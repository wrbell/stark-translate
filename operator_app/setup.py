"""Model bootstrap (Phase v2026.7).

Reads ``models.lock.json`` and ensures every entry is present in the
configured cache directory with the recorded SHA-256. Idempotent — re-runs
skip already-installed entries unless ``--refresh`` is passed.

Cache locations:
    Linux/Mac:  $XDG_CACHE_HOME/stark-translate/models
                (default ~/.cache/stark-translate/models)
    Windows:    %LOCALAPPDATA%\\stark-translate\\models

Override with ``STARK_MODELS_DIR`` env or ``--models-dir`` CLI flag.

Each downloaded entry gets a ``.installed`` sidecar JSON recording the
lockfile version + SHA-256 it satisfies; that's how we detect "already
done" without re-hashing 5 GB of GGUF on every startup.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import sys
import urllib.request
from pathlib import Path
from typing import Any

logger = logging.getLogger("stark-translate.setup")


# -- cache dir resolution -----------------------------------------------------


def default_models_dir() -> Path:
    """Platform-appropriate model cache root."""
    override = os.environ.get("STARK_MODELS_DIR")
    if override:
        return Path(override).expanduser()

    if platform.system() == "Windows":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home())
        return Path(base) / "stark-translate" / "models"

    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        return Path(xdg) / "stark-translate" / "models"
    return Path.home() / ".cache" / "stark-translate" / "models"


# -- lockfile loading ---------------------------------------------------------


def load_lockfile(project_root: Path | None = None) -> dict[str, Any]:
    """Load models.lock.json from project root or installed package."""
    if project_root is None:
        project_root = _find_project_root()
    lockfile = project_root / "models.lock.json"
    if not lockfile.exists():
        # When installed from a wheel, the lockfile lives next to the package.
        try:
            from importlib import resources

            with resources.as_file(resources.files("operator_app").joinpath("../models.lock.json")) as p:
                if p.exists():
                    lockfile = p
        except Exception:
            pass

    if not lockfile.exists():
        raise FileNotFoundError(f"models.lock.json not found (looked in {project_root})")
    return json.loads(lockfile.read_text())


def _find_project_root() -> Path:
    """Walk up from this file looking for pyproject.toml or models.lock.json."""
    here = Path(__file__).resolve().parent
    for candidate in [here, *here.parents]:
        if (candidate / "models.lock.json").exists() or (candidate / "pyproject.toml").exists():
            return candidate
    return here


# -- sidecar-based skip detection ---------------------------------------------


def _sidecar_path(target: Path) -> Path:
    return target.with_suffix(target.suffix + ".installed")


def _is_already_installed(target: Path, expected_sha256: str | None) -> bool:
    """Has this target already been satisfied by a previous run?"""
    if not target.exists():
        return False
    sidecar = _sidecar_path(target)
    if not sidecar.exists():
        return False
    try:
        meta = json.loads(sidecar.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    if expected_sha256 and meta.get("sha256") != expected_sha256:
        return False
    return True


def _write_sidecar(target: Path, lockfile_version: str, sha256_hex: str | None) -> None:
    sidecar = _sidecar_path(target)
    sidecar.write_text(
        json.dumps(
            {
                "lockfile_version": lockfile_version,
                "sha256": sha256_hex,
                "installed_at": _now_iso(),
            },
            indent=2,
        )
    )


def _now_iso() -> str:
    from datetime import datetime

    return datetime.now().isoformat(timespec="seconds")


# -- direct download (for GGUF files) -----------------------------------------


def _download_direct(url: str, target: Path, expected_size: int | None) -> None:
    """Stream a single file from ``url`` to ``target``, resumable.

    Resumes from a ``.partial`` sidecar if present. Honors HTTP Range so
    interrupted 5 GB downloads on flaky church Wi-Fi don't restart from zero.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".partial")

    start_byte = partial.stat().st_size if partial.exists() else 0
    if expected_size and start_byte >= expected_size:
        partial.rename(target)
        return

    req = urllib.request.Request(url)
    if start_byte:
        req.add_header("Range", f"bytes={start_byte}-")
        logger.info("resuming %s from byte %d", target.name, start_byte)

    try:
        with urllib.request.urlopen(req) as resp:
            total = expected_size or int(resp.headers.get("Content-Length") or 0) + start_byte
            mode = "ab" if start_byte else "wb"
            with partial.open(mode) as f:
                downloaded = start_byte
                last_pct = -1
                while True:
                    chunk = resp.read(1 << 20)  # 1 MiB
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = int(downloaded * 100 / total)
                        if pct != last_pct and pct % 5 == 0:
                            logger.info("  %s: %d%% (%d / %d MiB)", target.name, pct, downloaded >> 20, total >> 20)
                            last_pct = pct
    except urllib.error.HTTPError as exc:
        if exc.code == 416 and partial.exists() and expected_size and partial.stat().st_size >= expected_size:
            # Already complete — server rejected the range request.
            pass
        else:
            raise

    partial.rename(target)


# -- HF snapshot wrapper ------------------------------------------------------


def _download_hf_snapshot(repo_id: str, revision: str, target_dir: Path, allow_patterns: list[str] | None) -> None:
    """Wrap huggingface_hub.snapshot_download with our cache layout."""
    from huggingface_hub import snapshot_download

    target_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=str(target_dir),
        allow_patterns=allow_patterns,
    )


# -- SHA-256 verification ----------------------------------------------------


def _verify_sha256(path: Path, expected_hex: str) -> str:
    """Stream-hash a file and compare to the lockfile entry."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    actual = h.hexdigest()
    if actual.lower() != expected_hex.lower():
        raise ValueError(f"SHA-256 mismatch for {path}: expected {expected_hex[:16]}…, got {actual[:16]}…")
    return actual


# -- URL reachability check (no downloads) -----------------------------------


def check_lockfile_urls(
    project_root: Path | None = None,
    timeout_s: float = 10.0,
) -> dict:
    """HEAD every URL in the lockfile and return a dict of results.

    Direct entries are HEAD'd against the configured URL. HF snapshots are
    HEAD'd against the public ``/api/models/<repo_id>`` endpoint, which
    returns 200 for public repos and 401 for private/missing repos.

    Returns ``{"ok": bool, "checks": [{"name", "url", "status", "detail"}, ...]}``.
    Use to validate ``models.lock.json`` before a release tag.
    """
    try:
        lockfile = load_lockfile(project_root=project_root)
    except FileNotFoundError as exc:
        return {"ok": False, "error": str(exc), "checks": []}

    checks = []
    for key, entry in lockfile.get("models", {}).items():
        kind = entry.get("type", "direct")
        if kind == "direct":
            url = entry["url"]
        elif kind == "hf-snapshot":
            url = f"https://huggingface.co/api/models/{entry['repo_id']}"
        else:
            checks.append({"name": key, "url": "", "status": "fail", "detail": f"unknown type {kind!r}"})
            continue

        status, detail = _head_url(url, timeout_s=timeout_s)
        checks.append({"name": key, "url": url, "status": status, "detail": detail})

    ok = all(c["status"] == "pass" for c in checks)
    return {"ok": ok, "checks": checks}


def _head_url(url: str, timeout_s: float) -> tuple[str, str]:
    """Issue a HEAD against ``url``. Returns (status, detail).

    HF Hub blocks HEAD for some asset URLs but accepts them with a Range
    header that asks for byte 0; we fall back to that on 405.
    """
    req = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return ("pass", f"HTTP {resp.status}")
    except urllib.error.HTTPError as exc:
        if exc.code == 405:
            # Some HF buckets reject HEAD; try a tiny GET with Range: bytes=0-0.
            try:
                req2 = urllib.request.Request(url)
                req2.add_header("Range", "bytes=0-0")
                with urllib.request.urlopen(req2, timeout=timeout_s) as resp:
                    if resp.status in (200, 206):
                        return ("pass", f"HTTP {resp.status} (range)")
                    return ("fail", f"HTTP {resp.status}")
            except urllib.error.HTTPError as exc2:
                return ("fail", f"HTTP {exc2.code} {exc2.reason}")
            except Exception as exc2:
                return ("fail", f"{type(exc2).__name__}: {exc2}")
        return ("fail", f"HTTP {exc.code} {exc.reason}")
    except Exception as exc:
        return ("fail", f"{type(exc).__name__}: {exc}")


# -- public API --------------------------------------------------------------


def bootstrap_models(
    models_dir: Path | None = None,
    refresh: bool = False,
    allow_patterns: list[str] | None = None,
    project_root: Path | None = None,
) -> int:
    """Run the model setup flow. Returns process exit code."""
    if models_dir is None:
        models_dir = default_models_dir()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    try:
        lockfile = load_lockfile(project_root=project_root)
    except FileNotFoundError as exc:
        logger.error(str(exc))
        return 2

    lockfile_version = lockfile.get("version", "?")
    models = lockfile.get("models", {})
    if not models:
        logger.error("models.lock.json is empty")
        return 2

    logger.info("stark-translate setup")
    logger.info("  models_dir:       %s", models_dir)
    logger.info("  lockfile version: %s", lockfile_version)
    logger.info("  entries:          %d", len(models))
    models_dir.mkdir(parents=True, exist_ok=True)

    n_skipped = 0
    n_done = 0
    n_failed = 0

    for key, entry in models.items():
        kind = entry.get("type", "direct")
        try:
            if kind == "direct":
                target = models_dir / entry["filename"]
                if not refresh and _is_already_installed(target, entry.get("sha256")):
                    logger.info("[skip] %s — already installed", key)
                    n_skipped += 1
                    continue
                logger.info("[get ] %s ← %s", key, entry["url"])
                _download_direct(entry["url"], target, entry.get("size_bytes"))
                if entry.get("sha256"):
                    _verify_sha256(target, entry["sha256"])
                _write_sidecar(target, lockfile_version, entry.get("sha256"))
                n_done += 1
            elif kind == "hf-snapshot":
                target_dir = models_dir / entry["subdir"]
                marker = target_dir / ".installed"
                if not refresh and marker.exists():
                    logger.info("[skip] %s — already installed", key)
                    n_skipped += 1
                    continue
                logger.info("[get ] %s ← hf:%s@%s", key, entry["repo_id"], entry.get("revision", "main"))
                _download_hf_snapshot(
                    repo_id=entry["repo_id"],
                    revision=entry.get("revision", "main"),
                    target_dir=target_dir,
                    allow_patterns=allow_patterns,
                )
                marker.write_text(
                    json.dumps(
                        {
                            "lockfile_version": lockfile_version,
                            "repo_id": entry["repo_id"],
                            "revision": entry.get("revision", "main"),
                            "installed_at": _now_iso(),
                        },
                        indent=2,
                    )
                )
                n_done += 1
            else:
                logger.error("[fail] %s — unknown type %r", key, kind)
                n_failed += 1
        except Exception as exc:
            logger.error("[fail] %s — %s", key, exc)
            n_failed += 1

    logger.info("done: %d installed, %d skipped, %d failed", n_done, n_skipped, n_failed)
    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(bootstrap_models())
