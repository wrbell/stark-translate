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
import re
import sys
import urllib.request
from pathlib import Path
from typing import Any

from engines.model_paths import default_models_dir, resolve_backend, resolve_model_path
from operator_app.http_requests import open_http, validate_http_url

logger = logging.getLogger("stark-translate.setup")


# -- cache dir resolution -----------------------------------------------------


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
    validate_http_url(url)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".partial")

    start_byte = partial.stat().st_size if partial.exists() else 0
    if expected_size is not None and start_byte > expected_size:
        raise ValueError("Partial model exceeds its expected size; remove the partial and retry")
    if expected_size is not None and start_byte == expected_size and partial.exists():
        partial.replace(target)
        return

    req = urllib.request.Request(url)
    if start_byte:
        req.add_header("Range", f"bytes={start_byte}-")
        logger.info("resuming %s from byte %d", target.name, start_byte)

    with open_http(req, timeout=30) as resp:
        length = resp.headers.get("Content-Length")
        response_size = int(length) if length is not None else None
        if response_size is not None and response_size < 0:
            raise ValueError("Invalid negative model response Content-Length")
        total = expected_size
        if resp.status == 200:
            # Servers may ignore Range. Replace the partial from byte zero;
            # appending a full response would silently duplicate its prefix.
            start_byte = 0
            mode = "wb"
            if total is not None and response_size is not None and response_size != total:
                raise ValueError("Model response size differs from the expected full file")
            total = total if total is not None else response_size
        elif resp.status == 206:
            match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", resp.headers.get("Content-Range", ""))
            if not match:
                raise ValueError("Resumed model response requires an explicit Content-Range")
            first, last, remote_total = map(int, match.groups())
            if first != start_byte or not first <= last < remote_total:
                raise ValueError("Resumed model Content-Range does not match the requested offset")
            if total is not None and remote_total != total:
                raise ValueError("Resumed model total differs from its expected size")
            total = remote_total
            range_size = last - first + 1
            if response_size is not None and response_size != range_size:
                raise ValueError("Resumed model Content-Length differs from its range")
            response_size = range_size
            mode = "ab" if start_byte else "wb"
        else:
            raise ValueError(f"Unexpected HTTP {resp.status} for a model download")
        with partial.open(mode) as f:
            downloaded, received, last_pct = start_byte, 0, -1
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                if (total is not None and downloaded + len(chunk) > total) or (
                    response_size is not None and received + len(chunk) > response_size
                ):
                    raise ValueError("Model response exceeds its declared size")
                f.write(chunk)
                downloaded += len(chunk)
                received += len(chunk)
                if total:
                    pct = int(downloaded * 100 / total)
                    if pct != last_pct and pct % 5 == 0:
                        logger.info("  %s: %d%% (%d / %d MiB)", target.name, pct, downloaded >> 20, total >> 20)
                        last_pct = pct
            if (response_size is not None and received != response_size) or (total is not None and downloaded != total):
                raise ValueError("Model response is incomplete; partial retained for retry")

    partial.replace(target)


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
        elif kind == "derived-ct2":
            source = lockfile["models"][entry["source_model"]]
            checks.append(
                {
                    "name": key,
                    "url": "",
                    "status": "pass",
                    "detail": f"Derived locally from {source['repo_id']}@{source['revision']}",
                }
            )
            continue
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
    try:
        validate_http_url(url)
        req = urllib.request.Request(url, method="HEAD")
        with open_http(req, timeout=timeout_s) as resp:
            return ("pass", f"HTTP {resp.status}")
    except urllib.error.HTTPError as exc:
        if exc.code == 405:
            # Some HF buckets reject HEAD; try a tiny GET with Range: bytes=0-0.
            try:
                req2 = urllib.request.Request(url)
                req2.add_header("Range", "bytes=0-0")
                with open_http(req2, timeout=timeout_s) as resp:
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
    backend: str | None = None,
    include: list[str] | None = None,
    profile: str | None = None,
    offline: bool = False,
    build_native: bool = False,
    converter_python: str | None = None,
) -> int:
    """Run the model setup flow. Returns process exit code."""
    if models_dir is None:
        models_dir = default_models_dir()
    if project_root is None:
        project_root = Path(os.environ.get("STARK_PROJECT_ROOT", Path.cwd()))

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    try:
        lockfile = load_lockfile(project_root=project_root)
    except FileNotFoundError as exc:
        logger.error(str(exc))
        return 2

    lockfile_version = lockfile.get("version", "?")
    models = lockfile.get("models", {})
    from stark_translate.profiles import resolve_profile

    selected_profile = resolve_profile(profile, backend or "auto")
    if selected_profile.lite:
        keys = selected_profile.model_keys(include, models)
        missing_keys = keys - models.keys()
        if missing_keys:
            raise ValueError(f"Profile entries missing from manifest: {sorted(missing_keys)}")
        models = {k: v for k, v in models.items() if k in keys}
    elif backend is not None:
        selected_backend = resolve_backend(backend)
        selected_groups = set(include or [])
        models = {
            k: v
            for k, v in models.items()
            if selected_backend in v.get("required_for", []) or v.get("optional_group") in selected_groups
        }
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
                if offline:
                    raise FileNotFoundError(f"Offline artifact missing or unverified: {target}")
                logger.info("[get ] %s ← %s", key, entry["url"])
                _download_direct(entry["url"], target, entry.get("size_bytes"))
                if entry.get("sha256"):
                    _verify_sha256(target, entry["sha256"])
                _write_sidecar(target, lockfile_version, entry.get("sha256"))
                n_done += 1
            elif kind == "hf-snapshot":
                target_dir = models_dir / entry["subdir"]
                cached = resolve_model_path(key, models_dir=models_dir, project_root=project_root, local_only=True)
                if not refresh and cached and Path(cached) != target_dir and not entry.get("allow_patterns"):
                    logger.info("[skip] %s — using existing snapshot %s", key, cached)
                    n_skipped += 1
                    continue
                marker = target_dir / ".installed"
                marker_matches = False
                if marker.exists():
                    try:
                        installed = json.loads(marker.read_text())
                        marker_matches = (
                            installed.get("repo_id") == entry["repo_id"]
                            and installed.get("revision") == entry.get("revision", "main")
                            and installed.get("lockfile_version") == lockfile_version
                        )
                    except (OSError, ValueError):
                        pass
                if not refresh and marker_matches and cached is not None:
                    logger.info("[skip] %s — already installed", key)
                    n_skipped += 1
                    continue
                if offline:
                    raise FileNotFoundError(f"Offline pinned snapshot missing: {key}")
                logger.info("[get ] %s ← hf:%s@%s", key, entry["repo_id"], entry.get("revision", "main"))
                _download_hf_snapshot(
                    repo_id=entry["repo_id"],
                    revision=entry.get("revision", "main"),
                    target_dir=target_dir,
                    allow_patterns=allow_patterns or entry.get("allow_patterns"),
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
            elif kind == "derived-ct2":
                from tools.marian_ct2_setup import ensure_managed_marian

                artifact, created = ensure_managed_marian(
                    entry,
                    project_root=project_root or _find_project_root(),
                    models_dir=models_dir,
                    refresh=refresh,
                    **({"converter_python": converter_python} if converter_python else {}),
                    **({"offline": True} if offline else {}),
                    **({"managed_only": True} if selected_profile.lite else {}),
                )
                logger.info("[%s] %s — %s", "build" if created else "skip", key, artifact)
                n_done += int(created)
                n_skipped += int(not created)
            else:
                logger.error("[fail] %s — unknown type %r", key, kind)
                n_failed += 1
        except Exception as exc:
            logger.error("[fail] %s — %s", key, exc)
            n_failed += 1

    if selected_profile.final_engine == "llamacpp" and n_failed == 0:
        from tools.llama_runtime import install_native

        try:
            install_native(selected_profile.backend, models_dir, offline=offline, build=build_native)
        except Exception as exc:
            logger.error("[fail] native llama-server: %s", exc)
            n_failed += 1
    logger.info("done: %d installed, %d skipped, %d failed", n_done, n_skipped, n_failed)
    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(bootstrap_models())
