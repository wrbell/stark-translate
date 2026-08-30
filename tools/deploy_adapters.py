#!/usr/bin/env python3
"""Deploy versioned LoRA/CT2/GGUF adapters to inference endpoints.

Implements the six-phase pipeline from docs/deploy.md using the local
``adapters/manifest.json`` registry.

Phases: VERSION → TRANSFER → HEALTH CHECK → ACTIVATE → VERIFY.
Supports ``--dry-run`` and ``--rollback``.

Usage::

    python tools/deploy_adapters.py --cycle 1 --models whisper_turbo_ct2 \\
        --endpoints local --dry-run

    python tools/deploy_adapters.py --rollback --endpoints local --models whisper_turbo_ct2
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

from manage_adapters import (
    _ensure_model_entry,
    _load_manifest,
    _save_manifest,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("deploy_adapters")

ENDPOINTS_DEFAULT = {
    "mac-dev": {
        "host": "127.0.0.1",
        "user": None,
        "adapter_dir": str(PROJECT_ROOT / "adapters"),
        "reload_method": "local",
    },
    "local": {
        "host": "127.0.0.1",
        "user": None,
        "adapter_dir": str(PROJECT_ROOT / "adapters"),
        "reload_method": "local",
    },
}


def _endpoints_from_manifest(manifest: dict) -> dict:
    eps = dict(ENDPOINTS_DEFAULT)
    for name, cfg in (manifest.get("endpoints") or {}).items():
        eps[name] = cfg
    return eps


def _rsync_or_copy(src: Path, dest: Path, *, dry_run: bool) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        logger.info("[dry-run] would copy %s → %s", src, dest)
        return
    if dest.exists():
        shutil.rmtree(dest) if dest.is_dir() else dest.unlink()
    if src.is_dir():
        shutil.copytree(src, dest)
    else:
        shutil.copy2(src, dest)


def _activate_local(manifest: dict, model: str, version: str) -> None:
    """Swap active/previous slots in the manifest (no health check)."""
    model_entry = _ensure_model_entry(manifest, model)
    if version not in model_entry.get("versions", {}):
        raise KeyError(f"Version {version} not registered for {model}")
    current = model_entry.get("active")
    if current and current != version:
        model_entry["previous"] = current
    model_entry["active"] = version

    ver_path = Path(model_entry["versions"][version]["path"])
    active_dir = PROJECT_ROOT / "adapters" / model / "active"
    previous_dir = PROJECT_ROOT / "adapters" / model / "previous"
    if active_dir.exists():
        if previous_dir.exists():
            shutil.rmtree(previous_dir)
        active_dir.rename(previous_dir)
    if ver_path.exists():
        shutil.copytree(ver_path, active_dir)


def deploy(
    *,
    cycle: int | None,
    models: list[str],
    endpoints: list[str],
    dry_run: bool,
    skip_health: bool,
) -> int:
    manifest = _load_manifest()
    eps = _endpoints_from_manifest(manifest)
    record: dict = {
        "timestamp": datetime.now(UTC).isoformat(),
        "cycle": cycle,
        "models": models,
        "endpoints": endpoints,
        "status": "pending",
        "steps": [],
    }

    for model in models:
        entry = manifest.get(model) or {}
        active = entry.get("active")
        versions = entry.get("versions") or {}
        if not versions:
            logger.error("No versions registered for model %s — run manage_adapters register first", model)
            record["status"] = "failed"
            return 1
        if not active or active not in versions:
            active = sorted(versions.keys())[-1]
            logger.info("No active version for %s; using latest registered %s", model, active)

        version_meta = versions[active]
        src = Path(version_meta.get("path") or PROJECT_ROOT / "adapters" / model / active)
        if not src.exists():
            alt = PROJECT_ROOT / "adapters" / model / "active"
            src = alt if alt.exists() else src
        if not src.exists():
            logger.error("Adapter path missing for %s/%s: %s", model, active, src)
            record["status"] = "failed"
            return 1

        record["steps"].append({"phase": "version", "model": model, "version": active, "src": str(src)})

        for ep_name in endpoints:
            if ep_name not in eps:
                logger.error("Unknown endpoint %s (known: %s)", ep_name, sorted(eps))
                record["status"] = "failed"
                return 1
            ep = eps[ep_name]
            remote_root = Path(ep["adapter_dir"])
            staging = remote_root / model / "staging"
            record["steps"].append({"phase": "transfer", "endpoint": ep_name, "dest": str(staging)})

            if ep.get("reload_method") == "local" or ep.get("host") in ("127.0.0.1", "localhost"):
                _rsync_or_copy(src, staging, dry_run=dry_run)
            else:
                user_host = f"{ep['user']}@{ep['host']}" if ep.get("user") else ep["host"]
                remote = f"{user_host}:{staging}/"
                cmd = ["rsync", "-avz", "--progress", f"{src}/", remote]
                logger.info("TRANSFER: %s", " ".join(cmd))
                if not dry_run:
                    subprocess.run(cmd, check=True)

            is_stt = "whisper" in model or "ct2" in model
            if not skip_health and not dry_run and not is_stt:
                hc = [
                    sys.executable,
                    str(PROJECT_ROOT / "tools" / "health_check.py"),
                    "--adapter",
                    str(src),
                    "--n-canaries",
                    "8",
                ]
                logger.info("HEALTH CHECK: %s", " ".join(hc))
                result = subprocess.run(hc)
                if result.returncode != 0:
                    record["status"] = "health_failed"
                    manifest.setdefault("deployments", []).append(record)
                    _save_manifest(manifest)
                    return 1
                record["steps"].append({"phase": "health_check", "model": model, "pass": True})
            else:
                record["steps"].append({"phase": "health_check", "model": model, "skipped": True})

            if dry_run:
                logger.info("[dry-run] would activate %s on %s", model, ep_name)
            else:
                _activate_local(manifest, model, active)
                record["steps"].append({"phase": "activate", "model": model, "version": active})

    record["status"] = "success" if not dry_run else "dry_run"
    manifest.setdefault("deployments", []).append(record)
    _save_manifest(manifest)
    logger.info("Deploy %s", record["status"])
    print(json.dumps(record, indent=2))
    return 0


def rollback(*, models: list[str], dry_run: bool) -> int:
    manifest = _load_manifest()
    for model in models:
        entry = _ensure_model_entry(manifest, model)
        active = entry.get("active")
        previous = entry.get("previous")
        if not previous:
            logger.error("No previous version to rollback for %s", model)
            return 1
        if dry_run:
            logger.info("[dry-run] would rollback %s: %s ↔ %s", model, active, previous)
            continue
        entry["active"] = previous
        entry["previous"] = active
        logger.info("Rolled back %s: active=%s previous=%s", model, previous, active)
    if not dry_run:
        _save_manifest(manifest)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cycle", type=int, default=None)
    p.add_argument("--models", nargs="+", default=["whisper_turbo_ct2"])
    p.add_argument("--endpoints", nargs="+", default=["local"])
    p.add_argument("--all-adapters", action="store_true", help="Deploy every model in the manifest")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--rollback", action="store_true")
    p.add_argument("--skip-health", action="store_true")
    args = p.parse_args(argv)

    manifest = _load_manifest()
    if args.all_adapters:
        models = [
            m
            for m, v in manifest.items()
            if m not in ("endpoints", "deployments") and isinstance(v, dict) and "versions" in v
        ]
    else:
        models = args.models

    if args.rollback:
        return rollback(models=models, dry_run=args.dry_run)
    return deploy(
        cycle=args.cycle,
        models=models,
        endpoints=args.endpoints,
        dry_run=args.dry_run,
        skip_health=args.skip_health,
    )


if __name__ == "__main__":
    raise SystemExit(main())
