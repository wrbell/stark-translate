#!/usr/bin/env python3
"""
manage_adapters.py — Adapter Lifecycle Management

Register, activate, rollback, list, and export TranslateGemma/Whisper LoRA
adapters. Maintains a manifest at adapters/manifest.json tracking all versions,
active/previous status, eval metrics, and deployment timestamps.

Usage:
    # Register a new adapter version
    python tools/manage_adapters.py register \
        --adapter hybrid_runs/S8_deepl_only \
        --model gemma_4b \
        --eval-file hybrid_runs/S8_deepl_only_sermon_eval.json

    # Activate (runs health check first)
    python tools/manage_adapters.py activate --model gemma_4b --version S8_deepl_only

    # Rollback to previous
    python tools/manage_adapters.py rollback --model gemma_4b

    # List all versions
    python tools/manage_adapters.py list --model gemma_4b

    # Export active adapter to remote
    python tools/manage_adapters.py export --model gemma_4b --target user@mac:~/adapters/
"""

import argparse
import hashlib
import json
import logging
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

# Insert project root so sibling imports work
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

MANIFEST_PATH = PROJECT_ROOT / "adapters" / "manifest.json"
HEALTH_CHECK_SCRIPT = PROJECT_ROOT / "tools" / "health_check.py"


def _load_manifest() -> dict:
    """Load or initialize manifest."""
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_manifest(manifest: dict) -> None:
    """Save manifest to disk."""
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    logger.info(f"Manifest saved to {MANIFEST_PATH}")


def _ensure_model_entry(manifest: dict, model: str) -> dict:
    """Ensure a model entry exists in the manifest."""
    if model not in manifest:
        manifest[model] = {
            "active": None,
            "previous": None,
            "versions": {},
        }
    return manifest[model]


def _compute_sha256(filepath: Path) -> str:
    """Compute SHA-256 of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _find_safetensors(adapter_dir: Path) -> Path | None:
    """Find adapter_model.safetensors in the adapter directory."""
    candidate = adapter_dir / "adapter_model.safetensors"
    if candidate.exists():
        return candidate
    # Check subdirectories (e.g., checkpoint dirs)
    for p in sorted(adapter_dir.rglob("adapter_model.safetensors")):
        return p
    return None


def cmd_register(args) -> int:
    """Register a new adapter version."""
    adapter_dir = Path(args.adapter).resolve()
    if not adapter_dir.exists():
        logger.error(f"Adapter directory not found: {adapter_dir}")
        return 1

    model = args.model
    version = args.version or adapter_dir.name

    manifest = _load_manifest()
    model_entry = _ensure_model_entry(manifest, model)

    if version in model_entry["versions"]:
        logger.warning(f"Version '{version}' already registered for {model}. Overwriting.")

    # Compute SHA-256 of adapter weights
    safetensors = _find_safetensors(adapter_dir)
    adapter_sha256 = None
    if safetensors:
        logger.info(f"Computing SHA-256 of {safetensors}...")
        adapter_sha256 = _compute_sha256(safetensors)
        logger.info(f"SHA-256: {adapter_sha256[:16]}...")
    else:
        logger.warning("No adapter_model.safetensors found — SHA-256 will be null")

    # Read training manifest if exists
    training_manifest = None
    training_manifest_path = adapter_dir / "training_manifest.json"
    if training_manifest_path.exists():
        with open(training_manifest_path, encoding="utf-8") as f:
            training_manifest = json.load(f)
        logger.info(f"Read training manifest from {training_manifest_path}")

    # Read eval metrics if provided
    eval_metrics = {}
    if args.eval_file:
        eval_path = Path(args.eval_file)
        if eval_path.exists():
            with open(eval_path, encoding="utf-8") as f:
                eval_metrics = json.load(f)
            logger.info(f"Read eval metrics from {eval_path}")
        else:
            logger.warning(f"Eval file not found: {eval_path}")

    # Build version entry
    version_entry = {
        "path": str(adapter_dir),
        "registered": datetime.now(UTC).isoformat(),
        "adapter_sha256": adapter_sha256,
        "eval_metrics": eval_metrics,
        "training_manifest": training_manifest,
        "verdict": None,
        "deployed": None,
    }

    model_entry["versions"][version] = version_entry
    _save_manifest(manifest)

    logger.info(f"Registered {model}/{version} from {adapter_dir}")
    return 0


def cmd_activate(args) -> int:
    """Activate an adapter version (runs health check first)."""
    model = args.model
    version = args.version

    manifest = _load_manifest()
    model_entry = _ensure_model_entry(manifest, model)

    if version not in model_entry["versions"]:
        logger.error(f"Version '{version}' not found for {model}. Run 'register' first.")
        return 1

    ver = model_entry["versions"][version]
    adapter_path = ver["path"]

    if not Path(adapter_path).exists():
        logger.error(f"Adapter path no longer exists: {adapter_path}")
        return 1

    # Run health check
    logger.info(f"Running health check on {model}/{version}...")
    base_model_map = {
        "gemma_4b": "google/translategemma-4b-it",
        "gemma_12b": "google/translategemma-12b-it",
    }
    base_model = base_model_map.get(model, args.base_model or "google/translategemma-4b-it")

    health_cmd = [
        sys.executable,
        str(HEALTH_CHECK_SCRIPT),
        "--adapter",
        adapter_path,
        "--base-model",
        base_model,
    ]
    if args.max_latency:
        health_cmd.extend(["--max-latency", str(args.max_latency)])

    result = subprocess.run(health_cmd, capture_output=True, text=True)
    logger.info(result.stdout)
    if result.stderr:
        logger.info(result.stderr)

    if result.returncode != 0:
        logger.error(f"Health check FAILED for {model}/{version}. Aborting activation.")
        ver["verdict"] = "FAIL"
        _save_manifest(manifest)
        return 1

    ver["verdict"] = "PASS"

    # Move current active to previous
    current_active = model_entry["active"]
    if current_active and current_active != version:
        model_entry["previous"] = current_active
        logger.info(f"Moved {model}/{current_active} from active to previous")

    model_entry["active"] = version
    _save_manifest(manifest)

    logger.info(f"Activated {model}/{version}")
    return 0


def cmd_rollback(args) -> int:
    """Swap active and previous adapter versions."""
    model = args.model

    manifest = _load_manifest()
    model_entry = _ensure_model_entry(manifest, model)

    active = model_entry.get("active")
    previous = model_entry.get("previous")

    if not previous:
        logger.error(f"No previous version to rollback to for {model}")
        return 1

    model_entry["active"] = previous
    model_entry["previous"] = active
    _save_manifest(manifest)

    logger.info(f"Rolled back {model}: active={previous}, previous={active}")
    return 0


def cmd_list(args) -> int:
    """List all adapter versions for a model."""
    model = args.model

    manifest = _load_manifest()

    if model:
        models = [model] if model in manifest else []
        if not models:
            logger.info(f"No versions registered for '{model}'")
            # Show available models
            if manifest:
                logger.info(f"Available models: {', '.join(manifest.keys())}")
            return 0
    else:
        models = list(manifest.keys())
        if not models:
            logger.info("No adapters registered. Use 'register' to add one.")
            return 0

    for m in models:
        entry = manifest[m]
        active = entry.get("active")
        previous = entry.get("previous")
        versions = entry.get("versions", {})

        print(f"\n{'=' * 60}")
        print(f"Model: {m}")
        print(f"Active:   {active or '(none)'}")
        print(f"Previous: {previous or '(none)'}")
        print(f"{'=' * 60}")

        if not versions:
            print("  (no versions registered)")
            continue

        for ver_name, ver in sorted(versions.items()):
            status_flags = []
            if ver_name == active:
                status_flags.append("ACTIVE")
            if ver_name == previous:
                status_flags.append("PREVIOUS")
            status = f" [{', '.join(status_flags)}]" if status_flags else ""

            verdict = ver.get("verdict") or "—"
            registered = ver.get("registered", "?")
            sha_short = ver.get("adapter_sha256", "")[:12] or "—"
            deployed = ver.get("deployed") or "—"

            print(f"\n  {ver_name}{status}")
            print(f"    Path:       {ver.get('path', '?')}")
            print(f"    Registered: {registered}")
            print(f"    SHA-256:    {sha_short}...")
            print(f"    Verdict:    {verdict}")
            print(f"    Deployed:   {deployed}")

            metrics = ver.get("eval_metrics", {})
            if metrics:
                print("    Eval metrics:")
                for k, v in sorted(metrics.items()):
                    if isinstance(v, float):
                        print(f"      {k}: {v:.4f}")
                    else:
                        print(f"      {k}: {v}")

    print()
    return 0


def cmd_export(args) -> int:
    """Export active adapter to a remote target via rsync."""
    model = args.model
    target = args.target

    manifest = _load_manifest()
    model_entry = _ensure_model_entry(manifest, model)

    active = model_entry.get("active")
    if not active:
        logger.error(f"No active version for {model}. Run 'activate' first.")
        return 1

    ver = model_entry["versions"][active]
    adapter_path = ver["path"]

    if not Path(adapter_path).exists():
        logger.error(f"Adapter path no longer exists: {adapter_path}")
        return 1

    # Ensure trailing slash on source for rsync content sync
    source = adapter_path.rstrip("/") + "/"
    logger.info(f"Exporting {model}/{active} to {target}")
    logger.info(f"  Source: {source}")

    rsync_cmd = [
        "rsync",
        "-avz",
        "--progress",
        source,
        target,
    ]
    logger.info(f"  Command: {' '.join(rsync_cmd)}")

    if args.dry_run:
        rsync_cmd.insert(1, "--dry-run")
        logger.info("  (dry run)")

    result = subprocess.run(rsync_cmd, capture_output=False)

    if result.returncode != 0:
        logger.error(f"rsync failed with exit code {result.returncode}")
        return 1

    # Update deployment timestamp
    ver["deployed"] = datetime.now(UTC).isoformat()
    _save_manifest(manifest)

    logger.info(f"Export complete. Deployed timestamp updated for {model}/{active}.")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Adapter lifecycle management — register, activate, rollback, export")
    subparsers = parser.add_subparsers(dest="command", help="Subcommand")

    # --- register ---
    p_register = subparsers.add_parser("register", help="Register a new adapter version")
    p_register.add_argument(
        "--adapter",
        required=True,
        help="Path to adapter directory",
    )
    p_register.add_argument(
        "--model",
        required=True,
        help="Model name (e.g., gemma_4b, gemma_12b, whisper_turbo)",
    )
    p_register.add_argument(
        "--eval-file",
        default=None,
        help="Path to eval JSON with metrics",
    )
    p_register.add_argument(
        "--version",
        default=None,
        help="Version name (default: adapter directory name)",
    )

    # --- activate ---
    p_activate = subparsers.add_parser("activate", help="Activate an adapter (runs health check)")
    p_activate.add_argument("--model", required=True, help="Model name")
    p_activate.add_argument("--version", required=True, help="Version to activate")
    p_activate.add_argument(
        "--base-model",
        default=None,
        help="Override base model ID for health check",
    )
    p_activate.add_argument(
        "--max-latency",
        type=float,
        default=None,
        help="Override max latency for health check (seconds)",
    )

    # --- rollback ---
    p_rollback = subparsers.add_parser("rollback", help="Swap active and previous")
    p_rollback.add_argument("--model", required=True, help="Model name")

    # --- list ---
    p_list = subparsers.add_parser("list", help="List adapter versions")
    p_list.add_argument("--model", default=None, help="Model name (omit for all)")

    # --- export ---
    p_export = subparsers.add_parser("export", help="Export active adapter via rsync")
    p_export.add_argument("--model", required=True, help="Model name")
    p_export.add_argument(
        "--target",
        required=True,
        help="rsync target (e.g., user@mac:~/adapters/)",
    )
    p_export.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be transferred without actually copying",
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    cmd_map = {
        "register": cmd_register,
        "activate": cmd_activate,
        "rollback": cmd_rollback,
        "list": cmd_list,
        "export": cmd_export,
    }
    return cmd_map[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
