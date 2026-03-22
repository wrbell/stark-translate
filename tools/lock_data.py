#!/usr/bin/env python3
"""
lock_data.py — Data Lockfile for Reproducibility and Drift Detection

Computes SHA-256 hashes of registered training data files and saves them to
a lockfile. On verify, compares current file hashes against the lockfile and
reports mismatches, missing files, and new files.

Usage:
    # Generate lockfile
    python tools/lock_data.py generate

    # Verify against saved lockfile
    python tools/lock_data.py verify
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Project root setup
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Registered data files — key: short name, value: relative path from root
# ---------------------------------------------------------------------------
REGISTERED_FILES: dict[str, str] = {
    "verse_pairs_train": "bible_data/aligned/verse_pairs_train.jsonl",
    "verse_pairs_test": "bible_data/aligned/verse_pairs_test.jsonl",
    "glossary_pairs": "bible_data/glossary/glossary_pairs.jsonl",
    "tier1_boost": "bible_data/glossary/tier1_boost.json",
    "tier2_master": "bible_data/glossary/tier2_master.json",
    "sermon_whisper_chunks": "ablation/sermon_whisper_chunks_expanded.json",
}

LOCKFILE_PATH = _PROJECT_ROOT / "bible_data" / "data_lockfile.json"

# Buffer size for hashing large files (8 KB)
_HASH_BUF_SIZE = 8192


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    """Compute SHA-256 hex digest of a file using buffered reads."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(_HASH_BUF_SIZE)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def count_entries(path: Path) -> int:
    """Count lines for JSONL or top-level elements for JSON files."""
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        with open(path) as f:
            return sum(1 for _ in f)
    elif suffix == ".json":
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, list | dict):
            return len(data)
        return 1
    return 0


def build_file_entry(name: str, rel_path: str) -> dict | None:
    """Build a lockfile entry for one registered file."""
    abs_path = _PROJECT_ROOT / rel_path
    if not abs_path.exists():
        logger.warning("File not found, skipping: %s", rel_path)
        return None

    stat = abs_path.stat()
    entry: dict = {
        "path": rel_path,
        "sha256": sha256_file(abs_path),
        "size_bytes": stat.st_size,
    }

    suffix = abs_path.suffix.lower()
    if suffix == ".jsonl":
        entry["lines"] = count_entries(abs_path)
    elif suffix == ".json":
        entry["element_count"] = count_entries(abs_path)

    return entry


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


def cmd_generate(args: argparse.Namespace) -> int:
    """Generate the data lockfile."""
    logger.info("Generating data lockfile for %d registered files", len(REGISTERED_FILES))

    files: dict[str, dict] = {}
    for name, rel_path in REGISTERED_FILES.items():
        entry = build_file_entry(name, rel_path)
        if entry is not None:
            files[name] = entry
            logger.info(
                "  %-25s  sha256=%.12s...  size=%d",
                name,
                entry["sha256"],
                entry["size_bytes"],
            )

    lockfile = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "files": files,
    }

    LOCKFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOCKFILE_PATH, "w") as f:
        json.dump(lockfile, f, indent=2)
        f.write("\n")

    logger.info("Lockfile written to %s (%d files)", LOCKFILE_PATH, len(files))
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    """Verify current files against saved lockfile."""
    if not LOCKFILE_PATH.exists():
        logger.error("No lockfile found at %s — run 'generate' first", LOCKFILE_PATH)
        return 1

    with open(LOCKFILE_PATH) as f:
        lockfile = json.load(f)

    logger.info(
        "Verifying against lockfile generated at %s",
        lockfile.get("generated_at", "unknown"),
    )

    saved_files: dict[str, dict] = lockfile.get("files", {})
    mismatches = 0
    missing = 0
    ok = 0

    for name, entry in saved_files.items():
        abs_path = _PROJECT_ROOT / entry["path"]
        if not abs_path.exists():
            logger.error("  MISSING  %-25s  %s", name, entry["path"])
            missing += 1
            continue

        current_hash = sha256_file(abs_path)
        current_size = abs_path.stat().st_size

        if current_hash != entry["sha256"]:
            logger.error(
                "  MISMATCH %-25s  expected=%.12s...  got=%.12s...",
                name,
                entry["sha256"],
                current_hash,
            )
            if current_size != entry["size_bytes"]:
                logger.error(
                    "           %-25s  size: %d -> %d",
                    "",
                    entry["size_bytes"],
                    current_size,
                )
            mismatches += 1
        else:
            logger.info("  OK       %-25s  sha256=%.12s...", name, current_hash)
            ok += 1

    # Check for registered files not in the lockfile
    for name in REGISTERED_FILES:
        if name not in saved_files:
            logger.warning("  NEW      %-25s  (not in lockfile)", name)

    logger.info("Verification complete: %d ok, %d mismatches, %d missing", ok, mismatches, missing)

    if mismatches > 0 or missing > 0:
        return 1
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Data lockfile tool for reproducibility and drift detection",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("generate", help="Compute hashes and save lockfile")
    subparsers.add_parser("verify", help="Verify files against saved lockfile")

    args = parser.parse_args()

    if args.command == "generate":
        return cmd_generate(args)
    elif args.command == "verify":
        return cmd_verify(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
