#!/usr/bin/env python3
"""
build_eval_sets.py — Build Evaluation Sets with Stratification and Provenance

Creates two evaluation sets from existing training data:

A. Translation holdout: 500 stratified verse pairs sampled from the full
   training corpus. Removes them from verse_pairs_train.jsonl and saves
   to verse_pairs_test.jsonl.

B. Sermon eval chunks: Filters sermon_whisper_chunks_expanded.json to
   chunks with split == "eval" and saves to sermon_eval_chunks.json.

C. Generates eval_registry.json with metadata about both eval sets.

WARNING: This script MODIFIES verse_pairs_train.jsonl by removing sampled
verses. Use --dry-run to preview without changes.

Usage:
    # Dry run (no files modified)
    python tools/build_eval_sets.py --dry-run

    # Build eval sets (modifies training data!)
    python tools/build_eval_sets.py

    # Custom verse count and seed
    python tools/build_eval_sets.py --verse-count 300 --seed 123
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
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
# Paths
# ---------------------------------------------------------------------------
TRAIN_PATH = _PROJECT_ROOT / "bible_data" / "aligned" / "verse_pairs_train.jsonl"
TEST_PATH = _PROJECT_ROOT / "bible_data" / "aligned" / "verse_pairs_test.jsonl"
SERMON_CHUNKS_PATH = _PROJECT_ROOT / "ablation" / "sermon_whisper_chunks_expanded.json"
SERMON_EVAL_PATH = _PROJECT_ROOT / "ablation" / "sermon_eval_chunks.json"
REGISTRY_PATH = _PROJECT_ROOT / "bible_data" / "eval_registry.json"

# Hash buffer
_HASH_BUF_SIZE = 8192


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(_HASH_BUF_SIZE)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file into a list of dicts."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_jsonl(path: Path, rows: list[dict]) -> None:
    """Save a list of dicts as JSONL."""
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def classify_verse_bucket(pair: dict) -> str:
    """Assign a rough bucket to a verse pair for stratified sampling.

    Uses the verse_id field to determine position in the Bible corpus.
    Buckets are broad enough to give reasonable stratification without
    requiring external genre metadata.
    """
    verse_id = pair.get("verse_id")
    if verse_id is None:
        return "unknown"

    # verse_id can be an integer (sequential) or a string like "01001001"
    if isinstance(verse_id, str) and len(verse_id) >= 2:
        try:
            book_num = int(verse_id[:2])
        except ValueError:
            return "unknown"
    elif isinstance(verse_id, int):
        # Sequential IDs: rough mapping based on verse count per section
        # Genesis starts at 1, Psalms around 14000, Matthew around 23000
        if verse_id <= 5852:
            return "pentateuch"  # Genesis-Deuteronomy
        elif verse_id <= 12930:
            return "history"  # Joshua-Esther
        elif verse_id <= 18383:
            return "poetry"  # Job-Song of Solomon
        elif verse_id <= 23145:
            return "prophets"  # Isaiah-Malachi
        elif verse_id <= 27610:
            return "gospels_acts"  # Matthew-Acts
        else:
            return "epistles_rev"  # Romans-Revelation
    else:
        return "unknown"

    # Book-number based classification (for string verse_id format)
    if book_num <= 5:
        return "pentateuch"
    elif book_num <= 17:
        return "history"
    elif book_num <= 22:
        return "poetry"
    elif book_num <= 39:
        return "prophets"
    elif book_num <= 44:
        return "gospels_acts"
    else:
        return "epistles_rev"


# ---------------------------------------------------------------------------
# A. Translation holdout
# ---------------------------------------------------------------------------


def build_translation_holdout(verse_count: int, seed: int, dry_run: bool) -> dict | None:
    """Sample verse_count pairs from training set for eval holdout.

    Returns metadata dict for the registry, or None on failure.
    """
    if not TRAIN_PATH.exists():
        logger.error("Training file not found: %s", TRAIN_PATH)
        return None

    logger.info("Loading training corpus from %s", TRAIN_PATH)
    all_pairs = load_jsonl(TRAIN_PATH)
    total = len(all_pairs)
    logger.info("Loaded %d verse pairs", total)

    if verse_count > total:
        logger.error("Requested %d verses but only %d available", verse_count, total)
        return None

    # Classify into buckets for stratified sampling
    buckets: dict[str, list[int]] = {}
    for i, pair in enumerate(all_pairs):
        bucket = classify_verse_bucket(pair)
        buckets.setdefault(bucket, []).append(i)

    logger.info("Bucket distribution:")
    for bucket, indices in sorted(buckets.items()):
        logger.info("  %-15s  %d pairs", bucket, len(indices))

    # Stratified sampling: proportional allocation per bucket
    rng = random.Random(seed)
    selected_indices: set[int] = set()

    # Calculate proportional counts per bucket
    bucket_counts: dict[str, int] = {}
    allocated = 0
    sorted_buckets = sorted(buckets.keys())
    for i, bucket in enumerate(sorted_buckets):
        if i == len(sorted_buckets) - 1:
            # Last bucket gets the remainder to ensure exact total
            bucket_counts[bucket] = verse_count - allocated
        else:
            proportion = len(buckets[bucket]) / total
            count = round(proportion * verse_count)
            count = min(count, len(buckets[bucket]))
            bucket_counts[bucket] = count
            allocated += count

    logger.info("Sampling plan (seed=%d):", seed)
    for bucket in sorted_buckets:
        logger.info(
            "  %-15s  %d / %d (%.1f%%)",
            bucket,
            bucket_counts[bucket],
            len(buckets[bucket]),
            100.0 * bucket_counts[bucket] / max(len(buckets[bucket]), 1),
        )

    # Sample from each bucket
    for bucket in sorted_buckets:
        indices = buckets[bucket]
        count = bucket_counts[bucket]
        sampled = rng.sample(indices, min(count, len(indices)))
        selected_indices.update(sampled)

    # If rounding caused us to be off, adjust
    if len(selected_indices) < verse_count:
        remaining = [i for i in range(total) if i not in selected_indices]
        extra = rng.sample(remaining, verse_count - len(selected_indices))
        selected_indices.update(extra)
    elif len(selected_indices) > verse_count:
        excess = rng.sample(sorted(selected_indices), len(selected_indices) - verse_count)
        selected_indices -= set(excess)

    eval_pairs = [all_pairs[i] for i in sorted(selected_indices)]
    train_pairs = [all_pairs[i] for i in range(total) if i not in selected_indices]

    logger.info(
        "Split: %d eval + %d train = %d total",
        len(eval_pairs),
        len(train_pairs),
        len(eval_pairs) + len(train_pairs),
    )

    if dry_run:
        logger.info("[DRY RUN] Would write %d pairs to %s", len(eval_pairs), TEST_PATH)
        logger.info(
            "[DRY RUN] Would update %s: %d -> %d pairs",
            TRAIN_PATH,
            total,
            len(train_pairs),
        )
        return {
            "type": "translation_holdout",
            "count": len(eval_pairs),
            "source": str(TRAIN_PATH.relative_to(_PROJECT_ROOT)),
            "original_train_count": total,
            "remaining_train_count": len(train_pairs),
            "seed": seed,
            "dry_run": True,
        }

    # Write eval set (replaces existing test file)
    logger.info("Writing %d eval pairs to %s", len(eval_pairs), TEST_PATH)
    save_jsonl(TEST_PATH, eval_pairs)

    # Write updated training set
    logger.info(
        "Updating training set: %d -> %d pairs (%d removed)",
        total,
        len(train_pairs),
        total - len(train_pairs),
    )
    save_jsonl(TRAIN_PATH, train_pairs)

    return {
        "type": "translation_holdout",
        "count": len(eval_pairs),
        "source": str(TRAIN_PATH.relative_to(_PROJECT_ROOT)),
        "output": str(TEST_PATH.relative_to(_PROJECT_ROOT)),
        "original_train_count": total,
        "remaining_train_count": len(train_pairs),
        "seed": seed,
        "sha256": sha256_file(TEST_PATH),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }


# ---------------------------------------------------------------------------
# B. Sermon eval chunks
# ---------------------------------------------------------------------------


def build_sermon_eval(dry_run: bool) -> dict | None:
    """Filter sermon chunks to eval split and save."""
    if not SERMON_CHUNKS_PATH.exists():
        logger.error("Sermon chunks file not found: %s", SERMON_CHUNKS_PATH)
        return None

    logger.info("Loading sermon chunks from %s", SERMON_CHUNKS_PATH)
    with open(SERMON_CHUNKS_PATH) as f:
        all_chunks = json.load(f)

    logger.info("Total chunks: %d", len(all_chunks))

    eval_chunks = [c for c in all_chunks if c.get("split") == "eval"]
    logger.info("Eval chunks (split=='eval'): %d", len(eval_chunks))

    if not eval_chunks:
        logger.warning("No eval chunks found")
        return None

    # Report source sermons
    sources: dict[str, int] = {}
    for chunk in eval_chunks:
        src = chunk.get("source", "unknown")
        sources[src] = sources.get(src, 0) + 1

    logger.info("Eval chunks by source sermon:")
    for src, count in sorted(sources.items()):
        logger.info("  %-40s  %d chunks", src, count)

    if dry_run:
        logger.info("[DRY RUN] Would write %d eval chunks to %s", len(eval_chunks), SERMON_EVAL_PATH)
        return {
            "type": "sermon_eval",
            "count": len(eval_chunks),
            "sources": sources,
            "dry_run": True,
        }

    logger.info("Writing %d eval chunks to %s", len(eval_chunks), SERMON_EVAL_PATH)
    with open(SERMON_EVAL_PATH, "w") as f:
        json.dump(eval_chunks, f, indent=2, ensure_ascii=False)
        f.write("\n")

    return {
        "type": "sermon_eval",
        "count": len(eval_chunks),
        "source": str(SERMON_CHUNKS_PATH.relative_to(_PROJECT_ROOT)),
        "output": str(SERMON_EVAL_PATH.relative_to(_PROJECT_ROOT)),
        "sources": sources,
        "sha256": sha256_file(SERMON_EVAL_PATH),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }


# ---------------------------------------------------------------------------
# C. Eval registry
# ---------------------------------------------------------------------------


def save_registry(entries: list[dict], dry_run: bool) -> None:
    """Save eval registry JSON with metadata about all eval sets."""
    registry = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "eval_sets": entries,
    }

    if dry_run:
        logger.info("[DRY RUN] Would write eval registry to %s", REGISTRY_PATH)
        logger.info("[DRY RUN] Registry contents:")
        logger.info(json.dumps(registry, indent=2))
        return

    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)
        f.write("\n")

    logger.info("Eval registry written to %s", REGISTRY_PATH)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build evaluation sets with stratification and provenance",
    )
    parser.add_argument(
        "--verse-count",
        type=int,
        default=500,
        help="Number of verse pairs to hold out for eval (default: 500)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be created WITHOUT modifying any files",
    )

    args = parser.parse_args()

    if args.dry_run:
        logger.info("=" * 60)
        logger.info("DRY RUN MODE — no files will be modified")
        logger.info("=" * 60)
    else:
        logger.warning("=" * 60)
        logger.warning("LIVE MODE — training data WILL be modified")
        logger.warning("  verse_pairs_train.jsonl will lose %d verses", args.verse_count)
        logger.warning("  verse_pairs_test.jsonl will be REPLACED")
        logger.warning("  Run with --dry-run first to preview")
        logger.warning("=" * 60)

    registry_entries: list[dict] = []

    # A. Translation holdout
    logger.info("")
    logger.info("--- A. Translation Holdout ---")
    translation_meta = build_translation_holdout(args.verse_count, args.seed, args.dry_run)
    if translation_meta:
        registry_entries.append(translation_meta)

    # B. Sermon eval chunks
    logger.info("")
    logger.info("--- B. Sermon Eval Chunks ---")
    sermon_meta = build_sermon_eval(args.dry_run)
    if sermon_meta:
        registry_entries.append(sermon_meta)

    # C. Eval registry
    logger.info("")
    logger.info("--- C. Eval Registry ---")
    save_registry(registry_entries, args.dry_run)

    logger.info("")
    logger.info("Done. %d eval sets processed.", len(registry_entries))
    return 0


if __name__ == "__main__":
    sys.exit(main())
