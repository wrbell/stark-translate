#!/usr/bin/env python3
"""
build_hard_subset.py — Build a training subset from mined hard examples.

Reads the JSONL output from mine_hard_examples.py, filters by WER thresholds,
optionally includes all Tier 1 theological-term chunks, applies per-source
stratified caps, and outputs a chunks JSON compatible with align_deepgram_chunks.py.

Usage:
    python training/build_hard_subset.py \
        --mined whisper_ablation/w14_mined.jsonl \
        --chunks-json ablation/sermon_whisper_chunks_w14_combined.json \
        --output ablation/sermon_whisper_chunks_w15_hard.json \
        --target-size 10000 --include-tier1
"""

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path

# Ensure project root is on path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from training.align_deepgram_chunks import group_chunks_by_source

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_mined_jsonl(path: Path) -> list[dict]:
    """Load mined records from a JSONL file, skipping blank/malformed lines."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def build_chunk_lookup(chunks: list[dict]) -> dict[tuple[str, int], dict]:
    """Build a lookup from (source, chunk_idx) to full chunk dict.

    Uses the same sorted-group + enumerate ordering as mine_hard_examples.py
    to ensure chunk_idx alignment.
    """
    grouped = group_chunks_by_source(chunks)
    lookup = {}
    for source, source_chunks in sorted(grouped.items()):
        for chunk_idx, chunk in enumerate(source_chunks):
            lookup[(source, chunk_idx)] = chunk
    return lookup


def build_subset(args):
    """Filter mined records and produce a chunks JSON for align_deepgram_chunks.py."""
    # Load mined JSONL
    logger.info("Loading mined records from %s", args.mined)
    records = load_mined_jsonl(Path(args.mined))
    logger.info("Loaded %d mined records", len(records))

    if not records:
        logger.warning("No mined records found. Writing empty output.")
        with open(args.output, "w") as f:
            json.dump([], f)
        return

    # Load original chunks for lookup
    logger.info("Loading original chunks from %s", args.chunks_json)
    with open(args.chunks_json) as f:
        chunks = json.load(f)
    lookup = build_chunk_lookup(chunks)
    logger.info("Built lookup for %d chunks", len(lookup))

    # Step 1: Filter by WER bounds
    wer_filtered = [r for r in records if args.wer_min <= r.get("wer", 0) < args.wer_max]
    logger.info(
        "WER filter [%.2f, %.2f): %d / %d records",
        args.wer_min,
        args.wer_max,
        len(wer_filtered),
        len(records),
    )

    # Step 2: Include Tier 1 chunks (union with WER filter)
    selected_keys = {(r["source"], r["chunk_idx"]) for r in wer_filtered}
    tier1_added = 0

    if args.include_tier1:
        for r in records:
            key = (r["source"], r["chunk_idx"])
            if r.get("has_tier1") and key not in selected_keys:
                wer_filtered.append(r)
                selected_keys.add(key)
                tier1_added += 1
        logger.info("Tier 1 inclusion: +%d chunks (union)", tier1_added)

    # Step 3: Stratified per-source cap
    max_per_source = args.max_per_source or (args.target_size // 20)
    by_source = defaultdict(list)
    for r in wer_filtered:
        by_source[r["source"]].append(r)

    capped = []
    sources_capped = 0
    for source, source_records in by_source.items():
        # Sort by WER descending (hardest first) within each source
        source_records.sort(key=lambda r: -r.get("wer", 0))
        if len(source_records) > max_per_source:
            capped.extend(source_records[:max_per_source])
            sources_capped += 1
        else:
            capped.extend(source_records)

    logger.info(
        "Stratified cap (%d/source): %d records, %d sources capped",
        max_per_source,
        len(capped),
        sources_capped,
    )

    # Step 4: Sort by WER descending, take top target_size
    capped.sort(key=lambda r: -r.get("wer", 0))
    if len(capped) > args.target_size:
        selected = capped[: args.target_size]
    else:
        selected = capped

    logger.info("Selected %d / %d candidates (target: %d)", len(selected), len(capped), args.target_size)

    # Step 5: Map back to original chunk dicts
    output_chunks = []
    missing = 0
    for r in selected:
        key = (r["source"], r["chunk_idx"])
        chunk = lookup.get(key)
        if chunk is not None:
            output_chunks.append(chunk)
        else:
            missing += 1

    if missing:
        logger.warning("%d selected records not found in original chunks (skipped)", missing)

    # Step 6: Write output
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output_chunks, f, indent=2, ensure_ascii=False)
    logger.info("Wrote %d chunks to %s", len(output_chunks), args.output)

    # Stats
    unique_sources = {c.get("source", "") for c in output_chunks}
    tier1_count = sum(1 for r in selected if r.get("has_tier1"))
    wer_values = [r.get("wer", 0) for r in selected]
    logger.info("  Unique sources: %d", len(unique_sources))
    logger.info("  Tier 1 chunks: %d (%.1f%%)", tier1_count, 100 * tier1_count / max(len(selected), 1))
    if wer_values:
        logger.info(
            "  WER range: %.2f - %.2f (mean: %.2f)",
            min(wer_values),
            max(wer_values),
            sum(wer_values) / len(wer_values),
        )


def main():
    parser = argparse.ArgumentParser(description="Build a hard-example training subset from mined WER results.")
    parser.add_argument(
        "--mined",
        type=str,
        required=True,
        help="Path to mined JSONL from mine_hard_examples.py",
    )
    parser.add_argument(
        "--chunks-json",
        type=str,
        required=True,
        help="Path to original whisper chunks JSON (for full chunk records)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output filtered chunks JSON (compatible with align_deepgram_chunks.py)",
    )
    parser.add_argument(
        "--wer-min",
        type=float,
        default=0.15,
        help="Minimum WER threshold for hard examples (default: 0.15)",
    )
    parser.add_argument(
        "--wer-max",
        type=float,
        default=0.80,
        help="Maximum WER — exclude garbage chunks above this (default: 0.80)",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=10000,
        help="Maximum number of chunks to select (default: 10000)",
    )
    parser.add_argument(
        "--max-per-source",
        type=int,
        default=None,
        help="Max chunks per source sermon (default: target_size // 20)",
    )
    parser.add_argument(
        "--include-tier1",
        action="store_true",
        help="Always include chunks containing Tier 1 theological terms",
    )
    args = parser.parse_args()
    build_subset(args)


if __name__ == "__main__":
    main()
