#!/usr/bin/env python3
"""
filter_chunks_by_confidence.py — Filter whisper chunks JSON by quality score.

Selects the top-N chunks ranked by Whisper avg_logprob (or confidence).
Output is a new chunks JSON in the same format, ready for align_deepgram_chunks.py.

Usage:
    python training/filter_chunks_by_confidence.py \
        --input ablation/sermon_whisper_chunks_sttdata.json \
        --output ablation/sermon_whisper_chunks_sttdata_top22k.json \
        --target-size 22000 \
        --metric logprob
"""

import argparse
import json
import logging
import math
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def score_chunk(chunk: dict, metric: str) -> float:
    """Compute quality score for a chunk. Higher is better."""
    if metric == "confidence":
        return chunk.get("confidence", 0.0)
    if metric == "logprob":
        return chunk.get("avg_logprob", -10.0)
    if metric == "combined":
        return chunk.get("confidence", 0.0) * math.exp(chunk.get("avg_logprob", -10.0))
    raise ValueError(f"Unknown metric: {metric}")


def main():
    parser = argparse.ArgumentParser(description="Filter whisper chunks by confidence/logprob, keeping top N.")
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to input chunks JSON (e.g., ablation/sermon_whisper_chunks_sttdata.json)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to output filtered chunks JSON",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        required=True,
        help="Number of top chunks to keep",
    )
    parser.add_argument(
        "--metric",
        choices=["confidence", "logprob", "combined"],
        default="logprob",
        help="Quality metric for ranking (default: logprob, has more spread than confidence)",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=0.5,
        help="Drop chunks shorter than this many seconds (default: 0.5)",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=30.0,
        help="Drop chunks longer than this many seconds (default: 30.0, Whisper context limit)",
    )
    args = parser.parse_args()

    logger.info("Loading chunks from %s", args.input)
    with open(args.input) as f:
        chunks = json.load(f)
    logger.info("Loaded %d chunks", len(chunks))

    # Drop chunks outside duration bounds
    pre_filter = len(chunks)
    filtered = [c for c in chunks if args.min_duration <= (c.get("end", 0) - c.get("start", 0)) <= args.max_duration]
    logger.info(
        "Duration filter [%.1fs, %.1fs]: kept %d / %d",
        args.min_duration,
        args.max_duration,
        len(filtered),
        pre_filter,
    )

    # Sort by quality metric (descending)
    filtered.sort(key=lambda c: -score_chunk(c, args.metric))

    # Keep top N
    if args.target_size >= len(filtered):
        logger.warning(
            "target_size (%d) >= filtered chunks (%d), keeping all",
            args.target_size,
            len(filtered),
        )
        top = filtered
    else:
        top = filtered[: args.target_size]

    # Stats
    threshold = score_chunk(top[-1], args.metric)
    sources = {c.get("source", "") for c in top}
    logger.info(
        "Selected top %d chunks (metric=%s, threshold=%.4f)",
        len(top),
        args.metric,
        threshold,
    )
    logger.info("Source diversity: %d unique sermons", len(sources))

    # Write output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(top, f, indent=2, ensure_ascii=False)
    logger.info("Wrote %d chunks to %s", len(top), args.output)


if __name__ == "__main__":
    main()
