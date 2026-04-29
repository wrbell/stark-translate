#!/usr/bin/env python3
"""
qe_filter.py — Reference-free quality-estimation filter for translation pairs.

Loads CometKiwi (default Unbabel/wmt23-cometkiwi-da-xl), scores each (en, es)
pair, writes a filtered JSONL containing only pairs at or above the threshold.
Each kept row gains a `_kiwi_score` field (segment-level CometKiwi score).

Used in two places (see docs/gemma4_tuning/phase_a_infrastructure.md):
  1. Phase C — clean synthetic training data (general 0.80, sermon 0.85).
  2. Phase D — score candidates for preference triple construction.

NO-LEAKAGE RULE: filter / train with CometKiwi-XL (this script). Eval the
finished model with xCOMET-XL or COMET-22 (different family, supervised
reference-based). Using CometKiwi for both inflates apparent gains by 1–2
COMET points.

Usage:
    python training/qe_filter.py \\
        --input bible_data/synthetic/deepl_sermon_pairs_full.jsonl \\
        --output bible_data/synthetic/deepl_sermon_pairs_kiwi80.jsonl \\
        --threshold 0.80

    # Lower-VRAM alternative model:
    python training/qe_filter.py --model Unbabel/wmt22-cometkiwi-da ...
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
import sys
import time
from pathlib import Path

# Reduce VRAM fragmentation for the XL model.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("qe_filter")


def load_pairs(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                log.warning("skipping malformed JSON at line %d: %s", i, exc)
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def score_pairs(
    rows: list[dict],
    model_name: str,
    batch_size: int,
    src_key: str,
    tgt_key: str,
) -> list[float]:
    from comet import download_model, load_from_checkpoint

    log.info("downloading / loading %s ...", model_name)
    t0 = time.perf_counter()
    ckpt = download_model(model_name)
    model = load_from_checkpoint(ckpt)
    log.info("model loaded in %.1fs", time.perf_counter() - t0)

    samples = [{"src": r[src_key], "mt": r[tgt_key]} for r in rows]
    log.info("scoring %d pairs (batch_size=%d) ...", len(samples), batch_size)
    t0 = time.perf_counter()
    pred = model.predict(samples=samples, batch_size=batch_size, accelerator="auto", progress_bar=True)
    log.info(
        "scored in %.1fs (%.1f pairs/s)", time.perf_counter() - t0, len(samples) / max(time.perf_counter() - t0, 1e-9)
    )
    return [float(s) for s in pred.scores]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, required=True, help="JSONL of {en, es, ...} pairs")
    p.add_argument("--output", type=Path, required=True, help="filtered JSONL output")
    p.add_argument(
        "--rejected-output",
        type=Path,
        default=None,
        help="optional path for rejected pairs (for debugging / analysis)",
    )
    p.add_argument(
        "--scores-output",
        type=Path,
        default=None,
        help="optional path to dump every (input_index, score) for full distribution analysis",
    )
    p.add_argument(
        "--model",
        default="Unbabel/wmt23-cometkiwi-da-xl",
        help="HF model id for CometKiwi (default: XL). Smaller alternative: Unbabel/wmt22-cometkiwi-da",
    )
    p.add_argument("--threshold", type=float, default=0.80, help="keep pairs with score >= threshold")
    p.add_argument("--batch-size", type=int, default=16, help="CometKiwi batch size (16 fits A2000 Ada XL)")
    p.add_argument("--src-key", default="en", help="source-text key in input JSONL")
    p.add_argument("--tgt-key", default="es", help="target-text key in input JSONL")
    p.add_argument("--limit", type=int, default=None, help="score only first N pairs (smoke testing)")
    args = p.parse_args(argv)

    if not args.input.exists():
        p.error(f"input not found: {args.input}")
    if not 0.0 <= args.threshold <= 1.0:
        p.error("--threshold must be in [0.0, 1.0]")

    rows = load_pairs(args.input)
    log.info("loaded %d pairs from %s", len(rows), args.input)
    if args.limit:
        rows = rows[: args.limit]
        log.info("limited to first %d pairs", len(rows))

    # Validate keys exist on a sample.
    sample = rows[0] if rows else {}
    for key in (args.src_key, args.tgt_key):
        if key not in sample:
            p.error(f"key {key!r} missing from first row; got keys {list(sample.keys())}")

    scores = score_pairs(rows, args.model, args.batch_size, args.src_key, args.tgt_key)
    assert len(scores) == len(rows), f"score count {len(scores)} != row count {len(rows)}"

    kept: list[dict] = []
    rejected: list[dict] = []
    for row, score in zip(rows, scores):
        annotated = dict(row)
        annotated["_kiwi_score"] = round(score, 4)
        if score >= args.threshold:
            kept.append(annotated)
        else:
            rejected.append(annotated)

    write_jsonl(args.output, kept)
    log.info(
        "kept %d/%d (%.1f%%) at threshold %.2f -> %s",
        len(kept),
        len(rows),
        100 * len(kept) / max(len(rows), 1),
        args.threshold,
        args.output,
    )

    if args.rejected_output:
        write_jsonl(args.rejected_output, rejected)
        log.info("wrote %d rejected pairs -> %s", len(rejected), args.rejected_output)

    if args.scores_output:
        with args.scores_output.open("w", encoding="utf-8") as f:
            for i, score in enumerate(scores):
                f.write(json.dumps({"index": i, "score": round(score, 4)}) + "\n")
        log.info("wrote score dump -> %s", args.scores_output)

    if scores:
        log.info(
            "score distribution: min=%.3f p25=%.3f median=%.3f p75=%.3f max=%.3f mean=%.3f stdev=%.3f",
            min(scores),
            statistics.quantiles(scores, n=4)[0] if len(scores) >= 4 else min(scores),
            statistics.median(scores),
            statistics.quantiles(scores, n=4)[2] if len(scores) >= 4 else max(scores),
            max(scores),
            statistics.mean(scores),
            statistics.stdev(scores) if len(scores) >= 2 else 0.0,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
