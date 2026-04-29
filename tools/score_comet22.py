#!/usr/bin/env python3
"""
score_comet22.py — Reference-based COMET-22 head-to-head over two hypothesis files.

Both inputs are the per-translation JSONL produced by bench_translate_t1_t4.py
(Tier A `*_sermon.jsonl` files). Each row has en/es_hyp/es_ref. The two files
must be the same length and same row order (same source corpus).

Scores both systems with `Unbabel/wmt22-comet-da` (reference-based, different
family from CometKiwi-XL — satisfies the no-leakage rule documented in the
gemma4_tuning Phase A docs). Reports per-system mean COMET-22, the delta, and
chrF++ as a robustness check.

Usage:
    python tools/score_comet22.py \\
        --a metrics/prod_e4b_verses_sermon.jsonl --a-label prod_e4b \\
        --b metrics/v1_e4b_verses_sermon.jsonl  --b-label v1 \\
        --out metrics/comet22_v1_vs_prod.json
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics as stats
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("score_comet22")


def load_hyps(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def score_with_comet(samples: list[dict], model_name: str, batch_size: int) -> list[float]:
    from comet import download_model, load_from_checkpoint

    log.info("downloading / loading %s", model_name)
    t0 = time.perf_counter()
    ckpt = download_model(model_name)
    model = load_from_checkpoint(ckpt)
    log.info("model loaded in %.1fs", time.perf_counter() - t0)

    log.info("scoring %d samples (batch_size=%d)", len(samples), batch_size)
    t0 = time.perf_counter()
    pred = model.predict(samples=samples, batch_size=batch_size, accelerator="auto", progress_bar=True)
    log.info("scored in %.1fs", time.perf_counter() - t0)
    return [float(s) for s in pred.scores]


def chrf_pp(hyp: str, ref: str) -> float:
    """Cheap chrF++ via sacrebleu, per-segment."""
    import sacrebleu

    return sacrebleu.sentence_chrf(hyp, [ref], char_order=6, word_order=2, beta=2).score


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--a", type=Path, required=True, help="System A hypotheses JSONL (en, es_hyp, es_ref)")
    p.add_argument("--b", type=Path, required=True, help="System B hypotheses JSONL")
    p.add_argument("--a-label", default="A")
    p.add_argument("--b-label", default="B")
    p.add_argument("--model", default="Unbabel/wmt22-comet-da", help="HF model id for COMET (reference-based)")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)

    rows_a = load_hyps(args.a)
    rows_b = load_hyps(args.b)
    if len(rows_a) != len(rows_b):
        raise SystemExit(f"row count mismatch: A={len(rows_a)} B={len(rows_b)}")
    log.info("loaded %d rows per side", len(rows_a))

    # Sanity: src + ref must match across the two files (same source corpus)
    mismatches = sum(1 for ra, rb in zip(rows_a, rows_b) if ra["en"] != rb["en"] or ra["es_ref"] != rb["es_ref"])
    if mismatches:
        raise SystemExit(f"FATAL: {mismatches} rows have different src/ref between A and B")
    log.info("src/ref aligned across both files (0 mismatches)")

    # Build COMET sample lists
    samples_a = [{"src": r["en"], "mt": r["es_hyp"], "ref": r["es_ref"]} for r in rows_a]
    samples_b = [{"src": r["en"], "mt": r["es_hyp"], "ref": r["es_ref"]} for r in rows_b]

    # Score (single model load, score both runs)
    from comet import download_model, load_from_checkpoint

    log.info("downloading / loading %s", args.model)
    t0 = time.perf_counter()
    ckpt = download_model(args.model)
    model = load_from_checkpoint(ckpt)
    log.info("model loaded in %.1fs", time.perf_counter() - t0)

    log.info("scoring system %s (%d samples)", args.a_label, len(samples_a))
    t0 = time.perf_counter()
    pred_a = model.predict(samples=samples_a, batch_size=args.batch_size, accelerator="auto", progress_bar=True)
    log.info("  done in %.1fs", time.perf_counter() - t0)
    scores_a = [float(s) for s in pred_a.scores]

    log.info("scoring system %s (%d samples)", args.b_label, len(samples_b))
    t0 = time.perf_counter()
    pred_b = model.predict(samples=samples_b, batch_size=args.batch_size, accelerator="auto", progress_bar=True)
    log.info("  done in %.1fs", time.perf_counter() - t0)
    scores_b = [float(s) for s in pred_b.scores]

    # chrF++ per side
    log.info("computing chrF++ for both systems")
    chrf_a = [chrf_pp(r["es_hyp"], r["es_ref"]) for r in rows_a]
    chrf_b = [chrf_pp(r["es_hyp"], r["es_ref"]) for r in rows_b]

    summary = {
        "model": args.model,
        "n": len(rows_a),
        args.a_label: {
            "comet22_mean": round(stats.mean(scores_a), 4),
            "comet22_median": round(stats.median(scores_a), 4),
            "comet22_p25": round(stats.quantiles(scores_a, n=4)[0], 4),
            "comet22_p75": round(stats.quantiles(scores_a, n=4)[2], 4),
            "comet22_stdev": round(stats.stdev(scores_a), 4),
            "chrf_mean": round(stats.mean(chrf_a), 2),
        },
        args.b_label: {
            "comet22_mean": round(stats.mean(scores_b), 4),
            "comet22_median": round(stats.median(scores_b), 4),
            "comet22_p25": round(stats.quantiles(scores_b, n=4)[0], 4),
            "comet22_p75": round(stats.quantiles(scores_b, n=4)[2], 4),
            "comet22_stdev": round(stats.stdev(scores_b), 4),
            "chrf_mean": round(stats.mean(chrf_b), 2),
        },
        "delta": {
            "comet22": round(stats.mean(scores_b) - stats.mean(scores_a), 4),
            "chrf": round(stats.mean(chrf_b) - stats.mean(chrf_a), 2),
        },
    }

    # Per-row deltas: how often does B > A?
    deltas = [b - a for a, b in zip(scores_a, scores_b)]
    summary["per_row"] = {
        "b_better_than_a": sum(1 for d in deltas if d > 0),
        "a_better_than_b": sum(1 for d in deltas if d < 0),
        "tied": sum(1 for d in deltas if d == 0),
        "median_delta": round(stats.median(deltas), 4),
    }

    log.info("=== summary ===")
    for k, v in summary.items():
        log.info("  %s: %s", k, v)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    log.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
