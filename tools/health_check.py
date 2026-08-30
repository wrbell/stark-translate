#!/usr/bin/env python3
"""
health_check.py — Adapter Health Check for TranslateGemma / Gemma 4

Verifies an adapter produces sane translations before deployment.
Runs 8 canary sentences covering critical theological terms (including
partimiento del pan), checks expected substrings, measures latency, and
detects hallucination via word-count ratio.

Usage:
    # Base model only (no adapter)
    python tools/health_check.py

    # With adapter
    python tools/health_check.py --adapter hybrid_runs/S8_deepl_only

    # Custom latency threshold
    python tools/health_check.py --adapter hybrid_runs/S8_deepl_only --max-latency 15

    # Verbose (print full translations)
    python tools/health_check.py --adapter hybrid_runs/S8_deepl_only --verbose
"""

import argparse
import json
import logging
import os
import sys
import time

# Insert project root so training/ imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from training.evaluate_translation import load_gemma_model, translate_gemma
from training.theological_canaries import THEOLOGICAL_CANARIES, canary_sentences

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Back-compat alias — tests/docs may import CANARY_TESTS
CANARY_TESTS = THEOLOGICAL_CANARIES

# Hallucination ratio bounds (output words / input words)
HALLUCINATION_MIN = 0.5
HALLUCINATION_MAX = 2.5


def run_health_check(
    base_model: str,
    adapter_dir: str | None,
    max_latency: float,
    verbose: bool = False,
    n_canaries: int = 8,
) -> dict:
    """Run canary sentence health check. Returns results dict."""
    logger.info("Loading model...")
    model, tokenizer = load_gemma_model(base_model, adapter_dir=adapter_dir)

    results = []
    all_pass = True
    tests = canary_sentences(n_canaries)

    for i, test in enumerate(tests, 1):
        en = test["en"]
        expected = test["expected_substrings"]

        t0 = time.perf_counter()
        translation = translate_gemma(model, tokenizer, en, source_lang="en", target_lang="es")
        elapsed = time.perf_counter() - t0

        translation_l = (translation or "").lower()
        found = [s for s in expected if s.lower() in translation_l]
        missing = [s for s in expected if s.lower() not in translation_l]
        substring_pass = len(missing) == 0

        latency_pass = elapsed <= max_latency

        input_words = len(en.split())
        output_words = len(translation.split()) if translation else 0
        ratio = output_words / input_words if input_words > 0 else 0.0
        ratio_pass = HALLUCINATION_MIN <= ratio <= HALLUCINATION_MAX

        sentence_pass = substring_pass and latency_pass and ratio_pass
        if not sentence_pass:
            all_pass = False

        verdict = "PASS" if sentence_pass else "FAIL"
        results.append(
            {
                "sentence": i,
                "en": en,
                "translation": translation,
                "latency_s": round(elapsed, 3),
                "expected": expected,
                "found": found,
                "missing": missing,
                "word_ratio": round(ratio, 2),
                "substring_pass": substring_pass,
                "latency_pass": latency_pass,
                "ratio_pass": ratio_pass,
                "pass": sentence_pass,
            }
        )

        status_parts = []
        if not substring_pass:
            status_parts.append(f"missing: {missing}")
        if not latency_pass:
            status_parts.append(f"latency {elapsed:.1f}s > {max_latency}s")
        if not ratio_pass:
            status_parts.append(f"ratio {ratio:.2f} outside [{HALLUCINATION_MIN}, {HALLUCINATION_MAX}]")

        detail = f" ({'; '.join(status_parts)})" if status_parts else ""
        logger.info(f"  [{verdict}] Sentence {i}: {en}")
        if verbose:
            logger.info(f"         -> {translation}")
        logger.info(
            f"         latency={elapsed:.3f}s  ratio={ratio:.2f}  "
            f"found={len(found)}/{len(expected)}{detail}"
        )

    overall = "PASS" if all_pass else "FAIL"
    passed_count = sum(1 for r in results if r["pass"])
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Overall: {overall} ({passed_count}/{len(results)} sentences passed)")
    logger.info(f"{'=' * 60}")

    return {
        "verdict": overall,
        "passed": passed_count,
        "total": len(results),
        "all_pass": all_pass,
        "base_model": base_model,
        "adapter": adapter_dir,
        "max_latency": max_latency,
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Health check: verify TranslateGemma/Gemma 4 adapter produces sane translations"
    )
    parser.add_argument(
        "--adapter",
        default=None,
        help="Path to adapter directory (optional, runs base model if omitted)",
    )
    parser.add_argument(
        "--base-model",
        default="google/translategemma-4b-it",
        help="Base TranslateGemma/Gemma model ID (default: google/translategemma-4b-it)",
    )
    parser.add_argument(
        "--max-latency",
        type=float,
        default=5.0,
        help="Max seconds per sentence (default: 5)",
    )
    parser.add_argument(
        "--n-canaries",
        type=int,
        default=8,
        help="Number of canary sentences to run (default: 8)",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON path for results",
    )
    args = parser.parse_args()

    result = run_health_check(
        base_model=args.base_model,
        adapter_dir=args.adapter,
        max_latency=args.max_latency,
        verbose=args.verbose,
        n_canaries=args.n_canaries,
    )

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        logger.info(f"Wrote results to {args.output}")

    sys.exit(0 if result["all_pass"] else 1)


if __name__ == "__main__":
    main()
