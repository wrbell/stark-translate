#!/usr/bin/env python3
"""
health_check.py — Adapter Health Check for TranslateGemma / Gemma 4

Verifies an adapter produces sane translations before deployment.
Runs 8 canary sentences covering critical theological terms (including
partimiento del pan), checks expected substrings, measures latency, and
detects hallucination via word-count ratio.

Usage:
    # Base model only (no adapter) — auto backend
    python tools/health_check.py

    # Mac / Apple Silicon (MLX OptiQ — Mac default)
    python tools/health_check.py --backend mlx \\
        --base-model mlx-community/gemma-4-e4b-it-OptiQ-4bit \\
        --model-family gemma4

    # TranslateGemma opt-out
    python tools/health_check.py --backend mlx \\
        --base-model mlx-community/translategemma-4b-it-4bit \\
        --model-family translategemma \\
        --adapter adapters/translategemma_4b/active

    # CUDA / bitsandbytes (WSL training box)
    python tools/health_check.py --backend cuda --adapter hybrid_runs/S8_deepl_only

    # Custom latency threshold + verbose
    python tools/health_check.py --backend mlx --max-latency 15 --verbose
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections.abc import Callable
from typing import Any

# Insert project root so training/ imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

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

DEFAULT_MLX_MODEL = "mlx-community/gemma-4-e4b-it-OptiQ-4bit"
DEFAULT_CUDA_MODEL = "google/translategemma-4b-it"


def _detect_backend() -> str:
    """Prefer MLX on Apple Silicon, else CUDA when available, else cuda label for HF path."""
    try:
        import mlx.core  # noqa: F401

        return "mlx"
    except ImportError:
        pass
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cuda"  # HF/bnb path; will fail clearly if torch/CUDA missing


def _load_mlx(
    base_model: str,
    adapter_dir: str | None,
    model_family: str,
) -> tuple[Any, Any, Callable[..., str]]:
    """Load TranslateGemma via mlx-lm; return (model, tokenizer, translate_fn)."""
    import mlx.core as mx
    from mlx_lm import generate, load

    mx.set_cache_limit(256 * 1024 * 1024)
    load_kwargs: dict[str, Any] = {}
    if adapter_dir and os.path.exists(adapter_dir):
        load_kwargs["adapter_path"] = adapter_dir
        logger.info("MLX adapter: %s", adapter_dir)
    elif adapter_dir:
        logger.warning("Adapter path does not exist: %s — loading base only", adapter_dir)

    logger.info("Loading MLX model %s...", base_model)
    model, tokenizer = load(base_model, **load_kwargs)

    # EOS fix (same as dry_run_ab / MLXGemmaEngine)
    eot_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
    default_eos = tokenizer.eos_token_id
    if not hasattr(tokenizer, "_eos_token_ids") or eot_id not in tokenizer._eos_token_ids:
        tokenizer._eos_token_ids = {default_eos, eot_id}

    def translate_fn(
        _model: Any,
        _tokenizer: Any,
        text: str,
        *,
        source_lang: str = "en",
        target_lang: str = "es",
    ) -> str:
        from engines.translation_prompts import (
            build_chat_messages,
            chat_template_extra_kwargs,
            clean_translation,
            dynamic_max_tokens,
        )

        messages = build_chat_messages(
            text,
            source_lang=source_lang,
            target_lang=target_lang,
            model_family=model_family,
        )
        prompt = _tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            **chat_template_extra_kwargs(model_family=model_family),
        )
        max_tok = dynamic_max_tokens(text)
        result = generate(_model, _tokenizer, prompt=prompt, max_tokens=max_tok, verbose=False)
        return clean_translation(result, model_family=model_family)

    return model, tokenizer, translate_fn


def _load_cuda(
    base_model: str,
    adapter_dir: str | None,
) -> tuple[Any, Any, Callable[..., str]]:
    """Load via bitsandbytes / PEFT (WSL training path)."""
    from training.evaluate_translation import load_gemma_model, translate_gemma

    model, tokenizer = load_gemma_model(base_model, adapter_dir=adapter_dir)

    def translate_fn(
        _model: Any,
        _tokenizer: Any,
        text: str,
        *,
        source_lang: str = "en",
        target_lang: str = "es",
    ) -> str:
        # evaluate_translation.translate_gemma is EN→ES fixed; ignore lang kwargs
        _ = source_lang, target_lang
        return translate_gemma(_model, _tokenizer, text)

    return model, tokenizer, translate_fn


def run_health_check(
    base_model: str,
    adapter_dir: str | None,
    max_latency: float,
    verbose: bool = False,
    n_canaries: int = 8,
    backend: str = "auto",
    model_family: str = "gemma4",
) -> dict:
    """Run canary sentence health check. Returns results dict."""
    if backend == "auto":
        backend = _detect_backend()

    logger.info("Backend=%s  base=%s  adapter=%s", backend, base_model, adapter_dir or "(none)")

    if backend == "mlx":
        model, tokenizer, translate_fn = _load_mlx(base_model, adapter_dir, model_family)
    elif backend in ("cuda", "hf", "cpu"):
        model, tokenizer, translate_fn = _load_cuda(base_model, adapter_dir)
    else:
        raise ValueError(f"Unsupported backend: {backend!r}")

    results = []
    all_pass = True
    tests = canary_sentences(n_canaries)

    for i, test in enumerate(tests, 1):
        en = test["en"]
        expected = test["expected_substrings"]

        t0 = time.perf_counter()
        translation = translate_fn(model, tokenizer, en, source_lang="en", target_lang="es")
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
        logger.info(f"         latency={elapsed:.3f}s  ratio={ratio:.2f}  found={len(found)}/{len(expected)}{detail}")

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
        "backend": backend,
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
        "--backend",
        choices=["auto", "mlx", "cuda", "hf"],
        default="auto",
        help="Inference backend: auto (detect), mlx (Apple Silicon), cuda/hf (bitsandbytes)",
    )
    parser.add_argument(
        "--base-model",
        default=None,
        help=(f"Base model ID (default: {DEFAULT_MLX_MODEL} for mlx, {DEFAULT_CUDA_MODEL} for cuda)"),
    )
    parser.add_argument(
        "--model-family",
        choices=["translategemma", "gemma4"],
        default="gemma4",
        help="Chat template family for MLX path (default: gemma4)",
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

    backend = args.backend
    if backend == "auto":
        backend = _detect_backend()
    base_model = args.base_model
    if not base_model:
        base_model = DEFAULT_MLX_MODEL if backend == "mlx" else DEFAULT_CUDA_MODEL

    result = run_health_check(
        base_model=base_model,
        adapter_dir=args.adapter,
        max_latency=args.max_latency,
        verbose=args.verbose,
        n_canaries=args.n_canaries,
        backend=backend,
        model_family=args.model_family,
    )

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        logger.info(f"Wrote results to {args.output}")

    sys.exit(0 if result["all_pass"] else 1)


if __name__ == "__main__":
    main()
