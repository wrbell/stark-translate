#!/usr/bin/env python3
"""
mts_acceptance_probe.py — Gemma-4 assistant-drafter accept-rate soak (Mac GPU)

Loads the OptiQ E4B target + ``-assistant`` drafter via ``engines.mlx_spec``,
runs the 8 core theological canaries and ``TEST_SENTENCES`` from
``tools/benchmark_mlx_accel.py`` at gamma in {1, 2, 3}, and compares each
speculative decode against plain greedy ``mlx_lm.generate``.

Cloud / CI must not run this — it loads Metal models. Execute on M-series:

    python tools/mts_acceptance_probe.py
    python tools/mts_acceptance_probe.py --gammas 1,2 --runs 3 --output metrics/mts_probe.json

Prints per-gamma p50 latency, tok/s, accept_rate, and a byte-identical count
against greedy generate. Writes ``metrics/mts_probe_<ts>.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from datetime import datetime
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from engines.translation_prompts import (
    build_chat_messages,
    chat_template_extra_kwargs,
    clean_translation,
    dynamic_max_tokens,
)
from tools.benchmark_mlx_accel import TEST_SENTENCES
from training.theological_canaries import canary_sentences

DEFAULT_TARGET = "mlx-community/gemma-4-e4b-it-OptiQ-4bit"
DEFAULT_DRAFTER = "mlx-community/gemma-4-e4b-it-assistant-bf16"
MODEL_FAMILY = "gemma4"
N_CANARIES = 8


def _pct(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    idx = min(len(s) - 1, max(0, round((p / 100.0) * (len(s) - 1))))
    return s[idx]


def _prompt_ids(tokenizer: Any, text: str) -> list[int]:
    messages = build_chat_messages(
        text,
        source_lang="en",
        target_lang="es",
        model_family=MODEL_FAMILY,
    )
    prompt = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        **chat_template_extra_kwargs(model_family=MODEL_FAMILY),
    )
    if isinstance(prompt, str):
        bos = getattr(tokenizer, "bos_token", None)
        add_special = bos is None or not prompt.startswith(bos)
        return list(tokenizer.encode(prompt, add_special_tokens=add_special))
    if hasattr(prompt, "tolist"):
        prompt = prompt.tolist()
    return [int(x) for x in prompt]


def _apply_eos_fix(tokenizer: Any) -> None:
    """Honor Gemma-4 stop ids: <eos>=1, <turn|>=106, <tool_response>=50."""
    extra = {1, 106, 50}
    eot = None
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    if callable(convert):
        try:
            eot = convert("<end_of_turn>")
        except Exception:
            eot = None
    if eot is not None:
        extra.add(int(eot))
    current = set()
    for attr in ("eos_token_ids", "_eos_token_ids"):
        val = getattr(tokenizer, attr, None)
        if val is not None:
            current.update(int(x) for x in val if x is not None)
    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is not None:
        current.add(int(eos))
    merged = current | extra
    if hasattr(tokenizer, "eos_token_ids"):
        tokenizer.eos_token_ids = merged
    tokenizer._eos_token_ids = merged


def _assemble(stream) -> tuple[str, Any]:
    pieces: list[str] = []
    for tok in stream:
        pieces.append(tok.text)
    return "".join(pieces), stream.stats


def _greedy_generate(model: Any, tokenizer: Any, prompt_ids: list[int], max_tokens: int) -> str:
    from mlx_lm import generate

    return generate(
        model,
        tokenizer,
        prompt=prompt_ids,
        max_tokens=max_tokens,
        verbose=False,
    )


def _probe_prompts() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i, c in enumerate(canary_sentences(N_CANARIES)):
        out.append(
            {
                "id": f"canary_{i}",
                "en": c["en"],
                "expected_substrings": list(c["expected_substrings"]),
            }
        )
    for length, text in TEST_SENTENCES.items():
        out.append({"id": f"sent_{length}", "en": text, "expected_substrings": []})
    return out


def run_probe(
    *,
    target_id: str,
    drafter_id: str,
    gammas: list[int],
    runs: int,
    max_tokens: int | None,
    warmup: int,
) -> dict[str, Any]:
    from mlx_lm import load as mlx_load

    from engines.mlx_engine import materialize_mlx_model
    from engines.mlx_spec import load_gemma4_drafter, spec_stream

    print(f"Loading target {target_id} ...")
    t0 = time.perf_counter()
    model, tokenizer = mlx_load(target_id)
    _apply_eos_fix(tokenizer)
    materialize_mlx_model(model)
    print(f"  target ready ({time.perf_counter() - t0:.1f}s)")

    print(f"Loading drafter {drafter_id} ...")
    t1 = time.perf_counter()
    drafter = load_gemma4_drafter(drafter_id)
    print(f"  drafter ready ({time.perf_counter() - t1:.1f}s)")

    prompts = _probe_prompts()
    report: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "target": target_id,
        "drafter": drafter_id,
        "runs": runs,
        "warmup": warmup,
        "max_tokens": max_tokens,
        "gammas": {},
    }

    greedy_cache: dict[str, str] = {}

    for gamma in gammas:
        print(f"\n=== gamma={gamma} ===")
        if warmup > 0:
            warm_ids = _prompt_ids(tokenizer, TEST_SENTENCES["short"])
            warm_max = max_tokens or dynamic_max_tokens(TEST_SENTENCES["short"])
            for _ in range(warmup):
                list(spec_stream(model, tokenizer, drafter, warm_ids, gamma=gamma, max_tokens=warm_max))

        latencies_ms: list[float] = []
        tps_list: list[float] = []
        accept_list: list[float] = []
        prompt_rows: list[dict[str, Any]] = []
        n_identical = 0

        for spec in prompts:
            text = spec["en"]
            ids = _prompt_ids(tokenizer, text)
            cap = max_tokens or dynamic_max_tokens(text)

            if spec["id"] not in greedy_cache:
                greedy_cache[spec["id"]] = _greedy_generate(model, tokenizer, ids, cap)

            run_lats: list[float] = []
            run_tps: list[float] = []
            run_acc: list[float] = []
            last_text = ""
            last_stats = None
            for _ in range(runs):
                t_run = time.perf_counter()
                stream = spec_stream(model, tokenizer, drafter, ids, gamma=gamma, max_tokens=cap)
                last_text, last_stats = _assemble(stream)
                elapsed_s = time.perf_counter() - t_run
                elapsed_ms = elapsed_s * 1000.0
                n_tok = last_stats.n_tokens if last_stats is not None else 0
                tps = n_tok / elapsed_s if elapsed_s > 0 else 0.0
                run_lats.append(elapsed_ms)
                run_tps.append(tps)
                run_acc.append(last_stats.accept_rate if last_stats is not None else 0.0)

            latencies_ms.extend(run_lats)
            tps_list.extend(run_tps)
            accept_list.extend(run_acc)

            identical = last_text == greedy_cache[spec["id"]]
            if identical:
                n_identical += 1
            cleaned = clean_translation(last_text, model_family=MODEL_FAMILY)
            expect = spec["expected_substrings"]
            canary_ok = all(s.lower() in cleaned.lower() for s in expect) if expect else None

            row = {
                "id": spec["id"],
                "en": text,
                "spec_text": last_text,
                "greedy_text": greedy_cache[spec["id"]],
                "byte_identical": identical,
                "canary_ok": canary_ok,
                "latency_ms": {
                    "mean": statistics.mean(run_lats) if run_lats else 0.0,
                    "p50": _pct(run_lats, 50),
                },
                "tok_s_mean": statistics.mean(run_tps) if run_tps else 0.0,
                "accept_rate_mean": statistics.mean(run_acc) if run_acc else 0.0,
                "stats": None
                if last_stats is None
                else {
                    "n_tokens": last_stats.n_tokens,
                    "n_from_draft": last_stats.n_from_draft,
                    "n_verify_steps": last_stats.n_verify_steps,
                    "accept_rate": last_stats.accept_rate,
                    "gamma": last_stats.gamma,
                },
            }
            prompt_rows.append(row)
            ident = "MATCH" if identical else "DIFF"
            print(
                f"  {spec['id']}: p50={row['latency_ms']['p50']:.0f}ms "
                f"tok/s={row['tok_s_mean']:.1f} accept={row['accept_rate_mean']:.3f} {ident}"
            )

        gamma_row = {
            "p50_latency_ms": _pct(latencies_ms, 50),
            "mean_latency_ms": statistics.mean(latencies_ms) if latencies_ms else 0.0,
            "tok_s": statistics.mean(tps_list) if tps_list else 0.0,
            "accept_rate": statistics.mean(accept_list) if accept_list else 0.0,
            "byte_identical": {"matched": n_identical, "total": len(prompts)},
            "prompts": prompt_rows,
        }
        report["gammas"][str(gamma)] = gamma_row
        ident = gamma_row["byte_identical"]
        print(
            f"  SUMMARY gamma={gamma}  p50={gamma_row['p50_latency_ms']:.0f}ms  "
            f"tok/s={gamma_row['tok_s']:.1f}  accept_rate={gamma_row['accept_rate']:.3f}  "
            f"identical={ident['matched']}/{ident['total']}"
        )

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Gemma-4 MTS acceptance probe (Metal GPU)")
    parser.add_argument(
        "--gammas",
        type=str,
        default="1,2,3",
        help="Comma-separated gamma values (default: 1,2,3)",
    )
    parser.add_argument("--runs", type=int, default=3, help="Timed repetitions per prompt")
    parser.add_argument("--target", type=str, default=DEFAULT_TARGET, help="mlx-lm target repo id")
    parser.add_argument("--drafter", type=str, default=DEFAULT_DRAFTER, help="Gemma-4 assistant repo id")
    parser.add_argument("--max-tokens", type=int, default=None, help="Cap (default: dynamic_max_tokens)")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    gammas = [int(x.strip()) for x in args.gammas.split(",") if x.strip()]
    if not gammas or any(g < 1 for g in gammas):
        parser.error("--gammas must be a comma-separated list of integers >= 1")

    results = run_probe(
        target_id=args.target,
        drafter_id=args.drafter,
        gammas=gammas,
        runs=args.runs,
        max_tokens=args.max_tokens,
        warmup=args.warmup,
    )

    out_path = args.output or f"metrics/mts_probe_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
