#!/usr/bin/env python3
"""
benchmark_mlx_accel.py — MLX translation + STT acceleration matrix

Compares Mac candidates for cycle-time reduction (CUDA parity follow-up):

  Translation:
    1. TranslateGemma 4B 4-bit (baseline / current Mac default)
    2. Gemma 4 E4B OptiQ-4bit
    3. Gemma 4 E2B OptiQ-4bit
    4. E4B OptiQ + assistant drafter (MTS, γ=1)
    5. Winners ± TurboQuant KV

  STT:
    - mlx-whisper large-v3-turbo baseline
    - optional quantized turbo if --stt-model is passed
    - E2E overlap proxy: STT then MT on consecutive sentences (pipeline)

Cloud agents cannot run Metal — execute on M-series Mac:

    python tools/benchmark_mlx_accel.py --quick
    python tools/benchmark_mlx_accel.py --configs tg4b,e4b,e4b_mts --output metrics/mlx_accel.json

Gates (Mac soak):
  - E4B canary ≥ 7/8 (match CUDA llama.cpp T3)
  - No PLE garbage (ionoxff… / empty)
  - Translation p50 competitive with ~470 ms CUDA finals
"""

from __future__ import annotations

import argparse
import json
import resource
import statistics
import time
from datetime import datetime
from typing import Any

# Theological canaries shared with health_check / CUDA Phase 1A
try:
    from training.theological_canaries import THEOLOGICAL_CANARIES as _CANARY_DICTS

    CANARIES = [(c["en"], c["expected_substrings"][0]) for c in _CANARY_DICTS]
except ImportError:
    CANARIES = [
        ("The atonement of Christ", "expiación"),
        ("The grace of God is sufficient", "gracia"),
        ("The Holy Spirit", "Espíritu"),
        ("justification by faith", "justificación"),
        ("the resurrection of Jesus", "resurrección"),
        ("breaking of bread", "partimiento"),
        ("propitiation", "propiciación"),
        ("James the apostle", "Santiago"),
    ]

TEST_SENTENCES = {
    "short": "The grace of God is sufficient.",
    "medium": (
        "For God so loved the world that he gave his only begotten son, "
        "that whosoever believeth in him should not perish."
    ),
    "long": (
        "Brothers and sisters, let us turn to the book of Romans chapter eight, "
        "where the apostle Paul reminds us that there is therefore now no "
        "condemnation to them which are in Christ Jesus."
    ),
}

CONFIGS: dict[str, dict[str, Any]] = {
    "tg4b": {
        "label": "TranslateGemma 4B 4-bit (baseline)",
        "model_id": "mlx-community/translategemma-4b-it-4bit",
        "model_family": "translategemma",
        "draft_model_id": None,
        "num_draft_tokens": 0,
        "turboquant": False,
    },
    "e4b": {
        "label": "Gemma 4 E4B OptiQ-4bit",
        "model_id": "mlx-community/gemma-4-e4b-it-OptiQ-4bit",
        "model_family": "gemma4",
        "draft_model_id": None,
        "num_draft_tokens": 0,
        "turboquant": False,
    },
    "e2b": {
        "label": "Gemma 4 E2B OptiQ-4bit",
        "model_id": "mlx-community/gemma-4-e2b-it-OptiQ-4bit",
        "model_family": "gemma4",
        "draft_model_id": None,
        "num_draft_tokens": 0,
        "turboquant": False,
    },
    "e4b_mts": {
        "label": "Gemma 4 E4B OptiQ + assistant MTS (gamma=1)",
        "model_id": "mlx-community/gemma-4-e4b-it-OptiQ-4bit",
        "model_family": "gemma4",
        "draft_model_id": "mlx-community/gemma-4-e4b-it-assistant-bf16",
        "num_draft_tokens": 1,
        "turboquant": False,
    },
    "e4b_tq": {
        "label": "Gemma 4 E4B OptiQ + TurboQuant",
        "model_id": "mlx-community/gemma-4-e4b-it-OptiQ-4bit",
        "model_family": "gemma4",
        "draft_model_id": None,
        "num_draft_tokens": 0,
        "turboquant": True,
    },
    "e4b_mts_tq": {
        "label": "Gemma 4 E4B OptiQ + MTS + TurboQuant",
        "model_id": "mlx-community/gemma-4-e4b-it-OptiQ-4bit",
        "model_family": "gemma4",
        "draft_model_id": "mlx-community/gemma-4-e4b-it-assistant-bf16",
        "num_draft_tokens": 1,
        "turboquant": True,
    },
}


def _rss_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes; Linux reports KiB
    if usage > 10**9:  # likely bytes (mac)
        return usage / (1024 * 1024)
    return usage / 1024


def _pct(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    idx = min(len(s) - 1, max(0, round((p / 100.0) * (len(s) - 1))))
    return s[idx]


def _stats(latencies: list[float]) -> dict[str, float]:
    if not latencies:
        return {"n": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0}
    return {
        "n": len(latencies),
        "mean": statistics.mean(latencies),
        "p50": _pct(latencies, 50),
        "p95": _pct(latencies, 95),
    }


def _looks_like_garbage(text: str) -> bool:
    t = text.lower()
    if not t.strip():
        return True
    if "ionoxff" in t or "�" in text:
        return True
    # Extremely repetitive garbage
    if len(set(t.replace(" ", ""))) < 4 and len(t) > 20:
        return True
    return False


def run_canaries(engine) -> dict[str, Any]:
    """Run theological canaries; return pass count + details."""
    results = []
    passed = 0
    for en, expect_sub in CANARIES:
        out = engine.translate(en, source_lang="en", target_lang="es")
        text = out.text
        ok = expect_sub.lower() in text.lower() and not _looks_like_garbage(text)
        if ok:
            passed += 1
        results.append(
            {
                "en": en,
                "expect": expect_sub,
                "out": text[:200],
                "latency_ms": out.latency_ms,
                "pass": ok,
                "garbage": _looks_like_garbage(text),
            }
        )
    return {"passed": passed, "total": len(CANARIES), "details": results}


def bench_config(cfg_key: str, runs: int, warmup: int) -> dict[str, Any]:
    """Load one config, warm up, measure latency + canaries + RSS."""
    from engines.mlx_engine import MLXGemmaEngine

    cfg = CONFIGS[cfg_key]
    print(f"\n=== {cfg['label']} ===")
    rss_before = _rss_mb()

    engine = MLXGemmaEngine(
        model_id=cfg["model_id"],
        model_family=cfg["model_family"],
        use_prompt_cache=cfg["model_family"] == "translategemma" and not cfg["draft_model_id"],
        use_turboquant=cfg["turboquant"],
        draft_model_id=cfg["draft_model_id"],
        num_draft_tokens=cfg["num_draft_tokens"] or 1,
    )
    try:
        engine.load()
    except Exception as exc:
        print(f"  LOAD FAILED: {exc}")
        return {
            "config": cfg_key,
            "label": cfg["label"],
            "error": str(exc),
            "rss_mb": _rss_mb() - rss_before,
        }

    # Warmup
    for _ in range(warmup):
        engine.translate(TEST_SENTENCES["short"], source_lang="en", target_lang="es")

    per_length: dict[str, Any] = {}
    for length, text in TEST_SENTENCES.items():
        lats: list[float] = []
        tps_list: list[float] = []
        for _ in range(runs):
            result = engine.translate(text, source_lang="en", target_lang="es")
            if _looks_like_garbage(result.text):
                print(f"  WARNING: garbage output on {length}: {result.text[:80]!r}")
            lats.append(result.latency_ms)
            tps_list.append(result.tokens_per_second)
        per_length[length] = {
            "latency": _stats(lats),
            "tps_mean": statistics.mean(tps_list) if tps_list else 0.0,
            "sample": result.text[:120],
        }
        print(
            f"  {length}: p50={per_length[length]['latency']['p50']:.0f}ms "
            f"p95={per_length[length]['latency']['p95']:.0f}ms "
            f"tps={per_length[length]['tps_mean']:.1f}"
        )

    canary = run_canaries(engine)
    print(f"  canary: {canary['passed']}/{canary['total']}")

    rss_delta = _rss_mb() - rss_before
    engine.unload()

    return {
        "config": cfg_key,
        "label": cfg["label"],
        "model_id": cfg["model_id"],
        "model_family": cfg["model_family"],
        "draft_model_id": cfg["draft_model_id"],
        "turboquant": cfg["turboquant"],
        "per_length": per_length,
        "canary": {"passed": canary["passed"], "total": canary["total"]},
        "canary_details": canary["details"],
        "rss_delta_mb": rss_delta,
        "gate_canary_ok": canary["passed"] >= 7,
        "gate_no_garbage": not any(d.get("garbage") for d in canary["details"]),
    }


def bench_stt_baseline(runs: int, warmup: int, model_id: str, duration_s: float = 3.0) -> dict[str, Any]:
    """mlx-whisper latency on synthetic audio (Mac only)."""
    import numpy as np

    from engines.mlx_engine import MLXWhisperEngine

    print(f"\n=== STT {model_id} ===")
    audio = np.zeros(int(16000 * duration_s), dtype=np.float32)
    # Soft sine so VAD/energy isn't zeroed everywhere
    t = np.arange(len(audio)) / 16000.0
    audio = (0.05 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)

    engine = MLXWhisperEngine(model_id=model_id)
    try:
        engine.load()
    except Exception as exc:
        return {"error": str(exc), "model_id": model_id}

    for _ in range(warmup):
        engine.transcribe(audio, language="en", word_timestamps=False)

    lats = []
    for _ in range(runs):
        # Partials path: no word timestamps
        t0 = time.perf_counter()
        engine.transcribe(audio, language="en", word_timestamps=False)
        lats.append((time.perf_counter() - t0) * 1000)

    engine.unload()
    stats = _stats(lats)
    print(f"  STT (no timestamps): p50={stats['p50']:.0f}ms p95={stats['p95']:.0f}ms")
    return {"model_id": model_id, "latency": stats, "note": "synthetic audio; use real clips for WER"}


def bench_e2e_overlap_proxy(mt_result: dict, stt_result: dict) -> dict[str, Any]:
    """Estimate cycle time with STT(N)∥MT(N−1) overlap = max(stt, mt) + other."""
    if "error" in mt_result or "error" in stt_result:
        return {"error": "missing STT or MT result"}
    stt_p50 = stt_result.get("latency", {}).get("p50", 0)
    mt_p50 = mt_result.get("per_length", {}).get("medium", {}).get("latency", {}).get("p50", 0)
    serial = stt_p50 + mt_p50
    overlapped = max(stt_p50, mt_p50)
    return {
        "stt_p50_ms": stt_p50,
        "mt_medium_p50_ms": mt_p50,
        "serial_e2e_ms": serial,
        "overlapped_e2e_ms": overlapped,
        "overlap_savings_ms": serial - overlapped,
        "note": "Proxy assumes perfect STT∥MT overlap (#168 / multiprocess)",
    }


def bench_mlx_gemma4_accel(
    configs: list[str] | None = None,
    runs: int = 5,
    warmup: int = 2,
    include_stt: bool = True,
    stt_model: str = "mlx-community/whisper-large-v3-turbo",
) -> dict[str, Any]:
    """Entry point used by ``tools/benchmark_latency.py --only mlx-accel``."""
    selected = configs or ["tg4b", "e4b", "e2b", "e4b_mts"]
    out: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "runs": runs,
        "warmup": warmup,
        "configs": {},
    }

    for key in selected:
        if key not in CONFIGS:
            out["configs"][key] = {"error": f"unknown config {key}"}
            continue
        out["configs"][key] = bench_config(key, runs, warmup)

    if include_stt:
        out["stt"] = bench_stt_baseline(runs, warmup, stt_model)
        # Pair STT with best non-error MT config for overlap proxy
        for key in selected:
            mt = out["configs"].get(key, {})
            if "error" not in mt and "per_length" in mt:
                out.setdefault("e2e_overlap", {})[key] = bench_e2e_overlap_proxy(mt, out["stt"])

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="MLX Gemma 4 / TG acceleration matrix")
    parser.add_argument(
        "--configs",
        type=str,
        default="tg4b,e4b,e2b,e4b_mts",
        help=f"Comma-separated configs. Available: {','.join(CONFIGS)}",
    )
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--quick", action="store_true", help="3 runs, 1 warmup")
    parser.add_argument("--no-stt", action="store_true")
    parser.add_argument(
        "--stt-model",
        type=str,
        default="mlx-community/whisper-large-v3-turbo",
        help="mlx-whisper model id (probe quantized turbo by overriding)",
    )
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    runs = 3 if args.quick else args.runs
    warmup = 1 if args.quick else args.warmup
    configs = [c.strip() for c in args.configs.split(",") if c.strip()]

    results = bench_mlx_gemma4_accel(
        configs=configs,
        runs=runs,
        warmup=warmup,
        include_stt=not args.no_stt,
        stt_model=args.stt_model,
    )

    out_path = args.output or f"metrics/mlx_accel_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    import os

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_path}")

    # Summary table
    print("\n=== Summary ===")
    for key, row in results.get("configs", {}).items():
        if "error" in row:
            print(f"  {key}: ERROR {row['error']}")
            continue
        p50 = row.get("per_length", {}).get("medium", {}).get("latency", {}).get("p50", 0)
        can = row.get("canary", {})
        gates = []
        if row.get("gate_canary_ok"):
            gates.append("canary_ok")
        if row.get("gate_no_garbage"):
            gates.append("no_garbage")
        print(
            f"  {key}: medium_p50={p50:.0f}ms canary={can.get('passed')}/{can.get('total')} "
            f"rssΔ={row.get('rss_delta_mb', 0):.0f}MB gates={','.join(gates) or 'FAIL'}"
        )


if __name__ == "__main__":
    main()
