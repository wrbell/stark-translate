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
import hashlib
import json
import platform
import resource
import statistics
import time
from collections import Counter
from datetime import datetime
from importlib import metadata
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


# Separate full-model mlx-lm drafts; these are not the rejected assistant MTS head.
for _gamma in (1, 2, 3):
    CONFIGS[f"e4b_e2b_draft_g{_gamma}"] = {
        "label": f"Gemma 4 E4B OptiQ + E2B OptiQ draft (gamma={_gamma})",
        "model_id": CONFIGS["e4b"]["model_id"],
        "model_family": "gemma4",
        "draft_model_id": "mlx-community/gemma-4-e2b-it-OptiQ-4bit",
        "num_draft_tokens": _gamma,
        "turboquant": False,
    }


def select_sentences(selection: str = "default") -> dict[str, str]:
    """Stable sentence IDs shared across configs; preserve legacy length keys."""
    if selection not in {"default", "canaries", "all"}:
        raise ValueError(f"Unknown sentence selection: {selection}")
    sentences = dict(TEST_SENTENCES) if selection in {"default", "all"} else {}
    if selection in {"canaries", "all"}:
        from training.theological_canaries import canary_sentences

        sentences.update({f"canary_{i:02d}": c["en"] for i, c in enumerate(canary_sentences(), 1)})
    return sentences


def _text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _reset_metal_peak() -> None:
    try:
        import mlx.core as mx

        mx.reset_peak_memory()
    except (ImportError, AttributeError, RuntimeError, OSError):
        pass


def _metal_peak_mib() -> float | None:
    try:
        import mlx.core as mx

        peak = mx.get_peak_memory()
        return peak / (1024 * 1024) if isinstance(peak, (int, float)) else None
    except (ImportError, AttributeError, RuntimeError, OSError):
        return None


def _identity(configs: dict[str, dict], sentence_ids: list[str]) -> dict[str, bool]:
    """Require every config and every repeat; a missing/failed arm cannot pass."""
    identity = {}
    for sentence_id in sentence_ids:
        hashes = [row.get("per_length", {}).get(sentence_id, {}).get("text_sha256s", []) for row in configs.values()]
        identity[sentence_id] = all(hashes) and len({h for group in hashes for h in group}) == 1
    return identity


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


_GENERATION_FIELDS = (
    "generated_tokens",
    "prefill_ms",
    "ttft_ms",
    "decode_ms",
    "finish_reason",
    "draft_tokens",
    "draft_accept_rate",
)


def _generation_metrics(result) -> dict[str, Any]:
    return {field: getattr(result, field, None) for field in _GENERATION_FIELDS}


def _generation_summary(rows: list[dict]) -> dict[str, Any]:
    """Aggregate known telemetry; missing observations stay unknown, never zero."""
    summary: dict[str, Any] = {}
    for field, name in (
        ("generated_tokens", "gen_tokens_mean"),
        ("prefill_ms", "prefill_p50"),
        ("ttft_ms", "ttft_p50"),
        ("decode_ms", "decode_p50"),
    ):
        values = [row[field] for row in rows if row.get(field) is not None]
        summary[name] = (
            (statistics.mean(values) if field == "generated_tokens" else _pct(values, 50)) if values else None
        )
    for field in ("draft_tokens", "draft_accept_rate"):
        values = [row[field] for row in rows if row.get(field) is not None]
        summary[f"{field}_mean"] = statistics.mean(values) if values else None
        if field == "draft_accept_rate":
            summary["draft_accept_rate_min"] = min(values) if values else None
    reasons = Counter(row["finish_reason"] for row in rows if row.get("finish_reason") is not None)
    summary["finish_reason_counts"] = dict(reasons)
    summary["pct_hit_max_tokens"] = reasons["length"] / sum(reasons.values()) if reasons else None
    return summary


def _stops_before_max(rows: list[dict]) -> bool:
    """Require observed stop reasons for all runs, including canaries."""
    return bool(rows) and all(row.get("finish_reason") == "stop" for row in rows)


def _environment_versions() -> dict[str, str | None]:
    """Read installed distribution metadata without importing ML runtimes."""
    env: dict[str, str | None] = {"python": platform.python_version()}
    for name, distribution in (("mlx", "mlx"), ("mlx_lm", "mlx-lm"), ("optiq", "mlx-optiq")):
        try:
            env[name] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            env[name] = None
    return env


def run_canaries(engine) -> dict[str, Any]:
    """Run theological canaries; return pass count + details."""
    results = []
    passed = 0
    for i, (en, expect_sub) in enumerate(CANARIES, 1):
        out = engine.translate(en, source_lang="en", target_lang="es")
        text = out.text
        ok = expect_sub.lower() in text.lower() and not _looks_like_garbage(text)
        if ok:
            passed += 1
        results.append(
            {
                "id": f"canary_{i:02d}",
                "en": en,
                "expect": expect_sub,
                "out": text,
                "text_sha256": _text_sha256(text),
                "latency_ms": out.latency_ms,
                "pass": ok,
                "garbage": _looks_like_garbage(text),
                **_generation_metrics(out),
            }
        )
    return {"passed": passed, "total": len(CANARIES), "details": results}


def bench_config(cfg_key: str, runs: int, warmup: int, sentences: str = "default") -> dict[str, Any]:
    """Load one config, warm up, measure latency + canaries + RSS."""
    from engines.mlx_engine import MLXGemmaEngine

    selected_sentences = select_sentences(sentences)
    cfg = CONFIGS[cfg_key]
    _reset_metal_peak()
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
            "metal_peak_mib": _metal_peak_mib(),
        }

    # Warmup
    for _ in range(warmup):
        engine.translate(next(iter(selected_sentences.values())), source_lang="en", target_lang="es")

    per_length: dict[str, Any] = {}
    all_generation = []
    for length, text in selected_sentences.items():
        lats: list[float] = []
        tps_list: list[float] = []
        generation = []
        texts = []
        text_sha256s = []
        sample = ""
        for _ in range(runs):
            result = engine.translate(text, source_lang="en", target_lang="es")
            if _looks_like_garbage(result.text):
                print(f"  WARNING: garbage output on {length}: {result.text[:80]!r}")
            lats.append(result.latency_ms)
            if result.tokens_per_second is not None:
                tps_list.append(result.tokens_per_second)
            text_sha256 = _text_sha256(result.text)
            generation.append({**_generation_metrics(result), "text_sha256": text_sha256})
            texts.append(result.text)
            text_sha256s.append(text_sha256)
            sample = result.text[:120]
        per_length[length] = {
            "latency": _stats(lats),
            "tps_mean": statistics.mean(tps_list) if tps_list else 0.0,
            "sample": sample,
            "source": text,
            "texts": texts,
            "text_sha256s": text_sha256s,
            "all_identical": bool(text_sha256s) and len(set(text_sha256s)) == 1,
            "generation": generation,
            **_generation_summary(generation),
        }
        all_generation.extend(generation)
        print(
            f"  {length}: p50={per_length[length]['latency']['p50']:.0f}ms "
            f"p95={per_length[length]['latency']['p95']:.0f}ms "
            f"tps={per_length[length]['tps_mean']:.1f}"
        )

    canary = run_canaries(engine)
    print(f"  canary: {canary['passed']}/{canary['total']}")

    all_generation.extend(canary["details"])
    rss_delta = _rss_mb() - rss_before
    metal_peak_mib = _metal_peak_mib()
    engine.unload()

    return {
        "config": cfg_key,
        "label": cfg["label"],
        "model_id": cfg["model_id"],
        "model_family": cfg["model_family"],
        "draft_model_id": cfg["draft_model_id"],
        "num_draft_tokens": cfg["num_draft_tokens"],
        "turboquant": cfg["turboquant"],
        "per_length": per_length,
        "canary": {"passed": canary["passed"], "total": canary["total"]},
        "canary_details": canary["details"],
        "rss_delta_mb": rss_delta,
        "metal_peak_mib": metal_peak_mib,
        "gate_stops_before_max": _stops_before_max(all_generation),
        "gate_canary_ok": canary["passed"] >= 7,
        "gate_no_garbage": not any(d.get("garbage") for d in canary["details"]),
    }


def bench_stt_baseline(runs: int, warmup: int, model_id: str, duration_s: float = 3.0) -> dict[str, Any]:
    """Primary mlx-whisper latency; a startup fallback makes this arm ineligible."""
    import numpy as np

    from engines.mlx_engine import MLXWhisperEngine
    from tools.benchmark_identity import load_primary_model

    print(f"\n=== STT {model_id} ===")
    audio = np.zeros(int(16000 * duration_s), dtype=np.float32)
    # Soft sine so VAD/energy isn't zeroed everywhere
    t = np.arange(len(audio)) / 16000.0
    audio = (0.05 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)

    engine = MLXWhisperEngine(model_id=model_id, fallback_on_low_conf=False)
    try:
        identity = load_primary_model(engine, model_id)
    except Exception as exc:
        return {"error": str(exc), "model_id": model_id, "actual_model_id": engine.model_id, "eligible": False}

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
    return {
        "model_id": model_id,
        "model_identity": identity,
        "eligible": True,
        "latency": stats,
        "note": "synthetic audio; use real clips for WER",
    }


def bench_e2e_overlap_proxy(mt_result: dict, stt_result: dict) -> dict[str, Any]:
    """Estimate cycle time with STT(N)∥MT(N−1) overlap = max(stt, mt) + other."""
    if "error" in mt_result or "error" in stt_result:
        return {"error": "missing STT or MT result"}
    if "medium" not in mt_result.get("per_length", {}):
        return {"error": "overlap proxy requires the default medium sentence"}
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
    sentences: str = "default",
) -> dict[str, Any]:
    """Entry point used by ``tools/benchmark_latency.py --only mlx-accel``."""
    selected_sentences = select_sentences(sentences)
    selected = configs or ["tg4b", "e4b", "e2b", "e4b_mts"]
    out: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "env": _environment_versions(),
        "runs": runs,
        "warmup": warmup,
        "sentences": selected_sentences,
        "configs": {},
    }

    for key in selected:
        if key not in CONFIGS:
            out["configs"][key] = {"error": f"unknown config {key}"}
            continue
        out["configs"][key] = bench_config(key, runs, warmup, sentences=sentences)

    if len(out["configs"]) > 1:
        out["identity"] = _identity(out["configs"], list(selected_sentences))

    out["gate_stops_before_max"] = bool(out["configs"]) and all(
        row.get("gate_stops_before_max", False) for row in out["configs"].values()
    )

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
    parser.add_argument("--sentences", choices=("default", "canaries", "all"), default="default")
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
        sentences=args.sentences,
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
        lengths = row.get("per_length", {})
        sentence_id = "medium" if "medium" in lengths else next(iter(lengths), "medium")
        p50 = lengths.get(sentence_id, {}).get("latency", {}).get("p50", 0)
        can = row.get("canary", {})
        gates = []
        if row.get("gate_canary_ok"):
            gates.append("canary_ok")
        if row.get("gate_no_garbage"):
            gates.append("no_garbage")
        print(
            f"  {key}: {sentence_id}_p50={p50:.0f}ms canary={can.get('passed')}/{can.get('total')} "
            f"rssΔ={row.get('rss_delta_mb', 0):.0f}MB gates={','.join(gates) or 'FAIL'}"
        )


if __name__ == "__main__":
    main()
