#!/usr/bin/env python3
"""bench_translate_t1_t4.py — Phase 1A translation benchmark for v2026.5 Gate 1A.

Runs one of four configs (T1 HF baseline, T2/T3/T4 llama.cpp variants) over
sermon chunks + theological canary sentences, emits a JSON result file.
A separate ``--config merge`` step aggregates per-config JSONs into a single
``phase1a_benchmark.json`` and prints the comparison table.

Configs:
    T1 — HF CUDAGemmaStreamingEngine + NF4, E2B target          (production baseline post-#46)
    T2 — llama.cpp, E2B Q4_K_M target, no draft                 (runtime-only swap vs T1)
    T3 — llama.cpp, E4B Q4_K_M target, no draft                 (higher-quality target)
    T4 — llama.cpp, E4B Q4_K_M target + E2B Q4_K_M draft        (spec decode, the headline)

Gate 1A: best llama.cpp config must beat T1 by >30% in tok/s with canary >= 6/8.

Usage:
    # T1 — no server needed (HF NF4 in-process)
    python bench_translate_t1_t4.py --config t1 --out metrics/phase1a_t1.json

    # T2/T3/T4 — start llama-server first, then run
    ./start_server.sh --model models/gemma-4-e2b-it-q4km.gguf --no-draft &
    python bench_translate_t1_t4.py --config t2 --out metrics/phase1a_t2.json
    # ...stop server, restart with E4B, run --config t3, etc.

    # Merge all four into one comparison file
    python bench_translate_t1_t4.py --config merge \\
        --inputs metrics/phase1a_t1.json metrics/phase1a_t2.json \\
                 metrics/phase1a_t3.json metrics/phase1a_t4.json \\
        --out metrics/phase1a_benchmark.json
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault("USE_TF", "0")

_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from training.benchmark_gemma4 import (
    CANARY_SENTENCES,
    WARMUP_SENTENCES,
    _clean_gemma4_output,
    compute_hallucination_ratio,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Config registry ─────────────────────────────────────────────────────────

CONFIGS = {
    "t1": {
        "engine": "hf",
        "model_id": "google/gemma-4-e2b-it",
        "prompt_type": "gemma4",
        "label": "T1 — HF E2B NF4 (baseline)",
    },
    "t2": {
        "engine": "llamacpp",
        "label": "T2 — llama.cpp E2B Q4_K_M",
        "expected_target": "gemma-4-e2b-it-q4km.gguf",
        "expected_draft": None,
    },
    "t3": {
        "engine": "llamacpp",
        "label": "T3 — llama.cpp E4B Q4_K_M",
        "expected_target": "gemma-4-e4b-it-q4km.gguf",
        "expected_draft": None,
    },
    "t4": {
        "engine": "llamacpp",
        "label": "T4 — llama.cpp E4B + E2B draft (spec decode)",
        "expected_target": "gemma-4-e4b-it-q4km.gguf",
        "expected_draft": "gemma-4-e2b-it-q4km.gguf",
    },
    # Extended HF configs for the 4-model matrix (TG4B vs TG12B vs E2B vs E4B):
    "tg4b_hf": {
        "engine": "hf",
        "model_id": "google/translategemma-4b-it",
        "prompt_type": "translategemma",
        "label": "TG4B — HF TranslateGemma 4B NF4",
    },
    "tg12b_hf": {
        "engine": "hf",
        "model_id": "google/translategemma-12b-it",
        "prompt_type": "translategemma",
        "label": "TG12B — HF TranslateGemma 12B NF4",
    },
    "e4b_hf": {
        "engine": "hf",
        "model_id": "google/gemma-4-e4b-it",
        "prompt_type": "gemma4",
        "label": "E4B-HF — HF Gemma 4 E4B NF4",
    },
}


# ── Test data loaders ───────────────────────────────────────────────────────


def load_sermon_chunks(path: str | Path, n: int) -> list[dict]:
    """Load first N sermon chunks from deepl_sermon_benchmark_500.jsonl."""
    chunks: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            en = row.get("en", "")
            es = row.get("es", "")
            if en and es:
                chunks.append(
                    {
                        "en": en,
                        "es_ref": es,
                        "source": row.get("chunk_source", ""),
                    }
                )
            if len(chunks) >= n:
                break
    logger.info("Loaded %d sermon chunks from %s", len(chunks), path)
    return chunks


# ── Engine adapters ─────────────────────────────────────────────────────────


class HFBaselineRunner:
    """HF runner — supports Gemma 4 instruct (E2B/E4B) and TranslateGemma chat templates."""

    def __init__(self, model_id: str, prompt_type: str = "gemma4"):
        self._model_id = model_id
        self._prompt_type = prompt_type  # "gemma4" or "translategemma"
        self._model = None
        self._tokenizer = None
        self._load_time_s = 0.0

    def load(self) -> None:
        from training.benchmark_gemma4 import load_model_nf4

        t0 = time.time()
        self._model, self._tokenizer = load_model_nf4(self._model_id)
        self._load_time_s = time.time() - t0

    def translate(self, text: str) -> tuple[str, float, int]:
        """Return (translation, latency_ms, completion_tokens)."""
        import torch

        if self._prompt_type == "translategemma":
            # TranslateGemma's dedicated chat template requires structured content
            # blocks with source_lang_code/target_lang_code fields.
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "source_lang_code": "en",
                            "target_lang_code": "es",
                            "text": text,
                        }
                    ],
                }
            ]
        else:
            # Gemma 4 generic instruct template.
            messages = [
                {
                    "role": "user",
                    "content": (
                        "Translate the following English text to Spanish. "
                        "Output only the translation, nothing else.\n\n" + text
                    ),
                }
            ]

        prompt = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        prompt_len = inputs.input_ids.shape[1]

        t0 = time.perf_counter()
        with torch.no_grad():
            output = self._model.generate(**inputs, max_new_tokens=256, do_sample=False)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        completion_tokens = int(output.shape[1] - prompt_len)
        raw = self._tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True).strip()
        if self._prompt_type == "gemma4":
            raw = _clean_gemma4_output(raw)
        return raw, latency_ms, completion_tokens

    def unload(self) -> None:
        import torch

        del self._model
        del self._tokenizer
        self._model = None
        self._tokenizer = None
        gc.collect()
        gc.collect()
        torch.cuda.empty_cache()


class LlamaCppRunner:
    """T2/T3/T4 — translate via running llama-server, return same tuple shape."""

    def __init__(self, server_url: str, max_tokens: int = 256):
        from engines.llamacpp_engine import LlamaCppEngine

        self._engine = LlamaCppEngine(
            server_url=server_url,
            model_family="gemma4",
            max_tokens=max_tokens,
            timeout_s=60.0,
        )
        self._load_time_s = 0.0
        self._target_model = ""

    def load(self) -> None:
        t0 = time.time()
        self._engine.load()
        self._load_time_s = time.time() - t0
        self._target_model = self._engine.model_id

    def translate(self, text: str) -> tuple[str, float, int]:
        """Return (translation, latency_ms, completion_tokens).

        Re-issues the request directly to capture usage.completion_tokens —
        LlamaCppEngine.translate() drops it after computing tok/s.
        """
        import urllib.request

        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Translate the following English text to Spanish. "
                        "Output only the translation, nothing else.\n\n" + text
                    ),
                }
            ],
            "max_tokens": 256,
            "temperature": 0.0,
            "stream": False,
            # Gemma 4 chat template defaults to chain-of-thought "reasoning" mode,
            # which puts the actual answer in `reasoning_content` and leaves
            # `content` empty until the model finishes thinking — but it usually
            # hits max_tokens first. Disable for translation tasks.
            "chat_template_kwargs": {"enable_thinking": False},
        }
        body = json.dumps(payload).encode("utf-8")
        url = f"{self._engine._server_url}/v1/chat/completions"
        req = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read())
        latency_ms = (time.perf_counter() - t0) * 1000.0

        choices = result.get("choices", [])
        raw = choices[0].get("message", {}).get("content", "").strip() if choices else ""
        clean = _clean_gemma4_output(raw)

        completion_tokens = int(result.get("usage", {}).get("completion_tokens", 0))
        return clean, latency_ms, completion_tokens

    def unload(self) -> None:
        self._engine.unload()


# ── Benchmark loop ──────────────────────────────────────────────────────────


def _nvidia_smi_used_mib() -> int:
    """Total VRAM used on GPU 0 per nvidia-smi (works for both PyTorch and llama-server)."""
    import subprocess

    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", "0"],
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        return int(out.decode().strip().splitlines()[0])
    except Exception:
        return 0


class VramSampler:
    """Background thread that polls nvidia-smi at fixed interval and tracks running max.

    PyTorch's max_memory_allocated() misses bnb scratch + bf16 PLE on Gemma 4, and
    is useless for out-of-process llama-server. End-of-run nvidia-smi misses
    transient KV-cache peaks that shrink back. Continuous sampling catches both.
    """

    def __init__(self, interval_s: float = 1.0):
        import threading

        self._interval = interval_s
        self._max_mib = 0
        self._samples: list[int] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        import threading

        def _loop() -> None:
            while not self._stop.is_set():
                m = _nvidia_smi_used_mib()
                if m > 0:
                    self._samples.append(m)
                    if m > self._max_mib:
                        self._max_mib = m
                self._stop.wait(self._interval)

        self._thread = threading.Thread(target=_loop, daemon=True)
        self._thread.start()

    def stop(self) -> dict:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        return {
            "max_mib": self._max_mib,
            "max_gb": round(self._max_mib / 1024, 2),
            "n_samples": len(self._samples),
            "median_mib": int(statistics.median(self._samples)) if self._samples else 0,
        }


def parse_server_timings(log_path: str | Path) -> dict:
    """Parse llama-server stdout log for per-request prompt-eval and gen-eval timings.

    Each request emits a ``slot print_timing`` block with three lines we care about:

        prompt eval time =   42.80 ms /    45 tokens (    0.95 ms per token,  1051.28 tokens per second)
               eval time =  162.08 ms /    12 tokens (   13.51 ms per token,    74.04 tokens per second)
              total time =  204.89 ms /    57 tokens

    For spec-decode runs (T4), additionally captures ``n_drafted``/``n_accept`` if
    present in the log (llama.cpp emits these in the slot release line for spec
    decode runs).

    Returns a dict with median/mean of each metric. Empty dict if no blocks parsed.
    """
    import re

    p = Path(log_path)
    if not p.exists():
        return {}

    text = p.read_text(errors="replace")

    # Extract numbers from "prompt eval time = X ms / Y tokens" / "eval time = X ms / Y tokens"
    re_prompt = re.compile(
        r"prompt eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens.*?([\d.]+)\s*tokens per second",
        re.IGNORECASE,
    )
    re_gen = re.compile(
        r"^\s*eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens.*?([\d.]+)\s*tokens per second",
        re.IGNORECASE | re.MULTILINE,
    )
    re_total = re.compile(r"total time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens")
    re_drafted = re.compile(r"n_drafted\s*=\s*(\d+)")
    re_accept = re.compile(r"n_accept\s*=\s*(\d+)")

    prompt_tps = [float(m.group(3)) for m in re_prompt.finditer(text)]
    gen_tps = [float(m.group(3)) for m in re_gen.finditer(text)]
    prompt_ms = [float(m.group(1)) for m in re_prompt.finditer(text)]
    gen_ms = [float(m.group(1)) for m in re_gen.finditer(text)]
    total_ms = [float(m.group(1)) for m in re_total.finditer(text)]
    drafted = [int(m.group(1)) for m in re_drafted.finditer(text)]
    accepted = [int(m.group(1)) for m in re_accept.finditer(text)]

    if not gen_tps:
        return {"_warning": "no slot print_timing blocks found"}

    out: dict = {
        "n_requests": len(gen_tps),
        "prompt_eval_tps_median": round(statistics.median(prompt_tps), 1) if prompt_tps else 0.0,
        "gen_eval_tps_median": round(statistics.median(gen_tps), 1) if gen_tps else 0.0,
        "gen_eval_tps_mean": round(statistics.mean(gen_tps), 1) if gen_tps else 0.0,
        "prompt_ms_median": round(statistics.median(prompt_ms), 1) if prompt_ms else 0.0,
        "gen_ms_median": round(statistics.median(gen_ms), 1) if gen_ms else 0.0,
        "total_ms_median": round(statistics.median(total_ms), 1) if total_ms else 0.0,
    }
    if drafted and accepted:
        total_drafted = sum(drafted)
        total_accepted = sum(accepted)
        out["spec_decode"] = {
            "n_drafted_total": total_drafted,
            "n_accepted_total": total_accepted,
            "acceptance_rate": round(total_accepted / total_drafted, 3) if total_drafted else 0.0,
        }
    return out


def _summarize(latencies_ms: list[float], completion_tokens: list[int]) -> dict:
    if not latencies_ms:
        return {}
    sorted_lat = sorted(latencies_ms)
    p95 = sorted_lat[max(0, int(len(sorted_lat) * 0.95) - 1)]
    total_tokens = sum(completion_tokens)
    total_seconds = sum(latencies_ms) / 1000.0
    tok_per_s = total_tokens / total_seconds if total_seconds > 0 else 0.0
    return {
        "n": len(latencies_ms),
        "latency_median_ms": round(statistics.median(latencies_ms), 1),
        "latency_p95_ms": round(p95, 1),
        "latency_mean_ms": round(statistics.mean(latencies_ms), 1),
        "completion_tokens_median": int(statistics.median(completion_tokens)) if completion_tokens else 0,
        "tokens_per_second": round(tok_per_s, 2),
    }


def run_one_config(args) -> dict:
    cfg_key = args.config
    cfg = CONFIGS[cfg_key]

    # Build runner
    if cfg["engine"] == "hf":
        runner = HFBaselineRunner(
            model_id=cfg["model_id"],
            prompt_type=cfg.get("prompt_type", "gemma4"),
        )
    else:
        runner = LlamaCppRunner(server_url=args.server_url)

    logger.info("=" * 60)
    logger.info("Config: %s", cfg["label"])
    logger.info("=" * 60)

    # Snapshot baseline VRAM before loading anything
    vram_baseline_mib = _nvidia_smi_used_mib()

    # Start continuous VRAM sampler — catches KV-cache peaks during inference
    sampler = VramSampler(interval_s=0.5)
    sampler.start()

    runner.load()
    logger.info("Loaded in %.1fs", runner._load_time_s)

    # GPU info + post-load VRAM snapshot
    vram_post_load_mib = _nvidia_smi_used_mib()
    try:
        import torch

        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        gpu_vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else 0
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        gpu_name = "unknown"
        gpu_vram_gb = 0

    # Warmup
    logger.info("Warmup (%d sentences)...", args.warmup)
    for s in WARMUP_SENTENCES[: args.warmup]:
        runner.translate(s)

    # Tier A — sermon chunks
    sermon = load_sermon_chunks(args.sermon_path, args.n_sermon)
    sermon_records: list[dict] = []
    sermon_latencies: list[float] = []
    sermon_tokens: list[int] = []
    sermon_hall: list[float] = []

    logger.info("Tier A: translating %d sermon chunks...", len(sermon))
    for i, ch in enumerate(sermon):
        hyp, lat_ms, n_tok = runner.translate(ch["en"])
        sermon_records.append(
            {
                "en": ch["en"],
                "es_ref": ch["es_ref"],
                "es_hyp": hyp,
                "source": ch["source"],
                "latency_ms": round(lat_ms, 1),
                "completion_tokens": n_tok,
            }
        )
        sermon_latencies.append(lat_ms)
        sermon_tokens.append(n_tok)
        sermon_hall.append(compute_hallucination_ratio(ch["en"], hyp))
        if (i + 1) % 25 == 0:
            logger.info("  [%d/%d] median %.0fms", i + 1, len(sermon), statistics.median(sermon_latencies))

    sermon_summary = _summarize(sermon_latencies, sermon_tokens)
    sermon_summary["hallucination_ratio_mean"] = round(statistics.mean(sermon_hall), 3) if sermon_hall else 0.0
    sermon_summary["hallucination_flagged"] = sum(1 for r in sermon_hall if r > 1.5)

    # Tier B — canary sentences
    logger.info("Tier B: %d canary sentences...", len(CANARY_SENTENCES))
    canary_records: list[dict] = []
    canary_latencies: list[float] = []
    canary_tokens: list[int] = []
    canary_hits = 0
    for en, expected in CANARY_SENTENCES:
        hyp, lat_ms, n_tok = runner.translate(en)
        hit = expected.lower() in hyp.lower()
        canary_hits += int(hit)
        canary_records.append(
            {
                "en": en,
                "expected": expected,
                "es_hyp": hyp,
                "hit": hit,
                "latency_ms": round(lat_ms, 1),
                "completion_tokens": n_tok,
            }
        )
        canary_latencies.append(lat_ms)
        canary_tokens.append(n_tok)

    canary_summary = _summarize(canary_latencies, canary_tokens)
    canary_summary["score"] = f"{canary_hits}/{len(CANARY_SENTENCES)}"
    canary_summary["score_int"] = canary_hits

    # Peak VRAM — capture both metrics
    # - torch_max_allocated: PyTorch-only, undercounts bnb scratch + bf16 PLE on Gemma 4
    # - sampler_peak (nvidia-smi): total card usage from continuous polling, works for
    #   llama-server (out-of-process) too. Catches transient KV-cache peaks.
    torch_max_alloc_gb = 0.0
    try:
        import torch

        if torch.cuda.is_available():
            torch_max_alloc_gb = round(torch.cuda.max_memory_allocated() / 1024**3, 2)
    except Exception:
        pass
    sampler_summary = sampler.stop()
    vram_peak_smi_mib = sampler_summary["max_mib"]
    vram_peak_smi_gb = sampler_summary["max_gb"]

    runner.unload()

    return {
        "config": cfg_key,
        "label": cfg["label"],
        "engine": cfg["engine"],
        "model_id": getattr(runner, "_model_id", None) or getattr(runner, "_target_model", ""),
        "load_time_s": round(runner._load_time_s, 1),
        "vram_peak_gb": vram_peak_smi_gb,
        "vram": {
            "baseline_mib": vram_baseline_mib,
            "post_load_mib": vram_post_load_mib,
            "peak_during_inference_mib": vram_peak_smi_mib,
            "peak_during_inference_gb": vram_peak_smi_gb,
            "model_resident_mib": vram_post_load_mib - vram_baseline_mib,
            "model_resident_gb": round(max(0, vram_post_load_mib - vram_baseline_mib) / 1024, 2),
            "inference_overhead_mib": max(0, vram_peak_smi_mib - vram_post_load_mib),
            "torch_max_alloc_gb": torch_max_alloc_gb,
            "n_samples": sampler_summary["n_samples"],
            "median_mib": sampler_summary["median_mib"],
            "_note": (
                "peak_during_inference_gb is the truth for design budgeting. "
                "torch_max_alloc_gb only counts PyTorch tensors and undercounts "
                "bnb NF4 scratch buffers + bf16 PLE embeddings on Gemma 4."
            ),
        },
        "hardware": {
            "gpu": gpu_name,
            "gpu_vram_total_gb": round(gpu_vram_gb, 1),
        },
        "tier_a_sermon": {
            **sermon_summary,
            "records_path": str(Path(args.out).with_suffix("")) + "_sermon.jsonl",
        },
        "tier_b_canary": {
            **canary_summary,
            "details": canary_records,
        },
        "server_timings": parse_server_timings(args.server_log) if args.server_log else {},
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }, sermon_records


def cmd_run(args) -> int:
    result, sermon_records = run_one_config(args)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    logger.info("Wrote %s", out_path)

    sermon_jsonl = out_path.with_suffix("").with_name(out_path.stem + "_sermon.jsonl")
    with open(sermon_jsonl, "w") as f:
        for r in sermon_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info("Wrote %s", sermon_jsonl)

    return 0


# ── Merge ───────────────────────────────────────────────────────────────────


def cmd_merge(args) -> int:
    inputs = [Path(p) for p in args.inputs]
    results = []
    for p in inputs:
        if not p.exists():
            logger.warning("Skipping missing input: %s", p)
            continue
        results.append(json.loads(p.read_text()))

    if not results:
        logger.error("No input result files found")
        return 1

    # Sort by sermon tok/s descending (keep T1 visible as anchor)
    results.sort(key=lambda r: r["tier_a_sermon"]["tokens_per_second"], reverse=True)

    # Decide gate vs T1 baseline
    t1 = next((r for r in results if r["config"] == "t1"), None)
    t1_tps = t1["tier_a_sermon"]["tokens_per_second"] if t1 else 0.0

    gate_rows = []
    for r in results:
        if r["config"] == "t1":
            continue
        tps = r["tier_a_sermon"]["tokens_per_second"]
        speedup = (tps / t1_tps) if t1_tps > 0 else 0.0
        canary_score = r["tier_b_canary"]["score_int"]
        passes = speedup >= 1.30 and canary_score >= 6
        gate_rows.append(
            {
                "config": r["config"],
                "label": r["label"],
                "tokens_per_second": tps,
                "speedup_vs_t1": round(speedup, 2),
                "canary_score": r["tier_b_canary"]["score"],
                "passes_gate_1a": passes,
            }
        )

    any_pass = any(g["passes_gate_1a"] for g in gate_rows)
    verdict = "PASS" if any_pass else "FAIL"

    merged = {
        "results": results,
        "gate_1a": {
            "criterion": "best llama.cpp config tok/s >= 1.30 * T1 AND canary >= 6/8",
            "verdict": verdict,
            "rows": gate_rows,
        },
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(merged, indent=2, ensure_ascii=False))
    logger.info("Wrote %s", out)

    # Pretty-print: ASCII + markdown
    cols = ("config", "tok/s", "p50 ms", "p95 ms", "VRAM GB", "Canary", "Halluc.", "compl. tok")
    rows: list[tuple] = []
    for r in results:
        a = r["tier_a_sermon"]
        b = r["tier_b_canary"]
        rows.append(
            (
                r["config"],
                f"{a['tokens_per_second']:.2f}",
                f"{a['latency_median_ms']:.0f}",
                f"{a['latency_p95_ms']:.0f}",
                f"{r['vram_peak_gb']:.2f}",
                b["score"],
                f"{a.get('hallucination_ratio_mean', 0):.3f}",
                str(a.get("completion_tokens_median", "")),
            )
        )

    widths = [max(len(c), max(len(r[i]) for r in rows)) for i, c in enumerate(cols)]
    print()
    print("ASCII:")
    sep = "  ".join("-" * w for w in widths)
    print("  ".join(c.ljust(w) for c, w in zip(cols, widths, strict=False)))
    print(sep)
    for r in rows:
        print("  ".join(c.ljust(w) for c, w in zip(r, widths, strict=False)))

    print()
    print("Markdown:")
    print("| " + " | ".join(cols) + " |")
    print("|" + "|".join("---" for _ in cols) + "|")
    for r in rows:
        print("| " + " | ".join(r) + " |")

    print()
    print(f"Gate 1A: {verdict}")
    for g in gate_rows:
        flag = "✅ PASS" if g["passes_gate_1a"] else "❌ fail"
        print(f"  {g['config']:>10}: {g['speedup_vs_t1']:>5.2f}x  canary {g['canary_score']:>3}  -> {flag}")

    return 0 if any_pass else 2


# ── CLI ─────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--config",
        choices=[*CONFIGS.keys(), "merge"],
        required=True,
        help="Which config to run, or 'merge' to aggregate.",
    )
    parser.add_argument("--out", type=str, required=True, help="Output JSON path")
    parser.add_argument(
        "--sermon-path",
        type=str,
        default="bible_data/synthetic/deepl_sermon_benchmark_500.jsonl",
        help="Sermon JSONL with en/es fields",
    )
    parser.add_argument("--n-sermon", type=int, default=125, help="Number of sermon chunks (default 125)")
    parser.add_argument("--warmup", type=int, default=2, help="Warmup translations before timing")
    parser.add_argument(
        "--server-url",
        type=str,
        default="http://127.0.0.1:8090",
        help="llama-server URL for t2/t3/t4",
    )
    parser.add_argument(
        "--server-log",
        type=str,
        default="",
        help="Path to llama-server stdout log; if set, parsed for per-request timings + spec decode acceptance",
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=[],
        help="For --config merge: list of per-config JSON files",
    )

    args = parser.parse_args()

    if args.config == "merge":
        if not args.inputs:
            parser.error("--config merge requires --inputs")
        return cmd_merge(args)
    return cmd_run(args)


if __name__ == "__main__":
    sys.exit(main())
