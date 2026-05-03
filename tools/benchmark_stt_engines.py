#!/usr/bin/env python3
"""benchmark_stt_engines.py — STT engine latency + WER + VRAM benchmark.

Production replacement for ``scripts/benchmarks/_stt_baseline.py``. Compares
faster-whisper variants (off-the-shelf vs merged W16 CT2, int8 vs int8_float16)
and HF Whisper variants (with/without spec decode, with/without torch.compile)
on a fixed canonical bench manifest.

Output:
  - ``--output FILE.json`` — per-variant summary (latency p50/p95/mean per
    duration tier, RTF, WER vs Deepgram reference, peak VRAM, hardware metadata)
  - ``--output FILE.json`` companion ``FILE_clips.jsonl`` — one record per clip
    per iteration (variant, latency_ms, prediction, ground truth, etc.)

Each invocation runs a single variant. Sweep all variants by running the script
multiple times; merge by reading the JSONs (e.g. into the v2026.7 BENCHMARK.md).

Usage (assumes the WSL training venv is active):

    python tools/benchmark_stt_engines.py --variant fw_int8float16_offshelf \\
        --manifest tools/stt_bench_manifest.json \\
        --iterations 3 --warmup 2 \\
        --output metrics/stt_phase4_fw_int8float16_offshelf.json

    # List available variants
    python tools/benchmark_stt_engines.py --list

    # Run against a custom CT2 dir (e.g. a freshly exported W16)
    python tools/benchmark_stt_engines.py --variant fw_int8float16_w16 \\
        --model-id whisper_ct2/W16_mixed_w7
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import statistics
import struct
import subprocess
import sys
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.benchmarks.vram_sampler import VramSampler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("benchmark_stt_engines")

DEFAULT_MANIFEST = PROJECT_ROOT / "tools" / "stt_bench_manifest.json"

VARIANTS: dict[str, dict] = {
    "fw_int8_offshelf": {
        "engine": "faster-whisper",
        "model_id": "large-v3-turbo",
        "compute_type": "int8",
        "device": "cuda",
        "spec_decode": False,
    },
    "fw_int8float16_offshelf": {
        "engine": "faster-whisper",
        "model_id": "large-v3-turbo",
        "compute_type": "int8_float16",
        "device": "cuda",
        "spec_decode": False,
    },
    "fw_int8_w16": {
        "engine": "faster-whisper",
        "model_id": "whisper_ct2/W16_mixed_w7",  # may be overridden via --model-id
        "compute_type": "int8",
        "device": "cuda",
        "spec_decode": False,
        "requires_local_ct2": True,
    },
    "fw_int8float16_w16": {
        "engine": "faster-whisper",
        "model_id": "whisper_ct2/W16_mixed_w7",
        "compute_type": "int8_float16",
        "device": "cuda",
        "spec_decode": False,
        "requires_local_ct2": True,
    },
    "hf_fp16_nospec": {
        "engine": "hf",
        "model_id": "openai/whisper-large-v3-turbo",
        "draft_model_id": None,
        "device": "cuda",
        "compile_mode": None,
        "warmup_seconds": 1,
        "spec_decode": False,
    },
    "hf_fp16_nospec_compiled": {
        "engine": "hf",
        "model_id": "openai/whisper-large-v3-turbo",
        "draft_model_id": None,
        "device": "cuda",
        "compile_mode": "reduce-overhead",
        "warmup_seconds": 2,
        "spec_decode": False,
    },
    # Spec decode against turbo (4-layer decoder) is broken with distil-v3.5
    # (32-layer decoder) — see docs/archive/v2026.5/spec_decode_research.md.
    # The verified-compatible pairing is target=large-v3 (32 layers) + draft=turbo
    # (4 layers), which trades latency for quality. Uncomment if benching that
    # path; not enabled by default since it's a quality study, not a speed win.
    # "hf_fp16_spec_largev3": {
    #     "engine": "hf",
    #     "model_id": "openai/whisper-large-v3",
    #     "draft_model_id": "openai/whisper-large-v3-turbo",
    #     "device": "cuda",
    #     "compile_mode": None,
    #     "warmup_seconds": 1,
    #     "spec_decode": True,
    # },
}


# ---------------------------------------------------------------------------
# Audio loader (self-contained 16-bit PCM WAV reader to avoid scipy import)
# ---------------------------------------------------------------------------


def load_pcm16_wav(path: Path) -> tuple[np.ndarray, int]:
    """Return (float32 in [-1, 1], sample_rate) for a 16-bit PCM WAV."""
    with path.open("rb") as fh:
        riff = fh.read(12)
        if riff[:4] != b"RIFF" or riff[8:12] != b"WAVE":
            raise ValueError(f"not RIFF/WAVE: {path}")
        sample_rate = 16000
        bits_per_sample = 16
        num_channels = 1
        data = b""
        while True:
            hdr = fh.read(8)
            if len(hdr) < 8:
                break
            chunk_id, chunk_size = struct.unpack("<4sI", hdr)
            if chunk_id == b"fmt ":
                fmt = fh.read(chunk_size)
                _af, num_channels, sample_rate, _br, _ba, bits_per_sample = struct.unpack("<HHIIHH", fmt[:16])
            elif chunk_id == b"data":
                data = fh.read(chunk_size)
                break
            else:
                fh.seek(chunk_size, 1)
    if not data or bits_per_sample != 16:
        raise ValueError(f"unsupported WAV: {path}")
    pcm = np.frombuffer(data, dtype=np.int16)
    if num_channels > 1:
        pcm = pcm.reshape(-1, num_channels).mean(axis=1).astype(np.int16)
    return pcm.astype(np.float32) / 32768.0, sample_rate


# ---------------------------------------------------------------------------
# Text normalization + WER
# ---------------------------------------------------------------------------


def normalize_text(text: str) -> str:
    try:
        from whisper.normalizers import EnglishTextNormalizer

        return EnglishTextNormalizer()(text)
    except Exception:
        import re

        out = re.sub(r"[^\w\s']", " ", text.lower())
        return re.sub(r"\s+", " ", out).strip()


def wer(refs: list[str], preds: list[str]) -> float:
    import jiwer

    pairs = [(normalize_text(r), normalize_text(p)) for r, p in zip(refs, preds)]
    pairs = [(r, p) for r, p in pairs if r]
    if not pairs:
        return float("nan")
    refs_clean, preds_clean = zip(*pairs)
    return float(jiwer.wer(list(refs_clean), list(preds_clean)))


# ---------------------------------------------------------------------------
# Engine construction
# ---------------------------------------------------------------------------


def build_engine(variant_key: str, variant: dict, override_model_id: str | None) -> object:
    if variant["engine"] == "faster-whisper":
        from engines.cuda_engine import FasterWhisperEngine

        model_id = override_model_id or variant["model_id"]
        if variant.get("requires_local_ct2") and not Path(model_id).exists():
            raise SystemExit(
                f"variant {variant_key!r} requires a local CT2 model at {model_id}. "
                f"Run training/export_ct2.py first or pass --model-id."
            )
        return FasterWhisperEngine(
            model_id=model_id,
            compute_type=variant["compute_type"],
            device=variant["device"],
            fallback_on_low_conf=False,  # benchmark measures the primary path only
        )
    if variant["engine"] == "hf":
        from engines.hf_whisper_engine import HFWhisperEngine

        return HFWhisperEngine(
            model_id=override_model_id or variant["model_id"],
            draft_model_id=variant.get("draft_model_id"),
            device=variant["device"],
            compile_mode=variant.get("compile_mode"),
            warmup_seconds=variant.get("warmup_seconds", 1),
        )
    raise ValueError(f"unknown engine in variant {variant_key!r}: {variant['engine']!r}")


# ---------------------------------------------------------------------------
# Hardware metadata
# ---------------------------------------------------------------------------


def collect_hardware_info() -> dict:
    info = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cpu": platform.processor() or platform.machine(),
    }
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total,compute_cap",
                "--format=csv,noheader",
            ],
            timeout=5,
            stderr=subprocess.DEVNULL,
        )
        line = out.decode().strip().splitlines()[0]
        parts = [p.strip() for p in line.split(",")]
        info["gpu_name"] = parts[0] if len(parts) > 0 else None
        info["driver_version"] = parts[1] if len(parts) > 1 else None
        info["total_vram_mib"] = int(parts[2].split()[0]) if len(parts) > 2 else None
        info["gpu_compute_capability"] = parts[3] if len(parts) > 3 else None
    except Exception as exc:
        info["gpu_query_error"] = str(exc)
    try:
        import torch

        info["torch_version"] = torch.__version__
        info["cuda_runtime"] = torch.version.cuda
    except Exception:
        pass
    try:
        import psutil

        info["ram_gb"] = round(psutil.virtual_memory().total / 1024**3, 1)
    except Exception:
        pass
    return info


# ---------------------------------------------------------------------------
# Bench loop
# ---------------------------------------------------------------------------


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    idx = max(0, min(len(s) - 1, round(q * (len(s) - 1))))
    return s[idx]


def run_variant(
    variant_key: str,
    variant: dict,
    clips: list[dict],
    iterations: int,
    warmup: int,
    clips_jsonl: Path,
    override_model_id: str | None,
) -> dict:
    log.info("=== variant %s ===", variant_key)
    log.info("config: %s", variant)

    sampler = VramSampler(interval_s=0.5)
    sampler.start()

    t_load_start = time.perf_counter()
    engine = build_engine(variant_key, variant, override_model_id)
    engine.load()
    load_seconds = time.perf_counter() - t_load_start
    log.info("loaded in %.1fs", load_seconds)

    for w in range(warmup):
        log.info("warmup pass %d/%d", w + 1, warmup)
        for clip in clips:
            audio, _sr = load_pcm16_wav(PROJECT_ROOT / clip["audio_path_relative"])
            engine.transcribe(audio, language="en", beam_size=1)

    per_clip_records: list[dict] = []
    tier_latencies: dict[str, list[float]] = defaultdict(list)
    tier_rtfs: dict[str, list[float]] = defaultdict(list)
    refs: list[str] = []
    preds: list[str] = []
    tier1_refs: list[str] = []
    tier1_preds: list[str] = []
    cold_start_ms: float | None = None

    for it in range(iterations):
        log.info("measurement pass %d/%d", it + 1, iterations)
        for i, clip in enumerate(clips):
            audio, _sr = load_pcm16_wav(PROJECT_ROOT / clip["audio_path_relative"])
            t0 = time.perf_counter()
            result = engine.transcribe(audio, language="en")
            wall_ms = (time.perf_counter() - t0) * 1000.0
            duration_ms = clip["duration_s"] * 1000.0
            rtf = wall_ms / duration_ms if duration_ms else float("nan")
            if it == 0 and i == 0:
                cold_start_ms = wall_ms
            tier_latencies[clip["tier"]].append(wall_ms)
            tier_rtfs[clip["tier"]].append(rtf)
            refs.append(clip["ground_truth"])
            preds.append(result.text)
            if clip["tier1_terms_present"]:
                tier1_refs.append(clip["ground_truth"])
                tier1_preds.append(result.text)
            per_clip_records.append(
                {
                    "variant": variant_key,
                    "iteration": it + 1,
                    "clip_id": clip["id"],
                    "tier": clip["tier"],
                    "duration_s": clip["duration_s"],
                    "latency_ms": round(wall_ms, 2),
                    "rtf": round(rtf, 3),
                    "prediction": result.text,
                    "ground_truth": clip["ground_truth"],
                    "tier1_terms": clip["tier1_terms_present"],
                    "engine_avg_logprob": result.avg_logprob,
                    "engine_compression_ratio": result.compression_ratio,
                }
            )

    engine.unload()
    vram = sampler.stop()

    summary: dict = {
        "variant": variant_key,
        "config": variant,
        "load_seconds": round(load_seconds, 2),
        "iterations": iterations,
        "warmup": warmup,
        "vram": vram,
        "wer_normalized_overall": round(wer(refs, preds), 4),
        "wer_normalized_tier1_only": round(wer(tier1_refs, tier1_preds), 4) if tier1_refs else None,
        "cold_start_ms": round(cold_start_ms, 2) if cold_start_ms is not None else None,
        "tiers": {},
    }
    # Aggregate latency across all tiers for the headline number.
    all_latencies = [lat for ls in tier_latencies.values() for lat in ls]
    if all_latencies:
        summary["latency_ms_overall_mean"] = round(statistics.mean(all_latencies), 2)
        summary["latency_ms_overall_median"] = round(statistics.median(all_latencies), 2)
        summary["latency_ms_overall_p95"] = round(percentile(all_latencies, 0.95), 2)
    for tier in ("short", "medium", "long"):
        ls = tier_latencies[tier]
        rs = tier_rtfs[tier]
        if not ls:
            continue
        summary["tiers"][tier] = {
            "n_measurements": len(ls),
            "latency_ms_mean": round(statistics.mean(ls), 2),
            "latency_ms_median": round(statistics.median(ls), 2),
            "latency_ms_p95": round(percentile(ls, 0.95), 2),
            "latency_ms_min": round(min(ls), 2),
            "latency_ms_max": round(max(ls), 2),
            "rtf_median": round(statistics.median(rs), 3),
            "rtf_p95": round(percentile(rs, 0.95), 3),
        }

    with clips_jsonl.open("a") as fh:
        for rec in per_clip_records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def list_variants() -> None:
    print("Available variants:")
    for key, cfg in VARIANTS.items():
        engine = cfg["engine"]
        compute = cfg.get("compute_type", "")
        spec = " spec" if cfg.get("spec_decode") else ""
        compile_mode = f" compile={cfg.get('compile_mode')}" if cfg.get("compile_mode") else ""
        local = " (needs local CT2)" if cfg.get("requires_local_ct2") else ""
        print(f"  {key:30s}  {engine:14s} {compute}{spec}{compile_mode}{local}")


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--variant", type=str, default=None, choices=list(VARIANTS.keys()))
    p.add_argument("--list", action="store_true", help="list available variants and exit")
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--output", type=Path, default=None, help="JSON output (default: metrics/stt_bench_<variant>.json)")
    p.add_argument("--iterations", type=int, default=3)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument(
        "--model-id",
        type=str,
        default=None,
        help="Override the variant's default model_id (useful for fw_int8float16_w16 etc.)",
    )
    p.add_argument("--quiet", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    if args.list:
        list_variants()
        return 0
    if args.variant is None:
        log.error("--variant is required (use --list to see options)")
        return 2
    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)

    output = args.output or PROJECT_ROOT / "metrics" / f"stt_bench_{args.variant}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    clips_jsonl = output.with_suffix("").with_name(output.stem + "_clips.jsonl")
    if clips_jsonl.exists():
        clips_jsonl.unlink()

    manifest = json.loads(args.manifest.read_text())
    clips = manifest["clips"]
    log.info("loaded manifest with %d clips (tier1 chunks: %d)", len(clips), manifest.get("tier1_clip_count", 0))

    hardware = collect_hardware_info()
    log.info("hardware: %s (%s GB)", hardware.get("gpu_name"), hardware.get("total_vram_mib"))

    started = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    summary = run_variant(
        args.variant,
        VARIANTS[args.variant],
        clips,
        iterations=args.iterations,
        warmup=args.warmup,
        clips_jsonl=clips_jsonl,
        override_model_id=args.model_id,
    )
    finished = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    output_doc = {
        "version": "1.0",
        "tool": "tools/benchmark_stt_engines.py",
        "variant": args.variant,
        "started_at_utc": started,
        "finished_at_utc": finished,
        "manifest_path": str(args.manifest.relative_to(PROJECT_ROOT)),
        "manifest_total_clips": manifest["total_clips"],
        "iterations": args.iterations,
        "warmup": args.warmup,
        "hardware": hardware,
        "summary": summary,
    }
    output.write_text(json.dumps(output_doc, indent=2, ensure_ascii=False))
    log.info("wrote %s + %s", output, clips_jsonl)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
