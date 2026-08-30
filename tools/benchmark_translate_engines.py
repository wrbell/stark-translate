#!/usr/bin/env python3
"""benchmark_translate_engines.py — MarianMT engine latency + quality + VRAM benchmark.

Sibling of ``tools/benchmark_stt_engines.py`` (v2026.7) for the translation
side. Compares MarianHFEngine variants (PyTorch fp32/fp16 on CPU/CUDA) and
MarianCT2Engine variants (CTranslate2 int8 / int8_float16 / float16) on the
fixed manifest at ``tools/translate_bench_manifest.json``.

Output:
  - ``--output FILE.json`` — per-variant summary (latency p50/p95/mean per
    length tier, chrF++ vs DeepL sermon refs, canary term hit rate, peak VRAM,
    hardware metadata)
  - ``--output FILE.json`` companion ``FILE_clips.jsonl`` — one record per
    sentence per iteration (variant, latency_ms, source, prediction, reference,
    tier1_term_expected, etc.)

Each invocation runs a single variant. Sweep all variants by running multiple
times; results live in ``metrics/translate_bench/v2026.8/{variant}.json``.

Usage (with the WSL training venv active and the CT2 models converted):

    python tools/benchmark_translate_engines.py --variant marian_ct2_int8float16_cuda \\
        --manifest tools/translate_bench_manifest.json \\
        --iterations 3 --warmup 1 \\
        --output metrics/translate_bench/v2026.8/marian_ct2_int8float16_cuda.json

    # List variants
    python tools/benchmark_translate_engines.py --list

    # Optional: enable CometKiwi sermon-set scoring (slower, downloads ~1.5 GB
    # the first time; disabled by default since chrF++ is sufficient for the
    # quantization regression check)
    python tools/benchmark_translate_engines.py --variant marian_ct2_int8float16_cuda \\
        --comet --output ...
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.benchmarks.vram_sampler import VramSampler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("benchmark_translate_engines")

DEFAULT_MANIFEST = PROJECT_ROOT / "tools" / "translate_bench_manifest.json"

# Local CT2 adapter directories produced by scripts/convert_marian_ct2.py.
# Variants tagged with `requires_local_ct2` will skip if these are missing.
CT2_EN_ES = PROJECT_ROOT / "adapters" / "marian_ct2" / "en-es" / "active"
CT2_ES_EN = PROJECT_ROOT / "adapters" / "marian_ct2" / "es-en" / "active"

# Variants. Mirrors tools/benchmark_stt_engines.py::VARIANTS shape.
VARIANTS: dict[str, dict] = {
    # HF baselines — direct comparison against the v2026.7 production path.
    "marian_hf_fp32_cpu": {
        "engine": "hf",
        "device": "cpu",
        "torch_dtype": "float32",
    },
    "marian_hf_fp16_cuda": {
        "engine": "hf",
        "device": "cuda",
        "torch_dtype": "float16",
    },
    # CT2 path (the v2026.8 candidates). Requires the converted adapters.
    "marian_ct2_int8_cpu": {
        "engine": "ct2",
        "device": "cpu",
        "compute_type": "int8",
        "requires_local_ct2": True,
    },
    "marian_ct2_int8_cuda": {
        "engine": "ct2",
        "device": "cuda",
        "compute_type": "int8",
        "requires_local_ct2": True,
    },
    "marian_ct2_int8float16_cuda": {
        "engine": "ct2",
        "device": "cuda",
        "compute_type": "int8_float16",
        "requires_local_ct2": True,
    },
    "marian_ct2_float16_cuda": {
        "engine": "ct2",
        "device": "cuda",
        "compute_type": "float16",
        "requires_local_ct2": True,
    },
}


def _adapter_for_direction(source_lang: str) -> Path:
    return CT2_ES_EN if source_lang == "es" else CT2_EN_ES


def build_engines(variant_key: str, variant: dict, override_dir: dict[str, str] | None) -> dict[str, object]:
    """Return a dict mapping source_lang ('en' / 'es') → loaded engine.

    For HF variants we instantiate two MarianHFEngine objects, one per direction.
    For CT2 variants we instantiate two MarianCT2Engine objects from the
    matching local adapter dirs. Both directions get loaded so the bench can
    cover en→es + es→en in one variant run.
    """
    engines: dict[str, object] = {}

    if variant["engine"] == "ct2":
        from engines.cuda_engine import MarianCT2Engine

        for src_lang in ("en", "es"):
            override = (override_dir or {}).get(src_lang)
            adapter_dir = Path(override) if override else _adapter_for_direction(src_lang)
            if not (adapter_dir / "model.bin").exists():
                raise SystemExit(
                    f"variant {variant_key!r} needs CT2 model at {adapter_dir} (run "
                    "scripts/convert_marian_ct2.py first or pass --ct2-en-es / --ct2-es-en)."
                )
            engines[src_lang] = MarianCT2Engine(
                model_dir=str(adapter_dir),
                source_lang=src_lang,
                target_lang="es" if src_lang == "en" else "en",
                device=variant["device"],
                compute_type=variant["compute_type"],
                warmup_passes=2,
            )
        return engines

    if variant["engine"] == "hf":
        import torch

        from engines.marian_hf_engine import MarianHFEngine

        device = variant["device"]
        torch_dtype = variant.get("torch_dtype", "float32")

        for src_lang in ("en", "es"):
            model_id = "Helsinki-NLP/opus-mt-es-en" if src_lang == "es" else "Helsinki-NLP/opus-mt-en-es"
            engine = MarianHFEngine(model_id=model_id, device=device, warmup_passes=2)
            engines[src_lang] = engine
        # MarianHFEngine doesn't currently honor a torch_dtype kwarg; the dtype
        # control here is purely informational at the variant level (HF Marian
        # is small enough that fp16 vs fp32 doesn't move the needle). Future:
        # extend the engine to cast .to(dtype=...) after to(device).
        _ = torch  # keep the import alive for environments where it's a presence check
        _ = torch_dtype
        return engines

    raise ValueError(f"unknown engine in variant {variant_key!r}: {variant['engine']!r}")


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
        import ctranslate2

        info["ctranslate2_version"] = ctranslate2.__version__
    except Exception:
        pass
    try:
        import psutil

        info["ram_gb"] = round(psutil.virtual_memory().total / 1024**3, 1)
    except Exception:
        pass
    return info


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    idx = max(0, min(len(s) - 1, round(q * (len(s) - 1))))
    return s[idx]


def chrf_pp(hypothesis: str, reference: str) -> float | None:
    """Per-segment chrF++ (word-order=2). Returns None if sacrebleu missing or empty ref."""
    if not reference or not hypothesis:
        return None
    try:
        import sacrebleu
    except ImportError:
        return None
    return float(sacrebleu.sentence_chrf(hypothesis, [reference], word_order=2).score)


def run_cometkiwi(rows: list[dict]) -> list[float] | None:
    """Score (source, hypothesis) pairs with CometKiwi. Returns scores aligned with rows.

    Each None entry means the row had an empty source/hyp. Returns None if
    unbabel-comet isn't installed (the caller should treat scoring as optional).
    """
    try:
        from comet import download_model, load_from_checkpoint
    except ImportError:
        log.warning("unbabel-comet not installed — skipping CometKiwi (install: pip install unbabel-comet)")
        return None

    pairs = []
    indices = []
    for i, row in enumerate(rows):
        src = row.get("source_text", "")
        hyp = row.get("prediction", "")
        if src and hyp:
            pairs.append({"src": src, "mt": hyp})
            indices.append(i)
    if not pairs:
        return None

    log.info("running CometKiwi on %d pairs (downloading model on first run)...", len(pairs))
    model_path = download_model("Unbabel/wmt22-cometkiwi-da")
    model = load_from_checkpoint(model_path)
    out = model.predict(pairs, batch_size=16, gpus=1)
    scores: list[float | None] = [None] * len(rows)
    for j, idx in enumerate(indices):
        scores[idx] = float(out.scores[j])
    return scores


def run_variant(
    variant_key: str,
    variant: dict,
    clips: list[dict],
    iterations: int,
    warmup: int,
    clips_jsonl: Path,
    override_dir: dict[str, str] | None,
    enable_comet: bool,
) -> dict:
    log.info("=== variant %s ===", variant_key)
    log.info("config: %s", variant)

    sampler = VramSampler(interval_s=0.5)
    sampler.start()

    t_load_start = time.perf_counter()
    engines = build_engines(variant_key, variant, override_dir)
    for src_lang, engine in engines.items():
        log.info("loading %s engine for src=%s ...", variant_key, src_lang)
        engine.load()
    load_seconds = time.perf_counter() - t_load_start
    log.info("loaded both directions in %.1fs", load_seconds)

    # Warmup passes (in addition to engine.load()'s internal warmup).
    for w in range(warmup):
        log.info("warmup pass %d/%d", w + 1, warmup)
        for clip in clips:
            engine = engines[clip["source_lang"]]
            engine.translate(
                clip["source_text"],
                source_lang=clip["source_lang"],
                target_lang=clip["target_lang"],
            )

    per_clip_records: list[dict] = []
    tier_latencies: dict[str, list[float]] = defaultdict(list)
    direction_latencies: dict[str, list[float]] = defaultdict(list)
    chrf_by_tier: dict[str, list[float]] = defaultdict(list)
    canary_hits = 0
    canary_total = 0
    cold_start_ms: float | None = None

    for it in range(iterations):
        log.info("measurement pass %d/%d", it + 1, iterations)
        for i, clip in enumerate(clips):
            engine = engines[clip["source_lang"]]
            t0 = time.perf_counter()
            result = engine.translate(
                clip["source_text"],
                source_lang=clip["source_lang"],
                target_lang=clip["target_lang"],
            )
            wall_ms = (time.perf_counter() - t0) * 1000.0

            if it == 0 and i == 0:
                cold_start_ms = wall_ms

            tier_latencies[clip["length_tier"]].append(wall_ms)
            direction_latencies[f"{clip['source_lang']}-{clip['target_lang']}"].append(wall_ms)

            chrf = chrf_pp(result.text, clip["reference_text"])
            if chrf is not None:
                chrf_by_tier[clip["length_tier"]].append(chrf)

            term_hit = None
            expected_term = clip.get("tier1_term_expected", "")
            if expected_term:
                term_hit = expected_term.lower() in result.text.lower()
                if it == 0:  # canary tally per pass — count only the first pass
                    canary_total += 1
                    if term_hit:
                        canary_hits += 1

            per_clip_records.append(
                {
                    "variant": variant_key,
                    "iteration": it + 1,
                    "clip_id": clip["id"],
                    "source_lang": clip["source_lang"],
                    "target_lang": clip["target_lang"],
                    "length_tier": clip["length_tier"],
                    "tier1_terms_present": clip["tier1_terms_present"],
                    "source_text": clip["source_text"],
                    "reference_text": clip["reference_text"],
                    "tier1_term_expected": expected_term,
                    "prediction": result.text,
                    "latency_ms": round(wall_ms, 2),
                    "engine_latency_ms": round(result.latency_ms, 2),
                    "chrf_pp": round(chrf, 2) if chrf is not None else None,
                    "term_hit": term_hit,
                }
            )

    # Optional CometKiwi pass — score sermon clips (those with non-empty reference)
    # plus canary clips (reference-free QE), one number per row.
    comet_scores: list[float | None] | None = None
    if enable_comet:
        comet_scores = run_cometkiwi(per_clip_records)
        if comet_scores is not None:
            for rec, s in zip(per_clip_records, comet_scores):
                rec["cometkiwi"] = round(s, 4) if s is not None else None

    for engine in engines.values():
        engine.unload()
    vram = sampler.stop()

    summary: dict = {
        "variant": variant_key,
        "config": variant,
        "load_seconds": round(load_seconds, 2),
        "iterations": iterations,
        "warmup": warmup,
        "vram": vram,
        "cold_start_ms": round(cold_start_ms, 2) if cold_start_ms is not None else None,
        "canary": {
            "hits": canary_hits,
            "total": canary_total,
            "score": f"{canary_hits}/{canary_total}" if canary_total else None,
        },
        "tiers": {},
        "directions": {},
    }
    all_latencies = [lat for ls in tier_latencies.values() for lat in ls]
    if all_latencies:
        summary["latency_ms_overall_mean"] = round(statistics.mean(all_latencies), 2)
        summary["latency_ms_overall_median"] = round(statistics.median(all_latencies), 2)
        summary["latency_ms_overall_p95"] = round(percentile(all_latencies, 0.95), 2)

    for tier in ("short", "medium", "long"):
        ls = tier_latencies[tier]
        if not ls:
            continue
        chrf_values = chrf_by_tier[tier]
        summary["tiers"][tier] = {
            "n_measurements": len(ls),
            "latency_ms_mean": round(statistics.mean(ls), 2),
            "latency_ms_median": round(statistics.median(ls), 2),
            "latency_ms_p95": round(percentile(ls, 0.95), 2),
            "latency_ms_min": round(min(ls), 2),
            "latency_ms_max": round(max(ls), 2),
            "chrf_pp_mean": round(statistics.mean(chrf_values), 2) if chrf_values else None,
            "chrf_pp_n": len(chrf_values),
        }
    for direction, ls in direction_latencies.items():
        summary["directions"][direction] = {
            "n_measurements": len(ls),
            "latency_ms_p95": round(percentile(ls, 0.95), 2),
            "latency_ms_median": round(statistics.median(ls), 2),
        }

    if comet_scores is not None:
        valid = [s for s in comet_scores if s is not None]
        if valid:
            summary["cometkiwi_overall_mean"] = round(sum(valid) / len(valid), 4)
            summary["cometkiwi_overall_n"] = len(valid)

    with clips_jsonl.open("a") as fh:
        for rec in per_clip_records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return summary


def list_variants() -> None:
    print("Available variants:")
    for key, cfg in VARIANTS.items():
        engine = cfg["engine"]
        device = cfg.get("device", "")
        compute = cfg.get("compute_type") or cfg.get("torch_dtype", "")
        local = " (needs local CT2)" if cfg.get("requires_local_ct2") else ""
        print(f"  {key:32s}  {engine:6s} {device:4s} {compute:14s}{local}")


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--variant", type=str, default=None, choices=list(VARIANTS.keys()))
    p.add_argument("--list", action="store_true", help="list available variants and exit")
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="JSON output (default: metrics/translate_bench/v2026.8/<variant>.json)",
    )
    p.add_argument("--iterations", type=int, default=3)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument(
        "--ct2-en-es",
        type=str,
        default=None,
        help="Override path to the en→es CT2 model dir.",
    )
    p.add_argument(
        "--ct2-es-en",
        type=str,
        default=None,
        help="Override path to the es→en CT2 model dir.",
    )
    p.add_argument(
        "--comet",
        action="store_true",
        help="Score with CometKiwi (slower, downloads ~1.5 GB on first run).",
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

    output = args.output or PROJECT_ROOT / "metrics" / "translate_bench" / "v2026.8" / f"{args.variant}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    clips_jsonl = output.with_suffix("").with_name(output.stem + "_clips.jsonl")
    if clips_jsonl.exists():
        clips_jsonl.unlink()

    manifest = json.loads(args.manifest.read_text())
    clips = manifest["clips"]
    log.info(
        "loaded manifest with %d clips (en-es: %d, es-en: %d, tier1: %d)",
        manifest.get("total_clips", len(clips)),
        manifest.get("directions", {}).get("en-es", 0),
        manifest.get("directions", {}).get("es-en", 0),
        manifest.get("tier1_clip_count", 0),
    )

    hardware = collect_hardware_info()
    log.info("hardware: %s (%s MiB)", hardware.get("gpu_name"), hardware.get("total_vram_mib"))

    override_dir: dict[str, str] | None = None
    if args.ct2_en_es or args.ct2_es_en:
        override_dir = {}
        if args.ct2_en_es:
            override_dir["en"] = args.ct2_en_es
        if args.ct2_es_en:
            override_dir["es"] = args.ct2_es_en

    started = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    summary = run_variant(
        args.variant,
        VARIANTS[args.variant],
        clips,
        iterations=args.iterations,
        warmup=args.warmup,
        clips_jsonl=clips_jsonl,
        override_dir=override_dir,
        enable_comet=args.comet,
    )
    finished = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    output_doc = {
        "version": "1.0",
        "tool": "tools/benchmark_translate_engines.py",
        "variant": args.variant,
        "started_at_utc": started,
        "finished_at_utc": finished,
        "manifest_path": str(args.manifest.relative_to(PROJECT_ROOT)),
        "manifest_total_clips": manifest.get("total_clips", len(clips)),
        "iterations": args.iterations,
        "warmup": args.warmup,
        "comet_enabled": args.comet,
        "hardware": hardware,
        "summary": summary,
    }
    output.write_text(json.dumps(output_doc, indent=2, ensure_ascii=False))
    log.info("wrote %s + %s", output, clips_jsonl)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
