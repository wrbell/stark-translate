#!/usr/bin/env python3
"""
benchmark_gemma4.py — Benchmark Gemma 4 vs TranslateGemma on EN→ES translation.

Evaluates 4 models (TranslateGemma 4B/12B, Gemma 4 E2B/E4B) across 3 test tiers:
  - Tier 1: Bible verse holdout (formal register, with ES references → BLEU/chrF++/COMET)
  - Tier 2: Deepgram sermon chunks (informal register, no reference → COMET-QE + hallucination)
  - Tier 3: Theological canary sentences (informal, expected terms → term accuracy)

Usage:
    # Full benchmark
    python training/benchmark_gemma4.py

    # Quick smoke test
    python training/benchmark_gemma4.py --models tg4b --max-samples 5 --skip-comet

    # Specific models
    python training/benchmark_gemma4.py --models tg4b,e4b --output-dir metrics/gemma4_benchmark/
"""

import argparse
import gc
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

os.environ["USE_TF"] = "0"

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Model registry ──────────────────────────────────────────────────────────

MODEL_REGISTRY = {
    "tg4b": {
        "model_id": "google/translategemma-4b-it",
        "display_name": "TranslateGemma 4B",
        "prompt_type": "translategemma",
    },
    "tg12b": {
        "model_id": "google/translategemma-12b-it",
        "display_name": "TranslateGemma 12B",
        "prompt_type": "translategemma",
    },
    "e2b": {
        "model_id": "google/gemma-4-e2b-it",
        "display_name": "Gemma 4 E2B",
        "prompt_type": "gemma4_instruct",
    },
    "e4b": {
        "model_id": "google/gemma-4-e4b-it",
        "display_name": "Gemma 4 E4B",
        "prompt_type": "gemma4_instruct",
    },
}

# Theological canary sentences (Tier 3) — from evaluate_sermon.py
CANARY_SENTENCES = [
    ("And you know, when we think about the atonement — what Christ did for us on that cross...", "expiación"),
    ("God made a covenant with Abraham, and brothers, He keeps His promises.", "pacto"),
    ("It's only by grace, friends. We can't earn it — it's grace through faith.", "gracia"),
    ("And Paul writes about the righteousness of God here in Romans, and what does he mean by that?", "justicia"),
    ("Now, if you turn to the book of James — James has a lot to say about faith and works.", "Santiago"),
    ("You think about James and John, just regular fishermen on the Sea of Galilee.", "Jacobo"),
    ("The speaker tonight was talking about sanctification — what it means to be set apart.", "santificación"),
    ("Let us remember the Lord at the breaking of bread this morning.", "partimiento del pan"),
]

WARMUP_SENTENCES = [
    "The atonement of Christ reconciles us to God.",
    "James wrote about faith and works.",
    "The propitiation for our sins was the blood of Christ.",
]


# ── Model loading / unloading ───────────────────────────────────────────────


def load_model_nf4(model_id):
    """Load a HuggingFace model with NF4 quantization."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
    )
    model.eval()

    # EOS fix: TranslateGemma (and possibly Gemma 4) uses <end_of_turn> as
    # actual EOS but tokenizer defaults to <eos>. Without fix, model generates
    # 256 pad tokens wasting ~5s.
    eot_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
    if eot_id is not None and eot_id != tokenizer.unk_token_id:
        tokenizer._eos_token_ids = {tokenizer.eos_token_id, eot_id}

    return model, tokenizer


def unload_model(model, tokenizer):
    """Free GPU memory between model evaluations."""
    import torch

    del model
    del tokenizer
    gc.collect()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


# ── Translation functions ───────────────────────────────────────────────────


def translate_translategemma(model, tokenizer, text):
    """Translate using TranslateGemma's dedicated chat template."""
    import torch

    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "source_lang_code": "en", "target_lang_code": "es", "text": text}],
        }
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    return tokenizer.decode(output[0][inputs.input_ids.shape[1] :], skip_special_tokens=True).strip()


def _clean_gemma4_output(text):
    """Strip common instruct-model preambles from translation output."""
    prefixes = [
        "Translation:",
        "Here is the translation:",
        "Here's the translation:",
        "Traducción:",
        "Output:",
        "Spanish:",
        "The translation is:",
    ]
    for prefix in prefixes:
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix) :].strip()
    if len(text) >= 2 and text[0] in ('"', "'", "\u201c") and text[-1] in ('"', "'", "\u201d"):
        text = text[1:-1].strip()
    return text


def translate_gemma4(model, tokenizer, text):
    """Translate using Gemma 4 generic instruct template."""
    import torch

    messages = [
        {
            "role": "user",
            "content": (
                "Translate the following English text to Spanish. Output only the translation, nothing else.\n\n" + text
            ),
        }
    ]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    raw = tokenizer.decode(output[0][inputs.input_ids.shape[1] :], skip_special_tokens=True).strip()
    return _clean_gemma4_output(raw)


# ── Test data loading ───────────────────────────────────────────────────────


def load_verse_pairs(path, max_samples=None):
    """Load verse pairs from JSONL. Returns list of (en, es) tuples."""
    pairs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            en = record.get("source_text", record.get("en", ""))
            es = record.get("target_text", record.get("es", ""))
            if en and es:
                pairs.append((en, es))
    if max_samples and len(pairs) > max_samples:
        random.seed(42)
        pairs = random.sample(pairs, max_samples)
    logger.info("Loaded %d verse pairs from %s", len(pairs), path)
    return pairs


def load_sermon_chunks(deepgram_dir, num_per_sermon=25, min_words=20, seed=42):
    """Load Deepgram-transcribed sermon chunks as real-world test data.

    Returns list of dicts with 'text' and 'source' keys.
    """
    random.seed(seed)
    chunks = []
    dg_dir = Path(deepgram_dir)
    for dg_file in sorted(dg_dir.glob("*.deepgram.json")):
        with open(dg_file) as f:
            dg = json.load(f)
        utterances = dg.get("utterances", dg.get("segments", []))
        # Filter to substantive utterances
        long_utts = []
        for u in utterances:
            text = u.get("transcript", u.get("text", ""))
            if len(text.split()) >= min_words:
                long_utts.append({"text": text, "source": dg_file.stem})
        sampled = random.sample(long_utts, min(num_per_sermon, len(long_utts)))
        chunks.extend(sampled)
    logger.info(
        "Loaded %d sermon chunks from %s (%d files)",
        len(chunks),
        deepgram_dir,
        len(list(dg_dir.glob("*.deepgram.json"))),
    )
    return chunks


# ── Metrics ─────────────────────────────────────────────────────────────────


def compute_hallucination_ratio(source, translation):
    """Ratio of output words to input words. Flag if >1.5."""
    src_words = len(source.split())
    tgt_words = len(translation.split())
    return tgt_words / max(src_words, 1)


# ── Main benchmark ──────────────────────────────────────────────────────────


def run_benchmark(args):
    """Run the full benchmark across all requested models."""
    import torch

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Record GPU info
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_vram = torch.cuda.get_device_properties(0).total_mem / 1024**3
        logger.info("GPU: %s (%.1f GB)", gpu_name, gpu_vram)
    else:
        gpu_name = "CPU"
        gpu_vram = 0
        logger.warning("No CUDA device — running on CPU (will be slow)")

    # Load test data
    verse_pairs = load_verse_pairs(args.test, max_samples=args.max_samples) if os.path.exists(args.test) else []
    sermon_chunks = load_sermon_chunks(args.sermon_dir) if args.sermon_dir and os.path.exists(args.sermon_dir) else []

    if not verse_pairs and not sermon_chunks:
        logger.error("No test data found. Provide --test and/or --sermon-dir.")
        return

    # Parse requested models
    if args.models == "all":
        model_keys = list(MODEL_REGISTRY.keys())
    else:
        model_keys = [m.strip() for m in args.models.split(",")]

    # Results storage
    all_results = {}
    all_verse_hyps = {}  # model_key -> list of hypothesis strings (for batch COMET)
    all_sermon_hyps = {}

    for mi, model_key in enumerate(model_keys):
        if model_key not in MODEL_REGISTRY:
            logger.warning("Unknown model key: %s (skipping)", model_key)
            continue

        config = MODEL_REGISTRY[model_key]
        logger.info(
            "=" * 60 + "\n[%d/%d] %s (%s)\n" + "=" * 60,
            mi + 1,
            len(model_keys),
            config["display_name"],
            config["model_id"],
        )

        # Load
        load_start = time.time()
        try:
            model, tokenizer = load_model_nf4(config["model_id"])
        except Exception as e:
            logger.error("Failed to load %s: %s", config["model_id"], e)
            all_results[model_key] = {"error": str(e)}
            continue
        load_time = time.time() - load_start

        vram_loaded = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
        logger.info("Loaded in %.1fs (VRAM: %.2f GB)", load_time, vram_loaded)

        # Select translate function — bind model/tokenizer via default args
        if config["prompt_type"] == "translategemma":
            translate_fn = lambda text, m=model, t=tokenizer: translate_translategemma(m, t, text)
        else:
            translate_fn = lambda text, m=model, t=tokenizer: translate_gemma4(m, t, text)

        # Warmup
        logger.info("Warmup (%d sentences)...", args.warmup)
        for s in WARMUP_SENTENCES[: args.warmup]:
            translate_fn(s)

        result = {
            "model_id": config["model_id"],
            "display_name": config["display_name"],
            "prompt_type": config["prompt_type"],
            "load_time_s": round(load_time, 1),
            "vram_loaded_gb": round(vram_loaded, 2),
        }

        # ── Tier 1: Verse pairs ──
        verse_hyps = []
        verse_latencies = []
        if verse_pairs:
            logger.info("Tier 1: Translating %d verse pairs...", len(verse_pairs))
            for i, (en, es) in enumerate(verse_pairs):
                t0 = time.time()
                hyp = translate_fn(en)
                elapsed = time.time() - t0
                verse_hyps.append(hyp)
                verse_latencies.append(elapsed)
                if (i + 1) % 50 == 0:
                    logger.info(
                        "  [%d/%d] median %.2fs",
                        i + 1,
                        len(verse_pairs),
                        sorted(verse_latencies)[len(verse_latencies) // 2],
                    )

            all_verse_hyps[model_key] = verse_hyps
            result["tier1_count"] = len(verse_pairs)
            result["tier1_latency_median_s"] = round(sorted(verse_latencies)[len(verse_latencies) // 2], 2)
            result["tier1_latency_p95_s"] = round(sorted(verse_latencies)[int(len(verse_latencies) * 0.95)], 2)

        # ── Tier 2: Sermon chunks (Deepgram source of truth) ──
        sermon_hyps = []
        sermon_latencies = []
        hallucination_ratios = []
        if sermon_chunks:
            logger.info("Tier 2: Translating %d sermon chunks...", len(sermon_chunks))
            for i, chunk in enumerate(sermon_chunks):
                t0 = time.time()
                hyp = translate_fn(chunk["text"])
                elapsed = time.time() - t0
                sermon_hyps.append({"source": chunk["text"], "translation": hyp, "sermon": chunk["source"]})
                sermon_latencies.append(elapsed)
                hallucination_ratios.append(compute_hallucination_ratio(chunk["text"], hyp))

            all_sermon_hyps[model_key] = sermon_hyps
            result["tier2_count"] = len(sermon_chunks)
            result["tier2_latency_median_s"] = round(sorted(sermon_latencies)[len(sermon_latencies) // 2], 2)
            result["tier2_hallucination_mean"] = round(sum(hallucination_ratios) / len(hallucination_ratios), 3)
            result["tier2_hallucination_flagged"] = sum(1 for r in hallucination_ratios if r > 1.5)

        # ── Tier 3: Canary theological sentences ──
        logger.info("Tier 3: %d canary sentences...", len(CANARY_SENTENCES))
        canary_hits = 0
        canary_results = []
        for en, expected_es in CANARY_SENTENCES:
            hyp = translate_fn(en)
            hit = expected_es.lower() in hyp.lower()
            canary_hits += int(hit)
            canary_results.append({"en": en, "es_hyp": hyp, "expected": expected_es, "hit": hit})
        result["tier3_theological_accuracy"] = f"{canary_hits}/{len(CANARY_SENTENCES)}"
        result["tier3_canary_details"] = canary_results

        # Peak VRAM
        if torch.cuda.is_available():
            result["vram_peak_gb"] = round(torch.cuda.max_memory_allocated() / 1024**3, 2)

        # Save per-sentence JSONL
        sentences_path = output_dir / f"{model_key}_sentences.jsonl"
        with open(sentences_path, "w") as f:
            for i, (en, es) in enumerate(verse_pairs):
                hyp = verse_hyps[i] if i < len(verse_hyps) else ""
                lat = verse_latencies[i] if i < len(verse_latencies) else 0
                f.write(
                    json.dumps(
                        {"tier": 1, "en": en, "es_ref": es, "es_hyp": hyp, "latency_s": round(lat, 3)},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            for sh in sermon_hyps:
                f.write(
                    json.dumps(
                        {"tier": 2, "en": sh["source"], "es_hyp": sh["translation"], "sermon": sh["sermon"]},
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        all_results[model_key] = result
        logger.info(
            "Results so far: %s",
            json.dumps(
                {
                    k: {kk: vv for kk, vv in v.items() if kk != "tier3_canary_details"}
                    for k, v in all_results.items()
                    if isinstance(v, dict) and "error" not in v
                },
                indent=2,
            ),
        )

        # Unload
        unload_model(model, tokenizer)

    # ── Compute corpus metrics (BLEU, chrF++, COMET) ──
    if verse_pairs and all_verse_hyps:
        import sacrebleu

        refs = [es for _, es in verse_pairs]

        for model_key, hyps in all_verse_hyps.items():
            bleu = sacrebleu.corpus_bleu(hyps, [refs])
            chrf = sacrebleu.corpus_chrf(hyps, [refs], word_order=2)
            all_results[model_key]["tier1_bleu"] = round(bleu.score, 1)
            all_results[model_key]["tier1_chrf"] = round(chrf.score, 1)
            logger.info("[%s] BLEU=%.1f chrF++=%.1f", model_key, bleu.score, chrf.score)

        # COMET (reference-based for Tier 1)
        if not args.skip_comet:
            logger.info("Loading COMET model for Tier 1 scoring...")
            try:
                from comet import download_model, load_from_checkpoint

                comet_path = download_model("Unbabel/wmt22-comet-da")
                comet_model = load_from_checkpoint(comet_path)

                sources = [en for en, _ in verse_pairs]
                for model_key, hyps in all_verse_hyps.items():
                    data = [{"src": s, "mt": h, "ref": r} for s, h, r in zip(sources, hyps, refs)]
                    output = comet_model.predict(data, batch_size=32, gpus=1 if torch.cuda.is_available() else 0)
                    score = output.system_score if hasattr(output, "system_score") else output[1]
                    all_results[model_key]["tier1_comet"] = round(score, 4)
                    logger.info("[%s] COMET=%.4f", model_key, score)

                del comet_model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as e:
                logger.warning("COMET scoring failed: %s", e)

    # ── Print comparison table ──
    logger.info("\n" + "=" * 80)
    logger.info("GEMMA 4 vs TRANSLATEGEMMA BENCHMARK")
    logger.info("=" * 80)

    header = f"{'Model':<25} {'BLEU':>6} {'chrF++':>7} {'COMET':>7} {'Theo':>6} {'Lat(s)':>7} {'VRAM':>6}"
    logger.info(header)
    logger.info("-" * 72)
    for mk in model_keys:
        if mk not in all_results or "error" in all_results[mk]:
            continue
        r = all_results[mk]
        name = r["display_name"][:24]
        bleu = f"{r.get('tier1_bleu', '?'):>6}"
        chrf = f"{r.get('tier1_chrf', '?'):>7}"
        comet = f"{r.get('tier1_comet', '?'):>7}"
        theo = f"{r.get('tier3_theological_accuracy', '?'):>6}"
        lat = f"{r.get('tier1_latency_median_s', '?'):>7}"
        vram = f"{r.get('vram_peak_gb', '?'):>6}"
        logger.info(f"{name:<25} {bleu} {chrf} {comet} {theo} {lat} {vram}")

    if any(mk in all_results and "tier2_count" in all_results[mk] for mk in model_keys):
        logger.info("")
        logger.info("Sermon chunks (Tier 2, Deepgram source of truth):")
        logger.info(f"{'Model':<25} {'Halluc':>7} {'Flagged':>8} {'Lat(s)':>7}")
        logger.info("-" * 50)
        for mk in model_keys:
            if mk not in all_results or "tier2_count" not in all_results[mk]:
                continue
            r = all_results[mk]
            name = r["display_name"][:24]
            logger.info(
                f"{name:<25} {r['tier2_hallucination_mean']:>7.3f} {r['tier2_hallucination_flagged']:>8} {r['tier2_latency_median_s']:>7}"
            )

    # ── Save results ──
    comparison = {
        "benchmark_date": time.strftime("%Y-%m-%d"),
        "gpu": gpu_name,
        "gpu_vram_gb": round(gpu_vram, 1),
        "quantization": "NF4",
        "tier1_verse_count": len(verse_pairs),
        "tier2_sermon_count": len(sermon_chunks),
        "models": {k: v for k, v in all_results.items() if isinstance(v, dict)},
    }
    with open(output_dir / "comparison.json", "w") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)
    logger.info("Saved comparison to %s", output_dir / "comparison.json")

    # Save per-model JSONs
    for mk, r in all_results.items():
        with open(output_dir / f"{mk}.json", "w") as f:
            json.dump(r, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description="Benchmark Gemma 4 vs TranslateGemma on EN→ES translation")
    parser.add_argument(
        "--test",
        default="bible_data/synthetic/deepl_sermon_benchmark_500.jsonl",
        help="Path to verse pairs JSONL (Tier 1 eval data)",
    )
    parser.add_argument(
        "--sermon-dir",
        default="stark_data/eval_fresh_transcripts",
        help="Directory with Deepgram .deepgram.json transcripts (Tier 2 eval data)",
    )
    parser.add_argument(
        "--models",
        default="all",
        help="Comma-separated model shortnames: tg4b,tg12b,e2b,e4b (default: all)",
    )
    parser.add_argument("--max-samples", type=int, default=None, help="Max verse pairs for quick test")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup sentences before timing")
    parser.add_argument("--output-dir", default="metrics/gemma4_benchmark", help="Output directory")
    parser.add_argument("--skip-comet", action="store_true", help="Skip COMET scoring (saves time + VRAM)")
    args = parser.parse_args()

    run_benchmark(args)


if __name__ == "__main__":
    main()
