#!/usr/bin/env python3
"""
generate_hybrid_synthetic.py — Hybrid 12B + DeepL Pro Synthetic Sermon Data

Generates training data from Whisper-transcribed sermon chunks using a 70/30 split:
  - 70% translated by TranslateGemma-12B (with cache for existing translations)
  - 30% translated by DeepL Pro API (with glossary enforcement)

This hybrid approach combines 12B model quality with DeepL commercial fluency and
strict glossary enforcement for theological terms (Santiago, Jacobo, propiciación).

Input:  sermon_whisper_chunks.json (1880 chunks from 2 conference sermons)
Output: JSONL with {"en": "...", "es": "...", "source": "12b"|"deepl"}

Usage:
    python training/generate_hybrid_synthetic.py \\
        --deepl-key "$DEEPL_KEY" \\
        --max-chunks 1200

    # Quick test (300 chunks)
    python training/generate_hybrid_synthetic.py \\
        --deepl-key "$DEEPL_KEY" \\
        --max-chunks 300
"""

import argparse
import gc
import json
import logging
import os
import random
import sys

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_whisper_chunks(path, min_chars=20):
    """Load chunks, filter by min char length, deduplicate by exact text."""
    with open(path) as f:
        raw = json.load(f)

    logger.info(f"Loaded {len(raw)} raw whisper chunks from {path}")

    # Filter by min char length
    filtered = [c for c in raw if len(c.get("en", "")) >= min_chars]
    logger.info(f"  After min_chars={min_chars} filter: {len(filtered)}")

    # Deduplicate by exact text
    seen = set()
    deduped = []
    for c in filtered:
        text = c["en"]
        if text not in seen:
            seen.add(text)
            deduped.append(c)

    logger.info(f"  After deduplication: {len(deduped)}")
    return deduped


def load_12b_cache(cache_path):
    """Load existing 12B translations as {en_text: es_text} dict.

    Supports format: [{"en": ..., "es_12b": ...}]
    """
    if not cache_path or not os.path.exists(cache_path):
        logger.info("No 12B cache provided or file not found")
        return {}

    with open(cache_path) as f:
        data = json.load(f)

    cache = {entry["en"]: entry["es_12b"] for entry in data if entry.get("es_12b")}
    logger.info(f"Loaded {len(cache)} cached 12B translations from {cache_path}")
    return cache


def build_deepl_glossary(translator, glossary_path):
    """Filter glossary_pairs.jsonl to bare terms (len(en.split()) <= 2).

    Upload to DeepL via translator.create_glossary().
    Returns GlossaryInfo with .glossary_id.
    """
    if not os.path.exists(glossary_path):
        logger.warning(f"Glossary file not found: {glossary_path}")
        return None

    with open(glossary_path, encoding="utf-8") as f:
        all_entries = [json.loads(line) for line in f]

    # Filter to bare terms (1-2 words) for DeepL glossary
    bare_terms = {entry["en"]: entry["es"] for entry in all_entries if len(entry["en"].split()) <= 2}
    logger.info(f"Filtered {len(all_entries)} glossary entries to {len(bare_terms)} bare terms (len(en.split()) <= 2)")

    if not bare_terms:
        logger.warning("No bare terms found for DeepL glossary")
        return None

    glossary = translator.create_glossary(
        "stark-translate-theological",
        source_lang="en",
        target_lang="es",
        entries=bare_terms,
    )
    logger.info(f"Created DeepL glossary: {glossary.glossary_id} ({len(bare_terms)} entries)")
    return glossary


def assign_translation_source(chunks, ratio_deepl=0.30, seed=42):
    """Random split into (twelve_b_chunks, deepl_chunks). Seeded."""
    rng = random.Random(seed)
    shuffled = list(chunks)
    rng.shuffle(shuffled)

    n_deepl = int(len(shuffled) * ratio_deepl)
    deepl_chunks = shuffled[:n_deepl]
    twelve_b_chunks = shuffled[n_deepl:]

    logger.info(
        f"Split {len(chunks)} chunks: {len(twelve_b_chunks)} for 12B ({1 - ratio_deepl:.0%}), "
        f"{len(deepl_chunks)} for DeepL ({ratio_deepl:.0%})"
    )
    return twelve_b_chunks, deepl_chunks


def translate_with_12b(chunks, cache, base_model):
    """Translate using cache hits first. Load 12B model ONLY if cache misses exist.

    Import load_gemma_model + translate_gemma from evaluate_translation.
    Unload model after translation (gc.collect + cuda.empty_cache).
    Returns [{"en": ..., "es": ..., "source": "12b"}]
    """
    results = []
    uncached = []

    for c in chunks:
        text = c["en"]
        if text in cache:
            results.append({"en": text, "es": cache[text], "source": "12b"})
        else:
            uncached.append(c)

    logger.info(f"12B translation: {len(results)} cache hits, {len(uncached)} cache misses")

    if uncached:
        from evaluate_translation import load_gemma_model, translate_gemma

        logger.info(f"Loading {base_model} for {len(uncached)} uncached translations...")
        model, tokenizer = load_gemma_model(base_model, adapter_dir=None)

        for i, c in enumerate(uncached):
            try:
                translation = translate_gemma(model, tokenizer, c["en"])
            except Exception as e:
                logger.warning(f"12B translation failed for chunk {i}: {e}")
                translation = ""
            results.append({"en": c["en"], "es": translation, "source": "12b"})

            if (i + 1) % 50 == 0:
                logger.info(f"  12B progress: {i + 1}/{len(uncached)}")

        # Unload model
        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("12B model unloaded")

    return results


def translate_with_deepl(chunks, translator, glossary):
    """Translate via DeepL API with glossary enforcement.

    target_lang="ES", source_lang="EN".
    Returns [{"en": ..., "es": ..., "source": "deepl"}]
    """
    results = []

    for i, c in enumerate(chunks):
        text = c["en"]
        try:
            kwargs = {
                "text": text,
                "target_lang": "ES",
                "source_lang": "EN",
            }
            if glossary:
                kwargs["glossary"] = glossary

            result = translator.translate_text(**kwargs)
            translation = result.text
        except Exception as e:
            logger.warning(f"DeepL translation failed for chunk {i}: {e}")
            translation = ""

        results.append({"en": text, "es": translation, "source": "deepl"})

        if (i + 1) % 50 == 0:
            logger.info(f"  DeepL progress: {i + 1}/{len(chunks)}")

    logger.info(f"DeepL translation complete: {len(results)} chunks")
    return results


def main():
    """Orchestrate: load chunks -> split -> translate both -> merge -> save JSONL."""
    parser = argparse.ArgumentParser(description="Generate hybrid 12B + DeepL synthetic sermon training data")
    parser.add_argument(
        "--deepl-key",
        required=True,
        help="DeepL Pro API key",
    )
    parser.add_argument(
        "--whisper-chunks",
        default="ablation/sermon_whisper_chunks.json",
        help="Path to whisper-transcribed sermon chunks JSON",
    )
    parser.add_argument(
        "--output",
        default="bible_data/synthetic/hybrid_sermon_pairs.jsonl",
        help="Output JSONL path",
    )
    parser.add_argument(
        "--ratio-deepl",
        type=float,
        default=0.30,
        help="Fraction of chunks to translate with DeepL (default: 0.30)",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=1200,
        help="Max chunks to use (default: 1200)",
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=20,
        help="Minimum character length for chunks (default: 20)",
    )
    parser.add_argument(
        "--twelb-cache",
        default="ablation/sermon_12b_translations_v2.json",
        help="Path to pre-computed 12B translations JSON",
    )
    parser.add_argument(
        "--glossary-file",
        default="bible_data/glossary/glossary_pairs.jsonl",
        help="Path to glossary JSONL for DeepL glossary upload",
    )
    parser.add_argument(
        "--base-model",
        default="google/translategemma-12b-it",
        help="12B model ID for uncached translations",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible splitting",
    )
    args = parser.parse_args()

    # --- Load and filter chunks ---
    chunks = load_whisper_chunks(args.whisper_chunks, min_chars=args.min_chars)

    if args.max_chunks and len(chunks) > args.max_chunks:
        rng = random.Random(args.seed)
        chunks = rng.sample(chunks, args.max_chunks)
        logger.info(f"Capped to {args.max_chunks} chunks (seeded sample)")

    if not chunks:
        logger.error("No chunks remaining after filtering!")
        sys.exit(1)

    # --- Load 12B cache ---
    cache = load_12b_cache(args.twelb_cache)

    # --- Set up DeepL ---
    import deepl

    translator = deepl.Translator(args.deepl_key)
    glossary = build_deepl_glossary(translator, args.glossary_file)

    # --- Split and translate ---
    twelve_b_chunks, deepl_chunks = assign_translation_source(chunks, ratio_deepl=args.ratio_deepl, seed=args.seed)

    twelve_b_results = translate_with_12b(twelve_b_chunks, cache, args.base_model)
    deepl_results = translate_with_deepl(deepl_chunks, translator, glossary)

    # --- Merge and save ---
    all_results = twelve_b_results + deepl_results

    # Filter out empty translations
    valid = [r for r in all_results if r["es"]]
    logger.info(
        f"Total pairs: {len(valid)} valid / {len(all_results)} total "
        f"({len(all_results) - len(valid)} empty translations filtered)"
    )

    # Create output directory
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as f:
        for pair in valid:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    logger.info(f"Saved {len(valid)} pairs to {args.output}")

    # Summary
    n_12b = sum(1 for r in valid if r["source"] == "12b")
    n_deepl = sum(1 for r in valid if r["source"] == "deepl")
    logger.info(f"  12B: {n_12b} pairs, DeepL: {n_deepl} pairs")

    # Clean up DeepL glossary
    if glossary:
        try:
            translator.delete_glossary(glossary)
            logger.info("Deleted temporary DeepL glossary")
        except Exception as e:
            logger.warning(f"Could not delete DeepL glossary: {e}")


if __name__ == "__main__":
    main()
