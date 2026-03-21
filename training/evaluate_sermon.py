#!/usr/bin/env python3
"""
evaluate_sermon.py — Three-Model Sermon Evaluation (Phase 2.5)

Evaluates fine-tuned TranslateGemma adapter on real sermon content by comparing
three models: 4B base, 4B + adapter, and 12B base (quality ceiling).

Models are loaded sequentially (not simultaneously) to fit in 16GB VRAM.

Test data:
  Tier 1: Sermon chunks from JSON (exported from review queue TSV)
  Tier 2: 8 sermon-style theological sentences (hardcoded)
  Tier 3: Optional full sermon segment from text file

Automated metrics:
  1. 12B proximity gain (chrF++ distance to 12B ceiling)
  2. Hallucination ratio (output/input word count)
  3. Sermon-style theological term accuracy
  4. Register markers (archaic Spanish form counts)

Usage:
    python training/evaluate_sermon.py \\
        --chunks ablation/sermon_test_chunks.json \\
        --adapter ablation/B2_lr1e6_150 \\
        --output ablation/sermon_eval_results.json

    python training/evaluate_sermon.py \\
        --chunks ablation/sermon_test_chunks.json \\
        --adapter ablation/B2_lr1e6_150 \\
        --ceiling-model google/translategemma-12b-it \\
        --segment ablation/sermon_segment.txt \\
        --output ablation/sermon_eval_results.json
"""

import argparse
import gc
import json
import logging
import re
import sys

import sacrebleu
import torch

# Reuse model loading and translation from evaluate_translation.py
from evaluate_translation import load_gemma_model, translate_gemma

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# --- Tier 2: Sermon-style theological sentences ---
SERMON_SENTENCES = [
    (
        "And you know, when we think about the atonement — what Christ did for us on that cross...",
        "expiación",
    ),
    (
        "God made a covenant with Abraham, and brothers, He keeps His promises.",
        "pacto",
    ),
    (
        "It's only by grace, friends. We can't earn it — it's grace through faith.",
        "gracia",
    ),
    (
        "And Paul writes about the righteousness of God here in Romans, and what does he mean by that?",
        "justicia",
    ),
    (
        "Now, if you turn to the book of James — James has a lot to say about faith and works.",
        "Santiago",
    ),
    (
        "You think about James and John, just regular fishermen on the Sea of Galilee.",
        "Jacobo",
    ),
    (
        "The speaker tonight was talking about sanctification — what it means to be set apart.",
        "santificación",
    ),
    (
        "First John chapter 2 — He is the propitiation for our sins, and not for ours only.",
        "propiciación",
    ),
]

# --- Archaic Spanish register markers ---
ARCHAIC_MARKERS = [
    r"\bvosotros\b",
    r"\bhabéis\b",
    r"\bsois\b",
    r"\bhe aquí\b",
    r"\bmas\b",  # "mas" (but) instead of "pero"
    r"\bempero\b",
    r"\bbienaventurado\b",
    r"\bmenester\b",
    r"\bved aquí\b",
]


def _unload_model(model, tokenizer):
    """Free GPU memory after finishing with a model."""
    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _translate_all(model, tokenizer, texts):
    """Translate a list of texts, returning translations."""
    translations = []
    for text in texts:
        try:
            translation = translate_gemma(model, tokenizer, text)
        except Exception as e:
            logger.warning(f"Translation failed: {e}")
            translation = ""
        translations.append(translation)
    return translations


def _count_archaic_markers(text):
    """Count archaic Spanish register markers in text."""
    count = 0
    text_lower = text.lower()
    for pattern in ARCHAIC_MARKERS:
        count += len(re.findall(pattern, text_lower))
    return count


def _hallucination_ratio(input_text, output_text):
    """Compute output/input word count ratio."""
    input_words = len(input_text.split())
    output_words = len(output_text.split())
    if input_words == 0:
        return float("inf") if output_words > 0 else 1.0
    return output_words / input_words


def evaluate_sermon(
    chunks_path,
    adapter_dir,
    base_model="google/translategemma-4b-it",
    ceiling_model="google/translategemma-12b-it",
    segment_path=None,
    output_file=None,
):
    """Run three-model sermon evaluation.

    Loads models sequentially: 4B base → 4B+adapter → 12B base.
    """
    # --- Load test data ---
    all_inputs = []
    chunk_data = []
    segment_text = None

    # Tier 1: Sermon chunks
    if chunks_path:
        try:
            with open(chunks_path) as f:
                chunk_data = json.load(f)
            logger.info(f"Loaded {len(chunk_data)} sermon chunks from {chunks_path}")
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning(f"Could not load chunks: {e}")

    chunk_texts = [c["en"] for c in chunk_data]

    # Tier 2: Sermon-style sentences
    sermon_texts = [s[0] for s in SERMON_SENTENCES]

    # Tier 3: Optional segment
    if segment_path:
        try:
            with open(segment_path) as f:
                segment_text = f.read().strip()
            logger.info(f"Loaded sermon segment ({len(segment_text.split())} words) from {segment_path}")
        except FileNotFoundError:
            logger.warning(f"Segment file not found: {segment_path}")

    # Combine all inputs
    all_inputs = chunk_texts + sermon_texts
    if segment_text:
        all_inputs.append(segment_text)

    if not all_inputs:
        logger.error("No test data to evaluate!")
        sys.exit(1)

    n_chunks = len(chunk_texts)
    n_sermon = len(sermon_texts)
    logger.info(
        f"Total inputs: {len(all_inputs)} ({n_chunks} chunks + {n_sermon} sermon sentences"
        f"{' + 1 segment' if segment_text else ''})"
    )

    # --- Translate with each model sequentially ---
    results = {}

    # Model 1: 4B base (no adapter)
    logger.info("\n=== Model 1/3: 4B base (no adapter) ===")
    model, tokenizer = load_gemma_model(base_model, adapter_dir=None)
    results["base"] = _translate_all(model, tokenizer, all_inputs)
    _unload_model(model, tokenizer)

    # Model 2: 4B + adapter
    logger.info(f"\n=== Model 2/3: 4B + adapter ({adapter_dir}) ===")
    model, tokenizer = load_gemma_model(base_model, adapter_dir)
    results["adapter"] = _translate_all(model, tokenizer, all_inputs)
    _unload_model(model, tokenizer)

    # Model 3: 12B base (quality ceiling)
    logger.info(f"\n=== Model 3/3: 12B ceiling ({ceiling_model}) ===")
    model, tokenizer = load_gemma_model(ceiling_model, adapter_dir=None)
    results["ceiling"] = _translate_all(model, tokenizer, all_inputs)
    _unload_model(model, tokenizer)

    # --- Compute metrics ---
    logger.info("\n=== Computing metrics ===")

    # Split translations back into tiers
    base_chunks = results["base"][:n_chunks]
    adapter_chunks = results["adapter"][:n_chunks]
    ceiling_chunks = results["ceiling"][:n_chunks]

    base_sermon = results["base"][n_chunks : n_chunks + n_sermon]
    adapter_sermon = results["adapter"][n_chunks : n_chunks + n_sermon]
    ceiling_sermon = results["ceiling"][n_chunks : n_chunks + n_sermon]

    # --- Metric 1: 12B proximity gain (clean chunks only, confidence > 0.7) ---
    clean_indices = [i for i, c in enumerate(chunk_data) if float(c.get("confidence", 0)) > 0.7]
    proximity_gain = None
    if clean_indices and ceiling_chunks:
        clean_base = [base_chunks[i] for i in clean_indices]
        clean_adapter = [adapter_chunks[i] for i in clean_indices]
        clean_ceiling = [ceiling_chunks[i] for i in clean_indices]

        chrf_base_vs_ceiling = sacrebleu.corpus_chrf(clean_base, [clean_ceiling], word_order=2).score
        chrf_adapter_vs_ceiling = sacrebleu.corpus_chrf(clean_adapter, [clean_ceiling], word_order=2).score
        proximity_gain = chrf_adapter_vs_ceiling - chrf_base_vs_ceiling

        logger.info(f"  12B proximity gain: {proximity_gain:+.1f} chrF++ points")
        logger.info(f"    Base vs 12B:    chrF++ {chrf_base_vs_ceiling:.1f}")
        logger.info(f"    Adapter vs 12B: chrF++ {chrf_adapter_vs_ceiling:.1f}")
        logger.info(f"    Clean chunks used: {len(clean_indices)}")
    else:
        logger.warning("  No clean chunks (confidence > 0.7) for proximity gain")

    # --- Metric 2: Hallucination ratio ---
    low_conf_indices = [i for i, c in enumerate(chunk_data) if float(c.get("confidence", 1.0)) < 0.5]
    short_indices = [i for i, c in enumerate(chunk_data) if len(c.get("en", "").split()) < 6]
    risky_indices = list(set(low_conf_indices + short_indices))

    hallucination_result = None
    if risky_indices:
        base_ratios = [_hallucination_ratio(chunk_texts[i], base_chunks[i]) for i in risky_indices]
        adapter_ratios = [_hallucination_ratio(chunk_texts[i], adapter_chunks[i]) for i in risky_indices]
        avg_base_ratio = sum(base_ratios) / len(base_ratios)
        avg_adapter_ratio = sum(adapter_ratios) / len(adapter_ratios)
        hallucination_result = {
            "avg_base_ratio": round(avg_base_ratio, 2),
            "avg_adapter_ratio": round(avg_adapter_ratio, 2),
            "ratio_increase": round(avg_adapter_ratio / avg_base_ratio, 2) if avg_base_ratio > 0 else None,
            "risky_chunks": len(risky_indices),
        }
        logger.info(f"  Hallucination ratio (risky chunks): base={avg_base_ratio:.2f}, adapter={avg_adapter_ratio:.2f}")
        if avg_base_ratio > 0:
            logger.info(f"    Ratio increase: {avg_adapter_ratio / avg_base_ratio:.2f}x")
    else:
        logger.info("  No risky chunks for hallucination analysis")

    # --- Metric 3: Sermon-style theological terms ---
    theo_results = []
    base_correct = 0
    adapter_correct = 0
    ceiling_correct = 0
    for i, (sentence, expected_term) in enumerate(SERMON_SENTENCES):
        base_found = expected_term.lower() in base_sermon[i].lower()
        adapter_found = expected_term.lower() in adapter_sermon[i].lower()
        ceiling_found = expected_term.lower() in ceiling_sermon[i].lower()
        if base_found:
            base_correct += 1
        if adapter_found:
            adapter_correct += 1
        if ceiling_found:
            ceiling_correct += 1
        theo_results.append(
            {
                "sentence": sentence,
                "expected_term": expected_term,
                "base_found": base_found,
                "adapter_found": adapter_found,
                "ceiling_found": ceiling_found,
                "base_translation": base_sermon[i],
                "adapter_translation": adapter_sermon[i],
                "ceiling_translation": ceiling_sermon[i],
            }
        )

    logger.info(
        f"  Sermon-style theo terms: base={base_correct}/8, adapter={adapter_correct}/8, 12B={ceiling_correct}/8"
    )

    # --- Metric 4: Register markers ---
    all_base_text = " ".join(results["base"])
    all_adapter_text = " ".join(results["adapter"])
    all_ceiling_text = " ".join(results["ceiling"])

    archaic_base = _count_archaic_markers(all_base_text)
    archaic_adapter = _count_archaic_markers(all_adapter_text)
    archaic_ceiling = _count_archaic_markers(all_ceiling_text)

    logger.info(f"  Archaic markers: base={archaic_base}, adapter={archaic_adapter}, 12B={archaic_ceiling}")

    # --- Kill switch verdicts ---
    verdicts = {}

    # Proximity gain kill switch
    if proximity_gain is not None:
        if proximity_gain < 0:
            verdicts["proximity_gain"] = "KILL — adapter moved AWAY from 12B"
        elif proximity_gain < 2:
            verdicts["proximity_gain"] = "WARN — marginal gain, check human review"
        else:
            verdicts["proximity_gain"] = "PASS"
    else:
        verdicts["proximity_gain"] = "SKIP — no clean chunks"

    # Hallucination kill switch
    if hallucination_result and hallucination_result["ratio_increase"] is not None:
        if hallucination_result["ratio_increase"] > 1.5:
            verdicts["hallucination"] = "KILL — adapter hallucinates on noisy input"
        elif hallucination_result["ratio_increase"] > 1.2:
            verdicts["hallucination"] = "WARN — slight hallucination increase"
        else:
            verdicts["hallucination"] = "PASS"
    else:
        verdicts["hallucination"] = "SKIP — no risky chunks"

    # Register kill switch
    if archaic_ceiling > 0 and archaic_adapter > archaic_ceiling * 2:
        verdicts["register"] = "KILL — adapter more archaic than 12B"
    elif archaic_base > 0 and archaic_adapter > archaic_base * 2:
        verdicts["register"] = "WARN — adapter shifting toward archaic register"
    else:
        verdicts["register"] = "PASS"

    # Theological terms gate
    if adapter_correct >= base_correct:
        verdicts["theo_terms"] = "PASS"
    else:
        verdicts["theo_terms"] = f"WARN — adapter ({adapter_correct}/8) < base ({base_correct}/8)"

    # --- Print verdict summary ---
    logger.info("\n=== KILL SWITCH VERDICTS ===")
    any_kill = False
    for gate, verdict in verdicts.items():
        status = "KILL" if "KILL" in verdict else ("WARN" if "WARN" in verdict else "PASS")
        logger.info(f"  [{status}] {gate}: {verdict}")
        if "KILL" in verdict:
            any_kill = True

    if any_kill:
        logger.warning("\n  >>> ONE OR MORE KILL SWITCHES TRIGGERED — DO NOT PROCEED TO SCALE-UP <<<")
    else:
        logger.info("\n  >>> All gates passed — proceed to human review <<<")

    # --- Print comparison table ---
    logger.info("\n=== SERMON CHUNK COMPARISON ===")
    logger.info(f"{'ID':>4}  {'Conf':>5}  {'Model':>8}  Translation")
    logger.info("-" * 80)
    for i, chunk in enumerate(chunk_data):
        conf = float(chunk.get("confidence", 0))
        chunk_id = chunk.get("chunk_id", f"C{i}")
        logger.info(f"{chunk_id:>4}  {conf:5.2f}  {'base':>8}  {base_chunks[i][:80]}")
        logger.info(f"{'':>4}  {'':>5}  {'adapter':>8}  {adapter_chunks[i][:80]}")
        logger.info(f"{'':>4}  {'':>5}  {'12B':>8}  {ceiling_chunks[i][:80]}")
        logger.info("")

    logger.info("\n=== SERMON-STYLE THEOLOGICAL TERMS ===")
    for tr in theo_results:
        b_status = "PASS" if tr["base_found"] else "FAIL"
        a_status = "PASS" if tr["adapter_found"] else "FAIL"
        c_status = "PASS" if tr["ceiling_found"] else "FAIL"
        logger.info(f"  [{a_status}] '{tr['expected_term']}' — base:{b_status} adapter:{a_status} 12B:{c_status}")
        logger.info(f"         EN: {tr['sentence'][:70]}")
        logger.info(f"         Adapter: {tr['adapter_translation'][:70]}")

    # --- Build output ---
    output = {
        "metrics": {
            "proximity_gain": proximity_gain,
            "proximity_base_vs_ceiling": chrf_base_vs_ceiling if proximity_gain is not None else None,
            "proximity_adapter_vs_ceiling": chrf_adapter_vs_ceiling if proximity_gain is not None else None,
            "clean_chunks_used": len(clean_indices),
            "hallucination": hallucination_result,
            "theological_terms": {
                "base": base_correct,
                "adapter": adapter_correct,
                "ceiling": ceiling_correct,
                "total": len(SERMON_SENTENCES),
            },
            "archaic_markers": {
                "base": archaic_base,
                "adapter": archaic_adapter,
                "ceiling": archaic_ceiling,
            },
        },
        "verdicts": verdicts,
        "theological_details": theo_results,
        "chunk_translations": [
            {
                "chunk_id": chunk.get("chunk_id", f"C{i}"),
                "confidence": float(chunk.get("confidence", 0)),
                "en": chunk["en"],
                "base": base_chunks[i],
                "adapter": adapter_chunks[i],
                "ceiling": ceiling_chunks[i],
            }
            for i, chunk in enumerate(chunk_data)
        ],
    }

    if segment_text:
        output["segment"] = {
            "en": segment_text,
            "base": results["base"][-1],
            "adapter": results["adapter"][-1],
            "ceiling": results["ceiling"][-1],
        }

    # Save output
    if output_file:
        import os

        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        logger.info(f"\nResults saved to {output_file}")

    return output


def main():
    parser = argparse.ArgumentParser(description="Three-model sermon evaluation (Phase 2.5)")
    parser.add_argument(
        "--chunks",
        required=True,
        help="Path to sermon test chunks JSON (exported from review queue TSV)",
    )
    parser.add_argument(
        "--adapter",
        required=True,
        help="Path to QLoRA adapter directory (Phase 2 winner)",
    )
    parser.add_argument(
        "--base-model",
        default="google/translategemma-4b-it",
        help="Base TranslateGemma model (default: 4B)",
    )
    parser.add_argument(
        "--ceiling-model",
        default="google/translategemma-12b-it",
        help="Quality ceiling model (default: 12B). Used as pseudo-reference.",
    )
    parser.add_argument(
        "--segment",
        default=None,
        help="Path to full sermon segment text file (Tier 3, optional)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Save results to JSON file (e.g., ablation/sermon_eval_results.json)",
    )
    args = parser.parse_args()

    evaluate_sermon(
        chunks_path=args.chunks,
        adapter_dir=args.adapter,
        base_model=args.base_model,
        ceiling_model=args.ceiling_model,
        segment_path=args.segment,
        output_file=args.output,
    )


if __name__ == "__main__":
    main()
