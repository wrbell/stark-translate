#!/usr/bin/env python3
"""
evaluate_translation.py — Translation Evaluation: SacreBLEU/chrF++/COMET

Evaluate fine-tuned TranslateGemma on holdout Bible verses using three
complementary metrics. Includes per-genre breakdown and glossary-based
evaluation with optional base-model regression detection.

Runs on both WSL (after training) and Mac (post-transfer verification).

Usage:
    python evaluate_translation.py
    python evaluate_translation.py --adapter fine_tuned_gemma_mi_A --test bible_data/aligned/verse_pairs_test.jsonl
    python evaluate_translation.py --glossary-only
    python evaluate_translation.py --glossary-only --compare-base
    python evaluate_translation.py --glossary-file bible_data/glossary/glossary_pairs.jsonl
    python evaluate_translation.py --base-model google/translategemma-12b-it --adapter fine_tuned_gemma_mi_B
    python evaluate_translation.py --output-file metrics/translation_eval.json
    python evaluate_translation.py --marian fine_tuned_marian_mi   # Evaluate MarianMT instead
"""

import argparse
import json
import logging
import os
import time

import sacrebleu
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_gemma_model(base_model, adapter_dir=None):
    """Load TranslateGemma with optional QLoRA adapter."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(
        adapter_dir if adapter_dir and os.path.exists(adapter_dir) else base_model
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
    )

    if adapter_dir and os.path.exists(adapter_dir):
        model = PeftModel.from_pretrained(base, adapter_dir)
        logger.info(f"Loaded fine-tuned model from {adapter_dir}")
    else:
        model = base
        logger.info(f"Using base model: {base_model}")

    return model, tokenizer


def load_marian_model(model_dir):
    """Load fine-tuned MarianMT model."""
    from transformers import MarianMTModel, MarianTokenizer

    tokenizer = MarianTokenizer.from_pretrained(model_dir)
    model = MarianMTModel.from_pretrained(model_dir)
    if torch.cuda.is_available():
        model = model.to("cuda")
    logger.info(f"Loaded MarianMT from {model_dir}")
    return model, tokenizer


def translate_gemma(model, tokenizer, text):
    """Translate English to Spanish using TranslateGemma chat template."""
    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "source_lang_code": "en", "target_lang_code": "es", "text": text}],
        }
    ]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=256)

    translation = tokenizer.decode(output[0][inputs.input_ids.shape[1] :], skip_special_tokens=True)
    return translation.strip()


def translate_marian(model, tokenizer, text):
    """Translate English to Spanish using MarianMT."""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=128)

    translation = tokenizer.decode(output[0], skip_special_tokens=True)
    return translation.strip()


def evaluate_biblical_translation(
    adapter_dir="fine_tuned_gemma_mi_A",
    test_data="bible_data/aligned/verse_pairs_test.jsonl",
    base_model="google/translategemma-4b-it",
    max_samples=None,
    output_file=None,
    marian_dir=None,
):
    """Evaluate translation model on holdout Bible verses.

    Metrics:
    - SacreBLEU: n-gram precision (standard MT metric)
    - chrF++: character-level n-grams (better for Spanish morphology)
    - COMET: neural metric with highest correlation to human judgments

    Returns (scores_dict, model, tokenizer, translate_fn) so caller can
    reuse the loaded model for glossary evaluation.
    """
    # Load model
    if marian_dir:
        model, tokenizer = load_marian_model(marian_dir)
        translate_fn = translate_marian
    else:
        model, tokenizer = load_gemma_model(base_model, adapter_dir)
        translate_fn = translate_gemma

    # Load test set
    if not os.path.exists(test_data):
        logger.error(f"Test data not found: {test_data}")
        return None, model, tokenizer, translate_fn
    test = [json.loads(line) for line in open(test_data, encoding="utf-8")]
    if max_samples:
        test = test[:max_samples]

    logger.info(f"Translating {len(test)} verses...")
    sources, references, hypotheses = [], [], []
    start_time = time.time()

    for i, example in enumerate(test):
        try:
            translation = translate_fn(model, tokenizer, example["en"])
        except Exception as e:
            logger.warning(f"Translation failed for example {i}: {e}")
            translation = ""
        sources.append(example["en"])
        references.append(example.get("es", ""))
        hypotheses.append(translation)

        if (i + 1) % 100 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            eta = (len(test) - i - 1) / rate
            logger.info(f"  [{i + 1}/{len(test)}] {rate:.1f} verse/s, ETA: {eta:.0f}s")

    elapsed = time.time() - start_time
    logger.info(f"Translation complete in {elapsed:.0f}s ({len(test) / elapsed:.1f} verse/s)")

    # Filter to examples with references for metric computation
    has_refs = [bool(r) for r in references]
    if not any(has_refs):
        logger.warning("No reference translations found — skipping BLEU/chrF++/COMET metrics")
        scores = {
            "bleu": None,
            "chrf": None,
            "comet": None,
            "genres": {},
            "num_verses": len(test),
            "elapsed_s": round(elapsed, 1),
        }
        if output_file:
            os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
            with open(output_file, "w") as f:
                json.dump(scores, f, indent=2)
            logger.info(f"\nMetrics saved to {output_file}")
        return scores, model, tokenizer, translate_fn

    ref_hyp = [(r, h) for r, h, hr in zip(references, hypotheses, has_refs) if hr]
    ref_filtered = [r for r, _ in ref_hyp]
    hyp_filtered = [h for _, h in ref_hyp]

    # SacreBLEU
    bleu = sacrebleu.corpus_bleu(hyp_filtered, [ref_filtered])
    logger.info(f"\nSacreBLEU: {bleu.score:.1f}")

    # chrF++
    chrf = sacrebleu.corpus_chrf(hyp_filtered, [ref_filtered], word_order=2)
    logger.info(f"chrF++: {chrf.score:.1f}")

    # COMET (neural, highest human correlation)
    comet_val = None
    try:
        from comet import download_model, load_from_checkpoint

        comet_path = download_model("Unbabel/wmt22-comet-da")
        comet_model = load_from_checkpoint(comet_path)
        src_filtered = [s for s, hr in zip(sources, has_refs) if hr]
        comet_input = [{"src": s, "mt": h, "ref": r} for s, h, r in zip(src_filtered, hyp_filtered, ref_filtered)]
        comet_score = comet_model.predict(comet_input, batch_size=8)
        comet_val = comet_score.system_score
        logger.info(f"COMET: {comet_val:.4f}")
    except ImportError:
        logger.warning("COMET unavailable (install unbabel-comet). Skipping.")
    except Exception as e:
        logger.warning(f"COMET error: {e}")

    # Per-genre breakdown
    genres = {
        "pentateuch": range(1, 6),  # Genesis-Deuteronomy
        "history": range(6, 18),  # Joshua-Esther
        "poetry": range(18, 23),  # Job-Song of Solomon
        "prophecy": range(23, 40),  # Isaiah-Malachi
        "gospels": range(40, 44),  # Matthew-John
        "epistles": range(44, 66),  # Acts-Jude
        "apocalyptic": range(66, 67),  # Revelation
    }
    logger.info("\nPer-genre BLEU:")
    genre_scores = {}
    for genre, book_range in genres.items():
        genre_hyps = [
            h for h, ex in zip(hypotheses, test) if int(str(ex.get("verse_id", "01001001"))[:2]) in book_range
        ]
        genre_refs = [
            r for r, ex in zip(references, test) if int(str(ex.get("verse_id", "01001001"))[:2]) in book_range
        ]
        if genre_hyps:
            genre_bleu = sacrebleu.corpus_bleu(genre_hyps, [genre_refs])
            genre_scores[genre] = {
                "bleu": genre_bleu.score,
                "count": len(genre_hyps),
            }
            logger.info(f"  {genre:15s}: BLEU {genre_bleu.score:5.1f} ({len(genre_hyps)} verses)")

    scores = {
        "bleu": bleu.score,
        "chrf": chrf.score,
        "comet": comet_val,
        "genres": genre_scores,
        "num_verses": len(test),
        "elapsed_s": round(elapsed, 1),
    }

    # Save metrics to file
    if output_file:
        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(scores, f, indent=2)
        logger.info(f"\nMetrics saved to {output_file}")

    return scores, model, tokenizer, translate_fn


def evaluate_glossary_terms(
    model,
    tokenizer,
    translate_fn,
    glossary_file="bible_data/glossary/glossary_pairs.jsonl",
    base_results=None,
):
    """Evaluate translation quality on glossary sentence pairs.

    Loads sentence-level pairs from the glossary file (filtering out bare
    1-2 word terms via ``len(en.split()) > 4``), translates each sentence,
    and checks whether the expected Spanish term appears as a substring in
    the translation output.

    Args:
        model: Already-loaded translation model.
        tokenizer: Corresponding tokenizer.
        translate_fn: Callable(model, tokenizer, text) -> str.
        glossary_file: Path to JSONL glossary (each line has "en" and "es").
        base_results: Optional list of dicts from a prior base-model run.
            Each dict must contain an "found" bool.  When provided, any term
            that the base model got right but this model missed is counted as
            a regression.

    Returns:
        Dict with adapter_recall, base_recall (if base_results given), total,
        regressions count, regression_details list, and full details list.
    """
    if not os.path.exists(glossary_file):
        logger.error(f"Glossary file not found: {glossary_file}")
        return None

    # Load and filter to sentence-level pairs
    with open(glossary_file, encoding="utf-8") as f:
        all_entries = [json.loads(line) for line in f]

    pairs = [entry for entry in all_entries if len(entry["en"].split()) > 4]
    logger.info(
        f"Loaded {len(all_entries)} glossary entries, {len(pairs)} sentence-level pairs "
        f"(filtered {len(all_entries) - len(pairs)} bare terms)"
    )

    correct = 0
    details = []
    regressions = []

    for i, entry in enumerate(pairs):
        en_sentence = entry["en"]
        expected_es_term = entry["es"]

        try:
            translation = translate_fn(model, tokenizer, en_sentence)
        except Exception as e:
            translation = f"[ERROR: {e}]"

        found = expected_es_term.lower() in translation.lower()
        correct += found
        status = "PASS" if found else "FAIL"

        logger.info(f"  [{status}] '{en_sentence}'")
        logger.info(f"         -> '{translation}'")
        logger.info(f"         expected: '{expected_es_term}' {'(found)' if found else '(MISSING)'}")

        result = {
            "en": en_sentence,
            "expected_es": expected_es_term,
            "translation": translation,
            "found": found,
        }
        details.append(result)

        # Track regressions: base got it right, adapter missed it
        if base_results and i < len(base_results) and base_results[i].get("found") and not found:
            regressions.append(
                {
                    "en": en_sentence,
                    "expected_es": expected_es_term,
                    "adapter_translation": translation,
                }
            )

    total = len(pairs)
    logger.info(f"\nGlossary recall: {correct}/{total} ({correct / total:.0%})" if total else "\nNo pairs to evaluate")

    result_dict = {
        "adapter_recall": correct,
        "total": total,
        "details": details,
        "regressions": len(regressions),
        "regression_details": regressions,
    }

    if base_results is not None:
        base_recall = sum(1 for r in base_results if r.get("found"))
        result_dict["base_recall"] = base_recall
        logger.info(f"Base recall: {base_recall}/{total}")
        logger.info(f"Regressions (base got right, adapter missed): {len(regressions)}")
        if regressions:
            for reg in regressions:
                logger.info(f"  REGRESSION: '{reg['en']}' — expected '{reg['expected_es']}'")

    return result_dict


def evaluate_deepl_reference(
    model,
    tokenizer,
    translate_fn,
    test_data,
    deepl_key,
    glossary_file="bible_data/glossary/glossary_pairs.jsonl",
    max_samples=50,
):
    """Compare adapter vs DeepL on holdout verses.

    Translates a sample of holdout verses with both the adapter and DeepL,
    then computes COMET(adapter_output, deepl_output) to measure proximity
    to commercial quality.
    """
    import random

    import deepl

    # Load test set
    if not os.path.exists(test_data):
        logger.warning(f"Test data not found for DeepL comparison: {test_data}")
        return None

    with open(test_data, encoding="utf-8") as f:
        all_test = [json.loads(line) for line in f]

    # Sample
    rng = random.Random(42)
    sample = rng.sample(all_test, min(max_samples, len(all_test)))
    logger.info(f"DeepL reference comparison on {len(sample)} holdout verses")

    # Translate with adapter
    adapter_outputs = []
    for ex in sample:
        try:
            adapter_outputs.append(translate_fn(model, tokenizer, ex["en"]))
        except Exception:
            adapter_outputs.append("")

    # Translate with DeepL
    translator = deepl.Translator(deepl_key)

    # Build glossary
    glossary = None
    try:
        if os.path.exists(glossary_file):
            with open(glossary_file, encoding="utf-8") as f:
                all_entries = [json.loads(line) for line in f]
            bare_terms = {entry["en"]: entry["es"] for entry in all_entries if len(entry["en"].split()) <= 2}
            if bare_terms:
                glossary = translator.create_glossary(
                    "eval-deepl-reference",
                    source_lang="en",
                    target_lang="es",
                    entries=bare_terms,
                )
    except Exception as e:
        logger.warning(f"Could not create DeepL glossary: {e}")

    deepl_outputs = []
    for ex in sample:
        try:
            kwargs = {"text": ex["en"], "target_lang": "ES", "source_lang": "EN"}
            if glossary:
                kwargs["glossary"] = glossary
            result = translator.translate_text(**kwargs)
            deepl_outputs.append(result.text)
        except Exception:
            deepl_outputs.append("")

    # Clean up glossary
    if glossary:
        try:
            translator.delete_glossary(glossary)
        except Exception:
            pass

    # Compute COMET (adapter vs DeepL as reference)
    comet_score = None
    try:
        from comet import download_model, load_from_checkpoint

        comet_path = download_model("Unbabel/wmt22-comet-da")
        comet_model = load_from_checkpoint(comet_path)

        valid = [(ex, ao, do) for ex, ao, do in zip(sample, adapter_outputs, deepl_outputs) if ao and do]
        if valid:
            comet_input = [{"src": ex["en"], "mt": ao, "ref": do} for ex, ao, do in valid]
            comet_score = comet_model.predict(comet_input, batch_size=8).system_score
            logger.info(f"  COMET (adapter vs DeepL reference): {comet_score:.4f}")
            logger.info(f"  Valid pairs: {len(valid)}/{len(sample)}")

        del comet_model
    except ImportError:
        logger.warning("COMET unavailable. Skipping DeepL reference COMET.")
    except Exception as e:
        logger.warning(f"COMET error in DeepL comparison: {e}")

    # Also compute chrF++
    valid_adapter = [ao for ao, do in zip(adapter_outputs, deepl_outputs) if ao and do]
    valid_deepl = [do for ao, do in zip(adapter_outputs, deepl_outputs) if ao and do]
    chrf_score = None
    if valid_adapter:
        chrf_score = sacrebleu.corpus_chrf(valid_adapter, [valid_deepl], word_order=2).score
        logger.info(f"  chrF++ (adapter vs DeepL reference): {chrf_score:.1f}")

    return {
        "comet_vs_deepl": comet_score,
        "chrf_vs_deepl": chrf_score,
        "sample_size": len(sample),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned translation model quality")
    parser.add_argument(
        "--adapter", default="fine_tuned_gemma_mi_A", help="Path to QLoRA adapter directory (for TranslateGemma)"
    )
    parser.add_argument("--base-model", default="google/translategemma-4b-it", help="Base TranslateGemma model")
    parser.add_argument("--test", default="bible_data/aligned/verse_pairs_test.jsonl", help="Path to test JSONL")
    parser.add_argument(
        "--max-samples", type=int, default=None, help="Limit evaluation to N samples (useful for quick checks)"
    )
    parser.add_argument(
        "--glossary-only", action="store_true", help="Only run glossary evaluation (skip corpus metrics)"
    )
    parser.add_argument(
        "--glossary-file",
        default="bible_data/glossary/glossary_pairs.jsonl",
        help="Path to glossary JSONL file (default: bible_data/glossary/glossary_pairs.jsonl)",
    )
    parser.add_argument(
        "--compare-base",
        action="store_true",
        help="When an adapter is provided, also run base model on glossary to detect regressions",
    )
    parser.add_argument(
        "--output-file", default=None, help="Save metrics to JSON file (e.g., metrics/translation_eval.json)"
    )
    parser.add_argument("--marian", default=None, help="Evaluate MarianMT model instead (path to fine-tuned dir)")
    parser.add_argument(
        "--deepl-key",
        default=None,
        help="DeepL Pro API key (enables DeepL reference comparison on holdout verses)",
    )
    args = parser.parse_args()

    adapter = None if args.adapter.lower() == "none" else args.adapter

    # Track model/tokenizer/translate_fn so we don't double-load
    model, tokenizer, translate_fn = None, None, None

    if not args.glossary_only:
        logger.info("=== Corpus-level metrics ===")
        result = evaluate_biblical_translation(
            adapter_dir=adapter,
            test_data=args.test,
            base_model=args.base_model,
            max_samples=args.max_samples,
            output_file=args.output_file,
            marian_dir=args.marian,
        )
        if result is not None:
            _scores, model, tokenizer, translate_fn = result

    logger.info("\n=== Glossary evaluation ===")

    base_results = None

    if args.compare_base and adapter and not args.marian:
        # Run base model first to establish baseline
        logger.info("--- Running base model for regression comparison ---")
        base_model_obj, base_tokenizer = load_gemma_model(args.base_model, adapter_dir=None)
        base_translate_fn = translate_gemma

        base_result = evaluate_glossary_terms(
            model=base_model_obj,
            tokenizer=base_tokenizer,
            translate_fn=base_translate_fn,
            glossary_file=args.glossary_file,
        )
        if base_result is not None:
            base_results = base_result["details"]

        # Free base model before loading adapter
        del base_model_obj, base_tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Need to load adapter model fresh for glossary
        model, tokenizer, translate_fn = None, None, None

    # Load model for glossary eval if not already loaded
    if model is None:
        if args.marian:
            model, tokenizer = load_marian_model(args.marian)
            translate_fn = translate_marian
        else:
            model, tokenizer = load_gemma_model(args.base_model, adapter)
            translate_fn = translate_gemma

    evaluate_glossary_terms(
        model=model,
        tokenizer=tokenizer,
        translate_fn=translate_fn,
        glossary_file=args.glossary_file,
        base_results=base_results,
    )

    # --- Optional DeepL reference comparison ---
    if args.deepl_key and not args.marian:
        logger.info("\n=== DeepL reference comparison ===")
        evaluate_deepl_reference(
            model=model,
            tokenizer=tokenizer,
            translate_fn=translate_fn,
            test_data=args.test,
            deepl_key=args.deepl_key,
            glossary_file=args.glossary_file,
            max_samples=50,
        )


if __name__ == "__main__":
    main()
