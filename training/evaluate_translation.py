#!/usr/bin/env python3
"""
evaluate_translation.py — Translation Evaluation: SacreBLEU/chrF++/COMET

Evaluate fine-tuned TranslateGemma on holdout Bible verses using three
complementary metrics. Includes per-genre breakdown and theological term spot-check.

Runs on both WSL (after training) and Mac (post-transfer verification).

Usage:
    python evaluate_translation.py
    python evaluate_translation.py --adapter fine_tuned_gemma_mi_A --test bible_data/holdout/verse_pairs_test.jsonl
    python evaluate_translation.py --spot-check-only
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
    test_data="bible_data/holdout/verse_pairs_test.jsonl",
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
        return None
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
        references.append(example["es"])
        hypotheses.append(translation)

        if (i + 1) % 100 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            eta = (len(test) - i - 1) / rate
            logger.info(f"  [{i + 1}/{len(test)}] {rate:.1f} verse/s, ETA: {eta:.0f}s")

    elapsed = time.time() - start_time
    logger.info(f"Translation complete in {elapsed:.0f}s ({len(test) / elapsed:.1f} verse/s)")

    # SacreBLEU
    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    logger.info(f"\nSacreBLEU: {bleu.score:.1f}")

    # chrF++
    chrf = sacrebleu.corpus_chrf(hypotheses, [references], word_order=2)
    logger.info(f"chrF++: {chrf.score:.1f}")

    # COMET (neural, highest human correlation)
    comet_val = None
    try:
        from comet import download_model, load_from_checkpoint

        comet_path = download_model("Unbabel/wmt22-comet-da")
        comet_model = load_from_checkpoint(comet_path)
        comet_input = [{"src": s, "mt": h, "ref": r} for s, h, r in zip(sources, hypotheses, references)]
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

    return scores


def evaluate_theological_terms(
    adapter_dir="fine_tuned_gemma_mi_A",
    base_model="google/translategemma-4b-it",
    marian_dir=None,
):
    """Spot-check critical theological term translations."""
    if marian_dir:
        model, tokenizer = load_marian_model(marian_dir)
        translate_fn = translate_marian
    else:
        model, tokenizer = load_gemma_model(base_model, adapter_dir)
        translate_fn = translate_gemma

    test_sentences = [
        ("Christ's atonement covers all sins.", "expiación"),
        ("The covenant between God and Abraham.", "pacto"),
        ("We are saved by grace through faith.", "gracia"),
        ("The righteousness of God is revealed.", "justicia"),
        ("James wrote about faith and works.", "Santiago"),  # Epistle context
        ("James and John were fishermen.", "Jacobo"),  # Apostle context
        ("He preached about sanctification.", "santificación"),
        ("The propitiation for our sins.", "propiciación"),
    ]

    correct = 0
    results = []
    logger.info("")
    for en_sentence, expected_es_term in test_sentences:
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
        results.append(
            {
                "en": en_sentence,
                "translation": translation,
                "expected_term": expected_es_term,
                "found": found,
            }
        )

    accuracy = correct / len(test_sentences)
    logger.info(f"\nTheological term accuracy: {correct}/{len(test_sentences)} ({accuracy:.0%})")
    return correct, len(test_sentences), results


def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned translation model quality")
    parser.add_argument(
        "--adapter", default="fine_tuned_gemma_mi_A", help="Path to QLoRA adapter directory (for TranslateGemma)"
    )
    parser.add_argument("--base-model", default="google/translategemma-4b-it", help="Base TranslateGemma model")
    parser.add_argument("--test", default="bible_data/holdout/verse_pairs_test.jsonl", help="Path to test JSONL")
    parser.add_argument(
        "--max-samples", type=int, default=None, help="Limit evaluation to N samples (useful for quick checks)"
    )
    parser.add_argument("--spot-check-only", action="store_true", help="Only run theological term spot-check")
    parser.add_argument(
        "--output-file", default=None, help="Save metrics to JSON file (e.g., metrics/translation_eval.json)"
    )
    parser.add_argument("--marian", default=None, help="Evaluate MarianMT model instead (path to fine-tuned dir)")
    args = parser.parse_args()

    if not args.spot_check_only:
        logger.info("=== Corpus-level metrics ===")
        evaluate_biblical_translation(
            adapter_dir=args.adapter,
            test_data=args.test,
            base_model=args.base_model,
            max_samples=args.max_samples,
            output_file=args.output_file,
            marian_dir=args.marian,
        )

    logger.info("\n=== Theological term spot-check ===")
    evaluate_theological_terms(
        adapter_dir=args.adapter,
        base_model=args.base_model,
        marian_dir=args.marian,
    )


if __name__ == "__main__":
    main()
