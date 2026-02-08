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
"""

import argparse
import json
import os
import sys

import sacrebleu
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(base_model, adapter_dir=None):
    """Load TranslateGemma with optional QLoRA adapter."""
    tokenizer = AutoTokenizer.from_pretrained(
        adapter_dir if adapter_dir and os.path.exists(adapter_dir) else base_model
    )

    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        load_in_4bit=True,
    )

    if adapter_dir and os.path.exists(adapter_dir):
        model = PeftModel.from_pretrained(base, adapter_dir)
        print(f"Loaded fine-tuned model from {adapter_dir}")
    else:
        model = base
        print(f"Using base model: {base_model}")

    return model, tokenizer


def translate(model, tokenizer, text):
    """Translate English to Spanish using TranslateGemma chat template."""
    messages = [{"role": "user", "content": [
        {"type": "text", "source_lang_code": "en",
         "target_lang_code": "es", "text": text}
    ]}]
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=256)

    translation = tokenizer.decode(
        output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
    )
    return translation.strip()


def evaluate_biblical_translation(
    adapter_dir="fine_tuned_gemma_mi_A",
    test_data="bible_data/holdout/verse_pairs_test.jsonl",
    base_model="google/translategemma-4b-it",
    max_samples=None,
):
    """Evaluate fine-tuned TranslateGemma on holdout Bible verses.

    Metrics:
    - SacreBLEU: n-gram precision (standard MT metric)
    - chrF++: character-level n-grams (better for Spanish morphology)
    - COMET: neural metric with highest correlation to human judgments
    """
    model, tokenizer = load_model(base_model, adapter_dir)

    # Load test set
    if not os.path.exists(test_data):
        print(f"Test data not found: {test_data}")
        return None
    test = [json.loads(line) for line in open(test_data, encoding="utf-8")]
    if max_samples:
        test = test[:max_samples]

    print(f"Translating {len(test)} verses...")
    sources, references, hypotheses = [], [], []

    for i, example in enumerate(test):
        translation = translate(model, tokenizer, example["en"])
        sources.append(example["en"])
        references.append(example["es"])
        hypotheses.append(translation)

        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(test)}]")

    # SacreBLEU
    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    print(f"\nSacreBLEU: {bleu.score:.1f}")

    # chrF++
    chrf = sacrebleu.corpus_chrf(hypotheses, [references], word_order=2)
    print(f"chrF++: {chrf.score:.1f}")

    # COMET (neural, highest human correlation)
    try:
        from comet import download_model, load_from_checkpoint
        comet_path = download_model("Unbabel/wmt22-comet-da")
        comet_model = load_from_checkpoint(comet_path)
        comet_input = [{"src": s, "mt": h, "ref": r}
                       for s, h, r in zip(sources, hypotheses, references)]
        comet_score = comet_model.predict(comet_input, batch_size=8)
        print(f"COMET: {comet_score.system_score:.4f}")
        comet_val = comet_score.system_score
    except Exception as e:
        print(f"COMET unavailable: {e}")
        comet_val = None

    # Per-genre breakdown
    genres = {
        "pentateuch": range(1, 6),       # Genesis-Deuteronomy
        "history": range(6, 18),         # Joshua-Esther
        "poetry": range(18, 23),         # Job-Song of Solomon
        "prophecy": range(23, 40),       # Isaiah-Malachi
        "gospels": range(40, 44),        # Matthew-John
        "epistles": range(44, 66),       # Acts-Jude
        "apocalyptic": range(66, 67),    # Revelation
    }
    print("\nPer-genre BLEU:")
    for genre, book_range in genres.items():
        genre_hyps = [h for h, ex in zip(hypotheses, test)
                      if int(str(ex.get("verse_id", "01001001"))[:2]) in book_range]
        genre_refs = [r for r, ex in zip(references, test)
                      if int(str(ex.get("verse_id", "01001001"))[:2]) in book_range]
        if genre_hyps:
            genre_bleu = sacrebleu.corpus_bleu(genre_hyps, [genre_refs])
            print(f"  {genre:15s}: BLEU {genre_bleu.score:5.1f} ({len(genre_hyps)} verses)")

    return {"bleu": bleu.score, "chrf": chrf.score, "comet": comet_val}


def evaluate_theological_terms(
    adapter_dir="fine_tuned_gemma_mi_A",
    base_model="google/translategemma-4b-it",
):
    """Spot-check critical theological term translations."""
    model, tokenizer = load_model(base_model, adapter_dir)

    test_sentences = [
        ("Christ's atonement covers all sins.", "expiación"),
        ("The covenant between God and Abraham.", "pacto"),
        ("We are saved by grace through faith.", "gracia"),
        ("The righteousness of God is revealed.", "justicia"),
        ("James wrote about faith and works.", "Santiago"),       # Epistle context
        ("James and John were fishermen.", "Jacobo"),             # Apostle context
        ("He preached about sanctification.", "santificación"),
        ("The propitiation for our sins.", "propiciación"),
    ]

    correct = 0
    print()
    for en_sentence, expected_es_term in test_sentences:
        translation = translate(model, tokenizer, en_sentence)
        found = expected_es_term.lower() in translation.lower()
        correct += found
        status = "PASS" if found else "FAIL"
        print(f"  [{status}] '{en_sentence}'")
        print(f"         -> '{translation}'")
        print(f"         expected: '{expected_es_term}' {'(found)' if found else '(MISSING)'}")

    print(f"\nTheological term accuracy: {correct}/{len(test_sentences)} "
          f"({correct/len(test_sentences):.0%})")
    return correct, len(test_sentences)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate fine-tuned TranslateGemma translation quality"
    )
    parser.add_argument("--adapter", default="fine_tuned_gemma_mi_A",
                        help="Path to QLoRA adapter directory")
    parser.add_argument("--base-model", default="google/translategemma-4b-it",
                        help="Base TranslateGemma model")
    parser.add_argument("--test", default="bible_data/holdout/verse_pairs_test.jsonl",
                        help="Path to test JSONL")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit evaluation to N samples")
    parser.add_argument("--spot-check-only", action="store_true",
                        help="Only run theological term spot-check")
    args = parser.parse_args()

    if not args.spot_check_only:
        print("=== Corpus-level metrics ===")
        scores = evaluate_biblical_translation(
            adapter_dir=args.adapter,
            test_data=args.test,
            base_model=args.base_model,
            max_samples=args.max_samples,
        )

    print("\n=== Theological term spot-check ===")
    evaluate_theological_terms(
        adapter_dir=args.adapter,
        base_model=args.base_model,
    )


if __name__ == "__main__":
    main()
