#!/usr/bin/env python3
"""
translation_qe.py — Offline Translation Quality Estimation

Reads A/B test CSV output and runs heavier QE metrics:
  - Length ratio check (real-time lightweight, included for completeness)
  - Untranslated content detection
  - LaBSE cross-lingual cosine similarity
  - Back-translation via MarianMT → BERTScore
  - CometKiwi (if available)

Usage:
    python translation_qe.py metrics/ab_metrics_*.csv
    python translation_qe.py metrics/ab_metrics_*.csv --output metrics/translation_qe.jsonl
    python translation_qe.py metrics/ab_metrics_*.csv --tiers 1      # lightweight only
    python translation_qe.py metrics/ab_metrics_*.csv --tiers 1,2    # + back-translation
"""

import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Tier 1: Lightweight checks (no model downloads)
# ---------------------------------------------------------------------------

_EN_STOPWORDS = re.compile(
    r'\b(the|and|of|that|have|for|not|with|you|this|but|his|from|they|'
    r'been|said|each|which|their|will|other|about|many|then|them|these|'
    r'would|could|should|because|into|after|before|between|under|through)\b',
    re.IGNORECASE,
)


def qe_length_ratio(source, translation):
    """Score 0-1 based on length ratio. Spanish typically 15-25% longer."""
    if not source or not translation:
        return 0.0
    ratio = len(translation) / len(source)
    if 0.9 <= ratio <= 1.6:
        return 1.0
    elif 0.7 <= ratio <= 2.0:
        return 0.7
    elif 0.5 <= ratio <= 2.5:
        return 0.4
    return 0.1


def qe_untranslated(source, translation):
    """Score 0-1 detecting untranslated English content in Spanish output."""
    if not translation:
        return 0.0
    en_matches = _EN_STOPWORDS.findall(translation)
    words = translation.split()
    if not words:
        return 0.0
    en_ratio = len(en_matches) / len(words)
    if en_ratio < 0.05:
        return 1.0
    elif en_ratio < 0.15:
        return 0.7
    elif en_ratio < 0.30:
        return 0.4
    return 0.1


def tier1_score(source, translation):
    """Combined Tier 1 score (no dependencies)."""
    lr = qe_length_ratio(source, translation)
    ut = qe_untranslated(source, translation)
    return {"length_ratio": lr, "untranslated": ut, "tier1": round((lr + ut) / 2, 3)}


# ---------------------------------------------------------------------------
# Tier 2: Back-translation + BERTScore (needs MarianMT + bert-score)
# ---------------------------------------------------------------------------

_backtranslation_model = None
_backtranslation_tokenizer = None


def _load_backtranslation():
    global _backtranslation_model, _backtranslation_tokenizer
    if _backtranslation_model is not None:
        return
    from transformers import MarianMTModel, MarianTokenizer
    model_id = "Helsinki-NLP/opus-mt-es-en"
    print(f"  Loading {model_id} for back-translation...")
    _backtranslation_tokenizer = MarianTokenizer.from_pretrained(model_id)
    _backtranslation_model = MarianMTModel.from_pretrained(model_id)
    print(f"  MarianMT ready")


def backtranslate(spanish_text):
    """Translate Spanish back to English using MarianMT."""
    _load_backtranslation()
    inputs = _backtranslation_tokenizer(spanish_text, return_tensors="pt",
                                         truncation=True, max_length=512)
    outputs = _backtranslation_model.generate(**inputs, max_new_tokens=256)
    return _backtranslation_tokenizer.decode(outputs[0], skip_special_tokens=True)


def tier2_score(source, translation):
    """Back-translate and compare with BERTScore."""
    try:
        from bert_score import score as bert_score
    except ImportError:
        print("  bert-score not installed. pip install bert-score", file=sys.stderr)
        return {"tier2": None, "backtranslation": None, "bert_f1": None}

    back = backtranslate(translation)
    P, R, F1 = bert_score([back], [source], lang="en", verbose=False)
    return {
        "backtranslation": back,
        "bert_p": round(P.item(), 3),
        "bert_r": round(R.item(), 3),
        "bert_f1": round(F1.item(), 3),
        "tier2": round(F1.item(), 3),
    }


# ---------------------------------------------------------------------------
# Tier 3: LaBSE cross-lingual similarity
# ---------------------------------------------------------------------------

_labse_model = None


def _load_labse():
    global _labse_model
    if _labse_model is not None:
        return
    from sentence_transformers import SentenceTransformer
    print("  Loading LaBSE for cross-lingual similarity...")
    _labse_model = SentenceTransformer("sentence-transformers/LaBSE")
    print("  LaBSE ready")


def tier3_labse(source, translation):
    """LaBSE cross-lingual cosine similarity."""
    _load_labse()
    import numpy as np
    embeddings = _labse_model.encode([source, translation])
    cos_sim = float(np.dot(embeddings[0], embeddings[1]) /
                    (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])))
    return {"labse_similarity": round(cos_sim, 3)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def process_csv(csv_path, tiers, output_path):
    """Process a CSV file and run QE on all rows."""
    results = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"Processing {len(rows)} rows from {csv_path}")
    print(f"Tiers enabled: {tiers}")

    for i, row in enumerate(rows):
        english = row.get("english", "")
        spanish_a = row.get("spanish_a", "")
        spanish_b = row.get("spanish_b", "")

        if not english:
            continue

        entry = {
            "chunk_id": row.get("chunk_id"),
            "english": english,
        }

        # Score both translations
        for label, spanish in [("a", spanish_a), ("b", spanish_b)]:
            if not spanish:
                continue
            scores = {}

            # Tier 1: always run
            scores.update(tier1_score(english, spanish))

            # Tier 2: back-translation
            if 2 in tiers:
                scores.update(tier2_score(english, spanish))

            # Tier 3: LaBSE
            if 3 in tiers:
                scores.update(tier3_labse(english, spanish))

            entry[f"qe_{label}"] = scores

        results.append(entry)

        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(rows)} processed")

    # Write output
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\nWrote {len(results)} QE results to {output_path}")

    # Summary
    if results:
        t1_scores = []
        for r in results:
            for key in ["qe_a", "qe_b"]:
                if key in r and "tier1" in r[key]:
                    t1_scores.append(r[key]["tier1"])
        if t1_scores:
            import numpy as np
            print(f"\nTier 1 QE summary:")
            print(f"  Mean: {np.mean(t1_scores):.3f}")
            print(f"  Median: {np.median(t1_scores):.3f}")
            print(f"  Min: {np.min(t1_scores):.3f}")
            flagged = sum(1 for s in t1_scores if s < 0.5)
            print(f"  Flagged (< 0.5): {flagged}/{len(t1_scores)}")


def main():
    parser = argparse.ArgumentParser(
        description="Offline translation quality estimation on A/B test CSV"
    )
    parser.add_argument("csv_files", nargs="+", help="CSV file(s) from dry_run_ab.py")
    parser.add_argument("--output", "-o", default="metrics/translation_qe.jsonl",
                        help="Output JSONL path (default: metrics/translation_qe.jsonl)")
    parser.add_argument("--tiers", default="1",
                        help="Comma-separated tier numbers to run (default: 1). "
                             "1=lightweight, 2=back-translation, 3=LaBSE")
    args = parser.parse_args()

    tiers = set(int(t.strip()) for t in args.tiers.split(","))

    for csv_path in args.csv_files:
        if not os.path.exists(csv_path):
            print(f"File not found: {csv_path}", file=sys.stderr)
            continue
        process_csv(csv_path, tiers, args.output)


if __name__ == "__main__":
    main()
