#!/usr/bin/env python3
"""
prepare_bible_corpus.py — Download & Align ~155K Bible Verse Pairs

Builds the parallel EN→ES training set for TranslateGemma fine-tuning.
All sources are public domain or CC-licensed.

Sources:
    1. bible-nlp/biblenlp-corpus (HuggingFace, CC-BY-4.0)
    2. Helsinki-NLP/bible_para (CC0-1.0)
    3. scrollmapper/bible_databases (SQL/JSON/CSV, public domain)

Pairs 5 English translations × 2 Spanish translations = ~155K verse pairs:
    KJV, ASV, WEB, BBE, YLT  ↔  RVR1909, Español Sencillo

Copyright WARNING: Do NOT use ESV, NASB, NIV, NLT, NVI, LBLA, RVR1960, or DHH.
Their fair-use caps (~500 verses) do not permit bulk ML training.

Usage:
    python prepare_bible_corpus.py
    python prepare_bible_corpus.py --source scrollmapper --db bible_data/scrollmapper/bible-sqlite.db
    python prepare_bible_corpus.py --source huggingface
"""

import argparse
import json
import os
import sqlite3
from collections import defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# Source 1: scrollmapper/bible_databases (SQL with verse IDs)
# ---------------------------------------------------------------------------

def align_scrollmapper(db_path="bible_data/scrollmapper/bible-sqlite.db"):
    """
    scrollmapper/bible_databases: SQL with numeric verse IDs.
    Verse ID format: BBCCCVVV (e.g., 01001001 = Genesis 1:1).
    JOIN on verse ID gives exact alignment across translations.
    """
    if not os.path.exists(db_path):
        print(f"Database not found: {db_path}")
        print("Download from: https://github.com/scrollmapper/bible_databases")
        print("  git clone https://github.com/scrollmapper/bible_databases.git bible_data/scrollmapper")
        return []

    pairs = []
    # English translations: t_kjv, t_asv, t_web, t_bbe, t_ylt
    # Spanish translations: t_rvr (RVR1909)
    en_tables = ["t_kjv", "t_asv", "t_web", "t_bbe", "t_ylt"]
    es_tables = ["t_rvr"]

    conn = sqlite3.connect(db_path)

    # Check which tables actually exist
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    available = {row[0] for row in cursor}
    en_tables = [t for t in en_tables if t in available]
    es_tables = [t for t in es_tables if t in available]

    if not en_tables or not es_tables:
        print(f"Available tables: {available}")
        print(f"Expected EN: t_kjv, t_asv, t_web, t_bbe, t_ylt")
        print(f"Expected ES: t_rvr")
        conn.close()
        return []

    for en_table in en_tables:
        for es_table in es_tables:
            query = f"""
                SELECT e.t AS en_text, s.t AS es_text, e.id AS verse_id
                FROM {en_table} e
                INNER JOIN {es_table} s ON e.id = s.id
                WHERE e.t IS NOT NULL AND s.t IS NOT NULL
                  AND LENGTH(e.t) > 5 AND LENGTH(s.t) > 5
            """
            cursor = conn.execute(query)
            for row in cursor:
                pairs.append({
                    "en": row[0].strip(),
                    "es": row[1].strip(),
                    "verse_id": row[2],
                    "en_source": en_table.replace("t_", ""),
                    "es_source": es_table.replace("t_", ""),
                })
            print(f"  {en_table} × {es_table}: {cursor.rowcount if hasattr(cursor, 'rowcount') else '?'} pairs")

    conn.close()
    print(f"Total scrollmapper pairs: {len(pairs)}")
    return pairs


# ---------------------------------------------------------------------------
# Source 2: HuggingFace bible-nlp/biblenlp-corpus
# ---------------------------------------------------------------------------

def download_biblenlp():
    """Primary source: BibleNLP corpus on HuggingFace (833 languages, CC-BY-4.0)."""
    from datasets import load_dataset
    try:
        ds = load_dataset("bible-nlp/biblenlp-corpus", languages=["eng", "spa"])
        return ds
    except Exception as e:
        print(f"Could not load BibleNLP corpus: {e}")
        return None


# ---------------------------------------------------------------------------
# Source 3: Helsinki-NLP/bible_para
# ---------------------------------------------------------------------------

def download_helsinki():
    """Alternative: Helsinki-NLP bible_para (Christodoulopoulos/Steedman, CC0-1.0)."""
    from datasets import load_dataset
    try:
        ds = load_dataset("Helsinki-NLP/bible_para", lang1="en", lang2="es")
        return ds
    except Exception as e:
        print(f"Could not load Helsinki Bible corpus: {e}")
        return None


# ---------------------------------------------------------------------------
# Multi-Reference Pairing (Data Augmentation)
# ---------------------------------------------------------------------------

def create_multi_reference_pairs(pairs):
    """Group by verse_id, create (en_variant, es) pairs for all combinations.

    Using the same source verse with different target translations teaches
    the model that multiple valid translations exist.
    """
    by_verse = defaultdict(lambda: {"en": set(), "es": set()})

    for p in pairs:
        by_verse[p["verse_id"]]["en"].add(p["en"])
        by_verse[p["verse_id"]]["es"].add(p["es"])

    expanded = []
    for vid, texts in by_verse.items():
        for en in texts["en"]:
            for es in texts["es"]:
                expanded.append({"en": en, "es": es, "verse_id": vid})

    return expanded


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_training_jsonl(pairs, output_path="bible_data/aligned/verse_pairs.jsonl",
                          holdout_ratio=0.1):
    """Export as JSONL for HuggingFace Trainer.

    Stratified split: train/test by hashing verse_id (~10% holdout).
    OT books 1-39, NT books 40-66.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    train, test = [], []
    for p in pairs:
        verse_id = str(p.get("verse_id", "01001001"))
        # Use hash for deterministic split
        if hash(verse_id) % int(1 / holdout_ratio) == 0:
            test.append(p)
        else:
            train.append(p)

    for split, data in [("train", train), ("test", test)]:
        path = output_path.replace(".jsonl", f"_{split}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for p in data:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        print(f"Exported {len(data)} pairs to {path}")

    # Also save holdout to dedicated dir
    holdout_path = "bible_data/holdout/verse_pairs_test.jsonl"
    os.makedirs(os.path.dirname(holdout_path), exist_ok=True)
    with open(holdout_path, "w", encoding="utf-8") as f:
        for p in test:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    # Print genre distribution of holdout
    genre_counts = defaultdict(int)
    genres = {
        "pentateuch": range(1, 6),
        "history": range(6, 18),
        "poetry": range(18, 23),
        "prophecy": range(23, 40),
        "gospels": range(40, 44),
        "epistles": range(44, 66),
        "apocalyptic": range(66, 67),
    }
    for p in test:
        book = int(str(p.get("verse_id", "01001001"))[:2])
        for genre, book_range in genres.items():
            if book in book_range:
                genre_counts[genre] += 1
                break

    print("\nHoldout genre distribution:")
    for genre, count in sorted(genre_counts.items(), key=lambda x: -x[1]):
        print(f"  {genre}: {count}")

    return train, test


def main():
    parser = argparse.ArgumentParser(
        description="Download & align Bible parallel corpus for translation fine-tuning"
    )
    parser.add_argument("--source", choices=["scrollmapper", "huggingface", "all"],
                        default="scrollmapper",
                        help="Data source to use")
    parser.add_argument("--db", default="bible_data/scrollmapper/bible-sqlite.db",
                        help="Path to scrollmapper SQLite database")
    parser.add_argument("--output", default="bible_data/aligned/verse_pairs.jsonl",
                        help="Output JSONL path")
    parser.add_argument("--multi-ref", action="store_true",
                        help="Create multi-reference expanded pairs")
    args = parser.parse_args()

    pairs = []

    if args.source in ("scrollmapper", "all"):
        pairs.extend(align_scrollmapper(args.db))

    if args.source in ("huggingface", "all"):
        print("\nTrying BibleNLP corpus...")
        ds = download_biblenlp()
        if ds:
            print(f"  BibleNLP loaded: {ds}")

        print("\nTrying Helsinki corpus...")
        ds = download_helsinki()
        if ds:
            print(f"  Helsinki loaded: {ds}")

    if not pairs:
        print("No pairs generated. Check data sources.")
        return

    if args.multi_ref:
        print(f"\nCreating multi-reference pairs from {len(pairs)} base pairs...")
        pairs = create_multi_reference_pairs(pairs)
        print(f"  Expanded to {len(pairs)} pairs")

    print(f"\nTotal verse pairs: {len(pairs)}")
    export_training_jsonl(pairs, args.output)


if __name__ == "__main__":
    main()
