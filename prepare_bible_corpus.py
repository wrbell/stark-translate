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

def align_scrollmapper(db_dir="bible_data/scrollmapper/formats/sqlite"):
    """
    scrollmapper/bible_databases: individual SQLite DBs per translation.
    Each DB has {Name}_verses table with (id, book_id, chapter, verse, text).
    The 'id' column is a sequential verse index (1-31102) that aligns across
    all Protestant canon translations, making cross-translation JOIN trivial.

    Also supports legacy single-DB mode with t_kjv, t_asv, etc. tables.
    """
    db_dir = Path(db_dir)

    # Translation mappings: filename -> short name
    en_translations = {
        "KJV": "kjv",
        "ASV": "asv",
        "BBE": "bbe",
        "YLT": "ylt",
        "OEB": "web",      # Open English Bible = WEB equivalent
    }
    es_translations = {
        "SpaRV": "rvr1909",
        "SpaRVG": "rvg",
        "SpaPlatense": "platense",
    }

    # Load all verse texts keyed by (translation, verse_id)
    def load_translation(db_name):
        db_path = db_dir / f"{db_name}.db"
        if not db_path.exists():
            print(f"  WARNING: {db_path} not found, skipping")
            return {}
        conn = sqlite3.connect(str(db_path))
        table_name = f"{db_name}_verses"
        try:
            rows = conn.execute(
                f"SELECT id, text FROM {table_name} WHERE text IS NOT NULL AND LENGTH(text) > 5"
            ).fetchall()
        except sqlite3.OperationalError:
            print(f"  WARNING: table {table_name} not found in {db_path}")
            conn.close()
            return {}
        conn.close()
        return {row[0]: row[1].strip() for row in rows}

    # Load book names from KJV for verse_id -> reference mapping
    book_names = {}
    kjv_db = db_dir / "KJV.db"
    if kjv_db.exists():
        conn = sqlite3.connect(str(kjv_db))
        try:
            book_names = dict(conn.execute("SELECT id, name FROM KJV_books").fetchall())
        except sqlite3.OperationalError:
            pass
        conn.close()

    # Load all translations
    print("Loading English translations...")
    en_data = {}
    for db_name, short in en_translations.items():
        verses = load_translation(db_name)
        if verses:
            en_data[short] = verses
            print(f"  {db_name} ({short}): {len(verses)} verses")

    print("Loading Spanish translations...")
    es_data = {}
    for db_name, short in es_translations.items():
        verses = load_translation(db_name)
        if verses:
            es_data[short] = verses
            print(f"  {db_name} ({short}): {len(verses)} verses")

    if not en_data or not es_data:
        print(f"No translations found in {db_dir}")
        print("Download: git clone https://github.com/scrollmapper/bible_databases.git bible_data/scrollmapper")
        return []

    # Build pairs by joining on verse ID
    pairs = []
    for en_name, en_verses in en_data.items():
        for es_name, es_verses in es_data.items():
            common_ids = set(en_verses.keys()) & set(es_verses.keys())
            count = 0
            for vid in sorted(common_ids):
                en_text = en_verses[vid]
                es_text = es_verses[vid]
                if len(en_text) > 5 and len(es_text) > 5:
                    pairs.append({
                        "en": en_text,
                        "es": es_text,
                        "verse_id": vid,
                        "en_source": en_name,
                        "es_source": es_name,
                    })
                    count += 1
            print(f"  {en_name} x {es_name}: {count} pairs")

    print(f"\nTotal scrollmapper pairs: {len(pairs)}")
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

def export_individual_translations(pairs):
    """Export individual EN and ES translation files for reference."""
    en_by_source = defaultdict(list)
    es_by_source = defaultdict(list)
    for p in pairs:
        en_by_source[p["en_source"]].append(p)
        es_by_source[p["es_source"]].append(p)

    for lang, by_source, out_dir in [("en", en_by_source, "bible_data/en"),
                                      ("es", es_by_source, "bible_data/es")]:
        os.makedirs(out_dir, exist_ok=True)
        for source, items in by_source.items():
            # Deduplicate by verse_id
            seen = set()
            unique = []
            for p in items:
                if p["verse_id"] not in seen:
                    seen.add(p["verse_id"])
                    unique.append(p)
            path = os.path.join(out_dir, f"{source}.jsonl")
            with open(path, "w", encoding="utf-8") as f:
                for p in sorted(unique, key=lambda x: x["verse_id"]):
                    f.write(json.dumps({
                        "verse_id": p["verse_id"],
                        "text": p[lang],
                        "source": source,
                    }, ensure_ascii=False) + "\n")
            print(f"  {path}: {len(unique)} verses")


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
    parser.add_argument("--db-dir", default="bible_data/scrollmapper/formats/sqlite",
                        help="Directory containing per-translation SQLite databases")
    parser.add_argument("--output", default="bible_data/aligned/verse_pairs.jsonl",
                        help="Output JSONL path")
    parser.add_argument("--multi-ref", action="store_true",
                        help="Create multi-reference expanded pairs")
    args = parser.parse_args()

    pairs = []

    if args.source in ("scrollmapper", "all"):
        pairs.extend(align_scrollmapper(args.db_dir))

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

    print("\nExporting individual translation files...")
    export_individual_translations(pairs)

    print("\nExporting training JSONL...")
    export_training_jsonl(pairs, args.output)


if __name__ == "__main__":
    main()
