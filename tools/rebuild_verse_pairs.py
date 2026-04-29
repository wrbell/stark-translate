#!/usr/bin/env python3
"""
rebuild_verse_pairs.py — Rebuild the EN<->ES verse-pair corpus using realigned Platense.

Reads from the original `scrollmapper` SQLite databases (English: KJV, ASV, BBE,
WEB/OEB, YLT; Spanish: SpaRV [RVR1909] + the realigned Platense JSONL produced
by `tools/fix_platense_alignment.py`). JOINs every (en_translation, es_translation)
pair by canonical verse_id (which now matches across both Spanish sources, fixing
the bug). Writes a new file alongside the original — does NOT overwrite the
historical `bible_data/aligned/verse_pairs_train.jsonl`.

Output schema matches the original verse_pairs_train.jsonl:
  {"en": <text>, "es": <text>, "verse_id": <canonical>, "en_source": <name>, "es_source": <name>}

Usage:
    python tools/rebuild_verse_pairs.py
    python tools/rebuild_verse_pairs.py --output bible_data/aligned/verse_pairs_train_v2.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("rebuild_pairs")

DEFAULT_SCROLLMAPPER = Path("/mnt/d/Data/stt-data/bible_data/scrollmapper/formats/sqlite")

# Source mappings: scrollmapper db basename -> (short_name, side).
EN_SOURCES = {
    "KJV": "kjv",
    "ASV": "asv",
    "BBE": "bbe",
    "OEB": "web",  # scrollmapper labels Open English Bible as OEB; project calls it web
    "YLT": "ylt",
}
# Note: SpaPlatense is loaded from the realigned JSONL, not the SQLite directly.
ES_SQLITE_SOURCES = {
    "SpaRV": "rvr1909",
}


def load_sqlite_verses(db_path: Path, table_prefix: str) -> dict[int, str]:
    """Return {verse_id (canonical): text}."""
    if not db_path.exists():
        log.warning("missing %s — skipping", db_path)
        return {}
    con = sqlite3.connect(str(db_path))
    rows = con.execute(
        f"SELECT id, text FROM {table_prefix}_verses WHERE text IS NOT NULL AND LENGTH(text) > 5"
    ).fetchall()
    con.close()
    return {r[0]: r[1].strip() for r in rows}


def load_realigned_platense(path: Path) -> dict[int, str]:
    """Load the realigned platense JSONL: {canonical_verse_id: text}."""
    out: dict[int, str] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            out[row["verse_id"]] = row["text"]
    return out


def build(scrollmapper_dir: Path, platense_jsonl: Path, output: Path) -> dict:
    log.info("loading EN sources from %s", scrollmapper_dir)
    en_data: dict[str, dict[int, str]] = {}
    for db_name, short in EN_SOURCES.items():
        verses = load_sqlite_verses(scrollmapper_dir / f"{db_name}.db", db_name)
        if verses:
            en_data[short] = verses
            log.info("  %s: %d verses", short, len(verses))

    log.info("loading ES sources")
    es_data: dict[str, dict[int, str]] = {}
    for db_name, short in ES_SQLITE_SOURCES.items():
        verses = load_sqlite_verses(scrollmapper_dir / f"{db_name}.db", db_name)
        if verses:
            es_data[short] = verses
            log.info("  %s: %d verses", short, len(verses))

    log.info("loading realigned Platense from %s", platense_jsonl)
    platense_verses = load_realigned_platense(platense_jsonl)
    es_data["platense"] = platense_verses
    log.info("  platense (realigned): %d verses", len(platense_verses))

    if not en_data or not es_data:
        raise SystemExit("no source data loaded")

    log.info("joining all (en, es) pair combinations on canonical verse_id")
    pairs_per_combo: dict[tuple[str, str], int] = {}
    output.parent.mkdir(parents=True, exist_ok=True)
    n_total = 0
    with output.open("w", encoding="utf-8") as f:
        for en_short, en_verses in en_data.items():
            for es_short, es_verses in es_data.items():
                count = 0
                for vid in sorted(set(en_verses.keys()) & set(es_verses.keys())):
                    record = {
                        "en": en_verses[vid],
                        "es": es_verses[vid],
                        "verse_id": vid,
                        "en_source": en_short,
                        "es_source": es_short,
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    count += 1
                    n_total += 1
                pairs_per_combo[(en_short, es_short)] = count
                log.info("  %s <-> %s: %d pairs", en_short, es_short, count)

    log.info("wrote %d total pairs -> %s", n_total, output)

    # Diff against the OLD file for the report.
    old_path = Path("bible_data/aligned/verse_pairs_train.jsonl")
    old_count = 0
    old_per_combo: dict[tuple[str, str], int] = {}
    if old_path.exists():
        with old_path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                    key = (row.get("en_source"), row.get("es_source"))
                    old_per_combo[key] = old_per_combo.get(key, 0) + 1
                    old_count += 1
                except json.JSONDecodeError:
                    continue

    delta_per_combo = {}
    for k in set(list(pairs_per_combo.keys()) + list(old_per_combo.keys())):
        new_n = pairs_per_combo.get(k, 0)
        old_n = old_per_combo.get(k, 0)
        delta_per_combo[f"{k[0]}->{k[1]}"] = {"new": new_n, "old": old_n, "delta": new_n - old_n}

    report = {
        "output": str(output),
        "total_pairs_new": n_total,
        "total_pairs_old": old_count,
        "delta_total": n_total - old_count,
        "per_combo": delta_per_combo,
        "platense_source": str(platense_jsonl),
        "scrollmapper_dir": str(scrollmapper_dir),
    }
    report_path = output.with_suffix(".rebuild_report.json")
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    log.info("wrote rebuild report -> %s", report_path)
    return report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scrollmapper-dir", type=Path, default=DEFAULT_SCROLLMAPPER)
    p.add_argument("--platense-realigned", type=Path, default=Path("bible_data/es/platense_realigned.jsonl"))
    p.add_argument("--output", type=Path, default=Path("bible_data/aligned/verse_pairs_train_v2.jsonl"))
    args = p.parse_args(argv)
    if not args.platense_realigned.exists():
        p.error(f"platense_realigned not found: {args.platense_realigned}. Run tools/fix_platense_alignment.py first.")
    build(args.scrollmapper_dir, args.platense_realigned, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
