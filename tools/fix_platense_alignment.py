#!/usr/bin/env python3
"""
fix_platense_alignment.py — Re-key bible_data/es/platense.jsonl to canonical verse_ids.

PROBLEM (discovered 2026-04-29 during Phase A staging of Gemma 4 tuning):
the original alignment pipeline (`training/prepare_bible_corpus.py`) joined
per-source files on a flat `verse_id` column derived from each source's row
order. SpaPlatense (Catholic canon, 78 books, 37,255 verses) interleaves 12
deuterocanonical books (Tobit, Judith, Wisdom, Sirach, Baruch, I/II Maccabees,
etc.) with the Protestant 66-book ordering, so its row-order verse_id N maps
to a *different* canonical reference than KJV/RVR1909/WEB row-order verse_id N.

Result: ~120,777 of the 241,591 pairs in `bible_data/aligned/verse_pairs_train.jsonl`
were silently misaligned past Genesis. Spot-checked example: vid=13941 in EN
sources is Psalm 1:1, but in raw Platense it's Job 14:15 (the 813-vid drift is
the cumulative size of the apocryphal insertions before Psalms in the Catholic
ordering: Tobit 14ch + Judith 16ch + half of Esther + Wisdom 19ch + Sirach 51ch
+ part of Baruch ≈ 813 verses).

FIX: re-extract Platense from the original `scrollmapper` SQLite database (still
on disk at /mnt/d/Data/stt-data/bible_data/scrollmapper/formats/sqlite/) and
JOIN ON (book_name, chapter, verse) against SpaRV.db. The 66 Protestant-canon
book names are identical between the two databases — a clean structural join.
Apocryphal verses (no canonical match) are dropped, not misaligned.

This catches the bulk of the bug, but Psalms is special: Catholic Bibles use
Septuagint (LXX) Psalm numbering while Protestant Bibles use Masoretic (MT),
and they differ by ±1 in chapter from Psalm 9 onward (LXX combines MT 9+10,
splits 114+115 differently, etc.). A naive (book, chapter, verse) join across
the two traditions misaligns ~1700 Psalms verses. **Therefore the structural
join is followed by a Spanish-vs-Spanish content validator**: each realigned
Platense verse is scored against the RVR1909 verse at the same canonical
verse_id (RVR1909 is Protestant/Masoretic — the canonical reference), using
content-word Jaccard similarity. Verses below a threshold are dropped — this
catches the Psalms numbering shift AND any other latent misalignments.

Usage:
    python tools/fix_platense_alignment.py
    python tools/fix_platense_alignment.py --scrollmapper-dir <dir> --output ...
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sqlite3
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("fix_platense")

DEFAULT_SCROLLMAPPER = Path("/mnt/d/Data/stt-data/bible_data/scrollmapper/formats/sqlite")

# Spanish stopwords for content-word Jaccard validation between Platense and RVR1909.
# Same set as could be used for general Spanish text discrimination.
ES_STOPWORDS = {
    "el",
    "la",
    "los",
    "las",
    "de",
    "del",
    "y",
    "en",
    "a",
    "que",
    "un",
    "una",
    "es",
    "se",
    "su",
    "por",
    "con",
    "no",
    "o",
    "al",
    "le",
    "lo",
    "sus",
    "para",
    "como",
    "han",
    "ha",
    "sea",
    "si",
    "mas",
    "pero",
    "este",
    "esta",
    "estos",
    "estas",
    "ese",
    "esa",
    "eso",
    "fue",
    "fué",
    "ser",
    "está",
    "están",
    "yo",
    "tu",
    "tú",
    "ella",
    "nos",
    "vos",
    "vosotros",
    "ustedes",
    "mi",
    "mis",
    "tus",
    "ni",
    "muy",
    "más",
    "ya",
    "todo",
    "toda",
    "todos",
    "todas",
    "ah",
    "oh",
    "porque",
    "cuando",
    "donde",
    "entonces",
    "sobre",
    "hasta",
    "desde",
}
TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def content_tokens(text: str) -> set[str]:
    """Lowercase, split, drop stopwords + length<4. Returns content-word set."""
    return {w for w in TOKEN_RE.findall(text.lower()) if len(w) >= 4 and w not in ES_STOPWORDS}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def load_books(con: sqlite3.Connection, table_prefix: str) -> dict[int, str]:
    """Returns {book_id: book_name}."""
    rows = con.execute(f"SELECT id, name FROM {table_prefix}_books").fetchall()
    return dict(rows)


def load_verses(con: sqlite3.Connection, table_prefix: str) -> list[dict]:
    """Returns [{id, book_id, chapter, verse, text}, ...]."""
    cols = "id, book_id, chapter, verse, text"
    rows = con.execute(
        f"SELECT {cols} FROM {table_prefix}_verses WHERE text IS NOT NULL AND LENGTH(text) > 5"
    ).fetchall()
    return [{"id": r[0], "book_id": r[1], "chapter": r[2], "verse": r[3], "text": r[4].strip()} for r in rows]


def realign(scrollmapper_dir: Path, output_path: Path, report_path: Path, validate_threshold: float = 0.10) -> dict:
    plat_db = scrollmapper_dir / "SpaPlatense.db"
    rvr_db = scrollmapper_dir / "SpaRV.db"
    if not plat_db.exists():
        raise SystemExit(f"SpaPlatense.db not found at {plat_db}")
    if not rvr_db.exists():
        raise SystemExit(f"SpaRV.db not found at {rvr_db}")

    log.info("loading SpaRV (canonical reference) from %s", rvr_db)
    rvr_con = sqlite3.connect(str(rvr_db))
    rvr_books = load_books(rvr_con, "SpaRV")
    rvr_verses = load_verses(rvr_con, "SpaRV")
    rvr_con.close()
    log.info("RVR1909: %d books, %d verses", len(rvr_books), len(rvr_verses))

    # Build canonical lookup keyed by (book_name_lower, chapter, verse) -> canonical_id (== rvr verse_id).
    canonical_id: dict[tuple[str, int, int], int] = {}
    for v in rvr_verses:
        book_name = rvr_books[v["book_id"]].lower()
        canonical_id[(book_name, v["chapter"], v["verse"])] = v["id"]
    log.info("indexed %d canonical (book, chapter, verse) keys", len(canonical_id))

    log.info("loading SpaPlatense from %s", plat_db)
    plat_con = sqlite3.connect(str(plat_db))
    plat_books = load_books(plat_con, "SpaPlatense")
    plat_verses = load_verses(plat_con, "SpaPlatense")
    plat_con.close()
    log.info("Platense: %d books, %d verses", len(plat_books), len(plat_verses))

    rvr_book_names = {n.lower() for n in rvr_books.values()}
    apocryphal_books = [n for bid, n in plat_books.items() if n.lower() not in rvr_book_names]
    log.info("Platense has %d apocryphal books absent from RVR1909: %s", len(apocryphal_books), apocryphal_books)

    out_rows: list[dict] = []
    apocryphal_dropped = 0
    no_match_dropped = 0  # canonical book/chapter/verse exists in name but not in RVR
    by_orig_book: dict[str, int] = {}  # diagnostic: per-book count of realigned

    for v in plat_verses:
        book_name_raw = plat_books[v["book_id"]]
        book_name = book_name_raw.lower()
        if book_name not in rvr_book_names:
            apocryphal_dropped += 1
            continue
        key = (book_name, v["chapter"], v["verse"])
        if key not in canonical_id:
            no_match_dropped += 1
            continue
        canon_vid = canonical_id[key]
        out_rows.append(
            {
                "verse_id": canon_vid,
                "text": v["text"],
                "source": "platense",
                "_orig_id": v["id"],
                "_book_name": book_name_raw,
                "_chapter": v["chapter"],
                "_verse": v["verse"],
            }
        )
        by_orig_book[book_name_raw] = by_orig_book.get(book_name_raw, 0) + 1

    # Sanity: every output canonical_vid should be unique (book/chapter/verse keys are unique).
    canonical_vids = [r["verse_id"] for r in out_rows]
    assert len(canonical_vids) == len(set(canonical_vids)), (
        "duplicate canonical vids — book/chapter/verse should be unique"
    )

    # SECOND PASS: content validator. Compare each realigned Platense verse to
    # RVR1909 at the same canonical vid. RVR1909 is Protestant/Masoretic =
    # canonical truth. Low Jaccard => the (book, chapter, verse) join was misled
    # by tradition-specific numbering (e.g. Septuagint vs Masoretic Psalms shift).
    # Catches all latent misalignments structural join can't.
    log.info("validating realigned verses against RVR1909 by content (threshold=%.2f)", validate_threshold)
    rvr_vid_to_text = {v["id"]: v["text"] for v in rvr_verses}
    validated_rows: list[dict] = []
    validation_dropped: dict[str, int] = {}  # {book_name: count_dropped}
    for r in out_rows:
        rvr_text = rvr_vid_to_text.get(r["verse_id"])
        if rvr_text is None:
            # No RVR verse to compare against — keep, since structural join was clean.
            validated_rows.append(r)
            continue
        score = jaccard(content_tokens(r["text"]), content_tokens(rvr_text))
        r["_content_validate_score"] = round(score, 3)
        if score >= validate_threshold:
            validated_rows.append(r)
        else:
            validation_dropped[r["_book_name"]] = validation_dropped.get(r["_book_name"], 0) + 1

    n_validation_dropped = sum(validation_dropped.values())
    log.info(
        "content-validator dropped %d verses (per book: %s)",
        n_validation_dropped,
        dict(sorted(validation_dropped.items(), key=lambda x: -x[1])),
    )
    out_rows = validated_rows

    log.info("writing %d realigned + validated verses -> %s", len(out_rows), output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Compare against the OLD platense.jsonl to quantify the impact.
    old_path = Path("bible_data/es/platense.jsonl")
    old_count = 0
    if old_path.exists():
        with old_path.open(encoding="utf-8") as f:
            old_count = sum(1 for _ in f)

    report = {
        "fix_type": "structural_join_on_book_chapter_verse_plus_content_validator",
        "source": str(plat_db),
        "reference": str(rvr_db),
        "output": str(output_path),
        "platense_total_verses_in_db": len(plat_verses),
        "platense_books_total": len(plat_books),
        "rvr_books_total": len(rvr_books),
        "apocryphal_books_dropped": apocryphal_books,
        "apocryphal_verses_dropped": apocryphal_dropped,
        "no_match_dropped": no_match_dropped,
        "validation_threshold": validate_threshold,
        "validation_dropped_total": n_validation_dropped,
        "validation_dropped_per_book": dict(sorted(validation_dropped.items(), key=lambda x: -x[1])),
        "realigned_output_verses": len(out_rows),
        "old_platense_jsonl_verses": old_count,
        "old_jsonl_misaligned_estimate": old_count - len(out_rows),
        "per_book_kept_count": dict(sorted(by_orig_book.items())),
    }
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    log.info("wrote report -> %s", report_path)

    log.info(
        "summary: extracted %d Protestant-canonical Platense verses (dropped %d apocryphal, %d no-match)",
        len(out_rows),
        apocryphal_dropped,
        no_match_dropped,
    )
    return report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scrollmapper-dir", type=Path, default=DEFAULT_SCROLLMAPPER)
    p.add_argument("--output", type=Path, default=Path("bible_data/es/platense_realigned.jsonl"))
    p.add_argument("--report", type=Path, default=Path("bible_data/es/platense_realignment_report.json"))
    p.add_argument(
        "--validate-threshold",
        type=float,
        default=0.10,
        help="min Spanish-vs-Spanish Jaccard between Platense and RVR1909 at same canonical vid; "
        "verses below this are dropped (catches Septuagint vs Masoretic Psalms shift, etc.)",
    )
    args = p.parse_args(argv)
    realign(args.scrollmapper_dir, args.output, args.report, args.validate_threshold)
    return 0


if __name__ == "__main__":
    sys.exit(main())
