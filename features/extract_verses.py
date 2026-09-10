#!/usr/bin/env python3
"""
extract_verses.py — Bible Verse Reference Extraction from Sermon Transcripts

Two-pass extraction:
  Pass 1: Regex for explicit citations ("Romans 8:28", "1 Corinthians 13:4")
  Pass 2: Regex for spoken-form references ("turn to Romans chapter eight",
           "the passage in John three sixteen", "verse 28")

Handles all 66 book name variants (1 John, First John, I John, 1Jn, etc.).
Tracks explicit book/chapter context; impossible, hymn and unresolved citations are omitted.

Input: CSV (dry_run_ab.py) or JSONL (diarize.py)
Output: JSON with per-speaker (if diarized) or overall verse references.

Usage:
    python extract_verses.py metrics/ab_metrics_20260208_183356.csv
    python extract_verses.py metrics/diarization/sermon.jsonl
    python extract_verses.py transcript.csv -o metrics/verse_references/session.json
"""

import argparse
import csv
import json
import os
import re
import sys
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

try:
    from .bible_reference_bounds import CHAPTER_VERSE_COUNTS
except ImportError:  # Legacy direct script entry point.
    from bible_reference_bounds import CHAPTER_VERSE_COUNTS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OUTPUT_DIR = "metrics/verse_references"

# Context window: characters before and after a verse reference to include
CONTEXT_CHARS = 80


# ---------------------------------------------------------------------------
# Bible Book Name Mappings (all 66 books, all common variants)
# ---------------------------------------------------------------------------

# Canonical name -> list of regex-safe aliases (longest first to avoid greedy issues)
# Each alias is case-insensitive in the regex
BOOK_ALIASES = OrderedDict(
    [
        # --- Old Testament ---
        ("Genesis", ["Genesis", "Gen"]),
        ("Exodus", ["Exodus", "Exod", "Exo"]),
        ("Leviticus", ["Leviticus", "Lev"]),
        ("Numbers", ["Numbers", "Num"]),
        ("Deuteronomy", ["Deuteronomy", "Deut", "Deu"]),
        ("Joshua", ["Joshua", "Josh", "Jos"]),
        ("Judges", ["Judges", "Judg", "Jdg"]),
        ("Ruth", ["Ruth", "Rut"]),
        ("1 Samuel", ["First Samuel", "1st Samuel", "1 Samuel", "I Samuel", "1Samuel", "1Sam", "1 Sam"]),
        ("2 Samuel", ["Second Samuel", "2nd Samuel", "2 Samuel", "II Samuel", "2Samuel", "2Sam", "2 Sam"]),
        ("1 Kings", ["First Kings", "1st Kings", "1 Kings", "I Kings", "1Kings", "1Kgs", "1 Kgs"]),
        ("2 Kings", ["Second Kings", "2nd Kings", "2 Kings", "II Kings", "2Kings", "2Kgs", "2 Kgs"]),
        (
            "1 Chronicles",
            [
                "First Chronicles",
                "1st Chronicles",
                "1 Chronicles",
                "I Chronicles",
                "1Chronicles",
                "1Chr",
                "1 Chr",
                "1 Chron",
            ],
        ),
        (
            "2 Chronicles",
            [
                "Second Chronicles",
                "2nd Chronicles",
                "2 Chronicles",
                "II Chronicles",
                "2Chronicles",
                "2Chr",
                "2 Chr",
                "2 Chron",
            ],
        ),
        ("Ezra", ["Ezra", "Ezr"]),
        ("Nehemiah", ["Nehemiah", "Neh"]),
        ("Esther", ["Esther", "Est", "Esth"]),
        ("Job", ["Job"]),
        ("Psalms", ["Psalms", "Psalm", "Psa", "Ps"]),
        ("Proverbs", ["Proverbs", "Prov", "Pro"]),
        ("Ecclesiastes", ["Ecclesiastes", "Eccles", "Eccl", "Ecc"]),
        ("Song of Solomon", ["Song of Solomon", "Song of Songs", "Songs of Solomon", "Song", "SOS", "SS"]),
        ("Isaiah", ["Isaiah", "Isa"]),
        ("Jeremiah", ["Jeremiah", "Jer"]),
        ("Lamentations", ["Lamentations", "Lam"]),
        ("Ezekiel", ["Ezekiel", "Ezek", "Eze"]),
        ("Daniel", ["Daniel", "Dan"]),
        ("Hosea", ["Hosea", "Hos"]),
        ("Joel", ["Joel"]),
        ("Amos", ["Amos"]),
        ("Obadiah", ["Obadiah", "Obad", "Oba"]),
        ("Jonah", ["Jonah", "Jon"]),
        ("Micah", ["Micah", "Mic"]),
        ("Nahum", ["Nahum", "Nah"]),
        ("Habakkuk", ["Habakkuk", "Hab"]),
        ("Zephaniah", ["Zephaniah", "Zeph", "Zep"]),
        ("Haggai", ["Haggai", "Hag"]),
        ("Zechariah", ["Zechariah", "Zech", "Zec"]),
        ("Malachi", ["Malachi", "Mal"]),
        # --- New Testament ---
        ("Matthew", ["Matthew", "Matt", "Mat"]),
        ("Mark", ["Mark", "Mrk"]),
        ("Luke", ["Luke", "Luk"]),
        ("John", ["John", "Joh", "Jn"]),
        ("Acts", ["Acts", "Act"]),
        ("Romans", ["Romans", "Rom"]),
        (
            "1 Corinthians",
            ["First Corinthians", "1st Corinthians", "1 Corinthians", "I Corinthians", "1Corinthians", "1Cor", "1 Cor"],
        ),
        (
            "2 Corinthians",
            [
                "Second Corinthians",
                "2nd Corinthians",
                "2 Corinthians",
                "II Corinthians",
                "2Corinthians",
                "2Cor",
                "2 Cor",
            ],
        ),
        ("Galatians", ["Galatians", "Gal"]),
        ("Ephesians", ["Ephesians", "Eph"]),
        ("Philippians", ["Philippians", "Phil", "Php"]),
        ("Colossians", ["Colossians", "Col"]),
        (
            "1 Thessalonians",
            [
                "First Thessalonians",
                "1st Thessalonians",
                "1 Thessalonians",
                "I Thessalonians",
                "1Thessalonians",
                "1Thess",
                "1 Thess",
                "1Th",
                "1 Th",
            ],
        ),
        (
            "2 Thessalonians",
            [
                "Second Thessalonians",
                "2nd Thessalonians",
                "2 Thessalonians",
                "II Thessalonians",
                "2Thessalonians",
                "2Thess",
                "2 Thess",
                "2Th",
                "2 Th",
            ],
        ),
        ("1 Timothy", ["First Timothy", "1st Timothy", "1 Timothy", "I Timothy", "1Timothy", "1Tim", "1 Tim"]),
        ("2 Timothy", ["Second Timothy", "2nd Timothy", "2 Timothy", "II Timothy", "2Timothy", "2Tim", "2 Tim"]),
        ("Titus", ["Titus", "Tit"]),
        ("Philemon", ["Philemon", "Phlm", "Phm"]),
        ("Hebrews", ["Hebrews", "Heb"]),
        ("James", ["James", "Jas"]),
        ("1 Peter", ["First Peter", "1st Peter", "1 Peter", "I Peter", "1Peter", "1Pet", "1 Pet"]),
        ("2 Peter", ["Second Peter", "2nd Peter", "2 Peter", "II Peter", "2Peter", "2Pet", "2 Pet"]),
        ("1 John", ["First John", "1st John", "1 John", "I John", "1John", "1Jn", "1 Jn"]),
        ("2 John", ["Second John", "2nd John", "2 John", "II John", "2John", "2Jn", "2 Jn"]),
        ("3 John", ["Third John", "3rd John", "3 John", "III John", "3John", "3Jn", "3 Jn"]),
        ("Jude", ["Jude"]),
        ("Revelation", ["Revelation", "Revelations", "Rev"]),
    ]
)

# Spoken number words -> digits (for "chapter eight" -> 8)
SPOKEN_NUMBERS = {
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
    "thirteen": "13",
    "fourteen": "14",
    "fifteen": "15",
    "sixteen": "16",
    "seventeen": "17",
    "eighteen": "18",
    "nineteen": "19",
    "twenty": "20",
    "twenty-one": "21",
    "twenty-two": "22",
    "twenty-three": "23",
    "twenty-four": "24",
    "twenty-five": "25",
    "twenty-six": "26",
    "twenty-seven": "27",
    "twenty-eight": "28",
    "twenty-nine": "29",
    "thirty": "30",
    "thirty-one": "31",
    "thirty-two": "32",
    "thirty-three": "33",
    "thirty-four": "34",
    "thirty-five": "35",
    "thirty-six": "36",
    "thirty-seven": "37",
    "thirty-eight": "38",
    "thirty-nine": "39",
    "forty": "40",
    "fifty": "50",
    "sixty": "60",
    "seventy": "70",
    "eighty": "80",
    "ninety": "90",
    "hundred": "100",
    "first": "1",
    "second": "2",
    "third": "3",
    "fourth": "4",
    "fifth": "5",
    "sixth": "6",
    "seventh": "7",
    "eighth": "8",
    "ninth": "9",
    "tenth": "10",
}


# Whole compound numbers in either common STT spelling. Atomic NUM below
# prevents backtracking from "twenty three" to chapter 20, verse 3.
for _tens, _value in (
    ("twenty", 20),
    ("thirty", 30),
    ("forty", 40),
    ("fifty", 50),
    ("sixty", 60),
    ("seventy", 70),
    ("eighty", 80),
    ("ninety", 90),
):
    for _unit in ("one", "two", "three", "four", "five", "six", "seven", "eight", "nine"):
        for _separator in (" ", "-"):
            SPOKEN_NUMBERS[_tens + _separator + _unit] = str(_value + int(SPOKEN_NUMBERS[_unit]))
for _word, _number in list(SPOKEN_NUMBERS.items()):
    if (
        _word not in ("first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth")
        and 0 < int(_number) < 100
    ):
        for _prefix in ("one hundred ", "one hundred and ", "hundred ", "hundred and "):
            SPOKEN_NUMBERS[_prefix + _word] = str(100 + int(_number))
SPOKEN_NUMBERS["one hundred"] = "100"


# ---------------------------------------------------------------------------
# Build Regex Patterns
# ---------------------------------------------------------------------------


def _build_book_pattern():
    """Build a regex alternation matching any Bible book name.

    Returns a compiled regex with a named group 'book'.
    Sorts aliases longest-first to ensure greedy matching works correctly.
    """
    all_aliases = []
    alias_to_canonical = {}
    for canonical, aliases in BOOK_ALIASES.items():
        for alias in aliases:
            all_aliases.append(alias)
            alias_to_canonical[alias.lower()] = canonical

    # Sort longest first
    all_aliases.sort(key=len, reverse=True)

    # Escape for regex (handles periods in abbreviations)
    escaped = [re.escape(a) for a in all_aliases]
    pattern = "|".join(escaped)
    return pattern, alias_to_canonical


BOOK_PATTERN, ALIAS_TO_CANONICAL = _build_book_pattern()
BOOK_PATTERN = rf"\b(?:{BOOK_PATTERN})\b"


def _number_word_pattern():
    """Build regex alternation for spoken number words."""
    words = sorted(SPOKEN_NUMBERS.keys(), key=len, reverse=True)
    return "|".join(re.escape(w) for w in words)


NUMBER_WORD_PATTERN = _number_word_pattern()

# A number is either digits or a spoken word
NUM = rf"(?>\d+|{NUMBER_WORD_PATTERN})\b"


def normalize_number(s):
    """Convert a string (digit or spoken word) to an integer string."""
    s = s.strip().lower()
    if s.isdigit():
        return s
    return SPOKEN_NUMBERS.get(s, s)


def resolve_book(text):
    """Resolve a book name/alias to its canonical form."""
    return ALIAS_TO_CANONICAL.get(text.strip().lower(), text.strip())


# ---------------------------------------------------------------------------
# Pass 1: Explicit Citation Patterns
# ---------------------------------------------------------------------------

# Fillers are allowed around explicit reference markers, not as guessed digits.
FILL = r"(?:\s+(?:uh|um))*\s+"
VERSE_MARK = rf"\s*[,;]?\s*(?:and\s+)?verses?{FILL}"
RANGE = rf"(?:\s*(?:to|through|[-\u2013\u2014])\s*(?P<verse_end>{NUM}))?"
EXPLICIT_PATTERN = re.compile(rf"(?P<book>{BOOK_PATTERN})\s+(?P<chapter>\d+)\s*:\s*(?P<verse_start>\d+){RANGE}", re.I)
CHAPTER_VERSE_SPOKEN = re.compile(
    rf"(?P<book>{BOOK_PATTERN})\s+chapter{FILL}(?P<chapter>{NUM})"
    rf"(?:{VERSE_MARK}(?P<verse_start>{NUM}){RANGE})?",
    re.I,
)
CHAPTER_ONLY_PATTERN = re.compile(
    rf"(?P<book>{BOOK_PATTERN})\s+(?P<chapter>{NUM})"
    rf"(?:{VERSE_MARK}(?P<verse_start>{NUM}){RANGE})?",
    re.I,
)
SPOKEN_BARE_PATTERN = re.compile(
    rf"(?P<book>{BOOK_PATTERN})\s+(?P<chapter>{NUM})\s*,?\s+(?P<verse_start>{NUM}){RANGE}", re.I
)
BARE_CHAPTER_PATTERN = re.compile(
    rf"\bchapter{FILL}(?P<chapter>{NUM})(?:{VERSE_MARK}(?P<verse_start>{NUM}){RANGE})?", re.I
)
BARE_VERSE_PATTERN = re.compile(rf"\bverses?{FILL}(?P<verse_start>{NUM}){RANGE}", re.I)
BOOK_MENTION_PATTERN = re.compile(rf"(?P<book>{BOOK_PATTERN})(?=\s*(?:[,.;:]|$))", re.I)
HYMN_PATTERN = re.compile(r"\b(?:hymns?|stanzas?)\b", re.I)
AMBIGUOUS_BOOK_PATTERN = re.compile(r"\b(?:Peter|Samuel|Kings|Chronicles|Corinthians|Thessalonians|Timothy)\b", re.I)


# ---------------------------------------------------------------------------
# Extraction Engine
# ---------------------------------------------------------------------------


class VerseExtractor:
    """Stateful verse reference extractor with context tracking.

    Maintains the most recently mentioned book and chapter so that
    bare references like "verse 5" can be resolved.
    """

    def __init__(self):
        self.current_book = None
        self.current_chapter = None
        self.references = []  # list of extracted reference dicts
        self._seen = set()  # dedup key: (book, chapter, verse_start, verse_end)

    def _add_reference(self, book, chapter, verse_start, verse_end, timestamp, speaker, context, pattern_name):
        """Add a reference, updating context and deduplicating."""
        # Normalize
        canonical_book = resolve_book(book)
        chapter_str = normalize_number(str(chapter)) if chapter else None
        vs = normalize_number(str(verse_start)) if verse_start else None
        ve = normalize_number(str(verse_end)) if verse_end else None

        # Validate before committing chapter context. A changed book must never
        # inherit a chapter from the previous book, even if this match is invalid.
        bounds = CHAPTER_VERSE_COUNTS.get(canonical_book)
        if canonical_book != self.current_book:
            self.current_chapter = None
        self.current_book = canonical_book if bounds else None
        if chapter_str is None:
            return
        if not bounds or not chapter_str.isdigit() or not 1 <= int(chapter_str) <= len(bounds):
            self.current_chapter = None
            return
        limit = bounds[int(chapter_str) - 1]
        if (vs is not None and (not vs.isdigit() or not 1 <= int(vs) <= limit)) or (
            ve is not None and (not ve.isdigit() or not vs or not int(vs) <= int(ve) <= limit)
        ):
            self.current_chapter = None
            return
        self.current_chapter = chapter_str

        # Build reference string
        ref = canonical_book
        if chapter_str:
            ref += f" {chapter_str}"
            if vs:
                ref += f":{vs}"
                if ve and ve != vs:
                    ref += f"-{ve}"

        # Dedup within same speaker+timestamp window
        dedup_key = (canonical_book, chapter_str, vs, ve, speaker or "")
        if dedup_key in self._seen:
            return
        self._seen.add(dedup_key)

        entry = {
            "reference": ref,
            "book": canonical_book,
            "chapter": chapter_str,
            "verse_start": vs,
            "verse_end": ve,
            "timestamp": timestamp,
            "speaker": speaker,
            "context": context.strip()[: CONTEXT_CHARS * 2] if context else "",
            "pattern": pattern_name,
        }
        self.references.append(entry)

    def _overlaps(self, matched_spans, start, end):
        """Check if a span overlaps with any already-matched span."""
        for ms, me in matched_spans:
            if start < me and end > ms:
                return True
        return False

    def extract_from_text(self, text, timestamp="", speaker=None):
        """Run all extraction passes on a text segment.

        Pattern priority selects non-overlapping matches, then references and
        book/context changes are applied in their original text order.

        Args:
            text: English transcript text.
            timestamp: ISO timestamp or HH:MM:SS string.
            speaker: Speaker label (or None).
        """
        if not text:
            return

        # Priority resolves overlapping matches; context follows spoken order.
        matched_spans = []
        events = []
        patterns = (
            ("explicit", EXPLICIT_PATTERN),
            ("chapter_verse_spoken", CHAPTER_VERSE_SPOKEN),
            ("spoken_bare", SPOKEN_BARE_PATTERN),
            ("chapter_only", CHAPTER_ONLY_PATTERN),
            ("bare_chapter", BARE_CHAPTER_PATTERN),
            ("bare_verse", BARE_VERSE_PATTERN),
        )
        hymn = bool(HYMN_PATTERN.search(text))
        for name, pattern in patterns:
            for match in pattern.finditer(text):
                if self._overlaps(matched_spans, match.start(), match.end()):
                    continue
                matched_spans.append(match.span())
                events.append((match.start(), name, match))
        for name, pattern in (
            ("book_mention", BOOK_MENTION_PATTERN),
            ("ambiguous_book", AMBIGUOUS_BOOK_PATTERN),
            ("hymn", HYMN_PATTERN),
        ):
            for match in pattern.finditer(text):
                if self._overlaps(matched_spans, match.start(), match.end()):
                    continue
                matched_spans.append(match.span())
                events.append((match.start(), name, match))
        for _, name, match in sorted(events, key=lambda event: event[0]):
            if name in ("hymn", "ambiguous_book"):
                self.current_book = self.current_chapter = None
                continue
            if name == "book_mention":
                book = resolve_book(match.group("book"))
                if book != self.current_book:
                    self.current_chapter = None
                self.current_book = book
                continue
            if hymn and name in ("bare_chapter", "bare_verse"):
                continue
            groups = match.groupdict()
            book = groups.get("book") or self.current_book
            chapter = groups.get("chapter") or self.current_chapter
            if not book or not chapter:
                continue
            # A tens word at the end of an unpunctuated audio chunk can continue
            # ("verse thirty" / "uh thirty six"). Leave it unresolved.
            number = groups.get("verse_end") or groups.get("verse_start") or groups.get("chapter") or ""
            if match.end() == len(text.rstrip()) and number.lower() in {
                "twenty",
                "thirty",
                "forty",
                "fifty",
                "sixty",
                "seventy",
                "eighty",
                "ninety",
                "hundred",
                "one hundred",
            }:
                self.current_chapter = None
                continue
            # Do not publish the prefix of an unfinished verse/range/list.
            tail = text[match.end() :]
            unfinished = re.fullmatch(r"\s*(?:to|through|and|[-\u2013\u2014])(?:\s+(?:uh|um))*\s*[.,]?", tail, re.I)
            verse_marker = not groups.get("verse_start") and re.match(r"\s*(?:and\s+)?verses?\b", tail, re.I)
            verse_list = groups.get("verse_start") and re.match(rf"\s+and\s+{NUM}", tail, re.I)
            if unfinished or verse_marker or verse_list:
                self.current_chapter = None
                continue
            self._add_reference(
                book,
                chapter,
                groups.get("verse_start"),
                groups.get("verse_end"),
                timestamp,
                speaker,
                _extract_context(text, match.start(), match.end()),
                name,
            )


def _extract_context(text, start, end):
    """Extract surrounding context around a match position."""
    ctx_start = max(0, start - CONTEXT_CHARS)
    ctx_end = min(len(text), end + CONTEXT_CHARS)
    prefix = "..." if ctx_start > 0 else ""
    suffix = "..." if ctx_end < len(text) else ""
    return prefix + text[ctx_start:ctx_end] + suffix


# ---------------------------------------------------------------------------
# Transcript Loading (shared with summarize_sermon.py)
# ---------------------------------------------------------------------------


def load_csv_transcript(csv_path):
    """Load transcript from dry_run_ab.py CSV."""
    entries = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row.get("english", "").strip()
            if not text:
                continue
            entries.append(
                {
                    "timestamp": row.get("timestamp", ""),
                    "text": text,
                    "speaker": None,
                }
            )
    print(f"  Loaded {len(entries)} segments from CSV")
    return entries


def load_jsonl_transcript(jsonl_path):
    """Load transcript from diarize.py JSONL."""
    entries = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if "_metadata" in record:
                continue
            text = record.get("text", "")
            if not text or text.startswith("[transcription error"):
                continue
            start = record.get("start", 0)
            h = int(start // 3600)
            m = int((start % 3600) // 60)
            s = int(start % 60)
            ts = f"{h:02d}:{m:02d}:{s:02d}"
            entries.append(
                {
                    "timestamp": ts,
                    "text": text,
                    "speaker": record.get("speaker"),
                }
            )
    speakers = set(e["speaker"] for e in entries if e["speaker"])
    print(f"  Loaded {len(entries)} segments from JSONL")
    if speakers:
        print(f"  Speakers: {', '.join(sorted(speakers))}")
    return entries


def load_transcript(input_path):
    """Auto-detect file type and load transcript."""
    p = Path(input_path)
    if p.suffix.lower() == ".csv":
        return load_csv_transcript(input_path)
    elif p.suffix.lower() in (".jsonl", ".json"):
        return load_jsonl_transcript(input_path)
    else:
        print(f"ERROR: Unsupported file type: {p.suffix}", file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def format_output(extractor, input_path, entries):
    """Format extraction results as a structured dict for JSON output."""
    refs = extractor.references

    # Group by speaker if diarized
    speakers = set(e["speaker"] for e in entries if e.get("speaker"))
    has_diarization = len(speakers) >= 2

    if has_diarization:
        by_speaker = {}
        for ref in refs:
            spk = ref.get("speaker") or "Unknown"
            if spk not in by_speaker:
                by_speaker[spk] = []
            by_speaker[spk].append(
                {
                    "reference": ref["reference"],
                    "book": ref["book"],
                    "chapter": ref["chapter"],
                    "verse_start": ref["verse_start"],
                    "verse_end": ref["verse_end"],
                    "timestamp": ref["timestamp"],
                    "context": ref["context"],
                    "pattern": ref["pattern"],
                }
            )

        speaker_results = []
        for spk in sorted(by_speaker.keys()):
            speaker_results.append(
                {
                    "speaker": spk,
                    "verses": by_speaker[spk],
                    "count": len(by_speaker[spk]),
                }
            )
    else:
        speaker_results = None

    # Unique references (deduplicated by book+chapter+verse)
    unique_refs = []
    seen = set()
    for ref in refs:
        key = ref["reference"]
        if key not in seen:
            seen.add(key)
            unique_refs.append(ref["reference"])

    # Book frequency
    book_counts = {}
    for ref in refs:
        b = ref["book"]
        book_counts[b] = book_counts.get(b, 0) + 1

    return {
        "input": input_path,
        "timestamp": datetime.now().isoformat(),
        "total_references": len(refs),
        "unique_references": unique_refs,
        "books_referenced": dict(sorted(book_counts.items(), key=lambda x: -x[1])),
        "by_speaker": speaker_results,
        "all_references": [
            {
                "reference": r["reference"],
                "timestamp": r["timestamp"],
                "speaker": r["speaker"],
                "context": r["context"],
                "pattern": r["pattern"],
            }
            for r in refs
        ],
    }


def write_output(data, output_path):
    """Write verse references to JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"\n  Output: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Extract Bible verse references from sermon transcripts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python extract_verses.py metrics/ab_metrics_20260208_183356.csv
    python extract_verses.py metrics/diarization/sermon.jsonl
    python extract_verses.py transcript.csv -o metrics/verse_references/session.json
        """,
    )
    parser.add_argument("input", nargs="+", help="CSV or JSONL transcript file(s)")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output JSON path (default: metrics/verse_references/<input_name>.json)",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Print each reference as it's found")
    args = parser.parse_args()

    print(f"{'=' * 60}")
    print("  Bible Verse Reference Extraction")
    print(f"  Input(s): {len(args.input)} file(s)")
    print(f"{'=' * 60}\n")

    # Load and merge all input transcripts
    all_entries = []
    for input_path in args.input:
        print(f"Loading: {input_path}")
        entries = load_transcript(input_path)
        all_entries.extend(entries)

    if not all_entries:
        print("ERROR: No transcript content found.", file=sys.stderr)
        sys.exit(1)

    total_words = sum(len(e["text"].split()) for e in all_entries)
    print(f"\n  Total: {len(all_entries)} segments, ~{total_words} words")

    # Extract verse references
    extractor = VerseExtractor()

    print("\n  Extracting verse references...\n")
    for entry in all_entries:
        extractor.extract_from_text(
            text=entry["text"],
            timestamp=entry.get("timestamp", ""),
            speaker=entry.get("speaker"),
        )

        if args.verbose and extractor.references:
            latest = extractor.references[-1]
            print(f"    Found: {latest['reference']} [{latest['pattern']}]")

    # Format results
    data = format_output(extractor, args.input, all_entries)

    # Determine output path
    output_path = args.output
    if output_path is None:
        input_name = Path(args.input[0]).stem
        output_path = os.path.join(OUTPUT_DIR, f"{input_name}_verses.json")

    # Write output
    write_output(data, output_path)

    # Print summary
    print(f"\n{'=' * 60}")
    print("  VERSE REFERENCES SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Total references found: {data['total_references']}")
    print(f"  Unique references: {len(data['unique_references'])}")

    if data["unique_references"]:
        print("\n  Unique verses:")
        for ref in data["unique_references"]:
            print(f"    - {ref}")

    if data["books_referenced"]:
        print("\n  Books referenced (by frequency):")
        for book, count in data["books_referenced"].items():
            print(f"    {book}: {count}x")

    if data["by_speaker"]:
        print("\n  By speaker:")
        for spk_data in data["by_speaker"]:
            print(f"    {spk_data['speaker']}: {spk_data['count']} references")
            for v in spk_data["verses"][:5]:
                print(f"      - {v['reference']}")
            if spk_data["count"] > 5:
                print(f"      ... and {spk_data['count'] - 5} more")

    print(f"\n{'=' * 60}")


if __name__ == "__main__":
    main()
