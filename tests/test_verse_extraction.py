"""Tests for features/extract_verses.py — Bible verse reference extraction."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features.extract_verses import (
    BOOK_ALIASES,
    VerseExtractor,
    _extract_context,
    normalize_number,
    resolve_book,
)


class TestNormalizeNumber:
    def test_digit_passthrough(self):
        assert normalize_number("42") == "42"

    def test_spoken_numbers(self):
        assert normalize_number("eight") == "8"
        assert normalize_number("sixteen") == "16"
        assert normalize_number("twenty-one") == "21"

    def test_ordinals(self):
        assert normalize_number("first") == "1"
        assert normalize_number("third") == "3"
        assert normalize_number("tenth") == "10"

    def test_case_insensitive(self):
        assert normalize_number("Eight") == "8"
        assert normalize_number("SIXTEEN") == "16"

    def test_unknown_returns_input(self):
        assert normalize_number("bazillion") == "bazillion"


class TestResolveBook:
    def test_canonical_names(self):
        assert resolve_book("Genesis") == "Genesis"
        assert resolve_book("Romans") == "Romans"
        assert resolve_book("Revelation") == "Revelation"

    def test_abbreviations(self):
        assert resolve_book("Gen") == "Genesis"
        assert resolve_book("Rom") == "Romans"
        assert resolve_book("Rev") == "Revelation"

    def test_numbered_books(self):
        assert resolve_book("1 John") == "1 John"
        assert resolve_book("1Jn") == "1 John"
        assert resolve_book("First John") == "1 John"
        assert resolve_book("I John") == "1 John"

    def test_case_insensitive(self):
        assert resolve_book("genesis") == "Genesis"
        assert resolve_book("ROMANS") == "Romans"

    def test_all_66_books_covered(self):
        assert len(BOOK_ALIASES) == 66


class TestExtractContext:
    def test_short_text(self):
        ctx = _extract_context("Romans 8:28", 0, 11)
        assert ctx == "Romans 8:28"

    def test_with_surrounding(self):
        text = "x" * 100 + "Romans 8:28" + "y" * 100
        ctx = _extract_context(text, 100, 111)
        assert "Romans 8:28" in ctx
        assert ctx.startswith("...")
        assert ctx.endswith("...")


class TestVerseExtractor:
    def test_explicit_pattern(self):
        ext = VerseExtractor()
        ext.extract_from_text("Please read Romans 8:28 today.")

        assert len(ext.references) == 1
        ref = ext.references[0]
        assert ref["reference"] == "Romans 8:28"
        assert ref["book"] == "Romans"
        assert ref["chapter"] == "8"
        assert ref["verse_start"] == "28"
        assert ref["pattern"] == "explicit"

    def test_explicit_range(self):
        ext = VerseExtractor()
        ext.extract_from_text("Read 1 Corinthians 13:4-7 on love.")

        assert len(ext.references) == 1
        ref = ext.references[0]
        assert ref["reference"] == "1 Corinthians 13:4-7"
        assert ref["verse_end"] == "7"

    def test_chapter_verse_spoken(self):
        ext = VerseExtractor()
        ext.extract_from_text("Open to Romans chapter 8, verse 28.")

        assert len(ext.references) == 1
        ref = ext.references[0]
        assert ref["book"] == "Romans"
        assert ref["chapter"] == "8"
        assert ref["verse_start"] == "28"

    def test_spoken_intro_pattern(self):
        ext = VerseExtractor()
        ext.extract_from_text("Let's turn to John chapter three.")

        assert len(ext.references) >= 1
        ref = ext.references[0]
        assert ref["book"] == "John"
        assert ref["chapter"] == "3"

    def test_context_tracking_bare_verse(self):
        ext = VerseExtractor()
        # First establish context
        ext.extract_from_text("Read Romans 8:28.")
        # Then use bare verse reference
        ext.extract_from_text("Also verse 30 is important.")

        assert len(ext.references) == 2
        assert ext.references[1]["reference"] == "Romans 8:30"
        assert ext.references[1]["pattern"] == "bare_verse"

    def test_deduplication(self):
        ext = VerseExtractor()
        ext.extract_from_text("Romans 8:28 says something. Romans 8:28 again.")

        assert len(ext.references) == 1

    def test_multiple_verses_in_one_text(self):
        ext = VerseExtractor()
        ext.extract_from_text("Compare John 3:16 with Romans 5:8.")

        assert len(ext.references) == 2
        books = {ref["book"] for ref in ext.references}
        assert "John" in books
        assert "Romans" in books

    def test_chapter_only(self):
        ext = VerseExtractor()
        ext.extract_from_text("We are studying Genesis 1 tonight.")

        assert len(ext.references) >= 1
        ref = ext.references[0]
        assert ref["book"] == "Genesis"
        assert ref["chapter"] == "1"
        assert ref["verse_start"] is None

    def test_speaker_preserved(self):
        ext = VerseExtractor()
        ext.extract_from_text("Read John 3:16.", speaker="Pastor Smith")

        assert ext.references[0]["speaker"] == "Pastor Smith"

    def test_timestamp_preserved(self):
        ext = VerseExtractor()
        ext.extract_from_text("John 3:16", timestamp="00:15:30")

        assert ext.references[0]["timestamp"] == "00:15:30"

    def test_empty_text(self):
        ext = VerseExtractor()
        ext.extract_from_text("")
        assert len(ext.references) == 0

    def test_no_verses(self):
        ext = VerseExtractor()
        ext.extract_from_text("The weather is nice today.")
        assert len(ext.references) == 0

    def test_overlapping_patterns_no_duplicates(self):
        ext = VerseExtractor()
        ext.extract_from_text("Turn to Romans 8:28.")

        # Should not double-count from both spoken_intro and explicit patterns
        romans_refs = [r for r in ext.references if r["book"] == "Romans"]
        assert len(romans_refs) == 1
