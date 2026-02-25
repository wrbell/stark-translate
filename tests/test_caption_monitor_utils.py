"""Additional tests for utility functions in tools/live_caption_monitor.py.

Focuses on parse_vtt, load_local_csv, and edge cases not covered by
test_caption_monitor.py.  jiwer is a real dependency (installed in CI);
the conftest mock is cleared before importing the module.
"""

from __future__ import annotations

import csv
import sys
import textwrap
from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Clear the conftest jiwer mock so the real jiwer is used.
# ---------------------------------------------------------------------------
if "jiwer" in sys.modules and isinstance(sys.modules["jiwer"], MagicMock):
    del sys.modules["jiwer"]

from tools.live_caption_monitor import (
    TimedSegment,
    compute_wer_cer,
    detect_youtube_repetition,
    get_content_words,
    interpret_wer,
    load_local_csv,
    normalize_text,
    parse_vtt,
    segments_in_window,
)

# ===================================================================
# 1. normalize_text — additional edge cases
# ===================================================================


class TestNormalizeTextEdgeCases:
    """Edge cases for normalize_text() beyond the existing test suite."""

    def test_nested_brackets(self):
        assert normalize_text("hello [Music [loud]] world") == "hello world"

    def test_multiple_bracket_types(self):
        result = normalize_text("[Music] Hello (inaudible) there [Applause]")
        assert result == "hello there"

    def test_only_tags(self):
        assert normalize_text("[Music] (inaudible)") == ""

    def test_unicode_text(self):
        # Accented characters should survive (they match \w)
        result = normalize_text("cafe")
        assert result == "cafe"

    def test_numbers_preserved(self):
        result = normalize_text("Romans 8:28")
        assert "romans" in result
        assert "8" in result
        assert "28" in result


# ===================================================================
# 2. get_content_words — additional edge cases
# ===================================================================


class TestGetContentWordsEdgeCases:
    """Extra edge cases for get_content_words()."""

    def test_all_stopwords_returns_empty(self):
        assert get_content_words("the and is are was were") == set()

    def test_duplicates_collapsed_by_set(self):
        result = get_content_words("grace grace grace mercy mercy")
        assert result == {"grace", "mercy"}

    def test_mixed_content_and_stopwords(self):
        result = get_content_words("the lord is my shepherd")
        # "my" is in STOPWORDS, so only content words remain
        assert "my" not in result
        assert result == {"lord", "shepherd"}


# ===================================================================
# 3. interpret_wer — boundary precision
# ===================================================================


class TestInterpretWerBoundaries:
    """Extra boundary tests for interpret_wer()."""

    def test_exact_zero(self):
        assert interpret_wer(0.0) == "excellent"

    def test_just_above_excellent(self):
        assert interpret_wer(0.0500001) == "comparable_or_better"

    def test_just_above_comparable(self):
        assert interpret_wer(0.1500001) == "potential_issues"

    def test_just_above_potential(self):
        assert interpret_wer(0.2500001) == "likely_degradation"


# ===================================================================
# 4. compute_wer_cer — additional scenarios
# ===================================================================


class TestComputeWerCerExtra:
    """Additional WER/CER scenarios using real jiwer."""

    def test_insertion_only(self):
        wer, cer, ops = compute_wer_cer("the cat", "the big fat cat")
        assert ops["insertions"] > 0
        assert ops["hits"] >= 2

    def test_deletion_only(self):
        wer, cer, ops = compute_wer_cer("the big fat cat", "the cat")
        assert ops["deletions"] > 0

    def test_cer_lower_than_wer_for_similar_words(self):
        # "cat" vs "bat" is 1 sub at word level (WER=1/3) but only 1 char diff
        wer, cer, ops = compute_wer_cer("the cat sat", "the bat sat")
        assert cer < wer


# ===================================================================
# 5. detect_youtube_repetition — additional patterns
# ===================================================================


class TestDetectYoutubeRepetitionExtra:
    """Additional repetition detection patterns."""

    def test_no_false_positive_on_natural_repetition(self):
        # Natural speech might repeat a word once; should not flag
        text = "the grace the amazing grace of god is truly truly wonderful"
        is_degraded, _ = detect_youtube_repetition(text)
        assert is_degraded is False

    def test_adjacent_repeat_with_phrase_coverage(self):
        # YouTube stutter: phrase_coverage > 0.35 AND has_adjacent_repeats
        text = "and he said and he said and he said and he said word word"
        is_degraded, details = detect_youtube_repetition(text)
        assert is_degraded is True


# ===================================================================
# 6. TimedSegment + segments_in_window — additional edge cases
# ===================================================================


class TestTimedSegmentExtra:
    """Additional TimedSegment and segments_in_window tests."""

    def test_segment_end_equals_window_start_no_overlap(self):
        # seg.end must be strictly > window_start for overlap
        segs = [TimedSegment(0.0, 10.0, "first")]
        assert segments_in_window(segs, 10.0, 20.0) == ""

    def test_segment_start_equals_window_end_no_overlap(self):
        # seg.start must be strictly < window_end for overlap
        segs = [TimedSegment(20.0, 30.0, "later")]
        assert segments_in_window(segs, 10.0, 20.0) == ""

    def test_multiple_segments_joined_with_spaces(self):
        segs = [
            TimedSegment(0.0, 5.0, "hello"),
            TimedSegment(3.0, 8.0, "world"),
        ]
        result = segments_in_window(segs, 0.0, 10.0)
        assert result == "hello world"

    def test_single_point_overlap(self):
        # Segment from 5-10, window from 9.999-15 => tiny overlap
        segs = [TimedSegment(5.0, 10.0, "overlap")]
        result = segments_in_window(segs, 9.999, 15.0)
        assert result == "overlap"


# ===================================================================
# 7. parse_vtt — file-based tests
# ===================================================================


class TestParseVtt:
    """Tests for parse_vtt() using tmp_path fixtures."""

    def test_basic_vtt(self, tmp_path):
        vtt_file = tmp_path / "test.vtt"
        vtt_file.write_text(
            textwrap.dedent("""\
                WEBVTT

                00:00:01.000 --> 00:00:04.500
                Hello world

                00:00:05.000 --> 00:00:08.000
                Second segment
            """),
            encoding="utf-8",
        )
        segments = parse_vtt(str(vtt_file))
        assert len(segments) == 2
        assert segments[0].start == pytest.approx(1.0)
        assert segments[0].end == pytest.approx(4.5)
        assert segments[0].text == "Hello world"
        assert segments[1].start == pytest.approx(5.0)
        assert segments[1].end == pytest.approx(8.0)
        assert segments[1].text == "Second segment"

    def test_strips_html_tags(self, tmp_path):
        vtt_file = tmp_path / "tags.vtt"
        vtt_file.write_text(
            textwrap.dedent("""\
                WEBVTT

                00:00:00.000 --> 00:00:03.000
                <c.colorE5E5E5>some</c> <c.colorCCCCCC>styled</c> text
            """),
            encoding="utf-8",
        )
        segments = parse_vtt(str(vtt_file))
        assert len(segments) == 1
        assert segments[0].text == "some styled text"

    def test_skips_blank_text_cues(self, tmp_path):
        vtt_file = tmp_path / "empty.vtt"
        vtt_file.write_text(
            textwrap.dedent("""\
                WEBVTT

                00:00:00.000 --> 00:00:02.000

                00:00:03.000 --> 00:00:05.000
                actual content
            """),
            encoding="utf-8",
        )
        segments = parse_vtt(str(vtt_file))
        assert len(segments) == 1
        assert segments[0].text == "actual content"

    def test_hour_offset_calculation(self, tmp_path):
        vtt_file = tmp_path / "hours.vtt"
        vtt_file.write_text(
            textwrap.dedent("""\
                WEBVTT

                01:30:00.500 --> 01:30:05.000
                Late in the stream
            """),
            encoding="utf-8",
        )
        segments = parse_vtt(str(vtt_file))
        assert len(segments) == 1
        # 1*3600 + 30*60 + 0 + 0.5 = 5400.5
        assert segments[0].start == pytest.approx(5400.5)
        assert segments[0].end == pytest.approx(5405.0)

    def test_multiline_cue_text(self, tmp_path):
        vtt_file = tmp_path / "multiline.vtt"
        vtt_file.write_text(
            textwrap.dedent("""\
                WEBVTT

                00:00:00.000 --> 00:00:05.000
                First line of text
                Second line of text
            """),
            encoding="utf-8",
        )
        segments = parse_vtt(str(vtt_file))
        assert len(segments) == 1
        assert segments[0].text == "First line of text Second line of text"

    def test_empty_vtt_file(self, tmp_path):
        vtt_file = tmp_path / "empty.vtt"
        vtt_file.write_text("WEBVTT\n\n", encoding="utf-8")
        segments = parse_vtt(str(vtt_file))
        assert segments == []

    def test_cue_with_numeric_id(self, tmp_path):
        """VTT cues may have optional numeric identifiers before timestamps."""
        vtt_file = tmp_path / "numbered.vtt"
        vtt_file.write_text(
            textwrap.dedent("""\
                WEBVTT

                1
                00:00:01.000 --> 00:00:03.000
                First cue

                2
                00:00:04.000 --> 00:00:06.000
                Second cue
            """),
            encoding="utf-8",
        )
        segments = parse_vtt(str(vtt_file))
        assert len(segments) == 2
        assert segments[0].text == "First cue"
        assert segments[1].text == "Second cue"

    def test_millisecond_precision(self, tmp_path):
        vtt_file = tmp_path / "ms.vtt"
        vtt_file.write_text(
            textwrap.dedent("""\
                WEBVTT

                00:00:01.123 --> 00:00:02.456
                Precise timing
            """),
            encoding="utf-8",
        )
        segments = parse_vtt(str(vtt_file))
        assert segments[0].start == pytest.approx(1.123)
        assert segments[0].end == pytest.approx(2.456)


# ===================================================================
# 8. load_local_csv — file-based tests
# ===================================================================


class TestLoadLocalCsv:
    """Tests for load_local_csv() using tmp_path fixtures."""

    def _write_csv(self, path, rows):
        """Helper to write CSV rows with standard headers."""
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["chunk_id", "timestamp", "english", "spanish_a"])
            for row in rows:
                writer.writerow(row)

    def test_basic_three_rows(self, tmp_path):
        csv_file = tmp_path / "session.csv"
        t0 = datetime(2026, 1, 1, 10, 0, 0, tzinfo=UTC)
        t1 = datetime(2026, 1, 1, 10, 0, 5, tzinfo=UTC)
        t2 = datetime(2026, 1, 1, 10, 0, 12, tzinfo=UTC)
        self._write_csv(
            csv_file,
            [
                [1, t0.isoformat(), "First sentence", "Primera oracion"],
                [2, t1.isoformat(), "Second sentence", "Segunda oracion"],
                [3, t2.isoformat(), "Third sentence", "Tercera oracion"],
            ],
        )
        segments = load_local_csv(str(csv_file))
        assert len(segments) == 3
        # First segment starts at t=0
        assert segments[0].start == pytest.approx(0.0)
        assert segments[0].text == "First sentence"
        # Second segment offset = 5s from first
        assert segments[1].start == pytest.approx(5.0)
        assert segments[1].text == "Second sentence"
        # Third segment offset = 12s from first
        assert segments[2].start == pytest.approx(12.0)
        assert segments[2].text == "Third sentence"

    def test_empty_csv(self, tmp_path):
        csv_file = tmp_path / "empty.csv"
        self._write_csv(csv_file, [])
        segments = load_local_csv(str(csv_file))
        assert segments == []

    def test_skips_rows_with_empty_text(self, tmp_path):
        csv_file = tmp_path / "gaps.csv"
        t0 = datetime(2026, 1, 1, 10, 0, 0, tzinfo=UTC)
        t1 = datetime(2026, 1, 1, 10, 0, 5, tzinfo=UTC)
        self._write_csv(
            csv_file,
            [
                [1, t0.isoformat(), "Valid text", "Texto valido"],
                [2, t1.isoformat(), "", ""],  # empty english
            ],
        )
        segments = load_local_csv(str(csv_file))
        assert len(segments) == 1
        assert segments[0].text == "Valid text"

    def test_end_time_derived_from_next_segment(self, tmp_path):
        csv_file = tmp_path / "timing.csv"
        t0 = datetime(2026, 1, 1, 10, 0, 0, tzinfo=UTC)
        t1 = datetime(2026, 1, 1, 10, 0, 7, tzinfo=UTC)
        self._write_csv(
            csv_file,
            [
                [1, t0.isoformat(), "First", "Primero"],
                [2, t1.isoformat(), "Second", "Segundo"],
            ],
        )
        segments = load_local_csv(str(csv_file))
        # First segment's end should equal second segment's start
        assert segments[0].end == pytest.approx(7.0)
        # Last segment gets a default ~3s duration
        assert segments[1].end == pytest.approx(7.0 + 3.0)

    def test_sorts_by_timestamp(self, tmp_path):
        csv_file = tmp_path / "unordered.csv"
        t0 = datetime(2026, 1, 1, 10, 0, 10, tzinfo=UTC)
        t1 = datetime(2026, 1, 1, 10, 0, 0, tzinfo=UTC)
        t2 = datetime(2026, 1, 1, 10, 0, 5, tzinfo=UTC)
        # Write rows out of chronological order
        self._write_csv(
            csv_file,
            [
                [1, t0.isoformat(), "Third", "Tercero"],
                [2, t1.isoformat(), "First", "Primero"],
                [3, t2.isoformat(), "Second", "Segundo"],
            ],
        )
        segments = load_local_csv(str(csv_file))
        assert segments[0].text == "First"
        assert segments[1].text == "Second"
        assert segments[2].text == "Third"
        assert segments[0].start == pytest.approx(0.0)
        assert segments[1].start == pytest.approx(5.0)
        assert segments[2].start == pytest.approx(10.0)

    def test_skips_invalid_timestamp(self, tmp_path):
        csv_file = tmp_path / "bad_ts.csv"
        t0 = datetime(2026, 1, 1, 10, 0, 0, tzinfo=UTC)
        self._write_csv(
            csv_file,
            [
                [1, t0.isoformat(), "Good row", "Buena fila"],
                [2, "not-a-timestamp", "Bad row", "Mala fila"],
            ],
        )
        segments = load_local_csv(str(csv_file))
        assert len(segments) == 1
        assert segments[0].text == "Good row"
