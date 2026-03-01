"""Tests for tools/validate_session.py — session validation pipeline."""

import csv
import os
import sys
from datetime import datetime
from unittest.mock import patch

import numpy as np

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.validate_session import (
    AudioSegment,
    compare_stt,
    compare_stt_aligned,
    compare_translations,
    find_flagged_segments,
    generate_report,
    local_rows_to_timed_segments,
    match_screenshots_to_chunks,
    parse_screenshot_timestamp,
    read_session_csv,
    yt_segments_to_timed_segments,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CSV_HEADER = [
    "chunk_id",
    "timestamp",
    "english",
    "spanish_a",
    "spanish_b",
    "stt_latency_ms",
    "latency_a_ms",
    "latency_b_ms",
    "e2e_latency_ms",
    "stt_confidence",
    "tps_a",
    "tps_b",
    "qe_a",
    "qe_b",
    "utterance_dur",
]


def _make_csv(tmp_path, rows):
    """Write a test CSV with the standard header."""
    path = os.path.join(str(tmp_path), "test_session.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_HEADER)
        writer.writeheader()
        for row in rows:
            full_row = {k: "" for k in _CSV_HEADER}
            full_row.update(row)
            writer.writerow(full_row)
    return path


def _sample_row(**overrides):
    """Create a sample CSV row dict with sensible defaults."""
    row = {
        "chunk_id": "1",
        "timestamp": "2026-03-01T12:01:47.285906",
        "english": "For God so loved the world.",
        "spanish_a": "Porque de tal manera amó Dios al mundo.",
        "stt_confidence": "0.92",
        "qe_a": "0.95",
        "stt_latency_ms": "150.0",
        "latency_a_ms": "650.0",
        "utterance_dur": "3.84",
    }
    row.update(overrides)
    return row


# ---------------------------------------------------------------------------
# Tests: AudioSegment
# ---------------------------------------------------------------------------


class TestAudioSegment:
    def test_properties(self):
        audio = np.zeros(48000, dtype=np.float32)  # 3 seconds at 16kHz
        seg = AudioSegment(audio=audio, start_sample=16000, end_sample=64000)
        assert seg.start_seconds == 1.0
        assert seg.end_seconds == 4.0
        assert len(seg.audio) == 48000

    def test_zero_start(self):
        audio = np.zeros(16000, dtype=np.float32)
        seg = AudioSegment(audio=audio, start_sample=0, end_sample=16000)
        assert seg.start_seconds == 0.0
        assert seg.end_seconds == 1.0


# ---------------------------------------------------------------------------
# Tests: CSV Reading
# ---------------------------------------------------------------------------


class TestReadSessionCSV:
    def test_reads_valid_csv(self, tmp_path):
        path = _make_csv(tmp_path, [_sample_row()])
        rows = read_session_csv(path)
        assert len(rows) == 1
        assert rows[0]["english"] == "For God so loved the world."

    def test_reads_multiple_rows(self, tmp_path):
        path = _make_csv(
            tmp_path,
            [
                _sample_row(chunk_id="1"),
                _sample_row(chunk_id="2", english="He gave his only Son."),
            ],
        )
        rows = read_session_csv(path)
        assert len(rows) == 2
        assert rows[1]["english"] == "He gave his only Son."

    def test_empty_csv(self, tmp_path):
        path = _make_csv(tmp_path, [])
        rows = read_session_csv(path)
        assert rows == []


# ---------------------------------------------------------------------------
# Tests: Screenshot Timestamp Parsing
# ---------------------------------------------------------------------------


class TestParseScreenshotTimestamp:
    def test_standard_format(self):
        ts = parse_screenshot_timestamp("Screenshot 2026-03-01 at 12.59.35.png")
        assert ts is not None
        assert ts == datetime(2026, 3, 1, 12, 59, 35)

    def test_single_digit_hour(self):
        ts = parse_screenshot_timestamp("Screenshot 2026-03-01 at 9.05.00.png")
        assert ts is not None
        assert ts.hour == 9

    def test_no_match(self):
        assert parse_screenshot_timestamp("random_file.png") is None

    def test_invalid_date(self):
        assert parse_screenshot_timestamp("Screenshot 2026-13-01 at 12.00.00.png") is None


# ---------------------------------------------------------------------------
# Tests: Screenshot Matching
# ---------------------------------------------------------------------------


class TestMatchScreenshotsToChunks:
    def test_matches_close_timestamps(self, tmp_path):
        # Create screenshot files
        screenshot_dir = os.path.join(str(tmp_path), "screenshots")
        os.makedirs(screenshot_dir)
        # 12:01:50 — close to the sample row timestamp of 12:01:47
        with open(os.path.join(screenshot_dir, "Screenshot 2026-03-01 at 12.01.50.png"), "w") as f:
            f.write("fake")

        rows = [_sample_row(timestamp="2026-03-01T12:01:47.285906")]
        matches = match_screenshots_to_chunks(screenshot_dir, rows)
        assert len(matches) == 1
        assert matches[0]["matched_chunk"] is not None
        assert matches[0]["offset_seconds"] <= 5.0

    def test_no_match_when_too_far(self, tmp_path):
        screenshot_dir = os.path.join(str(tmp_path), "screenshots")
        os.makedirs(screenshot_dir)
        # 13:00:00 — 58 minutes away from 12:01:47
        with open(os.path.join(screenshot_dir, "Screenshot 2026-03-01 at 13.00.00.png"), "w") as f:
            f.write("fake")

        rows = [_sample_row(timestamp="2026-03-01T12:01:47.285906")]
        matches = match_screenshots_to_chunks(screenshot_dir, rows, max_offset_seconds=30.0)
        assert len(matches) == 1
        assert matches[0]["matched_chunk"] is None

    def test_nonexistent_dir(self):
        matches = match_screenshots_to_chunks("/nonexistent/dir", [])
        assert matches == []


# ---------------------------------------------------------------------------
# Tests: TimedSegment Conversion
# ---------------------------------------------------------------------------


class TestLocalRowsToTimedSegments:
    def test_basic_conversion(self):
        rows = [
            _sample_row(
                chunk_id="1",
                timestamp="2026-03-01T12:00:00.000000",
                english="First sentence.",
                utterance_dur="3.0",
            ),
            _sample_row(
                chunk_id="2",
                timestamp="2026-03-01T12:00:05.000000",
                english="Second sentence.",
                utterance_dur="4.0",
            ),
        ]
        segs = local_rows_to_timed_segments(rows)
        assert len(segs) == 2
        assert segs[0].start == 0.0
        assert segs[0].end == 3.0
        assert segs[0].text == "First sentence."
        assert segs[1].start == 5.0
        assert segs[1].end == 9.0

    def test_with_time_offset(self):
        rows = [
            _sample_row(
                timestamp="2026-03-01T12:00:00.000000",
                english="Hello world.",
                utterance_dur="2.0",
            ),
        ]
        segs = local_rows_to_timed_segments(rows, time_offset=100.0)
        assert segs[0].start == 100.0
        assert segs[0].end == 102.0

    def test_empty_rows(self):
        assert local_rows_to_timed_segments([]) == []

    def test_skips_empty_english(self):
        rows = [
            _sample_row(timestamp="2026-03-01T12:00:00.000000", english="", utterance_dur="2.0"),
            _sample_row(timestamp="2026-03-01T12:00:05.000000", english="Real text.", utterance_dur="3.0"),
        ]
        segs = local_rows_to_timed_segments(rows)
        assert len(segs) == 1
        assert segs[0].text == "Real text."

    def test_fallback_duration_from_gap(self):
        """When utterance_dur is missing, estimate from gap to next row."""
        rows = [
            _sample_row(
                chunk_id="1",
                timestamp="2026-03-01T12:00:00.000000",
                english="First.",
                utterance_dur="",
            ),
            _sample_row(
                chunk_id="2",
                timestamp="2026-03-01T12:00:07.000000",
                english="Second.",
                utterance_dur="3.0",
            ),
        ]
        segs = local_rows_to_timed_segments(rows)
        assert len(segs) == 2
        # First segment should span 7 seconds (gap to next)
        assert segs[0].end == 7.0

    def test_no_timestamps(self):
        rows = [_sample_row(timestamp="", english="No timestamp.")]
        segs = local_rows_to_timed_segments(rows)
        assert len(segs) == 0


class TestYtSegmentsToTimedSegments:
    def test_basic_conversion(self):
        yt_segs = [
            {"english": "Hello world", "start_seconds": 0.0, "end_seconds": 3.0},
            {"english": "Goodbye world", "start_seconds": 5.0, "end_seconds": 8.0},
        ]
        segs = yt_segments_to_timed_segments(yt_segs)
        assert len(segs) == 2
        assert segs[0].text == "Hello world"
        assert segs[0].start == 0.0
        assert segs[1].end == 8.0

    def test_missing_timing(self):
        """Segments without start_seconds/end_seconds are skipped."""
        yt_segs = [
            {"english": "Has timing", "start_seconds": 0.0, "end_seconds": 3.0},
            {"english": "No timing"},
        ]
        segs = yt_segments_to_timed_segments(yt_segs)
        assert len(segs) == 1

    def test_empty_english_skipped(self):
        yt_segs = [{"english": "", "start_seconds": 0.0, "end_seconds": 3.0}]
        segs = yt_segments_to_timed_segments(yt_segs)
        assert len(segs) == 0

    def test_empty_list(self):
        assert yt_segments_to_timed_segments([]) == []


# ---------------------------------------------------------------------------
# Tests: STT Comparison (legacy)
# ---------------------------------------------------------------------------


class TestCompareSTT:
    def test_with_text(self):
        """compare_stt returns a dict with stt_wer and word counts."""
        rows = [_sample_row(english="For God so loved the world")]
        yt_segs = [{"english": "For God so loved the world"}]
        result = compare_stt(rows, yt_segs)
        # jiwer is mocked in CI, so we just check structure
        assert "stt_wer" in result
        assert result["live_word_count"] > 0
        assert result["yt_word_count"] > 0

    def test_empty_live(self):
        rows = [_sample_row(english="")]
        yt_segs = [{"english": "something"}]
        result = compare_stt(rows, yt_segs)
        assert result["stt_wer"] is None

    def test_empty_youtube(self):
        rows = [_sample_row(english="something")]
        yt_segs = []
        result = compare_stt(rows, yt_segs)
        assert result["stt_wer"] is None


# ---------------------------------------------------------------------------
# Tests: Aligned STT Comparison
# ---------------------------------------------------------------------------


def _mock_compute_wer_cer(ref, hyp):
    """Return a plausible WER/CER for testing (avoids jiwer dependency)."""
    if not ref or not hyp:
        return 1.0, 1.0, {"substitutions": 0, "deletions": 0, "insertions": 0, "hits": 0}
    ref_words = set(ref.lower().split())
    hyp_words = set(hyp.lower().split())
    if ref_words == hyp_words:
        n = len(ref.split())
        return 0.0, 0.0, {"substitutions": 0, "deletions": 0, "insertions": 0, "hits": n}
    return 0.15, 0.10, {"substitutions": 1, "deletions": 0, "insertions": 0, "hits": 5}


# Patch path for compute_wer_cer as used inside live_caption_monitor
_PATCH_WER = "tools.live_caption_monitor.compute_wer_cer"


class TestCompareSTTAligned:
    @patch(_PATCH_WER, side_effect=_mock_compute_wer_cer)
    def test_single_speaker_with_timing(self, mock_wer):
        """Single speaker with timed YT segments produces aligned comparison."""
        rows = [
            _sample_row(
                chunk_id="1",
                timestamp="2026-03-01T12:00:00.000000",
                english="The grace of God is sufficient",
                utterance_dur="4.0",
            ),
            _sample_row(
                chunk_id="2",
                timestamp="2026-03-01T12:00:05.000000",
                english="for all who believe in him",
                utterance_dur="3.0",
            ),
        ]
        yt_segs = [
            {
                "english": "The grace of God is sufficient",
                "start_seconds": 0.0,
                "end_seconds": 4.0,
            },
            {
                "english": "for all who believe in him",
                "start_seconds": 5.0,
                "end_seconds": 8.0,
            },
        ]
        result = compare_stt_aligned([("Speaker 1", rows)], yt_segs)
        assert "combined_wer" in result
        assert "per_speaker_wer" in result
        assert "Speaker 1" in result["per_speaker_wer"]
        assert result["live_word_count"] > 0
        assert result["yt_word_count"] > 0

    @patch(_PATCH_WER, side_effect=_mock_compute_wer_cer)
    def test_multi_speaker(self, mock_wer):
        """Two speakers produce per-speaker WER entries."""
        sp1_rows = [
            _sample_row(
                chunk_id="1",
                timestamp="2026-03-01T11:37:00.000000",
                english="Speaker one is talking about grace",
                utterance_dur="4.0",
            ),
        ]
        sp2_rows = [
            _sample_row(
                chunk_id="1",
                timestamp="2026-03-01T12:01:00.000000",
                english="Speaker two discusses redemption",
                utterance_dur="4.0",
            ),
        ]
        yt_segs = [
            {
                "english": "Speaker one is talking about grace",
                "start_seconds": 0.0,
                "end_seconds": 4.0,
            },
            {
                "english": "Speaker two discusses redemption",
                "start_seconds": 5.0,
                "end_seconds": 9.0,
            },
        ]
        result = compare_stt_aligned(
            [("Speaker 1", sp1_rows), ("Speaker 2", sp2_rows)],
            yt_segs,
        )
        assert "Speaker 1" in result["per_speaker_wer"]
        assert "Speaker 2" in result["per_speaker_wer"]

    def test_empty_local(self):
        result = compare_stt_aligned([("Speaker 1", [])], [])
        assert result["combined_wer"] is None
        assert result["live_word_count"] == 0

    def test_fallback_when_no_yt_timing(self):
        """Falls back to legacy compare_stt when YT segments lack timing."""
        rows = [_sample_row(english="For God so loved the world")]
        yt_segs = [{"english": "For God so loved the world"}]  # no start/end_seconds
        result = compare_stt_aligned([("Speaker 1", rows)], yt_segs)
        # Should still produce a combined_wer via fallback
        assert "combined_wer" in result


# ---------------------------------------------------------------------------
# Tests: Translation Comparison
# ---------------------------------------------------------------------------


class TestCompareTranslations:
    def test_with_qe_scores(self):
        rows = [
            _sample_row(
                qe_a="0.95",
                retranslate_marian_qe="0.90",
                retranslate_12b_qe="0.92",
            )
        ]
        # Add re-translation keys that would be set by retranslate_session
        rows[0]["retranslate_marian_qe"] = "0.90"
        rows[0]["retranslate_12b_qe"] = "0.92"
        rows[0]["retranslate_marian"] = "Porque de tal manera amó Dios al mundo."

        result = compare_translations(rows)
        assert result["live_qe_a_mean"] is not None
        assert result["live_qe_a_mean"] == 0.95

    def test_empty_rows(self):
        result = compare_translations([])
        assert result["live_qe_a_mean"] is None
        assert result["marian_consistency_rate"] is None


# ---------------------------------------------------------------------------
# Tests: Flagged Segments
# ---------------------------------------------------------------------------


class TestFindFlaggedSegments:
    def test_flags_low_qe(self):
        rows = [_sample_row(qe_a="0.3")]
        flagged = find_flagged_segments(rows, qe_threshold=0.6)
        assert len(flagged) == 1
        assert "low live QE" in flagged[0]["reasons"][0]

    def test_flags_low_confidence(self):
        rows = [_sample_row(stt_confidence="0.4", qe_a="0.9")]
        flagged = find_flagged_segments(rows, qe_threshold=0.6)
        assert len(flagged) == 1
        assert "low STT confidence" in flagged[0]["reasons"][0]

    def test_no_flags_when_good(self):
        rows = [_sample_row(stt_confidence="0.92", qe_a="0.95")]
        flagged = find_flagged_segments(rows, qe_threshold=0.6)
        assert len(flagged) == 0

    def test_empty_english_skipped(self):
        rows = [_sample_row(english="", qe_a="0.1")]
        flagged = find_flagged_segments(rows)
        assert len(flagged) == 0

    def test_multiple_reasons(self):
        rows = [_sample_row(stt_confidence="0.3", qe_a="0.2")]
        flagged = find_flagged_segments(rows, qe_threshold=0.6)
        assert len(flagged) == 1
        assert len(flagged[0]["reasons"]) == 2


# ---------------------------------------------------------------------------
# Tests: Report Generation
# ---------------------------------------------------------------------------

# Common translation comparison dict for reuse
_EMPTY_TC = {
    "live_qe_a_mean": None,
    "live_qe_a_count": 0,
    "retranslate_marian_qe_mean": None,
    "retranslate_12b_qe_mean": None,
    "marian_consistency_rate": None,
    "marian_consistency_count": 0,
}


class TestGenerateReport:
    def test_generates_markdown(self):
        rows = [_sample_row()]
        report = generate_report(
            csv_path="metrics/test.csv",
            live_rows=rows,
            stt_comparison={"stt_wer": 0.12, "live_word_count": 100, "yt_word_count": 95},
            translation_comparison={
                "live_qe_a_mean": 0.85,
                "live_qe_a_count": 10,
                "retranslate_marian_qe_mean": 0.82,
                "retranslate_12b_qe_mean": 0.88,
                "marian_consistency_rate": 0.75,
                "marian_consistency_count": 10,
            },
            flagged_segments=[],
            yt_segments=None,
            feedback_matches=None,
            video_id="test123",
        )
        assert "# Session Validation Report" in report
        assert "test123" in report
        assert "12.0%" in report
        assert "STT Quality" in report
        assert "Translation Quality" in report

    def test_report_with_flagged_segments(self):
        rows = [_sample_row()]
        flagged = [
            {
                "chunk_id": "1",
                "timestamp": "2026-03-01T12:01:47",
                "english": "Test text",
                "spanish_a": "Texto de prueba",
                "retranslate_marian": "Texto prueba",
                "retranslate_12b": "",
                "reasons": ["low live QE (0.30)"],
            }
        ]
        report = generate_report(
            csv_path="metrics/test.csv",
            live_rows=rows,
            stt_comparison={"stt_wer": None, "live_word_count": 0, "yt_word_count": 0},
            translation_comparison=_EMPTY_TC,
            flagged_segments=flagged,
            yt_segments=None,
            feedback_matches=None,
            video_id=None,
        )
        assert "Flagged Segments" in report
        assert "1 segments flagged" in report
        assert "low live QE" in report

    def test_report_with_feedback(self):
        rows = [_sample_row()]
        feedback = [
            {
                "screenshot": "Screenshot 2026-03-01 at 12.01.50.png",
                "screenshot_path": "/path/to/screenshot.png",
                "screenshot_time": "12:01:50",
                "matched_chunk": _sample_row(),
                "offset_seconds": 3.0,
            }
        ]
        report = generate_report(
            csv_path="metrics/test.csv",
            live_rows=rows,
            stt_comparison={"stt_wer": None, "live_word_count": 0, "yt_word_count": 0},
            translation_comparison=_EMPTY_TC,
            flagged_segments=[],
            yt_segments=None,
            feedback_matches=feedback,
            video_id=None,
        )
        assert "Human-Flagged Segments" in report
        assert "Screenshot 2026-03-01 at 12.01.50.png" in report

    def test_report_with_yt_translation_qe(self):
        """Report includes YouTube Translation Quality section when yt_segments have QE scores."""
        rows = [_sample_row()]
        yt_segs = [
            {
                "english": "For God so loved the world",
                "start_seconds": 0.0,
                "end_seconds": 3.0,
                "yt_marian": "Porque de tal manera amó Dios al mundo",
                "yt_marian_qe": 0.88,
                "yt_4b": "Porque Dios amó tanto al mundo",
                "yt_4b_qe": 0.91,
                "yt_12b": "Porque de tal manera amó Dios al mundo",
                "yt_12b_qe": 0.93,
            },
            {
                "english": "He gave his only Son",
                "start_seconds": 3.0,
                "end_seconds": 6.0,
                "yt_marian": "Dio a su hijo unigénito",
                "yt_marian_qe": 0.85,
                "yt_4b": "Dio a su único hijo",
                "yt_4b_qe": 0.87,
                "yt_12b": "Dio a su hijo unigénito",
                "yt_12b_qe": 0.90,
            },
        ]
        report = generate_report(
            csv_path="metrics/test.csv",
            live_rows=rows,
            stt_comparison={"stt_wer": 0.10, "live_word_count": 50, "yt_word_count": 48},
            translation_comparison={
                "live_qe_a_mean": 0.90,
                "live_qe_a_count": 5,
                "retranslate_marian_qe_mean": None,
                "retranslate_12b_qe_mean": None,
                "marian_consistency_rate": None,
                "marian_consistency_count": 0,
            },
            flagged_segments=[],
            yt_segments=yt_segs,
            feedback_matches=None,
            video_id="test123",
        )
        assert "YouTube Translation Quality" in report
        assert "MarianMT" in report
        assert "Gemma 4B" in report
        assert "Gemma 12B" in report
        # Check that averages are present (0.865, 0.890, 0.915)
        assert "0.865" in report
        assert "0.890" in report
        assert "0.915" in report

    def test_report_no_yt_translation_qe_without_segments(self):
        """Report omits YouTube Translation Quality section when no yt_segments."""
        rows = [_sample_row()]
        report = generate_report(
            csv_path="metrics/test.csv",
            live_rows=rows,
            stt_comparison={"stt_wer": None, "live_word_count": 0, "yt_word_count": 0},
            translation_comparison=_EMPTY_TC,
            flagged_segments=[],
            yt_segments=None,
            feedback_matches=None,
            video_id=None,
        )
        assert "YouTube Translation Quality" not in report

    def test_report_no_youtube(self):
        """Report works even without YouTube comparison data."""
        rows = [_sample_row()]
        report = generate_report(
            csv_path="metrics/test.csv",
            live_rows=rows,
            stt_comparison={"stt_wer": None, "live_word_count": 0, "yt_word_count": 0},
            translation_comparison={
                "live_qe_a_mean": 0.90,
                "live_qe_a_count": 5,
                "retranslate_marian_qe_mean": None,
                "retranslate_12b_qe_mean": None,
                "marian_consistency_rate": None,
                "marian_consistency_count": 0,
            },
            flagged_segments=[],
            yt_segments=None,
            feedback_matches=None,
            video_id=None,
        )
        assert "not available" in report
        assert "No segments flagged" in report


# ---------------------------------------------------------------------------
# Tests: Multi-Speaker Report Generation
# ---------------------------------------------------------------------------


class TestGenerateReportMultiSpeaker:
    def _make_speaker_sessions(self):
        sp1_rows = [
            _sample_row(
                chunk_id="1",
                timestamp="2026-03-01T11:37:00.000000",
                english="Speaker one sentence.",
                stt_confidence="0.90",
                qe_a="0.85",
            ),
        ]
        sp2_rows = [
            _sample_row(
                chunk_id="1",
                timestamp="2026-03-01T12:01:00.000000",
                english="Speaker two sentence.",
                stt_confidence="0.88",
                qe_a="0.92",
            ),
        ]
        return [("Speaker 1", sp1_rows), ("Speaker 2", sp2_rows)]

    def test_multi_speaker_header(self):
        sessions = self._make_speaker_sessions()
        all_rows = sessions[0][1] + sessions[1][1]
        report = generate_report(
            csv_path=["sp1.csv", "sp2.csv"],
            live_rows=all_rows,
            stt_comparison={
                "combined_wer": 0.15,
                "per_speaker_wer": {
                    "Speaker 1": {"wer": 0.12, "cer": 0.08, "local_words": 50, "yt_words": 48},
                    "Speaker 2": {"wer": 0.18, "cer": 0.10, "local_words": 60, "yt_words": 55},
                },
                "per_window": [],
                "live_word_count": 110,
                "yt_word_count": 103,
                "comparable_windows": 5,
                "total_windows": 7,
                "global_offset": 1.5,
            },
            translation_comparison=[
                {
                    "live_qe_a_mean": 0.85,
                    "live_qe_a_count": 5,
                    "retranslate_marian_qe_mean": 0.80,
                    "retranslate_12b_qe_mean": None,
                    "marian_consistency_rate": 0.70,
                    "marian_consistency_count": 5,
                },
                {
                    "live_qe_a_mean": 0.92,
                    "live_qe_a_count": 8,
                    "retranslate_marian_qe_mean": 0.88,
                    "retranslate_12b_qe_mean": None,
                    "marian_consistency_rate": 0.80,
                    "marian_consistency_count": 8,
                },
            ],
            flagged_segments=[[], []],
            yt_segments=None,
            feedback_matches=None,
            video_id="testVID",
            speaker_sessions=sessions,
        )
        assert "Multi-Speaker" in report
        assert "Speaker 1" in report
        assert "Speaker 2" in report
        assert "testVID" in report
        assert "12.0%" in report  # Speaker 1 WER
        assert "18.0%" in report  # Speaker 2 WER
        assert "15.0%" in report  # Combined WER
        assert "Comparable windows" in report

    def test_multi_speaker_per_speaker_flagged(self):
        sessions = self._make_speaker_sessions()
        all_rows = sessions[0][1] + sessions[1][1]
        flagged_sp1 = [
            {
                "chunk_id": "1",
                "timestamp": "2026-03-01T11:37:00",
                "english": "Flagged text",
                "spanish_a": "Texto marcado",
                "retranslate_marian": "",
                "retranslate_12b": "",
                "reasons": ["low live QE (0.30)"],
            }
        ]
        report = generate_report(
            csv_path=["sp1.csv", "sp2.csv"],
            live_rows=all_rows,
            stt_comparison={
                "combined_wer": None,
                "per_speaker_wer": {},
                "per_window": [],
                "live_word_count": 0,
                "yt_word_count": 0,
            },
            translation_comparison=[_EMPTY_TC, _EMPTY_TC],
            flagged_segments=[flagged_sp1, []],
            yt_segments=None,
            feedback_matches=None,
            video_id=None,
            speaker_sessions=sessions,
        )
        assert "Flagged Segments" in report
        assert "1 segments flagged" in report
        assert "No segments flagged" in report  # Speaker 2

    def test_single_session_backward_compat(self):
        """Single session without speaker_sessions produces old-style report."""
        rows = [_sample_row()]
        report = generate_report(
            csv_path="metrics/test.csv",
            live_rows=rows,
            stt_comparison={"stt_wer": 0.10, "live_word_count": 50, "yt_word_count": 48},
            translation_comparison={
                "live_qe_a_mean": 0.90,
                "live_qe_a_count": 5,
                "retranslate_marian_qe_mean": None,
                "retranslate_12b_qe_mean": None,
                "marian_consistency_rate": None,
                "marian_consistency_count": 0,
            },
            flagged_segments=[],
            yt_segments=None,
            feedback_matches=None,
            video_id="abc123",
        )
        assert "# Session Validation Report" in report
        assert "Multi-Speaker" not in report
        assert "10.0%" in report
