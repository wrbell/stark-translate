"""Tests for training/align_deepgram_chunks.py — Deepgram word-to-chunk alignment."""

from __future__ import annotations

import json
import os
import sys

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from training.align_deepgram_chunks import (
    ALIGNMENT_TOLERANCE,
    determine_eval_sources,
    find_deepgram_words_for_chunk,
    group_chunks_by_source,
    load_deepgram_transcript,
    load_whisper_chunks,
    write_metadata_csv,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_word(word: str, start: float, end: float) -> dict:
    """Create a Deepgram word entry."""
    return {"word": word, "start": start, "end": end, "confidence": 0.99}


def _make_chunk(source: str, start: float, end: float, text: str = "") -> dict:
    """Create a Whisper chunk entry."""
    return {"source": source, "start": start, "end": end, "text": text}


# ---------------------------------------------------------------------------
# find_deepgram_words_for_chunk
# ---------------------------------------------------------------------------


class TestFindDeepgramWordsForChunk:
    def test_exact_match(self):
        words = [
            _make_word("hello", 1.0, 1.5),
            _make_word("world", 1.5, 2.0),
        ]
        matched = find_deepgram_words_for_chunk(words, 1.0, 2.0, tolerance=0.0)
        assert len(matched) == 2
        assert matched[0]["word"] == "hello"
        assert matched[1]["word"] == "world"

    def test_with_tolerance(self):
        words = [
            _make_word("for", 0.95, 1.2),
            _make_word("God", 1.2, 1.5),
            _make_word("so", 1.5, 1.7),
            _make_word("loved", 1.7, 2.05),
        ]
        # Without tolerance, "for" starts before chunk and "loved" ends after
        matched = find_deepgram_words_for_chunk(words, 1.0, 2.0, tolerance=0.0)
        assert len(matched) == 2  # only "God" and "so"

        # With tolerance, all four words match
        matched = find_deepgram_words_for_chunk(words, 1.0, 2.0, tolerance=ALIGNMENT_TOLERANCE)
        assert len(matched) == 4

    def test_no_matching_words(self):
        words = [
            _make_word("far", 10.0, 10.5),
            _make_word("away", 10.5, 11.0),
        ]
        matched = find_deepgram_words_for_chunk(words, 1.0, 2.0, tolerance=0.1)
        assert matched == []

    def test_partial_overlap_excluded(self):
        """Words that straddle the boundary beyond tolerance are excluded."""
        words = [
            _make_word("straddle", 0.5, 1.5),  # starts well before chunk
        ]
        matched = find_deepgram_words_for_chunk(words, 1.0, 2.0, tolerance=0.1)
        assert matched == []

    def test_word_ending_beyond_chunk(self):
        """A word ending far beyond chunk_end + tolerance is excluded."""
        words = [
            _make_word("inside", 1.0, 1.5),
            _make_word("outside", 1.5, 3.0),  # ends way past chunk_end=2.0
        ]
        matched = find_deepgram_words_for_chunk(words, 1.0, 2.0, tolerance=0.1)
        assert len(matched) == 1
        assert matched[0]["word"] == "inside"

    def test_empty_words_list(self):
        matched = find_deepgram_words_for_chunk([], 1.0, 2.0, tolerance=0.1)
        assert matched == []

    def test_boundary_word_at_exact_tolerance(self):
        """Word at exactly the tolerance boundary should be included."""
        words = [_make_word("edge", 0.9, 2.1)]
        # chunk 1.0-2.0 with tolerance 0.1: word.start=0.9 >= 0.9, word.end=2.1 <= 2.1
        matched = find_deepgram_words_for_chunk(words, 1.0, 2.0, tolerance=0.1)
        assert len(matched) == 1


# ---------------------------------------------------------------------------
# group_chunks_by_source
# ---------------------------------------------------------------------------


class TestGroupChunksBySource:
    def test_groups_correctly(self):
        chunks = [
            _make_chunk("sermon_a", 0.0, 5.0),
            _make_chunk("sermon_a", 5.0, 10.0),
            _make_chunk("sermon_b", 0.0, 3.0),
        ]
        groups = group_chunks_by_source(chunks)
        assert len(groups) == 2
        assert len(groups["sermon_a"]) == 2
        assert len(groups["sermon_b"]) == 1

    def test_empty_source_skipped(self):
        chunks = [
            _make_chunk("", 0.0, 1.0),
            _make_chunk("valid", 0.0, 1.0),
        ]
        groups = group_chunks_by_source(chunks)
        assert "valid" in groups
        assert "" not in groups

    def test_empty_list(self):
        groups = group_chunks_by_source([])
        assert len(groups) == 0


# ---------------------------------------------------------------------------
# load_deepgram_transcript
# ---------------------------------------------------------------------------


class TestLoadDeepgramTranscript:
    def test_loads_exact_match(self, tmp_path):
        data = {"words": [{"word": "hello", "start": 0.0, "end": 0.5}]}
        (tmp_path / "sermon_a.deepgram.json").write_text(json.dumps(data))

        result = load_deepgram_transcript(tmp_path, "sermon_a")
        assert result is not None
        assert len(result["words"]) == 1

    def test_loads_without_deepgram_suffix(self, tmp_path):
        data = {"words": [{"word": "hi", "start": 0.0, "end": 0.3}]}
        (tmp_path / "sermon_b.json").write_text(json.dumps(data))

        result = load_deepgram_transcript(tmp_path, "sermon_b")
        assert result is not None

    def test_returns_none_when_missing(self, tmp_path):
        result = load_deepgram_transcript(tmp_path, "nonexistent")
        assert result is None

    def test_fuzzy_match_by_substring(self, tmp_path):
        data = {"words": [{"word": "test", "start": 0.0, "end": 0.5}]}
        (tmp_path / "prefix_sermon_c_suffix.deepgram.json").write_text(json.dumps(data))

        result = load_deepgram_transcript(tmp_path, "sermon_c")
        assert result is not None
        assert result["words"][0]["word"] == "test"


# ---------------------------------------------------------------------------
# load_whisper_chunks
# ---------------------------------------------------------------------------


class TestLoadWhisperChunks:
    def test_loads_chunks(self, tmp_path):
        chunks = [
            _make_chunk("s1", 0.0, 5.0, "hello world"),
            _make_chunk("s1", 5.0, 10.0, "second chunk"),
        ]
        f = tmp_path / "chunks.json"
        f.write_text(json.dumps(chunks))

        result = load_whisper_chunks(f)
        assert len(result) == 2
        assert result[0]["text"] == "hello world"


# ---------------------------------------------------------------------------
# determine_eval_sources
# ---------------------------------------------------------------------------


class TestDetermineEvalSources:
    def test_from_cli_arg(self):
        chunks = [_make_chunk("s1", 0.0, 1.0)]
        result = determine_eval_sources(chunks, "source_a, source_b")
        assert result == {"source_a", "source_b"}

    def test_auto_detect_from_chunks(self):
        chunks = [
            {"source": "train_src", "split": "train"},
            {"source": "eval_src", "split": "eval"},
            {"source": "eval_src2", "split": "eval"},
        ]
        result = determine_eval_sources(chunks, None)
        assert result == {"eval_src", "eval_src2"}

    def test_no_eval_sources(self):
        chunks = [{"source": "s1", "split": "train"}]
        result = determine_eval_sources(chunks, None)
        assert result == set()

    def test_cli_arg_overrides_chunk_metadata(self):
        chunks = [{"source": "auto_eval", "split": "eval"}]
        result = determine_eval_sources(chunks, "manual_eval")
        assert result == {"manual_eval"}
        assert "auto_eval" not in result


# ---------------------------------------------------------------------------
# write_metadata_csv
# ---------------------------------------------------------------------------


class TestWriteMetadataCsv:
    def test_writes_csv(self, tmp_path):
        out_dir = tmp_path / "train"
        rows = [
            {"file_name": "chunk_00001.wav", "transcription": "hello world"},
            {"file_name": "chunk_00002.wav", "transcription": "second line"},
        ]
        write_metadata_csv(out_dir, rows)

        csv_path = out_dir / "metadata.csv"
        assert csv_path.exists()

        lines = csv_path.read_text().strip().split("\n")
        assert len(lines) == 3  # header + 2 rows
        assert "file_name" in lines[0]
        assert "transcription" in lines[0]
        assert "hello world" in lines[1]

    def test_creates_parent_dirs(self, tmp_path):
        out_dir = tmp_path / "nested" / "deep" / "train"
        write_metadata_csv(out_dir, [{"file_name": "a.wav", "transcription": "text"}])
        assert (out_dir / "metadata.csv").exists()

    def test_empty_rows(self, tmp_path):
        out_dir = tmp_path / "empty"
        write_metadata_csv(out_dir, [])
        lines = (out_dir / "metadata.csv").read_text().strip().split("\n")
        assert len(lines) == 1  # header only
