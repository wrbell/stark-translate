"""Tests for pure functions in features/summarize_sermon.py."""

import csv
import json

# ===================================================================
# _format_timestamp
# ===================================================================


class TestFormatTimestamp:
    def test_zero(self):
        from features.summarize_sermon import _format_timestamp

        assert _format_timestamp(0) == "00:00:00"

    def test_90_seconds(self):
        from features.summarize_sermon import _format_timestamp

        assert _format_timestamp(90) == "00:01:30"

    def test_over_one_hour(self):
        from features.summarize_sermon import _format_timestamp

        assert _format_timestamp(3661) == "01:01:01"

    def test_fractional_truncated(self):
        from features.summarize_sermon import _format_timestamp

        # int(seconds % 60) truncates fractional part
        assert _format_timestamp(90.9) == "00:01:30"


# ===================================================================
# has_diarization
# ===================================================================


class TestHasDiarization:
    def test_two_speakers(self):
        from features.summarize_sermon import has_diarization

        entries = [
            {"text": "Hello", "speaker": "Speaker A"},
            {"text": "World", "speaker": "Speaker B"},
        ]
        assert has_diarization(entries) is True

    def test_one_speaker(self):
        from features.summarize_sermon import has_diarization

        entries = [
            {"text": "Hello", "speaker": "Speaker A"},
            {"text": "World", "speaker": "Speaker A"},
        ]
        assert has_diarization(entries) is False

    def test_no_speaker_key(self):
        from features.summarize_sermon import has_diarization

        entries = [{"text": "Hello"}, {"text": "World"}]
        assert has_diarization(entries) is False

    def test_empty_list(self):
        from features.summarize_sermon import has_diarization

        assert has_diarization([]) is False


# ===================================================================
# build_transcript_text
# ===================================================================


class TestBuildTranscriptText:
    def test_short_passthrough(self):
        from features.summarize_sermon import build_transcript_text

        entries = [
            {"text": "Hello world", "speaker": None},
            {"text": "How are you", "speaker": None},
        ]
        result = build_transcript_text(entries, max_chars=1000)
        assert result == "Hello world\nHow are you"

    def test_with_speaker_labels(self):
        from features.summarize_sermon import build_transcript_text

        entries = [
            {"text": "Hello", "speaker": "Speaker A"},
            {"text": "Hi", "speaker": "Speaker B"},
        ]
        result = build_transcript_text(entries, max_chars=1000)
        assert "Speaker A: Hello" in result
        assert "Speaker B: Hi" in result

    def test_truncation(self):
        from features.summarize_sermon import build_transcript_text

        entries = [{"text": "X" * 200, "speaker": None} for _ in range(10)]
        result = build_transcript_text(entries, max_chars=500)
        assert "[... middle portion omitted for brevity ...]" in result

    def test_exact_boundary(self):
        from features.summarize_sermon import build_transcript_text

        entries = [{"text": "ABC", "speaker": None}]
        result = build_transcript_text(entries, max_chars=3)
        assert result == "ABC"


# ===================================================================
# get_speaker_texts
# ===================================================================


class TestGetSpeakerTexts:
    def test_two_speakers(self):
        from features.summarize_sermon import get_speaker_texts

        entries = [
            {"text": "Hello", "speaker": "A"},
            {"text": "World", "speaker": "B"},
            {"text": "Again", "speaker": "A"},
        ]
        result = get_speaker_texts(entries)
        assert result["A"] == "Hello Again"
        assert result["B"] == "World"

    def test_missing_speaker_defaults_to_unknown(self):
        from features.summarize_sermon import get_speaker_texts

        entries = [{"text": "Hello"}, {"text": "World"}]
        result = get_speaker_texts(entries)
        assert "Unknown" in result

    def test_single_speaker(self):
        from features.summarize_sermon import get_speaker_texts

        entries = [
            {"text": "One", "speaker": "X"},
            {"text": "Two", "speaker": "X"},
        ]
        result = get_speaker_texts(entries)
        assert len(result) == 1
        assert result["X"] == "One Two"


# ===================================================================
# load_csv_transcript
# ===================================================================


class TestLoadCsvTranscript:
    def test_valid_csv(self, tmp_path):
        from features.summarize_sermon import load_csv_transcript

        csv_path = str(tmp_path / "test.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["timestamp", "english"])
            writer.writeheader()
            writer.writerow({"timestamp": "00:00:01", "english": "Hello world"})
            writer.writerow({"timestamp": "00:00:05", "english": "God is love"})
        result = load_csv_transcript(csv_path)
        assert len(result) == 2
        assert result[0]["text"] == "Hello world"

    def test_empty_csv(self, tmp_path):
        from features.summarize_sermon import load_csv_transcript

        csv_path = str(tmp_path / "empty.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["timestamp", "english"])
            writer.writeheader()
        result = load_csv_transcript(csv_path)
        assert result == []


# ===================================================================
# load_jsonl_transcript
# ===================================================================


class TestLoadJsonlTranscript:
    def test_valid_jsonl(self, tmp_path):
        from features.summarize_sermon import load_jsonl_transcript

        path = str(tmp_path / "test.jsonl")
        with open(path, "w") as f:
            # Metadata header — should be skipped
            f.write(json.dumps({"_metadata": {"source": "test"}}) + "\n")
            f.write(json.dumps({"speaker": "A", "start": 10, "text": "Hello"}) + "\n")
            f.write(json.dumps({"speaker": "B", "start": 20, "text": "World"}) + "\n")
        result = load_jsonl_transcript(path)
        assert len(result) == 2
        assert result[0]["text"] == "Hello"
        assert result[0]["speaker"] == "A"

    def test_skips_transcription_errors(self, tmp_path):
        from features.summarize_sermon import load_jsonl_transcript

        path = str(tmp_path / "err.jsonl")
        with open(path, "w") as f:
            f.write(json.dumps({"speaker": "A", "start": 0, "text": "[transcription error: timeout]"}) + "\n")
            f.write(json.dumps({"speaker": "A", "start": 5, "text": "Valid text"}) + "\n")
        result = load_jsonl_transcript(path)
        assert len(result) == 1
        assert result[0]["text"] == "Valid text"


# ===================================================================
# write_summary
# ===================================================================


class TestWriteSummary:
    def test_creates_json_file(self, tmp_path):
        from features.summarize_sermon import write_summary

        output = str(tmp_path / "summaries" / "test.json")
        data = {"english": "Summary", "spanish": "Resumen"}
        write_summary(data, output)
        with open(output) as f:
            loaded = json.load(f)
        assert loaded["english"] == "Summary"
        assert loaded["spanish"] == "Resumen"
