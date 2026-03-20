"""Tests for training/filter_transcripts.py — confidence-based transcript filtering."""

import json

from training.filter_transcripts import (
    DEFAULT_MAX_COMPRESSION,
    DEFAULT_MAX_NO_SPEECH,
    DEFAULT_MIN_LOGPROB,
    filter_segments,
    filter_transcript_file,
    filter_transcripts,
)

# ===================================================================
# filter_segments
# ===================================================================


class TestFilterSegments:
    def test_keeps_high_confidence(self):
        segments = [{"avg_logprob": -0.2, "no_speech_prob": 0.05, "compression_ratio": 1.5}]
        kept, dropped = filter_segments(segments, -0.7, 0.3, 2.4)
        assert len(kept) == 1
        assert len(dropped) == 0

    def test_drops_low_logprob(self):
        segments = [{"avg_logprob": -0.9, "no_speech_prob": 0.05, "compression_ratio": 1.5}]
        kept, dropped = filter_segments(segments, -0.7, 0.3, 2.4)
        assert len(kept) == 0
        assert len(dropped) == 1

    def test_drops_high_no_speech(self):
        segments = [{"avg_logprob": -0.2, "no_speech_prob": 0.5, "compression_ratio": 1.5}]
        kept, dropped = filter_segments(segments, -0.7, 0.3, 2.4)
        assert len(kept) == 0
        assert len(dropped) == 1

    def test_drops_high_compression(self):
        segments = [{"avg_logprob": -0.2, "no_speech_prob": 0.05, "compression_ratio": 2.5}]
        kept, dropped = filter_segments(segments, -0.7, 0.3, 2.4)
        assert len(kept) == 0
        assert len(dropped) == 1

    def test_missing_fields_kept(self):
        """Segments without confidence fields should pass through."""
        segments = [{"text": "hello world"}]
        kept, dropped = filter_segments(segments, -0.7, 0.3, 2.4)
        assert len(kept) == 1
        assert len(dropped) == 0

    def test_boundary_values(self):
        """Values exactly at thresholds should pass."""
        segments = [{"avg_logprob": -0.7, "no_speech_prob": 0.3, "compression_ratio": 2.4}]
        kept, dropped = filter_segments(segments, -0.7, 0.3, 2.4)
        assert len(kept) == 1

    def test_mixed_segments(self):
        segments = [
            {"avg_logprob": -0.2, "text": "good"},
            {"avg_logprob": -0.9, "text": "bad"},
            {"avg_logprob": -0.5, "text": "ok"},
        ]
        kept, dropped = filter_segments(segments, -0.7, 0.3, 2.4)
        assert len(kept) == 2
        assert len(dropped) == 1
        assert dropped[0]["text"] == "bad"

    def test_empty_segments(self):
        kept, dropped = filter_segments([], -0.7, 0.3, 2.4)
        assert kept == []
        assert dropped == []

    def test_any_failing_metric_drops(self):
        """If any single metric fails, the segment is dropped."""
        segments = [{"avg_logprob": -0.2, "no_speech_prob": 0.05, "compression_ratio": 3.0}]
        kept, dropped = filter_segments(segments, -0.7, 0.3, 2.4)
        assert len(dropped) == 1


# ===================================================================
# filter_transcript_file
# ===================================================================


class TestFilterTranscriptFile:
    def test_filters_file(self, tmp_path):
        transcript = {
            "audio_path": "/audio/test.wav",
            "segments": [
                {"text": "good", "avg_logprob": -0.2, "no_speech_prob": 0.05, "compression_ratio": 1.5},
                {"text": "bad", "avg_logprob": -1.0, "no_speech_prob": 0.05, "compression_ratio": 1.5},
            ],
        }
        jf = tmp_path / "test.json"
        jf.write_text(json.dumps(transcript))

        original, kept, data = filter_transcript_file(str(jf), -0.7, 0.3, 2.4)
        assert original == 2
        assert kept == 1
        assert len(data["segments"]) == 1
        assert data["segments"][0]["text"] == "good"
        assert data["_filter_stats"]["dropped_segments"] == 1

    def test_empty_segments(self, tmp_path):
        transcript = {"audio_path": "/audio/test.wav", "segments": []}
        jf = tmp_path / "test.json"
        jf.write_text(json.dumps(transcript))

        original, kept, data = filter_transcript_file(str(jf), -0.7, 0.3, 2.4)
        assert original == 0
        assert kept == 0

    def test_preserves_audio_path(self, tmp_path):
        transcript = {
            "audio_path": "/audio/sermon.wav",
            "segments": [{"text": "hello", "avg_logprob": -0.2}],
        }
        jf = tmp_path / "test.json"
        jf.write_text(json.dumps(transcript))

        _, _, data = filter_transcript_file(str(jf), -0.7, 0.3, 2.4)
        assert data["audio_path"] == "/audio/sermon.wav"


# ===================================================================
# filter_transcripts (directory-level)
# ===================================================================


class TestFilterTranscripts:
    def _make_transcripts(self, tmp_path):
        """Create test transcript files."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()

        # File with mixed segments
        t1 = {
            "audio_path": "/audio/sermon1.wav",
            "segments": [
                {"text": "good1", "avg_logprob": -0.2},
                {"text": "bad1", "avg_logprob": -0.9},
            ],
        }
        (input_dir / "sermon1.json").write_text(json.dumps(t1))

        # File with all good segments
        t2 = {
            "audio_path": "/audio/sermon2.wav",
            "segments": [
                {"text": "good2", "avg_logprob": -0.3},
            ],
        }
        (input_dir / "sermon2.json").write_text(json.dumps(t2))

        # Underscore file (should be skipped)
        t3 = {"segments": [{"text": "skip"}]}
        (input_dir / "_summary.json").write_text(json.dumps(t3))

        return str(input_dir)

    def test_filters_directory(self, tmp_path):
        input_dir = self._make_transcripts(tmp_path)
        output_dir = str(tmp_path / "output")

        result = filter_transcripts(input_dir, output_dir, min_logprob=-0.7)
        assert result["total_segments"] == 3
        assert result["kept"] == 2
        assert result["dropped"] == 1

    def test_dry_run(self, tmp_path):
        input_dir = self._make_transcripts(tmp_path)
        output_dir = str(tmp_path / "output")

        result = filter_transcripts(input_dir, output_dir, min_logprob=-0.7, dry_run=True)
        assert result["dropped"] == 1
        # Output dir should not be created in dry run
        assert not (tmp_path / "output").exists()

    def test_output_files_created(self, tmp_path):
        input_dir = self._make_transcripts(tmp_path)
        output_dir = tmp_path / "output"

        filter_transcripts(str(input_dir), str(output_dir), min_logprob=-0.7)
        assert (output_dir / "sermon1.json").exists()
        assert (output_dir / "sermon2.json").exists()
        # Underscore file should not be in output
        assert not (output_dir / "_summary.json").exists()

    def test_empty_after_filter_not_written(self, tmp_path):
        """Files with all segments dropped should not be written."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"

        t = {"segments": [{"text": "bad", "avg_logprob": -1.5}]}
        (input_dir / "bad.json").write_text(json.dumps(t))

        result = filter_transcripts(str(input_dir), str(output_dir), min_logprob=-0.7)
        assert result["files_empty_after_filter"] == 1
        assert not (output_dir / "bad.json").exists()

    def test_empty_input_dir(self, tmp_path):
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = str(tmp_path / "output")

        result = filter_transcripts(str(input_dir), output_dir)
        assert result is None


# ===================================================================
# Default thresholds
# ===================================================================


class TestDefaults:
    def test_default_min_logprob(self):
        assert DEFAULT_MIN_LOGPROB == -0.7

    def test_default_max_no_speech(self):
        assert DEFAULT_MAX_NO_SPEECH == 0.3

    def test_default_max_compression(self):
        assert DEFAULT_MAX_COMPRESSION == 2.4
