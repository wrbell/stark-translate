"""Tests for training/assess_quality.py — smart sampling, interactive review, cross-check."""

import json
from unittest.mock import MagicMock, patch

from training.assess_quality import (
    CONFIDENCE_THRESHOLDS,
    _atomic_write_jsonl,
    _extract_segments,
    classify_segment,
    interactive_review,
    smart_sample_for_assessment,
)

# ===================================================================
# classify_segment
# ===================================================================


class TestClassifySegment:
    def test_high_confidence(self):
        seg = {"avg_logprob": -0.2, "no_speech_prob": 0.05, "compression_ratio": 1.5}
        assert classify_segment(seg) == "high"

    def test_medium_avg_logprob(self):
        # avg_logprob between good (-0.3) and flag (-0.5)
        seg = {"avg_logprob": -0.4, "no_speech_prob": 0.05, "compression_ratio": 1.5}
        assert classify_segment(seg) == "medium"

    def test_medium_no_speech_prob(self):
        seg = {"avg_logprob": -0.2, "no_speech_prob": 0.2, "compression_ratio": 1.5}
        assert classify_segment(seg) == "medium"

    def test_medium_compression_ratio(self):
        seg = {"avg_logprob": -0.2, "no_speech_prob": 0.05, "compression_ratio": 1.9}
        assert classify_segment(seg) == "medium"

    def test_low_avg_logprob(self):
        # avg_logprob in flag range: < -0.5 but > -1.0
        seg = {"avg_logprob": -0.7, "no_speech_prob": 0.05, "compression_ratio": 1.5}
        assert classify_segment(seg) == "low"

    def test_low_no_speech_prob(self):
        seg = {"avg_logprob": -0.2, "no_speech_prob": 0.4, "compression_ratio": 1.5}
        assert classify_segment(seg) == "low"

    def test_low_compression_ratio(self):
        seg = {"avg_logprob": -0.2, "no_speech_prob": 0.05, "compression_ratio": 2.1}
        assert classify_segment(seg) == "low"

    def test_reject_avg_logprob(self):
        seg = {"avg_logprob": -1.5, "no_speech_prob": 0.05, "compression_ratio": 1.5}
        assert classify_segment(seg) == "reject"

    def test_reject_no_speech_prob(self):
        seg = {"avg_logprob": -0.2, "no_speech_prob": 0.7, "compression_ratio": 1.5}
        assert classify_segment(seg) == "reject"

    def test_reject_compression_ratio(self):
        seg = {"avg_logprob": -0.2, "no_speech_prob": 0.05, "compression_ratio": 2.5}
        assert classify_segment(seg) == "reject"

    def test_all_fields_missing_returns_medium(self):
        seg = {"text": "hello world"}
        assert classify_segment(seg) == "medium"

    def test_partial_fields_missing(self):
        # Only avg_logprob present and good — should be high
        seg = {"avg_logprob": -0.2}
        assert classify_segment(seg) == "high"

    def test_reject_overrides_low(self):
        # One metric reject, another low — reject wins
        seg = {"avg_logprob": -1.5, "no_speech_prob": 0.4, "compression_ratio": 1.5}
        assert classify_segment(seg) == "reject"

    def test_low_overrides_medium(self):
        # One metric low, another medium — low wins
        seg = {"avg_logprob": -0.7, "no_speech_prob": 0.2, "compression_ratio": 1.5}
        assert classify_segment(seg) == "low"

    def test_boundary_good_avg_logprob(self):
        # Exactly at good threshold
        seg = {"avg_logprob": -0.3, "no_speech_prob": 0.05, "compression_ratio": 1.5}
        assert classify_segment(seg) == "high"

    def test_boundary_flag_avg_logprob(self):
        # Exactly at flag threshold
        seg = {"avg_logprob": -0.5, "no_speech_prob": 0.05, "compression_ratio": 1.5}
        assert classify_segment(seg) == "medium"

    def test_boundary_reject_avg_logprob(self):
        # Exactly at reject threshold
        seg = {"avg_logprob": -1.0, "no_speech_prob": 0.05, "compression_ratio": 1.5}
        assert classify_segment(seg) == "low"


# ===================================================================
# _extract_segments
# ===================================================================


class TestExtractSegments:
    def test_extracts_from_json(self, tmp_path):
        transcript = {
            "audio_path": "/audio/test.wav",
            "segments": [
                {
                    "timestamp": [0.0, 5.0],
                    "text": " Hello world ",
                    "avg_logprob": -0.2,
                    "no_speech_prob": 0.05,
                    "compression_ratio": 1.5,
                },
            ],
        }
        (tmp_path / "test.json").write_text(json.dumps(transcript))

        result = _extract_segments(str(tmp_path))
        assert len(result) == 1
        assert result[0]["audio_file"] == "/audio/test.wav"
        assert result[0]["start"] == 0.0
        assert result[0]["end"] == 5.0
        assert result[0]["auto_text"] == "Hello world"
        assert result[0]["avg_logprob"] == -0.2

    def test_skips_underscore_files(self, tmp_path):
        (tmp_path / "_summary.json").write_text(json.dumps({"segments": [{"text": "x"}]}))
        result = _extract_segments(str(tmp_path))
        assert len(result) == 0

    def test_missing_confidence_fields(self, tmp_path):
        transcript = {
            "segments": [{"timestamp": [1.0, 2.0], "text": "hi"}],
        }
        (tmp_path / "test.json").write_text(json.dumps(transcript))
        result = _extract_segments(str(tmp_path))
        assert len(result) == 1
        assert "avg_logprob" not in result[0]

    def test_empty_dir(self, tmp_path):
        result = _extract_segments(str(tmp_path))
        assert result == []


# ===================================================================
# smart_sample_for_assessment
# ===================================================================


class TestSmartSample:
    def _make_transcripts(self, tmp_path):
        """Create synthetic transcripts with known confidence distributions."""
        segments = []
        # 10 high-confidence segments
        for i in range(10):
            segments.append(
                {
                    "timestamp": [i * 5.0, (i + 1) * 5.0],
                    "text": f"high segment {i}",
                    "avg_logprob": -0.2,
                    "no_speech_prob": 0.05,
                    "compression_ratio": 1.5,
                }
            )
        # 6 medium-confidence segments
        for i in range(6):
            segments.append(
                {
                    "timestamp": [50.0 + i * 5.0, 55.0 + i * 5.0],
                    "text": f"medium segment {i}",
                    "avg_logprob": -0.4,
                    "no_speech_prob": 0.05,
                    "compression_ratio": 1.5,
                }
            )
        # 4 low-confidence segments
        for i in range(4):
            segments.append(
                {
                    "timestamp": [80.0 + i * 5.0, 85.0 + i * 5.0],
                    "text": f"low segment {i}",
                    "avg_logprob": -0.7,
                    "no_speech_prob": 0.05,
                    "compression_ratio": 1.5,
                }
            )
        # 2 reject segments
        for i in range(2):
            segments.append(
                {
                    "timestamp": [100.0 + i * 5.0, 105.0 + i * 5.0],
                    "text": f"reject segment {i}",
                    "avg_logprob": -1.5,
                    "no_speech_prob": 0.05,
                    "compression_ratio": 1.5,
                }
            )

        transcript = {"audio_path": "/audio/test.wav", "segments": segments}
        (tmp_path / "test.json").write_text(json.dumps(transcript))

    def test_bucket_distribution(self, tmp_path):
        self._make_transcripts(tmp_path)
        output = tmp_path / "sample.jsonl"
        result = smart_sample_for_assessment(str(tmp_path), n=20, output_path=str(output), seed=42)

        buckets = {}
        for s in result:
            b = s["confidence_bucket"]
            buckets[b] = buckets.get(b, 0) + 1

        # All 4 low segments should be included
        assert buckets.get("low", 0) == 4
        # No reject segments
        assert "reject" not in buckets

    def test_output_fields(self, tmp_path):
        self._make_transcripts(tmp_path)
        output = tmp_path / "sample.jsonl"
        result = smart_sample_for_assessment(str(tmp_path), n=10, output_path=str(output), seed=42)

        for entry in result:
            assert "confidence_bucket" in entry
            assert "reviewed" in entry
            assert "manual_text" in entry
            assert "auto_text" in entry
            assert entry["confidence_bucket"] in ("high", "medium", "low")

    def test_high_segments_auto_accepted(self, tmp_path):
        self._make_transcripts(tmp_path)
        output = tmp_path / "sample.jsonl"
        result = smart_sample_for_assessment(str(tmp_path), n=20, output_path=str(output), seed=42)

        for entry in result:
            if entry["confidence_bucket"] == "high":
                assert entry["reviewed"] is True
                assert entry["manual_text"] == entry["auto_text"]

    def test_low_segments_unreviewed(self, tmp_path):
        self._make_transcripts(tmp_path)
        output = tmp_path / "sample.jsonl"
        result = smart_sample_for_assessment(str(tmp_path), n=20, output_path=str(output), seed=42)

        for entry in result:
            if entry["confidence_bucket"] == "low":
                assert entry["reviewed"] is False
                assert entry["manual_text"] == ""

    def test_writes_valid_jsonl(self, tmp_path):
        self._make_transcripts(tmp_path)
        output = tmp_path / "sample.jsonl"
        smart_sample_for_assessment(str(tmp_path), n=10, output_path=str(output), seed=42)

        with open(output) as f:
            lines = f.readlines()
        assert len(lines) == 10
        for line in lines:
            parsed = json.loads(line)
            assert isinstance(parsed, dict)

    def test_n_larger_than_available(self, tmp_path):
        self._make_transcripts(tmp_path)
        output = tmp_path / "sample.jsonl"
        # 22 total segments (10 high + 6 medium + 4 low + 2 reject)
        # After removing reject: 20 available
        result = smart_sample_for_assessment(str(tmp_path), n=100, output_path=str(output), seed=42)
        assert len(result) <= 20

    def test_empty_dir(self, tmp_path):
        output = tmp_path / "sample.jsonl"
        result = smart_sample_for_assessment(str(tmp_path), n=10, output_path=str(output))
        assert result == []


# ===================================================================
# interactive_review
# ===================================================================


class TestInteractiveReview:
    def _make_review_jsonl(self, tmp_path, entries):
        path = tmp_path / "review.jsonl"
        with open(path, "w") as f:
            for e in entries:
                f.write(json.dumps(e) + "\n")
        return str(path)

    @patch("training.assess_quality.play_audio_clip")
    @patch("builtins.input")
    def test_accept_with_enter(self, mock_input, mock_play, tmp_path):
        entries = [
            {
                "audio_file": "/audio/test.wav",
                "start": 0.0,
                "end": 5.0,
                "auto_text": "hello world",
                "manual_text": "",
                "reviewed": False,
                "confidence_bucket": "low",
            }
        ]
        path = self._make_review_jsonl(tmp_path, entries)
        mock_input.return_value = ""  # Enter = accept

        result = interactive_review(path)
        assert result[0]["reviewed"] is True
        assert result[0]["manual_text"] == "hello world"

    @patch("training.assess_quality.play_audio_clip")
    @patch("builtins.input")
    def test_correct_with_typed_text(self, mock_input, mock_play, tmp_path):
        entries = [
            {
                "audio_file": "/audio/test.wav",
                "start": 0.0,
                "end": 5.0,
                "auto_text": "hello word",
                "manual_text": "",
                "reviewed": False,
                "confidence_bucket": "low",
            }
        ]
        path = self._make_review_jsonl(tmp_path, entries)
        mock_input.return_value = "hello world"

        result = interactive_review(path)
        assert result[0]["reviewed"] is True
        assert result[0]["manual_text"] == "hello world"

    @patch("training.assess_quality.play_audio_clip")
    @patch("builtins.input")
    def test_quit_saves(self, mock_input, mock_play, tmp_path):
        entries = [
            {
                "audio_file": "/audio/test.wav",
                "start": 0.0,
                "end": 5.0,
                "auto_text": "seg1",
                "manual_text": "",
                "reviewed": False,
                "confidence_bucket": "low",
            },
            {
                "audio_file": "/audio/test.wav",
                "start": 5.0,
                "end": 10.0,
                "auto_text": "seg2",
                "manual_text": "",
                "reviewed": False,
                "confidence_bucket": "medium",
            },
        ]
        path = self._make_review_jsonl(tmp_path, entries)
        mock_input.return_value = "/q"

        interactive_review(path)

        # File should be saved
        with open(path) as f:
            saved = [json.loads(line) for line in f]
        assert len(saved) == 2
        # Neither should be reviewed (quit before reviewing)
        assert saved[0]["reviewed"] is False
        assert saved[1]["reviewed"] is False

    @patch("training.assess_quality.play_audio_clip")
    @patch("builtins.input")
    def test_skip_leaves_unreviewed(self, mock_input, mock_play, tmp_path):
        entries = [
            {
                "audio_file": "/audio/test.wav",
                "start": 0.0,
                "end": 5.0,
                "auto_text": "hello",
                "manual_text": "",
                "reviewed": False,
                "confidence_bucket": "low",
            }
        ]
        path = self._make_review_jsonl(tmp_path, entries)
        mock_input.return_value = "/s"

        result = interactive_review(path)
        assert result[0]["reviewed"] is False

    @patch("training.assess_quality.play_audio_clip")
    @patch("builtins.input")
    def test_replay_then_accept(self, mock_input, mock_play, tmp_path):
        entries = [
            {
                "audio_file": "/audio/test.wav",
                "start": 0.0,
                "end": 5.0,
                "auto_text": "hello",
                "manual_text": "",
                "reviewed": False,
                "confidence_bucket": "low",
            }
        ]
        path = self._make_review_jsonl(tmp_path, entries)
        mock_input.side_effect = ["/r", ""]  # Replay then accept

        result = interactive_review(path)
        assert result[0]["reviewed"] is True
        # play_audio_clip called twice: initial + replay
        assert mock_play.call_count == 2

    @patch("training.assess_quality.play_audio_clip")
    @patch("builtins.input")
    def test_already_reviewed_skipped(self, mock_input, mock_play, tmp_path):
        entries = [
            {
                "audio_file": "/audio/test.wav",
                "start": 0.0,
                "end": 5.0,
                "auto_text": "hello",
                "manual_text": "hello",
                "reviewed": True,
                "confidence_bucket": "high",
            }
        ]
        path = self._make_review_jsonl(tmp_path, entries)

        interactive_review(path)
        # input() should never be called
        mock_input.assert_not_called()


# ===================================================================
# _atomic_write_jsonl
# ===================================================================


class TestAtomicWriteJsonl:
    def test_writes_and_replaces(self, tmp_path):
        path = tmp_path / "test.jsonl"
        entries = [{"a": 1}, {"b": 2}]
        _atomic_write_jsonl(str(path), entries)

        with open(path) as f:
            lines = f.readlines()
        assert len(lines) == 2
        assert json.loads(lines[0]) == {"a": 1}

    def test_no_tmp_file_left(self, tmp_path):
        path = tmp_path / "test.jsonl"
        _atomic_write_jsonl(str(path), [{"x": 1}])
        assert not (tmp_path / "test.jsonl.tmp").exists()


# ===================================================================
# cross_check (mocked)
# ===================================================================


class TestCrossCheck:
    def test_cross_check_wer_computation(self, tmp_path):
        """Test cross_check with mocked faster-whisper and soundfile."""
        import sys

        import numpy as np

        # Create transcript JSON
        transcript = {
            "audio_path": "/audio/test.wav",
            "segments": [
                {"timestamp": [0.0, 5.0], "text": "hello world", "avg_logprob": -0.2},
                {"timestamp": [5.0, 10.0], "text": "good morning", "avg_logprob": -0.3},
            ],
        }
        (tmp_path / "test.json").write_text(json.dumps(transcript))
        output = tmp_path / "cross.jsonl"

        # Mock segment result from faster-whisper
        mock_seg = MagicMock()
        mock_seg.text = " hello world"

        mock_model = MagicMock()
        mock_model.transcribe.side_effect = lambda *a, **kw: (iter([mock_seg]), None)

        # Create mock modules for faster_whisper and torch
        mock_fw = MagicMock()
        mock_fw.WhisperModel.return_value = mock_model

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False

        mock_sf = MagicMock()
        mock_sf.read.return_value = (np.zeros(16000, dtype=np.float32), 16000)

        # Temporarily restore real jiwer (conftest mocks it) and mock heavy deps
        saved = {}
        for mod_name in ("faster_whisper", "torch", "soundfile", "jiwer"):
            saved[mod_name] = sys.modules.get(mod_name)
        sys.modules["faster_whisper"] = mock_fw
        sys.modules["torch"] = mock_torch
        sys.modules["soundfile"] = mock_sf
        # Re-import real jiwer — conftest installs a MagicMock
        if "jiwer" in sys.modules:
            del sys.modules["jiwer"]
        import jiwer

        sys.modules["jiwer"] = jiwer

        try:
            from training.assess_quality import cross_check

            results = cross_check(
                str(tmp_path),
                n=2,
                model_name="openai/whisper-large-v3",
                output_path=str(output),
                seed=42,
            )
        finally:
            for mod_name, original in saved.items():
                if original is None:
                    sys.modules.pop(mod_name, None)
                else:
                    sys.modules[mod_name] = original

        assert len(results) == 2
        for r in results:
            assert "agreement_wer" in r
            assert "original_text" in r
            assert "cross_text" in r
            assert isinstance(r["agreement_wer"], float)

        # Output file should exist with valid JSONL
        assert output.exists()
        with open(output) as f:
            lines = f.readlines()
        assert len(lines) == 2


# ===================================================================
# CONFIDENCE_THRESHOLDS constant
# ===================================================================


class TestConfidenceThresholds:
    def test_all_metrics_present(self):
        assert "avg_logprob" in CONFIDENCE_THRESHOLDS
        assert "no_speech_prob" in CONFIDENCE_THRESHOLDS
        assert "compression_ratio" in CONFIDENCE_THRESHOLDS

    def test_all_tiers_present(self):
        for metric in CONFIDENCE_THRESHOLDS:
            assert "good" in CONFIDENCE_THRESHOLDS[metric]
            assert "flag" in CONFIDENCE_THRESHOLDS[metric]
            assert "reject" in CONFIDENCE_THRESHOLDS[metric]
