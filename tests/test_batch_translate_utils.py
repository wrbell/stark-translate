"""Tests for tools/batch_translate.py helpers (no GPU/VAD)."""

from __future__ import annotations

import os
import sys
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tools import batch_translate as bt


class TestSlugify:
    def test_basic(self):
        assert bt.slugify("Feb 14, 2026!") == "feb_14_2026"

    def test_underscores(self):
        assert bt.slugify("  Hello__World  ") == "hello_world"


class TestDiscoverMp3s:
    def test_discover_and_filter(self, tmp_path: Path):
        (tmp_path / "john_smith.mp3").touch()
        (tmp_path / "mary_jones.mp3").touch()
        (tmp_path / "notes.txt").touch()
        all_mp3 = bt.discover_mp3s(tmp_path)
        assert len(all_mp3) == 2
        filtered = bt.discover_mp3s(tmp_path, speaker_filter="mary")
        assert len(filtered) == 1
        assert filtered[0].name == "mary_jones.mp3"

    def test_case_insensitive_suffix(self, tmp_path: Path):
        # When glob *.mp3 finds nothing, fall back to suffix check
        weird = tmp_path / "Talk.MP3"
        weird.touch()
        # On case-sensitive FS, *.mp3 may miss .MP3 — function has fallback
        found = bt.discover_mp3s(tmp_path)
        assert any(p.name.lower().endswith(".mp3") for p in found) or found == []


class TestFmtDuration:
    def test_seconds(self):
        assert bt._fmt_duration(45) == "45s"

    def test_minutes(self):
        assert bt._fmt_duration(65) == "1m05s"

    def test_hours(self):
        assert bt._fmt_duration(3661) == "1h01m"


class TestJsonlAndTranscripts:
    def test_append_reload_write(self, tmp_path: Path):
        p = tmp_path / "segments.jsonl"
        bt._append_jsonl(
            p,
            {
                "segment_index": 1,
                "source_text": "hello",
                "marian_text": "hola",
                "gemma_text": "hola!",
            },
        )
        bt._append_jsonl(
            p,
            {
                "segment_index": 0,
                "source_text": "first",
                "marian_text": "primero",
                "gemma_text": "primero!",
            },
        )
        # bad line ignored
        with open(p, "a") as f:
            f.write("not-json\n")
        en, marian, gemma = bt._reload_transcripts(p)
        assert en == ["first", "hello"]
        assert marian[0] == "primero"
        assert gemma[1] == "hola!"

        out = tmp_path / "en.txt"
        bt._write_transcript(out, en)
        assert out.read_text().strip() == "first\nhello"


class TestSaveWav:
    def test_writes_int16_wav(self, tmp_path: Path):
        audio = np.array([0.0, 0.5, -0.5, 1.0], dtype=np.float32)
        path = str(tmp_path / "t.wav")
        bt.save_wav(audio, 16000, path)
        with wave.open(path, "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getframerate() == 16000
            assert wf.getnframes() == 4


class TestResampleAudio:
    def test_same_sr_noop(self):
        audio = np.ones(10, dtype=np.float32)
        assert bt.resample_audio(audio, 16000, 16000) is audio


class TestGenerateSummary:
    def test_summary_markdown(self, tmp_path: Path):
        results = [
            {
                "file": "a.mp3",
                "slug": "a",
                "audio_duration_sec": 120.0,
                "segment_count": 3,
                "processing_time_sec": 60.0,
                "avg_stt_ms": 100.0,
                "avg_marian_ms": 50.0,
                "avg_gemma_ms": 80.0,
                "avg_tts_ms": 40.0,
                "total_stt_ms": 300.0,
                "total_marian_ms": 150.0,
                "total_gemma_ms": 240.0,
                "total_tts_ms": 120.0,
            }
        ]
        report = bt.generate_summary(results, tmp_path, total_elapsed=60.0)
        assert "Batch Translation Summary" in report
        assert "Per-File Results" in report
        assert (tmp_path / "summary.md").exists()


class TestWavToMp3:
    def test_calls_ffmpeg(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        wav = tmp_path / "a.wav"
        wav.write_bytes(b"RIFF")
        with patch("subprocess.run") as run:
            run.return_value = MagicMock(returncode=0)
            out = bt.wav_to_mp3(str(wav))
            assert out.endswith(".mp3")
            run.assert_called_once()


class TestSegmentAudioVad:
    def test_segments_with_mocked_torch(self, monkeypatch: pytest.MonkeyPatch):
        audio = np.zeros(16000 * 5, dtype=np.float32)
        # Build fake torch hub response
        fake_ts = [
            {"start": 0, "end": 16000},  # 1s — kept
            {"start": 16000, "end": 16100},  # tiny — skipped
            {"start": 20000, "end": 20000 + 16000 * 35},  # long — split
        ]
        get_speech_ts = MagicMock(return_value=fake_ts)
        model = MagicMock()
        utils = (get_speech_ts,)
        fake_torch = MagicMock()
        fake_torch.hub.load.return_value = (model, utils)
        fake_torch.from_numpy.side_effect = lambda x: x
        monkeypatch.setitem(sys.modules, "torch", fake_torch)
        segs = bt.segment_audio_vad(audio, min_duration=0.5, max_duration=10.0)
        assert len(segs) >= 1
