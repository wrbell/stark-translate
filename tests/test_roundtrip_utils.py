"""Tests for tools/roundtrip_test.py helpers (no TTS/STT)."""

from __future__ import annotations

import os
import sys
import wave
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# roundtrip_test imports scipy.signal at module level; ensure submodule exists
# before import (conftest mocks top-level scipy as a bare MagicMock).
_scipy_signal = MagicMock()
_scipy_signal.resample = MagicMock(side_effect=lambda audio, n: np.ones(n, dtype=np.float32))
sys.modules["scipy.signal"] = _scipy_signal

from tools import roundtrip_test as rt  # noqa: E402

# Drop the fake submodule so other tests (audio_bridge) can importorskip real scipy.
del sys.modules["scipy.signal"]


class TestChunkText:
    def test_short_single_chunk(self):
        text = "Hello world. " * 5
        chunks = rt.chunk_text(text, target_words=200)
        assert len(chunks) == 1

    def test_paragraph_split(self):
        paras = ["\n\n".join(["word " * 50 for _ in range(6)])]
        text = paras[0]
        chunks = rt.chunk_text(text, target_words=80)
        assert len(chunks) >= 2

    def test_sentence_fallback(self):
        # Single huge paragraph without double newlines
        sentences = ". ".join([f"Sentence number {i} with several words" for i in range(40)])
        chunks = rt.chunk_text(sentences, target_words=30)
        assert len(chunks) >= 2


class TestLoadDocuments:
    def test_explicit_files(self, tmp_path: Path):
        a = tmp_path / "a.txt"
        a.write_text("alpha")
        missing = tmp_path / "gone.txt"
        docs = rt.load_documents([str(a), str(missing)], tmp_path / "texts")
        assert docs == {"a": "alpha"}

    def test_discover_dir(self, tmp_path: Path):
        texts = tmp_path / "texts"
        texts.mkdir()
        (texts / "sermon.txt").write_text("grace and mercy")
        docs = rt.load_documents(None, texts)
        assert docs["sermon"] == "grace and mercy"

    def test_builtin_fallback(self, tmp_path: Path):
        docs = rt.load_documents(None, tmp_path / "empty")
        assert "sample" in docs
        assert len(docs["sample"]) > 20


class TestWerDiffReport:
    def test_compute_wer_empty(self, monkeypatch: pytest.MonkeyPatch):
        # jiwer is mocked; provide predictable behavior
        fake = MagicMock()
        fake.wer = MagicMock(return_value=0.25)
        monkeypatch.setitem(sys.modules, "jiwer", fake)
        assert rt.compute_wer("", "") == 0.0
        assert rt.compute_wer("", "x") == 1.0
        assert rt.compute_wer("a b", "a c") == 0.25

    def test_word_diff(self):
        d = rt.word_diff("grace and mercy", "grace and love")
        assert "original" in d or "mercy" in d or "love" in d

    def test_generate_report(self, tmp_path: Path):
        results = {
            "doc1": [
                {
                    "chunk_idx": 0,
                    "stt_wer": 0.1,
                    "roundtrip_wer": 0.2,
                    "tts_en_ms": 10,
                    "stt_en_ms": 20,
                    "trans_es_ms": 30,
                    "tts_es_ms": 40,
                    "stt_es_ms": 50,
                    "trans_en_ms": 60,
                    "original": "grace and mercy abound",
                    "backtranslated": "grace and love abound",
                },
                {
                    "chunk_idx": 1,
                    "stt_wer": 0.0,
                    "roundtrip_wer": 0.05,
                    "tts_en_ms": 1,
                    "stt_en_ms": 2,
                    "trans_es_ms": 3,
                    "tts_es_ms": 4,
                    "stt_es_ms": 5,
                    "trans_en_ms": 6,
                    "original": "hello world",
                    "backtranslated": "hello world",
                },
            ]
        }
        report = rt.generate_report(results, tmp_path, total_elapsed=12.5)
        assert "ROUNDTRIP QUALITY REPORT" in report
        assert "COMBINED" in report
        assert (tmp_path / "roundtrip_report.txt").exists()


class TestSaveWavAndResample:
    def test_save_wav(self, tmp_path: Path):
        path = str(tmp_path / "x.wav")
        rt.save_wav(np.array([0.1, -0.1], dtype=np.float32), 16000, path)
        with wave.open(path, "rb") as wf:
            assert wf.getnframes() == 2

    def test_resample_same(self):
        a = np.ones(5, dtype=np.float32)
        assert rt.resample_audio(a, 16, 16) is a

    def test_resample_different(self, monkeypatch: pytest.MonkeyPatch):
        # roundtrip_test imports resample at module level from scipy
        monkeypatch.setattr(rt, "resample", lambda audio, n: np.ones(n, dtype=np.float32))
        out = rt.resample_audio(np.ones(10, dtype=np.float32), 8000, 16000)
        assert len(out) == 20
