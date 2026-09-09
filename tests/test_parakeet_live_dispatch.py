"""dry_run_ab dispatches STT to an STTEngine (Parakeet-MLX) when stt_pipe is an engine."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

import dry_run_ab
from engines.base import STTEngine, STTResult


class _FakeEngine(STTEngine):
    def __init__(self):
        self.calls = []

    def load(self):
        pass

    def unload(self):
        pass

    @property
    def model_id(self):
        return "fake"

    @property
    def backend(self):
        return "fake"

    def transcribe(self, audio, *, language="en", initial_prompt=None, word_timestamps=False, beam_size=None):
        self.calls.append({"language": language, "initial_prompt": initial_prompt, "word_timestamps": word_timestamps})
        return STTResult(
            text="  Grace and peace.  ",
            latency_ms=87.5,
            confidence=0.983,
            avg_logprob=-0.017,
            compression_ratio=0.9,
            segments=[{"avg_logprob": -0.017, "no_speech_prob": 0.0, "compression_ratio": 0.9}],
            low_confidence_words=[{"word": "peace", "probability": 0.4}],
        )


def test_run_stt_mlx_dispatches_to_engine(monkeypatch):
    eng = _FakeEngine()
    monkeypatch.setattr(dry_run_ab, "stt_pipe", eng)
    monkeypatch.setattr(dry_run_ab, "BACKEND", "mlx")
    monkeypatch.setattr(dry_run_ab, "SOURCE_LANG", "en")
    monkeypatch.setattr(dry_run_ab, "WORD_TIMESTAMPS", False)
    text, lat, conf, segs, low = dry_run_ab._run_stt_mlx(np.zeros(16000, dtype=np.float32), "prompt")
    assert text == "Grace and peace."
    assert lat == 87.5
    assert conf == 0.98
    assert segs and segs[0]["avg_logprob"] == -0.017
    assert low and low[0]["word"] == "peace"
    assert eng.calls[0] == {"language": "en", "initial_prompt": "prompt", "word_timestamps": False}


def test_run_stt_mlx_still_uses_whisper_for_model_id_string(monkeypatch):
    fake_whisper = MagicMock()
    fake_whisper.transcribe.return_value = {"text": " hola ", "segments": [{"avg_logprob": -0.2}]}
    monkeypatch.setattr(dry_run_ab, "mlx_whisper", fake_whisper, raising=False)
    monkeypatch.setattr(dry_run_ab, "stt_pipe", "mlx-community/whisper-large-v3-turbo")
    monkeypatch.setattr(dry_run_ab, "BACKEND", "mlx")
    monkeypatch.setattr(dry_run_ab, "SOURCE_LANG", "es")
    text, _lat, conf, _segs, _low = dry_run_ab._run_stt_mlx(np.zeros(16000, dtype=np.float32), None)
    assert text == "hola"
    assert conf == 0.8
    assert fake_whisper.transcribe.call_args.kwargs["path_or_hf_repo"] == "mlx-community/whisper-large-v3-turbo"


def test_load_whisper_returns_parakeet_engine_when_selected(monkeypatch):
    created = {}

    class _Eng:
        def __init__(self, model_id):
            created["model_id"] = model_id
            self.model_id = model_id

        def load(self):
            created["loaded"] = True

    import types

    fake_mod = types.SimpleNamespace(ParakeetMLXEngine=_Eng)
    monkeypatch.setitem(__import__("sys").modules, "engines.parakeet_mlx_engine", fake_mod)
    monkeypatch.setattr(dry_run_ab.settings.stt, "backend", "parakeet-mlx")
    monkeypatch.setattr(
        dry_run_ab.settings.stt, "parakeet_mlx_model", "mlx-community/parakeet-tdt-0.6b-v3", raising=False
    )
    engine = dry_run_ab.load_whisper("mlx")
    assert isinstance(engine, _Eng)
    assert created == {"model_id": "mlx-community/parakeet-tdt-0.6b-v3", "loaded": True}


def test_auto_backend_prefers_parakeet_for_english_when_installed(monkeypatch):
    import sys
    import types

    monkeypatch.setitem(sys.modules, "parakeet_mlx", types.ModuleType("parakeet_mlx"))
    monkeypatch.setattr(dry_run_ab.settings.stt, "backend", "auto")
    monkeypatch.setattr(dry_run_ab, "SOURCE_LANG", "en")
    assert dry_run_ab._resolve_mlx_stt_backend() == "parakeet-mlx"
    monkeypatch.setattr(dry_run_ab, "SOURCE_LANG", "es")
    assert dry_run_ab._resolve_mlx_stt_backend() == "mlx"


def test_auto_backend_falls_back_to_whisper_without_parakeet(monkeypatch):
    import builtins
    import sys

    monkeypatch.delitem(sys.modules, "parakeet_mlx", raising=False)
    real_import = builtins.__import__

    def _imp(name, *a, **k):
        if name == "parakeet_mlx":
            raise ImportError("not installed")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", _imp)
    monkeypatch.setattr(dry_run_ab.settings.stt, "backend", "auto")
    monkeypatch.setattr(dry_run_ab, "SOURCE_LANG", "en")
    assert dry_run_ab._resolve_mlx_stt_backend() == "mlx"


def test_explicit_backend_wins(monkeypatch):
    monkeypatch.setattr(dry_run_ab.settings.stt, "backend", "mlx")
    monkeypatch.setattr(dry_run_ab, "SOURCE_LANG", "en")
    assert dry_run_ab._resolve_mlx_stt_backend() == "mlx"
