"""Tests for HF Whisper engine with speculative decoding.

All tests mock HF transformers — runs in CI without GPU or model downloads.
"""

from unittest.mock import patch


class TestHFWhisperEngine:
    """Test HFWhisperEngine construction and configuration."""

    def test_default_draft_model(self):
        from engines.hf_whisper_engine import HFWhisperEngine

        with patch("engines.hf_whisper_engine.HF_WHISPER_AVAILABLE", True):
            engine = HFWhisperEngine()
        assert engine._model_id == "openai/whisper-large-v3-turbo"
        assert engine._draft_model_id == "distil-whisper/distil-large-v3.5"

    def test_no_draft_model(self):
        from engines.hf_whisper_engine import HFWhisperEngine

        with patch("engines.hf_whisper_engine.HF_WHISPER_AVAILABLE", True):
            engine = HFWhisperEngine(draft_model_id=None)
        assert engine._draft_model_id is None

    def test_custom_models(self):
        from engines.hf_whisper_engine import HFWhisperEngine

        with patch("engines.hf_whisper_engine.HF_WHISPER_AVAILABLE", True):
            engine = HFWhisperEngine(
                model_id="openai/whisper-large-v3",
                draft_model_id="openai/whisper-large-v3-turbo",
            )
        assert engine._model_id == "openai/whisper-large-v3"
        assert engine._draft_model_id == "openai/whisper-large-v3-turbo"

    def test_backend_property(self):
        from engines.hf_whisper_engine import HFWhisperEngine

        with patch("engines.hf_whisper_engine.HF_WHISPER_AVAILABLE", True):
            engine = HFWhisperEngine()
        assert engine.backend == "hf"

    def test_transcribe_before_load_raises(self):
        import numpy as np
        import pytest

        from engines.hf_whisper_engine import HFWhisperEngine

        with patch("engines.hf_whisper_engine.HF_WHISPER_AVAILABLE", True):
            engine = HFWhisperEngine()
            with pytest.raises(RuntimeError, match="not loaded"):
                engine.transcribe(np.zeros(16000))


class TestFactorySpecDecode:
    """Test that factory routes to HFWhisperEngine when spec_decode=True."""

    def test_spec_decode_creates_hf_engine(self):
        from engines.factory import create_stt_engine

        with patch("engines.hf_whisper_engine.HF_WHISPER_AVAILABLE", True):
            engine = create_stt_engine(backend="cuda", spec_decode=True)
        assert type(engine).__name__ == "HFWhisperEngine"
        assert engine._draft_model_id == "distil-whisper/distil-large-v3.5"

    def test_spec_decode_custom_draft(self):
        from engines.factory import create_stt_engine

        with patch("engines.hf_whisper_engine.HF_WHISPER_AVAILABLE", True):
            engine = create_stt_engine(
                backend="cuda",
                spec_decode=True,
                draft_model_id="openai/whisper-large-v3-turbo",
            )
        assert engine._draft_model_id == "openai/whisper-large-v3-turbo"

    def test_no_spec_decode_creates_faster_whisper(self):
        from engines.factory import create_stt_engine

        engine = create_stt_engine(backend="cuda", spec_decode=False)
        assert type(engine).__name__ == "FasterWhisperEngine"


class TestSTTSpecDecodeSettings:
    """Test spec_decode settings in STTSettings."""

    def test_defaults(self):
        from settings import STTSettings

        s = STTSettings()
        assert s.spec_decode is False
        assert s.draft_model == "distil-whisper/distil-large-v3.5"

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("STARK_STT_SPEC_DECODE", "true")
        monkeypatch.setenv("STARK_STT_DRAFT_MODEL", "openai/whisper-large-v3-turbo")
        from settings import STTSettings

        s = STTSettings()
        assert s.spec_decode is True
        assert s.draft_model == "openai/whisper-large-v3-turbo"
