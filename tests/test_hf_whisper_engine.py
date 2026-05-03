"""Tests for HF Whisper engine with speculative decoding.

All tests mock HF transformers — runs in CI without GPU or model downloads.
"""

from unittest.mock import patch


class TestHFWhisperEngine:
    """Test HFWhisperEngine construction and configuration."""

    def test_default_draft_model_is_none(self):
        # Distil-v3.5 + turbo is broken (different decoder layer counts);
        # see docs/archive/v2026.5/spec_decode_research.md. No default draft.
        from engines.hf_whisper_engine import HFWhisperEngine

        with patch("engines.hf_whisper_engine.HF_WHISPER_AVAILABLE", True):
            engine = HFWhisperEngine()
        assert engine._model_id == "openai/whisper-large-v3-turbo"
        assert engine._draft_model_id is None

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

    def test_spec_decode_without_draft_now_raises(self):
        # Removed the silent default that paired distil-v3.5 with turbo (broken).
        # Caller must provide draft_model_id explicitly.
        import pytest

        from engines.factory import create_stt_engine

        with (
            patch("engines.hf_whisper_engine.HF_WHISPER_AVAILABLE", True),
            pytest.raises(ValueError, match="explicit draft_model_id"),
        ):
            create_stt_engine(backend="cuda", spec_decode=True)

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
        # No safe default — see docs/archive/v2026.5/spec_decode_research.md
        assert s.draft_model is None

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("STARK_STT_SPEC_DECODE", "true")
        monkeypatch.setenv("STARK_STT_DRAFT_MODEL", "openai/whisper-large-v3-turbo")
        from settings import STTSettings

        s = STTSettings()
        assert s.spec_decode is True
        assert s.draft_model == "openai/whisper-large-v3-turbo"
