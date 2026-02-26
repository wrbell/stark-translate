"""Tests for settings.py — PipelineSettings, nested config, resolve_backend."""

from unittest.mock import patch


class TestAudioSettings:
    def test_defaults(self):
        from settings import AudioSettings

        s = AudioSettings()
        assert s.sample_rate == 16000
        assert s.mic_sample_rate == 48000
        assert s.chunk_duration == 2.0
        assert s.mic_device is None
        assert s.mic_gain is None
        assert s.target_rms == 0.08


class TestVADSettings:
    def test_defaults(self):
        from settings import VADSettings

        s = VADSettings()
        assert s.threshold == 0.3
        assert s.silence_trigger == 0.8
        assert s.max_utterance == 8.0
        assert s.partial_interval == 1.0


class TestSTTSettings:
    def test_defaults(self):
        from settings import STTSettings

        s = STTSettings()
        assert "whisper-large-v3-turbo" in s.whisper_model
        assert s.fallback_on_low_conf is True
        assert s.fallback_threshold == -1.2
        assert s.hallucination_threshold == 2.4

    def test_whisper_prompt_contains_theological_terms(self):
        from settings import STTSettings

        s = STTSettings()
        assert "atonement" in s.whisper_prompt.lower()
        assert "propitiation" in s.whisper_prompt.lower()

    def test_word_timestamps_default_false(self):
        from settings import STTSettings

        s = STTSettings()
        assert s.word_timestamps is False

    def test_beam_size_default_one(self):
        from settings import STTSettings

        s = STTSettings()
        assert s.beam_size == 1


class TestTranslationSettings:
    def test_defaults(self):
        from settings import TranslationSettings

        s = TranslationSettings()
        assert "translategemma-4b" in s.mlx_model_4b
        assert "translategemma-12b" in s.mlx_model_12b
        assert "opus-mt-en-es" in s.marian_model
        assert s.default_source_lang == "en"
        assert s.default_target_lang == "es"


class TestServerSettings:
    def test_defaults(self):
        from settings import ServerSettings

        s = ServerSettings()
        assert s.ws_port == 8765
        assert s.http_port == 8080


class TestTTSSettings:
    def test_defaults(self):
        from settings import TTSSettings

        s = TTSSettings()
        assert s.enabled is False
        assert s.voice_type == "stock"
        assert "en" in s.voices
        assert "es" in s.voices


class TestPipelineSettings:
    def test_defaults(self):
        from settings import PipelineSettings

        s = PipelineSettings()
        assert s.backend == "auto"
        assert s.run_ab is False
        assert s.low_vram is False
        assert s.multiprocess is False

    def test_nested_settings_accessible(self):
        from settings import PipelineSettings

        s = PipelineSettings()
        assert s.audio.sample_rate == 16000
        assert s.vad.threshold == 0.3
        assert s.stt.fallback_on_low_conf is True
        assert s.server.ws_port == 8765

    def test_resolve_backend_explicit(self):
        from settings import PipelineSettings

        s = PipelineSettings(backend="cuda")
        assert s.resolve_backend() == "cuda"

        s = PipelineSettings(backend="mlx")
        assert s.resolve_backend() == "mlx"

        s = PipelineSettings(backend="cpu")
        assert s.resolve_backend() == "cpu"

    def test_resolve_backend_auto_cpu_fallback(self):
        from settings import PipelineSettings

        s = PipelineSettings(backend="auto")

        def _import_block(name, *args, **kwargs):
            if name in ("mlx.core", "mlx", "torch"):
                raise ImportError(f"Mocked: {name}")
            return __import__(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=_import_block):
            assert s.resolve_backend() == "cpu"

    def test_env_var_override(self, monkeypatch):
        monkeypatch.setenv("STARK_BACKEND", "cuda")
        from settings import PipelineSettings

        s = PipelineSettings()
        assert s.backend == "cuda"

    def test_singleton_exists(self):
        from settings import settings

        assert settings is not None
        assert settings.backend in ("auto", "mlx", "cuda", "cpu")
