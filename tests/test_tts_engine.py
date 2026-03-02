"""Tests for TTS engine — TTSResult, TTSEngine ABC, PiperTTSEngine, factory, settings."""

import io
import sys
import wave
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from engines.base import TTSEngine, TTSResult

# ---------------------------------------------------------------------------
# TTSResult dataclass
# ---------------------------------------------------------------------------


class TestTTSResult:
    def test_required_fields(self):
        audio = np.zeros(22050, dtype=np.float32)
        r = TTSResult(audio=audio, sample_rate=22050, latency_ms=150.0, text="hola")
        assert r.text == "hola"
        assert r.sample_rate == 22050
        assert r.latency_ms == 150.0
        assert len(r.audio) == 22050

    def test_optional_defaults(self):
        audio = np.zeros(100, dtype=np.float32)
        r = TTSResult(audio=audio, sample_rate=22050, latency_ms=0.0, text="")
        assert r.wav_path is None

    def test_wav_path_set(self):
        audio = np.zeros(100, dtype=np.float32)
        r = TTSResult(audio=audio, sample_rate=22050, latency_ms=50.0, text="test", wav_path="/tmp/test.wav")
        assert r.wav_path == "/tmp/test.wav"

    def test_audio_is_numpy_array(self):
        audio = np.array([0.0, 0.5, -0.5], dtype=np.float32)
        r = TTSResult(audio=audio, sample_rate=16000, latency_ms=10.0, text="x")
        assert isinstance(r.audio, np.ndarray)
        assert r.audio.dtype == np.float32


# ---------------------------------------------------------------------------
# TTSEngine ABC
# ---------------------------------------------------------------------------


class TestTTSEngineABC:
    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            TTSEngine()

    def test_subclass_must_implement_all(self):
        class PartialTTS(TTSEngine):
            def load(self):
                pass

        with pytest.raises(TypeError):
            PartialTTS()

    def test_complete_subclass(self):
        class ConcreteTTS(TTSEngine):
            def load(self):
                pass

            def synthesize(self, text, *, language="es"):
                return TTSResult(
                    audio=np.zeros(100, dtype=np.float32),
                    sample_rate=22050,
                    latency_ms=0.0,
                    text=text,
                )

            def unload(self):
                pass

            @property
            def model_id(self):
                return "test-tts"

            @property
            def backend(self):
                return "cpu"

        engine = ConcreteTTS()
        assert engine.model_id == "test-tts"
        assert engine.backend == "cpu"
        result = engine.synthesize("hello")
        assert isinstance(result, TTSResult)


# ---------------------------------------------------------------------------
# PiperTTSEngine (with mocked piper)
# ---------------------------------------------------------------------------


def _make_wav_bytes(n_samples=4410, sample_rate=22050):
    """Create a minimal valid WAV file in memory."""
    buf = io.BytesIO()
    audio_int16 = np.zeros(n_samples, dtype=np.int16)
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())
    return buf.getvalue()


class TestPiperTTSEngine:
    def _make_mock_piper(self):
        """Set up mock piper.voice module."""
        mock_piper = MagicMock()
        mock_piper_voice = MagicMock()

        def mock_synthesize(text, wav_file):
            """Write valid WAV data to the file handle."""
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(22050)
            audio_int16 = np.zeros(4410, dtype=np.int16)
            wav_file.writeframes(audio_int16.tobytes())

        voice_instance = MagicMock()
        voice_instance.synthesize_wav = mock_synthesize
        mock_piper_voice.PiperVoice.load.return_value = voice_instance

        return mock_piper, mock_piper_voice, voice_instance

    def _load_engine(self, engine):
        """Helper: load a PiperTTSEngine with mocked filesystem and download."""
        from engines.mlx_engine import PiperTTSEngine

        with (
            patch("os.path.exists", return_value=False),
            patch.object(PiperTTSEngine, "_ensure_voice_downloaded", return_value="/tmp/voice.onnx"),
        ):
            engine.load()

    def test_load_creates_voices(self):
        mock_piper, mock_piper_voice, _ = self._make_mock_piper()
        with patch.dict(sys.modules, {"piper": mock_piper, "piper.voice": mock_piper_voice}):
            from engines.mlx_engine import PiperTTSEngine

            engine = PiperTTSEngine(voices={"es": "es_ES-carlfm-high"})
            self._load_engine(engine)

            assert engine._loaded is True
            assert "es" in engine._voices

    def test_synthesize_returns_valid_result(self):
        mock_piper, mock_piper_voice, _ = self._make_mock_piper()
        with patch.dict(sys.modules, {"piper": mock_piper, "piper.voice": mock_piper_voice}):
            from engines.mlx_engine import PiperTTSEngine

            engine = PiperTTSEngine(voices={"es": "es_ES-carlfm-high"})
            self._load_engine(engine)

            result = engine.synthesize("Hola mundo", language="es")
            assert isinstance(result, TTSResult)
            assert result.text == "Hola mundo"
            assert result.sample_rate == 22050
            assert result.audio.dtype == np.float32
            assert len(result.audio) == 4410
            assert result.latency_ms >= 0

    def test_synthesize_unknown_language_raises(self):
        mock_piper, mock_piper_voice, _ = self._make_mock_piper()
        with patch.dict(sys.modules, {"piper": mock_piper, "piper.voice": mock_piper_voice}):
            from engines.mlx_engine import PiperTTSEngine

            engine = PiperTTSEngine(voices={"es": "es_ES-carlfm-high"})
            self._load_engine(engine)

            with pytest.raises(ValueError, match="No voice loaded for language 'fr'"):
                engine.synthesize("Bonjour", language="fr")

    def test_synthesize_before_load_raises(self):
        mock_piper, mock_piper_voice, _ = self._make_mock_piper()
        with patch.dict(sys.modules, {"piper": mock_piper, "piper.voice": mock_piper_voice}):
            from engines.mlx_engine import PiperTTSEngine

            engine = PiperTTSEngine(voices={"es": "es_ES-carlfm-high"})
            with pytest.raises(RuntimeError, match="Engine not loaded"):
                engine.synthesize("Hola")

    def test_unload_clears_voices(self):
        mock_piper, mock_piper_voice, _ = self._make_mock_piper()
        with patch.dict(sys.modules, {"piper": mock_piper, "piper.voice": mock_piper_voice}):
            from engines.mlx_engine import PiperTTSEngine

            engine = PiperTTSEngine(voices={"es": "es_ES-carlfm-high"})
            self._load_engine(engine)

            engine.unload()
            assert engine._loaded is False
            assert len(engine._voices) == 0

    def test_model_id(self):
        mock_piper, mock_piper_voice, _ = self._make_mock_piper()
        with patch.dict(sys.modules, {"piper": mock_piper, "piper.voice": mock_piper_voice}):
            from engines.mlx_engine import PiperTTSEngine

            engine = PiperTTSEngine(voices={"es": "es_ES-carlfm-high", "en": "en_US-lessac-high"})
            assert "piper:" in engine.model_id
            assert "es=" in engine.model_id

    def test_backend(self):
        mock_piper, mock_piper_voice, _ = self._make_mock_piper()
        with patch.dict(sys.modules, {"piper": mock_piper, "piper.voice": mock_piper_voice}):
            from engines.mlx_engine import PiperTTSEngine

            engine = PiperTTSEngine()
            assert engine.backend == "onnx"

    def test_multiple_languages(self):
        mock_piper, mock_piper_voice, _ = self._make_mock_piper()
        with patch.dict(sys.modules, {"piper": mock_piper, "piper.voice": mock_piper_voice}):
            from engines.mlx_engine import PiperTTSEngine

            engine = PiperTTSEngine(voices={"es": "es_ES-carlfm-high", "en": "en_US-lessac-high"})
            self._load_engine(engine)

            # Both languages should work
            result_es = engine.synthesize("Hola", language="es")
            assert result_es.text == "Hola"

            result_en = engine.synthesize("Hello", language="en")
            assert result_en.text == "Hello"


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------


class TestCreateTTSEngine:
    def test_creates_piper_engine(self):
        mock_piper = MagicMock()
        mock_piper_voice = MagicMock()
        with patch.dict(sys.modules, {"piper": mock_piper, "piper.voice": mock_piper_voice}):
            from engines.factory import create_tts_engine

            engine = create_tts_engine()
            assert engine.backend == "onnx"
            assert "es" in engine._voice_specs

    def test_custom_voices(self):
        mock_piper = MagicMock()
        mock_piper_voice = MagicMock()
        with patch.dict(sys.modules, {"piper": mock_piper, "piper.voice": mock_piper_voice}):
            from engines.factory import create_tts_engine

            voices = {"en": "en_US-lessac-high", "es": "es_ES-carlfm-high"}
            engine = create_tts_engine(voices=voices)
            assert engine._voice_specs == voices


# ---------------------------------------------------------------------------
# TTSSettings
# ---------------------------------------------------------------------------


class TestTTSSettingsNew:
    def test_output_mode_default(self):
        from settings import TTSSettings

        s = TTSSettings()
        assert s.output_mode == "ws"

    def test_audio_ws_port_default(self):
        from settings import TTSSettings

        s = TTSSettings()
        assert s.audio_ws_port == 8766

    def test_output_mode_values(self):
        from settings import TTSSettings

        for mode in ("ws", "wav", "both"):
            s = TTSSettings(output_mode=mode)
            assert s.output_mode == mode

    def test_env_var_override(self, monkeypatch):
        monkeypatch.setenv("STARK_TTS_OUTPUT_MODE", "both")
        monkeypatch.setenv("STARK_TTS_AUDIO_WS_PORT", "9999")
        from settings import TTSSettings

        s = TTSSettings()
        assert s.output_mode == "both"
        assert s.audio_ws_port == 9999
