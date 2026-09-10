"""Piper TTS runtime, independent of Torch and MLX."""

import logging
import os
import time

import numpy as np

from engines.base import TTSEngine, TTSResult

logger = logging.getLogger(__name__)


class PiperTTSEngine(TTSEngine):
    """Text-to-speech engine wrapping Piper (ONNX Runtime).

    Piper uses ONNX Runtime for inference (CPU), not MLX. Despite living in
    mlx_engine.py, it's the Mac-side TTS engine. ONNX Runtime is thread-safe,
    so TTS can run on its own thread pool without the MLX single-thread restriction.

    Constructor args:
        voices:  Dict mapping language codes to Piper voice names or paths.
                 Default: ``{"es": "es_MX-claude-high"}``.
    """

    def __init__(self, voices: dict[str, str] | None = None):
        self._voice_specs = voices or {"es": "es_MX-claude-high"}
        self._voices: dict = {}  # lang -> PiperVoice instance
        self._loaded = False

    def load(self) -> None:
        """Load Piper voice models for each configured language."""
        from piper.voice import PiperVoice

        logger.info("Loading Piper TTS voices: %s", self._voice_specs)
        t0 = time.time()

        for lang, voice_spec in self._voice_specs.items():
            # Try custom path first, then resolve from Piper's download cache
            custom_path = f"piper_voices/{lang}/{voice_spec}.onnx"
            if os.path.exists(custom_path):
                voice = PiperVoice.load(custom_path)
                logger.info("  Loaded custom voice for %s: %s", lang, custom_path)
            else:
                # Stock voice: download only the registered, pinned files.
                model_path = self._ensure_voice_downloaded(voice_spec)
                voice = PiperVoice.load(model_path)
                logger.info("  Loaded stock voice for %s: %s", lang, voice_spec)
            self._voices[lang] = voice

        self._loaded = True
        logger.info("Piper TTS ready (%d voices, %.1fs)", len(self._voices), time.time() - t0)

    def synthesize(self, text: str, *, language: str = "es") -> TTSResult:
        """Synthesize text to float32 audio array using the voice for *language*."""
        if not self._loaded:
            raise RuntimeError("Engine not loaded -- call load() first")

        if language not in self._voices:
            raise ValueError(f"No voice loaded for language '{language}'. Available: {list(self._voices.keys())}")

        import io
        import wave

        voice = self._voices[language]

        t0 = time.perf_counter()

        # Synthesize to in-memory WAV
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wav_file:
            voice.synthesize_wav(text, wav_file)

        latency_ms = (time.perf_counter() - t0) * 1000

        # Extract raw PCM from WAV and convert int16 -> float32
        wav_buffer.seek(0)
        with wave.open(wav_buffer, "rb") as wav_file:
            sample_rate = wav_file.getframerate()
            n_frames = wav_file.getnframes()
            raw_bytes = wav_file.readframes(n_frames)

        audio_int16 = np.frombuffer(raw_bytes, dtype=np.int16)
        audio_float32 = audio_int16.astype(np.float32) / 32768.0

        return TTSResult(
            audio=audio_float32,
            sample_rate=sample_rate,
            latency_ms=latency_ms,
            text=text,
        )

    def play(self, audio_float32: np.ndarray, sample_rate: int, device: int | None = None) -> None:
        """Play synthesized audio to a sounddevice output device.

        Used by ``--tts-output local`` mode (Phase 9.4.1) so the operator
        can route the translated audio to a chosen speaker / monitor without
        going through the WebSocket clients.

        Args:
            audio_float32:  PCM as returned by ``synthesize().audio``.
            sample_rate:    Hz, also from the ``TTSResult``.
            device:         sounddevice output device index, or ``None`` for
                            system default. Get the index from
                            ``operator_app.audio.list_devices()`` outputs.

        Returns immediately after the audio is queued; the playback runs on
        sounddevice's own thread.
        PortAudioError propagates so the routing layer can re-resolve hotplugged devices.
        """
        import sounddevice as sd

        try:
            sd.play(audio_float32, samplerate=sample_rate, device=device)
        except sd.PortAudioError:
            raise
        except Exception as exc:
            logger.warning("PiperTTSEngine.play failed (device=%s): %s", device, exc)

    def unload(self) -> None:
        """Release voice models from memory."""
        self._voices.clear()
        self._loaded = False
        logger.info("PiperTTSEngine unloaded")

    @property
    def model_id(self) -> str:
        return f"piper:{','.join(f'{k}={v}' for k, v in self._voice_specs.items())}"

    @property
    def backend(self) -> str:
        return "onnx"

    @staticmethod
    def _ensure_voice_downloaded(voice_name: str) -> str:
        """Resolve a local voice or download its registered immutable HF revision."""
        from pathlib import Path

        from engines.model_paths import pinned_hf_entry, resolve_piper_voice

        local = resolve_piper_voice(voice_name)
        if local:
            return local

        if os.environ.get("STARK_PROFILE", "standard").startswith("lite-"):
            raise FileNotFoundError(f"Pinned Piper voice {voice_name} missing; run setup --include tts")

        entry = pinned_hf_entry(voice_name)
        files = entry.get("required_files", [])
        model_file = next((name for name in files if name.endswith(".onnx")), None)
        if not model_file or f"{model_file}.json" not in files:
            raise ValueError(f"Pinned Piper voice {voice_name} must register its ONNX model and config files")

        # Piper voices cache in ~/.local/share/piper_tts/ by convention.
        cache_dir = Path.home() / ".local" / "share" / "piper_tts"
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Download via huggingface_hub
        from huggingface_hub import hf_hub_download

        logger.info("Downloading Piper voice: %s", voice_name)
        source_kwargs = {
            "repo_id": entry["repo_id"],
            "revision": entry["revision"],
            "cache_dir": str(cache_dir),
        }
        onnx_local = hf_hub_download(filename=model_file, **source_kwargs)
        hf_hub_download(filename=f"{model_file}.json", **source_kwargs)

        return onnx_local
