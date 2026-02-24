#!/usr/bin/env python3
"""
settings.py — Unified Configuration for Stark Road Bilingual Pipeline

Centralizes all runtime configuration using pydantic-settings.
Supports environment variable overrides (prefix: STARK_) and .env files.

Environment variable examples:
    STARK_BACKEND=cuda
    STARK_SERVER__WS_PORT=9000
    STARK_VAD__THRESHOLD=0.4
    STARK_STT__WHISPER_MODEL=mlx-community/whisper-large-v3-turbo
    STARK_LOW_VRAM=true

Nested delimiter is "__" (double underscore), so:
    STARK_STT__WHISPER_MODEL   -> settings.stt.whisper_model
    STARK_AUDIO__MIC_DEVICE    -> settings.audio.mic_device
    STARK_VAD__SILENCE_TRIGGER -> settings.vad.silence_trigger

Usage:
    from settings import settings
    print(settings.backend)
    print(settings.stt.whisper_model)
    print(settings.resolve_backend())
"""

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings

# ---------------------------------------------------------------------------
# Sub-settings groups
# ---------------------------------------------------------------------------


class AudioSettings(BaseSettings):
    """Audio capture and processing configuration."""

    sample_rate: int = Field(
        default=16000,
        description="Whisper/VAD target sample rate (Hz)",
    )
    mic_sample_rate: int = Field(
        default=48000,
        description="Native mic sample rate; resampled to sample_rate internally",
    )
    chunk_duration: float = Field(
        default=2.0,
        description="Seconds of speech per chunk — more context = better word accuracy",
    )
    mic_device: int | None = Field(
        default=None,
        description="Audio input device index (None = auto-detect best input)",
    )
    mic_gain: float | None = Field(
        default=None,
        description="Mic gain multiplier (None = auto-calibrate from target_rms)",
    )
    target_rms: float = Field(
        default=0.08,
        description="Target RMS amplitude for auto-gain calibration",
    )

    model_config = {"env_prefix": "STARK_AUDIO_"}


class VADSettings(BaseSettings):
    """Voice Activity Detection (Silero VAD) configuration."""

    threshold: float = Field(
        default=0.3,
        description="Speech probability threshold (0-1). Lower = more sensitive.",
    )
    silence_trigger: float = Field(
        default=0.8,
        description="Seconds of silence before final STT/translation is triggered",
    )
    max_utterance: float = Field(
        default=8.0,
        description="Force-process utterance if speaker doesn't pause (seconds)",
    )
    partial_interval: float = Field(
        default=1.0,
        description="Seconds between partial STT updates while speaking",
    )

    model_config = {"env_prefix": "STARK_VAD_"}


class STTSettings(BaseSettings):
    """Speech-to-text model configuration."""

    whisper_model: str = Field(
        default="mlx-community/whisper-large-v3-turbo",
        description="Primary Whisper model (MLX community repo ID)",
    )
    whisper_fallback: str = Field(
        default="mlx-community/distil-whisper-large-v3",
        description="Fallback Whisper model if primary returns low-confidence output",
    )
    whisper_cuda_model: str = Field(
        default="large-v3-turbo",
        description="Whisper model name for faster-whisper (CUDA backend)",
    )
    whisper_cuda_compute_type: str = Field(
        default="int8",
        description="faster-whisper compute type: int8, float16, float32",
    )
    whisper_prompt: str = Field(
        default=(
            "Sermon at Stark Road Gospel Hall. "
            "Christ Jesus, the Holy Spirit, God the Father. "
            "Atonement, propitiation, mediator, covenant, righteousness, "
            "sanctification, justification, redemption, reconciliation, "
            "repentance, reign, grace, mercy, the Gospel, epistle, apostle, "
            "Scripture, the Lord, the Word of God."
        ),
        description="Whisper initial_prompt — biases decoder toward theological vocabulary",
    )
    use_chunked_algo: bool = Field(
        default=False,
        description="Use chunked inference for long audio (faster but stitching artifacts)",
    )
    chunk_length_s: int = Field(
        default=30,
        description="Chunk length in seconds for chunked inference mode",
    )
    fallback_on_low_conf: bool = Field(
        default=True,
        description="Retry with fallback model when avg_logprob < fallback_threshold",
    )
    fallback_threshold: float = Field(
        default=-1.2,
        description="avg_logprob threshold below which to retry with fallback model",
    )
    hallucination_threshold: float = Field(
        default=2.4,
        description="compression_ratio above which output is flagged as hallucination",
    )

    model_config = {"env_prefix": "STARK_STT_"}


class TranslationSettings(BaseSettings):
    """Translation model configuration (TranslateGemma + MarianMT)."""

    # MLX models (Apple Silicon)
    mlx_model_4b: str = Field(
        default="mlx-community/translategemma-4b-it-4bit",
        description="TranslateGemma 4B 4-bit MLX model (~2.2GB disk, ~2.5GB RAM)",
    )
    mlx_model_12b: str = Field(
        default="mlx-community/translategemma-12b-it-4bit",
        description="TranslateGemma 12B 4-bit MLX model (~6.6GB disk, ~7GB RAM)",
    )
    # CUDA models (NVIDIA)
    cuda_model_4b: str = Field(
        default="google/translategemma-4b-it",
        description="TranslateGemma 4B for CUDA (loaded with bitsandbytes 4-bit)",
    )
    # MarianMT (all backends — CPU inference, lightweight)
    marian_model: str = Field(
        default="Helsinki-NLP/opus-mt-en-es",
        description="MarianMT model for fast partial translations (~80ms PyTorch)",
    )
    num_draft_tokens: int = Field(
        default=3,
        description="Speculative decoding: tokens drafted by 4B for 12B to verify",
    )
    default_source_lang: str = Field(
        default="en",
        description="Source language code for TranslateGemma chat template",
    )
    default_target_lang: str = Field(
        default="es",
        description="Target language code for TranslateGemma chat template",
    )

    model_config = {"env_prefix": "STARK_TRANSLATE_"}


class ServerSettings(BaseSettings):
    """HTTP and WebSocket server configuration."""

    ws_port: int = Field(
        default=8765,
        description="WebSocket server port (browser displays connect here)",
    )
    http_port: int = Field(
        default=8080,
        description="HTTP server port for display pages (LAN/phone access)",
    )

    model_config = {"env_prefix": "STARK_SERVER_"}


class TTSSettings(BaseSettings):
    """Text-to-speech (Piper) configuration."""

    enabled: bool = Field(
        default=False,
        description="Enable TTS audio output of translations",
    )
    voice_type: Literal["custom", "stock"] = Field(
        default="stock",
        description="Use custom fine-tuned voice or stock Piper voice",
    )
    multi_channel: bool = Field(
        default=False,
        description="Route TTS to separate audio channels per language",
    )
    voices: dict = Field(
        default={
            "en": "en_US-lessac-high",
            "es": "es_ES-carlfm-high",
            "hi": "hi_IN-kusal-medium",
            "zh": "zh_CN-huayan-medium",
        },
        description="Piper voice names keyed by language code",
    )

    model_config = {"env_prefix": "STARK_TTS_"}


# ---------------------------------------------------------------------------
# Top-level pipeline settings
# ---------------------------------------------------------------------------


class PipelineSettings(BaseSettings):
    """Top-level pipeline configuration.

    All sub-settings are nested. Environment variables use STARK_ prefix
    with double-underscore delimiter for nesting:

        STARK_BACKEND=cuda            -> settings.backend
        STARK_SERVER__WS_PORT=9000    -> settings.server.ws_port
        STARK_VAD__THRESHOLD=0.4      -> settings.vad.threshold

    Also reads from .env file in the project root if present.
    """

    backend: Literal["auto", "mlx", "cuda", "cpu"] = Field(
        default="auto",
        description=(
            "Inference backend: auto (detect best available), mlx (Apple Silicon), cuda (NVIDIA), cpu (fallback)"
        ),
    )
    run_ab: bool = Field(
        default=False,
        description="Load both 4B and 12B TranslateGemma for A/B comparison",
    )
    low_vram: bool = Field(
        default=False,
        description="Minimal VRAM mode: MarianMT-only translation, no Gemma loaded",
    )

    # Nested sub-settings
    audio: AudioSettings = Field(default_factory=AudioSettings)
    vad: VADSettings = Field(default_factory=VADSettings)
    stt: STTSettings = Field(default_factory=STTSettings)
    translation: TranslationSettings = Field(default_factory=TranslationSettings)
    server: ServerSettings = Field(default_factory=ServerSettings)
    tts: TTSSettings = Field(default_factory=TTSSettings)

    model_config = {
        "env_prefix": "STARK_",
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "env_nested_delimiter": "__",
    }

    def resolve_backend(self) -> str:
        """Resolve 'auto' backend to a concrete value based on available hardware.

        Detection order: MLX (Apple Silicon) > CUDA (NVIDIA) > CPU.
        Heavy libraries (torch, mlx) are imported lazily to avoid slow startup
        when only the resolved value is needed.
        """
        if self.backend != "auto":
            return self.backend

        # Check for MLX (Apple Silicon with Metal)
        try:
            import mlx.core  # noqa: F401

            return "mlx"
        except ImportError:
            pass

        # Check for CUDA (NVIDIA GPU)
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass

        return "cpu"


# ---------------------------------------------------------------------------
# Singleton instance — this is the main API
#
#   from settings import settings
#   settings.stt.whisper_model
#   settings.vad.threshold
#   settings.resolve_backend()
# ---------------------------------------------------------------------------

settings = PipelineSettings()
