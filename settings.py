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
        description="Speech probability threshold (0-1). Lower = more sensitive. Matches --vad-threshold argparse default.",
    )
    silence_trigger: float = Field(
        default=0.5,
        description="Seconds of silence before final STT/translation is triggered. Tunable via STARK_VAD__SILENCE_TRIGGER.",
    )
    max_utterance: float = Field(
        default=8.0,
        description="Force-process utterance if speaker doesn't pause (seconds)",
    )
    partial_interval: float = Field(
        default=0.6,
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
        default="wbell7/distil-whisper-large-v3.5-mlx",
        description="Fallback Whisper model if primary returns low-confidence output",
    )
    whisper_cuda_model: str = Field(
        default="large-v3-turbo",
        description="Whisper model name for faster-whisper (CUDA backend)",
    )
    whisper_cuda_compute_type: str = Field(
        default="int8_float16",
        description=(
            "faster-whisper CTranslate2 compute type. int8_float16 is the "
            "Ampere/Ada sweet spot (~20%% faster than int8 alone, +30%% VRAM, "
            "<0.3 WER points). Tight VRAM budgets: set int8. Other options: "
            "int8_bfloat16, float16, bfloat16, float32."
        ),
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
    word_timestamps: bool = Field(
        default=False,
        description=(
            "Enable per-word timestamps and confidence scores (DTW alignment). "
            "Adds ~200-400ms to final STT. Useful for active learning, not needed for live display."
        ),
    )
    beam_size: int = Field(
        default=1,
        description=(
            "Beam search width for Whisper decoding (faster-whisper/CUDA only). "
            "1 = greedy (fastest), 5 = default beam search. "
            "mlx-whisper always uses greedy decoding (beam search not implemented)."
        ),
    )

    spec_decode: bool = Field(
        default=False,
        description=(
            "Opt into HF Whisper spec decode. Requires an explicit draft_model "
            "below — there is no safe default for whisper-large-v3-turbo. "
            "See docs/archive/v2026.5/spec_decode_research.md."
        ),
    )
    draft_model: str | None = Field(
        default=None,
        description=(
            "Draft model for STT speculative decoding. NO DEFAULT — distil-v3.5 "
            "is incompatible with whisper-large-v3-turbo (different decoder layer "
            "counts → 10x slower with hallucinated output, tested 2026-04-13). "
            "Verified pairing: target=openai/whisper-large-v3, "
            "draft=openai/whisper-large-v3-turbo (turbo IS a distilled v3)."
        ),
    )
    backend: str = Field(
        default="auto",
        description=(
            "Whisper implementation choice within a hardware tier: 'auto' "
            "(faster-whisper on cuda/cpu, mlx on Apple Silicon), 'faster-whisper' "
            "(CTranslate2, default on CUDA), 'hf' (transformers, supports "
            "torch.compile + spec decode), 'mlx' (Apple Silicon only), "
            "'parakeet' (NVIDIA Parakeet TDT — EN-only accelerator, requires nemo)."
        ),
    )
    parakeet_model: str = Field(
        default="nvidia/parakeet-tdt-0.6b-v3",
        description="NeMo Parakeet model id when STARK_STT__BACKEND=parakeet",
    )
    compile_mode: str | None = Field(
        default=None,
        description=(
            "torch.compile mode for HFWhisperEngine: 'reduce-overhead' enables CUDA "
            "graphs (best steady-state speed, recompiles on input shape changes), "
            "'default' is balanced. None disables. No effect on FasterWhisperEngine."
        ),
    )
    warmup_seconds: int = Field(
        default=1,
        description=(
            "Seconds of silence pushed through the engine after load to drive "
            "JIT/CUDA-graph capture before the first real call. 0 disables."
        ),
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
    cuda_model_12b: str = Field(
        default="google/translategemma-12b-it",
        description="TranslateGemma 12B for CUDA A/B mode (loaded with bitsandbytes 4-bit)",
    )
    # MarianMT (all backends — CPU inference, lightweight)
    marian_model: str = Field(
        default="Helsinki-NLP/opus-mt-en-es",
        description="MarianMT model for fast partial translations (~80ms PyTorch)",
    )
    model_family: str = Field(
        default="translategemma",
        description="Translation model family: 'translategemma' (structured lang codes) or 'gemma4' (instruct prompt)",
    )
    num_draft_tokens: int = Field(
        default=3,
        description="Speculative decoding: tokens drafted by 4B for 12B to verify",
    )
    default_source_lang: str = Field(
        default="en",
        description="Source language code for translation chat template",
    )
    default_target_lang: str = Field(
        default="es",
        description="Target language code for TranslateGemma chat template",
    )
    # TurboQuant KV cache compression (MLX only, via mlx-optiq)
    turboquant: bool = Field(
        default=False,
        description="Enable TurboQuant KV cache compression (4.6x, MLX only, requires mlx-optiq)",
    )
    turboquant_key_bits: int = Field(
        default=3,
        description="TurboQuant key quantization bits (3 = near-lossless)",
    )
    turboquant_val_bits: int = Field(
        default=4,
        description="TurboQuant value quantization bits (4 = near-lossless)",
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
    output_mode: Literal["ws", "wav", "both", "local"] = Field(
        default="ws",
        description=(
            "TTS output: 'ws' for WebSocket stream, 'wav' for file, 'both' for both, "
            "'local' for sounddevice playback to output_device (Phase 9.4.1)"
        ),
    )
    audio_ws_port: int = Field(
        default=8766,
        description="WebSocket port for TTS audio streaming (separate from text WS on 8765)",
    )
    output_device: int | None = Field(
        default=None,
        description=(
            "sounddevice output device index for output_mode='local'. None = system default. "
            "Get the index from /api/devices or `python -m sounddevice`."
        ),
    )

    model_config = {"env_prefix": "STARK_TTS_"}


class GlossarySettings(BaseSettings):
    """Tiered glossary configuration for Deepgram keyterms and normalization."""

    boost_size: int = Field(
        default=50,
        description="Number of Tier 1 boost terms for Deepgram keyterms (max ~80)",
    )
    master_size: int = Field(
        default=1000,
        description="Number of Tier 2 master terms for normalization and active learning",
    )
    boost_path: str = Field(
        default="bible_data/glossary/tier1_boost.json",
        description="Path to Tier 1 boost terms JSON (list of EN strings)",
    )
    master_path: str = Field(
        default="bible_data/glossary/tier2_master.json",
        description="Path to Tier 2 master glossary JSON (EN→ES dict)",
    )

    model_config = {"env_prefix": "STARK_GLOSSARY_"}


class DeepgramSettings(BaseSettings):
    """Deepgram Nova-3 oracle transcription configuration."""

    api_key: str = Field(
        default="",
        description="Deepgram API key (or set STARK_DEEPGRAM__API_KEY env var)",
    )
    model: str = Field(
        default="nova-3",
        description="Deepgram model name",
    )
    language: str = Field(
        default="en",
        description="Transcription language (en or es)",
    )
    smart_format: bool = Field(
        default=True,
        description="Enable Deepgram smart formatting (punctuation, casing)",
    )
    diarize: bool = Field(
        default=False,
        description="Enable speaker diarization",
    )
    mip_opt_out: bool = Field(
        default=True,
        description="Opt out of Deepgram model improvement program",
    )

    model_config = {"env_prefix": "STARK_DEEPGRAM_"}


# ---------------------------------------------------------------------------
# Top-level pipeline settings
# ---------------------------------------------------------------------------


class CUDASettings(BaseSettings):
    """CUDA/NVIDIA-specific inference configuration."""

    vram_tier: Literal["auto", "full_ab", "4b_only", "marian"] = Field(
        default="auto",
        description="VRAM tier: auto (detect GPU), full_ab (>=15GB), 4b_only (>=5.5GB), marian (<5.5GB)",
    )
    use_prompt_cache: bool = Field(
        default=True,
        description="Pre-compute KV cache for TranslateGemma prompt prefix (~50-80ms savings/call)",
    )
    use_speculative: bool = Field(
        default=True,
        description="Use 4B as speculative draft model for 12B verification (A/B mode only)",
    )
    pipeline_workers: int = Field(
        default=2,
        description="Thread pool workers for CUDA pipeline (2 = STT/Translation overlap)",
    )
    streaming_batch_size: int = Field(
        default=3,
        description="Tokens per WebSocket batch during streaming translation",
    )
    compute_type: str = Field(
        default="int8_float16",
        description=(
            "faster-whisper CTranslate2 compute type. int8_float16 is the "
            "Ampere/Ada default since v2026.7 (~20%% faster than int8 alone, "
            "+30%% VRAM). Set int8 for VRAM-constrained 6 GB cards."
        ),
    )
    engine: Literal["auto", "llamacpp", "hf"] = Field(
        default="auto",
        description=(
            "Translation engine selection on CUDA. 'auto' probes for a running "
            "llama-server + GGUF files in models/ and prefers llamacpp when "
            "available; falls back to HF NF4. 'llamacpp' / 'hf' force one path."
        ),
    )
    llamacpp_url: str = Field(
        default="http://127.0.0.1:8090",
        description="Primary llama-server URL (engine A). See start_server.sh.",
    )
    llamacpp_url_b: str = Field(
        default="http://127.0.0.1:8091",
        description="Secondary llama-server URL for --ab mode (engine B, e.g. E2B).",
    )

    model_config = {"env_prefix": "STARK_CUDA_"}


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
    multiprocess: bool = Field(
        default=False,
        description=(
            "Optional: run STT and translation in separate OS processes (separate Metal "
            "contexts). Default Mac path already overlaps STT∥translation in-process "
            "via MLX >= 0.31.2 thread-local streams (max_workers=2). Use this as an "
            "escape hatch for debugging or older mlx builds."
        ),
    )

    # Nested sub-settings
    audio: AudioSettings = Field(default_factory=AudioSettings)
    vad: VADSettings = Field(default_factory=VADSettings)
    stt: STTSettings = Field(default_factory=STTSettings)
    translation: TranslationSettings = Field(default_factory=TranslationSettings)
    server: ServerSettings = Field(default_factory=ServerSettings)
    tts: TTSSettings = Field(default_factory=TTSSettings)
    glossary: GlossarySettings = Field(default_factory=GlossarySettings)
    deepgram: DeepgramSettings = Field(default_factory=DeepgramSettings)
    cuda: CUDASettings = Field(default_factory=CUDASettings)

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
