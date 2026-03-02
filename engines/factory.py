"""Factory functions for creating STT and translation engines.

Provides ``create_stt_engine`` and ``create_translation_engine`` which
auto-detect the available backend (MLX on Apple Silicon, CUDA on NVIDIA,
CPU as fallback) and return the appropriate engine instance.

The returned engines are *not* loaded -- callers must invoke ``.load()``
before using ``.transcribe()`` or ``.translate()``.
"""

import logging
from typing import Any

from engines.base import STTEngine, TranslationEngine, TTSEngine

logger = logging.getLogger(__name__)


def create_stt_engine(
    backend: str = "auto",
    model_id: str | None = None,
    fallback_threshold: float | None = None,
    hallucination_threshold: float | None = None,
    fallback_on_low_conf: bool | None = None,
    **kwargs: Any,
) -> STTEngine:
    """Create an STT engine for the given backend.

    Args:
        backend:                 "mlx", "cuda", "cpu", or "auto" (detect best).
        model_id:                Override the default model identifier.
        fallback_threshold:      avg_logprob below which to retry with fallback
                                 model.  ``None`` uses the engine default (-1.2).
        hallucination_threshold: compression_ratio above which to retry.
                                 ``None`` uses the engine default (2.4).
        fallback_on_low_conf:    Enable/disable quality-based fallback retry.
                                 ``None`` uses the engine default (True).
        **kwargs:                Forwarded to the engine constructor.

    Returns:
        An *unloaded* ``STTEngine`` instance.  Call ``.load()`` to initialise.
    """
    if backend == "auto":
        backend = _detect_backend()

    # Build threshold kwargs, only including non-None values so engine
    # defaults are preserved when callers don't specify overrides.
    threshold_kwargs = {}
    if fallback_threshold is not None:
        threshold_kwargs["fallback_threshold"] = fallback_threshold
    if hallucination_threshold is not None:
        threshold_kwargs["hallucination_threshold"] = hallucination_threshold
    if fallback_on_low_conf is not None:
        threshold_kwargs["fallback_on_low_conf"] = fallback_on_low_conf

    merged_kwargs = {**threshold_kwargs, **kwargs}

    if backend == "mlx":
        from engines.mlx_engine import MLXWhisperEngine

        return MLXWhisperEngine(
            model_id=model_id or "mlx-community/whisper-large-v3-turbo",
            **merged_kwargs,
        )
    elif backend in ("cuda", "cpu"):
        from engines.cuda_engine import FasterWhisperEngine

        return FasterWhisperEngine(
            model_id=model_id or "large-v3-turbo",
            device=backend,
            **merged_kwargs,
        )
    else:
        raise ValueError(f"Unsupported STT backend: {backend!r}")


def create_translation_engine(
    backend: str = "auto",
    model_id: str | None = None,
    engine_type: str = "gemma",
    **kwargs: Any,
) -> TranslationEngine:
    """Create a translation engine.

    Args:
        backend:     "mlx", "cuda", "cpu", or "auto" (detect best available).
        model_id:    Override the default model identifier.
        engine_type: "gemma" for TranslateGemma, "marian" for MarianMT.
        **kwargs:    Forwarded to the engine constructor.

    Returns:
        An *unloaded* ``TranslationEngine`` instance.  Call ``.load()`` first.
    """
    if backend == "auto":
        backend = _detect_backend()

    if engine_type == "marian":
        from engines.mlx_engine import MarianEngine

        return MarianEngine(
            model_id=model_id or "Helsinki-NLP/opus-mt-en-es",
            **kwargs,
        )

    if backend == "mlx":
        from engines.mlx_engine import MLXGemmaEngine

        return MLXGemmaEngine(
            model_id=model_id or "mlx-community/translategemma-4b-it-4bit",
            **kwargs,
        )
    elif backend == "cuda":
        from engines.cuda_engine import CUDAGemmaEngine

        return CUDAGemmaEngine(
            model_id=model_id or "google/translategemma-4b-it",
            **kwargs,
        )
    else:
        raise ValueError(f"Unsupported translation backend: {backend!r}")


def create_tts_engine(voices: dict[str, str] | None = None) -> TTSEngine:
    """Create a TTS engine with the specified voice configuration.

    Args:
        voices:  Dict mapping language codes to Piper voice names.
                 Default: ``{"es": "es_ES-carlfm-high"}``.

    Returns:
        An *unloaded* ``TTSEngine`` instance.  Call ``.load()`` to initialise.
    """
    from engines.mlx_engine import PiperTTSEngine

    return PiperTTSEngine(voices=voices or {"es": "es_ES-carlfm-high"})


def _detect_backend() -> str:
    """Auto-detect the best available inference backend.

    Priority: MLX (Apple Silicon) > CUDA (NVIDIA) > CPU.
    """
    try:
        import mlx.core  # noqa: F401

        logger.info("Auto-detected backend: mlx (Apple Silicon)")
        return "mlx"
    except ImportError:
        pass

    try:
        import torch

        if torch.cuda.is_available():
            logger.info("Auto-detected backend: cuda (NVIDIA)")
            return "cuda"
    except ImportError:
        pass

    logger.info("Auto-detected backend: cpu (fallback)")
    return "cpu"
