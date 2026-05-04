"""Factory functions for creating STT and translation engines.

Provides ``create_stt_engine`` and ``create_translation_engine`` which
auto-detect the available backend (MLX on Apple Silicon, CUDA on NVIDIA,
CPU as fallback) and return the appropriate engine instance.

The returned engines are *not* loaded -- callers must invoke ``.load()``
before using ``.transcribe()`` or ``.translate()``.
"""

import logging
from pathlib import Path
from typing import Any

from engines.base import STTEngine, TranslationEngine, TTSEngine

logger = logging.getLogger(__name__)

# Adapter registry slot for the merged + CT2-converted Whisper fine-tune.
# When this directory exists at create_stt_engine() time and the caller did not
# pass an explicit model_id, the cuda/cpu branch loads from here instead of the
# off-the-shelf large-v3-turbo. The user can always override via the explicit
# model_id kwarg or STARK_STT__WHISPER_CUDA_MODEL env var.
_WHISPER_CT2_ACTIVE_PATH = Path("adapters/whisper_turbo_ct2/active")

# Adapter registry slot for the CT2-converted MarianMT vendor models (v2026.8).
# Built by ``scripts/convert_marian_ct2.py``. One subdir per direction.
_MARIAN_CT2_ROOT = Path("adapters/marian_ct2")


def _resolve_ct2_whisper_model(explicit_model_id: str | None) -> str:
    """Pick the local merged W16 CT2 dir when present, else fall back to the off-the-shelf id."""
    if explicit_model_id is not None:
        return explicit_model_id
    if _WHISPER_CT2_ACTIVE_PATH.exists() and (_WHISPER_CT2_ACTIVE_PATH / "model.bin").exists():
        logger.info("STT: using local CT2 fine-tune at %s", _WHISPER_CT2_ACTIVE_PATH)
        return str(_WHISPER_CT2_ACTIVE_PATH)
    return "large-v3-turbo"


def _resolve_ct2_marian_model(direction: str, explicit_path: str | None) -> str | None:
    """Return a usable CT2 Marian directory for *direction* or None if not available.

    *direction* is "en-es" or "es-en". Mirrors ``_resolve_ct2_whisper_model``:
    explicit override wins; otherwise check the active adapter slot.
    """
    if explicit_path is not None and Path(explicit_path).exists():
        return explicit_path
    candidate = _MARIAN_CT2_ROOT / direction / "active"
    if candidate.exists() and (candidate / "model.bin").exists():
        logger.info("Marian: using local CT2 model at %s", candidate)
        return str(candidate)
    return None


def _marian_direction_from_langs(source_lang: str, target_lang: str) -> str:
    """Map (source, target) lang codes to the adapter directory name."""
    src = (source_lang or "en").lower()
    tgt = (target_lang or "es").lower()
    if src == "en" and tgt == "es":
        return "en-es"
    if src == "es" and tgt == "en":
        return "es-en"
    raise ValueError(
        f"unsupported MarianMT direction: {src!r} -> {tgt!r} (only 'en-es' and 'es-en' have CT2 adapters today)"
    )


def create_stt_engine(
    backend: str = "auto",
    model_id: str | None = None,
    fallback_threshold: float | None = None,
    hallucination_threshold: float | None = None,
    fallback_on_low_conf: bool | None = None,
    spec_decode: bool = False,
    draft_model_id: str | None = None,
    stt_backend: str = "auto",
    **kwargs: Any,
) -> STTEngine:
    """Create an STT engine for the given backend.

    Args:
        backend:                 Hardware tier: "mlx", "cuda", "cpu", or "auto"
                                 (detect best). Decides which engine *families*
                                 are available.
        stt_backend:             Whisper implementation choice within a hardware
                                 tier: "auto" (default), "faster-whisper", "hf",
                                 or "mlx". On cuda/cpu, "auto" → "faster-whisper".
                                 On mlx hardware, "auto" → "mlx".
        model_id:                Override the default model identifier.
        fallback_threshold:      avg_logprob below which to retry with fallback
                                 model.  ``None`` uses the engine default (-1.2).
        hallucination_threshold: compression_ratio above which to retry.
                                 ``None`` uses the engine default (2.4).
        fallback_on_low_conf:    Enable/disable quality-based fallback retry.
                                 ``None`` uses the engine default (True).
        spec_decode:             Legacy shortcut for ``stt_backend="hf"``. NO
                                 default draft is attached — see the note on
                                 ``draft_model_id`` below. Useful only when the
                                 caller has a verified-compatible draft.
        draft_model_id:          Draft model for spec decode. **No default** —
                                 callers must supply a verified-compatible draft.
                                 The 2026-04-13 test (see
                                 ``docs/archive/v2026.5/spec_decode_research.md``)
                                 found that distil-large-v3.5 → whisper-large-v3-turbo
                                 produces 10× slower inference with hallucinated
                                 repetitions because they have incompatible
                                 decoder architectures (distil-v3.5 was distilled
                                 from large-v3's 32-layer decoder, turbo has 4).
                                 Valid pairing: turbo (draft, 4 layers) → large-v3
                                 (target, 32 layers), but that path costs latency
                                 for quality so it isn't wired as a default.
        **kwargs:                Forwarded to the engine constructor.

    Returns:
        An *unloaded* ``STTEngine`` instance.  Call ``.load()`` to initialise.
    """
    if backend == "auto":
        backend = _detect_backend()

    # Resolve stt_backend (Whisper implementation choice).
    if spec_decode and stt_backend == "faster-whisper":
        raise ValueError(
            "stt_backend='faster-whisper' is incompatible with spec_decode=True; "
            "use stt_backend='hf' (the spec-decode path is HF-only)"
        )
    if spec_decode:
        stt_backend = "hf"
    if stt_backend == "auto":
        stt_backend = "mlx" if backend == "mlx" else "faster-whisper"
    if stt_backend == "hf" and backend == "cuda" and not spec_decode:
        logger.warning(
            "stt_backend='hf' on CUDA without spec_decode: gives up CTranslate2 perf "
            "for no obvious gain. Use stt_backend='faster-whisper' or pass spec_decode=True."
        )

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

    if stt_backend == "hf":
        from engines.hf_whisper_engine import HFWhisperEngine

        # Extract HF-specific kwargs so we can forward only what the engine accepts
        hf_only_kwargs = {}
        for k in ("compile_mode", "warmup_seconds", "torch_dtype"):
            if k in kwargs:
                hf_only_kwargs[k] = kwargs.pop(k)
        if spec_decode and draft_model_id is None:
            # Refuse to silently attach distil-v3.5 — that pairing is broken with
            # whisper-large-v3-turbo (see docs/archive/v2026.5/spec_decode_research.md).
            raise ValueError(
                "spec_decode=True requires an explicit draft_model_id. "
                "The legacy default (distil-whisper/distil-large-v3.5) is "
                "incompatible with whisper-large-v3-turbo's 4-layer decoder "
                "(distil-v3.5 was distilled from large-v3's 32-layer decoder). "
                "See docs/archive/v2026.5/spec_decode_research.md for verified pairings."
            )
        return HFWhisperEngine(
            model_id=model_id or "openai/whisper-large-v3-turbo",
            draft_model_id=draft_model_id,
            device=backend if backend in ("cuda", "cpu") else None,
            **hf_only_kwargs,
            **kwargs,
        )

    if stt_backend == "mlx":
        from engines.mlx_engine import MLXWhisperEngine

        return MLXWhisperEngine(
            model_id=model_id or "mlx-community/whisper-large-v3-turbo",
            **merged_kwargs,
        )

    if stt_backend == "faster-whisper":
        if backend not in ("cuda", "cpu"):
            raise ValueError(f"stt_backend='faster-whisper' requires backend='cuda' or 'cpu' (got {backend!r})")
        from engines.cuda_engine import FasterWhisperEngine

        resolved_model_id = _resolve_ct2_whisper_model(model_id)
        return FasterWhisperEngine(
            model_id=resolved_model_id,
            device=backend,
            **merged_kwargs,
        )

    raise ValueError(f"Unsupported stt_backend: {stt_backend!r}")


def create_translation_engine(
    backend: str = "auto",
    model_id: str | None = None,
    engine_type: str = "gemma",
    streaming: bool = False,
    assistant_model_id: str | None = None,
    **kwargs: Any,
) -> TranslationEngine:
    """Create a translation engine.

    Args:
        backend:             "mlx", "cuda", "cpu", or "auto" (detect best available).
        model_id:            Override the default model identifier (or for
                             ``engine_type="llamacpp"``, the llama-server URL).
        engine_type:         "gemma" for HF TranslateGemma/Gemma 4 (default),
                             "marian" for MarianMT, "llamacpp" for llama.cpp via
                             HTTP. **For CUDA deployment, prefer "llamacpp"** —
                             v2026.5 Phase 1A benchmark shows 5–9× speedup and
                             4× VRAM reduction vs HF NF4 (see
                             ``docs/archive/v2026.5/BENCHMARK.md``). Caller is
                             responsible for starting llama-server first; see
                             ``start_server.sh``.
        streaming:           Use streaming engine (CUDA HF only).
        assistant_model_id:  Optional draft model for HF spec decode (CUDA only).
                             For llama.cpp spec decode, pass ``-md`` to llama-server
                             instead.
        **kwargs:            Forwarded to the engine constructor.

    Returns:
        An *unloaded* ``TranslationEngine`` instance.  Call ``.load()`` first.
    """
    if backend == "auto":
        backend = _detect_backend()

    if engine_type == "marian":
        # Marian-specific kwargs. Pop them off kwargs so the residual dict is
        # safe to forward to either engine (HF and CT2 share most names).
        marian_backend = kwargs.pop("marian_backend", "auto")
        compute_type = kwargs.pop("compute_type", "int8_float16")
        max_new_tokens = kwargs.pop("max_new_tokens", 128)
        warmup_passes = kwargs.pop("warmup_passes", 2)
        source_lang = kwargs.pop("source_lang", "en")
        target_lang = kwargs.pop("target_lang", "es")
        ct2_path_override = kwargs.pop("ct2_path", None)

        if marian_backend not in ("auto", "ct2", "hf"):
            raise ValueError(f"marian_backend must be one of 'auto', 'ct2', 'hf' (got {marian_backend!r})")

        try:
            direction = _marian_direction_from_langs(source_lang, target_lang)
        except ValueError:
            # Unsupported direction → can't route to CT2, drop to HF with explicit
            # model_id (caller knows what they're doing).
            direction = None

        ct2_dir = _resolve_ct2_marian_model(direction, ct2_path_override) if direction is not None else None

        # Strict ct2 mode: explicit request, but no model.bin present.
        if marian_backend == "ct2" and ct2_dir is None:
            raise ValueError(
                "marian_backend='ct2' requested but no CT2 model found at "
                f"{_MARIAN_CT2_ROOT / (direction or '<unknown>') / 'active'}/model.bin. "
                "Run scripts/convert_marian_ct2.py first, or set marian_backend='hf'."
            )

        # auto + CT2 available + ctranslate2 importable → CT2 path.
        if marian_backend in ("auto", "ct2") and ct2_dir is not None:
            try:
                import ctranslate2  # noqa: F401  -- presence check only
            except ImportError:
                if marian_backend == "ct2":
                    raise  # honor the strict mode signal
                logger.info("Marian: CT2 model present but ctranslate2 not importable; falling back to HF.")
            else:
                from engines.cuda_engine import MarianCT2Engine

                return MarianCT2Engine(
                    model_dir=ct2_dir,
                    source_lang=source_lang,
                    target_lang=target_lang,
                    device=backend if backend in ("cuda", "cpu") else "auto",
                    compute_type=compute_type,
                    max_new_tokens=max_new_tokens,
                    warmup_passes=warmup_passes,
                    **kwargs,
                )

        # Fall through to the HF path. ``model_id`` overrides the default for
        # callers passing a custom HF repo (rare but supported).
        from engines.marian_hf_engine import MarianHFEngine

        return MarianHFEngine(
            model_id=model_id or "Helsinki-NLP/opus-mt-en-es",
            max_new_tokens=max_new_tokens,
            warmup_passes=warmup_passes,
            **kwargs,
        )

    if engine_type == "llamacpp":
        from engines.llamacpp_engine import LlamaCppEngine

        return LlamaCppEngine(
            server_url=model_id or "http://127.0.0.1:8090",
            **kwargs,
        )

    if backend == "mlx":
        from engines.mlx_engine import MLXGemmaEngine

        return MLXGemmaEngine(
            model_id=model_id or "mlx-community/translategemma-4b-it-4bit",
            **kwargs,
        )
    elif backend == "cuda":
        if streaming:
            from engines.cuda_engine import CUDAGemmaStreamingEngine

            return CUDAGemmaStreamingEngine(
                model_id=model_id or "google/translategemma-4b-it",
                assistant_model_id=assistant_model_id,
                **kwargs,
            )
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
