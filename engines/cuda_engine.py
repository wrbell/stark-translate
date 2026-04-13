"""CUDA backend engines for NVIDIA GPU inference.

Provides:
  - FasterWhisperEngine        -- STT via faster-whisper (CTranslate2 backend)
  - CUDAGemmaEngine            -- Translation via transformers + bitsandbytes 4-bit (basic)
  - CUDAGemmaStreamingEngine   -- Enhanced: streaming, prompt cache, speculative decoding
  - detect_vram_tier()         -- Auto-detect GPU tier (full_ab / 4b_only / marian)
"""

from __future__ import annotations

import copy
import logging
import time
from collections.abc import Callable
from threading import Thread

import numpy as np

from engines.base import (
    STTEngine,
    STTResult,
    TranslationEngine,
    TranslationResult,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Guard CUDA-specific imports
# ---------------------------------------------------------------------------
try:
    from faster_whisper import WhisperModel

    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    WhisperModel = None
    FASTER_WHISPER_AVAILABLE = False

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

try:
    import bitsandbytes  # noqa: F401 -- presence check only

    BITSANDBYTES_AVAILABLE = True
except ImportError:
    BITSANDBYTES_AVAILABLE = False


# ---------------------------------------------------------------------------
# Faster-Whisper STT (CTranslate2 backend)
# ---------------------------------------------------------------------------


class FasterWhisperEngine(STTEngine):
    """Speech-to-text engine wrapping faster-whisper on CUDA or CPU.

    Constructor args:
        model_id:                Model size or HuggingFace repo
                                 (default: "large-v3-turbo").
        compute_type:            CTranslate2 compute type (default: "int8").
        device:                  "cuda" or "cpu" (default: "cuda").
        fallback_model_id:       Model to retry with on low-confidence output
                                 (default: "large-v3").
        fallback_threshold:      avg_logprob below which to retry with fallback
                                 model (default: -1.2).
        hallucination_threshold: compression_ratio above which output is flagged
                                 as hallucination and retried (default: 2.4).
        fallback_on_low_conf:    Enable/disable quality-based fallback retry
                                 (default: True).
    """

    def __init__(
        self,
        model_id: str = "large-v3-turbo",
        compute_type: str = "int8",
        device: str = "cuda",
        fallback_model_id: str = "large-v3",
        fallback_threshold: float = -1.2,
        hallucination_threshold: float = 2.4,
        fallback_on_low_conf: bool = True,
    ):
        if not FASTER_WHISPER_AVAILABLE:
            raise RuntimeError("faster-whisper is not installed. Install with: pip install faster-whisper")
        self._model_id_str = model_id
        self._compute_type = compute_type
        self._device = device
        self._fallback_model_id = fallback_model_id
        self._fallback_threshold = fallback_threshold
        self._hallucination_threshold = hallucination_threshold
        self._fallback_on_low_conf = fallback_on_low_conf
        self._model = None
        self._fallback_model = None
        self._fallback_loaded = False
        self._loaded = False

    # -- public interface ----------------------------------------------------

    def load(self) -> None:
        """Create the WhisperModel and warm up with 1 s of silence."""
        logger.info(
            "Loading faster-whisper %s (device=%s, compute=%s)...",
            self._model_id_str,
            self._device,
            self._compute_type,
        )
        t0 = time.time()
        self._model = WhisperModel(
            self._model_id_str,
            device=self._device,
            compute_type=self._compute_type,
        )

        # Warm up -- consume the generator to trigger actual inference
        silence = np.zeros(16000, dtype=np.float32)
        segments, _ = self._model.transcribe(silence, language="en")
        list(segments)

        logger.info("Whisper ready (%s) (%.1fs)", self._model_id_str, time.time() - t0)
        self._loaded = True

    def transcribe(
        self,
        audio: np.ndarray,
        *,
        language: str = "en",
        initial_prompt: str | None = None,
        word_timestamps: bool = False,
        beam_size: int | None = None,
    ) -> STTResult:
        """Transcribe *audio* (16 kHz float32 mono) to text.

        When ``fallback_on_low_conf`` is enabled, automatically retries with
        the fallback model (``large-v3`` by default) if the primary result
        has ``avg_logprob < fallback_threshold`` or
        ``compression_ratio > hallucination_threshold``.

        Returns the better of the two results (higher confidence).
        """
        if not self._loaded:
            raise RuntimeError("Engine not loaded -- call load() first")

        primary_result = self._raw_transcribe(
            audio,
            model=self._model,
            language=language,
            initial_prompt=initial_prompt,
            word_timestamps=word_timestamps,
            beam_size=beam_size,
        )

        # -- quality-based fallback retry ------------------------------------
        if not self._fallback_on_low_conf:
            return primary_result

        needs_fallback = self._should_fallback(primary_result)
        if not needs_fallback:
            return primary_result

        # Lazy-load fallback model on first use
        if not self._fallback_loaded:
            self._load_fallback_model()

        logger.info(
            "Fallback triggered (avg_logprob=%.3f, compression_ratio=%.2f) -- retrying with %s",
            primary_result.avg_logprob or 0.0,
            primary_result.compression_ratio or 0.0,
            self._fallback_model_id,
        )

        retry_result = self._raw_transcribe(
            audio,
            model=self._fallback_model,
            language=language,
            initial_prompt=initial_prompt,
            word_timestamps=word_timestamps,
            beam_size=beam_size,
        )

        # Pick the better result (higher confidence = less negative logprob)
        chosen, chosen_label = self._pick_best(primary_result, retry_result)
        chosen.used_fallback = True

        logger.info(
            "Fallback result: chose %s (primary conf=%.2f avg_lp=%.3f, retry conf=%.2f avg_lp=%.3f)",
            chosen_label,
            primary_result.confidence or 0.0,
            primary_result.avg_logprob or 0.0,
            retry_result.confidence or 0.0,
            retry_result.avg_logprob or 0.0,
        )

        # Log for active learning pipeline
        self._log_fallback(primary_result, retry_result, audio, chosen_label)

        return chosen

    # -- internal helpers ----------------------------------------------------

    def _raw_transcribe(
        self,
        audio: np.ndarray,
        *,
        model,
        language: str = "en",
        initial_prompt: str | None = None,
        word_timestamps: bool = False,
        beam_size: int | None = None,
    ) -> STTResult:
        """Run faster-whisper transcription against a specific model instance.

        This is the core transcription logic extracted so it can be called
        for both the primary and fallback models.
        """
        t0 = time.perf_counter()
        transcribe_kwargs: dict = dict(
            language=language,
            condition_on_previous_text=False,
            initial_prompt=initial_prompt,
            word_timestamps=word_timestamps,
        )
        if beam_size is not None:
            transcribe_kwargs["beam_size"] = beam_size
        segments_gen, _info = model.transcribe(audio, **transcribe_kwargs)

        # Consume the generator to get all segments
        segments_list = list(segments_gen)
        latency_ms = (time.perf_counter() - t0) * 1000

        # Build English text from segments
        english = " ".join(seg.text.strip() for seg in segments_list).strip()

        # -- extract segment metadata (mirrors dry_run_ab._run_stt_faster_whisper)
        confidence = None
        segment_meta = []
        low_conf_words = []
        overall_avg_logprob = None
        overall_compression_ratio = None
        if segments_list:
            avg_logprobs = []
            compression_ratios = []
            for seg in segments_list:
                meta = {
                    "avg_logprob": getattr(seg, "avg_logprob", None),
                    "no_speech_prob": getattr(seg, "no_speech_prob", None),
                    "compression_ratio": getattr(seg, "compression_ratio", None),
                }
                segment_meta.append(meta)
                if meta["avg_logprob"] is not None:
                    avg_logprobs.append(meta["avg_logprob"])
                if meta["compression_ratio"] is not None:
                    compression_ratios.append(meta["compression_ratio"])
                # Per-word confidence from faster-whisper Word objects
                for w in seg.words or []:
                    prob = getattr(w, "probability", 1.0)
                    if prob < 0.5:
                        low_conf_words.append(
                            {
                                "word": getattr(w, "word", ""),
                                "probability": round(prob, 3),
                                "start": getattr(w, "start", None),
                                "end": getattr(w, "end", None),
                            }
                        )
            if avg_logprobs:
                overall_avg_logprob = sum(avg_logprobs) / len(avg_logprobs)
                confidence = round(min(1.0, max(0.0, 1.0 + overall_avg_logprob)), 2)
            if compression_ratios:
                overall_compression_ratio = max(compression_ratios)

        return STTResult(
            text=english,
            latency_ms=latency_ms,
            confidence=confidence,
            avg_logprob=overall_avg_logprob,
            compression_ratio=overall_compression_ratio,
            segments=segment_meta,
            low_confidence_words=low_conf_words,
        )

    def _should_fallback(self, result: STTResult) -> bool:
        """Check whether a transcription result warrants a fallback retry."""
        if result.avg_logprob is not None and result.avg_logprob < self._fallback_threshold:
            return True
        if result.compression_ratio is not None and result.compression_ratio > self._hallucination_threshold:
            return True
        return False

    def _load_fallback_model(self) -> None:
        """Lazily load the fallback Whisper model (warmup with 1s silence).

        Only called on first fallback trigger, not at startup.
        """
        logger.info(
            "Lazy-loading fallback model %s (faster-whisper, device=%s, compute=%s)...",
            self._fallback_model_id,
            self._device,
            self._compute_type,
        )
        t0 = time.time()
        self._fallback_model = WhisperModel(
            self._fallback_model_id,
            device=self._device,
            compute_type=self._compute_type,
        )
        # Warm up
        silence = np.zeros(16000, dtype=np.float32)
        segments, _ = self._fallback_model.transcribe(silence, language="en")
        list(segments)
        logger.info(
            "Fallback model ready (%s) (%.1fs)",
            self._fallback_model_id,
            time.time() - t0,
        )
        self._fallback_loaded = True

    @staticmethod
    def _pick_best(primary: STTResult, retry: STTResult) -> tuple:
        """Return (best_result, label) choosing the higher-confidence result.

        Compares avg_logprob (less negative = better).

        Returns:
            Tuple of (STTResult, str) where str is "original" or "retry".
        """
        primary_score = primary.avg_logprob if primary.avg_logprob is not None else -999.0
        retry_score = retry.avg_logprob if retry.avg_logprob is not None else -999.0

        if retry_score > primary_score:
            return retry, "retry"
        return primary, "original"

    def _log_fallback(
        self,
        original: STTResult,
        retry: STTResult,
        audio: np.ndarray,
        chosen_label: str,
    ) -> None:
        """Log the fallback event for active learning.

        Computes a simple hash of the audio chunk for deduplication.
        """
        import hashlib

        from engines.active_learning import log_fallback_event

        audio_bytes = audio.tobytes()
        audio_hash = hashlib.sha256(audio_bytes).hexdigest()[:16]

        log_fallback_event(
            original=original,
            retry=retry,
            audio_hash=audio_hash,
            chosen=chosen_label,
            primary_model=self._model_id_str,
            fallback_model=self._fallback_model_id,
        )

    def unload(self) -> None:
        """Release primary and fallback models from memory."""
        self._model = None
        self._fallback_model = None
        self._fallback_loaded = False
        self._loaded = False
        logger.info("FasterWhisperEngine unloaded (%s)", self._model_id_str)

    @property
    def model_id(self) -> str:
        return self._model_id_str

    @property
    def backend(self) -> str:
        return self._device  # "cuda" or "cpu"


# ---------------------------------------------------------------------------
# CUDA TranslateGemma (transformers + bitsandbytes 4-bit)
# ---------------------------------------------------------------------------


class CUDAGemmaEngine(TranslationEngine):
    """Translation engine wrapping TranslateGemma on CUDA with 4-bit quantization.

    Uses transformers ``AutoModelForCausalLM`` with bitsandbytes NF4.

    Constructor args:
        model_id:  HuggingFace repo (default: "google/translategemma-4b-it").
    """

    def __init__(
        self,
        model_id: str = "google/translategemma-4b-it",
    ):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is not installed.")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. CUDAGemmaEngine requires an NVIDIA GPU.")
        self._model_id_str = model_id
        self._model = None
        self._tokenizer = None
        self._loaded = False

    # -- public interface ----------------------------------------------------

    def load(self) -> None:
        """Load TranslateGemma with bitsandbytes 4-bit quantization and apply EOS fix."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not BITSANDBYTES_AVAILABLE:
            raise RuntimeError(
                "bitsandbytes is not installed. Required for 4-bit quantization. Install with: pip install bitsandbytes"
            )

        logger.info("Loading %s (CUDA 4-bit)...", self._model_id_str)
        t0 = time.time()

        self._tokenizer = AutoTokenizer.from_pretrained(self._model_id_str)
        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_id_str,
            load_in_4bit=True,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        )
        self._model.eval()

        # -- EOS fix (mirrors dry_run_ab.load_cuda_translation_models) --------
        # TranslateGemma uses <end_of_turn> (id=106) as its actual EOS, but
        # the tokenizer default is <eos> (id=1) which the model never generates.
        eot_id = self._tokenizer.convert_tokens_to_ids("<end_of_turn>")
        default_eos = self._tokenizer.eos_token_id
        if hasattr(self._tokenizer, "_eos_token_ids"):
            self._tokenizer._eos_token_ids.add(eot_id)
        else:
            self._tokenizer._eos_token_ids = {default_eos, eot_id}
        logger.info(
            "EOS fix applied: added <end_of_turn> (id=%d) to EOS set",
            eot_id,
        )

        logger.info("TranslateGemma ready (%.1fs)", time.time() - t0)
        self._loaded = True

    def translate(
        self,
        text: str,
        *,
        source_lang: str = "en",
        target_lang: str = "es",
    ) -> TranslationResult:
        """Translate *text* using TranslateGemma on CUDA.

        Uses ``torch.no_grad()`` and greedy decoding (``do_sample=False``).
        """
        if not self._loaded:
            raise RuntimeError("Engine not loaded -- call load() first")

        if self._model is None or self._tokenizer is None:
            return TranslationResult(text="(model not loaded)", latency_ms=0.0)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "source_lang_code": source_lang, "target_lang_code": target_lang, "text": text}
                ],
            }
        ]
        prompt = self._tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        prompt = prompt.to("cuda")

        # Dynamic max-tokens cap: Spanish is ~15-25% longer than English
        input_words = len(text.split())
        max_tok = max(32, int(input_words * 1.8))

        t0 = time.perf_counter()
        with torch.no_grad():
            output = self._model.generate(prompt, max_new_tokens=max_tok, do_sample=False)
        generated = output[0][prompt.shape[1] :]
        result = self._tokenizer.decode(generated, skip_special_tokens=False)
        latency_ms = (time.perf_counter() - t0) * 1000

        clean = result.split("<end_of_turn>")[0].strip()
        out_tokens = len(generated)
        gen_tps = out_tokens / (latency_ms / 1000) if latency_ms > 0 else 0.0

        return TranslationResult(
            text=clean,
            latency_ms=latency_ms,
            tokens_per_second=gen_tps,
        )

    def unload(self) -> None:
        """Release model from GPU memory."""
        del self._model
        del self._tokenizer
        self._model = None
        self._tokenizer = None
        self._loaded = False
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("CUDAGemmaEngine unloaded (%s)", self._model_id_str)

    @property
    def model_id(self) -> str:
        return self._model_id_str

    @property
    def backend(self) -> str:
        return "cuda"


# ---------------------------------------------------------------------------
# VRAM tier detection
# ---------------------------------------------------------------------------

# Thresholds in MB
_VRAM_FULL_AB_MB = 15_000  # >=15 GB → 4B + 12B + Whisper
_VRAM_4B_ONLY_MB = 5_500  # >=5.5 GB → 4B + Whisper
# Below 5.5 GB → MarianMT only


def detect_vram_tier() -> tuple[str, int]:
    """Auto-detect GPU VRAM and determine inference configuration tier.

    Returns:
        (tier_name, vram_mb) where tier_name is one of:
        - ``"full_ab"``  (≥15 GB): Whisper + 4B + 12B (A/B comparison)
        - ``"4b_only"``  (≥5.5 GB): Whisper + 4B
        - ``"marian"``   (<5.5 GB): MarianMT only (no Gemma)
        - ``"none"``     (no CUDA): CPU fallback
    """
    if not TORCH_AVAILABLE or not torch.cuda.is_available():
        return ("none", 0)

    props = torch.cuda.get_device_properties(0)
    vram_mb = props.total_memory // (1024 * 1024)
    gpu_name = props.name

    if vram_mb >= _VRAM_FULL_AB_MB:
        tier = "full_ab"
    elif vram_mb >= _VRAM_4B_ONLY_MB:
        tier = "4b_only"
    else:
        tier = "marian"

    logger.info(
        "VRAM tier: %s (%s, %d MB)",
        tier,
        gpu_name,
        vram_mb,
    )
    return (tier, vram_mb)


# ---------------------------------------------------------------------------
# KV cache utilities
# ---------------------------------------------------------------------------


def _clone_past_key_values(
    past_key_values: tuple[tuple[object, ...], ...],
) -> tuple[tuple[object, ...], ...]:
    """Deep-clone ``past_key_values`` using tensor ``.clone()`` (fast GPU copy).

    ``copy.deepcopy()`` on GPU tensors goes through pickle and is slow.
    This traverses the nested tuple structure and clones each tensor in-place.
    """
    if past_key_values is None:
        return None  # type: ignore[return-value]

    # past_key_values is a tuple of layer tuples, each containing (key, value) tensors.
    # Some model implementations use DynamicCache objects instead of raw tuples.
    # Handle both cases.
    if hasattr(past_key_values, "key_cache"):
        # DynamicCache — use copy.deepcopy which calls __deepcopy__ on the cache
        return copy.deepcopy(past_key_values)

    return tuple(tuple(t.clone() if hasattr(t, "clone") else t for t in layer) for layer in past_key_values)


# ---------------------------------------------------------------------------
# Enhanced CUDA TranslateGemma (streaming + prompt cache + speculative)
# ---------------------------------------------------------------------------


class CUDAGemmaStreamingEngine(TranslationEngine):
    """Enhanced CUDA TranslateGemma with streaming, prompt caching, and speculative decoding.

    Provides full feature parity with the MLX inference path:

    - **Prompt cache**: Pre-computes KV cache for the fixed chat template prefix,
      saving ~50-80ms per translation by skipping re-prefilling ~30-40 template tokens.
    - **Streaming**: Token-by-token output via ``TextIteratorStreamer``, with configurable
      batching (default: every 3 tokens) for WebSocket broadcast.
    - **Speculative decoding**: Optional assistant (draft) model for faster 12B generation.

    Constructor args:
        model_id:            HuggingFace repo (default: ``"google/translategemma-4b-it"``).
        use_prompt_cache:    Pre-compute KV cache for fixed prefix (default: True).
        assistant_model_id:  Optional 4B model repo for speculative decoding (12B only).
        streaming_batch_size: Tokens per WebSocket batch during streaming (default: 3).
    """

    def __init__(
        self,
        model_id: str = "google/translategemma-4b-it",
        use_prompt_cache: bool = True,
        assistant_model_id: str | None = None,
        streaming_batch_size: int = 3,
        model_family: str = "translategemma",
    ):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is not installed.")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available.")
        self._model_id_str = model_id
        self._use_prompt_cache = use_prompt_cache
        self._assistant_model_id = assistant_model_id
        self._streaming_batch_size = streaming_batch_size
        self._model_family = model_family

        self._model = None
        self._tokenizer = None
        self._assistant_model = None
        self._prompt_cache_pkv = None  # past_key_values template
        self._suffix_tokens: list[int] = []
        self._loaded = False

    # -- public interface ----------------------------------------------------

    def load(self) -> None:
        """Load model with bitsandbytes 4-bit, apply EOS fix, build prompt cache."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not BITSANDBYTES_AVAILABLE:
            raise RuntimeError("bitsandbytes is not installed. Required for 4-bit quantization.")

        logger.info("Loading %s (CUDA 4-bit, streaming)...", self._model_id_str)
        t0 = time.time()

        self._tokenizer = AutoTokenizer.from_pretrained(self._model_id_str)
        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_id_str,
            load_in_4bit=True,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        )
        self._model.eval()

        # -- EOS fix --
        eot_id = self._tokenizer.convert_tokens_to_ids("<end_of_turn>")
        default_eos = self._tokenizer.eos_token_id
        if hasattr(self._tokenizer, "_eos_token_ids"):
            self._tokenizer._eos_token_ids.add(eot_id)
        else:
            self._tokenizer._eos_token_ids = {default_eos, eot_id}
        logger.info("EOS fix applied: added <end_of_turn> (id=%d)", eot_id)

        # -- Prompt cache --
        if self._use_prompt_cache:
            self._prompt_cache_pkv, self._suffix_tokens = self._build_prompt_cache()

        # -- Optional assistant model for speculative decoding --
        if self._assistant_model_id:
            logger.info(
                "Loading assistant model %s for speculative decoding...",
                self._assistant_model_id,
            )
            self._assistant_model = AutoModelForCausalLM.from_pretrained(
                self._assistant_model_id,
                load_in_4bit=True,
                torch_dtype=torch.bfloat16,
                device_map="cuda",
            )
            self._assistant_model.eval()
            logger.info("Assistant model ready")

        logger.info(
            "CUDAGemmaStreamingEngine ready (%.1fs, cache=%s, assistant=%s)",
            time.time() - t0,
            self._prompt_cache_pkv is not None,
            self._assistant_model_id or "none",
        )
        self._loaded = True

    def translate(
        self,
        text: str,
        *,
        source_lang: str = "en",
        target_lang: str = "es",
    ) -> TranslationResult:
        """Translate *text* using prompt cache for speed.

        If speculative decoding is configured, uses the assistant model.
        """
        if not self._loaded:
            raise RuntimeError("Engine not loaded -- call load() first")

        input_words = len(text.split())
        max_tok = max(32, int(input_words * 1.8))

        t0 = time.perf_counter()

        if self._prompt_cache_pkv is not None and self._suffix_tokens:
            # -- Cached path: prefix KV already computed --
            text_ids = self._tokenizer.encode(text, add_special_tokens=False)
            dynamic_ids = text_ids + self._suffix_tokens
            input_ids = torch.tensor([dynamic_ids], device="cuda")
            cached_pkv = _clone_past_key_values(self._prompt_cache_pkv)

            gen_kwargs: dict = dict(
                input_ids=input_ids,
                past_key_values=cached_pkv,
                max_new_tokens=max_tok,
                do_sample=False,
            )
        else:
            # -- Full prompt fallback --
            input_ids = self._build_full_prompt(text, source_lang, target_lang)
            gen_kwargs = dict(
                input_ids=input_ids,
                max_new_tokens=max_tok,
                do_sample=False,
            )

        if self._assistant_model is not None:
            gen_kwargs["assistant_model"] = self._assistant_model

        with torch.no_grad():
            output = self._model.generate(**gen_kwargs)

        generated = output[0][input_ids.shape[1] :]
        result = self._tokenizer.decode(generated, skip_special_tokens=False)
        latency_ms = (time.perf_counter() - t0) * 1000

        clean = result.split("<end_of_turn>")[0].strip()
        # Gemma 4 instruct may prepend preamble like "Here is the translation:"
        if self._model_family == "gemma4":
            for prefix in ("Here is the translation:\n", "Here is the translation:"):
                if clean.startswith(prefix):
                    clean = clean[len(prefix) :].strip()
                    break
        out_tokens = len(generated)
        gen_tps = out_tokens / (latency_ms / 1000) if latency_ms > 0 else 0.0

        return TranslationResult(
            text=clean,
            latency_ms=latency_ms,
            tokens_per_second=gen_tps,
        )

    def translate_streaming(
        self,
        text: str,
        *,
        source_lang: str = "en",
        target_lang: str = "es",
        token_callback: Callable[[str, int], None] | None = None,
    ) -> TranslationResult:
        """Translate with token-by-token streaming via ``TextIteratorStreamer``.

        Args:
            text: Source text to translate.
            source_lang: Source language code (default: ``"en"``).
            target_lang: Target language code (default: ``"es"``).
            token_callback: Called with ``(accumulated_text, tokens_so_far)``
                every ``streaming_batch_size`` tokens. Used to push partial
                translations to the WebSocket broadcaster.

        Returns:
            Final ``TranslationResult`` with complete translation.
        """
        if not self._loaded:
            raise RuntimeError("Engine not loaded -- call load() first")

        from transformers import TextIteratorStreamer

        input_words = len(text.split())
        max_tok = max(32, int(input_words * 1.8))

        if self._prompt_cache_pkv is not None and self._suffix_tokens:
            text_ids = self._tokenizer.encode(text, add_special_tokens=False)
            dynamic_ids = text_ids + self._suffix_tokens
            input_ids = torch.tensor([dynamic_ids], device="cuda")
            cached_pkv = _clone_past_key_values(self._prompt_cache_pkv)
            gen_kwargs: dict = dict(
                input_ids=input_ids,
                past_key_values=cached_pkv,
                max_new_tokens=max_tok,
                do_sample=False,
            )
        else:
            input_ids = self._build_full_prompt(text, source_lang, target_lang)
            gen_kwargs = dict(
                input_ids=input_ids,
                max_new_tokens=max_tok,
                do_sample=False,
            )

        # Note: speculative decoding is incompatible with TextIteratorStreamer
        # (assistant_model generates tokens internally, streamer can't intercept).
        # Streaming mode always uses standard generation.

        streamer = TextIteratorStreamer(
            self._tokenizer,
            skip_special_tokens=False,
            skip_prompt=True,
        )
        gen_kwargs["streamer"] = streamer

        t0 = time.perf_counter()
        gen_thread = Thread(
            target=self._generate_no_grad,
            kwargs=gen_kwargs,
            daemon=True,
        )
        gen_thread.start()

        accumulated = ""
        tokens_generated = 0
        batch_size = self._streaming_batch_size

        for token_text in streamer:
            accumulated += token_text
            tokens_generated += 1

            if token_callback and tokens_generated % batch_size == 0:
                partial = accumulated.split("<end_of_turn>")[0].strip()
                token_callback(partial, tokens_generated)

        gen_thread.join(timeout=5.0)
        latency_ms = (time.perf_counter() - t0) * 1000

        clean = accumulated.split("<end_of_turn>")[0].strip()
        gen_tps = tokens_generated / (latency_ms / 1000) if latency_ms > 0 else 0.0

        return TranslationResult(
            text=clean,
            latency_ms=latency_ms,
            tokens_per_second=gen_tps,
        )

    # -- internal helpers ----------------------------------------------------

    def _generate_no_grad(self, **kwargs: object) -> None:
        """Run ``model.generate()`` inside ``torch.no_grad()`` (thread target)."""
        with torch.no_grad():
            self._model.generate(**kwargs)

    def _build_full_prompt(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
    ) -> object:
        """Build the full chat template prompt as input_ids tensor on CUDA."""
        if self._model_family == "gemma4":
            # Gemma 4 instruct: plain text translation instruction
            lang_names = {"en": "English", "es": "Spanish", "hi": "Hindi", "zh": "Chinese"}
            src_name = lang_names.get(source_lang, source_lang)
            tgt_name = lang_names.get(target_lang, target_lang)
            messages = [
                {
                    "role": "user",
                    "content": f"Translate the following {src_name} text to {tgt_name}. "
                    f"Output only the translation, nothing else.\n\n{text}",
                }
            ]
        else:
            # TranslateGemma: structured content with lang codes
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "source_lang_code": source_lang,
                            "target_lang_code": target_lang,
                            "text": text,
                        }
                    ],
                }
            ]
        prompt = self._tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        return prompt.to("cuda")

    def _build_prompt_cache(self) -> tuple[object | None, list[int]]:
        """Pre-compute KV cache for the fixed TranslateGemma chat template prefix.

        Uses the same SPLIT_HERE marker technique as the MLX ``_build_prompt_cache``:
        1. Build prompt with marker text in place of user input
        2. Tokenize and locate marker tokens in the sequence
        3. Run a forward pass on prefix tokens to generate ``past_key_values``
        4. Store as reusable template (cloned per request)

        Returns:
            ``(past_key_values, suffix_token_ids)`` or ``(None, [])`` on failure.
        """
        marker = "SPLIT_HERE"
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "source_lang_code": "en",
                        "target_lang_code": "es",
                        "text": marker,
                    }
                ],
            }
        ]

        full_prompt = self._tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        if isinstance(full_prompt, str):
            full_tokens = self._tokenizer.encode(full_prompt, add_special_tokens=False)
        else:
            full_tokens = list(full_prompt)

        marker_tokens = self._tokenizer.encode(marker, add_special_tokens=False)
        marker_len = len(marker_tokens)

        # Locate marker in token sequence
        prefix_end = None
        for i in range(len(full_tokens) - marker_len + 1):
            if full_tokens[i : i + marker_len] == marker_tokens:
                prefix_end = i
                break

        if prefix_end is None:
            logger.warning("Could not locate marker in prompt, skipping cache")
            return (None, [])

        prefix_tokens = full_tokens[:prefix_end]
        suffix_tokens = full_tokens[prefix_end + marker_len :]

        if len(prefix_tokens) < 3:
            logger.warning(
                "Prefix too short (%d tokens), skipping cache",
                len(prefix_tokens),
            )
            return (None, suffix_tokens)

        # Forward pass on prefix to generate past_key_values
        prefix_ids = torch.tensor([prefix_tokens], device="cuda")
        with torch.no_grad():
            outputs = self._model(prefix_ids, use_cache=True)

        pkv = outputs.past_key_values
        logger.info(
            "Prompt cache built: %d prefix tokens cached, %d suffix tokens",
            len(prefix_tokens),
            len(suffix_tokens),
        )
        return (pkv, suffix_tokens)

    def unload(self) -> None:
        """Release all models and caches from GPU memory."""
        del self._model
        del self._tokenizer
        del self._assistant_model
        del self._prompt_cache_pkv
        self._model = None
        self._tokenizer = None
        self._assistant_model = None
        self._prompt_cache_pkv = None
        self._suffix_tokens = []
        self._loaded = False
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("CUDAGemmaStreamingEngine unloaded (%s)", self._model_id_str)

    @property
    def model_id(self) -> str:
        return self._model_id_str

    @property
    def backend(self) -> str:
        return "cuda"
