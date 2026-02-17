"""MLX backend engines for Apple Silicon inference.

Provides:
  - MLXWhisperEngine  -- STT via mlx-whisper (distil-whisper / whisper-large-v3-turbo)
  - MLXGemmaEngine    -- Translation via mlx-lm (TranslateGemma 4B/12B 4-bit)
  - MarianEngine      -- Fast partial translation via MarianMT PyTorch (any backend)
"""

import copy
import logging
import threading
import time
from typing import Optional

import numpy as np

from engines.base import (
    STTEngine, STTResult,
    TranslationEngine, TranslationResult,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Guard MLX imports -- this module may be imported on non-Apple hardware
# ---------------------------------------------------------------------------
try:
    import mlx.core as mx
    import mlx_lm
    import mlx_whisper
    MLX_AVAILABLE = True
except ImportError:
    mx = None
    mlx_lm = None
    mlx_whisper = None
    MLX_AVAILABLE = False

# PyTorch is always available (used by MarianEngine and as a fallback)
import torch


# ---------------------------------------------------------------------------
# MLX Whisper STT
# ---------------------------------------------------------------------------

class MLXWhisperEngine(STTEngine):
    """Speech-to-text engine wrapping mlx-whisper.

    Constructor args:
        model_id:                HuggingFace repo for the primary model
                                 (default: mlx-community/whisper-large-v3-turbo).
        fallback_model_id:       Used for quality-based retry when the primary
                                 model returns low-confidence output
                                 (default: mlx-community/distil-whisper-large-v3).
        cache_limit_mb:          Metal memory cache limit in megabytes (default: 256).
        fallback_threshold:      avg_logprob below which to retry with fallback
                                 model (default: -1.2).
        hallucination_threshold: compression_ratio above which output is flagged
                                 as hallucination and retried (default: 2.4).
        fallback_on_low_conf:    Enable/disable the quality-based fallback retry
                                 (default: True).
    """

    def __init__(
        self,
        model_id: str = "mlx-community/whisper-large-v3-turbo",
        fallback_model_id: str = "mlx-community/distil-whisper-large-v3",
        cache_limit_mb: int = 256,
        fallback_threshold: float = -1.2,
        hallucination_threshold: float = 2.4,
        fallback_on_low_conf: bool = True,
    ):
        if not MLX_AVAILABLE:
            raise RuntimeError(
                "MLX is not available. MLXWhisperEngine requires Apple Silicon "
                "with mlx, mlx-whisper, and mlx-lm installed."
            )
        self._model_id = model_id
        self._fallback_model_id = fallback_model_id
        self._cache_limit_mb = cache_limit_mb
        self._fallback_threshold = fallback_threshold
        self._hallucination_threshold = hallucination_threshold
        self._fallback_on_low_conf = fallback_on_low_conf
        self._fallback_loaded = False
        self._loaded = False

    # -- public interface ----------------------------------------------------

    def load(self) -> None:
        """Download (if needed) and warm up the Whisper model.

        Sets the MLX Metal cache limit, then runs a 1-second silence through
        the model to trigger compilation and weight loading.  Falls back to
        ``fallback_model_id`` if the primary model errors out.
        """
        mx.set_cache_limit(self._cache_limit_mb * 1024 * 1024)

        logger.info("Loading %s (MLX)...", self._model_id)
        t0 = time.time()
        silence = np.zeros(16000, dtype=np.float32)
        try:
            mlx_whisper.transcribe(
                silence,
                path_or_hf_repo=self._model_id,
                condition_on_previous_text=False,
            )
            logger.info("Whisper ready (%s) (%.1fs)", self._model_id, time.time() - t0)
        except Exception as exc:
            logger.warning(
                "Primary model %s failed (%s), falling back to %s",
                self._model_id, exc, self._fallback_model_id,
            )
            self._model_id = self._fallback_model_id
            t0 = time.time()
            mlx_whisper.transcribe(
                silence,
                path_or_hf_repo=self._model_id,
                condition_on_previous_text=False,
            )
            logger.info("Whisper ready (%s) (%.1fs)", self._model_id, time.time() - t0)

        self._loaded = True

    def transcribe(
        self,
        audio: np.ndarray,
        *,
        language: str = "en",
        initial_prompt: Optional[str] = None,
        word_timestamps: bool = False,
    ) -> STTResult:
        """Transcribe *audio* (16 kHz float32 mono) to text.

        Returns an ``STTResult`` with segment-level metadata and per-word
        confidence extraction.  When ``fallback_on_low_conf`` is enabled,
        automatically retries with the fallback model if the primary result
        has ``avg_logprob < fallback_threshold`` or
        ``compression_ratio > hallucination_threshold``.

        **MLX Metal is NOT thread-safe** -- the fallback model is loaded and
        run on the same thread as the primary model.  No threading is used.
        """
        if not self._loaded:
            raise RuntimeError("Engine not loaded -- call load() first")

        primary_result = self._raw_transcribe(
            audio,
            model_repo=self._model_id,
            language=language,
            initial_prompt=initial_prompt,
            word_timestamps=word_timestamps,
        )

        # -- quality-based fallback retry ------------------------------------
        if not self._fallback_on_low_conf:
            return primary_result

        needs_fallback = self._should_fallback(primary_result)
        if not needs_fallback:
            return primary_result

        # Lazy-load fallback model on first use (runs warmup transcription)
        if not self._fallback_loaded:
            self._load_fallback_model()

        logger.info(
            "Fallback triggered (avg_logprob=%.3f, compression_ratio=%.2f) "
            "-- retrying with %s",
            primary_result.avg_logprob or 0.0,
            primary_result.compression_ratio or 0.0,
            self._fallback_model_id,
        )

        retry_result = self._raw_transcribe(
            audio,
            model_repo=self._fallback_model_id,
            language=language,
            initial_prompt=initial_prompt,
            word_timestamps=word_timestamps,
        )

        # Pick the better result (higher confidence = less negative logprob)
        chosen, chosen_label = self._pick_best(primary_result, retry_result)
        chosen.used_fallback = True

        logger.info(
            "Fallback result: chose %s (primary conf=%.2f avg_lp=%.3f, "
            "retry conf=%.2f avg_lp=%.3f)",
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
        model_repo: str,
        language: str = "en",
        initial_prompt: Optional[str] = None,
        word_timestamps: bool = False,
    ) -> STTResult:
        """Run mlx-whisper transcription against a specific model repo.

        This is the core transcription logic extracted so it can be called
        for both the primary and fallback models.
        """
        t0 = time.perf_counter()
        result = mlx_whisper.transcribe(
            audio,
            path_or_hf_repo=model_repo,
            language=language,
            condition_on_previous_text=False,
            initial_prompt=initial_prompt,
            word_timestamps=word_timestamps,
        )
        latency_ms = (time.perf_counter() - t0) * 1000
        english = result["text"].strip()

        # -- extract segment metadata (mirrors dry_run_ab._run_stt_mlx) ------
        confidence = None
        segment_meta = []
        low_conf_words = []
        overall_avg_logprob = None
        overall_compression_ratio = None
        segments = result.get("segments", [])
        if segments:
            avg_logprobs = []
            compression_ratios = []
            for seg in segments:
                meta = {
                    "avg_logprob": seg.get("avg_logprob"),
                    "no_speech_prob": seg.get("no_speech_prob"),
                    "compression_ratio": seg.get("compression_ratio"),
                }
                segment_meta.append(meta)
                if meta["avg_logprob"] is not None:
                    avg_logprobs.append(meta["avg_logprob"])
                if meta["compression_ratio"] is not None:
                    compression_ratios.append(meta["compression_ratio"])
                # Per-word confidence
                for w in seg.get("words", []):
                    if w.get("probability", 1.0) < 0.5:
                        low_conf_words.append({
                            "word": w.get("word", ""),
                            "probability": round(w["probability"], 3),
                            "start": w.get("start"),
                            "end": w.get("end"),
                        })
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
        MLX Metal is NOT thread-safe -- this runs on the same thread as
        the primary model.
        """
        logger.info("Lazy-loading fallback model %s (MLX)...", self._fallback_model_id)
        t0 = time.time()
        silence = np.zeros(16000, dtype=np.float32)
        mlx_whisper.transcribe(
            silence,
            path_or_hf_repo=self._fallback_model_id,
            condition_on_previous_text=False,
        )
        logger.info(
            "Fallback model ready (%s) (%.1fs)",
            self._fallback_model_id, time.time() - t0,
        )
        self._fallback_loaded = True

    @staticmethod
    def _pick_best(primary: STTResult, retry: STTResult) -> tuple:
        """Return (best_result, label) choosing the higher-confidence result.

        Compares avg_logprob (less negative = better).  Falls back to
        confidence score if avg_logprob is unavailable.

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

        # Quick hash of audio bytes for dedup/linking
        audio_bytes = audio.tobytes()
        audio_hash = hashlib.sha256(audio_bytes).hexdigest()[:16]

        log_fallback_event(
            original=original,
            retry=retry,
            audio_hash=audio_hash,
            chosen=chosen_label,
            primary_model=self._model_id,
            fallback_model=self._fallback_model_id,
        )

    def unload(self) -> None:
        """Release references.  MLX models are managed by the framework cache."""
        self._loaded = False
        self._fallback_loaded = False
        logger.info("MLXWhisperEngine unloaded (%s)", self._model_id)

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def backend(self) -> str:
        return "mlx"


# ---------------------------------------------------------------------------
# MLX TranslateGemma
# ---------------------------------------------------------------------------

class MLXGemmaEngine(TranslationEngine):
    """Translation engine wrapping TranslateGemma via mlx-lm.

    Constructor args:
        model_id:        MLX-community 4-bit model repo
                         (default: mlx-community/translategemma-4b-it-4bit).
        cache_limit_mb:  Metal memory cache limit in megabytes (default: 256).
        use_prompt_cache: Pre-compute KV cache for the fixed chat template
                          prefix, saving ~50-80 ms per call (default: True).
    """

    def __init__(
        self,
        model_id: str = "mlx-community/translategemma-4b-it-4bit",
        cache_limit_mb: int = 256,
        use_prompt_cache: bool = True,
    ):
        if not MLX_AVAILABLE:
            raise RuntimeError(
                "MLX is not available. MLXGemmaEngine requires Apple Silicon "
                "with mlx and mlx-lm installed."
            )
        self._model_id = model_id
        self._cache_limit_mb = cache_limit_mb
        self._use_prompt_cache = use_prompt_cache

        self._model = None
        self._tokenizer = None
        self._prompt_cache_template = None
        self._suffix_tokens = None
        self._loaded = False

    # -- public interface ----------------------------------------------------

    def load(self) -> None:
        """Load the model weights, apply the EOS fix, and build the prompt cache."""
        from mlx_lm import load as mlx_load

        mx.set_cache_limit(self._cache_limit_mb * 1024 * 1024)

        logger.info("Loading %s (MLX 4-bit)...", self._model_id)
        t0 = time.time()
        self._model, self._tokenizer = mlx_load(self._model_id)

        # -- EOS fix (mirrors dry_run_ab.load_mlx_gemma) ----------------------
        # TranslateGemma uses <end_of_turn> (id=106) as its actual EOS, but
        # the tokenizer default is <eos> (id=1) which the model never generates.
        # Without this fix, generation runs to max_tokens (~5 s of pad tokens).
        eot_id = self._tokenizer.convert_tokens_to_ids("<end_of_turn>")
        default_eos = self._tokenizer.eos_token_id
        if not hasattr(self._tokenizer, "_eos_token_ids") or eot_id not in self._tokenizer._eos_token_ids:
            self._tokenizer._eos_token_ids = {default_eos, eot_id}
            logger.info(
                "EOS fix applied: added <end_of_turn> (id=%d) to EOS set "
                "(was only <eos> id=%d)", eot_id, default_eos,
            )
        else:
            logger.info("EOS tokens already correct: %s", self._tokenizer._eos_token_ids)

        elapsed = time.time() - t0
        logger.info("%s loaded (%.1fs)", self._model_id, elapsed)

        # -- prompt cache (mirrors dry_run_ab._build_prompt_cache) ------------
        if self._use_prompt_cache:
            self._prompt_cache_template, self._suffix_tokens = self._build_prompt_cache()

        self._loaded = True

    def translate(
        self,
        text: str,
        *,
        source_lang: str = "en",
        target_lang: str = "es",
    ) -> TranslationResult:
        """Translate *text* from *source_lang* to *target_lang*.

        Uses the pre-computed prompt cache when available.  Falls back to the
        full-prompt path when the cache was not built.
        """
        if not self._loaded:
            raise RuntimeError("Engine not loaded -- call load() first")

        from mlx_lm import generate

        if self._model is None or self._tokenizer is None:
            return TranslationResult(text="(model not loaded)", latency_ms=0.0)

        # Dynamic max-tokens cap: Spanish is ~15-25% longer than English
        input_words = len(text.split())
        max_tok = max(32, int(input_words * 1.8))

        use_cache = (
            self._prompt_cache_template is not None
            and self._suffix_tokens is not None
        )

        if use_cache:
            # Deep-copy the pre-computed KV cache so we don't mutate the template
            cached = copy.deepcopy(self._prompt_cache_template)
            text_tokens = self._tokenizer.encode(text, add_special_tokens=False)
            dynamic_tokens = text_tokens + self._suffix_tokens
            gen_kwargs = dict(
                prompt=dynamic_tokens,
                max_tokens=max_tok,
                verbose=False,
                prompt_cache=cached,
            )
        else:
            messages = [{"role": "user", "content": [
                {"type": "text",
                 "source_lang_code": source_lang,
                 "target_lang_code": target_lang,
                 "text": text}
            ]}]
            prompt = self._tokenizer.apply_chat_template(
                messages, add_generation_prompt=True,
            )
            gen_kwargs = dict(
                prompt=prompt,
                max_tokens=max_tok,
                verbose=False,
            )

        t0 = time.perf_counter()
        result = generate(self._model, self._tokenizer, **gen_kwargs)
        latency_ms = (time.perf_counter() - t0) * 1000

        clean = result.split("<end_of_turn>")[0].strip()
        out_tokens = len(self._tokenizer.encode(clean))
        gen_tps = out_tokens / (latency_ms / 1000) if latency_ms > 0 else 0.0

        return TranslationResult(
            text=clean,
            latency_ms=latency_ms,
            tokens_per_second=gen_tps,
        )

    def unload(self) -> None:
        """Release model and caches from memory."""
        self._model = None
        self._tokenizer = None
        self._prompt_cache_template = None
        self._suffix_tokens = None
        self._loaded = False
        logger.info("MLXGemmaEngine unloaded (%s)", self._model_id)

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def backend(self) -> str:
        return "mlx"

    # -- internals -----------------------------------------------------------

    def _build_prompt_cache(self):
        """Pre-compute KV cache for the fixed TranslateGemma chat template prefix.

        The chat template has a fixed prefix (system tokens + language codes)
        that is identical for every translation.  By pre-filling the KV cache
        once at load time, we skip re-computing those ~30-40 tokens each call
        (saves ~50-80 ms).

        Returns (prompt_cache, suffix_tokens) or (None, None) on failure.
        """
        from mlx_lm.models.cache import make_prompt_cache
        from mlx_lm.generate import generate_step

        marker = "SPLIT_HERE"
        messages = [{"role": "user", "content": [
            {"type": "text",
             "source_lang_code": "en",
             "target_lang_code": "es",
             "text": marker}
        ]}]
        full_prompt = self._tokenizer.apply_chat_template(
            messages, add_generation_prompt=True,
        )
        if isinstance(full_prompt, str):
            full_tokens = self._tokenizer.encode(full_prompt, add_special_tokens=False)
        else:
            full_tokens = list(full_prompt)

        marker_tokens = self._tokenizer.encode(marker, add_special_tokens=False)
        marker_len = len(marker_tokens)

        # Locate the marker in the token sequence
        prefix_end = None
        for i in range(len(full_tokens) - marker_len + 1):
            if full_tokens[i:i + marker_len] == marker_tokens:
                prefix_end = i
                break

        if prefix_end is None:
            logger.warning("Could not locate marker in prompt for %s, skipping cache", self._model_id)
            return None, None

        prefix_tokens = full_tokens[:prefix_end]
        suffix_tokens = full_tokens[prefix_end + marker_len:]

        if len(prefix_tokens) < 3:
            logger.warning("Prefix too short (%d tokens) for %s, skipping cache",
                           len(prefix_tokens), self._model_id)
            return None, suffix_tokens

        # Create and pre-fill the KV cache
        prompt_cache = make_prompt_cache(self._model)
        prompt_array = mx.array(prefix_tokens)

        for _ in generate_step(
            prompt_array, self._model,
            max_tokens=0,
            prompt_cache=prompt_cache,
        ):
            pass  # max_tokens=0 means no tokens generated, just prefill

        mx.eval([c.state for c in prompt_cache])
        logger.info(
            "Prompt cache built: %d prefix tokens cached, %d suffix tokens",
            len(prefix_tokens), len(suffix_tokens),
        )
        return prompt_cache, suffix_tokens


# ---------------------------------------------------------------------------
# MarianMT (PyTorch -- works on any backend)
# ---------------------------------------------------------------------------

# Module-level lock for PyTorch thread safety.  MarianMT inference and
# Silero VAD both use PyTorch; concurrent calls from different threads
# cause heap corruption on macOS.
_pytorch_lock = threading.Lock()


class MarianEngine(TranslationEngine):
    """Fast partial-translation engine wrapping Helsinki-NLP MarianMT.

    Uses PyTorch (CPU or CUDA).  Typical latency: ~80 ms on CPU,
    ~50 ms on CUDA.

    Constructor args:
        model_id:  HuggingFace repo (default: Helsinki-NLP/opus-mt-en-es).
        device:    "cpu", "cuda", or "auto" (auto-detect).  Default: "auto".
    """

    def __init__(
        self,
        model_id: str = "Helsinki-NLP/opus-mt-en-es",
        device: str = "auto",
    ):
        self._model_id_str = model_id
        self._requested_device = device
        self._device = None
        self._model = None
        self._tokenizer = None
        self._loaded = False

    # -- public interface ----------------------------------------------------

    def load(self) -> None:
        """Download (if needed) and warm up MarianMT."""
        from transformers import MarianMTModel, MarianTokenizer

        # Resolve device
        if self._requested_device == "auto":
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._device = self._requested_device

        logger.info("Loading %s (MarianMT, device=%s)...", self._model_id_str, self._device)
        t0 = time.time()
        self._tokenizer = MarianTokenizer.from_pretrained(self._model_id_str)
        self._model = MarianMTModel.from_pretrained(self._model_id_str)
        if self._device == "cuda":
            self._model = self._model.to("cuda")
        self._model.eval()

        # Warm up
        inputs = self._tokenizer("Hello", return_tensors="pt", padding=True)
        if self._device == "cuda":
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        with torch.no_grad():
            self._model.generate(**inputs, max_new_tokens=16)

        logger.info("MarianMT ready (%.1fs)", time.time() - t0)
        self._loaded = True

    def translate(
        self,
        text: str,
        *,
        source_lang: str = "en",
        target_lang: str = "es",
    ) -> TranslationResult:
        """Translate *text* via MarianMT.

        Thread-safe: uses ``_pytorch_lock`` to prevent concurrent PyTorch
        calls (which cause heap corruption on macOS).
        """
        if not self._loaded:
            raise RuntimeError("Engine not loaded -- call load() first")

        if self._model is None or self._tokenizer is None:
            return TranslationResult(text="(MarianMT not loaded)", latency_ms=0.0)

        t0 = time.perf_counter()
        inputs = self._tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        if self._device == "cuda":
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        with _pytorch_lock:
            with torch.no_grad():
                translated = self._model.generate(**inputs, max_new_tokens=128)
        result = self._tokenizer.decode(translated[0], skip_special_tokens=True)
        latency_ms = (time.perf_counter() - t0) * 1000

        return TranslationResult(
            text=result,
            latency_ms=latency_ms,
        )

    def unload(self) -> None:
        """Release model from memory."""
        self._model = None
        self._tokenizer = None
        self._loaded = False
        logger.info("MarianEngine unloaded (%s)", self._model_id_str)

    @property
    def model_id(self) -> str:
        return self._model_id_str

    @property
    def backend(self) -> str:
        if self._device is not None:
            return self._device
        return "cpu"
