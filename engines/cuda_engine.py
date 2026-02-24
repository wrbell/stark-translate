"""CUDA backend engines for NVIDIA GPU inference.

Provides:
  - FasterWhisperEngine  -- STT via faster-whisper (CTranslate2 backend)
  - CUDAGemmaEngine      -- Translation via transformers + bitsandbytes 4-bit
"""

import logging
import time

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
    ) -> STTResult:
        """Run faster-whisper transcription against a specific model instance.

        This is the core transcription logic extracted so it can be called
        for both the primary and fallback models.
        """
        t0 = time.perf_counter()
        segments_gen, _info = model.transcribe(
            audio,
            language=language,
            condition_on_previous_text=False,
            initial_prompt=initial_prompt,
            word_timestamps=word_timestamps,
        )

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
