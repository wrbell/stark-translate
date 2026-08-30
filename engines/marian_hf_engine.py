"""HF transformers MarianMT translation engine.

Extracted from ``engines.mlx_engine.MarianEngine`` (v2026.7) and renamed to
``MarianHFEngine`` to disambiguate from the new CTranslate2 path
(``engines.cuda_engine.MarianCT2Engine``). Behaviour is preserved bit-for-bit;
only the lock import changed (now from ``engines._locks`` so Silero VAD and
the HF partial translator actually share serialization — see _locks.py).

Path selection:
  - On CUDA, prefer ``MarianCT2Engine`` (3–5× faster on short utterances).
  - On Mac (MLX backend), this engine is the production path: PyTorch CPU
    runs alongside Metal, no contention.
  - As a fallback when the CT2 conversion artifact is missing.
"""

from __future__ import annotations

import logging
import time

import torch

from engines._locks import _pytorch_lock
from engines.base import TranslationEngine, TranslationResult

logger = logging.getLogger(__name__)


class MarianHFEngine(TranslationEngine):
    """Fast partial-translation engine wrapping Helsinki-NLP MarianMT (HF transformers).

    Uses PyTorch (CPU or CUDA). Typical latency: ~80 ms on CPU, ~50 ms on CUDA.

    Constructor args:
        model_id:  HuggingFace repo (default: Helsinki-NLP/opus-mt-en-es).
        device:    "cpu", "cuda", or "auto" (auto-detect).  Default: "auto".
        max_new_tokens: Max decoding length (default: 128).
        warmup_passes: Warmup translations performed at load() (default: 2).
    """

    def __init__(
        self,
        model_id: str = "Helsinki-NLP/opus-mt-en-es",
        device: str = "auto",
        max_new_tokens: int = 128,
        warmup_passes: int = 2,
    ):
        self._model_id_str = model_id
        self._requested_device = device
        self._device: str | None = None
        self._model = None
        self._tokenizer = None
        self._max_new_tokens = max_new_tokens
        self._warmup_passes = max(0, int(warmup_passes))
        self._loaded = False

    # -- public interface ----------------------------------------------------

    def load(self) -> None:
        """Download (if needed) and warm up MarianMT."""
        from transformers import MarianMTModel, MarianTokenizer

        if self._requested_device == "auto":
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._device = self._requested_device

        logger.info("Loading %s (MarianMT-HF, device=%s)...", self._model_id_str, self._device)
        t0 = time.time()
        self._tokenizer = MarianTokenizer.from_pretrained(self._model_id_str)
        self._model = MarianMTModel.from_pretrained(self._model_id_str)
        if self._device == "cuda":
            self._model = self._model.to("cuda")
        self._model.eval()

        # Warmup: short greeting + theological-term sentence covers both the
        # short-input codepath and SentencePiece subword splits on rarer tokens.
        warmup_texts = ["Hello", "Lord, have mercy on us."]
        for text in warmup_texts[: self._warmup_passes]:
            inputs = self._tokenizer(text, return_tensors="pt", padding=True)
            if self._device == "cuda":
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            with torch.no_grad():
                self._model.generate(**inputs, max_new_tokens=16)

        logger.info("MarianMT-HF ready (%.1fs)", time.time() - t0)
        self._loaded = True

    def translate(
        self,
        text: str,
        *,
        source_lang: str = "en",
        target_lang: str = "es",
    ) -> TranslationResult:
        """Translate *text* via MarianMT.

        Thread-safe via the shared ``_pytorch_lock`` from ``engines._locks``,
        which also serializes against Silero VAD calls in the live pipeline.
        """
        if not self._loaded:
            raise RuntimeError("Engine not loaded -- call load() first")

        if self._model is None or self._tokenizer is None:
            return TranslationResult(text="(MarianMT not loaded)", latency_ms=0.0)

        t0 = time.perf_counter()
        inputs = self._tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        if self._device == "cuda":
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        with _pytorch_lock, torch.no_grad():
            translated = self._model.generate(**inputs, max_new_tokens=self._max_new_tokens)
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
        logger.info("MarianHFEngine unloaded (%s)", self._model_id_str)

    @property
    def model_id(self) -> str:
        return self._model_id_str

    @property
    def backend(self) -> str:
        # Format: "hf-{device}" — disambiguates from MarianCT2Engine's
        # "ct2-{compute_type}" so operators can see which path is hot.
        return f"hf-{self._device or 'cpu'}"
