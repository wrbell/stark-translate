"""NVIDIA Parakeet TDT STT engine (EN-only finals accelerator).

Optional backend for ``--lang en`` / ``STARK_STT__BACKEND=parakeet``. Requires
``nemo_toolkit[asr]`` (or ``nvidia-nemo``) at runtime. Not a bilingual default —
Spanish coverage stays on Whisper + W16.

Usage via factory::

    create_stt_engine(stt_backend="parakeet", model_id="nvidia/parakeet-tdt-0.6b-v3")
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

from engines.base import STTEngine, STTResult

logger = logging.getLogger(__name__)

DEFAULT_PARAKEET_MODEL = "nvidia/parakeet-tdt-0.6b-v3"


class ParakeetEngine(STTEngine):
    """NeMo Parakeet TDT wrapper implementing ``STTEngine``."""

    def __init__(
        self,
        model_id: str | None = None,
        device: str = "cuda",
        fallback_threshold: float = -1.2,
        hallucination_threshold: float = 2.4,
        fallback_on_low_conf: bool = False,
        **kwargs: Any,
    ) -> None:
        self._model_id = model_id or DEFAULT_PARAKEET_MODEL
        self._device = device
        self._fallback_threshold = fallback_threshold
        self._hallucination_threshold = hallucination_threshold
        self._fallback_on_low_conf = fallback_on_low_conf
        self._model = None
        self._loaded = False
        # Accept unused kwargs for factory parity
        _ = kwargs

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def backend(self) -> str:
        return "parakeet"

    def load(self) -> None:
        if self._loaded:
            return
        try:
            import nemo.collections.asr as nemo_asr  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "ParakeetEngine requires nemo_toolkit[asr]. "
                "Install on WSL CUDA: pip install 'nemo_toolkit[asr]'. "
                "Keep Whisper as the bilingual default."
            ) from exc

        logger.info("Loading Parakeet model %s on %s", self._model_id, self._device)
        self._model = nemo_asr.models.ASRModel.from_pretrained(model_name=self._model_id)
        if self._device == "cuda":
            self._model = self._model.cuda()
        self._model.eval()
        self._loaded = True

    def unload(self) -> None:
        self._model = None
        self._loaded = False

    def transcribe(
        self,
        audio: np.ndarray,
        *,
        language: str = "en",
        initial_prompt: str | None = None,
        word_timestamps: bool = False,
        beam_size: int | None = None,
    ) -> STTResult:
        if not self._loaded or self._model is None:
            raise RuntimeError("ParakeetEngine.load() must be called before transcribe()")
        if language not in ("en", "english"):
            logger.warning(
                "Parakeet is EN-focused; language=%r may degrade. Prefer Whisper for ES.",
                language,
            )
        _ = initial_prompt, word_timestamps, beam_size  # API parity; not all forwarded

        # NeMo expects float32 mono 16 kHz
        wav = np.asarray(audio, dtype=np.float32).reshape(-1)
        t0 = time.perf_counter()
        # transcribe() accepts a list of numpy arrays in recent NeMo builds
        try:
            hypotheses = self._model.transcribe([wav], batch_size=1)
        except TypeError:
            # Older API: write temp wav path — avoid when possible
            import tempfile

            import soundfile as sf

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
                sf.write(tmp.name, wav, 16000)
                hypotheses = self._model.transcribe([tmp.name])

        latency_ms = (time.perf_counter() - t0) * 1000
        text = ""
        if hypotheses:
            hyp0 = hypotheses[0]
            text = hyp0 if isinstance(hyp0, str) else getattr(hyp0, "text", str(hyp0))
        text = (text or "").strip()

        return STTResult(
            text=text,
            latency_ms=latency_ms,
            confidence=None,
            avg_logprob=None,
            compression_ratio=None,
            segments=[],
            low_confidence_words=[],
            used_fallback=False,
        )
