"""HuggingFace Transformers Whisper engine with speculative decoding support.

Uses ``WhisperForConditionalGeneration`` from transformers, enabling the
``assistant_model`` parameter for speculative decoding with a distilled
draft model (e.g., distil-whisper/distil-large-v3.5 drafting for
openai/whisper-large-v3-turbo).

This engine trades the speed optimizations of faster-whisper (CTranslate2)
for access to HF's speculative decoding, which provides 1.5-2x speedup
with mathematically identical output.

Usage:
    engine = HFWhisperEngine(
        model_id="openai/whisper-large-v3-turbo",
        draft_model_id="distil-whisper/distil-large-v3.5",
    )
    engine.load()
    result = engine.transcribe(audio_array)
"""

import logging
import time

import numpy as np

from engines.base import STTEngine, STTResult

logger = logging.getLogger(__name__)

try:
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    HF_WHISPER_AVAILABLE = True
except ImportError:
    HF_WHISPER_AVAILABLE = False


class HFWhisperEngine(STTEngine):
    """HuggingFace Whisper STT engine with speculative decoding.

    Constructor args:
        model_id:        Target Whisper model (default: whisper-large-v3-turbo).
        draft_model_id:  Draft model for speculative decoding (default: distil-large-v3.5).
                         Set to None to disable spec decode.
        device:          'cuda' or 'cpu' (default: auto-detect).
        torch_dtype:     Model dtype (default: float16 for CUDA, float32 for CPU).
    """

    def __init__(
        self,
        model_id: str = "openai/whisper-large-v3-turbo",
        draft_model_id: str | None = "distil-whisper/distil-large-v3.5",
        device: str | None = None,
        torch_dtype: str | None = None,
    ):
        if not HF_WHISPER_AVAILABLE:
            raise RuntimeError("transformers and torch are required for HFWhisperEngine.")
        self._model_id = model_id
        self._draft_model_id = draft_model_id
        if device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._device = device
        if torch_dtype is None:
            self._torch_dtype = torch.float16 if self._device == "cuda" else torch.float32
        else:
            self._torch_dtype = getattr(torch, torch_dtype)

        self._model = None
        self._draft_model = None
        self._processor = None
        self._loaded = False

    def load(self) -> None:
        """Load target and draft models. Encoder weights are shared."""
        t0 = time.time()
        logger.info("Loading %s (HF Whisper, %s)...", self._model_id, self._device)

        self._processor = AutoProcessor.from_pretrained(self._model_id)
        self._model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self._model_id,
            torch_dtype=self._torch_dtype,
            device_map=self._device,
        )
        self._model.eval()
        logger.info("Target model loaded (%.1fs)", time.time() - t0)

        if self._draft_model_id:
            t1 = time.time()
            logger.info("Loading draft model %s...", self._draft_model_id)
            self._draft_model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self._draft_model_id,
                torch_dtype=self._torch_dtype,
                device_map=self._device,
            )
            self._draft_model.eval()
            logger.info("Draft model loaded (%.1fs)", time.time() - t1)

        elapsed = time.time() - t0
        logger.info(
            "HFWhisperEngine ready (%.1fs, spec_decode=%s)",
            elapsed,
            self._draft_model_id or "disabled",
        )
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
        """Transcribe audio using HF generate() with optional speculative decoding."""
        if not self._loaded:
            raise RuntimeError("Engine not loaded -- call load() first")

        t0 = time.perf_counter()

        # Process audio to input features
        inputs = self._processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",
        )
        input_features = inputs.input_features.to(self._device, dtype=self._torch_dtype)

        # Build generate kwargs
        gen_kwargs = {
            "input_features": input_features,
            "max_new_tokens": 128,
            "language": language,
            "return_timestamps": word_timestamps,
        }

        if initial_prompt:
            prompt_ids = self._processor.get_prompt_ids(initial_prompt, return_tensors="pt")
            gen_kwargs["prompt_ids"] = prompt_ids.to(self._device)

        if beam_size is not None:
            gen_kwargs["num_beams"] = beam_size

        # Speculative decoding: pass draft model as assistant
        if self._draft_model is not None:
            gen_kwargs["assistant_model"] = self._draft_model

        with torch.no_grad():
            output = self._model.generate(**gen_kwargs)

        text = self._processor.batch_decode(output, skip_special_tokens=True)[0].strip()
        latency_ms = (time.perf_counter() - t0) * 1000

        return STTResult(
            text=text,
            latency_ms=latency_ms,
        )

    def unload(self) -> None:
        """Release models from memory."""
        self._model = None
        self._draft_model = None
        self._processor = None
        self._loaded = False
        if HF_WHISPER_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("HFWhisperEngine unloaded")

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def backend(self) -> str:
        return "hf"
