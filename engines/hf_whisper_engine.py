"""HuggingFace Transformers Whisper engine.

Uses ``WhisperForConditionalGeneration`` from transformers. Optionally
supports HF's `assistant_model` speculative decoding when caller supplies a
verified-compatible draft (note: distil-large-v3.5 + whisper-large-v3-turbo
is broken — see docs/archive/v2026.5/spec_decode_research.md). Trades the
speed of faster-whisper (CTranslate2) for torch.compile + spec-decode
flexibility.

Usage:
    # Vanilla (default — no spec decode, no draft)
    engine = HFWhisperEngine(model_id="openai/whisper-large-v3-turbo")

    # Spec decode against a verified-compatible draft (turbo drafts for v3)
    engine = HFWhisperEngine(
        model_id="openai/whisper-large-v3",
        draft_model_id="openai/whisper-large-v3-turbo",
    )
    engine.load()
    result = engine.transcribe(audio_array)
"""

import logging
import time
from typing import Any

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
    """HuggingFace Whisper STT engine.

    Constructor args:
        model_id:         Target Whisper model (default: whisper-large-v3-turbo).
        draft_model_id:   Optional draft model for HF speculative decoding.
                          Default: None (no spec decode). Caller must supply a
                          verified-compatible draft — distil-large-v3.5 + turbo
                          is broken (see docs/archive/v2026.5/spec_decode_research.md).
        device:           'cuda' or 'cpu' (default: auto-detect).
        torch_dtype:      Model dtype (default: float16 for CUDA, float32 for CPU).
        compile_mode:     If set (e.g. 'reduce-overhead', 'default'), wraps the
                          target + draft models in torch.compile after load. Note:
                          'reduce-overhead' enables CUDA graphs which require
                          fixed input shapes — Whisper's variable-length log-Mel
                          recompiles on each new shape (~5–10 s first call).
        warmup_seconds:   Seconds of silence to push through the model after load
                          to drive any JIT/CUDA-graph capture before the first
                          real call. 0 disables.
    """

    def __init__(
        self,
        model_id: str = "openai/whisper-large-v3-turbo",
        draft_model_id: str | None = None,
        device: str | None = None,
        torch_dtype: str | None = None,
        compile_mode: str | None = None,
        warmup_seconds: int = 1,
    ):
        # NOTE: draft_model_id default is None, NOT distil-large-v3.5. Distil-v3.5
        # was distilled from whisper-large-v3 (32 decoder layers); whisper-large-v3-turbo
        # has 4. Pairing them produces 10x slower inference with hallucinated
        # repetitions (tested 2026-04-13, see docs/archive/v2026.5/spec_decode_research.md).
        # Verified-compatible pairing: turbo (4 layers) drafts for large-v3 (32 layers).
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
        self._compile_mode = compile_mode
        self._warmup_seconds = max(0, int(warmup_seconds))

        self._model: Any = None
        self._draft_model: Any = None
        self._processor: Any = None
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

        if self._compile_mode:
            logger.info(
                "torch.compile(mode=%r) on target%s...", self._compile_mode, " + draft" if self._draft_model else ""
            )
            t_c = time.time()
            self._model = torch.compile(self._model, mode=self._compile_mode)
            if self._draft_model is not None:
                self._draft_model = torch.compile(self._draft_model, mode=self._compile_mode)
            logger.info("compile graphs registered (%.2fs); first call still pays JIT cost", time.time() - t_c)

        self._loaded = True

        if self._warmup_seconds > 0:
            logger.info("warmup pass on %ds of silence...", self._warmup_seconds)
            t_w = time.time()
            silence = np.zeros(16000 * self._warmup_seconds, dtype=np.float32)
            self.transcribe(silence, language="en")
            logger.info("warmup done (%.2fs)", time.time() - t_w)

        elapsed = time.time() - t0
        logger.info(
            "HFWhisperEngine ready (%.1fs, spec_decode=%s, compile=%s, warmup=%ds)",
            elapsed,
            self._draft_model_id or "disabled",
            self._compile_mode or "off",
            self._warmup_seconds,
        )

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
