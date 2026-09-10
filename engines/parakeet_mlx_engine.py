"""Parakeet TDT v3 STT via the optional parakeet-mlx backend."""

from __future__ import annotations

import logging
import math
import re
import time

import numpy as np

from engines.base import STTEngine, STTResult, text_compression_ratio

logger = logging.getLogger(__name__)
DEFAULT_PARAKEET_MLX_MODEL = "mlx-community/parakeet-tdt-0.6b-v3"


def _audio_float32(audio: np.ndarray) -> np.ndarray:
    """Require normalized mono audio; sample-rate/PCM conversion is caller-owned."""
    audio = np.asarray(audio)
    if audio.ndim != 1:
        raise ValueError("Parakeet requires 16 kHz mono audio (one-dimensional)")
    if not np.issubdtype(audio.dtype, np.floating):
        raise ValueError("Parakeet requires normalized floating-point audio, not integer PCM")
    return audio.astype(np.float32, copy=False)


def _avg_logprob(tokens: list) -> float | None:
    if not tokens:
        return None
    return sum(math.log(max(float(token.confidence), 1e-6)) for token in tokens) / len(tokens)


def _words(tokens: list) -> list[dict]:
    """Join whitespace-delimited subwords, using geometric-mean confidence."""
    words: list[dict] = []
    pieces: list[str] = []
    word_tokens: list = []

    def flush():
        if pieces:
            words.append(
                {
                    "word": "".join(pieces),
                    "probability": math.exp(_avg_logprob(word_tokens) or 0.0),
                    "start": word_tokens[0].start,
                    "end": word_tokens[-1].end,
                }
            )
            pieces.clear()
            word_tokens.clear()

    for token in tokens:
        for piece in re.findall(r"\s+|\S+", token.text):
            if piece.isspace():
                flush()
            else:
                pieces.append(piece)
                word_tokens.append(token)
    flush()
    return words


class ParakeetStream:
    """An opened streaming context. Feed only new 16 kHz samples on its worker.

    Use ``with engine.open_stream() as stream`` or explicitly call ``close()``.
    Closing restores the model's attention mode; text/tokens remain readable.
    """

    def __init__(self, context):
        self._context = context
        self._stream = context.__enter__()
        self._closed = False

    def __enter__(self) -> ParakeetStream:
        if self._closed:
            raise RuntimeError("Parakeet stream is closed")
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._close(exc_type, exc_value, traceback)

    def add_audio(self, audio: np.ndarray) -> None:
        if self._closed:
            raise RuntimeError("Parakeet stream is closed")
        import mlx.core as mx

        self._stream.add_audio(mx.array(_audio_float32(audio)))

    @property
    def text(self) -> str:
        return self._stream.result.text

    @property
    def finalized_text(self) -> str:
        return "".join(token.text for token in self._stream.finalized_tokens).strip()

    @property
    def tokens(self) -> list:
        """Current finalized and draft AlignedTokens, in decoding order."""
        return list(self._stream.finalized_tokens) + list(self._stream.draft_tokens)

    def _close(self, exc_type, exc_value, traceback) -> None:
        if not self._closed:
            self._closed = True
            self._context.__exit__(exc_type, exc_value, traceback)

    def close(self) -> None:
        self._close(None, None, None)


class ParakeetMLXEngine(STTEngine):
    """Multilingual Parakeet TDT v3 with Whisper-compatible quality proxies.

    Run load, transcription, all stream operations, and unload on one dedicated
    worker thread: MLX lazy arrays bind to their creating thread's stream. Load
    materializes weights on that thread. No internal locks are needed. Close a
    stream before batch transcription, another stream, or unload: streaming
    temporarily changes the model's attention mode.

    English and Spanish are supported by v3; ``language`` is accepted but cannot
    force a language through this API. ``beam_size`` is ignored (greedy decode).
    Token confidence is a TDT proxy, not a calibrated Whisper probability.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_PARAKEET_MLX_MODEL,
        dtype: str = "bfloat16",
        cache_limit_mb: int = 256,
        warmup_seconds: float = 1.0,
        stream_context: tuple[int, int] = (256, 256),
        stream_depth: int = 1,
    ):
        if dtype not in ("bfloat16", "float16", "float32"):
            raise ValueError(f"Unsupported Parakeet MLX dtype: {dtype!r}")
        self._model_id = model_id
        self._dtype = dtype
        self._cache_limit_mb = cache_limit_mb
        self._warmup_seconds = warmup_seconds
        self._stream_context = stream_context
        self._stream_depth = stream_depth
        self._model = None
        self._prompt_warned = False

    def load(self) -> None:
        if self._model is not None:
            return
        import mlx.core as mx
        from parakeet_mlx import from_pretrained

        from engines.mlx_engine import materialize_mlx_model

        mx.set_cache_limit(self._cache_limit_mb * 1024 * 1024)
        from engines.model_paths import resolve_model_path

        logger.info("Loading %s (Parakeet MLX)", self._model_id)
        self._model = from_pretrained(resolve_model_path(self._model_id), dtype=getattr(mx, self._dtype))
        try:
            if self._warmup_seconds > 0:
                self.transcribe(np.zeros(int(self._warmup_seconds * 16000), dtype=np.float32))
            materialize_mlx_model(self._model)
            mx.synchronize()
        except Exception:
            self._model = None
            raise

    def transcribe(
        self,
        audio: np.ndarray,
        *,
        language: str = "en",
        initial_prompt: str | None = None,
        word_timestamps: bool = False,
        beam_size: int | None = None,
    ) -> STTResult:
        """Decode normalized float audio, with no resampling or language forcing.

        ``no_speech_prob`` is a conservative rejection proxy: 1 for no tokens
        despite RMS > 0.008, 0 when tokens exist, otherwise unknown (None).
        Parakeet supplies no silence posterior. Prompts cannot bias decoding.
        """
        if self._model is None:
            raise RuntimeError("Engine not loaded -- call load() first")
        import mlx.core as mx
        from parakeet_mlx.audio import get_logmel

        audio = _audio_float32(audio)
        if initial_prompt and not self._prompt_warned:
            logger.warning("Parakeet has no prompt biasing; theological terms rely on correct_stt_output")
            self._prompt_warned = True
        t0 = time.perf_counter()
        mel = get_logmel(mx.array(audio), self._model.preprocessor_config)
        res = self._model.generate(mel)[0]
        mx.synchronize()
        text = res.text.strip()
        tokens = [token for sentence in res.sentences for token in sentence.tokens]
        avg_logprob = _avg_logprob(tokens)
        rms = float(np.sqrt(np.mean(np.square(audio), dtype=np.float64))) if audio.size else 0.0
        no_speech_prob = 0.0 if tokens else (1.0 if rms > 0.008 else None)
        segments = []
        low_confidence_words = []
        for sentence in res.sentences:
            words = _words(sentence.tokens)
            segment = {
                "avg_logprob": _avg_logprob(sentence.tokens),
                "no_speech_prob": 0.0 if sentence.tokens else no_speech_prob,
                "compression_ratio": text_compression_ratio(sentence.text.strip()),
                "start": sentence.start,
                "end": sentence.end,
                "text": sentence.text.strip(),
            }
            if word_timestamps:
                segment["words"] = words
            segments.append(segment)
            low_confidence_words.extend(word for word in words if word["probability"] < 0.5)
        return STTResult(
            text=text,
            latency_ms=(time.perf_counter() - t0) * 1000,
            avg_logprob=avg_logprob,
            confidence=min(1.0, max(0.0, 1.0 + avg_logprob)) if avg_logprob is not None else None,
            compression_ratio=text_compression_ratio(text),
            no_speech_prob=no_speech_prob,
            segments=segments,
            low_confidence_words=low_confidence_words,
        )

    def open_stream(self, context_size: tuple[int, int] | None = None, depth: int | None = None) -> ParakeetStream:
        if self._model is None:
            raise RuntimeError("Engine not loaded -- call load() first")
        return ParakeetStream(
            self._model.transcribe_stream(
                context_size=self._stream_context if context_size is None else context_size,
                depth=self._stream_depth if depth is None else depth,
            )
        )

    def unload(self) -> None:
        if self._model is not None:
            import mlx.core as mx

            mx.synchronize()
            self._model = None
            mx.clear_cache()

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def backend(self) -> str:
        return "parakeet-mlx"
