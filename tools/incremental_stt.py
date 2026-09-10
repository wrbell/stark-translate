"""Preview-only incremental STT adapter; final batch recognition stays separate."""

from __future__ import annotations

import math
import time


class StreamingPreview:
    """All calls, including close, must run on the STT owning worker.

    Streams consume only new audio. Closing before batch recognition restores
    Parakeet's original attention mode; no streaming text is a final result.
    """

    def __init__(self, engine):
        if not callable(getattr(engine, "open_stream", None)):
            raise ValueError("Streaming previews require an STT engine with open_stream (Parakeet MLX)")
        self.engine = engine
        self.stream = None
        self.utterance_id = None
        self.consumed_samples = 0

    def transcribe(self, audio, utterance_id):
        from engines.base import text_compression_ratio

        started = time.perf_counter()
        if self.utterance_id != utterance_id or len(audio) < self.consumed_samples:
            self.close()
        if self.stream is None:
            self.stream = self.engine.open_stream()
            self.utterance_id = utterance_id
        if len(audio) > self.consumed_samples:
            self.stream.add_audio(audio[self.consumed_samples :])
        processed = len(audio) - self.consumed_samples
        self.consumed_samples = len(audio)
        text = self.stream.text.strip()
        scores = [math.log(max(float(t.confidence), 1e-6)) for t in self.stream.tokens]
        confidence = max(0, min(1, 1 + sum(scores) / len(scores))) if scores else None
        return (
            text,
            (time.perf_counter() - started) * 1000,
            confidence,
            None,
            text_compression_ratio(text),
        ), processed

    def close(self):
        if self.stream is not None:
            self.stream.close()
        self.stream = None
        self.utterance_id = None
        self.consumed_samples = 0
