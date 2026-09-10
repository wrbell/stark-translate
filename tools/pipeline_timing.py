"""Clock-local pipeline timing, including frame-preserving utterance cuts.

Speech end means the end of the last VAD-positive frame, not a forced claim
about acoustic ground truth. No timestamp here is comparable across hosts.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, replace
from typing import Any

TIMING_SCHEMA_VERSION = 2
SAMPLE_COLUMNS = ("sample_start", "sample_end", "sample_rate", "speech_end_sample", "padding_samples")
TIMING_COLUMNS = (
    "timing_schema_version",
    "source_lang",
    "target_lang",
    "session_kind",
    "utterance_id",
    "endpoint_reason",
    "timing_source",
    "speech_end_to_final_ms",
    "vad_wait_ms",
    "stt_queue_wait_ms",
    "translation_queue_wait_ms",
    "finalization_overhead_ms",
    "broadcast_ms",
    "input_audio_path",
    *SAMPLE_COLUMNS,
)


def milliseconds(end: float | None, start: float | None) -> float | None:
    if end is None or start is None:
        return None
    return round(max(0.0, end - start) * 1000, 1)


@dataclass(frozen=True)
class CaptureStamp:
    start: float
    end: float
    source: str
    # Half-open original stream coordinates, before per-frame STT resampling.
    # Virtual replay tail/block padding has no source samples of its own.
    sample_start: int | None = None
    sample_end: int | None = None
    sample_rate: int | None = None
    padding_samples: int = 0

    def slice(self, start: int, end: int, size: int) -> CaptureStamp:
        """Slice using buffered PCM indices while retaining source coordinates."""
        duration = self.end - self.start
        clock_start = self.start + duration * start / size
        clock_end = self.start + duration * end / size
        if self.sample_start is None or self.sample_end is None:
            return replace(self, start=clock_start, end=clock_end)
        actual = self.sample_end - self.sample_start
        total = actual + self.padding_samples
        first, last = round(total * start / size), round(total * end / size)
        return replace(
            self,
            start=clock_start,
            end=clock_end,
            sample_start=self.sample_start + min(first, actual),
            sample_end=self.sample_start + min(last, actual),
            padding_samples=max(0, last - actual) - max(0, first - actual),
        )

    @property
    def audio_end(self) -> float:
        if self.sample_start is None or self.sample_end is None or not self.padding_samples:
            return self.end
        actual = self.sample_end - self.sample_start
        return self.start + (self.end - self.start) * actual / (actual + self.padding_samples)

    @property
    def has_audio(self) -> bool:
        return self.sample_start is None or self.sample_end is None or self.sample_end > self.sample_start


def capture_stamp(frames: int, rate: int, time_info: Any, received: float | None = None) -> CaptureStamp:
    """Map PortAudio's ADC clock using its callback-current clock offset.

    Replay already supplies pipeline-clock sample endpoints. A bridge with no
    source clock gets a receipt estimate that explicitly excludes upstream lag.
    """
    received = time.perf_counter() if received is None else received
    if isinstance(time_info, CaptureStamp):
        return time_info
    duration = frames / rate
    try:
        adc = float(time_info.inputBufferAdcTime)
        current = float(time_info.currentTime)
        start = received + adc - current
        if math.isfinite(start) and adc > 0 and current > 0 and start + duration <= received + duration:
            return CaptureStamp(start, start + duration, "portaudio_adc")
    except (AttributeError, TypeError, ValueError):
        pass
    return CaptureStamp(received - duration, received, "callback_receipt_estimate")


@dataclass
class CaptureSampleClock:
    """Assign stream positions in the producer callback, including discarded frames."""

    next_sample: int = 0

    def capture(self, frames: int, rate: int, time_info: Any, received: float | None = None) -> CaptureStamp:
        stamp = capture_stamp(frames, rate, time_info, received)
        if stamp.sample_start is None or stamp.sample_end is None:
            stamp = replace(
                stamp, sample_start=self.next_sample, sample_end=self.next_sample + frames, sample_rate=rate
            )
        assert stamp.sample_end is not None
        self.next_sample = max(self.next_sample, stamp.sample_end)
        return stamp


@dataclass(frozen=True)
class AudioFrame:
    samples: Any
    stamp: CaptureStamp

    @property
    def sample_start(self) -> int | None:
        return self.stamp.sample_start

    @property
    def sample_end(self) -> int | None:
        return self.stamp.sample_end

    @property
    def sample_rate(self) -> int | None:
        return self.stamp.sample_rate


@dataclass
class AudioTimeline:
    """One span per buffered VAD frame; splits retain original capture times."""

    spans: list[tuple[int, CaptureStamp, bool]] = field(default_factory=list)

    def append(self, samples: int, stamp: CaptureStamp, speech: bool) -> None:
        self.spans.append((samples, stamp, speech))

    def split(self, count: int) -> AudioTimeline:
        taken = []
        rest = []
        for size, stamp, speech in self.spans:
            n = min(max(count, 0), size)
            if n:
                taken.append((n, stamp.slice(0, n, size), speech))
            if n < size:
                rest.append((size - n, stamp.slice(n, size, size), speech))
            count -= n
        self.spans = rest
        return AudioTimeline(taken)

    @property
    def first(self) -> float | None:
        return self.spans[0][1].start if self.spans else None

    @property
    def last(self) -> float | None:
        return self.spans[-1][1].end if self.spans else None

    @property
    def speech_end(self) -> float | None:
        # Retain schema-2 VAD frame-end timing semantics. Sample metadata below
        # separately identifies real source audio versus virtual replay padding.
        return next((stamp.end for _, stamp, speech in reversed(self.spans) if speech), None)

    def sample_metadata(self) -> dict:
        rates = {stamp.sample_rate for _, stamp, _ in self.spans}
        return {
            "sample_start": self.spans[0][1].sample_start if self.spans else None,
            "sample_end": self.spans[-1][1].sample_end if self.spans else None,
            "sample_rate": next(iter(rates)) if len(rates) == 1 else None,
            "speech_end_sample": next(
                (stamp.sample_end for _, stamp, speech in reversed(self.spans) if speech and stamp.has_audio), None
            ),
            "padding_samples": sum(stamp.padding_samples for _, stamp, _ in self.spans),
        }

    @property
    def source(self) -> str:
        sources = {stamp.source for _, stamp, _ in self.spans}
        return next(iter(sources)) if len(sources) == 1 else "mixed_or_unknown"


@dataclass
class ChunkTiming:
    utterance_id: int | None = None
    endpoint_reason: str = "unknown"
    timing_source: str = "unknown"
    captured_start: float | None = None
    speech_end: float | None = None
    vad_finalized: float | None = None
    submitted: float | None = None
    dequeued: float | None = None
    stt_requested: float | None = None
    stt_started: float | None = None
    stt_finished: float | None = None
    translation_requested: float | None = None
    translation_lock_acquired: float | None = None
    translation_started: float | None = None
    translation_finished: float | None = None
    final_ready: float | None = None
    broadcast_finished: float | None = None
    sample_start: int | None = None
    sample_end: int | None = None
    sample_rate: int | None = None
    speech_end_sample: int | None = None
    padding_samples: int = 0

    @classmethod
    def from_timeline(cls, timeline: AudioTimeline, utterance_id: int, reason: str) -> ChunkTiming:
        return cls(
            utterance_id,
            reason,
            timeline.source,
            timeline.first,
            timeline.speech_end,
            time.perf_counter(),
            **timeline.sample_metadata(),
        )

    def metrics(self) -> dict:
        return {
            "timing_schema_version": TIMING_SCHEMA_VERSION,
            "utterance_id": self.utterance_id,
            "endpoint_reason": self.endpoint_reason,
            "timing_source": self.timing_source,
            **self.sample_metadata(),
            "speech_end_to_final_ms": (
                milliseconds(self.final_ready, self.speech_end) if self.timing_source != "replay_nonrealtime" else None
            ),
            "vad_wait_ms": milliseconds(self.vad_finalized, self.speech_end),
            "stt_queue_wait_ms": milliseconds(self.stt_started, self.submitted),
            "translation_queue_wait_ms": milliseconds(self.translation_started, self.translation_requested),
            "finalization_overhead_ms": milliseconds(self.final_ready, self.translation_finished),
            "broadcast_ms": milliseconds(self.broadcast_finished, self.final_ready),
        }

    def relative_stages(self, origin: float) -> dict:
        return {
            name: round((value - origin) * 1000, 3)
            for name, value in vars(self).items()
            if name not in {"utterance_id", "endpoint_reason", "timing_source", *self.sample_metadata()}
            and value is not None
        }

    def sample_metadata(self) -> dict:
        return {name: getattr(self, name) for name in SAMPLE_COLUMNS}


class RenderTracker:
    """Bounded per-connection pending events; acknowledgments never block finals."""

    def __init__(self, max_pending: int = 2048, ttl_s: float = 60):
        self.pending: dict[tuple[int, str], tuple[float, float | None, str, dict, dict]] = {}
        self.max_pending, self.ttl_s = max_pending, ttl_s

    def sent(
        self,
        client: Any,
        event_id: str,
        sent: float,
        speech_end: float | None,
        stage: str,
        metadata: dict | None = None,
        *,
        preview_start: float | None = None,
        preview_end: float | None = None,
        preview_speech_end: float | None = None,
    ) -> None:
        self.pending = {key: item for key, item in self.pending.items() if sent - item[0] < self.ttl_s}
        while len(self.pending) >= self.max_pending:
            self.pending.pop(next(iter(self.pending)))
        # These references come from the producer's clock, never the browser.
        # A pause preview's last captured frame includes the pause; its last
        # VAD-positive frame is a separate endpoint and must remain separate.
        preview_references = {
            "speech_start_to_preview_ack_upper_bound_ms": preview_start,
            "captured_end_to_preview_ack_upper_bound_ms": preview_end,
            "speech_end_to_preview_ack_upper_bound_ms": preview_speech_end,
        }
        self.pending[id(client), event_id] = (sent, speech_end, stage, metadata or {}, preview_references)

    def acknowledge(self, client: Any, message: dict, now: float) -> dict | None:
        event_id = message.get("event_id")
        if not isinstance(event_id, str):
            return None
        item = self.pending.pop((id(client), event_id), None)
        if item is None or now - item[0] > self.ttl_s:
            return None
        render_ms = message.get("receive_to_render_ms")
        if isinstance(render_ms, bool) or not isinstance(render_ms, (int, float)):
            return None
        if not math.isfinite(render_ms) or not 0 <= render_ms <= self.ttl_s * 1000:
            return None
        sent, speech_end, stage, metadata, preview_references = item
        visible = message.get("visible") is True
        return {
            **metadata,
            "event": "caption_rendered",
            "event_id": event_id,
            "stage": stage,
            "client_id": str(id(client)),
            "visible": visible,
            "receive_to_render_ms": round(render_ms, 1),
            "send_to_ack_ms": milliseconds(now, sent),
            "speech_end_to_ack_upper_bound_ms": milliseconds(now, speech_end) if visible else None,
            **{
                name: milliseconds(now, reference) if visible and stage == "partial" else None
                for name, reference in preview_references.items()
            },
            "timing_schema_version": TIMING_SCHEMA_VERSION,
        }

    def disconnect(self, client: Any) -> None:
        self.pending = {key: item for key, item in self.pending.items() if key[0] != id(client)}
