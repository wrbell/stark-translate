"""Clock-local pipeline timing, including frame-preserving utterance cuts.

Speech end means the end of the last VAD-positive frame, not a forced claim
about acoustic ground truth. No timestamp here is comparable across hosts.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

TIMING_SCHEMA_VERSION = 2
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


@dataclass(frozen=True)
class AudioFrame:
    samples: Any
    stamp: CaptureStamp


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
                boundary = stamp.start + (stamp.end - stamp.start) * n / size
                taken.append((n, CaptureStamp(stamp.start, boundary, stamp.source), speech))
            if n < size:
                boundary = stamp.start + (stamp.end - stamp.start) * n / size
                rest.append((size - n, CaptureStamp(boundary, stamp.end, stamp.source), speech))
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
        return next((stamp.end for _, stamp, speech in reversed(self.spans) if speech), None)

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

    @classmethod
    def from_timeline(cls, timeline: AudioTimeline, utterance_id: int, reason: str) -> ChunkTiming:
        return cls(utterance_id, reason, timeline.source, timeline.first, timeline.speech_end, time.perf_counter())

    def metrics(self) -> dict:
        return {
            "timing_schema_version": TIMING_SCHEMA_VERSION,
            "utterance_id": self.utterance_id,
            "endpoint_reason": self.endpoint_reason,
            "timing_source": self.timing_source,
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
            if name not in {"utterance_id", "endpoint_reason", "timing_source"} and value is not None
        }


class RenderTracker:
    """Bounded per-connection pending events; acknowledgments never block finals."""

    def __init__(self, max_pending: int = 2048, ttl_s: float = 60):
        self.pending: dict[tuple[int, str], tuple[float, float | None, str, dict]] = {}
        self.max_pending, self.ttl_s = max_pending, ttl_s

    def sent(
        self,
        client: Any,
        event_id: str,
        sent: float,
        speech_end: float | None,
        stage: str,
        metadata: dict | None = None,
    ) -> None:
        self.pending = {key: item for key, item in self.pending.items() if sent - item[0] < self.ttl_s}
        while len(self.pending) >= self.max_pending:
            self.pending.pop(next(iter(self.pending)))
        self.pending[id(client), event_id] = (sent, speech_end, stage, metadata or {})

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
        sent, speech_end, stage, metadata = item
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
            "timing_schema_version": TIMING_SCHEMA_VERSION,
        }

    def disconnect(self, client: Any) -> None:
        self.pending = {key: item for key, item in self.pending.items() if key[0] != id(client)}
