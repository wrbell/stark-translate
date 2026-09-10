"""Bounded cross-thread capture handoff with at most one scheduled drain."""

import time
from collections import deque
from threading import Condition


class CaptureHandoff:
    def __init__(
        self,
        loop,
        consume,
        can_accept,
        on_drop,
        *,
        capacity=32,
        wait_for_space=False,
        trace=None,
        frame_metadata=None,
        clock=time.perf_counter,
    ):
        self.loop, self.consume, self.can_accept, self.on_drop = loop, consume, can_accept, on_drop
        self.capacity, self.wait_for_space = capacity, wait_for_space
        self._items = deque()
        self._condition = Condition()
        self._scheduled = self._closed = False
        self.dropped = 0
        self.trace, self.frame_metadata, self.clock = trace, frame_metadata, clock
        self.origin = trace.origin if trace is not None else clock()
        self.submitted = self.dequeued = self.high_water = 0
        self.max_producer_wait_ms = self.max_handoff_wait_ms = self.max_capture_age_ms = 0.0
        self.drop_spans = deque(maxlen=32)

    def _event(self, event, item, now, **fields):
        metadata = dict(self.frame_metadata(item)) if self.frame_metadata is not None else {}
        captured_end = metadata.pop("capture_end_perf_counter_s", None)
        if captured_end is not None:
            fields["capture_age_ms"] = max(0.0, (now - captured_end) * 1000)
            self.max_capture_age_ms = max(self.max_capture_age_ms, fields["capture_age_ms"])
        record = {**metadata, **fields}
        if event == "capture_handoff_dropped":
            self.drop_spans.append({"at_ms": (now - self.origin) * 1000, **record})
        if self.trace is not None:
            self.trace.record(event, **record)

    def put(self, *item):
        requested = self.clock()
        with self._condition:
            while self.wait_for_space and len(self._items) >= self.capacity and not self._closed:
                self._condition.wait(0.1)
            if self._closed:
                return
            now = self.clock()
            producer_wait_ms = max(0.0, (now - requested) * 1000)
            self.max_producer_wait_ms = max(self.max_producer_wait_ms, producer_wait_ms)
            if len(self._items) >= self.capacity:
                queued_at, dropped = self._items.popleft()
                self.dropped += 1
                self._event(
                    "capture_handoff_dropped", dropped, now, reason="overflow", wait_ms=(now - queued_at) * 1000
                )
                self.on_drop()
            self._items.append((now, item))
            self.submitted += 1
            self.high_water = max(self.high_water, len(self._items))
            self._event(
                "capture_handoff_enqueued", item, now, producer_wait_ms=producer_wait_ms, queue_depth=len(self._items)
            )
            if not self._scheduled:
                self._scheduled = True
                self.loop.call_soon_threadsafe(self._drain)

    def _drain(self):
        for _ in range(8):
            with self._condition:
                if self._closed or not self._items:
                    self._scheduled = False
                    return
                if not self.can_accept():
                    break
                queued_at, item = self._items.popleft()
                now = self.clock()
                wait_ms = max(0.0, (now - queued_at) * 1000)
                self.dequeued += 1
                self.max_handoff_wait_ms = max(self.max_handoff_wait_ms, wait_ms)
                self._event("capture_handoff_dequeued", item, now, wait_ms=wait_ms, queue_depth=len(self._items))
                self._condition.notify_all()
            try:
                self.consume(*item)
            except Exception:
                # A malformed frame must not strand later audio behind a stuck
                # scheduled flag. Recording completeness remains explicitly false.
                with self._condition:
                    self.dropped += 1
                    self._event("capture_handoff_dropped", item, self.clock(), reason="consume_error")
                    self.on_drop()
        with self._condition:
            if not self._closed:
                self.loop.call_later(0.005, self._drain)

    def qsize(self):
        with self._condition:
            return len(self._items)

    def snapshot(self):
        with self._condition:
            return {
                "schema_version": 1,
                "clock": "session_monotonic",
                "origin_perf_counter_s": self.origin,
                "capacity": self.capacity,
                "closed": self._closed,
                "wait_for_space": self.wait_for_space,
                "submitted": self.submitted,
                "dequeued": self.dequeued,
                "pending": len(self._items),
                "high_water": self.high_water,
                "dropped_frames": self.dropped,
                "max_producer_wait_ms": self.max_producer_wait_ms,
                "max_handoff_wait_ms": self.max_handoff_wait_ms,
                "max_capture_age_ms": self.max_capture_age_ms,
                "drop_spans": list(self.drop_spans),
                "drop_spans_truncated": self.dropped > len(self.drop_spans),
            }

    def __enter__(self):
        return self

    def close(self, *, record_discard=True):
        with self._condition:
            self._closed = True
            if self._items and record_discard:
                self.dropped += len(self._items)
                for queued_at, item in self._items:
                    now = self.clock()
                    self._event("capture_handoff_dropped", item, now, reason="close", wait_ms=(now - queued_at) * 1000)
                self.on_drop()
            self._items.clear()
            self._condition.notify_all()

    def __exit__(self, *args):
        self.close()


class CaptureTransportSummary:
    """Retain closed capture segments across pause/resume with bounded detail."""

    def __init__(self):
        self.segments = deque(maxlen=16)
        self.count = self.handoff_drops = self.upstream_dropped_samples = 0

    def record(self, stream, handoff):
        record = {"stream_type": type(stream).__name__, "handoff": handoff.snapshot()}
        if hasattr(stream, "capture_snapshot"):
            record["pipe"] = stream.capture_snapshot()
            self.upstream_dropped_samples += record["pipe"]["upstream_dropped_samples"]
        self.handoff_drops += record["handoff"]["dropped_frames"]
        self.count += 1
        self.segments.append(record)

    def snapshot(self):
        return {
            "schema_version": 1,
            "scope": "closed capture segments; original source frames; no PCM",
            "segments_completed": self.count,
            "handoff_dropped_frames": self.handoff_drops,
            "upstream_dropped_samples": self.upstream_dropped_samples,
            "segments": list(self.segments),
            "segments_truncated": self.count > len(self.segments),
        }
