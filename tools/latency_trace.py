"""Bounded monotonic research trace, emitted once with the session summary."""

from __future__ import annotations

import threading
import time
from collections import deque
from contextlib import contextmanager


class LatencyTrace:
    def __init__(self, enabled: bool = False, capacity: int = 8192, *, origin: float | None = None):
        if capacity < 1:
            raise ValueError("Trace capacity must be positive")
        self.enabled = enabled
        self.origin = time.perf_counter() if origin is None else origin
        self._records: deque = deque(maxlen=capacity)
        self._lock = threading.Lock()
        self._total = 0

    def record(self, event: str, **fields):
        if not self.enabled:
            return
        record = {
            **fields,
            "event": event,
            "at_ms": (time.perf_counter() - self.origin) * 1000,
            "thread_id": threading.get_ident(),
            "thread_name": threading.current_thread().name,
        }
        with self._lock:
            self._total += 1
            self._records.append(record)

    @contextmanager
    def span(self, event: str, **fields):
        """Record physical lifetime including failures; no extra device synchronization."""
        if not self.enabled:
            yield
            return
        started = time.perf_counter()
        self.record(event + "_started", **fields)
        failed = False
        try:
            yield
        except BaseException:
            failed = True
            raise
        finally:
            self.record(event + "_finished", elapsed_ms=(time.perf_counter() - started) * 1000, failed=failed, **fields)

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "schema_version": 2,
                "clock": "session_monotonic",
                "origin_perf_counter_s": self.origin,
                "clock_scope": "same pipeline process only; not comparable across runs or hosts",
                "enabled": self.enabled,
                "total_events": self._total,
                "discarded_old_events": self._total - len(self._records),
                "events": list(self._records),
            }
