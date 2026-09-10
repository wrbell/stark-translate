"""Bounded monotonic research trace, emitted once with the session summary."""

from __future__ import annotations

import threading
import time
from collections import deque


class LatencyTrace:
    def __init__(self, enabled: bool = False, capacity: int = 8192):
        self.enabled = enabled
        self.origin = time.perf_counter()
        self._records: deque = deque(maxlen=capacity)
        self._lock = threading.Lock()
        self._total = 0

    def record(self, event: str, **fields):
        if not self.enabled:
            return
        record = {"event": event, "at_ms": (time.perf_counter() - self.origin) * 1000, **fields}
        with self._lock:
            self._total += 1
            self._records.append(record)

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "schema_version": 1,
                "clock": "session_monotonic",
                "enabled": self.enabled,
                "total_events": self._total,
                "discarded_old_events": self._total - len(self._records),
                "events": list(self._records),
            }
