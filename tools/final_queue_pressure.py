"""Bounded, event-loop-owned telemetry for the final pipeline's existing queue.

This observes public ``put``/dequeue boundaries without changing capacity or
backpressure. Payloads must be distinct objects for each live submission (the
pipeline creates a fresh tuple). Utterance IDs are descriptive, never ledger keys.
"""

from __future__ import annotations

import time
from collections import deque


class FinalQueuePressure:
    def __init__(self, *, origin=None, on_event=None, capacity=1024, window_size=16, clock=None):
        if capacity < 1 or window_size < 1:
            raise ValueError("Bookkeeping capacity and wait window must be positive")
        self._clock = clock or time.perf_counter
        self.origin = self._clock() if origin is None else origin
        self._on_event = on_event
        self.capacity = capacity
        self.window_size = window_size
        self._active = {}
        self._first = []
        self._last = deque(maxlen=window_size)
        self._attempts = self._submitted = self._dequeued = self._failed = 0
        self._dropped = self._unmatched = 0
        self._max_pending = self._max_queued = self._max_producers = 0
        self._max_oldest = self._max_queue_wait = self._max_producer_wait = 0.0
        self._queue_capacity = None

    def _counts(self):
        producers = self._attempts - self._submitted - self._failed
        queued = max(0, self._submitted - self._dequeued)
        return queued, producers

    def _sample(self, now):
        queued, producers = self._counts()
        self._max_pending = max(self._max_pending, queued + producers)
        self._max_queued = max(self._max_queued, queued)
        self._max_producers = max(self._max_producers, producers)
        # Insertion order is request order. Sampling at removal and snapshot
        # captures the oldest age without a timer or scanning the queue.
        oldest = next(iter(self._active.values()), None)
        age = max(0.0, (now - oldest["requested"]) * 1000) if oldest else 0.0
        self._max_oldest = max(self._max_oldest, age)
        return queued, producers, age

    def _event(self, event, record, now, **fields):
        if self._on_event is not None:
            queued, producers = self._counts()
            self._on_event(
                event,
                submission_id=record["submission_id"] if record else None,
                utterance_id=record["utterance_id"] if record else None,
                queue_at_ms=(now - self.origin) * 1000,
                queued=queued,
                producer_waiting=producers,
                pending=queued + producers,
                **fields,
            )

    async def put(self, queue, item, *, utterance_id=None):
        """Await the original Queue.put, counting blocked/cancelled producers."""
        now = self._clock()
        self._attempts += 1
        self._queue_capacity = queue.maxsize
        record = {
            "submission_id": self._attempts,
            "utterance_id": utterance_id,
            "requested": now,
            "enqueued": None,
        }
        key = id(item)
        retained = len(self._active) < self.capacity and key not in self._active
        if retained:
            self._active[key] = record
        else:
            self._dropped += 1
        self._sample(now)
        self._event("final_queue_submit_started", record, now)
        try:
            await queue.put(item)
        except BaseException:
            now = self._clock()
            self._sample(now)
            self._max_producer_wait = max(self._max_producer_wait, (now - record["requested"]) * 1000)
            self._failed += 1
            if retained:
                self._active.pop(key)
            self._event("final_queue_submit_failed", record, now)
            raise
        # asyncio.Queue.put has no suspension after insertion, so this boundary
        # precedes the consumer receiving this item, including a formerly full queue.
        now = self._clock()
        record["enqueued"] = now
        self._submitted += 1
        self._max_producer_wait = max(self._max_producer_wait, (now - record["requested"]) * 1000)
        self._sample(now)
        self._event("final_queue_submitted", record, now)

    def dequeued(self, item, *, now=None):
        """Record a successful get before filtering, STT, or downstream failures."""
        now = self._clock() if now is None else now
        self._sample(now)
        record = self._active.pop(id(item), None)
        self._dequeued += 1
        fields = {}
        if record is None or record["enqueued"] is None:
            self._unmatched += 1
        else:
            wait = max(0.0, (now - record["requested"]) * 1000)
            queue_wait = max(0.0, (now - record["enqueued"]) * 1000)
            self._max_queue_wait = max(self._max_queue_wait, queue_wait)
            if len(self._first) < self.window_size:
                self._first.append(wait)
            self._last.append(wait)
            fields = {"wait_ms": wait, "queue_wait_ms": queue_wait}
        self._event("final_queue_dequeued", record, now, **fields)

    def snapshot(self):
        now = self._clock()
        queued, producers, oldest = self._sample(now)
        first, last = list(self._first), list(self._last)
        return {
            "schema_version": 1,
            "clock": "session_monotonic",
            "origin_perf_counter_s": self.origin,
            "scope": "final input queue and waiting producers; excludes sentinel, active STT and translation",
            "wait_definition": "submission request to dequeue; max includes observed outstanding/cancelled ages",
            "sampling": "submit request, successful put, failed put, successful get, summary; no polling",
            "queue_capacity": self._queue_capacity,
            "submit_attempts": self._attempts,
            "submitted": self._submitted,
            "dequeued": self._dequeued,
            "put_failed": self._failed,
            "max_pending": self._max_pending,
            "max_queued": self._max_queued,
            "max_producer_waiting": self._max_producers,
            "max_wait_ms": self._max_oldest,
            "max_oldest_wait_ms": self._max_oldest,
            "max_queue_wait_ms": self._max_queue_wait,
            "max_producer_wait_ms": self._max_producer_wait,
            "first_window_wait_ms": first,
            "last_window_wait_ms": last,
            "first_window_mean_wait_ms": sum(first) / len(first) if first else None,
            "last_window_mean_wait_ms": sum(last) / len(last) if last else None,
            "wait_window_size": self.window_size,
            "wait_windows_overlap": self._dequeued < 2 * self.window_size,
            "terminal_outstanding": queued + producers,
            "terminal_queued": queued,
            "terminal_producer_waiting": producers,
            "terminal_oldest_wait_ms": oldest,
            "bookkeeping_capacity": self.capacity,
            "bookkeeping_active": len(self._active),
            "bookkeeping_dropped": self._dropped,
            "unmatched_dequeues": self._unmatched,
            "bookkeeping_truncated": bool(self._dropped or self._unmatched),
        }
