"""Bounded, acknowledged background persistence; never hide a failed write."""

from __future__ import annotations

import threading
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor


class PersistenceExecutor(ThreadPoolExecutor):
    """Executor-compatible writer with a bounded admission queue and failure ledger.

    Submission never blocks the audio loop. Overload rejects a write explicitly,
    marks the recording incomplete, and returns a failed Future. Shutdown drains
    admitted writes; callers must consult ``snapshot`` before declaring success.
    """

    def __init__(self, max_workers=2, *, max_pending=256, thread_name_prefix="persistence"):
        super().__init__(max_workers=max_workers, thread_name_prefix=thread_name_prefix)
        self._slots = threading.BoundedSemaphore(max_pending)
        self._ledger_lock = threading.RLock()
        self._submitted = self._completed = self._failed = self._pending = 0
        self._failures = deque(maxlen=32)
        self._capacity = max_pending

    def submit(self, fn, /, *args, **kwargs):
        label = getattr(fn, "__name__", type(fn).__name__)
        with self._ledger_lock:
            self._submitted += 1
        if not self._slots.acquire(blocking=False):
            future = Future()
            future.set_exception(RuntimeError("Persistence queue is full; recording is incomplete"))
            self.record_failure(label, "queue_full")
            return future
        with self._ledger_lock:
            self._pending += 1
        try:
            future = super().submit(fn, *args, **kwargs)
        except BaseException:
            self._slots.release()
            with self._ledger_lock:
                self._pending -= 1
            self.record_failure(label, "submission_failed")
            raise

        def finished(result):
            try:
                failure = result.exception() if not result.cancelled() else RuntimeError("cancelled")
                if failure is not None:
                    # Exception text may contain transcript text or private paths.
                    self.record_failure(label, type(failure).__name__)
                else:
                    with self._ledger_lock:
                        self._completed += 1
            finally:
                with self._ledger_lock:
                    self._pending -= 1
                self._slots.release()

        future.add_done_callback(finished)
        return future

    def record_failure(self, operation, code):
        with self._ledger_lock:
            self._failed += 1
            self._failures.append({"operation": operation, "code": code})

    def snapshot(self):
        with self._ledger_lock:
            return {
                "submitted": self._submitted,
                "completed": self._completed,
                "pending": self._pending,
                "failed": self._failed,
                "capacity": self._capacity,
                "ok": self._failed == 0,
                "failures": list(self._failures),
            }
