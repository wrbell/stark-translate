"""CPU-testable scheduling primitives; no inference imports or model state."""

from __future__ import annotations

import threading
import time
from collections import OrderedDict, deque
from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any


@dataclass
class _Request:
    future: Future
    fn: Callable
    args: tuple
    key: Any
    submitted: float


class LatestSTTWorker:
    """One physical worker, FIFO finals and one replaceable pending partial.

    Cancellation never pretends that a running model has stopped. Final work
    outranks the pending partial; running inference remains non-preemptible.
    """

    def __init__(self, on_event: Callable | None = None, max_finals: int = 8):
        self._condition = threading.Condition()
        self._finals: deque[_Request] = deque()
        self._partial: _Request | None = None
        self._active: _Request | None = None
        self._closing = False
        self._max_finals = max_finals
        self._event = on_event or (lambda *args, **kwargs: None)
        self._thread = threading.Thread(target=self._run, name="stt-owner", daemon=True)
        self._thread.start()

    def submit(self, kind: str, fn: Callable, *args, key=None) -> Future:
        if kind not in {"partial", "final"}:
            raise ValueError("STT work must be partial or final")
        future: Future = Future()
        request = _Request(future, fn, args, key, time.perf_counter())
        with self._condition:
            if self._closing:
                raise RuntimeError("STT worker is closed")
            if kind == "partial":
                if self._partial is not None:
                    self._partial.future.cancel()
                    self._event("partial_replaced")
                self._partial = request
            else:
                if len(self._finals) >= self._max_finals:
                    raise RuntimeError("STT final queue capacity exceeded")
                self.cancel_partial(key)
                self._finals.append(request)
            self._condition.notify()
        return future

    def cancel_partial(self, key=None) -> bool:
        with self._condition:
            if self._partial is not None and (key is None or self._partial.key == key):
                self._partial.future.cancel()
                self._partial = None
                self._event("partial_cancelled_pending")
                return True
        return False

    @property
    def busy(self) -> bool:
        with self._condition:
            return self._active is not None or bool(self._finals) or self._partial is not None

    def _run(self):
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._closing or self._finals or self._partial)
                if self._finals:
                    request = self._finals.popleft()
                    kind = "final"
                elif self._partial is not None:
                    request, self._partial = self._partial, None
                    kind = "partial"
                elif self._closing:
                    return
                else:
                    continue
                if not request.future.set_running_or_notify_cancel():
                    continue
                self._active = request
            self._event("stt_worker_started", kind=kind, wait_ms=(time.perf_counter() - request.submitted) * 1000)
            try:
                result = request.fn(*request.args)
            except BaseException as exc:
                request.future.set_exception(exc)
            else:
                request.future.set_result(result)
            finally:
                with self._condition:
                    self._active = None
                self._event("stt_worker_finished", kind=kind)

    def shutdown(self, wait: bool = True):
        with self._condition:
            self._closing = True
            self.cancel_partial()
            self._condition.notify_all()
        if wait:
            self._thread.join()


class ExactTextMemo:
    """Bounded session-local LRU. Exact request identity; no text normalization."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self._values: OrderedDict[tuple, str] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: tuple) -> str | None:
        with self._lock:
            if key not in self._values:
                return None
            self._values.move_to_end(key)
            return self._values[key]

    def put(self, key: tuple, value: str):
        if self.capacity <= 0:
            return
        with self._lock:
            self._values[key] = value
            self._values.move_to_end(key)
            while len(self._values) > self.capacity:
                self._values.popitem(last=False)
