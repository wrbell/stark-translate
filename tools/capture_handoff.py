"""Bounded cross-thread capture handoff with at most one scheduled drain."""

from collections import deque
from threading import Condition


class CaptureHandoff:
    def __init__(self, loop, consume, can_accept, on_drop, *, capacity=32, wait_for_space=False):
        self.loop, self.consume, self.can_accept, self.on_drop = loop, consume, can_accept, on_drop
        self.capacity, self.wait_for_space = capacity, wait_for_space
        self._items = deque()
        self._condition = Condition()
        self._scheduled = self._closed = False
        self.dropped = 0

    def put(self, *item):
        with self._condition:
            while self.wait_for_space and len(self._items) >= self.capacity and not self._closed:
                self._condition.wait(0.1)
            if self._closed:
                return
            if len(self._items) >= self.capacity:
                self._items.popleft()
                self.dropped += 1
                self.on_drop()
            self._items.append(item)
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
                item = self._items.popleft()
                self._condition.notify_all()
            try:
                self.consume(*item)
            except Exception:
                # A malformed frame must not strand later audio behind a stuck
                # scheduled flag. Recording completeness remains explicitly false.
                self.dropped += 1
                self.on_drop()
        with self._condition:
            if not self._closed:
                self.loop.call_later(0.005, self._drain)

    def qsize(self):
        with self._condition:
            return len(self._items)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        with self._condition:
            self._closed = True
            if self._items:
                self.dropped += len(self._items)
                self.on_drop()
            self._items.clear()
            self._condition.notify_all()
