"""One model/audio job per operator; reservations include subprocess lifetime."""

from __future__ import annotations

import threading
import uuid
from contextlib import contextmanager
from pathlib import Path


class WorkBusyError(RuntimeError):
    def __init__(self, work):
        self.work = {k: v for k, v in work.items() if k != "token"}
        super().__init__(f"Finish {work['kind']} before starting another job")


class WorkLease:
    def __init__(self):
        self._lock = threading.Lock()
        self._work = None

    def acquire(self, kind, job_id=None):
        with self._lock:
            if self._work is not None:
                raise WorkBusyError(self._work.copy())
            token = uuid.uuid4().hex
            self._work = {"kind": kind, "id": job_id or token, "token": token}
            return token

    def release(self, token):
        with self._lock:
            if self._work and self._work["token"] == token:
                self._work = None

    def snapshot(self):
        with self._lock:
            return {k: v for k, v in self._work.items() if k != "token"} if self._work else None

    @contextmanager
    def reserve(self, kind):
        token = self.acquire(kind)
        try:
            yield
        finally:
            self.release(token)


_leases = {}
_lock = threading.Lock()


def get_work_lease(root: Path):
    key = str(root.resolve())
    with _lock:
        return _leases.setdefault(key, WorkLease())
