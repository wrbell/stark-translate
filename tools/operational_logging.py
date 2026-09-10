"""Bounded asynchronous operational logs, separate from required recording data."""

from __future__ import annotations

import json
import logging
import logging.handlers
import os
import queue
import re
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

MAX_BYTES = 20 * 1024 * 1024
BACKUPS = 5
RETENTION_DAYS = 30


def redact_message(value):
    text = str(value)
    text = re.sub(r"(?i)(authorization|token|api[_-]?key|password|secret)([\s=:]+)[^\s,;]+", r"\1\2[REDACTED]", text)
    text = re.sub(r"(?i)bearer\s+\S+", "Bearer [REDACTED]", text)
    text = re.sub(r"\b(?:hf_|ghp_|github_pat_|sk-)[A-Za-z0-9_-]{16,}\b", "[REDACTED]", text)
    text = text.replace(str(Path.home()), "~")
    return text[:4000]


class JsonLogFormatter(logging.Formatter):
    def format(self, record):
        return json.dumps(
            {
                "schema_version": 1,
                "at": datetime.fromtimestamp(record.created, UTC).isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "session_id": getattr(record, "session_id", None),
                "task_id": getattr(record, "task_id", None),
                "event": getattr(record, "event", "log"),
                "stage": getattr(record, "stage", None),
                "message": redact_message(record.getMessage()),
            },
            ensure_ascii=False,
        )


class _PrivateRotatingFileHandler(logging.handlers.RotatingFileHandler):
    def _open(self):
        stream = open(
            self.baseFilename,
            self.mode,
            encoding=self.encoding,
            errors=self.errors,
            opener=lambda path, flags: os.open(path, flags, 0o600),
        )
        if os.name != "nt":
            os.fchmod(stream.fileno(), 0o600)
        return stream


class AsyncOperationalHandler(logging.Handler):
    """Logging overload is counted and visible, never allowed to block capture."""

    def __init__(
        self,
        path: Path,
        *,
        capacity=2048,
        max_bytes=MAX_BYTES,
        backups=BACKUPS,
        private=False,
        prune_backups=True,
    ):
        super().__init__(logging.INFO)
        self.path = path.absolute()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.queue = queue.Queue(maxsize=capacity)
        self.dropped = self.write_failures = 0
        self._closed_event = threading.Event()
        sink_class = _PrivateRotatingFileHandler if private else logging.handlers.RotatingFileHandler
        self._sink = sink_class(self.path, maxBytes=max_bytes, backupCount=backups, encoding="utf-8")
        self._sink.setFormatter(JsonLogFormatter())
        # RotatingFileHandler normally swallows disk errors and prints stderr;
        # expose them as health counters instead.
        self._sink.handleError = lambda record: self._write_failed()
        if prune_backups:
            self.prune()
        self._thread = threading.Thread(target=self._run, name="operational-log", daemon=True)
        self._thread.start()

    def _write_failed(self):
        self.write_failures += 1

    def emit(self, record):
        if self._closed_event.is_set():
            self.dropped += 1
            return
        try:
            self.queue.put_nowait(record)
        except queue.Full:
            self.dropped += 1

    def _run(self):
        try:
            while not self._closed_event.is_set() or not self.queue.empty():
                try:
                    record = self.queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                try:
                    self._sink.handle(record)
                except Exception:
                    self.write_failures += 1
                finally:
                    self.queue.task_done()
        finally:
            # The writer owns the sink, including after close() times out on a
            # slow filesystem. It must eventually close rather than leak a file.
            self._sink.close()

    def snapshot(self):
        return {"queued": self.queue.qsize(), "dropped": self.dropped, "write_failures": self.write_failures}

    def prune(self, now=None):
        cutoff = (now or time.time()) - RETENTION_DAYS * 86400
        for number in range(1, BACKUPS + 1):
            path = self.path.with_name(f"{self.path.name}.{number}")
            if path.is_file() and not path.is_symlink() and path.stat().st_mtime < cutoff:
                path.unlink()

    def close(self):
        if not self._closed_event.is_set():
            self._closed_event.set()
            self._thread.join(timeout=3)
        super().close()


def configure_log(logger, path):
    """Idempotent per logger/path. Import/reload never leaks another handler."""
    path = Path(path).absolute()
    for handler in logger.handlers:
        if isinstance(handler, AsyncOperationalHandler) and handler.path == path:
            return handler
    handler = AsyncOperationalHandler(path)
    logger.addHandler(handler)
    if logger.level > logging.INFO or logger.level == logging.NOTSET:
        logger.setLevel(logging.INFO)
    return handler


def prune_completed_logs(root: Path, *, now=None):
    """Apply 30-day retention only to operational logs of proven completed runs.

    Audio, predictions, corrections and export artifacts are never auto-deleted.
    Unknown/interrupted recordings require explicit operator attention.
    """
    from tools.session_lifecycle import session_status

    directory = root / "metrics"
    if not directory.is_dir() or directory.is_symlink():
        return 0
    cutoff = (now or time.time()) - RETENTION_DAYS * 86400
    removed = 0
    completed = {}
    for path in directory.glob("*.log*"):
        match = re.fullmatch(r"(?:session|llama)_([A-Za-z0-9][A-Za-z0-9_.-]{0,159})\.log(?:\.[1-5])?", path.name)
        if not match or path.is_symlink() or not path.is_file() or path.stat().st_mtime >= cutoff:
            continue
        session = match[1]
        if ".." in session:
            continue
        if session not in completed:
            completed[session] = session_status(root, session).get("exportable", False)
        if completed[session]:
            path.unlink()
            removed += 1
    return removed
