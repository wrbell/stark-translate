"""Low-rate local health/control channel, independent of inference imports."""

from __future__ import annotations

import json
import math
import shutil
import threading
import time
from collections import deque
from pathlib import Path

from tools.caption_delivery import FinalCaptionHistory, partial_utterance_id
from tools.session_lifecycle import _path, _write


class PipelineHealth:
    def __init__(self, root: Path, session: str, *, persistence=None, record_audio=True):
        self.root, self.session = root, session
        self.persistence = persistence
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread = None
        self._phase = "loading"
        self._last_input = self._last_caption = None
        self._level = 0.0
        self._errors = deque(maxlen=32)
        self._error_count = 0
        self._required_failures = 0
        self._captions = deque(maxlen=8)
        self._caption_finals = FinalCaptionHistory()
        self._provider = lambda: {}
        self._control = None
        self._control_seq = 0
        self.record_audio = record_audio
        self.paused = False
        self.publish_failures = 0

    def start(self, provider=None, control=None):
        self._provider = provider or self._provider
        self._control = control
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="pipeline-health", daemon=True)
            self._thread.start()

    def phase(self, phase):
        with self._lock:
            self._phase = phase

    def input(self, level=0.0):
        with self._lock:
            self._last_input = time.monotonic()
            self._level = min(1.0, max(0.0, float(level))) if math.isfinite(float(level)) else 0.0
            if self._phase in {"listening", "loading"}:
                self._phase = "ready"

    def caption(self, record):
        with self._lock:
            if record.get("session_id", self.session) != self.session:
                return
            uid = partial_utterance_id(record)
            if record.get("stage") == "partial" and self._caption_finals.blocks(uid):
                return
            if record.get("stage", "complete") == "complete":
                # Final chunk numbers are not capture utterance identities.
                self._caption_finals.observe(record.get("utterance_id"))
                self._captions = deque(
                    (
                        row
                        for row in self._captions
                        if not (
                            row.get("stage") == "partial" and self._caption_finals.blocks(partial_utterance_id(row))
                        )
                    ),
                    maxlen=8,
                )
            self._last_caption = time.monotonic()
            self._captions.append(
                {
                    k: record.get(k)
                    for k in (
                        "session_id",
                        "utterance_id",
                        "chunk_id",
                        "stage",
                        "english",
                        "spanish_a",
                        "source_lang",
                        "target_lang",
                        "speaker",
                    )
                }
            )

    def discard_utterance(self, session_id, utterance_id):
        """Remove discarded interim snapshots, preserving finals with equal IDs."""
        if session_id != self.session:
            return
        with self._lock:
            self._captions = deque(
                (
                    row
                    for row in self._captions
                    if not (
                        row.get("stage") == "partial"
                        and row.get("session_id") in (None, session_id)
                        and (row.get("utterance_id") if row.get("utterance_id") is not None else row.get("chunk_id"))
                        == utterance_id
                    )
                ),
                maxlen=8,
            )

    def error(self, stage, code):
        with self._lock:
            self._error_count += 1
            if stage in {"stt", "translation", "capture"}:
                self._required_failures += 1
            self._errors.append({"stage": stage, "code": code, "at": time.time()})

    def snapshot(self):
        with self._lock:
            now = time.monotonic()
            data = {
                "schema_version": 1,
                "session_id": self.session,
                "updated_at": time.time(),
                "phase": "paused" if self.paused else self._phase,
                "input_seen": self._last_input is not None,
                "input_age_s": now - self._last_input if self._last_input else None,
                "caption_age_s": now - self._last_caption if self._last_caption else None,
                "input_level": self._level,
                "errors": list(self._errors),
                "error_count": self._error_count,
                "captions": list(self._captions),
                "recording": {"audio_enabled": self.record_audio, "required_failures": self._required_failures},
                "control_sequence": self._control_seq,
                "publish_failures": self.publish_failures,
            }
        data["persistence"] = self.persistence.snapshot() if self.persistence else None
        data["recording"]["ok"] = not data["recording"]["required_failures"] and (
            data["persistence"] is None or data["persistence"]["ok"]
        )
        try:
            free = shutil.disk_usage(self.root).free
            data["storage"] = {"free_bytes": free, "low_space": free < 1024**3}
        except OSError:
            data["storage"] = {"free_bytes": None, "low_space": None}
        data.update(self._provider())
        return data

    def _run(self):
        while not self._stop.is_set():
            try:
                path = _path(self.root, self.session, "control", "json")
                if path.is_file():
                    command = json.loads(path.read_text())
                    sequence = command.get("sequence", 0)
                    if (
                        isinstance(sequence, int)
                        and sequence > self._control_seq
                        and command.get("session_id") == self.session
                        and command.get("operation") in {"pause", "resume", "stop"}
                        and self._control
                    ):
                        self._control(command["operation"])
                        self._control_seq = sequence
                _write(_path(self.root, self.session, "health", "json"), self.snapshot())
            except (OSError, ValueError, TypeError):
                self.publish_failures += 1
            self._stop.wait(0.5)

    def close(self, outcome):
        self.paused = False
        self.phase(outcome)
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        try:
            _write(_path(self.root, self.session, "health", "json"), self.snapshot())
        except OSError:
            self.publish_failures += 1


def read_health(root, session, *, stale_after=3.0):
    try:
        data = json.loads(_path(root, session, "health", "json").read_text())
        if data.get("session_id") != session or data.get("schema_version") != 1:
            raise ValueError("Wrong health session")
        age = max(0, time.time() - float(data["updated_at"]))
        data.update(age_s=age, stale=age > stale_after)
        return data
    except (OSError, ValueError, TypeError, KeyError):
        return {"session_id": session, "phase": "unknown", "stale": True, "age_s": None}


def send_control(root, session, operation):
    if operation not in {"pause", "resume", "stop"}:
        raise ValueError("Unsupported operation")
    sequence = time.time_ns()
    _write(
        _path(root, session, "control", "json"), {"session_id": session, "sequence": sequence, "operation": operation}
    )
    return sequence
