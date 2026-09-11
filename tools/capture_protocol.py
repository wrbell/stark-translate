"""Metadata-only capture status and terminal receipts; no native imports."""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path


def count(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f"Invalid capture counter: {name}")
    return value


class CaptureStatus(str):
    """Preserve historical status strings while retaining independent loss flags."""

    def __new__(cls, text="", *, fifo_dropped_samples=0, input_overflow_callbacks=0):
        count(fifo_dropped_samples, "fifo_dropped_samples")
        count(input_overflow_callbacks, "input_overflow_callbacks")
        display = f"capture_overflow:{fifo_dropped_samples}" if fifo_dropped_samples else str(text)
        if input_overflow_callbacks and not display:
            display = "input overflow"
        result = super().__new__(cls, display)
        result.fifo_dropped_samples = fifo_dropped_samples
        result.input_overflow_callbacks = input_overflow_callbacks
        result.portaudio_status = str(text)
        return result


def capture_status(status):
    """Accept modern status, historical strings and direct CallbackFlags."""
    if isinstance(status, CaptureStatus):
        return status
    text = str(status) if status else ""
    fifo = 0
    if text.startswith("capture_overflow:"):
        fifo = count(int(text.removeprefix("capture_overflow:")), "legacy dropped")
    flags = {part.strip() for part in text.split(",")}
    overflow = getattr(status, "input_overflow", False) is True or "input overflow" in flags
    return CaptureStatus(text, fifo_dropped_samples=fifo, input_overflow_callbacks=int(overflow))


def status_from_metadata(metadata, previous_fifo, previous_overflows):
    fifo = count(metadata["dropped"], "dropped")
    if fifo < previous_fifo:
        raise ValueError("Capture FIFO counter decreased")
    if (
        "worker_fifo_dropped_samples" in metadata
        and count(metadata["worker_fifo_dropped_samples"], "worker_fifo_dropped_samples") != fifo
    ):
        raise ValueError("Capture FIFO aliases disagree")
    raw_status = metadata.get("status", "")
    if not isinstance(raw_status, str):
        raise ValueError("Invalid capture status")
    legacy = capture_status(raw_status)
    if raw_status.startswith("capture_overflow:") and legacy.fifo_dropped_samples != fifo - previous_fifo:
        raise ValueError("Capture FIFO status and counter disagree")
    if "portaudio_input_overflow" in metadata and type(metadata["portaudio_input_overflow"]) is not bool:
        raise ValueError("Invalid PortAudio input-overflow flag")
    flag = metadata.get("portaudio_input_overflow", bool(legacy.input_overflow_callbacks))
    if "portaudio_input_overflow" in metadata and flag != bool(legacy.input_overflow_callbacks):
        raise ValueError("PortAudio input-overflow flag and status disagree")
    overflows = count(
        metadata.get("portaudio_input_overflow_callbacks", previous_overflows + int(flag)),
        "portaudio_input_overflow_callbacks",
    )
    if overflows < previous_overflows or (flag and overflows == previous_overflows):
        raise ValueError("PortAudio input-overflow counter is inconsistent")
    return (
        fifo,
        overflows,
        CaptureStatus(
            metadata.get("status", ""),
            fifo_dropped_samples=fifo - previous_fifo,
            input_overflow_callbacks=overflows - previous_overflows,
        ),
    )


def write_terminal_receipt(path, receipt):
    """Atomic side channel, outside the native callback and congested audio pipe."""
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix="capture-terminal-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(receipt, stream, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_terminal_receipt(path, token, *, observed_fifo, observed_overflows, observed_end):
    path = Path(path)
    if not path.is_file() or path.stat().st_size > 8192:
        raise ValueError("Capture terminal receipt unavailable")
    raw = path.read_bytes()
    if len(raw) > 8192:
        raise ValueError("Capture terminal receipt oversized")
    receipt = json.loads(raw)
    if (
        not isinstance(receipt, dict)
        or type(receipt.get("schema_version")) is not int
        or receipt.get("schema_version") != 1
        or not re.fullmatch(r"[0-9a-f]{32}", token)
        or receipt.get("capture_token") != token
        or receipt.get("stream_closed") is not True
        or receipt.get("stop_reason") not in {"requested_stop", "capture_error"}
    ):
        raise ValueError("Capture terminal identity or closure unavailable")
    fields = (
        "callback_samples",
        "callback_count",
        "worker_fifo_dropped_samples",
        "worker_fifo_dropped_callbacks",
        "worker_fifo_admitted_samples",
        "portaudio_input_overflow_callbacks",
        "pipe_written_samples",
        "worker_fifo_pending_samples",
        "writer_unfinished_samples",
    )
    for field in fields:
        count(receipt.get(field), field)
    if (
        receipt["worker_fifo_dropped_samples"] < observed_fifo
        or receipt["portaudio_input_overflow_callbacks"] < observed_overflows
        or receipt["callback_samples"] < observed_end
        or receipt["worker_fifo_dropped_callbacks"] > receipt["callback_count"]
        or receipt["worker_fifo_dropped_callbacks"] > receipt["worker_fifo_dropped_samples"]
        or receipt["callback_count"] - receipt["worker_fifo_dropped_callbacks"]
        > receipt["worker_fifo_admitted_samples"]
        or bool(receipt["callback_count"] - receipt["worker_fifo_dropped_callbacks"])
        != bool(receipt["worker_fifo_admitted_samples"])
        or receipt["portaudio_input_overflow_callbacks"] > receipt["callback_count"]
        or bool(receipt["worker_fifo_dropped_callbacks"]) != bool(receipt["worker_fifo_dropped_samples"])
        or bool(receipt["callback_count"]) != bool(receipt["callback_samples"])
        or receipt["callback_count"] > receipt["callback_samples"]
        or receipt["worker_fifo_admitted_samples"]
        != receipt["callback_samples"] - receipt["worker_fifo_dropped_samples"]
        # Already parsed frames cannot still be queued in the worker. Permit
        # post-flush/pre-bookkeeping interruption in writer_unfinished_samples.
        or receipt["pipe_written_samples"] + receipt["writer_unfinished_samples"] < observed_end - observed_fifo
        or receipt["callback_samples"]
        != sum(
            receipt[field]
            for field in (
                "worker_fifo_dropped_samples",
                "pipe_written_samples",
                "worker_fifo_pending_samples",
                "writer_unfinished_samples",
            )
        )
    ):
        raise ValueError("Capture terminal counters are inconsistent")
    return receipt
