"""Bounded device probes and framed live capture from disposable child processes."""

from __future__ import annotations

import json
import struct
import subprocess
import sys
import threading
import time
from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from operator_app.processes import cleanup_children
from tools.pipeline_timing import capture_stamp


class AudioCaptureError(RuntimeError):
    pass


def worker_argv(options):
    return [sys.executable, "-m", "tools.capture_worker", json.dumps(options)]


def probe_audio(mode, device=None, duration_s=2, *, argv=None):
    """A hung native device open is terminated after a bounded timeout."""
    if not 0 < duration_s <= 5:
        raise ValueError("Audio tests must last at most five seconds")
    if mode not in {"probe", "output"}:
        raise ValueError("Invalid audio test")
    proc = subprocess.Popen(
        argv or worker_argv({"mode": mode, "device": device, "duration_s": duration_s}),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=duration_s + 3)
        if proc.returncode:
            raise AudioCaptureError("Audio device could not open. Check its connection and microphone permission.")
        result = json.loads(stdout)
        if not result.get("ok"):
            raise AudioCaptureError("The audio test did not complete")
        return result
    except subprocess.TimeoutExpired as exc:
        raise AudioCaptureError(
            "Audio device did not respond. Check microphone permission and reconnect the selected device."
        ) from exc
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.communicate(timeout=2)
        cleanup_children(proc.pid)


class IsolatedInputStream:
    def __init__(
        self,
        *,
        callback,
        samplerate,
        channels,
        dtype,
        blocksize,
        device,
        startup_timeout=5.0,
        idle_timeout=3.0,
        argv=None,
    ):
        self.callback = callback
        self.rate, self.channels = samplerate, channels
        self.argv = argv or worker_argv(
            {"samplerate": samplerate, "channels": channels, "dtype": dtype, "blocksize": blocksize, "device": device}
        )
        self.startup_timeout, self.idle_timeout = startup_timeout, idle_timeout
        self.finished = threading.Event()
        self.error = None
        self.dropped_samples = 0
        self.sample_offset = 0
        self._stop = threading.Event()
        self._last_frame = None
        self._started = None
        self._proc = self._reader = self._watcher = None

    def __enter__(self):
        # No native calls or wait for permission on the inference event loop.
        self._started = time.monotonic()
        self._proc = subprocess.Popen(
            self.argv, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, start_new_session=False
        )
        self._reader = threading.Thread(target=self._read, name="capture-reader", daemon=True)
        self._watcher = threading.Thread(target=self._watch, name="capture-watchdog", daemon=True)
        self._reader.start()
        self._watcher.start()
        return self

    def _bytes(self, size):
        result = bytearray()
        while len(result) < size:
            block = self._proc.stdout.read(size - len(result))
            if not block:
                raise AudioCaptureError("Audio capture process exited or disconnected")
            result.extend(block)
        return bytes(result)

    def _read(self):
        try:
            while not self._stop.is_set():
                size = struct.unpack("!I", self._bytes(4))[0]
                if not 1 <= size <= 8192:
                    raise AudioCaptureError("Invalid capture frame")
                metadata = json.loads(self._bytes(size))
                frames, channels = metadata["frames"], metadata["channels"]
                if not isinstance(frames, int) or not 1 <= frames <= self.rate or channels != self.channels:
                    raise AudioCaptureError("Invalid capture sample bounds")
                samples = np.frombuffer(self._bytes(frames * channels * 4), dtype="float32").reshape(frames, channels)
                self._last_frame = time.monotonic()
                newly_dropped = metadata["dropped"] - self.dropped_samples
                self.dropped_samples = metadata["dropped"]
                # perf_counter is a common host monotonic clock. Anchor the ADC
                # offset at child callback receipt, never at delayed pipe receipt.
                stamp = capture_stamp(frames, self.rate, SimpleNamespace(**metadata), received=metadata["received"])
                stamp = replace(
                    stamp,
                    sample_start=self.sample_offset + metadata["sample_start"],
                    sample_end=self.sample_offset + metadata["sample_start"] + frames,
                    sample_rate=self.rate,
                )
                status = f"capture_overflow:{newly_dropped}" if newly_dropped else metadata["status"] or None
                self.callback(samples, frames, stamp, status)
        except Exception as exc:
            if not self._stop.is_set() and self.error is None:
                self.error = exc if isinstance(exc, AudioCaptureError) else AudioCaptureError(type(exc).__name__)
                self.finished.set()

    def _watch(self):
        while not self._stop.wait(0.1) and not self.finished.is_set():
            elapsed = time.monotonic() - (self._last_frame or self._started)
            limit = self.idle_timeout if self._last_frame else self.startup_timeout
            if elapsed > limit:
                self.error = AudioCaptureError(
                    "Microphone delivered no samples. Check permission and reconnect the device."
                )
                self.finished.set()
                self._proc.kill()
                return

    def __exit__(self, *args):
        self._stop.set()
        if self._proc is not None:
            if self._proc.poll() is None:
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=0.5)
                except subprocess.TimeoutExpired:
                    self._proc.kill()
            self._proc.wait(timeout=2)
            cleanup_children(self._proc.pid, group=False)
        for thread in (self._reader, self._watcher):
            if thread:
                thread.join(timeout=2)
        if self._proc and self._proc.stdout:
            self._proc.stdout.close()
