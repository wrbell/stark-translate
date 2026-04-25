"""Audio device enumeration + hotplug watcher (Phase 9.4).

The mic in the church is on a long USB run; if it drops mid-session the
sounddevice callback hangs silently and the operator has no signal. This
module:

- Exposes ``list_devices()`` returning input + output device descriptors.
- Runs a background ``DeviceWatcher`` thread that polls the device list
  every ~2 s and bumps a ``change_seq`` counter when the set changes.
  Frontend reads the counter from the metrics WS frame and re-fetches
  ``/api/devices``, surfacing a toast when a device disappears.

Multi-channel TTS routing (EN main / ES monitor) is out of scope for 9.4
— the plumbing requires PiperTTSEngine output-device support and a
restart-on-change flow. Documented in plans/we-haven-t-worked-on-lexical-moth.md
as a 9.4.1 follow-up; the current code surfaces output devices in the
listing so the UI can be ready when that lands.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class DeviceInfo:
    index: int
    name: str
    direction: str  # "input" | "output"
    channels: int
    default_sample_rate: float

    def to_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class DeviceListing:
    inputs: list[DeviceInfo] = field(default_factory=list)
    outputs: list[DeviceInfo] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "inputs": [d.to_dict() for d in self.inputs],
            "outputs": [d.to_dict() for d in self.outputs],
            "error": self.error,
        }

    def fingerprint(self) -> tuple:
        """Stable identity used to detect changes between polls."""
        return (
            tuple((d.index, d.name, d.channels) for d in self.inputs),
            tuple((d.index, d.name, d.channels) for d in self.outputs),
        )


def list_devices() -> DeviceListing:
    """Enumerate input + output audio devices via sounddevice."""
    try:
        import sounddevice as sd

        raw = sd.query_devices()
    except Exception as exc:
        return DeviceListing(error=f"sounddevice unavailable: {exc}")

    inputs: list[DeviceInfo] = []
    outputs: list[DeviceInfo] = []
    for idx, d in enumerate(raw):
        name = str(d.get("name", "?"))
        sr = float(d.get("default_samplerate", 0) or 0)
        in_ch = int(d.get("max_input_channels", 0) or 0)
        out_ch = int(d.get("max_output_channels", 0) or 0)
        if in_ch > 0:
            inputs.append(DeviceInfo(index=idx, name=name, direction="input", channels=in_ch, default_sample_rate=sr))
        if out_ch > 0:
            outputs.append(
                DeviceInfo(index=idx, name=name, direction="output", channels=out_ch, default_sample_rate=sr)
            )
    return DeviceListing(inputs=inputs, outputs=outputs)


class DeviceWatcher:
    """Polls audio device list at fixed cadence; tracks a change sequence number."""

    POLL_INTERVAL_S = 2.0

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_fingerprint: tuple | None = None
        self._change_seq = 0
        self._last_listing: DeviceListing | None = None
        self._last_change_ts: float | None = None

    def start(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._loop, name="audio-watcher", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "change_seq": self._change_seq,
                "last_change_ts": self._last_change_ts,
                "last_listing": self._last_listing.to_dict() if self._last_listing else None,
            }

    def force_poll(self) -> DeviceListing:
        """Used by /api/devices to return the latest listing without waiting for the loop."""
        listing = list_devices()
        with self._lock:
            self._update(listing)
        return listing

    def _loop(self) -> None:
        # Prime the fingerprint on first iteration so we don't false-positive.
        first = True
        while not self._stop_event.wait(timeout=self.POLL_INTERVAL_S if not first else 0.05):
            first = False
            listing = list_devices()
            with self._lock:
                self._update(listing)

    def _update(self, listing: DeviceListing) -> None:
        fp = listing.fingerprint()
        if self._last_fingerprint is not None and fp != self._last_fingerprint:
            self._change_seq += 1
            self._last_change_ts = time.time()
            logger.info(
                "audio device set changed (seq=%d): %d inputs, %d outputs",
                self._change_seq,
                len(listing.inputs),
                len(listing.outputs),
            )
        self._last_fingerprint = fp
        self._last_listing = listing


# Module-level singleton.
_watcher: DeviceWatcher | None = None
_watcher_lock = threading.Lock()


def get_watcher() -> DeviceWatcher:
    global _watcher
    with _watcher_lock:
        if _watcher is None:
            _watcher = DeviceWatcher()
            _watcher.start()
        return _watcher


def reset_watcher_for_tests() -> None:
    global _watcher
    with _watcher_lock:
        if _watcher is not None:
            try:
                _watcher.stop()
            except Exception:
                pass
        _watcher = None
