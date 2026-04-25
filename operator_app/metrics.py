"""Live metrics collector for the operator UI (Phase 9.2).

Owns small ring buffers of per-segment latency, confidence, and queue depth
plus periodic samples of GPU/CPU resource use. Designed for a ~1 Hz
WebSocket stream — every consumer takes ``snapshot()`` and the WS endpoint
sends the JSON shape directly.

Real pipeline event hooks (`record_segment_*`) are populated by the runner
in 9.3. Until then the metrics stay empty and the sparklines are flat —
that's intentional, the infrastructure ships separately from the data
plumbing.
"""

from __future__ import annotations

import collections
import logging
import statistics
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class _Sample:
    ts: float  # epoch seconds
    value: float


@dataclass
class _Segment:
    """One pipeline segment — STT + translate + display."""

    chunk_id: int
    stt_ms: float
    translate_ms: float
    total_ms: float
    confidence: float
    text_len: int


class MetricsCollector:
    """Thread-safe metrics aggregator with bounded ring buffers."""

    # Resource samples are taken every ``RESOURCE_INTERVAL_S`` seconds.
    RESOURCE_INTERVAL_S = 1.0

    # Ring buffer cap — 5 minutes at 1 Hz = 300 samples; 60 segments at
    # ~one-per-2s = 2 minutes. Both are tiny in memory.
    RESOURCE_BUFFER = 300
    SEGMENT_BUFFER = 60

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._vram_samples: collections.deque[_Sample] = collections.deque(maxlen=self.RESOURCE_BUFFER)
        self._cpu_samples: collections.deque[_Sample] = collections.deque(maxlen=self.RESOURCE_BUFFER)
        self._segments: collections.deque[_Segment] = collections.deque(maxlen=self.SEGMENT_BUFFER)
        self._queue_depth = 0  # current pending inference jobs
        self._error_count = 0
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._started_at: float | None = None

    # -- lifecycle ------------------------------------------------------------

    def start(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._started_at = time.time()
            self._thread = threading.Thread(target=self._sampler_loop, name="metrics-sampler", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    # -- ingest hooks (called by pipeline) ------------------------------------

    def record_segment(
        self,
        *,
        chunk_id: int,
        stt_ms: float,
        translate_ms: float,
        total_ms: float,
        confidence: float,
        text_len: int = 0,
    ) -> None:
        with self._lock:
            self._segments.append(
                _Segment(
                    chunk_id=chunk_id,
                    stt_ms=float(stt_ms),
                    translate_ms=float(translate_ms),
                    total_ms=float(total_ms),
                    confidence=float(confidence),
                    text_len=int(text_len),
                )
            )

    def set_queue_depth(self, depth: int) -> None:
        with self._lock:
            self._queue_depth = int(depth)

    def record_error(self) -> None:
        with self._lock:
            self._error_count += 1

    # -- snapshot -------------------------------------------------------------

    def snapshot(self) -> dict:
        """Return a JSON-serializable dict suitable for WS broadcast or /healthz."""
        with self._lock:
            segs = list(self._segments)
            vram = list(self._vram_samples)
            cpu = list(self._cpu_samples)
            queue_depth = self._queue_depth
            error_count = self._error_count
            started_at = self._started_at

        # Latency aggregates from the last N segments.
        if segs:
            totals = [s.total_ms for s in segs]
            stt = [s.stt_ms for s in segs]
            translate = [s.translate_ms for s in segs]
            confidences = [s.confidence for s in segs]
            latency = {
                "n": len(segs),
                "total_ms_p50": round(statistics.median(totals), 1),
                "total_ms_p95": round(_p95(totals), 1),
                "stt_ms_p50": round(statistics.median(stt), 1),
                "translate_ms_p50": round(statistics.median(translate), 1),
                "confidence_mean": round(statistics.mean(confidences), 3),
            }
        else:
            latency = {"n": 0}

        return {
            "ts": time.time(),
            "uptime_s": round(time.time() - started_at, 1) if started_at else 0.0,
            "queue_depth": queue_depth,
            "error_count": error_count,
            "latency": latency,
            "resources": {
                "vram_mib_recent": _recent_values(vram, 30),
                "cpu_percent_recent": _recent_values(cpu, 30),
                "vram_mib_current": vram[-1].value if vram else 0.0,
                "cpu_percent_current": cpu[-1].value if cpu else 0.0,
            },
            "segments_recent": [
                {
                    "chunk_id": s.chunk_id,
                    "total_ms": s.total_ms,
                    "confidence": s.confidence,
                }
                for s in list(segs)[-20:]
            ],
        }

    # -- internals ------------------------------------------------------------

    def _sampler_loop(self) -> None:
        while not self._stop_event.wait(timeout=self.RESOURCE_INTERVAL_S):
            now = time.time()
            vram = _read_vram_mib()
            cpu = _read_cpu_percent()
            with self._lock:
                if vram is not None:
                    self._vram_samples.append(_Sample(ts=now, value=vram))
                if cpu is not None:
                    self._cpu_samples.append(_Sample(ts=now, value=cpu))


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = max(0, int(len(s) * 0.95) - 1)
    return s[idx]


def _recent_values(samples: list[_Sample], n: int) -> list[float]:
    return [s.value for s in samples[-n:]]


def _read_vram_mib() -> float | None:
    """Total VRAM used on GPU 0 via nvidia-smi, or None if unavailable."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", "0"],
            stderr=subprocess.DEVNULL,
            timeout=2,
        )
        return float(out.decode().strip().splitlines()[0])
    except (FileNotFoundError, subprocess.SubprocessError, ValueError):
        return None


def _read_cpu_percent() -> float | None:
    """Whole-system CPU percent over the last RESOURCE_INTERVAL_S, or None."""
    try:
        import psutil

        # interval=None returns the value since the previous call without
        # blocking — perfect for our sampler loop after the first iteration.
        return float(psutil.cpu_percent(interval=None))
    except Exception:
        return None


# Module-level singleton — same pattern as PipelineRunner.
_collector: MetricsCollector | None = None
_collector_lock = threading.Lock()


def get_collector() -> MetricsCollector:
    global _collector
    with _collector_lock:
        if _collector is None:
            _collector = MetricsCollector()
            _collector.start()
        return _collector


def reset_collector_for_tests() -> None:
    global _collector
    with _collector_lock:
        if _collector is not None:
            try:
                _collector.stop()
            except Exception:
                pass
        _collector = None


def healthz_snapshot() -> dict:
    """Used by /healthz endpoint — small subset, never blocks."""
    snap = get_collector().snapshot()
    return {
        "status": "ok",
        "service": "stark-translate-operator",
        "uptime_s": snap["uptime_s"],
        "queue_depth": snap["queue_depth"],
        "error_count": snap["error_count"],
        "vram_mib": snap["resources"]["vram_mib_current"],
        "cpu_percent": snap["resources"]["cpu_percent_current"],
        "ts": datetime.fromtimestamp(snap["ts"]).isoformat(timespec="seconds"),
    }
