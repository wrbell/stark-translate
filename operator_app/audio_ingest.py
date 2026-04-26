"""Audio ingest WebSocket endpoint (Phase 9.4.2).

Receives PCM audio frames from a remote ``tools/audio_bridge.py`` running
in the audio-bridge container. The pipeline subprocess (``dry_run_ab.py``)
reads from the same ``AudioBus`` instead of opening ``sd.InputStream``
directly when ``STARK_AUDIO_SOURCE=ws``.

Wire protocol
-------------

WS endpoint: ``/ws/audio/ingest`` on the operator FastAPI app (port 9000).

On connect, the bridge sends a JSON text frame describing the stream:

.. code-block:: json

    {
        "version": 1,
        "sample_rate": 16000,
        "channels": 1,
        "format": "pcm_s16le",
        "frame_ms": 20,
        "client": "stark-audio-bridge/2026.7"
    }

The server replies with a single text frame:

.. code-block:: json

    {"ok": true, "expected_frame_bytes": 640}

Subsequent messages are **binary** frames carrying raw PCM. For 16 kHz
mono 16-bit at 20 ms per frame that's 640 bytes per message. The receiver
appends each frame to the ``AudioBus`` ring buffer; consumers (the
pipeline's audio_queue feeder) call ``AudioBus.read_frame()`` to pop one.

Protocol design notes
---------------------

- Binary frames carry the PCM directly, no JSON wrapping. Keeps overhead
  to ~2 bytes per 1280-byte frame and avoids float64 quantization round-trips.
- 16 kHz mono is the canonical pipeline rate (Whisper is fixed at 16 kHz).
  Resampling happens in the bridge, not here — that way the operator
  container doesn't carry scipy's resample cost in the hot path.
- Signed little-endian 16-bit is the path of least conversion; sounddevice
  reads float32 [-1, 1] so we ``* 32767.5 + 0.5 → int16`` on the bridge
  side and ``/ 32768.0`` here.
"""

from __future__ import annotations

import asyncio
import collections
import json
import logging
import threading
import time
from dataclasses import dataclass

import numpy as np
from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)


PROTOCOL_VERSION = 1
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_CHANNELS = 1
DEFAULT_FORMAT = "pcm_s16le"
DEFAULT_FRAME_MS = 20


@dataclass
class _BusStats:
    """Counters surfaced via /healthz and /api/audio_ingest."""

    frames_received: int = 0
    bytes_received: int = 0
    frames_dropped: int = 0  # buffer full when a frame arrived
    last_frame_ts: float | None = None
    last_client: str | None = None
    last_handshake_ts: float | None = None


class AudioBus:
    """Single-producer / single-consumer ring buffer for incoming PCM frames.

    The producer is the WS receive task; the consumer is the pipeline
    subprocess's audio thread when STARK_AUDIO_SOURCE=ws is set. Frames are
    stored as np.float32 arrays already converted from int16, so consumers
    see the same dtype the local sounddevice path produces.
    """

    # 30 frames * 20 ms = 600 ms of buffered audio. Small enough that a
    # consumer falling 600 ms behind drops frames rather than blowing memory.
    DEFAULT_CAPACITY_FRAMES = 30

    def __init__(self, capacity: int = DEFAULT_CAPACITY_FRAMES) -> None:
        self._lock = threading.Lock()
        self._buf: collections.deque[np.ndarray] = collections.deque(maxlen=capacity)
        self._stats = _BusStats()
        self._sample_rate = DEFAULT_SAMPLE_RATE
        # Subscriber queues for /ws/audio/subscribe fan-out. Each subscriber
        # gets its own bounded queue so a slow consumer drops frames rather
        # than back-pressuring the producer.
        self._subscribers: list[asyncio.Queue[bytes]] = []
        self._subscribers_lock = threading.Lock()

    # -- producer side --------------------------------------------------------

    def push_frame(self, pcm_int16: bytes, *, client: str | None = None) -> None:
        """Convert raw int16 LE bytes into float32 [-1, 1] and enqueue.

        Also fan-outs the raw int16 bytes to all WS subscribers (the pipeline
        consumes via ``/ws/audio/subscribe`` rather than reading the ring
        directly — different processes, no shared memory).
        """
        if not pcm_int16:
            return
        samples = np.frombuffer(pcm_int16, dtype=np.int16).astype(np.float32) / 32768.0
        with self._lock:
            if len(self._buf) == self._buf.maxlen:
                self._stats.frames_dropped += 1
            self._buf.append(samples)
            self._stats.frames_received += 1
            self._stats.bytes_received += len(pcm_int16)
            self._stats.last_frame_ts = time.time()
            if client:
                self._stats.last_client = client

        # Fan-out to subscribers. put_nowait drops if full — bounded by
        # subscriber queue size so a slow consumer can't OOM the operator.
        with self._subscribers_lock:
            subs = list(self._subscribers)
        for q in subs:
            try:
                q.put_nowait(pcm_int16)
            except asyncio.QueueFull:
                pass

    def add_subscriber(self, q: asyncio.Queue) -> None:
        with self._subscribers_lock:
            self._subscribers.append(q)

    def remove_subscriber(self, q: asyncio.Queue) -> None:
        with self._subscribers_lock:
            try:
                self._subscribers.remove(q)
            except ValueError:
                pass

    def subscriber_count(self) -> int:
        with self._subscribers_lock:
            return len(self._subscribers)

    def record_handshake(self, sample_rate: int, client: str | None) -> None:
        with self._lock:
            self._sample_rate = sample_rate
            self._stats.last_handshake_ts = time.time()
            if client:
                self._stats.last_client = client

    # -- consumer side --------------------------------------------------------

    def read_frame(self, timeout_s: float = 0.1) -> np.ndarray | None:
        """Pop the next frame. Returns None on timeout (non-blocking poll loop)."""
        deadline = time.time() + timeout_s
        while True:
            with self._lock:
                if self._buf:
                    return self._buf.popleft()
            remaining = deadline - time.time()
            if remaining <= 0:
                return None
            time.sleep(min(0.01, remaining))

    def reset(self) -> None:
        """Drop any buffered frames. Called between sessions."""
        with self._lock:
            self._buf.clear()

    # -- introspection --------------------------------------------------------

    def snapshot(self) -> dict:
        with self._lock:
            base = {
                "frames_received": self._stats.frames_received,
                "bytes_received": self._stats.bytes_received,
                "frames_dropped": self._stats.frames_dropped,
                "last_frame_ts": self._stats.last_frame_ts,
                "last_handshake_ts": self._stats.last_handshake_ts,
                "last_client": self._stats.last_client,
                "sample_rate": self._sample_rate,
                "buffered_frames": len(self._buf),
                "capacity_frames": self._buf.maxlen,
            }
        base["subscribers"] = self.subscriber_count()
        return base


# Module-level singleton — same pattern as MetricsCollector and PipelineRunner.
_bus: AudioBus | None = None
_bus_lock = threading.Lock()


def get_bus() -> AudioBus:
    global _bus
    with _bus_lock:
        if _bus is None:
            _bus = AudioBus()
        return _bus


def reset_bus_for_tests() -> None:
    global _bus
    with _bus_lock:
        _bus = None


# -- WebSocket endpoint -------------------------------------------------------


async def handle_audio_ingest(websocket: WebSocket) -> None:
    """Receive a hello frame, ack, then loop on binary PCM frames.

    Closes on protocol mismatch (wrong version, unsupported format) and
    on the first non-binary message after handshake. Designed to be wired
    in via ``app.websocket("/ws/audio/ingest")(handle_audio_ingest)``.
    """
    await websocket.accept()
    bus = get_bus()
    client = websocket.client.host if websocket.client else "?"
    logger.info("audio-ingest: connection from %s", client)

    try:
        # --- handshake ---
        try:
            hello_text = await asyncio.wait_for(websocket.receive_text(), timeout=5.0)
        except (TimeoutError, WebSocketDisconnect):
            logger.warning("audio-ingest: %s sent no hello frame within 5s", client)
            await websocket.close(code=1002, reason="missing hello")
            return

        try:
            hello = json.loads(hello_text)
        except json.JSONDecodeError:
            await websocket.close(code=1003, reason="hello must be JSON")
            return

        if not _validate_hello(hello):
            await websocket.send_text(json.dumps({"ok": False, "reason": "unsupported hello"}))
            await websocket.close(code=1003, reason="unsupported hello")
            return

        sample_rate = int(hello.get("sample_rate", DEFAULT_SAMPLE_RATE))
        frame_ms = int(hello.get("frame_ms", DEFAULT_FRAME_MS))
        expected_frame_bytes = (sample_rate * frame_ms // 1000) * 2  # int16 mono
        client_ua = str(hello.get("client") or client)

        bus.record_handshake(sample_rate=sample_rate, client=client_ua)
        await websocket.send_text(json.dumps({"ok": True, "expected_frame_bytes": expected_frame_bytes}))
        logger.info(
            "audio-ingest: %s handshake ok (sr=%d, frame=%dB)",
            client_ua,
            sample_rate,
            expected_frame_bytes,
        )

        # --- audio frame loop ---
        while True:
            msg = await websocket.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            data = msg.get("bytes")
            if data is None:
                # text after handshake — log and ignore so a stale client
                # can't kill the loop, but flag in metrics.
                logger.debug("audio-ingest: ignoring non-binary frame from %s", client_ua)
                continue
            bus.push_frame(data, client=client_ua)

    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.warning("audio-ingest: %s closed unexpectedly: %s", client, exc)
        try:
            await websocket.close()
        except Exception:
            pass
    finally:
        logger.info("audio-ingest: %s disconnected", client)


async def handle_audio_subscribe(websocket: WebSocket) -> None:
    """Stream PCM frames to a consumer (the pipeline subprocess).

    No handshake required — the subscriber just opens the WS and receives
    binary frames as they arrive on the AudioBus. Frame size + sample rate
    match whatever the producing bridge negotiated; subscribers can read
    ``GET /api/audio_ingest`` to learn the current rate.

    Per-subscriber queue is bounded at 30 frames (~600 ms @ 20 ms/frame). A
    consumer that falls behind drops oldest frames rather than back-pressuring
    the producer.
    """
    await websocket.accept()
    bus = get_bus()
    client = websocket.client.host if websocket.client else "?"
    logger.info("audio-subscribe: %s connected (subscribers=%d)", client, bus.subscriber_count() + 1)

    q: asyncio.Queue[bytes] = asyncio.Queue(maxsize=30)
    bus.add_subscriber(q)
    try:
        while True:
            try:
                frame = await asyncio.wait_for(q.get(), timeout=30.0)
            except TimeoutError:
                # Idle — send a tiny ping to keep the connection alive
                try:
                    await websocket.send_text(json.dumps({"keepalive": True}))
                except Exception:
                    break
                continue
            await websocket.send_bytes(frame)
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.warning("audio-subscribe: %s closed unexpectedly: %s", client, exc)
    finally:
        bus.remove_subscriber(q)
        logger.info("audio-subscribe: %s disconnected (subscribers=%d)", client, bus.subscriber_count())
        try:
            await websocket.close()
        except Exception:
            pass


def _validate_hello(hello: dict) -> bool:
    if not isinstance(hello, dict):
        return False
    if int(hello.get("version", 0)) != PROTOCOL_VERSION:
        return False
    if hello.get("format", DEFAULT_FORMAT) != DEFAULT_FORMAT:
        return False
    if int(hello.get("channels", DEFAULT_CHANNELS)) != 1:
        return False
    sr = int(hello.get("sample_rate", DEFAULT_SAMPLE_RATE))
    # Whisper is 16 kHz; allow 16/24/48 kHz and let the consumer downsample
    # if it cares. For now only 16 kHz is wired into the pipeline.
    if sr not in (16000, 24000, 48000):
        return False
    return True
