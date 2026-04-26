"""Client-side audio-bridge adapter for the pipeline subprocess (Phase 9.4.2).

When ``STARK_AUDIO_SOURCE=ws`` is set, ``dry_run_ab.py`` swaps its
``sd.InputStream(...)`` for a ``WebsocketAudioStream`` instance from this
module. The adapter mimics the ``sd.InputStream`` context manager interface
so the rest of the audio loop is unchanged.

Frame flow:

    audio-bridge container
      tools.audio_bridge.AudioBridge
        sounddevice → resample 16 kHz → int16 → WS client →
          ws://operator:9000/ws/audio/ingest
                                    │
                                    ▼
        operator_app.audio_ingest.AudioBus  (push_frame fan-out)
                                    │
                                    ▼
              ws://operator:9000/ws/audio/subscribe
                                    │
                                    ▼
    pipeline subprocess (this module)
      WebsocketAudioStream — reads frames, calls user's callback
        callback pushes float32 frames into audio_queue (same as sd path)
"""

from __future__ import annotations

import json
import logging
import os
import threading
from collections.abc import Callable
from urllib.parse import urlparse, urlunparse

import numpy as np

logger = logging.getLogger(__name__)


def _operator_url_to_subscribe_ws(url: str) -> str:
    """``http://operator:9000`` → ``ws://operator:9000/ws/audio/subscribe``."""
    parsed = urlparse(url)
    scheme = "wss" if parsed.scheme == "https" else "ws"
    return urlunparse((scheme, parsed.netloc, "/ws/audio/subscribe", "", "", ""))


class WebsocketAudioStream:
    """Mimics ``sd.InputStream`` for the WS-source audio path.

    Public surface matches what dry_run_ab.py uses on ``sd.InputStream``:

    - ``with stream:`` enters/exits cleanly (context manager)
    - the constructor takes a ``callback`` that receives ``(indata, frames,
      time_info, status)`` — same signature as sounddevice. Frames are
      np.float32 mono [-1, 1], shape ``(N, 1)``.

    Internally a daemon thread holds the WS connection, decodes each binary
    frame to float32, and invokes the callback synchronously. Reconnects
    with exponential backoff up to 10 s on transient errors.
    """

    def __init__(
        self,
        url: str,
        *,
        callback: Callable | None = None,
        samplerate: int = 16000,
        channels: int = 1,
        dtype: str = "float32",
        blocksize: int | None = None,
        device: int | None = None,
    ) -> None:
        # We accept and ignore the sounddevice-style kwargs (samplerate,
        # blocksize, device) so callers can swap the stream constructors
        # without code duplication. The bridge has already resampled to
        # 16 kHz mono int16 before getting here.
        del samplerate, channels, dtype, blocksize, device

        self.url = _operator_url_to_subscribe_ws(url)
        self._callback = callback
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._frames_received = 0
        self._frames_dropped = 0  # callback raised; we swallow + count

    # -- context manager ------------------------------------------------------

    def __enter__(self) -> WebsocketAudioStream:
        if self._callback is None:
            raise RuntimeError("WebsocketAudioStream requires a callback")
        self._stop.clear()
        self._thread = threading.Thread(target=self._reader_loop, name="ws-audio-reader", daemon=True)
        self._thread.start()
        logger.info("ws-audio: connected to %s", self.url)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        logger.info(
            "ws-audio: closed (frames=%d dropped=%d)",
            self._frames_received,
            self._frames_dropped,
        )

    # -- reader thread --------------------------------------------------------

    def _reader_loop(self) -> None:
        from websockets.exceptions import ConnectionClosed
        from websockets.sync.client import connect

        backoff_s = 0.5
        while not self._stop.is_set():
            try:
                with connect(self.url, max_size=2**20) as ws:
                    backoff_s = 0.5  # reset on a clean connect
                    while not self._stop.is_set():
                        try:
                            msg = ws.recv(timeout=5.0)
                        except TimeoutError:
                            # Idle — server will send keepalive JSON every 30 s
                            continue
                        if isinstance(msg, str):
                            # Server keepalive — ignore
                            try:
                                obj = json.loads(msg)
                                if obj.get("keepalive"):
                                    continue
                            except json.JSONDecodeError:
                                pass
                            continue
                        # Binary PCM frame: int16 LE → float32 [-1, 1]
                        samples = np.frombuffer(msg, dtype=np.int16).astype(np.float32) / 32768.0
                        # Sounddevice callback signature: (indata, frames, time_info, status).
                        # sounddevice gives indata as 2D (N, channels); we match that shape.
                        indata = samples.reshape(-1, 1)
                        try:
                            self._callback(indata, len(samples), None, None)
                            self._frames_received += 1
                        except Exception as exc:
                            self._frames_dropped += 1
                            logger.debug("ws-audio: callback raised: %s", exc)
            except ConnectionClosed as exc:
                logger.warning("ws-audio: ws closed: %s — reconnecting in %.1fs", exc, backoff_s)
            except Exception as exc:
                logger.warning("ws-audio: ws error %s — reconnecting in %.1fs", exc, backoff_s)

            if not self._stop.is_set():
                self._stop.wait(timeout=backoff_s)
                backoff_s = min(backoff_s * 2, 10.0)


def open_audio_stream(
    callback: Callable,
    *,
    samplerate: int,
    channels: int,
    dtype: str,
    blocksize: int,
    device: int | None,
):
    """Factory: return ``sd.InputStream`` or ``WebsocketAudioStream`` per env.

    When ``STARK_AUDIO_SOURCE=ws`` is set, returns a ``WebsocketAudioStream``
    pointed at ``STARK_OPERATOR_URL`` (default ``http://localhost:9000``).
    Otherwise opens a normal sounddevice ``InputStream`` with the given args
    — drop-in-compatible swap point for the pipeline.
    """
    if os.environ.get("STARK_AUDIO_SOURCE") == "ws":
        url = os.environ.get("STARK_OPERATOR_URL", "http://localhost:9000")
        logger.info("STARK_AUDIO_SOURCE=ws — opening WebsocketAudioStream at %s", url)
        return WebsocketAudioStream(
            url=url,
            callback=callback,
            samplerate=samplerate,
            channels=channels,
            dtype=dtype,
            blocksize=blocksize,
            device=device,
        )
    import sounddevice as sd

    return sd.InputStream(
        samplerate=samplerate,
        channels=channels,
        dtype=dtype,
        blocksize=blocksize,
        callback=callback,
        device=device,
    )
