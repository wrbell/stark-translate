"""Audio bridge sender (Phase 9.4.2).

Captures host microphone audio via ``sounddevice`` and streams it to the
operator's ``/ws/audio/ingest`` endpoint as 16 kHz mono 16-bit PCM frames.

Use this when the operator container can't reach the host mic directly —
e.g. a Wayland-only host with no PulseAudio socket, WSL2 without WSLg
audio, or a remote head where the operator runs on a different machine
than the audio source.

CLI:

    python -m tools.audio_bridge --operator http://operator:9000
    python -m tools.audio_bridge --device 3 --gain 1.5 --frame-ms 20
    python -m tools.audio_bridge --test            # log frame counts only

Inside the audio-bridge Docker service, this is launched by
``docker/entrypoint.sh`` with the same env vars the compose file sets.

Wire protocol (matches operator_app.audio_ingest)
-------------------------------------------------

1. Connect WS to ``ws://operator:9000/ws/audio/ingest``
2. Send a JSON text hello:
     {"version": 1, "sample_rate": 16000, "channels": 1, "format": "pcm_s16le",
      "frame_ms": 20, "client": "stark-audio-bridge/<ver>"}
3. Receive ack: ``{"ok": true, "expected_frame_bytes": 640}``
4. Stream binary PCM frames at the negotiated cadence
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import queue
import signal
import sys
import threading
import time
from urllib.parse import urlparse, urlunparse

logger = logging.getLogger(__name__)


PROTOCOL_VERSION = 1
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_CHANNELS = 1
DEFAULT_FRAME_MS = 20
DEFAULT_CLIENT_UA = "stark-audio-bridge/2026.7"


def _operator_url_to_ws(url: str) -> str:
    """``http://operator:9000`` → ``ws://operator:9000/ws/audio/ingest``."""
    parsed = urlparse(url)
    scheme = "wss" if parsed.scheme == "https" else "ws"
    return urlunparse((scheme, parsed.netloc, "/ws/audio/ingest", "", "", ""))


def _resample_to_16k(pcm_float32, src_sr: int):
    """Decimate or resample a float32 buffer to 16 kHz. Lazy-imports scipy."""
    import numpy as np

    if src_sr == DEFAULT_SAMPLE_RATE:
        return pcm_float32.astype(np.float32)
    if src_sr % DEFAULT_SAMPLE_RATE == 0:
        from scipy.signal import decimate

        factor = src_sr // DEFAULT_SAMPLE_RATE
        return decimate(pcm_float32, factor, zero_phase=False).astype(np.float32)
    from scipy.signal import resample

    target_len = int(len(pcm_float32) * DEFAULT_SAMPLE_RATE / src_sr)
    return resample(pcm_float32, target_len).astype(np.float32)


def _float_to_int16(pcm_float32) -> bytes:
    import numpy as np

    clipped = np.clip(pcm_float32, -1.0, 1.0)
    return (clipped * 32767.0).astype("<i2").tobytes()


# ---------------------------------------------------------------------------
# Bridge
# ---------------------------------------------------------------------------


class AudioBridge:
    """sounddevice → WS pipe.

    Capture and send run on different threads so a brief network stall
    can't drop mic frames; the queue caps at 0.5 s and overflowing drops
    oldest frames so we stay close to real-time.
    """

    QUEUE_CAPACITY_FRAMES = 25  # 25 * 20 ms = 500 ms max network slip

    def __init__(
        self,
        operator_url: str,
        *,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        frame_ms: int = DEFAULT_FRAME_MS,
        device: int | None = None,
        gain: float = 1.0,
        client_ua: str = DEFAULT_CLIENT_UA,
    ) -> None:
        self.ws_url = _operator_url_to_ws(operator_url)
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.device = device
        self.gain = gain
        self.client_ua = client_ua

        self._stop = threading.Event()
        self._frames_q: queue.Queue[bytes] = queue.Queue(maxsize=self.QUEUE_CAPACITY_FRAMES)
        self._dropped = 0

    # -- capture side ---------------------------------------------------------

    def _capture_loop(self) -> None:
        """sounddevice InputStream → resample → int16 → enqueue."""
        import sounddevice as sd

        # Use the device's native sample rate, then resample. Avoids letting
        # sounddevice choose a downsample we may not want.
        try:
            info = sd.query_devices(self.device, "input")
            native_sr = int(info.get("default_samplerate") or 48000)
        except Exception as exc:
            logger.warning("could not query device, falling back to 48 kHz: %s", exc)
            native_sr = 48000

        block_frames = int(native_sr * self.frame_ms / 1000)
        logger.info(
            "audio-bridge: capturing device=%s native_sr=%d frame_ms=%d block_frames=%d",
            self.device,
            native_sr,
            self.frame_ms,
            block_frames,
        )

        def callback(indata, frames, _time_info, status):
            import numpy as np

            if status:
                logger.debug("sounddevice status: %s", status)
            mono = indata[:, 0] if indata.ndim > 1 else indata
            mono = np.clip(mono * self.gain, -1.0, 1.0)
            resampled = _resample_to_16k(mono, native_sr)
            payload = _float_to_int16(resampled)
            try:
                self._frames_q.put_nowait(payload)
            except queue.Full:
                # Drop oldest, enqueue newest. Keeps latency bounded under
                # network back-pressure.
                try:
                    self._frames_q.get_nowait()
                except queue.Empty:
                    pass
                self._frames_q.put_nowait(payload)
                self._dropped += 1

        with sd.InputStream(
            samplerate=native_sr,
            channels=1,
            dtype="float32",
            blocksize=block_frames,
            callback=callback,
            device=self.device,
        ):
            while not self._stop.is_set():
                time.sleep(0.05)

    # -- send side ------------------------------------------------------------

    def _send_loop(self) -> None:
        """WS connect → handshake → drain frames queue → send."""
        from websockets.exceptions import ConnectionClosed
        from websockets.sync.client import connect

        backoff_s = 0.5
        sent_total = 0
        while not self._stop.is_set():
            try:
                logger.info("audio-bridge: connecting to %s", self.ws_url)
                with connect(self.ws_url, max_size=2**20) as ws:
                    ws.send(
                        json.dumps(
                            {
                                "version": PROTOCOL_VERSION,
                                "sample_rate": DEFAULT_SAMPLE_RATE,
                                "channels": DEFAULT_CHANNELS,
                                "format": "pcm_s16le",
                                "frame_ms": self.frame_ms,
                                "client": self.client_ua,
                            }
                        )
                    )
                    ack = json.loads(ws.recv(timeout=5.0))
                    if not ack.get("ok"):
                        logger.error("audio-bridge: server rejected hello: %s", ack)
                        return
                    logger.info("audio-bridge: handshake ok (%s)", ack)
                    backoff_s = 0.5  # reset on successful connect

                    while not self._stop.is_set():
                        try:
                            frame = self._frames_q.get(timeout=0.5)
                        except queue.Empty:
                            continue
                        ws.send(frame)
                        sent_total += 1
                        if sent_total % 50 == 0:
                            logger.info(
                                "audio-bridge: sent=%d dropped=%d buffered=%d",
                                sent_total,
                                self._dropped,
                                self._frames_q.qsize(),
                            )
            except ConnectionClosed as exc:
                logger.warning("audio-bridge: ws closed: %s — reconnecting in %.1fs", exc, backoff_s)
            except Exception as exc:
                logger.warning("audio-bridge: ws error %s — reconnecting in %.1fs", exc, backoff_s)
            if not self._stop.is_set():
                self._stop.wait(timeout=backoff_s)
                backoff_s = min(backoff_s * 2, 10.0)

    # -- public ---------------------------------------------------------------

    def run(self) -> int:
        capture = threading.Thread(target=self._capture_loop, name="audio-bridge-capture", daemon=True)
        send = threading.Thread(target=self._send_loop, name="audio-bridge-send", daemon=True)
        capture.start()
        send.start()

        def _sigterm(_sig, _frame):
            logger.info("audio-bridge: received SIGTERM, shutting down")
            self._stop.set()

        signal.signal(signal.SIGTERM, _sigterm)
        signal.signal(signal.SIGINT, _sigterm)

        while not self._stop.is_set():
            time.sleep(0.5)

        # Capture/send threads are daemons; let them die with the process
        # rather than gating shutdown on a clean disconnect.
        return 0


def run_test_mode(device: int | None) -> int:
    """Print frame counts only — no network. Useful for verifying the mic."""
    import sounddevice as sd

    counter = {"frames": 0, "rms": 0.0}

    def callback(indata, frames, _time_info, status):
        import numpy as np

        if status:
            print(f"  sounddevice status: {status}", file=sys.stderr)
        counter["frames"] += 1
        counter["rms"] = float(np.sqrt(np.mean(indata**2)))

    info = sd.query_devices(device, "input") if device is not None else sd.query_devices(kind="input")
    native_sr = int(info.get("default_samplerate") or 48000)
    print(f"audio-bridge --test: device={info.get('name')!r} sr={native_sr}")
    with sd.InputStream(
        samplerate=native_sr,
        channels=1,
        dtype="float32",
        blocksize=int(native_sr * 0.5),
        callback=callback,
        device=device,
    ):
        try:
            while True:
                time.sleep(1.0)
                print(f"  frames={counter['frames']} last_rms={counter['rms']:.4f}")
        except KeyboardInterrupt:
            return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    p = argparse.ArgumentParser(prog="audio-bridge", description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--operator",
        default=os.environ.get("STARK_OPERATOR_URL", "http://operator:9000"),
        help="Operator base URL (env: STARK_OPERATOR_URL). Default: http://operator:9000",
    )
    p.add_argument(
        "--device",
        type=int,
        default=int(os.environ["STARK_MIC_DEVICE"]) if os.environ.get("STARK_MIC_DEVICE") else None,
        help="sounddevice input index. Default: system default mic",
    )
    p.add_argument("--gain", type=float, default=1.0, help="Mic gain multiplier (default 1.0)")
    p.add_argument("--frame-ms", type=int, default=DEFAULT_FRAME_MS, help="Frame size in ms (default 20)")
    p.add_argument("--test", action="store_true", help="Capture-only test mode — no network")
    args = p.parse_args(argv)

    if args.test:
        return run_test_mode(device=args.device)

    bridge = AudioBridge(
        operator_url=args.operator,
        device=args.device,
        gain=args.gain,
        frame_ms=args.frame_ms,
    )
    return bridge.run()


if __name__ == "__main__":
    sys.exit(main())
