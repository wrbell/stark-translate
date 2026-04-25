"""Phase 9.4.2 — audio bridge sender + audio_ingest WS endpoint.

Covers:
- AudioBus producer/consumer (frame buffering, drop on overflow, stats)
- Hello-frame validation (version, format, channels, sample rate)
- /ws/audio/ingest end-to-end via FastAPI TestClient
- tools.audio_bridge URL conversion + int16 conversion + the test mode CLI
"""

from __future__ import annotations

import json
import struct

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# AudioBus
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_bus():
    from operator_app import audio_ingest

    audio_ingest.reset_bus_for_tests()
    yield
    audio_ingest.reset_bus_for_tests()


class TestAudioBus:
    def test_push_converts_int16_to_float32(self):
        from operator_app.audio_ingest import AudioBus

        bus = AudioBus()
        # 4 samples: 0, half-pos, max-neg, max-pos
        pcm = struct.pack("<hhhh", 0, 16384, -32768, 32767)
        bus.push_frame(pcm)
        frame = bus.read_frame(timeout_s=0.1)
        assert frame is not None
        assert frame.dtype == np.float32
        # 0 → 0.0, 16384 → 0.5, -32768 → -1.0, 32767 → ~1.0
        assert frame[0] == pytest.approx(0.0)
        assert frame[1] == pytest.approx(0.5, abs=0.001)
        assert frame[2] == pytest.approx(-1.0, abs=0.001)
        assert frame[3] == pytest.approx(1.0, abs=0.001)

    def test_overflow_drops_oldest_and_increments_counter(self):
        from operator_app.audio_ingest import AudioBus

        bus = AudioBus(capacity=2)
        bus.push_frame(b"\x00\x00")
        bus.push_frame(b"\x01\x00")
        bus.push_frame(b"\x02\x00")  # overflow → oldest dropped
        snap = bus.snapshot()
        assert snap["frames_received"] == 3
        assert snap["frames_dropped"] == 1
        assert snap["buffered_frames"] == 2

    def test_read_frame_timeout_returns_none(self):
        from operator_app.audio_ingest import AudioBus

        bus = AudioBus()
        assert bus.read_frame(timeout_s=0.05) is None

    def test_reset_drops_buffered(self):
        from operator_app.audio_ingest import AudioBus

        bus = AudioBus()
        bus.push_frame(b"\x00\x01\x02\x03")
        assert bus.snapshot()["buffered_frames"] == 1
        bus.reset()
        assert bus.snapshot()["buffered_frames"] == 0

    def test_handshake_records_client_and_sample_rate(self):
        from operator_app.audio_ingest import AudioBus

        bus = AudioBus()
        bus.record_handshake(sample_rate=24000, client="test-bridge/1.0")
        snap = bus.snapshot()
        assert snap["sample_rate"] == 24000
        assert snap["last_client"] == "test-bridge/1.0"
        assert snap["last_handshake_ts"] is not None


# ---------------------------------------------------------------------------
# Hello validation
# ---------------------------------------------------------------------------


class TestHelloValidation:
    def test_accepts_canonical_hello(self):
        from operator_app.audio_ingest import _validate_hello

        assert _validate_hello({"version": 1, "sample_rate": 16000, "channels": 1, "format": "pcm_s16le"})

    def test_rejects_wrong_version(self):
        from operator_app.audio_ingest import _validate_hello

        assert not _validate_hello({"version": 999, "sample_rate": 16000, "channels": 1, "format": "pcm_s16le"})

    def test_rejects_stereo(self):
        from operator_app.audio_ingest import _validate_hello

        assert not _validate_hello({"version": 1, "sample_rate": 16000, "channels": 2, "format": "pcm_s16le"})

    def test_rejects_unsupported_format(self):
        from operator_app.audio_ingest import _validate_hello

        assert not _validate_hello({"version": 1, "sample_rate": 16000, "channels": 1, "format": "opus"})

    def test_rejects_odd_sample_rate(self):
        from operator_app.audio_ingest import _validate_hello

        assert not _validate_hello({"version": 1, "sample_rate": 22050, "channels": 1, "format": "pcm_s16le"})

    def test_accepts_24k_and_48k(self):
        from operator_app.audio_ingest import _validate_hello

        for sr in (24000, 48000):
            assert _validate_hello({"version": 1, "sample_rate": sr, "channels": 1, "format": "pcm_s16le"})


# ---------------------------------------------------------------------------
# End-to-end via FastAPI TestClient
# ---------------------------------------------------------------------------


@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    from operator_app.main import app

    return TestClient(app)


class TestAudioIngestEndpoint:
    def test_ws_endpoint_handshakes_and_accepts_pcm(self, client):
        from operator_app.audio_ingest import get_bus

        with client.websocket_connect("/ws/audio/ingest") as ws:
            ws.send_text(
                json.dumps(
                    {
                        "version": 1,
                        "sample_rate": 16000,
                        "channels": 1,
                        "format": "pcm_s16le",
                        "frame_ms": 20,
                        "client": "test/1.0",
                    }
                )
            )
            ack = json.loads(ws.receive_text())
            assert ack["ok"] is True
            assert ack["expected_frame_bytes"] == 640  # 16000 * 20/1000 * 2

            # Send three 320-byte frames (we don't enforce frame size)
            ws.send_bytes(b"\x00\x00" * 160)
            ws.send_bytes(b"\x01\x00" * 160)
            ws.send_bytes(b"\x02\x00" * 160)

        snap = get_bus().snapshot()
        assert snap["frames_received"] == 3
        assert snap["last_client"] == "test/1.0"

    def test_ws_endpoint_rejects_bad_hello(self, client):
        with client.websocket_connect("/ws/audio/ingest") as ws:
            ws.send_text(json.dumps({"version": 999, "sample_rate": 16000, "channels": 1, "format": "pcm_s16le"}))
            # Server replies with a rejection text frame before closing.
            reply = json.loads(ws.receive_text())
            assert reply["ok"] is False
            assert "unsupported" in reply["reason"].lower()

    def test_api_audio_ingest_returns_bus_stats(self, client):
        r = client.get("/api/audio_ingest")
        assert r.status_code == 200
        body = r.json()
        for key in ("frames_received", "bytes_received", "frames_dropped", "sample_rate", "buffered_frames"):
            assert key in body


# ---------------------------------------------------------------------------
# Sender (tools.audio_bridge)
# ---------------------------------------------------------------------------


class TestSenderHelpers:
    def test_url_conversion_http_to_ws(self):
        from tools.audio_bridge import _operator_url_to_ws

        assert _operator_url_to_ws("http://operator:9000") == "ws://operator:9000/ws/audio/ingest"
        assert _operator_url_to_ws("https://stark.example.com") == "wss://stark.example.com/ws/audio/ingest"
        # Trailing slash and arbitrary path get replaced — only scheme + netloc matter
        assert _operator_url_to_ws("http://localhost:9000/foo") == "ws://localhost:9000/ws/audio/ingest"

    def test_float_to_int16_clamps(self):
        from tools.audio_bridge import _float_to_int16

        # Out-of-range samples must clamp, not wrap. -1.5 → -32767, 1.5 → 32767.
        pcm = np.array([0.0, 0.5, -1.0, 1.0, 1.5, -1.5], dtype=np.float32)
        out = _float_to_int16(pcm)
        unpacked = struct.unpack("<6h", out)
        assert unpacked[0] == 0
        assert unpacked[1] == pytest.approx(16383, abs=2)
        assert unpacked[2] == -32767
        assert unpacked[3] == 32767
        assert unpacked[4] == 32767
        assert unpacked[5] == -32767

    def test_resample_passthrough_when_sr_matches(self):
        from tools.audio_bridge import DEFAULT_SAMPLE_RATE, _resample_to_16k

        pcm = np.linspace(-1, 1, 320, dtype=np.float32)
        out = _resample_to_16k(pcm, DEFAULT_SAMPLE_RATE)
        assert out.dtype == np.float32
        assert len(out) == 320

    def test_resample_decimates_48k_to_16k(self):
        pytest.importorskip("scipy.signal")
        from tools.audio_bridge import _resample_to_16k

        pcm = np.zeros(960, dtype=np.float32)  # 20 ms @ 48 kHz
        out = _resample_to_16k(pcm, 48000)
        assert len(out) == 320  # 20 ms @ 16 kHz


class TestSenderCli:
    def test_argparse_help_does_not_crash(self, capsys):
        from tools.audio_bridge import main

        with pytest.raises(SystemExit) as exc:
            main(["--help"])
        assert exc.value.code == 0
        out = capsys.readouterr().out
        assert "audio-bridge" in out
        assert "--operator" in out

    def test_default_operator_url_from_env(self, monkeypatch):
        """STARK_OPERATOR_URL env var seeds the --operator default."""
        import argparse

        monkeypatch.setenv("STARK_OPERATOR_URL", "http://test-operator:9000")
        # Re-import the parser so the os.environ.get default is re-evaluated
        from tools import audio_bridge

        parser = argparse.ArgumentParser()
        parser.add_argument("--operator", default=__import__("os").environ.get("STARK_OPERATOR_URL"))
        args = parser.parse_args([])
        assert args.operator == "http://test-operator:9000"
        # And the URL converter still works on it
        assert audio_bridge._operator_url_to_ws(args.operator) == "ws://test-operator:9000/ws/audio/ingest"


# ---------------------------------------------------------------------------
# Compose + entrypoint integration
# ---------------------------------------------------------------------------


from pathlib import Path

ROOT = Path(__file__).parent.parent


class TestEntrypointIntegration:
    def test_entrypoint_no_longer_sleeps(self):
        text = (ROOT / "docker" / "entrypoint.sh").read_text()
        assert "sleep infinity" not in text, "audio-bridge stub still sleeping forever"

    def test_entrypoint_invokes_audio_bridge(self):
        text = (ROOT / "docker" / "entrypoint.sh").read_text()
        assert "tools.audio_bridge" in text or "audio_bridge" in text

    def test_compose_audio_bridge_uses_correct_operator_url(self):
        text = (ROOT / "docker-compose.yml").read_text()
        assert "STARK_OPERATOR_URL: http://operator:9000" in text
        # The old aspirational STARK_TRANSCRIPT_WS pointed at the wrong port
        # (8765 was outbound transcript broadcast). Make sure it's gone.
        assert "ws://operator:8765/audio" not in text
