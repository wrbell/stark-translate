"""Phase 9.4 — audio device enumeration + hotplug watcher tests."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def _reset_singletons():
    from operator_app import audio, metrics, pipeline_manager

    metrics.reset_collector_for_tests()
    pipeline_manager.reset_runner_for_tests()
    audio.reset_watcher_for_tests()
    yield
    metrics.reset_collector_for_tests()
    pipeline_manager.reset_runner_for_tests()
    audio.reset_watcher_for_tests()


@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    from operator_app.main import app

    return TestClient(app)


# -- list_devices ------------------------------------------------------------


class TestListDevices:
    def test_filters_inputs_and_outputs(self):
        from operator_app.audio import list_devices

        fake = [
            {"name": "Mic A", "max_input_channels": 1, "max_output_channels": 0, "default_samplerate": 16000},
            {"name": "Mic B", "max_input_channels": 2, "max_output_channels": 0, "default_samplerate": 48000},
            {"name": "Speaker", "max_input_channels": 0, "max_output_channels": 2, "default_samplerate": 48000},
            {"name": "Combo", "max_input_channels": 1, "max_output_channels": 2, "default_samplerate": 44100},
        ]
        with patch("sounddevice.query_devices", return_value=fake):
            listing = list_devices()

        assert listing.error is None
        assert [d.name for d in listing.inputs] == ["Mic A", "Mic B", "Combo"]
        assert [d.name for d in listing.outputs] == ["Speaker", "Combo"]
        assert all(d.direction == "input" for d in listing.inputs)
        assert all(d.direction == "output" for d in listing.outputs)

    def test_returns_error_on_sounddevice_failure(self):
        from operator_app.audio import list_devices

        with patch("sounddevice.query_devices", side_effect=OSError("device busy")):
            listing = list_devices()

        assert listing.error is not None
        assert "device busy" in listing.error
        assert listing.inputs == []
        assert listing.outputs == []


# -- DeviceWatcher -----------------------------------------------------------


class TestDeviceWatcher:
    def test_change_seq_increments_on_change(self):
        from operator_app.audio import DeviceWatcher

        w = DeviceWatcher()
        snap_a = [
            {"name": "Mic", "max_input_channels": 1, "max_output_channels": 0, "default_samplerate": 16000},
        ]
        snap_b = [
            {"name": "Mic", "max_input_channels": 1, "max_output_channels": 0, "default_samplerate": 16000},
            {"name": "USB Mic", "max_input_channels": 1, "max_output_channels": 0, "default_samplerate": 48000},
        ]
        with patch("sounddevice.query_devices", return_value=snap_a):
            w.force_poll()
        assert w.snapshot()["change_seq"] == 0
        with patch("sounddevice.query_devices", return_value=snap_b):
            w.force_poll()
        assert w.snapshot()["change_seq"] == 1
        assert w.snapshot()["last_change_ts"] is not None

    def test_no_change_keeps_seq(self):
        from operator_app.audio import DeviceWatcher

        w = DeviceWatcher()
        snap = [
            {"name": "Mic", "max_input_channels": 1, "max_output_channels": 0, "default_samplerate": 16000},
        ]
        with patch("sounddevice.query_devices", return_value=snap):
            w.force_poll()
            w.force_poll()
            w.force_poll()
        assert w.snapshot()["change_seq"] == 0


# -- /api/devices ------------------------------------------------------------


class TestApiDevices:
    def test_returns_inputs_and_outputs_and_change_seq(self, client):
        fake = [
            {"name": "MicX", "max_input_channels": 2, "max_output_channels": 0, "default_samplerate": 48000},
            {"name": "Headphones", "max_input_channels": 0, "max_output_channels": 2, "default_samplerate": 48000},
        ]
        with patch("sounddevice.query_devices", return_value=fake):
            resp = client.get("/api/devices")

        assert resp.status_code == 200
        body = resp.json()
        assert "inputs" in body
        assert "outputs" in body
        assert "change_seq" in body
        assert any(d["name"] == "MicX" for d in body["inputs"])
        assert any(d["name"] == "Headphones" for d in body["outputs"])

    def test_returns_503_when_sounddevice_unavailable(self, client):
        with patch("sounddevice.query_devices", side_effect=OSError("not found")):
            resp = client.get("/api/devices")
        # 503 with the same body shape so the frontend can show the error.
        assert resp.status_code == 503
        body = resp.json()
        assert body["error"]
        assert body["inputs"] == []
        assert body["outputs"] == []


# -- metrics frame includes audio summary ------------------------------------


class TestMetricsAudioField:
    def test_snapshot_contains_audio_change_seq(self, client):
        # Force a known change_seq via watcher.
        from operator_app.audio import get_watcher

        snap_a = [
            {"name": "X", "max_input_channels": 1, "max_output_channels": 0, "default_samplerate": 16000},
        ]
        snap_b = [
            {"name": "X", "max_input_channels": 1, "max_output_channels": 0, "default_samplerate": 16000},
            {"name": "Y", "max_input_channels": 1, "max_output_channels": 0, "default_samplerate": 16000},
        ]
        watcher = get_watcher()
        with patch("sounddevice.query_devices", return_value=snap_a):
            watcher.force_poll()
        with patch("sounddevice.query_devices", return_value=snap_b):
            watcher.force_poll()

        with client.websocket_connect("/ws/control") as ws:
            frame = json.loads(ws.receive_text())
            assert "audio" in frame
            assert frame["audio"]["change_seq"] >= 1
            assert frame["audio"]["last_change_ts"] is not None
