"""Phase 9.1 — operator FastAPI control plane tests."""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def _reset_runner():
    """Make sure each test gets a fresh PipelineRunner singleton."""
    from operator_app import pipeline_manager

    pipeline_manager.reset_runner_for_tests()
    yield
    pipeline_manager.reset_runner_for_tests()


@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    from operator_app.main import app

    return TestClient(app)


# -- preflight ---------------------------------------------------------------


class TestPreflight:
    def test_run_all_returns_structured_payload(self, tmp_path):
        from operator_app.preflight import run_all_checks

        payload = run_all_checks(project_root=tmp_path, llamacpp_url="http://127.0.0.1:65535")
        assert "checks" in payload
        assert "ok" in payload
        assert "status_counts" in payload
        for c in payload["checks"]:
            assert c["status"] in ("pass", "warn", "fail")
            assert isinstance(c["name"], str)
            assert isinstance(c["detail"], str)

    def test_check_models_warns_when_missing(self, tmp_path):
        from operator_app.preflight import check_models

        result = check_models(tmp_path)
        assert result["status"] == "warn"
        assert "No Gemma 4 GGUFs" in result["detail"]

    def test_check_models_passes_when_both_present(self, tmp_path):
        from operator_app.preflight import check_models

        models = tmp_path / "models"
        models.mkdir()
        (models / "gemma-4-e2b-it-q4km.gguf").touch()
        (models / "gemma-4-e4b-it-q4km.gguf").touch()
        result = check_models(tmp_path)
        assert result["status"] == "pass"

    def test_check_adapter_manifest_fails_on_invalid_json(self, tmp_path):
        from operator_app.preflight import check_adapter_manifest

        (tmp_path / "adapters").mkdir()
        (tmp_path / "adapters" / "manifest.json").write_text("{not json")
        result = check_adapter_manifest(tmp_path)
        assert result["status"] == "fail"

    def test_check_llamacpp_warns_when_unreachable(self):
        from operator_app.preflight import check_llamacpp_server

        # Use port 65535 which is almost never bound
        result = check_llamacpp_server("http://127.0.0.1:65535")
        assert result["status"] == "warn"
        assert "Not reachable" in result["detail"]


# -- FastAPI endpoints --------------------------------------------------------


class TestHealthz:
    def test_healthz_returns_200(self, client):
        resp = client.get("/healthz")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"


class TestPreflightEndpoint:
    def test_returns_payload(self, client):
        resp = client.get("/api/preflight")
        assert resp.status_code == 200
        body = resp.json()
        assert "checks" in body
        assert "ok" in body


class TestSessionStatus:
    def test_idle_initial_state(self, client):
        resp = client.get("/api/session/status")
        assert resp.status_code == 200
        body = resp.json()
        assert body["state"] == "idle"


class TestSessionStart:
    def test_start_then_status_running(self, client):
        resp = client.post(
            "/api/session/start",
            json={"lang": "en", "backend": "auto", "engine": "auto"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["state"] in ("starting", "running")
        # Give the thread a moment to flip to "running" via the placeholder loop
        for _ in range(20):
            status = client.get("/api/session/status").json()
            if status["state"] == "running":
                break
            time.sleep(0.05)
        assert status["state"] == "running"

    def test_double_start_returns_409(self, client):
        first = client.post("/api/session/start", json={"lang": "en"})
        assert first.status_code == 200
        second = client.post("/api/session/start", json={"lang": "es"})
        assert second.status_code == 409
        assert "already running" in second.json()["detail"].lower()

    def test_invalid_lang_rejected(self, client):
        resp = client.post("/api/session/start", json={"lang": "fr"})
        assert resp.status_code == 422  # pydantic validation error

    def test_invalid_engine_rejected(self, client):
        resp = client.post("/api/session/start", json={"lang": "en", "engine": "magic"})
        assert resp.status_code == 422


class TestSessionStop:
    def test_stop_when_idle_is_idempotent(self, client):
        resp = client.post("/api/session/stop")
        assert resp.status_code == 200
        assert resp.json()["state"] == "idle"

    def test_start_then_stop_returns_idle(self, client):
        client.post("/api/session/start", json={"lang": "en"})
        # Wait for running state before stopping (ensures we're testing the real path)
        for _ in range(20):
            status = client.get("/api/session/status").json()
            if status["state"] == "running":
                break
            time.sleep(0.05)
        resp = client.post("/api/session/stop")
        assert resp.status_code == 200
        body = resp.json()
        assert body["state"] == "idle"
        assert body.get("stopped_at") is not None


# -- /api/devices -------------------------------------------------------------


class TestDevices:
    def test_returns_inputs_or_503(self, client):
        # sounddevice is mocked in conftest.py; the endpoint should either
        # return a payload or a 503 explaining why.
        resp = client.get("/api/devices")
        assert resp.status_code in (200, 503)
        body = resp.json()
        if resp.status_code == 200:
            assert "inputs" in body
        else:
            assert "error" in body

    def test_returns_filtered_inputs_when_sounddevice_works(self, client):
        fake_devices = [
            {"name": "Mic A", "max_input_channels": 1, "default_samplerate": 16000},
            {"name": "Speaker", "max_input_channels": 0, "default_samplerate": 48000},
            {"name": "Mic B", "max_input_channels": 2, "default_samplerate": 48000},
        ]
        with patch("sounddevice.query_devices", return_value=fake_devices):
            resp = client.get("/api/devices")
        if resp.status_code == 503:
            pytest.skip("sounddevice unavailable in CI mock")
        body = resp.json()
        names = [d["name"] for d in body["inputs"]]
        assert "Mic A" in names
        assert "Mic B" in names
        assert "Speaker" not in names
