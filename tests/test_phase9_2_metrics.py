"""Phase 9.2 — metrics collector + /ws/control + /healthz tests."""

from __future__ import annotations

import json
import time

import pytest


@pytest.fixture(autouse=True)
def _reset_collector():
    from operator_app import metrics

    metrics.reset_collector_for_tests()
    yield
    metrics.reset_collector_for_tests()


@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    from operator_app.main import app

    return TestClient(app)


# -- collector ---------------------------------------------------------------


class TestMetricsCollector:
    def test_snapshot_when_empty(self):
        from operator_app.metrics import MetricsCollector

        c = MetricsCollector()
        snap = c.snapshot()
        assert "ts" in snap
        assert snap["queue_depth"] == 0
        assert snap["error_count"] == 0
        assert snap["latency"] == {"n": 0}
        assert snap["resources"]["vram_mib_recent"] == []
        assert snap["resources"]["cpu_percent_recent"] == []

    def test_record_segment_populates_latency_aggregates(self):
        from operator_app.metrics import MetricsCollector

        c = MetricsCollector()
        c.record_segment(chunk_id=1, stt_ms=100, translate_ms=200, total_ms=300, confidence=0.9, text_len=15)
        c.record_segment(chunk_id=2, stt_ms=120, translate_ms=180, total_ms=300, confidence=0.85, text_len=20)
        c.record_segment(chunk_id=3, stt_ms=110, translate_ms=290, total_ms=400, confidence=0.92, text_len=18)

        snap = c.snapshot()
        lat = snap["latency"]
        assert lat["n"] == 3
        assert lat["total_ms_p50"] == 300.0
        assert lat["confidence_mean"] == pytest.approx((0.9 + 0.85 + 0.92) / 3, rel=1e-3)
        assert len(snap["segments_recent"]) == 3

    def test_segment_buffer_caps_at_60(self):
        from operator_app.metrics import MetricsCollector

        c = MetricsCollector()
        for i in range(100):
            c.record_segment(chunk_id=i, stt_ms=10, translate_ms=10, total_ms=20, confidence=1.0)
        snap = c.snapshot()
        assert snap["latency"]["n"] == 60  # SEGMENT_BUFFER cap

    def test_queue_depth_and_errors(self):
        from operator_app.metrics import MetricsCollector

        c = MetricsCollector()
        c.set_queue_depth(7)
        c.record_error()
        c.record_error()
        snap = c.snapshot()
        assert snap["queue_depth"] == 7
        assert snap["error_count"] == 2


# -- /healthz ----------------------------------------------------------------


class TestHealthz:
    def test_healthz_includes_metrics_fields(self, client):
        resp = client.get("/healthz")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        for key in ("uptime_s", "queue_depth", "error_count", "vram_mib", "cpu_percent", "ts"):
            assert key in body, f"missing {key}"


# -- pull-mode metrics --------------------------------------------------------


class TestApiMetrics:
    def test_returns_snapshot_shape(self, client):
        resp = client.get("/api/metrics")
        assert resp.status_code == 200
        body = resp.json()
        assert "latency" in body
        assert "resources" in body
        assert "segments_recent" in body


# -- /ws/control --------------------------------------------------------------


class TestWsControl:
    def test_streams_at_least_one_frame(self, client):
        with client.websocket_connect("/ws/control") as ws:
            raw = ws.receive_text()
            frame = json.loads(raw)
            assert "ts" in frame
            assert "resources" in frame
            assert "latency" in frame

    def test_segment_recorded_appears_in_next_frame(self, client):
        from operator_app.metrics import get_collector

        collector = get_collector()
        collector.record_segment(chunk_id=42, stt_ms=80, translate_ms=120, total_ms=200, confidence=0.95, text_len=10)
        with client.websocket_connect("/ws/control") as ws:
            frame = json.loads(ws.receive_text())
            assert frame["latency"]["n"] >= 1
            chunk_ids = [s["chunk_id"] for s in frame["segments_recent"]]
            assert 42 in chunk_ids


# -- sampler thread (smoke test) ----------------------------------------------


class TestSamplerThread:
    def test_sampler_loop_does_not_block_test_exit(self):
        # Smoke test only — the sampler thread is daemon=True so it will not
        # block process exit even if it never wakes during our test window.
        from operator_app.metrics import MetricsCollector

        c = MetricsCollector()
        c.start()
        time.sleep(0.05)
        c.stop()
        # Should not raise
