"""Track 1 (v2026.7) — Linux/CUDA Docker packaging tests.

Smoke tests that don't require Docker itself:
- Dockerfile / docker-compose.yml / entrypoint shape
- /metrics Prometheus endpoint emits the expected gauges/counters
- docker.yml workflow scaffolds the GHCR push correctly

Run with: ``pytest tests/test_v2026_7_linux_docker.py``
"""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent


# ---------------------------------------------------------------------------
# Dockerfile shape
# ---------------------------------------------------------------------------


class TestDockerfile:
    def test_present(self):
        assert (ROOT / "Dockerfile").exists(), "Track 1 Dockerfile missing"

    def test_multi_stage(self):
        text = (ROOT / "Dockerfile").read_text()
        assert "AS builder" in text, "Dockerfile must have a builder stage"
        assert "AS runtime" in text, "Dockerfile must have a runtime stage"
        # devel for build, runtime for ship — keeps final image lean
        assert "nvidia/cuda" in text and "devel" in text and "runtime" in text

    def test_builds_llama_cpp(self):
        text = (ROOT / "Dockerfile").read_text()
        assert "llama.cpp" in text
        assert "GGML_CUDA=ON" in text
        assert "llama-server" in text

    def test_installs_cuda_extras(self):
        text = (ROOT / "Dockerfile").read_text()
        assert ".[cuda]" in text or "[cuda]" in text, "must pip install '.[cuda]'"

    def test_exposes_operator_ports(self):
        text = (ROOT / "Dockerfile").read_text()
        for port in ("9000", "8080", "8765", "8766", "8090"):
            assert port in text, f"Dockerfile must reference port {port}"

    def test_entrypoint_default_cmd(self):
        text = (ROOT / "Dockerfile").read_text()
        assert "ENTRYPOINT" in text and "/app/docker/entrypoint.sh" in text
        assert 'CMD ["operator"]' in text


# ---------------------------------------------------------------------------
# docker-compose.yml shape
# ---------------------------------------------------------------------------


class TestDockerCompose:
    def test_present(self):
        assert (ROOT / "docker-compose.yml").exists()

    def test_three_services(self):
        text = (ROOT / "docker-compose.yml").read_text()
        for svc in ("operator:", "llama-server:", "audio-bridge:"):
            assert svc in text, f"compose missing service block for {svc}"

    def test_cdi_gpu_mode(self):
        text = (ROOT / "docker-compose.yml").read_text()
        # 2025 default — CDI mode rather than legacy --gpus all
        assert "nvidia.com/gpu=all" in text

    def test_audio_bridge_profile_gated(self):
        text = (ROOT / "docker-compose.yml").read_text()
        assert "audio-bridge" in text
        assert "profiles:" in text and "audio-bridge" in text

    def test_models_bind_mount(self):
        text = (ROOT / "docker-compose.yml").read_text()
        assert "STARK_MODELS_DIR" in text
        assert "/app/models" in text


# ---------------------------------------------------------------------------
# entrypoint shape
# ---------------------------------------------------------------------------


class TestEntrypoint:
    def test_present_and_executable(self):
        ep = ROOT / "docker" / "entrypoint.sh"
        assert ep.exists()
        # In git, the +x bit comes back through a checkout — assert by content shape
        text = ep.read_text()
        assert text.startswith("#!"), "entrypoint must start with a shebang"

    def test_dispatches_known_commands(self):
        text = (ROOT / "docker" / "entrypoint.sh").read_text()
        for cmd in ("operator)", "llama-server)", "audio-bridge)", "bash"):
            assert cmd in text, f"entrypoint missing dispatch for {cmd}"

    def test_operator_runs_uvicorn(self):
        text = (ROOT / "docker" / "entrypoint.sh").read_text()
        assert "uvicorn" in text and "operator_app.main:app" in text

    def test_llama_server_falls_back_to_e2b(self):
        text = (ROOT / "docker" / "entrypoint.sh").read_text()
        assert "gemma-4-e4b-it-q4km.gguf" in text
        assert "gemma-4-e2b-it-q4km.gguf" in text


# ---------------------------------------------------------------------------
# /metrics Prometheus endpoint
# ---------------------------------------------------------------------------


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


class TestPrometheusEndpoint:
    def test_returns_text_plain(self, client):
        r = client.get("/metrics")
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/plain")

    def test_emits_required_gauges(self, client):
        body = client.get("/metrics").text
        for metric in (
            "stark_uptime_seconds",
            "stark_queue_depth",
            "stark_errors_total",
            "stark_vram_mib",
            "stark_cpu_percent",
            "stark_latency_total_ms_p50",
            "stark_latency_total_ms_p95",
            "stark_confidence_mean",
            "stark_audio_device_change_seq",
        ):
            assert metric in body, f"/metrics missing {metric}"

    def test_has_help_and_type_lines(self, client):
        body = client.get("/metrics").text
        # Every metric must be preceded by HELP + TYPE for valid exposition format
        for line in body.splitlines():
            if line.startswith("# HELP "):
                name = line.split(" ", 2)[2].split(" ", 1)[0]
                assert f"# TYPE {name} " in body, f"missing TYPE for {name}"

    def test_uptime_is_nonnegative_number(self, client):
        body = client.get("/metrics").text
        for line in body.splitlines():
            if line.startswith("stark_uptime_seconds "):
                value = float(line.split()[-1])
                assert value >= 0
                break
        else:
            pytest.fail("stark_uptime_seconds line not found")


# ---------------------------------------------------------------------------
# CI workflow shape
# ---------------------------------------------------------------------------


class TestDockerWorkflow:
    def test_present(self):
        assert (ROOT / ".github" / "workflows" / "docker.yml").exists()

    def test_pushes_to_ghcr(self):
        text = (ROOT / ".github" / "workflows" / "docker.yml").read_text()
        assert "ghcr.io" in text
        assert "packages: write" in text

    def test_uses_buildx_and_metadata_actions(self):
        text = (ROOT / ".github" / "workflows" / "docker.yml").read_text()
        assert "docker/setup-buildx-action" in text
        assert "docker/metadata-action" in text
        assert "docker/build-push-action" in text

    def test_triggers_on_version_tags(self):
        text = (ROOT / ".github" / "workflows" / "docker.yml").read_text()
        assert 'tags:\n      - "v*"' in text or "tags:\n      - 'v*'" in text
