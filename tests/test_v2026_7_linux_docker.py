"""Track 1 (v2026.7) — Linux/CUDA Docker packaging tests.

Smoke tests that don't require Docker itself:
- Dockerfile / docker-compose.yml / entrypoint shape
- /metrics Prometheus endpoint emits the expected gauges/counters
- docker.yml workflow scaffolds the GHCR push correctly

Run with: ``pytest tests/test_v2026_7_linux_docker.py``
"""

from __future__ import annotations

import ast
import os
import re
import socket
import subprocess
from pathlib import Path
from urllib.parse import urlsplit

import pytest
import yaml

ROOT = Path(__file__).parent.parent


# ---------------------------------------------------------------------------
# .dockerignore (build context size guard)
# ---------------------------------------------------------------------------


class TestDockerignore:
    def test_present(self):
        assert (ROOT / ".dockerignore").exists(), (
            "Without .dockerignore the build context exceeds 60 GB on this repo "
            "(stark_data alone is 43 GB). Required before first GHCR build."
        )

    def test_excludes_heavy_dirs(self):
        text = (ROOT / ".dockerignore").read_text()
        for path in ("stark_data/", "models/", "whisper_ablation/", ".venv/", ".git/"):
            assert path in text, f".dockerignore must exclude {path}"

    def test_keeps_models_lock_json(self):
        # The Dockerfile COPY copies models.lock.json by name. If a wildcard
        # excludes it, the build breaks. Defend against future regressions.
        text = (ROOT / ".dockerignore").read_text()
        assert "!models.lock.json" in text


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

    def test_llama_cpp_ref_locked_to_start_server_pin(self):
        docker = (ROOT / "Dockerfile").read_text()
        server = (ROOT / "start_server.sh").read_text()
        assert "ARG LLAMA_CPP_REF=b10883" in docker, "Dockerfile pin must match start_server.sh (Gemma 4 MTP ≥ b10883)"
        assert "b10883" in server, "start_server.sh pin comment must name b10883"

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

    def test_operator_runs_guarded_cli(self):
        # Behavior check: tests/test_operator_security.py runs this entrypoint.
        text = (ROOT / "docker" / "entrypoint.sh").read_text()
        assert "-m operator_app.cli operator" in text

    def test_llama_server_falls_back_to_e2b(self):
        text = (ROOT / "docker" / "entrypoint.sh").read_text()
        assert "gemma-4-e4b-it-q4km.gguf" in text
        assert "gemma-4-e2b-it-q4km.gguf" in text

    def test_llama_server_defaults_to_no_draft(self):
        text = (ROOT / "docker" / "entrypoint.sh").read_text()
        assert "--no-draft" in text
        assert "STARK_LLAMA_MTP" in text
        assert "--mtp" in text


# ---------------------------------------------------------------------------
# start_server.sh (copied into the image; entrypoint execs it)
# ---------------------------------------------------------------------------


class TestStartServer:
    def test_present(self):
        assert (ROOT / "start_server.sh").exists()

    def test_default_is_no_draft(self):
        text = (ROOT / "start_server.sh").read_text()
        assert "NO_DRAFT=true" in text
        assert "--mtp" in text
        assert "draft-mtp" in text
        assert "SPEC_N" in text
        # Legacy E2B spec is opt-in only (measured single-GPU loss).
        assert "--e2b-draft" in text
        assert "--draft 16 --draft-min 5" in text

    def test_mtp_drops_q8_kv(self):
        text = (ROOT / "start_server.sh").read_text()
        assert "Quantized KV" in text or "f16 KV" in text
        assert "--spec-type draft-mtp" in text


# ---------------------------------------------------------------------------
# Container network: run the real shell scripts with stub programs
# ---------------------------------------------------------------------------


def _free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _recording_stub(path, argv_file):
    path.write_text(f'#!/bin/sh\nprintf "%s\\n" "$@" > "{argv_file}"\n')
    path.chmod(0o755)


def _run_start_server(tmp_path, *args):
    """Run start_server.sh with a stub llama-server; return the stub's argv."""
    (tmp_path / "gemma-4-e4b-it-q4km.gguf").touch()
    _recording_stub(tmp_path / "llama-server", tmp_path / "server-argv")
    env = {"PATH": os.environ["PATH"], "HOME": str(tmp_path), "SERVER": str(tmp_path / "llama-server")}
    env["MODEL_DIR"] = str(tmp_path)
    cmd = ["bash", str(ROOT / "start_server.sh"), "--port", str(_free_port()), *args]
    result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=10)
    assert (tmp_path / "server-argv").exists(), result.stdout + result.stderr
    return (tmp_path / "server-argv").read_text().splitlines()


class TestContainerNetwork:
    @staticmethod
    def services():
        return yaml.safe_load((ROOT / "docker-compose.yml").read_text())["services"]

    def test_operator_reaches_llama_server_by_service_name(self):
        url = urlsplit(self.services()["operator"]["environment"]["STARK_CUDA__LLAMACPP_URL"])
        assert url.hostname in self.services()
        assert str(url.port) == self.services()[url.hostname]["environment"]["LLAMA_PORT"]

    def test_only_loopback_publishes_operator_and_no_llama_publish(self):
        ports = [re.sub(r"\$\{\w+:-([^}]*)\}", r"\1", p) for p in self.services()["operator"]["ports"]]
        assert "127.0.0.1:9000:9000" in ports
        assert not self.services()["llama-server"].get("ports")

    def test_entrypoint_makes_llama_server_listen_on_all_interfaces(self, tmp_path):
        (tmp_path / "models").mkdir()
        (tmp_path / "models" / "gemma-4-e4b-it-q4km.gguf").touch()
        _recording_stub(tmp_path / "start_server.sh", tmp_path / "entry-argv")
        env = {
            **self.services()["llama-server"]["environment"],
            "PATH": os.environ["PATH"],
            "STARK_PROJECT_ROOT": str(tmp_path),
            "MODEL_DIR": str(tmp_path / "models"),
        }
        entry = ROOT / "docker" / "entrypoint.sh"
        result = subprocess.run(["bash", str(entry), "llama-server"], env=env, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        args = (tmp_path / "entry-argv").read_text().splitlines()
        assert args[args.index("--host") + 1] == "0.0.0.0"
        # start_server.sh accepts the same flags and gives them to llama-server.
        passed = [a for i, a in enumerate(args) if a != "--port" and args[i - 1] != "--port"]
        served = _run_start_server(tmp_path / "models", *passed)
        assert served[served.index("--host") + 1] == "0.0.0.0"

    def test_start_server_default_stays_loopback(self, tmp_path):
        served = _run_start_server(tmp_path)
        assert served[served.index("--host") + 1] == "127.0.0.1"


def _missing_copy_sources(dockerfile_text):
    missing = []
    for line in dockerfile_text.replace("\\\n", " ").splitlines():
        words = line.split()
        if not words or words[0] != "COPY" or any(w.startswith("--from=") for w in words):
            continue
        sources = [w for w in words[1:] if not w.startswith("--")][:-1]
        missing += [s for s in sources if not (ROOT / s).exists()]
    return missing


class TestDockerfileCopySources:
    def test_every_build_context_copy_source_exists(self):
        assert _missing_copy_sources((ROOT / "Dockerfile").read_text()) == []

    def test_check_detects_a_missing_source(self):
        text = "COPY --from=builder /opt/venv /opt/venv\nCOPY settings.py \\\n    nope.py /app/\n"
        assert _missing_copy_sources(text) == ["nope.py"]


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


def _release_condition(expression, event_name, ref_type, push):
    """Evaluate the workflow's boolean subset against an independent event matrix."""
    expression = expression.removeprefix("${{").removesuffix("}}").strip()
    tree = ast.parse(expression.replace("&&", " and ").replace("||", " or "), mode="eval")
    context = {"github": {"event_name": event_name, "ref_type": ref_type}, "inputs": {"push": push}}

    def value(node):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            return context[node.id]
        if isinstance(node, ast.Attribute):
            return value(node.value)[node.attr]
        if isinstance(node, ast.BoolOp):
            if isinstance(node.op, ast.And):
                return all(value(item) for item in node.values)
            if isinstance(node.op, ast.Or):
                return any(value(item) for item in node.values)
        if isinstance(node, ast.Compare) and len(node.ops) == 1:
            left, right = value(node.left), value(node.comparators[0])
            if isinstance(node.ops[0], ast.Eq):
                return left == right
            if isinstance(node.ops[0], ast.NotEq):
                return left != right
        raise AssertionError(f"Unsupported workflow condition syntax: {ast.dump(node)}")

    return bool(value(tree.body))


class TestDockerPublicationPolicy:
    @staticmethod
    def workflow():
        # BaseLoader preserves the GitHub Actions `on` key (YAML1.1 calls it True).
        return yaml.load((ROOT / ".github/workflows/docker.yml").read_text(), Loader=yaml.BaseLoader)

    @pytest.mark.parametrize(
        "event_name,ref_type,push,expected",
        [
            ("push", "branch", False, False),
            ("push", "branch", True, False),
            ("push", "tag", False, True),
            ("push", "tag", True, True),
            ("workflow_dispatch", "branch", False, False),
            ("workflow_dispatch", "branch", True, True),
            ("workflow_dispatch", "tag", False, False),
            ("workflow_dispatch", "tag", True, True),
            ("pull_request", "branch", False, False),
            ("pull_request", "tag", True, False),
        ],
    )
    def test_login_upload_and_report_obey_event_boundary(self, event_name, ref_type, push, expected):
        steps = self.workflow()["jobs"]["build-and-push"]["steps"]
        login = next(s for s in steps if s.get("uses", "").startswith("docker/login-action@"))
        build = next(s for s in steps if s.get("uses", "").startswith("docker/build-push-action@"))
        summary = next(s for s in steps if s.get("name") == "Image build result")
        for expression in (login["if"], build["with"]["push"], summary["env"]["IMAGE_PUSHED"]):
            assert _release_condition(expression, event_name, ref_type, push) is expected

    def test_main_still_builds_and_manual_push_defaults_false(self):
        workflow = self.workflow()
        assert "main" in workflow["on"]["push"]["branches"]
        assert workflow["on"]["workflow_dispatch"]["inputs"]["push"]["default"] == "false"
        job = workflow["jobs"]["build-and-push"]
        assert not job.get("if"), "The build must not be skipped when publication is disabled"
        build = next(s for s in job["steps"] if s.get("uses", "").startswith("docker/build-push-action@"))
        assert not build.get("if")

    @pytest.mark.parametrize("pushed", [False, True])
    def test_actual_summary_shell_reports_build_or_upload_without_evaluating_tag_text(self, tmp_path, pushed):
        summary = next(
            s for s in self.workflow()["jobs"]["build-and-push"]["steps"] if s.get("name") == "Image build result"
        )
        marker = tmp_path / "tag-must-remain-data"
        tag = f"ghcr.io/example/image:$(touch {marker})"
        result = subprocess.run(
            ["bash", "-e", "-c", summary["run"]],
            env={**os.environ, "IMAGE_PUSHED": str(pushed).lower(), "IMAGE_TAGS": tag},
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        assert ("Uploaded images to GHCR:" in result.stdout) is pushed
        assert ("Built images; no registry upload:" in result.stdout) is not pushed
        assert tag in result.stdout
        assert not marker.exists()
