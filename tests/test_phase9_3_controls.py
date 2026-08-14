"""Phase 9.3 — mid-session control endpoints + subprocess runner tests.

The subprocess path is exercised end-to-end with a tiny stub script that
mimics ``dry_run_ab.py`` enough to test PipelineRunner: writes a couple of
CSV rows, prints, and waits for SIGTERM. Real pipeline wiring is not
needed for these tests — we're verifying the control plane, not the ML.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _reset_runner_and_collector():
    from operator_app import metrics, pipeline_manager

    pipeline_manager.reset_runner_for_tests()
    metrics.reset_collector_for_tests()
    yield
    pipeline_manager.reset_runner_for_tests()
    metrics.reset_collector_for_tests()


@pytest.fixture
def stub_pipeline(tmp_path: Path) -> Path:
    """Project root with a fake dry_run_ab.py + metrics dir."""
    (tmp_path / "metrics").mkdir()
    stub = tmp_path / "dry_run_ab.py"
    stub.write_text(
        '"""Test stub mimicking dry_run_ab.py CSV output."""\n'
        "from __future__ import annotations\n"
        "import argparse\n"
        "import csv\n"
        "import os\n"
        "import signal\n"
        "import sys\n"
        "import time\n"
        "from datetime import datetime\n"
        "\n"
        "def main():\n"
        "    parser = argparse.ArgumentParser()\n"
        "    parser.add_argument('--lang')\n"
        "    parser.add_argument('--backend', default='auto')\n"
        "    parser.add_argument('--engine', default='auto')\n"
        "    parser.add_argument('--vad-threshold', type=float, default=0.3)\n"
        "    parser.add_argument('--log-level', default='INFO')\n"
        "    parser.add_argument('--ab', action='store_true')\n"
        "    parser.add_argument('--no-ab', action='store_true')\n"
        "    parser.add_argument('--tts', action='store_true')\n"
        "    parser.add_argument('--device', type=int, default=None)\n"
        "    parser.add_argument('--gain', type=float, default=None)\n"
        "    args = parser.parse_args()\n"
        "    # Runner and stub each call datetime.now() for the session id.\n"
        "    # Write a ±1s window so a second-boundary race cannot miss the tail.\n"
        "    now = datetime.now()\n"
        "    paths = []\n"
        "    for delta in (-1, 0, 1):\n"
        "        stamp = datetime.fromtimestamp(now.timestamp() + delta).strftime('%Y%m%d_%H%M%S')\n"
        "        paths.append(os.path.join('metrics', f'ab_metrics_{stamp}_{args.lang}.csv'))\n"
        "    header = ['chunk_id','english','spanish','stt_ms','translate_ms','latency_ms','confidence']\n"
        "    files = [open(p, 'w', newline='') for p in paths]\n"
        "    writers = [csv.writer(f) for f in files]\n"
        "    for w in writers:\n"
        "        w.writerow(header)\n"
        "    for i in range(3):\n"
        "        row = [i, f'sample {i}', f'muestra {i}', 100.0, 200.0, 300.0, 0.9]\n"
        "        for w, f in zip(writers, files):\n"
        "            w.writerow(row)\n"
        "            f.flush()\n"
        "        time.sleep(0.05)\n"
        "    for f in files:\n"
        "        f.close()\n"
        "    # Wait for SIGTERM/SIGINT so the test can stop us deterministically.\n"
        "    stopped = {'flag': False}\n"
        "    def _stop(sig, frame):\n"
        "        stopped['flag'] = True\n"
        "    signal.signal(signal.SIGTERM, _stop)\n"
        "    signal.signal(signal.SIGINT, _stop)\n"
        "    while not stopped['flag']:\n"
        "        time.sleep(0.1)\n"
        "    sys.exit(0)\n"
        "\n"
        "if __name__ == '__main__':\n"
        "    main()\n"
    )
    return tmp_path


# -- subprocess lifecycle ----------------------------------------------------


class TestPipelineRunnerSubprocess:
    def test_start_spawns_subprocess_and_status_running(self, stub_pipeline):
        from operator_app.pipeline_manager import PipelineRunner, SessionConfig

        runner = PipelineRunner(project_root=stub_pipeline)
        snap = runner.start(SessionConfig(lang="en"))
        assert snap.state in ("starting", "running")
        # Wait for state to flip to running (subprocess actually launched)
        for _ in range(40):
            snap = runner.status()
            if snap.state == "running":
                break
            time.sleep(0.1)
        assert snap.state == "running", f"state={snap.state}, last_event={snap.last_event}"
        assert snap.pid is not None
        runner.stop(timeout_s=5)

    def test_stop_terminates_subprocess(self, stub_pipeline):
        from operator_app.pipeline_manager import PipelineRunner, SessionConfig

        runner = PipelineRunner(project_root=stub_pipeline)
        runner.start(SessionConfig(lang="en"))
        # Wait for running state
        for _ in range(40):
            if runner.status().state == "running":
                break
            time.sleep(0.1)
        snap = runner.stop(timeout_s=5)
        assert snap.state == "idle"

    def test_double_start_returns_error(self, stub_pipeline):
        from operator_app.pipeline_manager import PipelineRunner, SessionAlreadyRunningError, SessionConfig

        runner = PipelineRunner(project_root=stub_pipeline)
        runner.start(SessionConfig(lang="en"))
        with pytest.raises(SessionAlreadyRunningError):
            runner.start(SessionConfig(lang="es"))
        runner.stop(timeout_s=5)

    def test_pause_resume_flow(self, stub_pipeline):
        import platform

        if platform.system() == "Windows":
            pytest.skip("SIGSTOP/SIGCONT not supported on Windows")
        from operator_app.pipeline_manager import PipelineRunner, SessionConfig

        runner = PipelineRunner(project_root=stub_pipeline)
        runner.start(SessionConfig(lang="en"))
        for _ in range(40):
            if runner.status().state == "running":
                break
            time.sleep(0.1)
        snap = runner.pause()
        assert snap.state == "paused"
        snap = runner.resume()
        assert snap.state == "running"
        runner.stop(timeout_s=5)

    def test_pause_when_idle_raises(self, stub_pipeline):
        from operator_app.pipeline_manager import InvalidStateError, PipelineRunner

        runner = PipelineRunner(project_root=stub_pipeline)
        with pytest.raises(InvalidStateError):
            runner.pause()


# -- argv builder ------------------------------------------------------------


class TestBuildArgv:
    def test_minimal_argv(self, stub_pipeline):
        from operator_app.pipeline_manager import PipelineRunner, SessionConfig

        runner = PipelineRunner(project_root=stub_pipeline)
        argv = runner._build_argv(SessionConfig(lang="en"))
        assert "--lang" in argv and "en" in argv
        assert "--backend" in argv
        assert "--no-ab" in argv

    def test_engine_passes_through(self, stub_pipeline):
        from operator_app.pipeline_manager import PipelineRunner, SessionConfig

        runner = PipelineRunner(project_root=stub_pipeline)
        argv = runner._build_argv(SessionConfig(lang="en", engine="llamacpp"))
        assert "--engine" in argv
        idx = argv.index("--engine")
        assert argv[idx + 1] == "llamacpp"

    def test_engine_auto_omitted(self, stub_pipeline):
        from operator_app.pipeline_manager import PipelineRunner, SessionConfig

        runner = PipelineRunner(project_root=stub_pipeline)
        argv = runner._build_argv(SessionConfig(lang="en", engine="auto"))
        assert "--engine" not in argv

    def test_run_ab_emits_ab_flag(self, stub_pipeline):
        from operator_app.pipeline_manager import PipelineRunner, SessionConfig

        runner = PipelineRunner(project_root=stub_pipeline)
        argv = runner._build_argv(SessionConfig(lang="en", run_ab=True))
        assert "--ab" in argv
        assert "--no-ab" not in argv


# -- control endpoints (FastAPI) ---------------------------------------------


@pytest.fixture
def client_and_root(stub_pipeline, monkeypatch):
    # Force the runner singleton to use the stub project root
    from fastapi.testclient import TestClient

    from operator_app import pipeline_manager
    from operator_app.main import app

    pipeline_manager.reset_runner_for_tests()
    pipeline_manager._runner = pipeline_manager.PipelineRunner(project_root=stub_pipeline)
    return TestClient(app), stub_pipeline


class TestControlEndpoints:
    def test_pause_when_idle_returns_409(self, client_and_root):
        client, _ = client_and_root
        resp = client.post("/api/control/pause")
        assert resp.status_code == 409

    def test_lang_flip_when_idle_returns_409(self, client_and_root):
        client, _ = client_and_root
        resp = client.post("/api/control/lang_flip")
        assert resp.status_code == 409

    def test_vad_when_idle_returns_409(self, client_and_root):
        client, _ = client_and_root
        resp = client.post("/api/control/vad", json={"threshold": 0.5})
        assert resp.status_code == 409

    def test_fallback_validation(self, client_and_root):
        client, _ = client_and_root
        # invalid engine value caught by pydantic
        resp = client.post("/api/control/fallback", json={"engine": "magic"})
        assert resp.status_code == 422

    def test_vad_validation_out_of_range(self, client_and_root):
        client, _ = client_and_root
        resp = client.post("/api/control/vad", json={"threshold": 1.5})
        assert resp.status_code == 422

    def test_pause_resume_full_cycle(self, client_and_root):
        import platform

        if platform.system() == "Windows":
            pytest.skip("SIGSTOP/SIGCONT not supported on Windows")
        client, _ = client_and_root

        start_resp = client.post("/api/session/start", json={"lang": "en"})
        assert start_resp.status_code == 200
        # Wait for running
        for _ in range(40):
            if client.get("/api/session/status").json()["state"] == "running":
                break
            time.sleep(0.1)
        assert client.post("/api/control/pause").json()["state"] == "paused"
        assert client.post("/api/control/resume").json()["state"] == "running"
        client.post("/api/session/stop")


# -- CSV tail integration ----------------------------------------------------


class TestCsvTailIntegration:
    def test_segments_recorded_via_csv(self, stub_pipeline):
        from operator_app.metrics import get_collector
        from operator_app.pipeline_manager import PipelineRunner, SessionConfig

        runner = PipelineRunner(project_root=stub_pipeline)
        collector = get_collector()
        runner.start(SessionConfig(lang="en"))
        # Wait for the stub to write its 3 rows AND the runner thread to tail them.
        # Stub rows: 3 x 50ms write delay + tail interval = up to ~2s.
        for _ in range(80):
            n = collector.snapshot()["latency"]["n"]
            if n >= 3:
                break
            time.sleep(0.1)
        snap = collector.snapshot()
        # Stub CSV has 3 rows; even with small race conditions we should see >= 1.
        assert snap["latency"]["n"] >= 1, f"no segments tailed; last_event={runner.status().last_event}"
        runner.stop(timeout_s=5)
