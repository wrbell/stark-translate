"""Phase 9.5 — crash recovery + production hardening tests.

Lifespan + log rotation are tricky to assert without I/O, so the tests
focus on the contract: shutting the FastAPI app down with a running
session triggers ``runner.stop()``, the rotating handler is wired up,
and the systemd/launchd files are valid.
"""

from __future__ import annotations

import logging
from pathlib import Path

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


# -- log rotation -------------------------------------------------------------


class TestLogRotation:
    def test_rotating_handler_attached_to_root(self, tmp_path, monkeypatch):
        # Reset existing handlers so the test environment is clean.
        root = logging.getLogger()
        original_handlers = list(root.handlers)
        for h in list(root.handlers):
            if str(getattr(h, "path", "")).endswith("operator.log"):
                root.removeHandler(h)
                h.close()

        monkeypatch.setenv("STARK_OPERATOR_LOG_DIR", str(tmp_path))

        # Re-import operator_app.main with the env override
        import importlib

        from operator_app import main as operator_main

        importlib.reload(operator_main)

        try:
            file_handlers = [h for h in root.handlers if str(getattr(h, "path", "")).endswith("operator.log")]
            assert file_handlers, "expected RotatingFileHandler to be attached"
            handler = file_handlers[0]
            assert handler._sink.maxBytes == 20 * 1024 * 1024
            assert handler._sink.backupCount == 5
            assert (tmp_path / "operator.log").parent.exists()
        finally:
            for h in list(root.handlers):
                if h not in original_handlers:
                    root.removeHandler(h)
                    try:
                        h.close()
                    except Exception:
                        pass


# -- graceful shutdown via lifespan ------------------------------------------


class TestLifespanShutdown:
    @pytest.mark.parametrize("ready", [False, True])
    def test_shutdown_stops_running_session(self, tmp_path, ready, monkeypatch):
        """When the app is torn down, the lifespan context stops the pipeline.

        Use a stub project root with a no-op dry_run_ab so the subprocess
        actually starts, then exercise the lifespan via TestClient as a
        context manager (which fires startup/shutdown).
        """
        from fastapi.testclient import TestClient

        from operator_app import pipeline_manager
        from operator_app.main import app

        monkeypatch.setattr("operator_app.main.run_all_checks", lambda **kw: {"ok": True, "checks": []})
        # Both model loading and a ready pipeline must stop at app shutdown.
        (tmp_path / "metrics").mkdir()
        stub = tmp_path / "dry_run_ab.py"
        stub.write_text(
            "import signal, sys, time, json\n"
            "from pathlib import Path\n"
            "session = sys.argv[sys.argv.index('--session-id') + 1]\n"
            f"if {ready!r}: Path(f'metrics/ab_metrics_{{session}}.csv').write_text('chunk_id,stt_latency_ms\\n')\n"
            "stopped = {'flag': False}\n"
            "def _stop(s, f): stopped['flag'] = True\n"
            "signal.signal(signal.SIGTERM, _stop)\n"
            "signal.signal(signal.SIGINT, _stop)\n"
            "while not stopped['flag']:\n"
            f"    if {ready!r}: Path(f'metrics/health_{{session}}.json').write_text(json.dumps(dict(schema_version=1,session_id=session,updated_at=time.time(),phase='ready')))\n"
            "    time.sleep(0.1)\n"
            "sys.exit(0)\n"
        )

        # Wire the runner singleton to the stub root.
        pipeline_manager.reset_runner_for_tests()
        runner = pipeline_manager.PipelineRunner(project_root=tmp_path)
        pipeline_manager._runner = runner

        with TestClient(app) as client:
            resp = client.post("/api/session/start", json={"lang": "en"})
            assert resp.status_code == 200
            expected_state = "running" if ready else "starting"
            # Wait for the actual child and its expected readiness state.
            import time

            for _ in range(40):
                snapshot = client.get("/api/session/status").json()
                if snapshot["state"] == expected_state and snapshot.get("pid"):
                    break
                time.sleep(0.1)
            assert client.get("/api/session/status").json()["state"] == expected_state
        # Exiting the TestClient context fires app shutdown -> lifespan -> stop().
        assert runner.status().state == "idle"


# -- service file sanity ------------------------------------------------------


class TestSystemdUnit:
    def test_systemd_unit_present_and_well_formed(self):
        path = Path(__file__).parent.parent / "systemd" / "stark-translate.service"
        assert path.exists()
        text = path.read_text()
        for required in (
            "[Unit]",
            "[Service]",
            "[Install]",
            "uvicorn operator_app.main:app",
            "Restart=always",
            "WantedBy=multi-user.target",
        ):
            assert required in text, f"systemd unit missing: {required}"


class TestLaunchdPlist:
    def test_launchd_plist_present_and_well_formed(self):
        path = Path(__file__).parent.parent / "launchd" / "com.starkroad.translate.plist"
        assert path.exists()
        # Verify XML parses.
        import plistlib

        with path.open("rb") as f:
            doc = plistlib.load(f)
        assert doc.get("Label") == "com.starkroad.translate"
        # The checked-in plist is reference-only; installation renders actual paths.
        from operator_app.launchd import launchd_plist

        doc = launchd_plist(path.parent.parent, Path("/test/venv/bin/python"))
        assert "ProgramArguments" in doc
        assert "operator_app.cli" in doc["ProgramArguments"]
        assert doc.get("RunAtLoad") is True
        assert doc.get("KeepAlive", {}).get("SuccessfulExit") is False


class TestRunOperatorScript:
    def test_present_and_executable_pattern(self):
        path = Path(__file__).parent.parent / "run_operator.sh"
        assert path.exists()
        text = path.read_text()
        assert "exec" in text
        assert "uvicorn operator_app.main:app" in text
