"""Deterministic overlapping controls; no actual child processes or models."""

import threading
from types import SimpleNamespace
from unittest.mock import Mock

from fastapi.testclient import TestClient

from operator_app.pipeline_manager import InvalidStateError, PipelineRunner, SessionConfig, SessionStatus


def test_queued_stop_cannot_command_join_or_relabel_new_session(tmp_path, monkeypatch):
    from operator_app import metrics, pipeline_manager

    real_thread = threading.Thread
    waiting = threading.Event()
    operations = []
    errors = []

    class OperationGate:
        def __init__(self):
            self.lock = threading.RLock()

        def __enter__(self):
            if threading.current_thread().name == "stale-stop":
                waiting.set()
            self.lock.acquire()

        def __exit__(self, *args):
            self.lock.release()

    class OwnedThread:
        def __init__(self, *, target=None, args=None, name="old", daemon=None):
            self.name = name
            self.alive = False
            self.joins = 0

        def start(self):
            self.alive = True

        def is_alive(self):
            return self.alive

        def join(self, timeout):
            self.joins += 1
            self.alive = False

    class OldProcess:
        pid = 12345
        returncode = None

        def poll(self):
            return self.returncode

        def terminate(self):
            operations.append("terminate-old")
            self.returncode = 0

        def wait(self, timeout):
            self.returncode = 0
            return 0

    monkeypatch.setattr(pipeline_manager, "read_health", lambda *args: {"phase": "warming", "stale": True})
    monkeypatch.setattr(pipeline_manager, "send_control", lambda root, sid, op: operations.append((sid, op)))
    monkeypatch.setattr(pipeline_manager, "descendants", lambda pid: {})
    monkeypatch.setattr(pipeline_manager, "cleanup_children", lambda *args: None)
    monkeypatch.setattr(metrics, "get_collector", lambda: SimpleNamespace(reset_session=lambda sid: None))
    runner = PipelineRunner(tmp_path)
    runner._control_lock = OperationGate()
    runner._generation = 1
    runner._status = SessionStatus(state="running", session_id="old_en")
    runner._proc = OldProcess()
    old_thread = runner._thread = OwnedThread()

    def stale_stop():
        try:
            runner.stop(timeout_s=0.1)
        except InvalidStateError as exc:
            errors.append(str(exc))

    thread = real_thread(target=stale_stop, name="stale-stop")
    try:
        with runner._control_lock:
            thread.start()
            assert waiting.wait(1), "The second Stop never reached the operation gate"
            # This owner represents the first Stop finishing and a new Start
            # winning the gate before the already queued second Stop proceeds.
            assert runner.stop(timeout_s=0.1).state == "idle"
            monkeypatch.setattr(pipeline_manager.threading, "Thread", OwnedThread)
            new = runner.start(SessionConfig(lang="es"))
            new_thread = runner._thread
        thread.join(timeout=1)
        assert not thread.is_alive()
        assert errors and "Session changed" in errors[0]
        assert runner.status().session_id == new.session_id
        assert runner.status().state == "starting"
        assert old_thread.joins == 1
        assert new_thread.joins == 0
        assert operations == [("old_en", "stop"), "terminate-old"]
    finally:
        thread.join(timeout=1)
        runner.stop(timeout_s=0.1)
        if runner._lease_token:
            runner._lease.release(runner._lease_token)
            runner._lease_token = None


def test_stale_stop_api_is_conflict_not_server_error(monkeypatch):
    from operator_app.main import app
    from operator_app.pipeline_manager import get_runner

    runner = SimpleNamespace(stop=Mock(side_effect=InvalidStateError("Session changed while waiting")))
    monkeypatch.setitem(app.dependency_overrides, get_runner, lambda: runner)
    response = TestClient(app).post("/api/session/stop")
    assert response.status_code == 409
    assert "Session changed" in response.json()["detail"]
