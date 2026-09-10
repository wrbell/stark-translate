"""Run the shell rehearsal over HTTP through real FastAPI routes; no real pipeline."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from operator_app.pipeline_manager import SessionStatus

ROOT = Path(__file__).parent.parent
SCRIPT = ROOT / "scripts" / "dry_run_rehearsal.sh"


@pytest.fixture(scope="module", autouse=True)
def require_bash():
    if shutil.which("bash") is None:
        pytest.skip("bash not available on this platform")


class ControlledRunner:
    """Only process/model work is replaced; routes serialize real status objects."""

    def __init__(self, root):
        self._project_root = root
        self.current = SessionStatus(state="idle")
        self.scenario = "healthy"
        self.starts = []
        self.stops = []
        self.flips = []
        self.polls = 0
        self.polling_started = threading.Event()

    def status(self):
        if self.current.state in {"starting", "running"}:
            self.polls += 1
            self.polling_started.set()
            if self.scenario == "replacement":
                self.current = replace(self.current, session_id="someone_else_en")
            elif self.scenario == "input_error":
                self.current.readiness.update(phase="input_error", reason="No capture samples")
            elif self.scenario == "stale":
                self.current.state = "running"
                self.current.readiness.update(ready=True, stale=True)
                self.current.health.update(stale=True, input_seen=True)
            elif self.polls >= 3:
                self.current.state = "running"
                self.current.readiness.update(phase="ready", ready=True, stale=False, reason="ready")
                self.current.health.update(phase="ready", stale=False, input_seen=True)
        return self.current

    def start(self, config):
        sid = f"rehearsal_{len(self.starts) + 1}_{config.lang}"
        self.starts.append((sid, config.lang))
        self.polls = 0
        self.current = SessionStatus(
            state="starting",
            session_id=sid,
            config=config.__dict__.copy(),
            readiness={"phase": "warming", "ready": False, "stale": False, "reason": "models warming"},
            health={"phase": "warming", "stale": False, "input_seen": False, "recording": {"ok": True}},
        )
        return self.current

    def restart_with(self, config):
        # A second flip before readiness is an actual invalid use of this API.
        assert self.current.state == "running" and self.current.readiness["ready"]
        previous = self.current.session_id
        self.flips.append((previous, config.lang))
        snapshot = self.start(config)
        if self.scenario == "wrong_language":
            snapshot.config["lang"] = "en"
        elif self.scenario == "reused_identity":
            snapshot.session_id = previous
        return snapshot

    def stop(self):
        self.stops.append(self.current.session_id)
        self.current.state = "idle"
        self.current.outcome = "failed" if self.scenario == "failed_stop" else "completed"
        self.current.readiness.update(phase=self.current.outcome, ready=False)
        return self.current


@pytest.fixture
def operator_http(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from operator_app import main
    from operator_app.pipeline_manager import get_runner

    runner = ControlledRunner(tmp_path)
    preflight_calls = []
    monkeypatch.delenv("STARK_PROFILE", raising=False)
    monkeypatch.setitem(main.app.dependency_overrides, get_runner, lambda: runner)
    monkeypatch.setattr(main, "healthz_snapshot", lambda: {"status": "ok", "uptime_s": 1})
    monkeypatch.setattr(main, "get_verse_watcher", lambda **kwargs: None)

    def preflight(config, root):
        preflight_calls.append(config.lang)
        fail = runner.scenario == "preflight_fail" or (runner.scenario == "start_rejected" and len(preflight_calls) > 1)
        return {"ok": not fail, "checks": [], "status_counts": {"pass": 4, "warn": 0, "fail": int(fail)}}

    monkeypatch.setattr(main, "_preflight_config", preflight)
    client = TestClient(main.app)  # No lifespan: no global device watchers or child processes.

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def dispatch(self):
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            response = client.request(
                self.command, self.path, content=body, headers={"Content-Type": "application/json"}
            )
            self.send_response(response.status_code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response.content)))
            self.end_headers()
            self.wfile.write(response.content)

        do_GET = do_POST = dispatch

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", runner, preflight_calls
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        client.close()
        assert not thread.is_alive()


def environment(url, **overrides):
    return {
        "PATH": os.environ["PATH"],
        "STARK_PYTHON": sys.executable,
        "OPERATOR_URL": url,
        "REHEARSAL_TIMEOUT_S": "0.5",
        "REHEARSAL_POLL_S": "0.02",
        **overrides,
    }


def run_rehearsal(url, **overrides):
    return subprocess.run(
        [str(SCRIPT)] if os.name != "nt" else ["bash", str(SCRIPT)],
        capture_output=True,
        text=True,
        env=environment(url, **overrides),
        timeout=10,
    )


def test_waits_for_each_real_status_and_completed_stop(operator_http):
    url, runner, preflight = operator_http
    result = run_rehearsal(url)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "rehearsal passed" in result.stdout
    assert len(runner.starts) == 3
    assert [lang for _, lang in runner.starts] == ["en", "es", "en"]
    assert len(set(sid for sid, _ in runner.starts)) == 3
    assert len(runner.flips) == 2
    assert runner.stops == [runner.starts[-1][0]]
    assert runner.current.outcome == "completed"
    assert preflight == ["en", "en", "es", "en"]
    assert result.stdout.count("✓ ready:") == 3
    assert all(f"[{step}]" in result.stdout for step in range(1, 7))


@pytest.mark.parametrize(
    "scenario,message",
    [
        ("preflight_fail", "Pre-flight"),
        ("start_rejected", "HTTP 422"),
        ("stale", "Timed out"),
        ("input_error", "input_error"),
        ("wrong_language", "expected language es"),
        ("reused_identity", "reused the previous session identity"),
        ("failed_stop", "stop outcome=failed"),
        ("replacement", "ownership changed"),
    ],
)
def test_failures_do_not_report_green_or_stop_another_session(operator_http, scenario, message):
    url, runner, _ = operator_http
    runner.scenario = scenario
    result = run_rehearsal(url, REHEARSAL_TIMEOUT_S="0.15")
    assert result.returncode == 1
    assert "rehearsal failed" in result.stderr
    assert message in result.stderr
    assert "rehearsal passed" not in result.stdout
    if scenario in {"preflight_fail", "start_rejected", "replacement"}:
        assert not runner.stops
    else:
        assert runner.stops == [runner.current.session_id]
        assert runner.current.state == "idle"


def test_refuses_operator_already_in_use(operator_http):
    url, runner, _ = operator_http
    runner.current = SessionStatus(state="paused", session_id="someone_else_en")
    result = run_rehearsal(url)
    assert result.returncode == 1
    assert "not idle" in result.stderr
    assert not runner.starts and not runner.stops


def test_interruption_cleans_only_session_started_by_script(operator_http):
    url, runner, _ = operator_http
    runner.scenario = "stale"
    process = subprocess.Popen(
        ["bash", str(SCRIPT)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=environment(url, REHEARSAL_TIMEOUT_S="5"),
    )
    try:
        assert runner.polling_started.wait(4)
        process.terminate()
        stdout, stderr = process.communicate(timeout=5)
        assert process.returncode == 1
        assert "Interrupted by signal" in stderr
        assert "rehearsal passed" not in stdout
        assert runner.stops == [runner.starts[0][0]]
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=3)


def test_invalid_deadline_has_no_mutations(operator_http):
    url, runner, _ = operator_http
    result = run_rehearsal(url, REHEARSAL_TIMEOUT_S="nan")
    assert result.returncode == 1
    assert "positive finite" in result.stderr
    assert not runner.starts and not runner.stops


def test_script_is_executable_shell_and_runbook_link_is_current():
    if os.name != "nt":
        assert os.access(SCRIPT, os.X_OK)
    assert SCRIPT.read_text().startswith("#!/usr/bin/env bash")
    assert "set -euo pipefail" in SCRIPT.read_text()
    result = subprocess.run(["bash", "-n", str(SCRIPT)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "scripts/dry_run_rehearsal.sh" in (ROOT / "docs/operator_runbook.md").read_text()
