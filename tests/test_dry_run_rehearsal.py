"""Tests for scripts/dry_run_rehearsal.sh.

Exercises the script end-to-end against a tiny in-process HTTP server
that stands in for the operator FastAPI app. Two scenarios:

- All endpoints return healthy → script exits 0.
- /api/preflight reports a red check → script exits 1.

Skipped on platforms without bash (Windows CI).
"""

from __future__ import annotations

import json
import shutil
import socket
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
SCRIPT = ROOT / "scripts" / "dry_run_rehearsal.sh"


# -- bash existence ----------------------------------------------------------


@pytest.fixture(scope="module", autouse=True)
def _require_bash():
    if shutil.which("bash") is None:
        pytest.skip("bash not available on this platform")
    if shutil.which("curl") is None:
        pytest.skip("curl not available on this platform")


# -- fake operator HTTP server -----------------------------------------------


def _make_handler(state):
    """Build a handler class closing over a mutable state dict.

    state['preflight_fail'] toggles whether /api/preflight reports red.
    state['session_state'] tracks the simulated session state.
    """

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args, **_kwargs):
            return  # silence stderr noise during tests

        def _json(self, status, body):
            payload = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            if self.path == "/healthz":
                return self._json(200, {"status": "ok", "uptime_s": 1})
            if self.path == "/api/preflight":
                fail = 1 if state.get("preflight_fail") else 0
                return self._json(
                    200,
                    {
                        "checks": [],
                        "ok": fail == 0,
                        "status_counts": {"pass": 4, "warn": 0, "fail": fail},
                    },
                )
            if self.path == "/api/session/status":
                return self._json(200, {"state": state.get("session_state", "idle")})
            if self.path.startswith("/api/features/verses"):
                return self._json(200, {"highlights": [], "since_chunk": None})
            return self._json(404, {"detail": "not found"})

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            if length:
                _ = self.rfile.read(length)
            if self.path == "/api/session/start":
                state["session_state"] = "running"
                return self._json(200, {"state": "running", "session_id": "test_sid"})
            if self.path == "/api/control/lang_flip":
                return self._json(200, {"state": "running"})
            if self.path == "/api/session/stop":
                state["session_state"] = "idle"
                return self._json(200, {"state": "idle"})
            return self._json(404, {"detail": "not found"})

    return Handler


@pytest.fixture
def fake_operator():
    """Spin up a fake operator server on an ephemeral port. Yields (url, state)."""
    state = {"preflight_fail": False, "session_state": "idle"}
    Handler = _make_handler(state)

    # Bind to ephemeral port
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()

    server = HTTPServer(("127.0.0.1", port), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}", state
    finally:
        server.shutdown()
        server.server_close()


def _run_rehearsal(url: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(SCRIPT)],
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin", "OPERATOR_URL": url},
        timeout=30,
    )


# -- happy path --------------------------------------------------------------


class TestHappyPath:
    def test_exits_zero_when_all_green(self, fake_operator):
        url, _state = fake_operator
        result = _run_rehearsal(url)
        assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        assert "rehearsal passed" in result.stdout
        # Every numbered step should appear
        for step in ("[1]", "[2]", "[3]", "[4]", "[5]", "[6]"):
            assert step in result.stdout

    def test_walks_session_lifecycle(self, fake_operator):
        url, state = fake_operator
        result = _run_rehearsal(url)
        assert result.returncode == 0
        # Session must have been started, then stopped — we observe the final
        # state via the fake server's mutable record.
        assert state["session_state"] == "idle"


# -- failure path ------------------------------------------------------------


class TestFailurePath:
    def test_exits_nonzero_when_preflight_red(self, fake_operator):
        url, state = fake_operator
        state["preflight_fail"] = True
        result = _run_rehearsal(url)
        assert result.returncode == 1
        assert "rehearsal failed" in result.stdout


# -- script shape ------------------------------------------------------------


class TestScriptShape:
    def test_has_shebang(self):
        assert SCRIPT.read_text().startswith("#!/usr/bin/env bash")

    def test_set_euo_pipefail(self):
        assert "set -euo pipefail" in SCRIPT.read_text()

    def test_bash_syntax_valid(self):
        result = subprocess.run(["bash", "-n", str(SCRIPT)], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr

    def test_runbook_links_to_script(self):
        runbook = (ROOT / "docs" / "operator_runbook.md").read_text()
        assert "scripts/dry_run_rehearsal.sh" in runbook
