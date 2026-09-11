"""Real ASGI routes enforce operator boundaries without model/device work."""

import os
import shutil
import socket
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from operator_app.security import _policy


@pytest.fixture
def guarded_operator(monkeypatch, tmp_path):
    from operator_app import main
    from operator_app.pipeline_manager import SessionStatus, get_runner

    monkeypatch.delenv("STARK_OPERATOR_BIND_HOST", raising=False)
    monkeypatch.delenv("STARK_OPERATOR_ALLOWED_HOSTS", raising=False)
    _policy.cache_clear()
    runner = SimpleNamespace(
        _project_root=tmp_path,
        stop=Mock(return_value=SessionStatus(state="idle", outcome="completed")),
    )
    monkeypatch.setitem(main.app.dependency_overrides, get_runner, lambda: runner)
    monkeypatch.setattr(main, "get_collector", lambda: SimpleNamespace(snapshot=lambda: {"control": "ok"}))
    yield main.app, runner
    _policy.cache_clear()


@pytest.mark.parametrize("origin", [None, "http://localhost", "http://localhost:80"])
def test_local_spa_and_originless_cli_can_control(guarded_operator, origin):
    app, runner = guarded_operator
    headers = {} if origin is None else {"Origin": origin}
    response = TestClient(app).post("/api/session/stop", headers=headers)
    assert response.status_code == 200
    runner.stop.assert_called_once()


@pytest.mark.parametrize(
    "headers",
    [
        {"Origin": "https://evil.example"},
        {"Origin": "http://localhost:9001"},
        {"Origin": "https://localhost"},
        {"Origin": "null"},
        {"Origin": "http://localhost@evil.example"},
        {"Origin": "http://localhost/private"},
        {"Origin": "http://localhost:0"},
        {"Sec-Fetch-Site": "cross-site"},
        {"Sec-Fetch-Site": "same-site"},
    ],
)
def test_cross_origin_controls_fail_before_runner(guarded_operator, headers):
    app, runner = guarded_operator
    assert TestClient(app).post("/api/session/stop", headers=headers).status_code == 403
    runner.stop.assert_not_called()


@pytest.mark.parametrize("host", ["evil.example", "localhost.evil.example", "localhost@evil.example"])
def test_dns_rebinding_host_cannot_read_private_routes(guarded_operator, host):
    app, runner = guarded_operator
    response = TestClient(app).get("/api/storage", headers={"Host": host})
    assert response.status_code == 403
    runner.stop.assert_not_called()


def test_raw_remote_bind_does_not_bypass_local_peer_policy(guarded_operator):
    app, runner = guarded_operator
    client = TestClient(app, client=("192.0.2.40", 50500))
    assert client.get("/api/storage").status_code == 403
    assert client.post("/api/session/stop").status_code == 403
    runner.stop.assert_not_called()


@pytest.mark.parametrize("route", ["/ws/control", "/ws/audio/ingest", "/ws/audio/subscribe"])
def test_websockets_reject_foreign_origins_before_accept(guarded_operator, route):
    app, _ = guarded_operator
    with (
        pytest.raises(WebSocketDisconnect) as rejected,
        TestClient(app).websocket_connect(route, headers={"Origin": "http://evil.example"}),
    ):
        raise AssertionError("Untrusted WebSocket was accepted")
    assert rejected.value.code == 1008


@pytest.mark.parametrize("origin", [None, "http://localhost"])
def test_local_control_websocket_works(guarded_operator, origin):
    app, _ = guarded_operator
    headers = {} if origin is None else {"Origin": origin}
    with TestClient(app).websocket_connect("/ws/control", headers=headers) as ws:
        assert ws.receive_json() == {"control": "ok"}


def test_explicit_remote_bind_allows_local_interface_and_named_proxy(guarded_operator, monkeypatch, caplog):
    import psutil

    from operator_app.security import configure_operator_host

    app, runner = guarded_operator
    monkeypatch.setattr(
        psutil,
        "net_if_addrs",
        lambda: {"ethernet": [SimpleNamespace(family=socket.AF_INET, address="192.0.2.10")]},
    )
    monkeypatch.setenv("STARK_OPERATOR_ALLOWED_HOSTS", "operator.example")
    configure_operator_host("0.0.0.0")
    assert "without authentication" in caplog.text
    for host in ("192.0.2.10", "operator.example"):
        client = TestClient(app, base_url=f"http://{host}:9000", client=("192.0.2.40", 5000))
        assert client.post("/api/session/stop", headers={"Origin": f"http://{host}:9000"}).status_code == 200
        assert client.post("/api/session/stop", headers={"Origin": "http://evil.example"}).status_code == 403
    assert runner.stop.call_count == 2
    assert TestClient(app, base_url="http://arbitrary.example").get("/api/storage").status_code == 403


def test_duplicate_host_or_origin_is_rejected(guarded_operator):
    app, runner = guarded_operator
    for headers in (
        [("Host", "localhost"), ("Host", "evil.example")],
        [("Origin", "http://localhost"), ("Origin", "http://evil.example")],
    ):
        assert TestClient(app).post("/api/session/stop", headers=headers).status_code == 403
    runner.stop.assert_not_called()


def test_cli_default_configures_loopback_without_launching_server(monkeypatch):
    import uvicorn

    from operator_app.cli import main

    monkeypatch.delenv("STARK_OPERATOR_BIND_HOST", raising=False)
    start = Mock()
    monkeypatch.setattr(uvicorn, "run", start)
    assert main(["operator", "--no-browser"]) == 0
    assert start.call_args.kwargs["host"] == "127.0.0.1"
    assert os.environ["STARK_OPERATOR_BIND_HOST"] == "127.0.0.1"


@pytest.mark.skipif(os.name == "nt", reason="POSIX source launcher")
def test_source_launcher_passes_loopback_and_explicit_remote_host_to_guarded_cli(tmp_path):
    root = Path(__file__).resolve().parents[1]
    shutil.copy(root / "run_operator.sh", tmp_path / "run_operator.sh")
    (tmp_path / "scripts").mkdir()
    shutil.copy(root / "scripts" / "runtime_env.sh", tmp_path / "scripts" / "runtime_env.sh")
    python = tmp_path / "python"
    python.write_text('#!/bin/sh\nif [ "$1" = "-c" ]; then exit 0; fi\nprintf "%s\\n" "$@"\n')
    python.chmod(0o755)
    for requested, expected in ((None, "127.0.0.1"), ("0.0.0.0", "0.0.0.0")):
        env = {"PATH": os.environ["PATH"], "STARK_PYTHON": str(python)}
        if requested:
            env["HOST"] = requested
        result = subprocess.run(
            ["bash", "run_operator.sh"], cwd=tmp_path, env=env, text=True, capture_output=True, timeout=3
        )
        assert result.returncode == 0, result.stderr
        assert f"python: {python} " in result.stdout.splitlines()[0]
        args = result.stdout.splitlines()[1:]
        assert args[:3] == ["-m", "operator_app.cli", "operator"]
        assert args[args.index("--host") + 1] == expected


def test_launchd_default_is_loopback(tmp_path):
    from operator_app.launchd import launchd_plist

    args = launchd_plist(tmp_path, Path("/python"))["ProgramArguments"]
    assert args[args.index("--host") + 1] == "127.0.0.1"
