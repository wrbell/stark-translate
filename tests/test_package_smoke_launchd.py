"""Standalone artifact checks and login services preserve real runtime contracts."""

import json
import os
import plistlib
import subprocess
import sys
from pathlib import Path

import pytest

from operator_app import cli
from operator_app.launchd import manage_launchd
from stark_translate.profiles import PROFILE_NAMES

ROOT = Path(__file__).resolve().parents[1]


def test_standalone_smoke_obeys_actual_localhost_boundary():
    # A child interpreter deliberately excludes conftest's localhost TestClient
    # wrapper. This regression must exercise the real standalone smoke client.
    script = """
import json
from fastapi import FastAPI
from fastapi.testclient import TestClient
from operator_app.security import OperatorBoundaryMiddleware
from tools.installed_smoke import check_operator_routes
app = FastAPI()
app.add_middleware(OperatorBoundaryMiddleware)
@app.get('/{path:path}')
def healthy(path):
    return {'ok': True}
assert TestClient(app).get('/healthz').status_code == 403
assert TestClient(app, base_url='http://127.0.0.1', client=('127.0.0.1', 50000)).get(
    '/healthz', headers={'origin': 'https://untrusted.example'}
).status_code == 403
print(json.dumps(check_operator_routes(app)))
"""
    env = {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
        "STARK_OPERATOR_BIND_HOST": "127.0.0.1",
        "STARK_OPERATOR_ALLOWED_HOSTS": "",
    }
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=ROOT, env=env, capture_output=True, text=True, timeout=10, check=True
    )
    responses = json.loads(result.stdout)
    assert responses == {
        "/healthz": 200,
        "/operator/": 200,
        "/operator/review.js": 200,
        "/operator/widgets/qr.js": 200,
        "/api/capabilities": 200,
    }


@pytest.mark.parametrize("profile", PROFILE_NAMES)
def test_rendered_cli_service_restarts_with_selected_profile(tmp_path, monkeypatch, capsys, profile):
    import uvicorn

    monkeypatch.setenv("STARK_PROFILE", "standard")
    monkeypatch.setenv("STARK_OPERATOR_BIND_HOST", "127.0.0.1")
    monkeypatch.setenv("STARK_MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("STARK_OPERATOR_LOG_DIR", str(tmp_path / "logs"))
    interpreter = tmp_path / "my env" / "bin" / "python"
    interpreter.parent.mkdir(parents=True)
    interpreter.symlink_to(sys.executable)
    calls = []
    monkeypatch.setattr("operator_app.launchd.subprocess.run", lambda *a, **kw: pytest.fail("render used launchctl"))
    assert (
        cli.main(
            ["launchd", "render", "--project-root", str(tmp_path), "--python", str(interpreter), "--profile", profile]
        )
        == 0
    )
    config = plistlib.loads(capsys.readouterr().out.encode())
    assert config["ProgramArguments"][0] == str(interpreter)
    assert config["WorkingDirectory"] == str(tmp_path)
    assert config["EnvironmentVariables"]["STARK_MODELS_DIR"] == str(tmp_path / "models")
    # Simulate login with a different ambient profile; persisted argv wins.
    monkeypatch.setenv("STARK_PROFILE", "standard")
    monkeypatch.setattr(uvicorn, "run", lambda *a, **kw: calls.append((os.environ["STARK_PROFILE"], kw)))
    assert cli.main(config["ProgramArguments"][3:]) == 0
    assert calls[0][0] == profile and calls[0][1]["host"] == "127.0.0.1"


@pytest.mark.parametrize("environment_profile", [None, "lite-cpu-quality", "lite-cuda-8gb"])
def test_lite_alias_render_preserves_its_effective_default(tmp_path, monkeypatch, capsys, environment_profile):
    if environment_profile is None:
        monkeypatch.delenv("STARK_PROFILE", raising=False)
    else:
        monkeypatch.setenv("STARK_PROFILE", environment_profile)
    monkeypatch.setattr("operator_app.launchd.subprocess.run", lambda *a, **kw: pytest.fail("render used launchctl"))
    assert cli.lite_main(["launchd", "render", "--project-root", str(tmp_path)]) == 0
    config = plistlib.loads(capsys.readouterr().out.encode())
    args = config["ProgramArguments"]
    assert args[args.index("--profile") + 1] == (environment_profile or "lite-cpu")


def test_invalid_environment_profile_cannot_write_plist(tmp_path, monkeypatch):
    monkeypatch.setenv("STARK_PROFILE", "not-a-profile")
    target = tmp_path / "agent.plist"
    monkeypatch.setattr("operator_app.launchd.subprocess.run", lambda *a, **kw: pytest.fail("render used launchctl"))
    with pytest.raises(ValueError, match="Unknown profile"):
        manage_launchd("render", project_root=tmp_path, python=Path(sys.executable), output=target)
    assert not target.exists()


def test_invalid_cli_profile_rejected_before_writing(tmp_path, monkeypatch):
    monkeypatch.setenv("STARK_PROFILE", "standard")
    target = tmp_path / "agent.plist"
    with pytest.raises(SystemExit) as caught:
        cli.main(["launchd", "render", "--profile", "not-a-profile", "--output", str(target)])
    assert caught.value.code == 2 and not target.exists()
