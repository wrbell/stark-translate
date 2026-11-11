"""Run the double-click Mac launcher with fake commands in a folder whose name has a space."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = "Stark Translate.command"
pytestmark = pytest.mark.skipif(os.name == "nt", reason="POSIX launcher")

USABLE = '#!/bin/sh\n[ "$1" = "-c" ] && exit 0\nexit 0\n'
UNUSABLE = '#!/bin/sh\n[ "$1" = "-c" ] && exit 1\nexit 0\n'


@pytest.fixture
def launcher(tmp_path):
    project = tmp_path / "Church Mac" / "stark translate"
    shims = tmp_path / "fake bin"
    (project / "scripts").mkdir(parents=True)
    shims.mkdir()
    shutil.copy2(ROOT / LAUNCHER, project / LAUNCHER)
    shutil.copy2(ROOT / "scripts/runtime_env.sh", project / "scripts/runtime_env.sh")
    for name, body in {
        "uname": "echo arm64",
        "curl": 'exit "${FAKE_HEALTHZ:-7}"',
        "open": 'echo "opened $*" >> "$FAKE_LOG"',
        "python3": UNUSABLE,
        "python3.11": UNUSABLE,
        "bootstrap.sh": 'echo "bootstrap $* python=$STARK_PYTHON" >> "$FAKE_LOG"; echo "[bootstrap] ERROR: model setup failed" >&2; exit 3',
        "run_operator.sh": 'echo "operator browser=$STARK_OPEN_BROWSER" >> "$FAKE_LOG"',
    }.items():
        target = (project if name.endswith(".sh") else shims) / name
        target.write_text(body if body.startswith("#!") else "#!/bin/sh\n" + body + "\n")
        target.chmod(0o755)
    for tool in ("dirname", "mkdir", "tee"):
        (shims / tool).symlink_to(shutil.which(tool))
    log = tmp_path / "calls.txt"
    env = {k: v for k, v in os.environ.items() if k not in {"STARK_PYTHON", "VENV", "VIRTUAL_ENV", "CONDA_PREFIX"}}
    env.update(PATH=str(shims), FAKE_LOG=str(log), HOME=str(tmp_path))

    def run(*args, **overrides):
        result = subprocess.run(
            [str(project / LAUNCHER), *args],
            cwd=tmp_path,
            env={**env, **overrides},
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result, (log.read_text() if log.exists() else "")

    def runtime():
        python = project / "venv/bin/python"
        python.parent.mkdir(parents=True)
        python.write_text(USABLE)
        python.chmod(0o755)

    run.project, run.runtime, run.shims = project, runtime, shims
    return run


def test_launcher_is_executable_and_valid_bash():
    path = ROOT / LAUNCHER
    assert os.access(path, os.X_OK), f"chmod +x '{LAUNCHER}' (git mode 100755)"
    assert subprocess.run(["bash", "-n", str(path)]).returncode == 0


def test_dry_run_without_runtime_plans_bootstrap_then_start(launcher):
    result, calls = launcher("--dry-run")
    assert result.returncode == 0, result.stderr
    plan = [line for line in result.stdout.splitlines() if line.startswith("would run:")]
    assert plan == [
        f"would run: {launcher.project}/bootstrap.sh --skip-systemd",
        f"would run: {launcher.project}/run_operator.sh (browser opens)",
    ]
    assert calls == ""


def test_dry_run_with_runtime_plans_start_only(launcher):
    launcher.runtime()
    result, _ = launcher(STARK_LAUNCHER_DRY_RUN="1")
    assert result.returncode == 0, result.stderr
    plan = [line for line in result.stdout.splitlines() if line.startswith("would run:")]
    assert plan == [f"would run: {launcher.project}/run_operator.sh (browser opens)"]


def test_running_operator_only_opens_the_browser(launcher):
    result, calls = launcher(FAKE_HEALTHZ="0")
    assert result.returncode == 0, result.stderr
    assert calls == "opened http://localhost:9000/operator/\n"


def test_start_uses_run_operator_with_the_browser(launcher):
    launcher.runtime()
    result, calls = launcher()
    assert result.returncode == 0, result.stderr
    assert calls == "operator browser=1\n"


def test_bootstrap_failure_names_the_failure_and_the_log(launcher):
    result, calls = launcher()
    assert result.returncode == 1
    assert calls == f"bootstrap --skip-systemd python={launcher.shims}/python3.11\n"
    log = launcher.project / "metrics/install.log"
    assert "model setup failed" in log.read_text()
    assert f"The full log is in {log}." in result.stdout
    assert "Press Return to close this window." in result.stdout


def test_missing_python_311_stops_with_one_plain_sentence(launcher):
    (launcher.shims / "python3.11").unlink()
    result, calls = launcher()
    assert result.returncode == 1
    assert "This Mac does not have Python 3.11." in result.stdout
    assert calls == ""


def test_run_operator_opens_the_browser_only_when_asked(tmp_path):
    shutil.copy(ROOT / "run_operator.sh", tmp_path / "run_operator.sh")
    (tmp_path / "scripts").mkdir()
    shutil.copy(ROOT / "scripts/runtime_env.sh", tmp_path / "scripts/runtime_env.sh")
    python = tmp_path / "python"
    python.write_text('#!/bin/sh\nif [ "$1" = "-c" ]; then exit 0; fi\nprintf "%s\\n" "$@"\n')
    python.chmod(0o755)
    for flag, expected in (("", True), ("1", False)):
        env = {"PATH": os.environ["PATH"], "STARK_PYTHON": str(python), "STARK_OPEN_BROWSER": flag}
        result = subprocess.run(["bash", "run_operator.sh"], cwd=tmp_path, env=env, text=True, capture_output=True)
        assert result.returncode == 0, result.stderr
        assert ("--no-browser" in result.stdout.splitlines()) is expected
