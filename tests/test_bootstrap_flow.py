"""Execute bootstrap with inert installers/services and isolated temporary targets."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

SHIM = r"""
import json, os
from pathlib import Path
import sys

exe = Path(sys.argv[0]).absolute()
args = sys.argv[1:]
root = Path(os.environ['BOOTSTRAP_TEST_ROOT'])
with Path(os.environ['BOOTSTRAP_TEST_EVENTS']).open('a') as stream:
    stream.write(json.dumps({'exe': str(exe), 'args': args}) + '\n')
if exe.name == 'sudo':
    if args[0] == 'tee':
        Path(os.environ['BOOTSTRAP_TEST_DROPIN']).write_text(sys.stdin.read())
    raise SystemExit(0)
if args[:1] == ['-c']:
    if 'version_info' in args[1]:
        print('3.11')
    elif 'sys.prefix' in args[1]:
        print(os.environ.get('BOOTSTRAP_TEST_PREFIX', '' if exe.name == 'python3' else str(exe.parent.parent)))
    else:
        raise SystemExit('Unexpected -c command')
elif args[:2] == ['-m', 'venv']:
    target = Path(args[2]).absolute()
    assert target.is_relative_to(root), target
    binary = target / 'bin' / 'python'
    binary.parent.mkdir(parents=True, exist_ok=True)
    binary.write_text(exe.read_text())
    binary.chmod(0o755)
elif args[:2] == ['-m', 'pip']:
    pass
elif args[:2] == ['-m', 'operator_app.cli']:
    if args[2] == os.environ.get('BOOTSTRAP_TEST_FAIL_STAGE'):
        raise SystemExit(17)
    assert args[2] in ('setup', 'doctor', 'launchd'), args
else:
    raise SystemExit('Unexpected executable invocation: ' + repr(args))
"""


@pytest.fixture
def bootstrap(tmp_path):
    project = tmp_path / "project"
    shims = tmp_path / "shims"
    project.mkdir()
    shims.mkdir()
    (project / "scripts").mkdir()
    shutil.copyfile(ROOT / "scripts/runtime_env.sh", project / "scripts/runtime_env.sh")
    (project / "systemd").mkdir()
    (project / "systemd/stark-translate.service").write_text("[Service]\n")
    # Isolate only the legacy fixed log destination; execute the real control flow.
    script = (ROOT / "bootstrap.sh").read_text().replace("/tmp/bootstrap-pip.log", str(tmp_path / "pip.log"))
    (project / "bootstrap.sh").write_text(script)
    for name in ("python3", "sudo"):
        path = shims / name
        path.write_text(f"#!{sys.executable}\n" + SHIM)
        path.chmod(0o755)
    for name, body in {
        "uname": 'if [ "$1" = "-s" ]; then echo Linux; else echo x86_64; fi',
        "ffmpeg": "exit 0",
        "systemctl": "exit 0",
    }.items():
        path = shims / name
        path.write_text("#!/bin/sh\n" + body + "\n")
        path.chmod(0o755)
    (shims / "dirname").symlink_to(shutil.which("dirname"))
    events = tmp_path / "events.jsonl"
    dropin = tmp_path / "dropin.conf"
    env = {k: v for k, v in os.environ.items() if k not in {"STARK_PYTHON", "VENV", "VIRTUAL_ENV", "CONDA_PREFIX"}}
    env.update(
        PATH=str(shims),
        USER="inert-test-user",
        BOOTSTRAP_TEST_ROOT=str(tmp_path),
        BOOTSTRAP_TEST_EVENTS=str(events),
        BOOTSTRAP_TEST_DROPIN=str(dropin),
    )

    class Harness:
        def environment(self, name):
            path = project / name
            binary = path / "bin/python"
            binary.parent.mkdir(parents=True, exist_ok=True)
            binary.write_text((shims / "python3").read_text())
            binary.chmod(0o755)
            return path

        def run(self, overrides=None, *, service=None):
            argv = [] if service == "systemd" else ["--skip-systemd"]
            if service == "launchd":
                argv.append("--install-launchd")
            result = subprocess.run(
                ["/bin/bash", str(project / "bootstrap.sh"), *argv],
                cwd=project,
                env={**env, **(overrides or {})},
                capture_output=True,
                text=True,
                timeout=10,
            )
            records = [json.loads(line) for line in events.read_text().splitlines()] if events.exists() else []
            return result, records

    harness = Harness()
    harness.project, harness.dropin = project, dropin
    return harness


def installer_records(records):
    return [row for row in records if row["args"][:2] == ["-m", "pip"]]


@pytest.mark.parametrize("empty_prefix", [True, False])
def test_bootstrap_refuses_stt_env_when_pointer_names_other_env(bootstrap, empty_prefix):
    rollback = bootstrap.environment("stt_env")
    other = bootstrap.environment("other")
    (bootstrap.project / ".stark-python").write_text(str(other / "bin/python") + "\n")
    # Empty prefixes fall back to venv; a symlink can make that the rollback env.
    (bootstrap.project / "venv").symlink_to(rollback, target_is_directory=True)
    result, records = bootstrap.run({"BOOTSTRAP_TEST_PREFIX": "" if empty_prefix else str(rollback)})
    assert result.returncode == 3, result.stderr
    assert "refusing to install into the rollback environment stt_env" in result.stderr
    assert str(other / "bin/python") in result.stderr
    assert installer_records(records) == []
    assert not any(row["args"][:2] == ["-m", "venv"] for row in records)


def test_bootstrap_with_pointer_installs_into_the_pointed_env(bootstrap):
    rollback = bootstrap.environment("stt_env")
    other = bootstrap.environment("other")
    (bootstrap.project / ".stark-python").write_text(str(other / "bin/python") + "\n")
    result, records = bootstrap.run()
    assert result.returncode == 0, result.stderr
    assert {row["exe"] for row in installer_records(records)} == {str(other / "bin/python")}
    assert not any(row["args"][:2] == ["-m", "venv"] for row in records)
    assert all(row["exe"] != str(rollback / "bin/python") for row in records)


def service_records(records):
    return [
        row
        for row in records
        if Path(row["exe"]).name == "sudo" or row["args"][:3] == ["-m", "operator_app.cli", "launchd"]
    ]


@pytest.mark.parametrize("precreated", [False, True])
def test_bootstrap_creates_explicit_relative_target_even_if_directory_exists(bootstrap, precreated):
    target = bootstrap.project / "new env"
    if precreated:
        target.mkdir()
    # Creation uses an available active interpreter, but installation must use target.
    active = bootstrap.environment("active")
    result, records = bootstrap.run({"VENV": "new env", "VIRTUAL_ENV": str(active)}, service="systemd")
    assert result.returncode == 0, result.stderr
    creation = next(row for row in records if row["args"][:2] == ["-m", "venv"])
    assert creation["exe"] == str(active / "bin/python")
    assert creation["args"][2] == "new env"
    assert (target / "bin/python").is_file()
    assert {row["exe"] for row in installer_records(records)} == {str(target / "bin/python")}
    dropin = bootstrap.dropin.read_text()
    assert f'ExecStart="{target}/bin/uvicorn" operator_app.main:app' in dropin


@pytest.mark.parametrize("selected", ["explicit", "target", "pointer", "active", "conda", "stt_env", "venv"])
def test_bootstrap_preserves_authorized_environment_precedence(bootstrap, selected):
    paths = {
        name: bootstrap.environment(name)
        for name in ["explicit", "target", "pointer", "active", "conda", "stt_env", "venv"]
    }
    overrides = {}
    if selected == "explicit":
        overrides["STARK_PYTHON"] = str(paths["explicit"] / "bin/python")
    if selected in ("explicit", "target"):
        overrides["VENV"] = str(paths["target"])
    if selected in ("explicit", "target", "pointer"):
        (bootstrap.project / ".stark-python").write_text(str(paths["pointer"] / "bin/python") + "\n")
    if selected in ("explicit", "target", "pointer", "active"):
        overrides["VIRTUAL_ENV"] = str(paths["active"])
    if selected in ("explicit", "target", "pointer", "active", "conda"):
        overrides["CONDA_PREFIX"] = str(paths["conda"])
    if selected == "venv":
        (paths["stt_env"] / "bin/python").unlink()
    result, records = bootstrap.run(overrides)
    assert result.returncode == 0, result.stderr
    assert {row["exe"] for row in installer_records(records)} == {str(paths[selected] / "bin/python")}
    assert not any(row["args"][:2] == ["-m", "venv"] for row in records)
    if selected == "pointer":
        assert all(row["exe"] != str(paths["stt_env"] / "bin/python") for row in records)


@pytest.mark.parametrize("service", ["systemd", "launchd"])
def test_services_are_installed_only_after_successful_setup_and_doctor(bootstrap, service):
    env = bootstrap.environment("venv")
    result, records = bootstrap.run({"VENV": str(env)}, service=service)
    assert result.returncode == 0, result.stderr
    setup_at = next(i for i, row in enumerate(records) if row["args"][:3] == ["-m", "operator_app.cli", "setup"])
    doctor_at = next(i for i, row in enumerate(records) if row["args"][:3] == ["-m", "operator_app.cli", "doctor"])
    actual_services = service_records(records)
    assert actual_services
    assert setup_at < doctor_at < min(records.index(row) for row in actual_services)


@pytest.mark.parametrize("service", ["systemd", "launchd"])
@pytest.mark.parametrize("failed_stage,exit_code", [("setup", 3), ("doctor", 5)])
def test_failed_setup_or_doctor_never_mutates_services(bootstrap, service, failed_stage, exit_code):
    env = bootstrap.environment("venv")
    result, records = bootstrap.run({"VENV": str(env), "BOOTSTRAP_TEST_FAIL_STAGE": failed_stage}, service=service)
    assert result.returncode == exit_code, result.stderr
    assert service_records(records) == []
    assert not bootstrap.dropin.exists()
