"""Offline runtime audit checks, including the shell entry point with inert tools."""

import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pytest

from tools.check_dependency_audit import MAC_REQUIRED, REQUIRED, validate_audit

ROOT = Path(__file__).resolve().parents[1]


def report(names):
    dependencies = [{"name": name, "version": "1.0", "vulns": []} for name in sorted(names)]
    local = next(item for item in dependencies if item["name"] == "stark-translate")
    local.clear()
    local.update(name="stark-translate", skip_reason="Dependency not found on PyPI and could not be audited")
    return {"dependencies": dependencies}


def test_mac_accepts_torch_while_lite_rejects_it():
    data = report(MAC_REQUIRED | REQUIRED)
    assert validate_audit(data, runtime="mac")["known_vulnerabilities"] == 0
    with pytest.raises(ValueError, match=r"Non-Lite dependencies installed.*torch"):
        validate_audit(data)


def test_mac_requires_mlx():
    with pytest.raises(ValueError, match=r"missing from audit.*mlx"):
        validate_audit(report(MAC_REQUIRED - {"mlx"}), runtime="mac")


@pytest.mark.parametrize("runtime,names", [("lite", REQUIRED), ("mac", MAC_REQUIRED)])
@pytest.mark.parametrize("problem", ["vulnerability", "duplicate", "skip", "incomplete"])
def test_both_runtimes_reject_bad_inventory(runtime, names, problem):
    data = report(names)
    item = data["dependencies"][0]
    if problem == "vulnerability":
        item["vulns"] = [{"id": "CVE-test"}]
    elif problem == "duplicate":
        data["dependencies"].append(dict(item))
    elif problem == "skip":
        item["skip_reason"] = "Dependency not found on PyPI"
    else:
        del item["version"]
    with pytest.raises(ValueError):
        validate_audit(data, runtime=runtime)


@pytest.mark.parametrize("runtime", [None, "lite", "mac"])
def test_cli_reports_runtime(tmp_path, runtime):
    path = tmp_path / "audit.json"
    path.write_text(json.dumps(report(MAC_REQUIRED if runtime == "mac" else REQUIRED)))
    args = ["--runtime", runtime] if runtime else []
    result = subprocess.run(
        [sys.executable, str(ROOT / "tools/check_dependency_audit.py"), *args, str(path)],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["runtime"] == (runtime or "lite")


@pytest.fixture
def audit_runtime(tmp_path):
    project = tmp_path / "project"
    for relative in ("scripts/audit_mac_runtime.sh", "scripts/runtime_env.sh", "tools/check_dependency_audit.py"):
        target = project / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    interpreter = project / "pointed env/bin/python"
    interpreter.parent.mkdir(parents=True)
    interpreter.write_text(
        f"#!{sys.executable}\n"
        "import json, os, runpy, sys\n"
        "from pathlib import Path\n"
        "with Path(os.environ['AUDIT_TEST_EVENTS']).open('a') as stream:\n"
        "    stream.write(json.dumps(sys.argv) + '\\n')\n"
        "if sys.argv[1] == '-c':\n"
        "    assert sys.argv[2] == 'import sysconfig; print(sysconfig.get_paths()[\"purelib\"])'\n"
        "    print(os.environ['AUDIT_TEST_SITE'])\n"
        "elif sys.argv[1] == '-':\n"
        "    sys.argv = sys.argv[1:]\n"
        "    exec(sys.stdin.read())\n"
        "else:\n"
        "    sys.argv = sys.argv[1:]\n"
        "    runpy.run_path(sys.argv[0], run_name='__main__')\n"
    )
    interpreter.chmod(0o755)
    (project / ".stark-python").write_text("pointed env/bin/python\n")
    auditor = tmp_path / "auditor tools/pip-audit"
    auditor.parent.mkdir()
    auditor.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "with Path(os.environ['AUDIT_TEST_EVENTS']).open('a') as stream:\n"
        "    stream.write(json.dumps(sys.argv) + '\\n')\n"
        "assert sys.argv[1:5] == ['--path', os.environ['AUDIT_TEST_SITE'], '--format', 'json']\n"
        "assert sys.argv[5] == '--output'\n"
        "Path(sys.argv[6]).write_text(os.environ['AUDIT_TEST_REPORT'])\n"
        "raise SystemExit(int(os.environ['AUDIT_TEST_EXIT']))\n"
    )
    auditor.chmod(0o755)
    events = tmp_path / "events.jsonl"
    env = {k: v for k, v in os.environ.items() if k not in {"STARK_PYTHON", "VENV", "VIRTUAL_ENV", "CONDA_PREFIX"}}
    env.update(
        AUDIT_TEST_EVENTS=str(events),
        AUDIT_TEST_SITE=str(interpreter.parent.parent / "lib/site-packages"),
        AUDIT_TEST_REPORT=json.dumps(report(MAC_REQUIRED)),
        AUDIT_TEST_EXIT="0",
        PATH=str(auditor.parent) + os.pathsep + os.environ["PATH"],
    )

    def run(args, overrides=None):
        result = subprocess.run(
            ["/bin/bash", str(project / "scripts/audit_mac_runtime.sh"), *args],
            cwd=tmp_path,
            env={**env, **(overrides or {})},
            capture_output=True,
            text=True,
            timeout=10,
        )
        records = [json.loads(line) for line in events.read_text().splitlines()] if events.exists() else []
        return result, records

    return run, interpreter, auditor, env["AUDIT_TEST_SITE"]


@pytest.mark.parametrize("explicit_auditor", [False, True])
@pytest.mark.parametrize("problem", [None, "findings", "missing-mlx", "auditor-failure", "malformed"])
def test_audit_script_uses_pointer_and_records_failures(tmp_path, audit_runtime, explicit_auditor, problem):
    run, interpreter, auditor, site = audit_runtime
    data = report(MAC_REQUIRED - ({"mlx"} if problem == "missing-mlx" else set()))
    if problem == "findings":
        data["dependencies"][0]["vulns"] = [{"id": "CVE-test"}]
    audit_exit = 1 if problem == "findings" else 7 if problem == "auditor-failure" else 0
    validation_exit = 1 if problem in {"findings", "missing-mlx", "malformed"} else 0
    output = tmp_path / "audit output"
    args = ["--output", str(output)]
    if explicit_auditor:
        args += ["--auditor", str(auditor)]
    result, records = run(
        args,
        {"AUDIT_TEST_REPORT": "{" if problem == "malformed" else json.dumps(data), "AUDIT_TEST_EXIT": str(audit_exit)},
    )
    assert result.returncode == (audit_exit or validation_exit), result.stderr
    receipt = json.loads((output / "receipt.json").read_text())
    assert receipt["interpreter"] == str(interpreter)
    assert receipt["site_packages"] == site
    assert receipt["auditor"] == str(auditor)
    assert datetime.fromisoformat(receipt["timestamp"]).utcoffset().total_seconds() == 0
    assert receipt["exit_codes"] == {"site_packages": 0, "audit": audit_exit, "validation": validation_exit}
    validation = next(row for row in records if "--runtime" in row)
    assert validation == [
        str(interpreter),
        str(interpreter.parents[2] / "tools/check_dependency_audit.py"),
        "--runtime",
        "mac",
        str(output / "installed-audit.json"),
    ]
    assert {row[0] for row in records} == {str(interpreter), str(auditor)}


@pytest.mark.parametrize("problem", ["missing-output", "existing-output", "invalid-auditor", "missing-value"])
def test_audit_script_rejects_invalid_arguments(tmp_path, audit_runtime, problem):
    run, _, _, _ = audit_runtime
    output = tmp_path / "output"
    args = ["--output", str(output)]
    if problem == "missing-output":
        args = []
    elif problem == "existing-output":
        output.mkdir()
    elif problem == "invalid-auditor":
        args += ["--auditor", str(tmp_path / "missing")]
    else:
        args += ["--auditor"]
    result, records = run(args)
    assert result.returncode == 2
    assert records == []
    assert not (output / "receipt.json").exists()
