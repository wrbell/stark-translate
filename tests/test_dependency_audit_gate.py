"""Exercise the installed dependency report validator and the actual CI shell gate."""

import copy
import subprocess
from pathlib import Path

import pytest
import yaml

from tools.check_dependency_audit import REQUIRED, validate_lite_audit

ROOT = Path(__file__).resolve().parents[1]


def report():
    dependencies = [{"name": name, "version": "1.0", "vulns": []} for name in sorted(REQUIRED)]
    local = next(item for item in dependencies if item["name"] == "stark-translate")
    local.clear()
    local.update(name="stark-translate", skip_reason="Dependency not found on PyPI and could not be audited")
    return {"dependencies": dependencies}


def test_audit_accepts_only_expected_unpublished_project_skip():
    assert validate_lite_audit(report()) == {
        "audited": len(REQUIRED) - 1,
        "skipped_local_project": ["stark-translate"],
        "known_vulnerabilities": 0,
    }


@pytest.mark.parametrize("problem", ["vulnerable", "unknown", "missing", "empty", "duplicate", "torch", "invalid"])
def test_audit_fails_closed_on_vulnerable_or_incomplete_inventory(problem):
    data = report()
    dependency = data["dependencies"][0]
    if problem == "vulnerable":
        dependency["vulns"] = [{"id": "CVE-test"}]
    elif problem == "unknown":
        dependency["skip_reason"] = "Dependency not found on PyPI"
    elif problem == "missing":
        data["dependencies"].pop(0)
    elif problem == "empty":
        data["dependencies"] = []
    elif problem == "duplicate":
        data["dependencies"].append(copy.deepcopy(dependency))
    elif problem == "torch":
        data["dependencies"].append({"name": "torch", "version": "1.0", "vulns": []})
    elif problem == "invalid":
        del dependency["vulns"]
    with pytest.raises(ValueError):
        validate_lite_audit(data)


@pytest.mark.parametrize(
    "detection,legacy_required,lite_required,legacy_result,lite_result,passed",
    [
        ("success", "false", "false", "skipped", "skipped", True),
        ("success", "false", "true", "skipped", "success", True),
        ("success", "true", "true", "success", "success", True),
        ("success", "false", "true", "skipped", "skipped", False),
        ("success", "false", "true", "skipped", "failure", False),
        ("success", "false", "true", "skipped", "cancelled", False),
        ("success", "true", "true", "failure", "success", False),
        ("failure", "false", "false", "skipped", "skipped", False),
    ],
)
def test_security_gate_does_not_hide_failed_or_unexpectedly_skipped_audits(
    detection, legacy_required, lite_required, legacy_result, lite_result, passed
):
    workflow = yaml.safe_load((ROOT / ".github/workflows/security.yml").read_text())
    job = workflow["jobs"]["security"]
    assert "audit-lite" in job["needs"] and job["if"] == "always()"
    script = job["steps"][0]["run"]
    replacements = {
        "needs.detect-changes.result": detection,
        "needs.detect-changes.outputs.requirements_changed == 'true' || github.event_name == 'schedule'": legacy_required,
        "needs.detect-changes.outputs.lite_changed == 'true' || github.event_name == 'schedule'": lite_required,
        "needs.audit.result": legacy_result,
        "needs.audit-lite.result": lite_result,
    }
    for expression, value in replacements.items():
        script = script.replace("${{ " + expression + " }}", value)
    assert "${{" not in script
    result = subprocess.run(["bash", "-e", "-c", script], capture_output=True, text=True, timeout=5)
    assert (result.returncode == 0) is passed, result.stdout + result.stderr
