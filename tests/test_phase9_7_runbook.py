"""Phase 9.7 — operator runbook + bootstrap.sh shape tests.

These are file-existence + content-pattern tests. The bootstrap script
is hard to exercise end-to-end without spinning up sudo, systemd, and
a fresh venv — that's covered by manual install on the church PC.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).parent.parent


class TestRunbook:
    def test_runbook_exists(self):
        path = ROOT / "docs" / "operator_runbook.md"
        assert path.exists()
        text = path.read_text()
        assert "Operator Runbook" in text

    def test_runbook_has_required_sections(self):
        text = (ROOT / "docs" / "operator_runbook.md").read_text()
        # Sections that map to the runbook structure described in the plan
        for section in (
            "Before the event",
            "During the event",
            "Start the session",
            "When something goes wrong",
            "End-of-event checklist",
            "Pre-event dry-run",
        ):
            assert section in text, f"runbook missing section: {section}"

    def test_runbook_references_real_endpoints_and_paths(self):
        text = (ROOT / "docs" / "operator_runbook.md").read_text()
        # Cross-check the runbook against URLs/paths the code actually exposes.
        for token in (
            "http://localhost:9000/operator/",
            "/healthz",
            "audience_display.html",
            "metrics/",
            "Romans 8:28",  # used as a test phrase in the dry-run
        ):
            assert token in text, f"runbook missing reference: {token}"


class TestBootstrap:
    def test_bootstrap_exists_and_executable(self):
        path = ROOT / "bootstrap.sh"
        assert path.exists()
        # On Unix, executable bit may not survive Windows checkouts, so just
        # check we have a shebang + bash compatibility.
        assert path.read_text().startswith("#!/usr/bin/env bash")

    def test_bootstrap_includes_required_steps(self):
        text = (ROOT / "bootstrap.sh").read_text()
        for step in (
            "venv",
            # v2026.7+ installs via pyproject extras (`pip install '.[cuda|cpu]'`).
            # Pre-v2026.7 used `pip install -r requirements-nvidia.txt`.
            ".[$EXTRA]",
            "systemctl",
            "/api/preflight",
            "ExecStart=",
            "operator_app.main:app",
        ):
            assert step in text, f"bootstrap.sh missing step: {step}"

    def test_bootstrap_supports_skip_systemd_flag(self):
        text = (ROOT / "bootstrap.sh").read_text()
        assert "--skip-systemd" in text

    def test_bootstrap_set_euo_pipefail(self):
        # safer-by-default scripting; catch typos and silent failures
        text = (ROOT / "bootstrap.sh").read_text()
        assert "set -euo pipefail" in text

    def test_bootstrap_syntax_valid(self):
        import shutil
        import subprocess

        if not shutil.which("bash"):
            return  # CI runner may not have bash; harmless skip
        result = subprocess.run(["bash", "-n", str(ROOT / "bootstrap.sh")], capture_output=True, text=True)
        assert result.returncode == 0, f"bash -n failed:\n{result.stderr}"
