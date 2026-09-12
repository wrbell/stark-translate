"""Documentation contract tests — backlog, links, guide navigation.

These tests deliberately avoid asserting changing numbers (test counts,
coverage, latency). They check structure, canonical references and that the
guides do not carry known-stale claims.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
BACKLOG_JSON = ROOT / "docs" / "backlog.json"
BACKLOG_MD = ROOT / "docs" / "backlog.md"
RENDER_SCRIPT = ROOT / "tools" / "render_backlog.py"
PYPROJECT = ROOT / "pyproject.toml"
WORKFLOWS_DIR = ROOT / ".github" / "workflows"

GUIDE_PAIRS = [
    ("engines/AGENTS.md", "engines/CLAUDE.md"),
    ("training/AGENTS.md", "training/CLAUDE.md"),
    ("tools/AGENTS.md", "tools/CLAUDE.md"),
    ("displays/AGENTS.md", "displays/CLAUDE.md"),
    ("features/AGENTS.md", "features/CLAUDE.md"),
]

ROOT_GUIDES = [
    "README.md",
    "CLAUDE.md",
    "AGENTS.md",
    "CLAUDE-macbook.md",
    "CLAUDE-windows.md",
]

CANONICAL_DOCS = (
    ROOT_GUIDES
    + [
        "docs/roadmap.md",
        "docs/backlog.md",
        "docs/current_architecture.md",
        "docs/overnight_status.md",
        "docs/README.md",
        "docs/mac_implementation_status.md",
    ]
    + [pair[0] for pair in GUIDE_PAIRS]
    + [pair[1] for pair in GUIDE_PAIRS]
)

# Claims that source review showed to be stale. Each entry is (regex, reason).
STALE_CLAIMS = [
    (
        r"Seven (CI workflows|GitHub Actions)",
        "there are more than seven workflow files; count them from .github/workflows",
    ),
    (r"scale ~linearly with memory bandwidth", "unsupported hardware-scaling extrapolation"),
    (r"not yet integrated with (the )?live pipeline", "verses/summary/diarization are wired into the operator app"),
    (r"5-canary", "health_check.py uses the 8-canary set"),
    (r"Codex-(macbook|windows)\.md", "the platform guides are CLAUDE-macbook.md / CLAUDE-windows.md"),
]


def _run_render_backlog(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(RENDER_SCRIPT), *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def _load_backlog() -> dict:
    return json.loads(BACKLOG_JSON.read_text(encoding="utf-8"))


class TestBacklogJson:
    def test_backlog_exists_and_parses(self):
        assert BACKLOG_JSON.exists()
        data = _load_backlog()
        assert data["schema_version"] == 2
        assert len(data["items"]) >= 20

    def test_backlog_validate_script_passes(self):
        result = _run_render_backlog("validate")
        assert result.returncode == 0, result.stderr

    def test_backlog_md_matches_json(self):
        result = _run_render_backlog("render", "--check")
        assert result.returncode == 0, result.stderr

    def test_integration_version_matches_pyproject(self):
        local_version = _load_backlog()["integration"]["local_version"]
        pyproject_text = PYPROJECT.read_text(encoding="utf-8")
        assert f'version = "{local_version}"' in pyproject_text

    def test_status_and_certification_values_are_canonical(self):
        sys.path.insert(0, str(ROOT / "tools"))
        try:
            import render_backlog
        finally:
            sys.path.pop(0)
        data = _load_backlog()
        statuses = {item["status"] for item in data["items"]}
        certifications = {item["certification"] for item in data["items"]}
        assert statuses <= render_backlog.ALLOWED_STATUS
        assert certifications <= render_backlog.ALLOWED_CERTIFICATION
        assert set(data["status_definitions"]) == render_backlog.ALLOWED_STATUS

    def test_in_progress_items_never_claim_met_acceptance(self):
        for item in _load_backlog()["items"]:
            if item["status"] in {"in_progress", "pending_input_or_hardware"}:
                assert item["certification"] != "met", item["id"]

    def test_pr_status_has_consistent_merge_evidence(self):
        integration = _load_backlog()["integration"]
        records = [integration]
        if "previous_integration" in integration:
            records.append(integration["previous_integration"])
        for record in records:
            assert re.search(r"https://github\.com/wrbell/stark-translate/pull/[1-9]\d*", record["draft_pr"])
            state = record["pr_state"]
            assert state in {"open", "merged", "closed"}
            if record.get("is_draft"):
                assert state == "open"
            if state == "merged":
                assert re.fullmatch(r"[0-9a-f]{40}", record["merge_commit"])
                assert "not merged" not in record["draft_pr"].lower()
                assert "merged" in record["draft_pr"].lower()
            else:
                assert "not merged" in record["draft_pr"].lower()
                assert not record.get("merge_commit")

    def test_known_issue_items_carry_issue_acceptance(self):
        items = {item["id"]: item for item in _load_backlog()["items"]}
        for item_id in (
            "issue-131-smoke",
            "issue-132-tts-routing",
            "issue-133-diarize-gate",
            "issue-134-sunday-dry-run",
            "issue-137-active-learning",
            "issue-138-hindi-zero-shot",
            "issue-176-multiprocess",
        ):
            assert item_id in items, item_id
            assert items[item_id].get("issue_acceptance"), f"{item_id} must quote the issue acceptance"


class TestGuideNavigation:
    @pytest.mark.parametrize("agents_path,claude_path", GUIDE_PAIRS)
    def test_subdirectory_guide_pairs_exist(self, agents_path: str, claude_path: str):
        agents = ROOT / agents_path
        claude = ROOT / claude_path
        assert agents.exists(), agents_path
        assert claude.exists(), claude_path
        agents_text = agents.read_text(encoding="utf-8")
        claude_text = claude.read_text(encoding="utf-8")
        assert "CLAUDE.md" in agents_text
        assert "Paired with" in agents_text
        assert "AGENTS.md" in claude_text, f"{claude_path} should point back to its AGENTS.md"

    def test_root_agents_points_to_real_mac_windows_guides(self):
        text = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
        assert "CLAUDE-macbook.md" in text
        assert "CLAUDE-windows.md" in text

    def test_root_claude_and_agents_reference_backlog(self):
        for name in ("CLAUDE.md", "AGENTS.md"):
            text = (ROOT / name).read_text(encoding="utf-8")
            assert "docs/backlog.json" in text
            assert "docs/current_architecture.md" in text
            assert "v2026.14" in text

    def test_ci_workflow_count_matches_guides(self):
        workflow_count = len(sorted(WORKFLOWS_DIR.glob("*.yml")))
        assert workflow_count > 0
        pattern = re.compile(r"(\d+) GitHub Actions workflow files")
        for name in ("CLAUDE.md", "AGENTS.md", "README.md", "docs/current_architecture.md"):
            text = (ROOT / name).read_text(encoding="utf-8")
            match = pattern.search(text)
            assert match, f"{name} must state the workflow-file count as 'N GitHub Actions workflow files'"
            assert int(match.group(1)) == workflow_count, f"{name} says {match.group(1)}, found {workflow_count}"

    @pytest.mark.parametrize("doc_path", CANONICAL_DOCS)
    def test_guides_do_not_carry_known_stale_claims(self, doc_path: str):
        text = (ROOT / doc_path).read_text(encoding="utf-8")
        for pattern, reason in STALE_CLAIMS:
            assert re.search(pattern, text) is None, f"{doc_path}: stale claim {pattern!r} — {reason}"


class TestLocalLinks:
    def test_canonical_docs_local_links_resolve(self):
        result = _run_render_backlog("check-links", *CANONICAL_DOCS)
        assert result.returncode == 0, result.stderr or result.stdout

    def test_readme_documents_backlog_and_architecture(self):
        text = (ROOT / "README.md").read_text(encoding="utf-8")
        assert "docs/backlog.json" in text
        assert "docs/current_architecture.md" in text
