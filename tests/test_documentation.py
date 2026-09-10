"""Documentation contract tests — backlog, links, guide navigation."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
BACKLOG_JSON = ROOT / "docs" / "backlog.json"
BACKLOG_MD = ROOT / "docs" / "backlog.md"
RENDER_SCRIPT = ROOT / "tools" / "render_backlog.py"
PYPROJECT = ROOT / "pyproject.toml"

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

CANONICAL_DOCS = ROOT_GUIDES + [
    "docs/roadmap.md",
    "docs/backlog.md",
    "docs/backlog.json",
    "docs/current_architecture.md",
    "docs/overnight_status.md",
    "docs/mac_implementation_status.md",
] + [pair[0] for pair in GUIDE_PAIRS] + [pair[1] for pair in GUIDE_PAIRS]


def _run_render_backlog(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(RENDER_SCRIPT), *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


class TestBacklogJson:
    def test_backlog_exists_and_parses(self):
        assert BACKLOG_JSON.exists()
        data = json.loads(BACKLOG_JSON.read_text(encoding="utf-8"))
        assert data["schema_version"] == 1
        assert len(data["items"]) >= 20

    def test_backlog_validate_script_passes(self):
        result = _run_render_backlog("validate")
        assert result.returncode == 0, result.stderr

    def test_backlog_md_matches_json(self):
        result = _run_render_backlog("render", "--check")
        assert result.returncode == 0, result.stderr

    def test_integration_version_matches_pyproject(self):
        backlog = json.loads(BACKLOG_JSON.read_text(encoding="utf-8"))
        local_version = backlog["integration"]["local_version"]
        pyproject_text = PYPROJECT.read_text(encoding="utf-8")
        assert f'version = "{local_version}"' in pyproject_text

    def test_status_values_are_canonical(self):
        allowed = {
            "implemented",
            "validated",
            "experimental",
            "pending_input_or_hardware",
            "deferred",
        }
        backlog = json.loads(BACKLOG_JSON.read_text(encoding="utf-8"))
        statuses = {item["status"] for item in backlog["items"]}
        assert statuses <= allowed


class TestGuideNavigation:
    @pytest.mark.parametrize("agents_path,claude_path", GUIDE_PAIRS)
    def test_subdirectory_guide_pairs_exist(self, agents_path: str, claude_path: str):
        agents = ROOT / agents_path
        claude = ROOT / claude_path
        assert agents.exists(), agents_path
        assert claude.exists(), claude_path
        agents_text = agents.read_text(encoding="utf-8")
        assert "CLAUDE.md" in agents_text
        assert "Paired with" in agents_text

    def test_root_agents_points_to_real_mac_windows_guides(self):
        text = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
        assert "CLAUDE-macbook.md" in text
        assert "CLAUDE-windows.md" in text
        assert "Codex-macbook.md" not in text
        assert "Codex-windows.md" not in text

    def test_root_claude_and_agents_reference_backlog(self):
        for name in ("CLAUDE.md", "AGENTS.md"):
            text = (ROOT / name).read_text(encoding="utf-8")
            assert "docs/backlog.json" in text
            assert "docs/current_architecture.md" in text
            assert "v2026.14 candidate" in text.lower() or "v2026.14" in text


class TestLocalLinks:
    def test_canonical_docs_local_links_resolve(self):
        result = _run_render_backlog("check-links", *CANONICAL_DOCS)
        assert result.returncode == 0, result.stderr or result.stdout

    def test_readme_documents_backlog_and_architecture(self):
        text = (ROOT / "README.md").read_text(encoding="utf-8")
        assert "docs/backlog.json" in text
        assert "docs/current_architecture.md" in text
