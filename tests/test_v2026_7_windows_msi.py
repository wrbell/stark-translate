"""Track 3 (v2026.7) — Windows MSI scaffold tests.

These are shape-only tests. The actual MSI is built on a Windows runner via
``release-win.yml``; CI here just verifies the scaffold files are present
and have the expected structure.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).parent.parent


class TestPyAppConfig:
    def test_present(self):
        assert (ROOT / "packaging" / "windows" / "pyapp-config.toml").exists()

    def test_pinned_python_version(self):
        text = (ROOT / "packaging" / "windows" / "pyapp-config.toml").read_text()
        assert "3.12" in text, "PyApp must pin Python 3.12 for cold-start parity with PyPI build"

    def test_extras_detection_via_env_var(self):
        text = (ROOT / "packaging" / "windows" / "pyapp-config.toml").read_text()
        assert "STARK_INSTALL_EXTRAS" in text

    def test_exec_spec_matches_cli(self):
        text = (ROOT / "packaging" / "windows" / "pyapp-config.toml").read_text()
        assert "operator_app.cli:main" in text


class TestWixFragment:
    def test_present(self):
        assert (ROOT / "packaging" / "windows" / "wix-fragment.wxi").exists()

    def test_start_menu_shortcut(self):
        text = (ROOT / "packaging" / "windows" / "wix-fragment.wxi").read_text()
        assert "Stark Translate" in text
        assert "ProgramMenuFolder" in text

    def test_nvidia_detection(self):
        text = (ROOT / "packaging" / "windows" / "wix-fragment.wxi").read_text()
        assert "Win32_VideoController" in text
        assert "STARK_INSTALL_EXTRAS" in text
        # Both possible outcomes must be set so PyApp always sees the var
        assert "[cuda]" in text and "[cpu]" in text

    def test_arp_metadata_links_to_repo(self):
        text = (ROOT / "packaging" / "windows" / "wix-fragment.wxi").read_text()
        assert "github.com/wrbell/stark-translate" in text


class TestReleaseWorkflow:
    def test_present(self):
        assert (ROOT / ".github" / "workflows" / "release-win.yml").exists()

    def test_runs_on_windows(self):
        text = (ROOT / ".github" / "workflows" / "release-win.yml").read_text()
        assert "windows-2022" in text or "windows-latest" in text

    def test_builds_pyapp_with_rust_toolchain(self):
        text = (ROOT / ".github" / "workflows" / "release-win.yml").read_text()
        assert "dtolnay/rust-toolchain" in text or "rust-toolchain" in text
        assert "cargo install pyapp" in text or "pyapp" in text.lower()

    def test_briefcase_package_unsigned(self):
        text = (ROOT / ".github" / "workflows" / "release-win.yml").read_text()
        assert "briefcase package windows" in text
        # v2026.7.2.0 ships unsigned; signing is the v2026.7.2.1 follow-up
        assert "--no-sign" in text

    def test_uploads_msi_artifact(self):
        text = (ROOT / ".github" / "workflows" / "release-win.yml").read_text()
        assert "upload-artifact" in text
        assert "*.msi" in text or ".msi" in text

    def test_triggers_on_version_tags(self):
        text = (ROOT / ".github" / "workflows" / "release-win.yml").read_text()
        assert 'tags:\n      - "v*"' in text or "tags:\n      - 'v*'" in text
