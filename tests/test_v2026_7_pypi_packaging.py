"""Track 4 (v2026.7) — pypi wheel + uv install tests.

Covers:
- ``operator_app.cli`` argparse dispatch
- ``operator_app.setup`` model bootstrap (with mocked downloads)
- ``models.lock.json`` shape + SHA-256 of in-repo GGUFs
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).parent.parent


# -- pyproject + lockfile shape ----------------------------------------------


class TestPyprojectShape:
    def test_pyproject_has_required_fields(self):
        text = (ROOT / "pyproject.toml").read_text()
        for field in ('name = "stark-translate"', "[project.scripts]", "[project.optional-dependencies]"):
            assert field in text, f"pyproject.toml missing: {field}"

    def test_pyproject_extras_present(self):
        text = (ROOT / "pyproject.toml").read_text()
        for extra in ("cuda = [", "mlx = [", "cpu = [", "dev = ["):
            assert extra in text, f"missing extras block: {extra}"

    def test_pyproject_console_script_wires_cli(self):
        text = (ROOT / "pyproject.toml").read_text()
        assert 'stark-translate = "operator_app.cli:main"' in text

    def test_pyproject_build_system_hatchling(self):
        text = (ROOT / "pyproject.toml").read_text()
        assert "[build-system]" in text
        assert 'requires = ["hatchling' in text


class TestModelsLockShape:
    def test_lockfile_loads(self):
        data = json.loads((ROOT / "models.lock.json").read_text())
        assert data["version"]
        assert isinstance(data["models"], dict)
        assert len(data["models"]) >= 1

    def test_each_entry_has_required_fields(self):
        data = json.loads((ROOT / "models.lock.json").read_text())
        for key, entry in data["models"].items():
            assert "type" in entry, f"{key} missing type"
            assert entry["type"] in ("direct", "hf-snapshot"), f"{key} unknown type"
            assert "required_for" in entry, f"{key} missing required_for"
            if entry["type"] == "direct":
                assert "url" in entry
                assert "filename" in entry
                assert "sha256" in entry
            else:
                assert "repo_id" in entry
                assert "subdir" in entry

    def test_gguf_entries_match_local_files(self):
        """If we have the GGUFs locally, their SHA-256 should match the lockfile."""
        data = json.loads((ROOT / "models.lock.json").read_text())
        models_dir = ROOT / "models"
        for key, entry in data["models"].items():
            if entry["type"] != "direct":
                continue
            local = models_dir / entry["filename"]
            if not local.exists():
                pytest.skip(f"{local} not present; CI environments may not have GGUFs")
            h = hashlib.sha256()
            with local.open("rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            assert h.hexdigest() == entry["sha256"], f"SHA-256 mismatch for {key}"


# -- cli dispatch ------------------------------------------------------------


class TestCli:
    def test_no_args_prints_help_and_returns_2(self, capsys):
        from operator_app.cli import main

        rc = main([])
        captured = capsys.readouterr()
        assert rc == 2
        assert "operator" in captured.err
        assert "setup" in captured.err
        assert "doctor" in captured.err

    def test_version_prints_and_returns_0(self, capsys):
        from operator_app.cli import main

        rc = main(["version"])
        captured = capsys.readouterr()
        assert rc == 0
        # Should look like a CalVer or "0.0.0+dev"
        assert any(c.isdigit() for c in captured.out)

    def test_doctor_runs_preflight(self, capsys):
        from operator_app.cli import main

        rc = main(["doctor", "--json"])
        captured = capsys.readouterr()
        # Returns 0 if no fail, 1 if any fail. Either is valid here — we
        # just want to confirm it produced JSON output.
        assert rc in (0, 1)
        payload = json.loads(captured.out)
        assert "checks" in payload
        assert "ok" in payload

    def test_doctor_human_format(self, capsys):
        from operator_app.cli import main

        rc = main(["doctor"])
        captured = capsys.readouterr()
        assert rc in (0, 1)
        assert "Pre-flight:" in captured.out

    def test_operator_subcommand_dispatches(self, capsys):
        """Operator subcommand should attempt to call uvicorn.run.

        We mock uvicorn so the test doesn't actually start a server.
        """
        pytest.importorskip("uvicorn")
        from operator_app import cli

        with patch("uvicorn.run") as mock_run:
            rc = cli.main(["operator", "--no-browser", "--port", "9999"])
        assert rc == 0
        mock_run.assert_called_once()
        kwargs = mock_run.call_args.kwargs
        assert kwargs["port"] == 9999
        assert kwargs["host"] == "0.0.0.0"


# -- setup module ------------------------------------------------------------


class TestSetupModule:
    def test_default_models_dir_unix(self, monkeypatch):
        from operator_app.setup import default_models_dir

        monkeypatch.delenv("STARK_MODELS_DIR", raising=False)
        monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
        monkeypatch.setattr("platform.system", lambda: "Linux")

        result = default_models_dir()
        assert result.name == "models"
        assert "stark-translate" in str(result)

    def test_default_models_dir_windows(self, monkeypatch, tmp_path):
        from operator_app.setup import default_models_dir

        monkeypatch.delenv("STARK_MODELS_DIR", raising=False)
        monkeypatch.setattr("platform.system", lambda: "Windows")
        monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))

        result = default_models_dir()
        assert "stark-translate" in str(result)
        assert str(tmp_path) in str(result)

    def test_default_models_dir_env_override(self, monkeypatch, tmp_path):
        from operator_app.setup import default_models_dir

        monkeypatch.setenv("STARK_MODELS_DIR", str(tmp_path / "custom"))
        result = default_models_dir()
        assert result == tmp_path / "custom"

    def test_load_lockfile_finds_repo_root(self):
        from operator_app.setup import load_lockfile

        data = load_lockfile(project_root=ROOT)
        assert data["version"]
        assert data["models"]

    def test_bootstrap_skip_when_sidecar_matches(self, tmp_path, monkeypatch):
        """If a target file + .installed sidecar exist with matching SHA,
        bootstrap should skip the entry."""
        from operator_app.setup import bootstrap_models

        # Build a tiny lockfile in tmp
        (tmp_path / "models.lock.json").write_text(
            json.dumps(
                {
                    "version": "test",
                    "models": {
                        "fake": {
                            "type": "direct",
                            "url": "http://example.invalid/fake.bin",
                            "filename": "fake.bin",
                            "sha256": hashlib.sha256(b"hello").hexdigest(),
                            "size_bytes": 5,
                            "required_for": ["cuda"],
                        }
                    },
                }
            )
        )
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        (models_dir / "fake.bin").write_bytes(b"hello")
        sidecar_meta = {
            "lockfile_version": "test",
            "sha256": hashlib.sha256(b"hello").hexdigest(),
            "installed_at": "2026-04-25T00:00:00",
        }
        (models_dir / "fake.bin.installed").write_text(json.dumps(sidecar_meta))

        # urlopen should NOT be called — we expect the skip path.
        with patch("urllib.request.urlopen") as mock_open:
            rc = bootstrap_models(models_dir=models_dir, project_root=tmp_path)
        assert rc == 0
        mock_open.assert_not_called()

    def test_bootstrap_downloads_when_missing(self, tmp_path):
        """When the target file is missing, urlopen is called."""
        from operator_app import setup

        payload = b"hello world"
        sha = hashlib.sha256(payload).hexdigest()
        (tmp_path / "models.lock.json").write_text(
            json.dumps(
                {
                    "version": "test",
                    "models": {
                        "fake": {
                            "type": "direct",
                            "url": "http://example.invalid/fake.bin",
                            "filename": "fake.bin",
                            "sha256": sha,
                            "size_bytes": len(payload),
                            "required_for": ["cuda"],
                        }
                    },
                }
            )
        )

        models_dir = tmp_path / "models"

        # Replace _download_direct with a stub that just writes the bytes.
        def fake_download(url, target, expected_size):
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)

        with patch("operator_app.setup._download_direct", side_effect=fake_download):
            rc = setup.bootstrap_models(models_dir=models_dir, project_root=tmp_path)

        assert rc == 0
        assert (models_dir / "fake.bin").exists()
        assert (models_dir / "fake.bin.installed").exists()
        sidecar = json.loads((models_dir / "fake.bin.installed").read_text())
        assert sidecar["sha256"] == sha

    def test_bootstrap_fails_on_sha_mismatch(self, tmp_path):
        from operator_app import setup

        (tmp_path / "models.lock.json").write_text(
            json.dumps(
                {
                    "version": "test",
                    "models": {
                        "fake": {
                            "type": "direct",
                            "url": "http://example.invalid/fake.bin",
                            "filename": "fake.bin",
                            "sha256": "0" * 64,  # never going to match
                            "size_bytes": 5,
                            "required_for": ["cuda"],
                        }
                    },
                }
            )
        )

        def write_wrong_bytes(url, target, expected_size):
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"wrong")

        with patch("operator_app.setup._download_direct", side_effect=write_wrong_bytes):
            rc = setup.bootstrap_models(models_dir=tmp_path / "models", project_root=tmp_path)

        assert rc == 1  # SHA mismatch → failure
