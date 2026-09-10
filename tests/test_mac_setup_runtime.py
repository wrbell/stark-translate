"""Offline setup, preflight and packaging contracts; no ML model loads."""

from __future__ import annotations

import json
import os
import plistlib
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from engines.model_paths import resolve_model_path
from operator_app import preflight, setup
from operator_app.launchd import launchd_plist, manage_launchd
from tools.release_artifacts import build_mac_bundle, validate_version, verify_artifact

ROOT = Path(__file__).resolve().parent.parent


def _manifest(root, revision="a" * 40):
    entry = {
        "type": "hf-snapshot",
        "repo_id": "test/model",
        "revision": revision,
        "subdir": "selected",
        "required_for": ["mlx"],
    }
    (root / "models.lock.json").write_text(json.dumps({"version": "test", "models": {"selected": entry}}))
    return entry


def _weights(root):
    root.mkdir(parents=True)
    (root / "config.json").write_text("{}")
    (root / "model.safetensors").write_bytes(b"weights")


def test_setup_cache_is_used_before_hf_cache(tmp_path, monkeypatch):
    _manifest(tmp_path)
    cache = tmp_path / "custom"
    _weights(cache / "selected")
    monkeypatch.setenv("STARK_MODELS_DIR", str(cache))
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hf"))
    assert resolve_model_path("test/model", project_root=tmp_path, local_only=True) == str(cache / "selected")


def test_incomplete_snapshot_fails_offline_lookup(tmp_path, monkeypatch):
    _manifest(tmp_path)
    cache = tmp_path / "custom"
    (cache / "selected").mkdir(parents=True)
    (cache / "selected" / "config.json").write_text("{}")
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hf"))
    assert resolve_model_path("test/model", models_dir=cache, project_root=tmp_path, local_only=True) is None
    assert resolve_model_path("test/model", models_dir=cache, project_root=tmp_path) == "test/model"


@pytest.mark.parametrize(
    "marker", [{"repo_id": "wrong", "revision": "a" * 40}, {"repo_id": "test/model", "revision": "old"}, []]
)
def test_managed_snapshot_mismatch_falls_back_to_pinned_cache(tmp_path, monkeypatch, marker):
    entry = _manifest(tmp_path)
    cache = tmp_path / "custom"
    _weights(cache / "selected")
    (cache / "selected" / ".installed").write_text(json.dumps(marker))
    hub = tmp_path / "hf"
    snapshot = hub / "models--test--model" / "snapshots" / entry["revision"]
    _weights(snapshot)
    monkeypatch.setenv("HF_HUB_CACHE", str(hub))
    assert resolve_model_path("test/model", models_dir=cache, project_root=tmp_path, local_only=True) == str(snapshot)
    # An explicit user path remains a deliberate override even with an old marker.
    assert resolve_model_path(str(cache / "selected"), project_root=tmp_path, local_only=True) == str(
        cache / "selected"
    )


def test_matching_managed_snapshot_remains_first_choice(tmp_path, monkeypatch):
    entry = _manifest(tmp_path)
    cache = tmp_path / "custom"
    _weights(cache / "selected")
    (cache / "selected" / ".installed").write_text(
        json.dumps({"repo_id": entry["repo_id"], "revision": entry["revision"]})
    )
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "empty-hf"))
    assert resolve_model_path("test/model", models_dir=cache, project_root=tmp_path, local_only=True) == str(
        cache / "selected"
    )


def test_default_tts_factory_only_uses_setup_languages(monkeypatch):
    from types import SimpleNamespace

    from engines.factory import create_tts_engine
    from settings import settings

    monkeypatch.setitem(sys.modules, "engines.mlx_engine", SimpleNamespace(PiperTTSEngine=lambda **kwargs: kwargs))
    monkeypatch.setattr(
        settings.tts, "voices", {"en": "english", "es": "spanish", "hi": "future-hindi", "zh": "future-chinese"}
    )
    assert create_tts_engine() == {"voices": {"en": "english", "es": "spanish"}}
    assert create_tts_engine({"hi": "custom-hindi"}) == {"voices": {"hi": "custom-hindi"}}


def test_setup_only_downloads_selected_backend(tmp_path):
    _manifest(tmp_path)
    manifest = json.loads((tmp_path / "models.lock.json").read_text())
    manifest["models"]["cuda-only"] = {**manifest["models"]["selected"], "required_for": ["cuda"]}
    (tmp_path / "models.lock.json").write_text(json.dumps(manifest))
    cache = tmp_path / "cache"

    def download(**kwargs):
        _weights(kwargs["target_dir"])

    with patch.object(setup, "_download_hf_snapshot", side_effect=download) as get:
        assert setup.bootstrap_models(cache, project_root=tmp_path, backend="mlx") == 0
    assert get.call_count == 1
    assert (cache / "selected" / ".installed").is_file()


def test_changed_revision_refreshes_snapshot(tmp_path):
    entry = _manifest(tmp_path)
    cache = tmp_path / "cache"
    _weights(cache / "selected")
    (cache / "selected" / ".installed").write_text(
        json.dumps({"repo_id": entry["repo_id"], "revision": "old", "lockfile_version": "test"})
    )
    with patch.object(setup, "_download_hf_snapshot") as get:
        assert setup.bootstrap_models(cache, project_root=tmp_path, backend="mlx") == 0
    get.assert_called_once()


def test_mlx_preflight_never_probes_llama(tmp_path):
    with (
        patch.object(preflight, "check_gpu", return_value=preflight._check("GPU", "pass", "Mac")),
        patch.object(preflight, "check_dependencies", return_value=preflight._check("deps", "pass", "ok")),
        patch.object(preflight, "check_models", return_value=preflight._check("models", "pass", "ok")),
        patch.object(preflight, "check_microphone", return_value=preflight._check("mic", "pass", "ok")),
        patch.object(preflight, "check_llamacpp_server") as llama,
    ):
        payload = preflight.run_all_checks(tmp_path, backend="mlx")
    assert payload["ok"]
    llama.assert_not_called()


def test_mac_model_selection_changes_with_language(tmp_path):
    with patch.object(preflight, "resolve_model_path", return_value="/cached") as resolve:
        assert preflight.check_models(tmp_path, backend="mlx", lang="es")["status"] == "pass"
    requested = [call.args[0] for call in resolve.call_args_list]
    assert "mlx-whisper-large-v3-turbo" in requested
    assert "mlx-parakeet-v3" not in requested
    assert "marian-es-en" in requested


def test_missing_selected_model_blocks_start(tmp_path):
    with patch.object(preflight, "resolve_model_path", return_value=None):
        result = preflight.check_models(tmp_path, backend="mlx")
    assert result["status"] == "fail"
    assert "mlx-parakeet-v3" in result["detail"]
    assert "GGUF" not in result["detail"]


def test_launchd_render_uses_actual_paths_and_preserves_venv(tmp_path):
    python = tmp_path / "my env" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.symlink_to(sys.executable)
    config = launchd_plist(tmp_path, python)
    assert config["ProgramArguments"][0] == str(python)
    assert config["WorkingDirectory"] == str(tmp_path)
    target = tmp_path / "agent.plist"
    assert manage_launchd("render", project_root=tmp_path, python=python, output=target) == 0
    assert plistlib.loads(target.read_bytes())["ProgramArguments"][0] == str(python)


def test_launchd_install_and_uninstall_are_explicit(tmp_path, monkeypatch):
    monkeypatch.setattr("operator_app.launchd.platform.system", lambda: "Darwin")
    monkeypatch.setenv("STARK_OPERATOR_LOG_DIR", str(tmp_path / "logs"))
    target = tmp_path / "agents" / "service.plist"
    with patch("operator_app.launchd.subprocess.run") as launchctl:
        manage_launchd("install", project_root=tmp_path, python=Path(sys.executable), output=target)
        assert target.is_file()
        assert launchctl.call_args.args[0][:2] == ["launchctl", "bootstrap"]
        manage_launchd("uninstall", project_root=tmp_path, output=target)
        assert not target.exists()
        assert launchctl.call_args.args[0][:2] == ["launchctl", "bootout"]


def test_missing_dependency_blocks_preflight_without_importing_models():
    from importlib.metadata import PackageNotFoundError

    with patch("operator_app.preflight.importlib.metadata.version", side_effect=PackageNotFoundError):
        result = preflight.check_dependencies("mlx", lang="en")
    assert result["status"] == "fail"
    assert "parakeet-mlx" in result["detail"]
    assert "stark-translate[mlx]" in result["detail"]


@pytest.mark.parametrize(
    "explicit,active,expected", [(True, True, "explicit"), (False, True, "active"), (False, False, "stt_env")]
)
def test_launch_environment_precedence(tmp_path, explicit, active, expected):
    env = {k: v for k, v in os.environ.items() if k not in {"STARK_PYTHON", "VENV", "VIRTUAL_ENV"}}
    for name in ["explicit", "active", "stt_env", "venv"]:
        binary = tmp_path / name / "bin" / "python"
        binary.parent.mkdir(parents=True)
        binary.symlink_to(sys.executable)
    if explicit:
        env["STARK_PYTHON"] = str(tmp_path / "explicit" / "bin" / "python")
    if active:
        env["VIRTUAL_ENV"] = str(tmp_path / "active")
    result = subprocess.check_output(
        [
            "bash",
            "-c",
            'source "$1"; stark_resolve_python "$2"',
            "test",
            str(ROOT / "scripts/runtime_env.sh"),
            str(tmp_path),
        ],
        env=env,
        text=True,
    )
    assert result.strip() == str(tmp_path / expected / "bin" / "python")


def test_release_rejects_version_mismatch():
    with pytest.raises(ValueError, match="does not match"):
        validate_version(ROOT, "v0.0.0")


def test_mac_bundle_contains_installable_runtime(tmp_path):
    bundle = build_mac_bundle(ROOT, tmp_path, f"v{validate_version(ROOT)}")
    verify_artifact(bundle)


def test_source_cli_version_does_not_report_an_old_installed_distribution():
    from operator_app.cli import _resolve_version

    with patch("operator_app.cli.metadata.version", return_value="2026.7.0.0"):
        assert _resolve_version() == validate_version(ROOT)
