"""Managed Marian setup/lookup contracts without inference or downloads."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from engines.model_paths import MARIAN_CT2_REQUIRED_FILES, resolve_marian_ct2
from operator_app import setup
from tools import marian_ct2_setup as managed


def complete(path):
    path.mkdir(parents=True, exist_ok=True)
    for name in MARIAN_CT2_REQUIRED_FILES:
        (path / name).write_bytes(b"{}")
    return path


def manifest(root):
    models = {}
    for direction in ("en-es", "es-en"):
        models[f"marian-{direction}"] = {"type": "hf-snapshot", "repo_id": f"org/{direction}", "revision": "a" * 40}
        models[f"marian-ct2-{direction}"] = {
            "type": "derived-ct2",
            "source_model": f"marian-{direction}",
            "direction": direction,
            "quantization": "int8",
            "subdir": f"marian-ct2/{direction}",
            "required_for": ["mlx"],
        }
    result = {"version": "test", "models": models}
    (root / "models.lock.json").write_text(json.dumps(result))
    return result


def fake_conversion(source, target, quantization, direction):
    complete(target)
    return {"command": [sys.executable, "mocked-converter"], "smoke": {"device": "cpu", "nonempty": True}}


def prepare_sources(tmp_path, monkeypatch):
    data = manifest(tmp_path)
    monkeypatch.setattr(managed.importlib.metadata, "version", lambda name: "4.8.2")
    monkeypatch.setattr(
        managed,
        "resolve_hf_source",
        lambda model_id, **kwargs: (
            tmp_path / "pinned-source",
            {"model_id": model_id, "source_revision": "a" * 40, "source_path": str(tmp_path / "pinned-source")},
        ),
    )
    monkeypatch.setattr(managed, "convert_source", fake_conversion)
    return data


def test_explicit_then_adapter_then_managed_and_both_directions(tmp_path, monkeypatch):
    data = prepare_sources(tmp_path, monkeypatch)
    cache = tmp_path / "cache"
    for direction in ("en-es", "es-en"):
        artifact, created = managed.ensure_managed_marian(
            data["models"][f"marian-ct2-{direction}"], project_root=tmp_path, models_dir=cache
        )
        assert created
        assert resolve_marian_ct2(direction, project_root=tmp_path, models_dir=cache) == str(artifact)
        adapter = complete(tmp_path / "adapters" / "marian_ct2" / direction / "active")
        assert resolve_marian_ct2(direction, project_root=tmp_path, models_dir=cache) == str(adapter)
        override = complete(tmp_path / ("explicit-" + direction))
        assert resolve_marian_ct2(direction, explicit_path=override, project_root=tmp_path, models_dir=cache) == str(
            override
        )
        with pytest.raises(ValueError, match="incomplete"):
            resolve_marian_ct2(direction, explicit_path=tmp_path / "missing", project_root=tmp_path, models_dir=cache)


def test_setup_preserves_existing_adapters_even_on_refresh(tmp_path, monkeypatch):
    manifest(tmp_path)
    before = {}
    for direction in ("en-es", "es-en"):
        path = complete(tmp_path / "adapters" / "marian_ct2" / direction / "active")
        before[str(path)] = (path / "model.bin").read_bytes()
    conversion = Mock(side_effect=AssertionError("Working adapters must not be rebuilt"))
    monkeypatch.setattr(managed, "convert_source", conversion)
    assert (
        setup.bootstrap_models(project_root=tmp_path, models_dir=tmp_path / "cache", backend="mlx", refresh=True) == 0
    )
    conversion.assert_not_called()
    assert all((Path(path) / "model.bin").read_bytes() == value for path, value in before.items())


def test_failed_refresh_keeps_previous_pointer_and_artifact(tmp_path, monkeypatch):
    data = prepare_sources(tmp_path, monkeypatch)
    entry = data["models"]["marian-ct2-en-es"]
    cache = tmp_path / "cache"
    first, _ = managed.ensure_managed_marian(entry, project_root=tmp_path, models_dir=cache)
    pointer = cache / entry["subdir"] / "active.json"
    original_pointer = pointer.read_bytes()
    monkeypatch.setattr(managed, "convert_source", Mock(side_effect=RuntimeError("conversion failed")))
    with pytest.raises(RuntimeError, match="conversion failed"):
        managed.ensure_managed_marian(entry, project_root=tmp_path, models_dir=cache, refresh=True)
    assert pointer.read_bytes() == original_pointer
    assert first.is_dir()
    assert resolve_marian_ct2("en-es", project_root=tmp_path, models_dir=cache) == str(first)
    assert not list(pointer.parent.glob(".staging-*"))


def test_managed_manifest_revision_mismatch_is_not_ready(tmp_path, monkeypatch):
    data = prepare_sources(tmp_path, monkeypatch)
    entry = data["models"]["marian-ct2-en-es"]
    cache = tmp_path / "cache"
    managed.ensure_managed_marian(entry, project_root=tmp_path, models_dir=cache)
    data["models"]["marian-en-es"]["revision"] = "b" * 40
    (tmp_path / "models.lock.json").write_text(json.dumps(data))
    assert resolve_marian_ct2("en-es", project_root=tmp_path, models_dir=cache) is None


def test_converter_uses_selected_python_and_offline_local_source(tmp_path, monkeypatch):
    source, target = complete(tmp_path / "source"), complete(tmp_path / "target")
    run = Mock(side_effect=[SimpleNamespace(returncode=0), SimpleNamespace(stdout='{"nonempty":true}\n')])
    monkeypatch.setattr(managed.subprocess, "run", run)
    managed.convert_source(source, target, "int8", "en-es")
    first = run.call_args_list[0]
    assert first.args[0][:3] == [sys.executable, "-m", "ctranslate2.converters.transformers"]
    assert str(source) in first.args[0]
    assert first.kwargs["env"]["HF_HUB_OFFLINE"] == "1"
    assert run.call_args_list[1].args[0][0] == sys.executable


def test_unversioned_local_candidate_does_not_gain_a_pinned_revision(tmp_path, monkeypatch):
    manifest(tmp_path)
    unversioned = complete(tmp_path / "unversioned")
    snapshot = complete(tmp_path / "snapshots" / ("a" * 40))
    monkeypatch.setattr(managed, "resolve_model_path", lambda *args, **kwargs: str(unversioned))
    download = Mock(return_value=str(snapshot))
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=download))
    resolved, provenance = managed.resolve_hf_source("org/en-es", project_root=tmp_path)
    assert resolved == snapshot
    assert provenance["source_revision"] == "a" * 40
    download.assert_called_once_with(repo_id="org/en-es", revision="a" * 40)


def test_pinned_snapshot_reuse_never_downloads(tmp_path, monkeypatch):
    manifest(tmp_path)
    snapshot = complete(tmp_path / "snapshots" / ("a" * 40))
    monkeypatch.setattr(managed, "resolve_model_path", lambda *args, **kwargs: str(snapshot))
    download = Mock(side_effect=AssertionError("network"))
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=download))
    assert managed.resolve_hf_source("org/en-es", project_root=tmp_path)[0] == snapshot
    download.assert_not_called()
