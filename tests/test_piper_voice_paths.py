"""Keep a local Piper ONNX/config pair together across Hugging Face symlinks.

Every weight/config file is a tiny fixture; Piper and downloads are mocked.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from engines import model_paths


@pytest.fixture
def snapshot_voice(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(model_paths, "load_model_manifest", lambda *args: {"models": {}})
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    repo = tmp_path / "models--fixture--piper-voices"
    blobs = repo / "blobs"
    blobs.mkdir(parents=True)
    onnx_blob = blobs / ("a" * 64)
    config_blob = blobs / ("b" * 64)
    onnx_blob.write_bytes(b"mock ONNX fixture, never loaded")
    config_blob.write_text(json.dumps({"audio": {"sample_rate": 22050}, "fixture": "correct voice"}))
    voice = repo / "snapshots" / ("c" * 40) / "en/en_US/fixture/high/en_US-fixture-high.onnx"
    voice.parent.mkdir(parents=True)
    voice.symlink_to(os.path.relpath(onnx_blob, voice.parent))
    config = Path(f"{voice}.json")
    config.symlink_to(os.path.relpath(config_blob, config.parent))
    assert not Path(f"{onnx_blob}.json").exists()
    return voice, onnx_blob


@pytest.mark.parametrize("relative", [False, True])
def test_explicit_snapshot_voice_keeps_adjacent_config(snapshot_voice, tmp_path, relative):
    voice, blob = snapshot_voice
    configured = str(voice.relative_to(tmp_path) if relative else voice)

    resolved = model_paths.resolve_piper_voice(configured)

    assert resolved == str(voice)
    assert Path(resolved).resolve() == blob
    # This is the actual Piper default: append .json to the supplied model path.
    assert json.loads(Path(f"{resolved}.json").read_text())["fixture"] == "correct voice"


@pytest.mark.parametrize("misleading_blob_config", [False, True])
def test_engine_passes_the_configured_pair_to_piper_without_download(
    snapshot_voice, monkeypatch, misleading_blob_config
):
    from engines.tts_engine import PiperTTSEngine

    voice, blob = snapshot_voice
    if misleading_blob_config:
        Path(f"{blob}.json").write_text(json.dumps({"fixture": "wrong adjacent blob config"}))
    loaded_voice = SimpleNamespace()

    def load(model_path):
        assert json.loads(Path(f"{model_path}.json").read_text())["fixture"] == "correct voice"
        return loaded_voice

    loader = Mock(side_effect=load)
    download = Mock(side_effect=AssertionError("Explicit local voice attempted a download"))
    monkeypatch.setitem(sys.modules, "piper.voice", SimpleNamespace(PiperVoice=SimpleNamespace(load=loader)))
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(hf_hub_download=download))
    engine = PiperTTSEngine(voices={"en": str(voice)})

    engine.load()

    loader.assert_called_once_with(str(voice))
    download.assert_not_called()
    assert engine._voices["en"] is loaded_voice


@pytest.mark.parametrize("missing", ["model", "config"])
def test_broken_snapshot_pair_is_not_resolved(snapshot_voice, missing):
    voice, blob = snapshot_voice
    (blob if missing == "model" else Path(f"{voice}.json")).unlink()

    assert model_paths.resolve_piper_voice(str(voice)) is None


def test_explicit_parent_directory_symlink_keeps_configured_path(snapshot_voice, tmp_path):
    voice, _ = snapshot_voice
    directory = tmp_path / "configured_voice_directory"
    directory.symlink_to(voice.parent, target_is_directory=True)
    configured = directory / voice.name

    resolved = model_paths.resolve_piper_voice(str(configured))

    assert resolved == str(configured)
    assert json.loads(Path(f"{resolved}.json").read_text())["fixture"] == "correct voice"
