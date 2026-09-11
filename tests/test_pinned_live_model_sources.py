"""Live fallback downloads must use registered commits; all loaders are mocked."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from engines import model_paths
from engines.tts_engine import PiperTTSEngine


@pytest.fixture
def isolated_models(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("STARK_MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("STARK_PROJECT_ROOT", str(tmp_path / "project"))
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hf"))
    monkeypatch.setenv("STARK_PROFILE", "standard")
    return tmp_path


def _voice_pair(model):
    model.parent.mkdir(parents=True, exist_ok=True)
    model.write_bytes(b"mock ONNX")
    model.with_suffix(".onnx.json").write_text("{}")
    return str(model)


@pytest.mark.parametrize("voice", ["en_US-lessac-high", "es_MX-claude-high"])
def test_stock_voice_downloads_model_and_config_from_same_lockfile_commit(isolated_models, monkeypatch, voice):
    entry = model_paths.pinned_hf_entry(voice)
    download = Mock(return_value="/downloaded/voice.onnx")
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(hf_hub_download=download))

    assert PiperTTSEngine._ensure_voice_downloaded(voice) == "/downloaded/voice.onnx"

    assert len(download.call_args_list) == 2
    assert [call.kwargs["filename"] for call in download.call_args_list] == entry["required_files"]
    for call in download.call_args_list:
        assert call.kwargs["revision"] == entry["revision"]
        assert call.kwargs["repo_id"] == entry["repo_id"]


def test_piper_uses_pinned_legacy_hf_snapshot_and_ignores_other_revisions(isolated_models):
    entry = model_paths.pinned_hf_entry("en_US-lessac-high")
    snapshots = isolated_models / ".local/share/piper_tts/models--rhasspy--piper-voices/snapshots"
    _voice_pair(snapshots / ("b" * 40) / entry["required_files"][0])
    assert model_paths.resolve_piper_voice(entry["voice"]) is None
    expected = _voice_pair(snapshots / entry["revision"] / entry["required_files"][0])
    assert model_paths.resolve_piper_voice(entry["voice"]) == expected


@pytest.mark.parametrize("kind", ["explicit", "flat", "managed"])
def test_complete_local_voice_never_downloads(isolated_models, monkeypatch, kind):
    voice = "es_MX-claude-high"
    entry = model_paths.pinned_hf_entry(voice)
    if kind == "managed":
        model = isolated_models / "models" / entry["subdir"] / entry["required_files"][0]
    elif kind == "flat":
        model = isolated_models / ".local/share/piper_tts" / f"{voice}.onnx"
    else:
        model = isolated_models / "custom" / "my-voice.onnx"
        voice = str(model)
    expected = _voice_pair(model)
    download = Mock(side_effect=AssertionError("local voice attempted network"))
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(hf_hub_download=download))

    assert PiperTTSEngine._ensure_voice_downloaded(voice) == expected
    download.assert_not_called()


def test_custom_stock_voice_requires_pinned_registration_before_download(isolated_models, monkeypatch):
    download = Mock(side_effect=AssertionError("unpinned network request"))
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(hf_hub_download=download))
    with pytest.raises(ValueError, match=r"models\.lock\.json"):
        PiperTTSEngine._ensure_voice_downloaded("en_US-custom-high")
    download.assert_not_called()


def test_registered_custom_voice_uses_explicit_repo_commit_and_filenames(isolated_models, monkeypatch):
    entry = {
        "type": "hf-snapshot",
        "repo_id": "custom/voices",
        "revision": "c" * 40,
        "voice": "my-custom-voice",
        "subdir": "custom-voice",
        "required_files": ["reviewed/speaker.onnx", "reviewed/speaker.onnx.json"],
        "weights_required": False,
    }
    monkeypatch.setattr(model_paths, "load_model_manifest", lambda *args: {"models": {"custom": entry}})
    download = Mock(return_value="/pinned/speaker.onnx")
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(hf_hub_download=download))
    assert PiperTTSEngine._ensure_voice_downloaded(entry["voice"]) == "/pinned/speaker.onnx"
    assert [call.kwargs["filename"] for call in download.call_args_list] == entry["required_files"]
    for call in download.call_args_list:
        assert call.kwargs["repo_id"] == entry["repo_id"]
        assert call.kwargs["revision"] == entry["revision"]


def test_incomplete_flat_voice_does_not_bypass_pinned_download(isolated_models, monkeypatch):
    path = isolated_models / ".local/share/piper_tts/en_US-lessac-high.onnx"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"incomplete")
    download = Mock(return_value="/pinned/voice.onnx")
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(hf_hub_download=download))
    assert PiperTTSEngine._ensure_voice_downloaded("en_US-lessac-high") == "/pinned/voice.onnx"
    assert download.call_count == 2


def test_missing_lite_voice_remains_offline(isolated_models, monkeypatch):
    monkeypatch.setenv("STARK_PROFILE", "lite-cpu")
    download = Mock(side_effect=AssertionError("Lite attempted live download"))
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(hf_hub_download=download))
    with pytest.raises(FileNotFoundError, match="setup --include tts"):
        PiperTTSEngine._ensure_voice_downloaded("en_US-lessac-high")
    download.assert_not_called()


@pytest.mark.parametrize("revision", [None, "main", "v1", "a" * 39])
def test_unpinned_registered_source_is_rejected(tmp_path, revision):
    (tmp_path / "models.lock.json").write_text(
        json.dumps({"models": {"test": {"type": "hf-snapshot", "repo_id": "org/test", "revision": revision}}})
    )
    with pytest.raises(ValueError, match="full commit"):
        model_paths.pinned_hf_entry("org/test", project_root=tmp_path)


@pytest.mark.parametrize("direction", ["en-es", "es-en"])
def test_marian_remote_model_and_tokenizer_share_manifest_revision(isolated_models, monkeypatch, direction):
    from engines.marian_hf_engine import MarianHFEngine

    model_id = f"Helsinki-NLP/opus-mt-{direction}"
    model = Mock()
    tokenizer_loader, model_loader = Mock(), Mock(return_value=model)
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            MarianTokenizer=SimpleNamespace(from_pretrained=tokenizer_loader),
            MarianMTModel=SimpleNamespace(from_pretrained=model_loader),
        ),
    )
    MarianHFEngine(model_id=model_id, device="cpu", warmup_passes=0).load()
    expected = {"revision": model_paths.pinned_hf_entry(model_id)["revision"]}
    tokenizer_loader.assert_called_once_with(model_id, **expected)
    model_loader.assert_called_once_with(model_id, **expected)


def test_hf_explicit_local_override_is_local_only_without_invented_revision(tmp_path):
    local = tmp_path / "custom"
    local.mkdir()
    source, kwargs = model_paths.resolve_hf_model_source(str(local))
    assert source == str(local)
    assert kwargs == {"local_files_only": True}


def test_hf_unknown_remote_fails_without_following_main(isolated_models):
    with pytest.raises(ValueError, match="explicit local model path"):
        model_paths.resolve_hf_model_source("unregistered/custom-marian")
