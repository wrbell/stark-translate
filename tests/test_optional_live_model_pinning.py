"""Optional live boundaries: tiny local fixtures, real policy, mocked native APIs."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest

from engines import model_paths
from engines.model_paths import UnpinnedModelError


@pytest.fixture(autouse=True)
def isolated_sources(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("STARK_MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("STARK_PROJECT_ROOT", str(tmp_path / "project"))
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hf"))
    download = Mock(side_effect=AssertionError("No network permitted in loader tests"))
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=download, hf_hub_download=download),
    )
    yield
    download.assert_not_called()


@pytest.fixture
def manifest(monkeypatch):
    entries = {}
    monkeypatch.setattr(model_paths, "load_model_manifest", lambda *args: {"models": entries})
    return entries


def register(manifest, model_id, *, files=None, revision="a" * 40):
    entry = {
        "type": "hf-snapshot",
        "repo_id": model_id,
        "revision": revision,
        "subdir": model_id.replace("/", "--"),
        "required_files": files or ["config.json", "model.bin"],
        "weights_required": False,
    }
    manifest[model_id] = entry
    return entry


def install(tmp_path, entry):
    root = tmp_path / "models" / entry["subdir"]
    for name in entry["required_files"]:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture only, never loaded")
    return str(root)


def transformers_engine(kind, monkeypatch, *, primary="fixture/target", draft=None):
    from engines import cuda_engine, hf_whisper_engine

    tokenizer = MagicMock()
    tokenizer.convert_tokens_to_ids.return_value = 106
    tokenizer.eos_token_id = 1
    tokenizer._eos_token_ids = {1}
    tokenizer_loader = Mock(return_value=tokenizer)
    model_loader = Mock(return_value=MagicMock())
    monkeypatch.setattr(cuda_engine, "TORCH_AVAILABLE", True)
    monkeypatch.setattr(cuda_engine, "BITSANDBYTES_AVAILABLE", True)
    monkeypatch.setattr(cuda_engine, "torch", MagicMock())
    monkeypatch.setattr(hf_whisper_engine, "HF_WHISPER_AVAILABLE", True)
    monkeypatch.setattr(hf_whisper_engine, "AutoProcessor", SimpleNamespace(from_pretrained=tokenizer_loader))
    monkeypatch.setattr(hf_whisper_engine, "AutoModelForSpeechSeq2Seq", SimpleNamespace(from_pretrained=model_loader))
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoTokenizer=SimpleNamespace(from_pretrained=tokenizer_loader),
            AutoModelForCausalLM=SimpleNamespace(from_pretrained=model_loader),
        ),
    )
    if kind == "hf-whisper":
        engine = hf_whisper_engine.HFWhisperEngine(
            model_id=primary, draft_model_id=draft, device="cpu", warmup_seconds=0
        )
    elif kind == "cuda-basic":
        engine = cuda_engine.CUDAGemmaEngine(model_id=primary)
    else:
        engine = cuda_engine.CUDAGemmaStreamingEngine(
            model_id=primary, assistant_model_id=draft, use_prompt_cache=False
        )
    return engine, tokenizer_loader, model_loader


@pytest.mark.parametrize("kind", ["hf-whisper", "cuda-basic", "cuda-streaming"])
@pytest.mark.parametrize("local", [False, True])
def test_transformers_pins_processor_target_and_draft(manifest, tmp_path, monkeypatch, kind, local):
    ids = ["fixture/target"] + ([] if kind == "cuda-basic" else ["fixture/draft"])
    expected = []
    for i, model_id in enumerate(ids):
        entry = register(manifest, model_id, revision=("a" if i == 0 else "b") * 40)
        expected.append(
            (install(tmp_path, entry), {"local_files_only": True})
            if local
            else (model_id, {"revision": entry["revision"]})
        )
    engine, tokenizer, loader = transformers_engine(kind, monkeypatch, draft=ids[1] if len(ids) > 1 else None)
    engine.load()
    tokenizer.assert_called_once_with(expected[0][0], **expected[0][1])
    assert len(loader.call_args_list) == len(expected)
    for call, (source, policy) in zip(loader.call_args_list, expected, strict=True):
        assert call.args == (source,)
        assert all(call.kwargs[k] == v for k, v in policy.items())
        assert ("revision" in call.kwargs) is not local
        assert ("local_files_only" in call.kwargs) is local


@pytest.mark.parametrize(
    ("kind", "missing"),
    [
        ("hf-whisper", "target"),
        ("hf-whisper", "draft"),
        ("cuda-basic", "target"),
        ("cuda-streaming", "target"),
        ("cuda-streaming", "draft"),
    ],
)
def test_transformers_rejects_missing_pin_before_any_load(manifest, monkeypatch, kind, missing):
    if missing == "draft":
        register(manifest, "fixture/target")
    engine, tokenizer, loader = transformers_engine(
        kind, monkeypatch, draft="fixture/draft" if missing == "draft" else None
    )
    with pytest.raises(UnpinnedModelError, match=rf"fixture/{missing}.*models\.lock\.json"):
        engine.load()
    tokenizer.assert_not_called()
    loader.assert_not_called()


@pytest.mark.parametrize("fallback", [False, True])
@pytest.mark.parametrize("state", ["local", "explicit", "missing-pin", "not-installed", "partial"])
def test_faster_whisper_always_receives_local_only_source(manifest, tmp_path, monkeypatch, fallback, state):
    from engines import cuda_engine

    model_id = "fixture/whisper"
    entry = register(manifest, model_id)
    source = install(tmp_path, entry) if state in {"local", "explicit", "partial"} else None
    if state == "explicit":
        model_id = source
        manifest.clear()
    elif state == "missing-pin":
        manifest.clear()
    elif state == "partial":
        (Path(source) / "model.bin").unlink()
    loader = Mock(return_value=SimpleNamespace(transcribe=Mock(return_value=([], None))))
    monkeypatch.setattr(cuda_engine, "FASTER_WHISPER_AVAILABLE", True)
    monkeypatch.setattr(cuda_engine, "WhisperModel", loader)
    engine = cuda_engine.FasterWhisperEngine(model_id=model_id, fallback_model_id=model_id, local_files_only=False)
    load = engine._load_fallback_model if fallback else engine.load
    if state in {"local", "explicit"}:
        load()
        assert loader.call_args.args == (source,)
        assert loader.call_args.kwargs["local_files_only"] is True
    else:
        with pytest.raises(UnpinnedModelError, match=r"fixture/whisper.*models\.lock\.json"):
            load()
        loader.assert_not_called()


@pytest.mark.parametrize("state", ["local", "explicit", "missing-pin", "not-installed", "ambiguous", "wrong-type"])
def test_nemo_restores_only_local_checkpoint(manifest, tmp_path, monkeypatch, state):
    from engines.parakeet_engine import ParakeetEngine

    model_id = "fixture/parakeet"
    entry = register(manifest, model_id, files=["voice.nemo"])
    root = install(tmp_path, entry) if state != "not-installed" else None
    if state == "explicit":
        model_id = str(Path(root) / "voice.nemo")
        manifest.clear()
    elif state == "missing-pin":
        manifest.clear()
    elif state == "ambiguous":
        (Path(root) / "another.nemo").write_bytes(b"fixture")
    elif state == "wrong-type":
        model_id = str(Path(root) / "config.json")
        Path(model_id).write_text("{}")
    nemo = MagicMock()
    monkeypatch.setitem(sys.modules, "nemo", nemo)
    monkeypatch.setitem(sys.modules, "nemo.collections", nemo.collections)
    monkeypatch.setitem(sys.modules, "nemo.collections.asr", nemo.collections.asr)
    api = nemo.collections.asr.models.ASRModel
    engine = ParakeetEngine(model_id=model_id, device="cpu")
    if state in {"local", "explicit"}:
        engine.load()
        api.restore_from.assert_called_once_with(restore_path=str(Path(root) / "voice.nemo"))
    else:
        with pytest.raises(UnpinnedModelError, match=r"models\.lock\.json"):
            engine.load()
        api.restore_from.assert_not_called()
    api.from_pretrained.assert_not_called()


@pytest.mark.parametrize("state", ["local", "explicit", "missing-pin", "not-installed", "missing-config"])
def test_legacy_piper_resolves_pair_before_loading(manifest, tmp_path, monkeypatch, state):
    from training import evaluate_piper

    voice = "en_US-fixture-high"
    entry = register(manifest, "fixture/voices", files=["voice.onnx", "voice.onnx.json"])
    entry["voice"] = voice
    root = install(tmp_path, entry) if state != "not-installed" else None
    if state == "explicit":
        voice = str(Path(root) / "voice.onnx")
        manifest.clear()
    elif state == "missing-pin":
        manifest.clear()
    elif state == "missing-config":
        (Path(root) / "voice.onnx.json").unlink()
    loader = Mock()
    monkeypatch.setattr(evaluate_piper, "PIPER_AVAILABLE", True)
    monkeypatch.setattr(evaluate_piper, "PiperVoice", SimpleNamespace(load=loader), raising=False)
    if state in {"local", "explicit"}:
        evaluate_piper.load_piper_voice(voice)
        loader.assert_called_once_with(str(Path(root) / "voice.onnx"), config_path=str(Path(root) / "voice.onnx.json"))
    else:
        with pytest.raises(UnpinnedModelError, match=r"models\.lock\.json"):
            evaluate_piper.load_piper_voice(voice)
        loader.assert_not_called()


@pytest.mark.parametrize("pinned", [False, True])
def test_live_monitor_reuses_pinned_snapshot_for_warmup_and_chunks(manifest, tmp_path, monkeypatch, pinned):
    from tools import live_caption_monitor as monitor

    entry = register(manifest, monitor.WHISPER_MODEL)
    source = install(tmp_path, entry)
    if not pinned:
        manifest.clear()
    transcribe = Mock(return_value={"text": "fixture transcript"})
    monkeypatch.setattr(monitor, "_import_mlx_whisper", lambda: SimpleNamespace(transcribe=transcribe))
    check = Mock()
    streamlink, ffmpeg = MagicMock(), MagicMock()
    ffmpeg.stdout.read.side_effect = [b"\0\0" * 160, b""]
    popen = Mock(side_effect=[streamlink, ffmpeg])
    monkeypatch.setattr(monitor.subprocess, "run", check)
    monkeypatch.setattr(monitor.subprocess, "Popen", popen)
    if pinned:
        segments = monitor.capture_and_transcribe_live("fixture", 60)
        assert len(segments) == 1
        assert transcribe.call_count == 2
        for call in transcribe.call_args_list:
            assert call.kwargs["path_or_hf_repo"] == source
        streamlink.terminate.assert_called_once()
        ffmpeg.terminate.assert_called_once()
    else:
        with pytest.raises(UnpinnedModelError, match=r"models\.lock\.json"):
            monitor.capture_and_transcribe_live("fixture", 60)
        transcribe.assert_not_called()
        check.assert_not_called()
        popen.assert_not_called()


@pytest.mark.parametrize("revision", [None, "main", "a" * 39])
def test_cached_moving_ref_cannot_bypass_hf_policy(manifest, tmp_path, revision):
    entry = register(manifest, "fixture/unpinned", revision=revision)
    install(tmp_path, entry)
    with pytest.raises(UnpinnedModelError, match=r"fixture/unpinned.*models\.lock\.json"):
        model_paths.resolve_hf_model_source("fixture/unpinned")
    with pytest.raises(UnpinnedModelError, match=r"fixture/unpinned.*models\.lock\.json"):
        model_paths.resolve_local_model_for_loading("fixture/unpinned")


def test_live_monitor_acquires_only_the_registered_commit(manifest, tmp_path, monkeypatch):
    from tools import live_caption_monitor as monitor

    entry = register(manifest, monitor.WHISPER_MODEL)
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    for name in entry["required_files"]:
        (snapshot / name).write_bytes(b"fixture only")
    download = Mock(return_value=str(snapshot))
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=download))
    transcribe = Mock()
    monkeypatch.setattr(monitor, "_import_mlx_whisper", lambda: SimpleNamespace(transcribe=transcribe))
    monkeypatch.setattr(monitor.subprocess, "run", Mock())
    monkeypatch.setattr(monitor.subprocess, "Popen", Mock(return_value=MagicMock()))

    monitor.capture_and_transcribe_live("fixture", 0)

    download.assert_called_once_with(repo_id=entry["repo_id"], revision=entry["revision"])
    assert transcribe.call_count == 1
    assert transcribe.call_args.kwargs["path_or_hf_repo"] == str(snapshot)
