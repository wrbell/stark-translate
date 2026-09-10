"""Real manifest/cache resolution with fake downloads and model runtimes only."""

import builtins
import json
import sys
import wave
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from engines import model_paths


@pytest.fixture
def sources(tmp_path, monkeypatch):
    manifest = model_paths.load_model_manifest()
    manifest["models"]["test-model"] = {
        "type": "hf-snapshot",
        "repo_id": "test/model",
        "revision": "a" * 40,
        "subdir": "test-model",
        "required_files": ["config.json", "model.safetensors"],
        "allow_patterns": ["config.json", "*.safetensors"],
    }
    root = tmp_path / "project"
    root.mkdir()
    path = root / "models.lock.json"
    path.write_text(json.dumps(manifest))
    original_manifest = model_paths.load_model_manifest
    monkeypatch.setattr(model_paths, "load_model_manifest", lambda project_root=None: original_manifest(root))
    for key, value in {
        "HF_HOME": tmp_path / "hf",
        "HF_HUB_CACHE": tmp_path / "hub",
        "STARK_MODELS_DIR": tmp_path / "managed",
        "STARK_PROJECT_ROOT": root,
        "STARK_PROFILE": "standard",
    }.items():
        monkeypatch.setenv(key, str(value))

    def snapshot(repo_id, revision, **kwargs):
        entry = next(item for item in manifest["models"].values() if item.get("repo_id") == repo_id)
        folder = tmp_path / "hub" / ("models--" + repo_id.replace("/", "--")) / "snapshots" / revision
        folder.mkdir(parents=True, exist_ok=True)
        for name in entry.get("required_files", ["config.json"]):
            (folder / name).parent.mkdir(parents=True, exist_ok=True)
            (folder / name).write_text("{}" if name.endswith(".json") else "fixture")
        if entry.get("weights_required", True) and not any(folder.glob("*.bin")):
            (folder / "model.safetensors").touch()
        return str(folder)

    download = MagicMock(side_effect=snapshot)
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=download))
    return SimpleNamespace(root=root, manifest=manifest, path=path, download=download, snapshot=snapshot)


def test_offline_lookup_remains_pure_on_cache_miss(sources, monkeypatch):
    original_import = builtins.__import__

    def no_hub(name, *args, **kwargs):
        if name.startswith("huggingface_hub"):
            pytest.fail("offline lookup imported the downloader")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_hub)
    assert model_paths.resolve_model_path("test/model") == "test/model"
    assert model_paths.resolve_model_path("test/model", local_only=True) is None
    sources.download.assert_not_called()


def test_loading_downloads_full_revision_then_reuses_complete_cache(sources):
    local = model_paths.resolve_model_for_loading("test-model")
    assert Path(local).name == "a" * 40
    sources.download.assert_called_once_with(
        repo_id="test/model", revision="a" * 40, allow_patterns=["config.json", "*.safetensors"]
    )
    assert model_paths.resolve_model_for_loading("test/model") == local
    sources.download.assert_called_once()


def test_explicit_local_directory_does_not_need_manifest_or_downloader(sources, monkeypatch, tmp_path):
    explicit = tmp_path / "explicit"
    explicit.mkdir()
    monkeypatch.setattr(model_paths, "load_model_manifest", MagicMock(side_effect=AssertionError("manifest")))
    assert model_paths.resolve_model_for_loading(str(explicit)) == str(explicit.resolve())
    sources.download.assert_not_called()


def test_managed_cache_preserves_existing_local_override(sources, monkeypatch, tmp_path):
    managed = tmp_path / "managed" / "test-model"
    managed.mkdir(parents=True)
    (managed / "config.json").write_text("{}")
    (managed / "model.safetensors").touch()
    assert model_paths.resolve_model_for_loading("test/model") == str(managed)
    sources.download.assert_not_called()


@pytest.mark.parametrize("revision", [None, "main", "v1", "a" * 39])
def test_unpinned_remote_rejected_before_download(sources, revision):
    sources.manifest["models"]["test-model"]["revision"] = revision
    sources.path.write_text(json.dumps(sources.manifest))
    with pytest.raises(model_paths.UnpinnedModelError, match="register its full commit"):
        model_paths.resolve_model_for_loading("test/model")
    sources.download.assert_not_called()


def test_unknown_remote_rejected_before_download(sources):
    with pytest.raises(model_paths.UnpinnedModelError, match="unknown/model"):
        model_paths.resolve_model_for_loading("unknown/model")
    sources.download.assert_not_called()


@pytest.mark.parametrize("missing", ["required_file", "indexed_shard"])
def test_partial_download_never_reaches_wrapper(sources, missing):
    def incomplete(**kwargs):
        path = Path(sources.snapshot(**kwargs))
        if missing == "required_file":
            (path / "model.safetensors").unlink()
        else:
            (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"layer": "absent"}}))
        return str(path)

    sources.download.side_effect = incomplete
    with pytest.raises(ValueError, match="Incomplete pinned model snapshot"):
        model_paths.resolve_model_for_loading("test/model")


def test_whisper_policy_error_does_not_silently_fallback(sources, monkeypatch):
    from engines import mlx_engine

    transcribe = MagicMock()
    monkeypatch.setattr(mlx_engine, "MLX_AVAILABLE", True)
    monkeypatch.setattr(mlx_engine, "mx", MagicMock())
    monkeypatch.setattr(mlx_engine, "mlx_whisper", SimpleNamespace(transcribe=transcribe))
    engine = mlx_engine.MLXWhisperEngine(model_id="unknown/model")
    with pytest.raises(model_paths.UnpinnedModelError):
        engine.load()
    sources.download.assert_not_called()
    transcribe.assert_not_called()
    assert not engine._loaded


def test_actual_live_whisper_and_worker_use_pinned_paths(sources, monkeypatch):
    import dry_run_ab
    import workers

    transcribe = MagicMock(return_value={"text": "", "segments": []})
    monkeypatch.setitem(sys.modules, "mlx_whisper", SimpleNamespace(transcribe=transcribe))
    monkeypatch.setattr(dry_run_ab.settings.stt, "backend", "mlx")
    monkeypatch.setattr(dry_run_ab.settings.stt, "whisper_model", "test/model")
    local = dry_run_ab.load_whisper("mlx")
    conn = MagicMock()
    conn.recv.return_value = None
    workers.stt_worker_main(conn, "test/model")
    assert Path(local).name == "a" * 40
    assert {call.kwargs["path_or_hf_repo"] for call in transcribe.call_args_list} == {local}
    conn.send.assert_called_once_with("ready")
    sources.download.assert_called_once()


@pytest.mark.parametrize("legacy", [False, True])
def test_speechbrain_pins_secondary_weights_with_actual_api_shapes(sources, monkeypatch, legacy):
    from features import live_diarize

    calls = []

    class LegacyClassifier:
        @classmethod
        def from_hparams(cls, source, revision=None, **kwargs):
            calls.append({"source": source, "revision": revision, **kwargs})
            return object()

    class CurrentClassifier:
        @classmethod
        def from_hparams(cls, source, hparams_file="hyperparams.yaml", **kwargs):
            calls.append({"source": source, **kwargs})
            return object()

    @dataclass
    class FetchConfig:
        revision: str
        allow_network: bool

    classifier = LegacyClassifier if legacy else CurrentClassifier
    monkeypatch.setitem(sys.modules, "speechbrain.inference.speaker", SimpleNamespace(EncoderClassifier=classifier))
    monkeypatch.setitem(sys.modules, "speechbrain.utils.fetching", SimpleNamespace(FetchConfig=FetchConfig))
    assert live_diarize._load_speechbrain_embedder() is not None
    call = calls[0]
    revision = sources.manifest["models"]["speechbrain-ecapa"]["revision"]
    assert Path(call["source"]).name == revision
    assert call["overrides"] == {"pretrained_path": call["source"]}
    assert call["run_opts"] == {"device": "cpu"}
    if legacy:
        assert call["revision"] == revision
        assert "fetch_config" not in call
    else:
        assert call["fetch_config"] == FetchConfig(revision, False)
        assert "revision" not in call
    sources.download.assert_called_once_with(
        repo_id="speechbrain/spkrec-ecapa-voxceleb",
        revision=revision,
        allow_patterns=sources.manifest["models"]["speechbrain-ecapa"]["allow_patterns"],
    )


def test_optional_speechbrain_download_failure_is_graceful(sources, monkeypatch):
    from features import live_diarize

    classifier = MagicMock()
    monkeypatch.setitem(sys.modules, "speechbrain.inference.speaker", SimpleNamespace(EncoderClassifier=classifier))
    sources.download.side_effect = OSError("offline")
    assert live_diarize._load_speechbrain_embedder() is None
    classifier.from_hparams.assert_not_called()


def test_pyannote_embedding_receives_checkpoint_and_uses_existing_token(sources, monkeypatch):
    from features import live_diarize

    inference = MagicMock()
    monkeypatch.setitem(sys.modules, "pyannote.audio", SimpleNamespace(Inference=inference))
    monkeypatch.setattr(live_diarize, "hf_token", lambda: "fixture-token")
    assert live_diarize._load_pyannote_embedder() is not None
    path = Path(inference.call_args.args[0])
    assert path.name == "pytorch_model.bin"
    assert path.parent.name == sources.manifest["models"]["pyannote-embedding"]["revision"]
    assert inference.call_args.kwargs == {"window": "whole"}
    assert sources.download.call_args.kwargs["token"] == "fixture-token"


@pytest.mark.parametrize("invalid", [None, "nested_source", "class", "download"])
def test_pyannote_full_pipeline_never_receives_remote_nested_ids(sources, monkeypatch, invalid):
    from features import live_diarize

    entry = sources.manifest["models"]["pyannote-diarization-3.1"]
    config = {
        "pipeline": {
            "name": "pyannote.audio.pipelines.SpeakerDiarization",
            "params": {**entry["dependencies"], "embedding_exclude_overlap": True, "embedding_batch_size": 32},
        },
        "params": {"clustering": {"threshold": 0.7045654963945799}, "segmentation": {"min_duration_off": 0.0}},
    }
    if invalid == "nested_source":
        config["pipeline"]["params"]["segmentation"] = "unknown/moving"
    elif invalid == "class":
        config["pipeline"]["name"] = "unknown.CustomPipeline"

    def snapshot_with_config(**kwargs):
        root = Path(sources.snapshot(**kwargs))
        if kwargs["repo_id"] == entry["repo_id"]:
            # JSON is valid YAML: exercise the actual safe_load parser too.
            (root / "config.yaml").write_text(json.dumps(config))
        return str(root)

    sources.download.side_effect = snapshot_with_config
    if invalid == "download":
        sources.download.side_effect = OSError("no gated access")
    model = MagicMock()
    models = [object(), object()]
    model.from_pretrained.side_effect = models
    pipeline = MagicMock()
    monkeypatch.setitem(sys.modules, "pyannote.audio", SimpleNamespace(Model=model))
    monkeypatch.setitem(sys.modules, "pyannote.audio.pipelines", SimpleNamespace(SpeakerDiarization=pipeline))
    monkeypatch.setattr(live_diarize, "hf_token", lambda: "fixture-token")
    result = live_diarize._load_pyannote()
    if invalid:
        assert result is None
        pipeline.assert_not_called()
        model.from_pretrained.assert_not_called()
        assert sources.download.call_count == 1
    else:
        assert result is pipeline.return_value
        assert pipeline.call_args.kwargs["segmentation"] is models[0]
        assert pipeline.call_args.kwargs["embedding"] is models[1]
        assert pipeline.call_args.kwargs["embedding_batch_size"] == 32
        result.instantiate.assert_called_once_with(config["params"])
        for call in model.from_pretrained.call_args_list:
            assert Path(call.args[0]).is_file()
            assert call.kwargs == {"map_location": "cpu", "strict": False}
        assert sources.download.call_count == 3


def test_pyannote_without_credential_never_fetches(sources, monkeypatch):
    from features import live_diarize

    monkeypatch.setattr(live_diarize, "hf_token", lambda: None)
    assert live_diarize._load_pyannote() is None
    assert live_diarize._load_pyannote_embedder() is None
    sources.download.assert_not_called()


@pytest.mark.parametrize("group", ["whisper-fallback", "diarization", "diarization-pyannote"])
def test_actual_setup_parser_forwards_registered_optional_group(sources, monkeypatch, group):
    from operator_app import cli, setup

    bootstrap = MagicMock(return_value=0)
    monkeypatch.setattr(setup, "bootstrap_models", bootstrap)
    assert cli.main(["setup", "--backend", "mlx", "--profile", "standard", "--include", group]) == 0
    assert bootstrap.call_args.kwargs["include"] == [group]
    assert bootstrap.call_args.kwargs["backend"] == "mlx"
    assert bootstrap.call_args.kwargs["profile"] == "standard"
    assert any(entry.get("optional_group") == group for entry in sources.manifest["models"].values())
    sources.download.assert_not_called()


@pytest.mark.parametrize("group", ["whisper-fallback", "diarization", "diarization-pyannote"])
def test_new_setup_groups_preserve_lite_tts_only_policy(sources, monkeypatch, group):
    from operator_app import cli, setup

    monkeypatch.setattr(setup, "load_lockfile", lambda **kwargs: sources.manifest)
    with pytest.raises(ValueError, match="Lite profiles accept only --include tts"):
        cli.main(["setup", "--profile", "lite-cpu", "--include", group])
    sources.download.assert_not_called()


@pytest.mark.parametrize("revision", [None, "main", "v1", "a" * 39, "a" * 41, 12])
@pytest.mark.parametrize("existing_cache", [False, True])
def test_setup_rejects_unpinned_manifest_before_any_cache_mutation(sources, monkeypatch, revision, existing_cache):
    from operator_app import setup

    entry = dict(sources.manifest["models"]["test-model"], required_for=["mlx"])
    if revision is None:
        entry.pop("revision")
    else:
        entry["revision"] = revision
    # Even a valid earlier direct entry must not download before a later bad pin.
    direct = {
        "type": "direct",
        "url": "https://example.invalid/model",
        "filename": "first.bin",
        "required_for": ["mlx"],
    }
    manifest = {"version": "fixture", "models": {"first": direct, "bad": entry}}
    monkeypatch.setattr(setup, "load_lockfile", lambda **kwargs: manifest)
    cache = sources.root / "setup-target"
    if existing_cache:
        target = cache / entry["subdir"]
        target.mkdir(parents=True)
        (target / "config.json").write_text("{}")
        (target / "model.safetensors").write_text("existing local weights")
        (target / ".installed").write_text(json.dumps({"repo_id": entry["repo_id"], "revision": revision}))
    before = {str(path.relative_to(cache)): path.read_bytes() for path in cache.rglob("*") if path.is_file()}
    lookup = MagicMock(side_effect=AssertionError("cache lookup before manifest validation"))
    direct_download = MagicMock(side_effect=AssertionError("earlier entry must not download"))
    monkeypatch.setattr(setup, "resolve_model_path", lookup)
    monkeypatch.setattr(setup, "_download_direct", direct_download)
    assert setup.bootstrap_models(models_dir=cache, project_root=sources.root, backend="mlx") == 2
    assert cache.exists() is existing_cache
    after = {str(path.relative_to(cache)): path.read_bytes() for path in cache.rglob("*") if path.is_file()}
    assert before == after
    lookup.assert_not_called()
    direct_download.assert_not_called()
    sources.download.assert_not_called()


@pytest.mark.parametrize("revision", [None, "main", "a" * 39])
def test_setup_snapshot_wrapper_rejects_unpinned_direct_call(sources, revision):
    from operator_app import setup

    target = sources.root / "never-created"
    with pytest.raises(ValueError, match="full 40-character commit"):
        setup._download_hf_snapshot("test/model", revision, target, None)
    assert not target.exists()
    sources.download.assert_not_called()


def test_summary_loaders_use_actual_pinned_resolution_and_preserve_family(sources, monkeypatch):
    from features import summarize_sermon as summary

    core = MagicMock()
    monkeypatch.setitem(sys.modules, "mlx", SimpleNamespace(core=core))
    monkeypatch.setitem(sys.modules, "mlx.core", core)
    tokenizer = SimpleNamespace(
        eos_token_id=1, unk_token_id=3, _eos_token_ids={1, 50}, convert_tokens_to_ids=lambda _: 106
    )
    model = object()
    load = MagicMock(return_value=(model, tokenizer))
    monkeypatch.setitem(sys.modules, "mlx_lm", SimpleNamespace(load=load))
    warm = MagicMock(return_value="Hello")
    translate = MagicMock(return_value="Resumen")
    monkeypatch.setattr(summary, "generate_text", warm)
    monkeypatch.setattr(summary, "translate_summary", translate)
    assert summary.load_summarization_model() == (model, tokenizer)
    first = Path(load.call_args.args[0])
    assert first.name == model_paths.pinned_hf_entry(summary.settings.translation.mlx_model_gemma4_e4b)["revision"]
    warm.assert_called_once_with(model, tokenizer, "Hello.", max_tokens=1)
    assert tokenizer._stark_summary_model_family == "gemma4"
    assert tokenizer._eos_token_ids == {1, 50, 106}
    core.synchronize.assert_called_once()
    assert summary.translate_with_translategemma("Summary") == "Resumen"
    second = Path(load.call_args.args[0])
    assert second.name == model_paths.pinned_hf_entry(summary.TRANSLATE_MODEL_ID)["revision"]
    assert tokenizer._stark_summary_model_family == "translategemma"
    translate.assert_called_once_with(model, tokenizer, "Summary", max_tokens=600)


@pytest.mark.parametrize("input_state", ["missing", "directory", "empty", "wav"])
def test_pyannote_daemon_defers_models_until_valid_file_and_does_not_retry_failure(sources, monkeypatch, input_state):
    from features import live_diarize

    wav = sources.root / "rolling.wav"
    if input_state == "directory":
        wav.mkdir()
    elif input_state == "empty":
        wav.touch()
    elif input_state == "wav":
        with wave.open(str(wav), "wb") as stream:
            stream.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            stream.writeframes(b"\x00\x00" * 16)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "live_diarize.py",
            "--mode",
            "pyannote",
            "--rolling-wav",
            str(wav),
            "--output",
            str(sources.root / "diarize.jsonl"),
            "--interval-s",
            "0",
            "--max-iters",
            "3",
        ],
    )
    loader = MagicMock(return_value=None)
    monkeypatch.setattr(live_diarize, "_load_pyannote", loader)
    monkeypatch.setattr(live_diarize.signal, "signal", MagicMock())
    monkeypatch.setattr(live_diarize.time, "sleep", MagicMock())
    assert live_diarize.main() == 0
    assert loader.call_count == (1 if input_state == "wav" else 0)
    sources.download.assert_not_called()
    assert not (sources.root / "diarize.jsonl").exists()
