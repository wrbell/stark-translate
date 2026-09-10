"""Shared lite contract tests; no model loading or accelerator required."""

import asyncio
import json
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest

from settings import PipelineSettings
from stark_translate.profiles import apply_profile, resolve_profile, session_overrides

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def isolate_profile_environment(monkeypatch):
    monkeypatch.setenv("STARK_PROFILE", "standard")


@pytest.mark.parametrize(
    "name,backend,model,final",
    [
        ("lite-cpu", "cpu", "whisper-small", "marian"),
        ("lite-cpu-quality", "cpu", "whisper-small", "llamacpp"),
        ("lite-cuda-8gb", "cuda", "whisper-large-v3-turbo", "llamacpp"),
    ],
)
def test_profile_is_identical_across_settings_and_operator(name, backend, model, final):
    settings = PipelineSettings()
    p = apply_profile(settings, name)
    assert p.backend == settings.backend == backend
    assert p.stt_model == settings.stt.whisper_cuda_model == model
    assert p.final_engine == final
    assert settings.low_vram == (final == "marian")
    assert settings.stt.cpu_threads == 3 and settings.stt.num_workers == 1
    assert settings.translation.marian_device == "cpu"
    assert settings.translation.marian_intra_threads == 1
    assert not settings.stt.fallback_on_low_conf
    assert settings.stt.local_files_only
    assert not settings.run_ab and not settings.multiprocess
    assert session_overrides(name)["low_vram"] == settings.low_vram


def test_profiles_reject_incompatible_hardware_and_ambiguous_optional_groups():
    with pytest.raises(ValueError, match="requires backend cpu"):
        resolve_profile("lite-cpu", "mlx")
    with pytest.raises(ValueError, match="lite-cpu-quality"):
        resolve_profile("lite-cpu").model_keys(["e2b"])
    baseline = PipelineSettings()
    before = baseline.model_dump()
    apply_profile(baseline, "standard")
    assert baseline.model_dump() == before


def test_exact_cpu_setup_selection_has_no_unrequested_gemma_or_turbo():
    manifest = json.loads((ROOT / "models.lock.json").read_text())["models"]
    cpu = resolve_profile("lite-cpu").model_keys(manifest=manifest)
    cuda = resolve_profile("lite-cuda-8gb").model_keys(manifest=manifest)
    assert cpu <= manifest.keys()
    assert "whisper-small" in cpu and "whisper-large-v3-turbo" not in cpu
    assert not any("gemma" in key for key in cpu)
    assert "gemma-4-e2b-it-q4km.gguf" in cuda
    assert "gemma-4-e4b-it-q4km.gguf" not in cuda
    assert "mlx-parakeet-v3" not in cuda
    assert manifest["whisper-small"]["revision"] == "536b0662742c02347bc0e980a01041f333bce120"


def test_alias_sets_cpu_profile_before_operator_import(monkeypatch):
    from operator_app import cli

    monkeypatch.delenv("STARK_PROFILE", raising=False)
    called = []
    monkeypatch.setattr(cli, "main", lambda argv: called.append((argv, os.environ["STARK_PROFILE"])) or 0)
    assert cli.lite_main(["operator"]) == 0
    assert called == [(["operator"], "lite-cpu")]


def test_onnx_vad_numpy_state_and_discontinuity(monkeypatch):
    from tools.vad_runtime import NumpyOnnxVAD

    feeds = []

    def run(_, feed):
        feeds.append({k: v.copy() for k, v in feed.items()})
        return np.array([[0.7]], dtype=np.float32), feed["state"] + 1

    session = SimpleNamespace(run=run)
    monkeypatch.setitem(
        sys.modules,
        "onnxruntime",
        SimpleNamespace(SessionOptions=lambda: SimpleNamespace(), InferenceSession=lambda *a, **k: session),
    )
    vad = NumpyOnnxVAD("pinned.onnx")
    assert vad(np.ones(512, dtype=np.float32), 16000).item() == pytest.approx(0.7)
    vad(np.zeros(512, dtype=np.float32), 16000)
    assert feeds[0]["input"].shape == (1, 576)
    assert not feeds[0]["input"][:, :64].any()
    assert feeds[1]["input"][:, :64].all()
    assert feeds[1]["state"].min() == 1
    vad.reset_states()
    assert not vad._state.any() and not vad._context.any()
    with pytest.raises(ValueError, match="512"):
        vad(np.ones(300), 16000)
    with pytest.raises(ValueError, match="16000"):
        vad(np.ones(512), 8000)


def test_pipeline_cpu_loader_honors_selected_artifact_and_caps(monkeypatch):
    import dry_run_ab as pipeline
    import engines.factory as factory
    import stark_translate.profiles as profiles

    selected = PipelineSettings()
    apply_profile(selected, "lite-cpu")
    monkeypatch.setattr(pipeline, "settings", selected)
    monkeypatch.setattr(
        profiles,
        "resolve_profile_model",
        lambda model, **kw: "/cache/small" if model == "whisper-small" else None,
    )
    engine = Mock()
    create = Mock(return_value=engine)
    monkeypatch.setattr(factory, "create_stt_engine", create)
    assert pipeline.load_whisper("cpu") is engine
    assert create.call_args.kwargs["model_id"] == "/cache/small"
    assert create.call_args.kwargs["compute_type"] == "int8"
    assert create.call_args.kwargs["cpu_threads"] == 3
    assert create.call_args.kwargs["num_workers"] == 1
    assert create.call_args.kwargs["fallback_on_low_conf"] is False
    engine.load.assert_called_once()


@pytest.mark.parametrize("backend", ["cpu", "mlx"])
def test_absent_gemma_emits_real_marian_final(monkeypatch, backend):
    import dry_run_ab as pipeline

    for key, value in {
        "BACKEND": backend,
        "MULTIPROCESS": False,
        "_RUN_AB": False,
        "mlx_a_model": None,
        "mlx_b_model": None,
        "DIARIZE_ENABLED": False,
        "tts_engine": None,
        "_io_pool": Mock(),
        "all_results": [],
        "partial_translations": {},
        "partial_latencies": {},
        "_last_gen_stats": {},
        "_experiment_counters": {},
        "_translation_active": threading.Event(),
    }.items():
        monkeypatch.setattr(pipeline, key, value)
    monkeypatch.setattr(pipeline, "translate_marian", lambda text: ("Texto final real", 30))
    monkeypatch.setattr(pipeline, "translate_mlx_streaming", Mock(side_effect=AssertionError("sentinel path called")))
    monkeypatch.setattr(pipeline, "get_resource_snapshot", lambda: {})
    monkeypatch.setattr(pipeline, "qe_score", lambda *args: 1)
    monkeypatch.setattr(pipeline, "broadcast", AsyncMock())
    for name in ("check_homophones", "check_bad_split", "check_near_miss"):
        monkeypatch.setattr(pipeline, name, lambda *args: None)

    async def exercise():
        monkeypatch.setattr(pipeline, "_pipeline_translation_lock", asyncio.Lock())
        await pipeline._pipeline_translate_and_finalize(1, "Real source", 20, 0.9, [], [], np.ones(16000), 1)

    with ThreadPoolExecutor(max_workers=1) as pool:
        monkeypatch.setattr(pipeline, "_pytorch_pool", pool)
        asyncio.run(exercise())
    assert pipeline.all_results[0]["spanish_a"] == "Texto final real"
    assert pipeline._experiment_snapshot()["final_marian_routes"] == 1


def test_lite_dependencies_do_not_require_torch():
    import tomllib

    doc = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert doc["project"]["scripts"]["stark-translate-lite"] == "operator_app.cli:lite_main"
    extra = doc["project"]["optional-dependencies"]["lite-cpu"]
    assert not any(dep.split(">", 1)[0] in {"torch", "silero-vad", "torchaudio", "mlx"} for dep in extra)


def test_torch_free_pipeline_and_tts_import_in_fresh_process():
    # No suite-wide module mocks here. Make installed Torch/MLX unavailable,
    # matching an actual slim environment; import only, never load models.
    code = """
import sys
for name in ("torch", "torchvision", "torchaudio", "mlx", "mlx_lm", "silero_vad"):
    sys.modules[name] = None
import dry_run_ab
from engines.tts_engine import PiperTTSEngine
from tools.vad_runtime import NumpyOnnxVAD
assert dry_run_ab.torch is None
assert not dry_run_ab.MLX_AVAILABLE
assert dry_run_ab.RUNTIME_PROFILE.name == "lite-cpu"
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env={**os.environ, "STARK_PROFILE": "lite-cpu"},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_setup_offline_never_downloads_missing_profile_artifacts(tmp_path, monkeypatch):
    import operator_app.setup as setup

    monkeypatch.setattr(setup, "resolve_model_path", lambda *a, **k: None)
    download = Mock(side_effect=AssertionError("network attempted"))
    monkeypatch.setattr(setup, "_download_direct", download)
    monkeypatch.setattr(setup, "_download_hf_snapshot", download)
    monkeypatch.setattr(
        "tools.marian_ct2_setup.ensure_managed_marian", Mock(side_effect=FileNotFoundError("not prepared"))
    )
    assert setup.bootstrap_models(tmp_path, project_root=ROOT, profile="lite-cpu", offline=True) == 1
    download.assert_not_called()


@pytest.mark.parametrize("explicit", [False, True])
def test_standard_default_preserves_adapter_preference_but_explicit_alias_wins(monkeypatch, explicit):
    import dry_run_ab as pipeline
    import engines.factory as factory
    import engines.model_paths as paths

    selected = PipelineSettings()
    if explicit:
        selected.stt.whisper_cuda_model = "large-v3-turbo"
    monkeypatch.setattr(pipeline, "settings", selected)
    resolved = Mock(return_value="/cache/stock-turbo")
    monkeypatch.setattr(paths, "resolve_model_path", resolved)
    create = Mock(return_value=Mock(model_id="/chosen-model"))
    monkeypatch.setattr(factory, "create_stt_engine", create)
    pipeline.load_whisper("cpu")
    assert create.call_args.kwargs["model_id"] == ("/cache/stock-turbo" if explicit else None)
    assert create.call_args.kwargs["compute_type"] == "int8"
    assert resolved.call_count == int(explicit)


def test_standard_cpu_does_not_require_unused_gemma_or_server(tmp_path, monkeypatch):
    from operator_app import preflight

    monkeypatch.setattr(preflight, "resolve_marian_ct2", lambda *a, **k: "/cache/marian")
    lookup = Mock(return_value="/cache/whisper")
    monkeypatch.setattr(preflight, "resolve_model_path", lookup)
    result = preflight.check_models(tmp_path, backend="cpu")
    assert result["status"] == "pass"
    assert [call.args[0] for call in lookup.call_args_list] == ["whisper-large-v3-turbo"]
    monkeypatch.setattr(preflight, "check_dependencies", lambda *a, **k: preflight._check("deps", "pass", "ok"))
    monkeypatch.setattr(preflight, "check_microphone", lambda *a: preflight._check("mic", "pass", "ok"))
    llama = Mock(side_effect=AssertionError("unused server probed"))
    monkeypatch.setattr(preflight, "check_llamacpp_server", llama)
    assert preflight.run_all_checks(tmp_path, backend="cpu", profile="standard")["ok"]
    llama.assert_not_called()
    manifest = json.loads((ROOT / "models.lock.json").read_text())["models"]
    assert "cpu" not in manifest["gemma-4-e2b-it-q4km.gguf"]["required_for"]
    assert manifest["gemma-4-e2b-it-q4km.gguf"]["optional_group"] == "e2b"
