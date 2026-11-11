"""TranslateGemma retirement, stage 1: no default, fallback or setup group resolves it.

A person can still pass an explicit model ID. These tests do not load a model.
"""

import ast
import json
import re
import sys
from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
RETIRED = "translategemma"
RETIRED_REPO = re.compile(r"translategemma-(4b|12b)-it")
E4B = "mlx-community/gemma-4-e4b-it-OptiQ-4bit"
E2B = "mlx-community/gemma-4-e2b-it-OptiQ-4bit"


def _strings(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield str(key)
            yield from _strings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _strings(item)
    elif isinstance(value, str):
        yield value


def test_lock_file_has_no_translategemma_entry():
    lock = json.loads((ROOT / "models.lock.json").read_text())
    assert [s for s in _strings(lock) if RETIRED in s.lower()] == []


def test_setup_include_rejects_translategemma(monkeypatch):
    from operator_app import cli, setup

    bootstrap = MagicMock(return_value=0)  # never run a real setup from a test
    monkeypatch.setattr(setup, "bootstrap_models", bootstrap)
    with pytest.raises(SystemExit):
        cli.main(["setup", "--backend", "mlx", "--include", "translategemma"])
    bootstrap.assert_not_called()


def test_settings_defaults_name_no_translategemma_model(monkeypatch):
    import os

    for name in list(os.environ):
        if name.startswith("STARK_"):
            monkeypatch.delenv(name)
    from settings import PipelineSettings

    values = PipelineSettings(_env_file=None).model_dump()
    assert [s for s in _strings(values) if RETIRED in s.lower()] == []


@pytest.mark.parametrize(
    "path",
    [
        "settings.py",
        "engines/factory.py",
        "engines/cuda_engine.py",
        "dry_run_ab.py",
        "features/summarize_sermon.py",
        "tools/health_check.py",
        "tools/manage_adapters.py",
        "tools/benchmark_latency.py",
        "tools/test_adaptive_model.py",
        "tools/validate_session.py",
    ],
)
def test_no_translategemma_repo_literal_outside_docstrings(path):
    tree = ast.parse((ROOT / path).read_text())
    docstrings = {
        id(node.body[0].value)
        for node in ast.walk(tree)
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        and node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
    }
    literals = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in docstrings
        and RETIRED_REPO.search(node.value)
    ]
    assert literals == []


def test_factory_translategemma_family_needs_explicit_model():
    from engines.factory import resolve_mlx_translation_model_id

    with pytest.raises(ValueError, match="retired"):
        resolve_mlx_translation_model_id(model_family="translategemma")
    assert resolve_mlx_translation_model_id("local/tg", model_family="translategemma") == "local/tg"
    assert resolve_mlx_translation_model_id(model_family="gemma4") == E4B


@pytest.mark.parametrize("streaming", [True, False])
def test_factory_cuda_hf_engine_needs_explicit_model(streaming):
    from engines.factory import create_translation_engine

    with pytest.raises(ValueError, match="retired"):
        create_translation_engine(backend="cuda", streaming=streaming)


@pytest.mark.parametrize("name", ["CUDAGemmaEngine", "CUDAGemmaStreamingEngine"])
def test_cuda_hf_engine_load_needs_explicit_model(name):
    from engines import cuda_engine

    with patch("engines.cuda_engine.TORCH_AVAILABLE", True), patch("engines.cuda_engine.torch") as torch:
        torch.cuda.is_available.return_value = True
        engine = getattr(cuda_engine, name)()
    with pytest.raises(ValueError, match="retired"):
        engine.load()


def test_dry_run_mlx_model_selection():
    import dry_run_ab

    assert RETIRED not in dry_run_ab.MLX_MODEL_A.lower()
    assert dry_run_ab.MLX_MODEL_B is None
    assert dry_run_ab._select_mlx_models("gemma4", None, "e4b") == (E4B, E2B)
    assert dry_run_ab._select_mlx_models("gemma4", "local/x", "e4b") == ("local/x", E2B)
    with pytest.raises(ValueError, match="retired"):
        dry_run_ab._select_mlx_models("translategemma", None, "e4b")
    assert dry_run_ab._select_mlx_models("translategemma", "local/tg", "e4b") == ("local/tg", None)
    with pytest.raises(ValueError, match="retired"):
        dry_run_ab._select_mlx_models("translategemma", "local/tg", "e4b", run_ab=True)


def test_summary_has_no_translategemma_default():
    from features import summarize_sermon as summary

    assert not hasattr(summary, "TRANSLATE_MODEL_ID")
    assert not hasattr(summary, "translate_with_translategemma")
    assert summary.DEFAULT_MODEL == E4B


def test_health_check_cuda_needs_explicit_base_model(monkeypatch):
    from tools import health_check as hc

    run = MagicMock()
    monkeypatch.setattr(hc, "run_health_check", run)
    monkeypatch.setattr(sys, "argv", ["health_check.py", "--backend", "cuda"])
    with pytest.raises(SystemExit):
        hc.main()
    run.assert_not_called()


def test_manage_adapters_activate_passes_no_retired_base(tmp_path, monkeypatch):
    from tools import manage_adapters as ma

    monkeypatch.setattr(ma, "MANIFEST_PATH", tmp_path / "manifest.json")
    adapter = tmp_path / "v1"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"1")
    ma.cmd_register(Namespace(adapter=str(adapter), model="gemma_4b", version="v1", eval_file=None))
    run = MagicMock(return_value=MagicMock(returncode=0, stdout="ok", stderr=""))
    monkeypatch.setattr(ma.subprocess, "run", run)
    ma.cmd_activate(Namespace(model="gemma_4b", version="v1", base_model=None, max_latency=None))
    command = run.call_args.args[0]
    assert "--base-model" not in command
    assert not any(RETIRED_REPO.search(str(part)) for part in command)


def test_benchmark_mlx_accel_default_configs_exclude_translategemma(monkeypatch, tmp_path):
    from tools import benchmark_mlx_accel as bench

    seen = {}
    output = tmp_path / "accel.json"
    monkeypatch.setattr(sys, "argv", ["benchmark_mlx_accel.py", "--no-stt", "--output", str(output)])
    monkeypatch.setattr(bench, "bench_mlx_gemma4_accel", lambda **kw: seen.update(kw) or {})
    bench.main()
    assert seen["configs"]
    assert not any(bench.CONFIGS[name]["model_family"] == RETIRED for name in seen["configs"])


def test_preflight_rejects_translategemma_family_without_lookup(tmp_path):
    from operator_app import preflight

    with patch.object(preflight, "resolve_model_path") as resolve:
        result = preflight.check_models(tmp_path, backend="mlx", model_family="translategemma")
    assert result["status"] == "fail"
    assert "retired" in result["detail"]
    resolve.assert_not_called()


def test_raw_load_tools_refuse_without_explicit_model():
    from tools import benchmark_latency, test_adaptive_model

    assert benchmark_latency.MLX_MODEL_A is None and benchmark_latency.MLX_MODEL_B is None
    with pytest.raises(ValueError, match="retired"):
        benchmark_latency._load_mlx_gemma(None, "4B")
    assert test_adaptive_model.TG_MODEL_ID is None
    with pytest.raises(ValueError, match="retired"):
        test_adaptive_model.load_gemma_4b()


@pytest.mark.parametrize("flag,expected", [("--all", False), ("--gemma", True)])
def test_convert_all_excludes_translategemma(monkeypatch, tmp_path, flag, expected):
    from tools import convert_models_to_both as convert

    for name in dir(convert):
        if name.startswith(("export_", "validate_")):
            monkeypatch.setattr(
                convert,
                name,
                MagicMock(return_value=dict.fromkeys(("model", "target", "format", "size"), "x") | {"status": "SKIP"}),
            )
    argv = ["convert_models_to_both.py", flag, "--dry-run", "--output-dir", str(tmp_path)]
    monkeypatch.setattr(sys, "argv", argv)
    convert.main()
    assert convert.export_gemma_mlx.called is expected
    assert convert.export_gemma_cuda.called is expected
