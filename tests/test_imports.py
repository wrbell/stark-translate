"""Smoke tests: verify every module in engines/, features/, tools/ can be imported.

Catches broken imports, missing __init__.py, syntax errors, and circular imports
that would otherwise be invisible until runtime.
"""

import importlib

import pytest

# Modules to import as dotted paths.
# Root scripts (dry_run_ab.py, download_sermons.py) are skipped — they have
# top-level side effects. training/ is skipped — CUDA-only, different env.
ENGINE_MODULES = [
    "engines.base",
    "engines.factory",
    "engines.mlx_engine",
    "engines.cuda_engine",
    "engines.active_learning",
]

# features/ and tools/ have no __init__.py — import via importlib with spec_from_file
FEATURE_FILES = [
    ("features.diarize", "features/diarize.py"),
    ("features.extract_verses", "features/extract_verses.py"),
    ("features.summarize_sermon", "features/summarize_sermon.py"),
]

TOOL_FILES = [
    ("tools.benchmark_latency", "tools/benchmark_latency.py"),
    ("tools.convert_models_to_both", "tools/convert_models_to_both.py"),
    ("tools.live_caption_monitor", "tools/live_caption_monitor.py"),
    ("tools.stt_benchmark", "tools/stt_benchmark.py"),
    ("tools.test_adaptive_model", "tools/test_adaptive_model.py"),
    ("tools.translation_qe", "tools/translation_qe.py"),
]


@pytest.mark.parametrize("module_name", ENGINE_MODULES)
def test_import_engine(module_name):
    """Engine modules import without error."""
    mod = importlib.import_module(module_name)
    assert mod is not None


@pytest.mark.parametrize("module_name,file_path", FEATURE_FILES)
def test_import_feature(module_name, file_path):
    """Feature modules import without error."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    assert spec is not None, f"Cannot find {file_path}"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod is not None


@pytest.mark.parametrize("module_name,file_path", TOOL_FILES)
def test_import_tool(module_name, file_path):
    """Tool modules import without error."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    assert spec is not None, f"Cannot find {file_path}"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod is not None
