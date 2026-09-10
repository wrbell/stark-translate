"""Actual loaders and transcription paths with mocked ML runtimes only."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

import dry_run_ab
import engines.mlx_engine as mlx_engine
import engines.model_paths as model_paths


@pytest.fixture
def runtime(monkeypatch):
    transcribe = MagicMock(return_value={"text": "", "segments": []})
    fake = SimpleNamespace(transcribe=transcribe)
    monkeypatch.setitem(__import__("sys").modules, "mlx_whisper", fake)
    monkeypatch.setattr(mlx_engine, "mlx_whisper", fake)
    monkeypatch.setattr(mlx_engine, "mx", MagicMock())
    monkeypatch.setattr(mlx_engine, "MLX_AVAILABLE", True)
    resolve = MagicMock(side_effect=lambda name: "/installed/" + name)
    monkeypatch.setattr(model_paths, "resolve_model_path", resolve)
    monkeypatch.setattr(mlx_engine, "resolve_model_path", resolve)
    monkeypatch.setattr(dry_run_ab.settings.stt, "backend", "mlx")
    monkeypatch.setattr(dry_run_ab.settings.stt, "whisper_model", "chosen-primary")
    monkeypatch.setattr(dry_run_ab.settings.stt, "whisper_fallback", "chosen-fallback")
    return transcribe, resolve


@pytest.mark.parametrize("language", ["en", "es"])
def test_live_primary_load_uses_selected_model_and_resolver(runtime, monkeypatch, language):
    transcribe, resolve = runtime
    monkeypatch.setattr(dry_run_ab, "SOURCE_LANG", language)
    assert dry_run_ab.load_whisper("mlx") == "/installed/chosen-primary"
    resolve.assert_called_once_with("chosen-primary")
    assert transcribe.call_count == 1


def test_live_english_retains_configured_startup_fallback(runtime, monkeypatch):
    transcribe, resolve = runtime
    monkeypatch.setattr(dry_run_ab, "SOURCE_LANG", "en")
    transcribe.side_effect = [ValueError("bad primary"), {"text": ""}]
    assert dry_run_ab.load_whisper("mlx") == "/installed/chosen-fallback"
    assert [call.args[0] for call in resolve.call_args_list] == ["chosen-primary", "chosen-fallback"]
    assert [call.kwargs["path_or_hf_repo"] for call in transcribe.call_args_list] == [
        "/installed/chosen-primary",
        "/installed/chosen-fallback",
    ]


def test_live_spanish_load_failure_never_resolves_or_loads_fallback(runtime, monkeypatch):
    transcribe, resolve = runtime
    monkeypatch.setattr(dry_run_ab, "SOURCE_LANG", "es")
    transcribe.side_effect = ValueError("bad primary")
    with pytest.raises(RuntimeError, match="refusing fallback for language='es'"):
        dry_run_ab.load_whisper("mlx")
    transcribe.assert_called_once()
    resolve.assert_called_once_with("chosen-primary")


@pytest.mark.parametrize("language", ["en", "es"])
def test_engine_startup_language_contract(runtime, language):
    transcribe, resolve = runtime
    transcribe.side_effect = [ValueError("bad primary"), {"text": ""}]
    engine = mlx_engine.MLXWhisperEngine(model_id="primary", fallback_model_id="fallback", session_language=language)
    if language == "es":
        with pytest.raises(RuntimeError, match="validated only for English"):
            engine.load()
        assert not engine._loaded
        assert engine.model_id == "primary"
        resolve.assert_called_once_with("primary")
        transcribe.assert_called_once()
    else:
        engine.load()
        assert engine._loaded
        assert engine.model_id == "fallback"
        assert transcribe.call_count == 2


def test_engine_rejects_spanish_after_english_startup_fallback(runtime):
    transcribe, _ = runtime
    transcribe.side_effect = [ValueError("bad primary"), {"text": ""}]
    engine = mlx_engine.MLXWhisperEngine()
    engine.load()
    with pytest.raises(RuntimeError, match="refusing fallback for language='es'"):
        engine.transcribe(np.zeros(16000, dtype=np.float32), language="es")
    assert transcribe.call_count == 2  # no Spanish inference on the EN fallback


@pytest.mark.parametrize("language", ["en", "es"])
@pytest.mark.parametrize("avg_logprob,compression_ratio", [(-1.3, 1.0), (-0.1, 2.5)])
def test_quality_fallback_preserves_thresholds_and_only_retries_english(
    runtime, monkeypatch, language, avg_logprob, compression_ratio
):
    transcribe, _ = runtime
    engine = mlx_engine.MLXWhisperEngine(model_id="primary", fallback_model_id="fallback", session_language=language)
    engine.load()
    monkeypatch.setattr(engine, "_log_fallback", MagicMock())
    primary = {
        "text": "original",
        "segments": [{"avg_logprob": avg_logprob, "compression_ratio": compression_ratio}],
    }
    retry = {"text": "retry", "segments": [{"avg_logprob": -0.01, "compression_ratio": 1.0}]}
    transcribe.reset_mock()
    transcribe.side_effect = [primary, {"text": ""}, retry]
    result = engine.transcribe(np.zeros(16000, dtype=np.float32), language=language)
    if language == "es":
        assert result.text == "original"
        assert result.avg_logprob == avg_logprob
        assert not result.used_fallback
        assert not engine._fallback_loaded
        transcribe.assert_called_once()
    else:
        assert result.text == "retry"
        assert result.used_fallback
        assert engine._fallback_loaded
        assert transcribe.call_count == 3  # primary, lazy warmup, retry
    assert transcribe.call_args.kwargs["language"] == language


def test_engine_disabled_retry_stays_disabled_for_english(runtime):
    transcribe, _ = runtime
    engine = mlx_engine.MLXWhisperEngine(fallback_on_low_conf=False)
    engine.load()
    transcribe.reset_mock()
    transcribe.return_value = {"text": "original", "segments": [{"avg_logprob": -2.0}]}
    assert engine.transcribe(np.zeros(16000, dtype=np.float32)).text == "original"
    transcribe.assert_called_once()
