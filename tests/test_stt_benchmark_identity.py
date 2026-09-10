"""Fallback failures must never turn into mislabeled primary-model measurements."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from tools import benchmark_mlx_accel, benchmark_stt_engines, stt_roundtrip_compare
from tools.benchmark_identity import load_primary_model


def swapped_engine():
    engine = MagicMock(model_id="primary")
    engine.load.side_effect = lambda: setattr(engine, "model_id", "fallback")
    return engine


def test_actual_mlx_startup_fallback_is_rejected_before_measurement(monkeypatch):
    import engines.mlx_engine as mlx_engine

    transcribe = MagicMock(side_effect=[ValueError("primary failed to load"), {"text": ""}])
    monkeypatch.setattr(mlx_engine, "MLX_AVAILABLE", True)
    monkeypatch.setattr(mlx_engine, "mx", MagicMock())
    monkeypatch.setattr(mlx_engine, "mlx_whisper", SimpleNamespace(transcribe=transcribe))
    monkeypatch.setattr(mlx_engine, "resolve_model_path", lambda name: "/installed/" + name)
    engine = mlx_engine.MLXWhisperEngine(model_id="primary", fallback_model_id="fallback", fallback_on_low_conf=False)
    with pytest.raises(RuntimeError, match="requested 'primary', loaded 'fallback'"):
        load_primary_model(engine, "primary")
    assert transcribe.call_count == 2  # only startup attempts; no benchmark samples
    assert not engine._loaded


def test_matching_identity_is_recorded_without_resolving_away_explicit_override():
    engine = MagicMock(model_id="/chosen/primary")
    assert load_primary_model(engine, "/chosen/primary") == {
        "requested_model_id": "/chosen/primary",
        "actual_model_id": "/chosen/primary",
        "primary_identity_verified": True,
    }
    engine.unload.assert_not_called()


def test_load_failure_keeps_original_error_even_if_cleanup_fails():
    engine = MagicMock()
    engine.load.side_effect = ValueError("original loader failure")
    engine.unload.side_effect = RuntimeError("cleanup failure")
    with pytest.raises(ValueError, match="original loader failure"):
        load_primary_model(engine, "primary")


@pytest.mark.parametrize("load_error", [False, True])
def test_variant_failure_stops_sampler_and_never_writes_measurements(tmp_path, monkeypatch, load_error):
    engine = swapped_engine()
    if load_error:
        engine.load.side_effect = ValueError("no primary")
    sampler = MagicMock()
    monkeypatch.setattr(benchmark_stt_engines, "build_engine", MagicMock(return_value=engine))
    monkeypatch.setattr(benchmark_stt_engines, "VramSampler", MagicMock(return_value=sampler))
    audio = MagicMock()
    monkeypatch.setattr(benchmark_stt_engines, "load_pcm16_wav", audio)
    output = tmp_path / "clips.jsonl"
    with pytest.raises((RuntimeError, ValueError)):
        benchmark_stt_engines.run_variant("fake", {"model_id": "primary", "device": "cuda"}, [], 1, 1, output, None)
    sampler.start.assert_called_once()
    sampler.stop.assert_called_once()
    engine.unload.assert_called_once()
    engine.transcribe.assert_not_called()
    audio.assert_not_called()
    assert not output.exists()


def test_acceleration_stt_arm_marks_fallback_ineligible(monkeypatch):
    engine = swapped_engine()
    constructor = MagicMock(return_value=engine)
    monkeypatch.setattr("engines.mlx_engine.MLXWhisperEngine", constructor)
    report = benchmark_mlx_accel.bench_stt_baseline(1, 1, "primary", duration_s=0.01)
    assert report["eligible"] is False
    assert report["model_id"] == "primary" and report["actual_model_id"] == "fallback"
    assert "identity mismatch" in report["error"] and "latency" not in report
    assert constructor.call_args.kwargs["fallback_on_low_conf"] is False
    engine.unload.assert_called_once()
    engine.transcribe.assert_not_called()


def test_roundtrip_stt_fallback_fails_before_any_measured_item(tmp_path, monkeypatch):
    engine = swapped_engine()
    monkeypatch.setattr(stt_roundtrip_compare, "make_engine", lambda _: engine)
    monkeypatch.setattr(stt_roundtrip_compare, "build_items", lambda *_: [{"id": "one"}])
    monkeypatch.setattr(stt_roundtrip_compare, "synthesize", lambda *_: [np.zeros(160)])
    output = tmp_path / "roundtrip.json"
    monkeypatch.setattr("sys.argv", ["roundtrip", "--engines", "mlx", "--output", str(output)])
    with pytest.raises(RuntimeError, match="identity mismatch"):
        stt_roundtrip_compare.main()
    engine.transcribe.assert_not_called()
    engine.unload.assert_called_once()
    assert not output.exists()
