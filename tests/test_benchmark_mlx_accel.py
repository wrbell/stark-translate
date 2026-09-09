"""MLX benchmark aggregation tests without loading models."""

from unittest.mock import MagicMock, patch

import pytest

from engines.base import TranslationResult
from tools import benchmark_mlx_accel as b


def test_generation_summary_none_safe():
    rows = [
        dict(generated_tokens=10, prefill_ms=20, ttft_ms=30, decode_ms=40, finish_reason="stop"),
        dict(generated_tokens=30, prefill_ms=40, ttft_ms=50, decode_ms=60, finish_reason="length"),
        dict(generated_tokens=None, prefill_ms=None, ttft_ms=None, decode_ms=None, finish_reason=None),
    ]
    summary = b._generation_summary(rows)
    assert summary == dict(
        gen_tokens_mean=20,
        prefill_p50=20,
        ttft_p50=30,
        decode_p50=40,
        finish_reason_counts={"stop": 1, "length": 1},
        pct_hit_max_tokens=0.5,
    )
    assert not b._stops_before_max(rows)
    assert b._stops_before_max(rows[:1])
    assert not b._stops_before_max([rows[0], rows[2]])
    for rows in ([], [{}]):
        summary = b._generation_summary(rows)
        assert summary["gen_tokens_mean"] is None
        assert summary["prefill_p50"] is None
        assert summary["pct_hit_max_tokens"] is None
        assert summary["finish_reason_counts"] == {}
        assert not b._stops_before_max(rows)


@pytest.mark.parametrize("canary_reason,gate", [("stop", True), ("length", False), (None, False)])
def test_bench_config_includes_canaries(monkeypatch, canary_reason, gate):
    monkeypatch.setattr(b, "TEST_SENTENCES", {"short": "hello"})
    monkeypatch.setattr(b, "CANARIES", [("grace", "gracia")])
    engine = MagicMock()
    measured = TranslationResult(
        "Hola", 100, 40, generated_tokens=4, prefill_ms=10, ttft_ms=20, decode_ms=80, finish_reason="stop"
    )
    canary = TranslationResult("gracia", 100, finish_reason=canary_reason)
    engine.translate.side_effect = [measured, measured, canary]
    with patch("engines.mlx_engine.MLXGemmaEngine", return_value=engine):
        result = b.bench_config("e4b", runs=2, warmup=0)
    assert result["gate_stops_before_max"] is gate
    metrics = result["per_length"]["short"]
    assert metrics["gen_tokens_mean"] == 4
    assert metrics["prefill_p50"] == 10
    assert metrics["ttft_p50"] == 20
    assert metrics["decode_p50"] == 80
    assert metrics["finish_reason_counts"] == {"stop": 2}
    assert metrics["pct_hit_max_tokens"] == 0
    assert result["canary_details"][0]["finish_reason"] == canary_reason
    engine.unload.assert_called_once()


def test_environment_and_top_level_gate():
    with (
        patch.object(b.metadata, "version", side_effect=["1", "2", b.metadata.PackageNotFoundError()]),
        patch.object(b, "bench_config", return_value={"gate_stops_before_max": True}),
    ):
        result = b.bench_mlx_gemma4_accel(configs=["e4b"], include_stt=False)
    assert result["env"]["mlx"] == "1"
    assert result["env"]["mlx_lm"] == "2"
    assert result["env"]["optiq"] is None
    assert result["env"]["python"]
    assert result["gate_stops_before_max"] is True
