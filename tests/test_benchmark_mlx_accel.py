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
        draft_tokens_mean=None,
        draft_accept_rate_mean=None,
        draft_accept_rate_min=None,
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


@pytest.mark.parametrize("gamma", [1, 2, 3])
def test_e2b_draft_configs(gamma):
    config = b.CONFIGS[f"e4b_e2b_draft_g{gamma}"]
    assert config == {
        "label": f"Gemma 4 E4B OptiQ + E2B OptiQ draft (gamma={gamma})",
        "model_id": b.CONFIGS["e4b"]["model_id"],
        "model_family": "gemma4",
        "draft_model_id": "mlx-community/gemma-4-e2b-it-OptiQ-4bit",
        "num_draft_tokens": gamma,
        "turboquant": False,
    }


def test_draft_summary_preserves_unknown_and_observed_zero():
    from types import SimpleNamespace

    missing = b._generation_metrics(SimpleNamespace())
    assert missing["draft_tokens"] is None
    assert missing["draft_accept_rate"] is None
    summary = b._generation_summary(
        [
            {"draft_tokens": 0, "draft_accept_rate": 0.0},
            {"draft_tokens": 6, "draft_accept_rate": 0.6},
            missing,
        ]
    )
    assert summary["draft_tokens_mean"] == 3
    assert summary["draft_accept_rate_mean"] == 0.3
    assert summary["draft_accept_rate_min"] == 0
    for rows in ([], [missing]):
        summary = b._generation_summary(rows)
        assert summary["draft_tokens_mean"] is None
        assert summary["draft_accept_rate_mean"] is None
        assert summary["draft_accept_rate_min"] is None


@pytest.mark.parametrize("variation", ["none", "between_configs", "between_runs", "load_failure"])
def test_benchmark_full_text_identity(monkeypatch, variation):
    import hashlib

    monkeypatch.setattr(b, "TEST_SENTENCES", {"short": "hello"})
    monkeypatch.setattr(b, "CANARIES", [("grace", "gracia")])
    full = "gracia " * 40 + "á"
    different = full + " "  # Same truncated sample; byte identity must still fail.
    engines = [MagicMock(), MagicMock()]
    for i, engine in enumerate(engines):
        outputs = [full, full, full]
        if i and variation == "between_configs":
            outputs = [different] * 3
        elif i and variation == "between_runs":
            outputs[1] = different
        elif i and variation == "load_failure":
            engine.load.side_effect = RuntimeError("unavailable")
        engine.translate.side_effect = [TranslationResult(t, 100, finish_reason="stop") for t in outputs]
    with patch("engines.mlx_engine.MLXGemmaEngine", side_effect=engines) as factory:
        result = b.bench_mlx_gemma4_accel(configs=["e4b", "e4b_e2b_draft_g2"], runs=2, warmup=0, include_stt=False)
    assert result["identity"] == {"short": variation == "none"}
    baseline = result["configs"]["e4b"]
    metrics = baseline["per_length"]["short"]
    digest = hashlib.sha256(full.encode("utf-8")).hexdigest()
    assert metrics["texts"] == [full, full]
    assert metrics["text_sha256s"] == [digest, digest]
    assert metrics["all_identical"] is True
    assert [r["text_sha256"] for r in metrics["generation"]] == [digest, digest]
    assert baseline["canary_details"][0]["text_sha256"] == digest
    assert baseline["canary_details"][0]["out"] == full
    if variation == "between_runs":
        assert result["configs"]["e4b_e2b_draft_g2"]["per_length"]["short"]["all_identical"] is False
    assert factory.call_args.kwargs["draft_model_id"] == b.CONFIGS["e2b"]["model_id"]
    assert factory.call_args.kwargs["num_draft_tokens"] == 2


def test_cli_selects_canary_sources(monkeypatch, tmp_path):
    import json

    from training.theological_canaries import canary_sentences

    path = tmp_path / "bench.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "benchmark_mlx_accel.py",
            "--sentences",
            "canaries",
            "--configs",
            "e4b",
            "--runs",
            "1",
            "--warmup",
            "0",
            "--no-stt",
            "--output",
            str(path),
        ],
    )
    engine = MagicMock()
    engine.translate.return_value = TranslationResult("gracia", 100, finish_reason="stop")
    with patch("engines.mlx_engine.MLXGemmaEngine", return_value=engine):
        b.main()
    result = json.loads(path.read_text())
    expected = {f"canary_{i:02d}": c["en"] for i, c in enumerate(canary_sentences(), 1)}
    assert result["sentences"] == expected
    assert set(result["configs"]["e4b"]["per_length"]) == set(expected)
    assert [call.args[0] for call in engine.translate.call_args_list[: len(expected)]] == list(expected.values())
    assert "identity" not in result
    assert b.select_sentences("all") == {**b.TEST_SENTENCES, **expected}
    assert b.select_sentences() == b.TEST_SENTENCES


def test_metal_peak_is_reset_and_recorded_before_unload(monkeypatch):
    import mlx.core as mx

    events = []
    monkeypatch.setattr(mx, "reset_peak_memory", lambda: events.append("reset"))

    def peak():
        events.append("peak")
        return 123 * 1024 * 1024

    monkeypatch.setattr(mx, "get_peak_memory", peak)
    engine = MagicMock()
    engine.load.side_effect = lambda: events.append("load")
    engine.unload.side_effect = lambda: events.append("unload")
    engine.translate.return_value = TranslationResult("gracia", 100)
    with patch("engines.mlx_engine.MLXGemmaEngine", return_value=engine):
        result = b.bench_config("e4b", 1, 0)
    assert result["metal_peak_mib"] == 123
    assert events == ["reset", "load", "peak", "unload"]
    monkeypatch.setattr(mx, "get_peak_memory", MagicMock(side_effect=RuntimeError("unavailable")))
    assert b._metal_peak_mib() is None
    monkeypatch.setitem(__import__("sys").modules, "mlx.core", None)
    assert b._metal_peak_mib() is None


def test_canary_only_selection_cannot_invent_medium_overlap_latency():
    result = b.bench_e2e_overlap_proxy({"per_length": {"canary_01": {}}}, {"latency": {"p50": 100}})
    assert result == {"error": "overlap proxy requires the default medium sentence"}
