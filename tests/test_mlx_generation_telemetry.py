"""Mocked MLX response accounting, streaming callbacks, and pipeline exports."""

import csv
import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from engines.mlx_engine import MLXGemmaEngine, generate_translation


def responses(finish_reason="stop", prompt_tps=200):
    return [
        SimpleNamespace(
            text=text,
            token=i,
            prompt_tokens=10,
            prompt_tps=prompt_tps,
            generation_tokens=i,
            generation_tps=40.0,
            from_draft=i in (1, 3),
            finish_reason=finish_reason if i == 4 else None,
        )
        for i, text in enumerate(["Hola", " mundo", ".", "<turn|>"], 1)
    ]


@pytest.fixture
def engine():
    with patch("engines.mlx_engine.MLX_AVAILABLE", True):
        engine = MLXGemmaEngine(use_prompt_cache=False)
    engine._loaded = True
    engine._model = object()
    engine._tokenizer = MagicMock()
    engine._tokenizer.apply_chat_template.return_value = [1, 2]
    return engine


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("finish_reason", ["stop", "length"])
def test_generation_telemetry(engine, streaming, finish_reason):
    callback = MagicMock()
    with (
        patch("mlx_lm.stream_generate", return_value=iter(responses(finish_reason))) as gen,
        patch("engines.mlx_engine.time.perf_counter", side_effect=[1.0, 1.1, 1.5]),
    ):
        result = (
            engine.translate_streaming("hello", token_callback=callback) if streaming else engine.translate("hello")
        )
    assert result.text == "Hola mundo."
    assert result.latency_ms == pytest.approx(500)
    assert result.prompt_tokens == 10
    assert result.generated_tokens == 4
    assert result.prefill_ms == 50
    assert result.ttft_ms == pytest.approx(100)
    assert result.decode_ms == pytest.approx(400)
    assert result.tokens_per_second == 40
    assert result.finish_reason == finish_reason
    assert result.draft_tokens == 2
    assert result.draft_accept_rate == 0.5
    engine._tokenizer.encode.assert_not_called()
    assert gen.call_args.kwargs == {"prompt": [1, 2], "max_tokens": 64}
    if streaming:
        callback.assert_called_once_with("Hola mundo.", 3)


def test_streaming_custom_batch_cleans_partial(engine):
    callback = MagicMock()
    with patch("mlx_lm.stream_generate", return_value=iter(responses())):
        engine.translate_streaming("hello", token_callback=callback, batch_size=2)
    assert [call.args for call in callback.call_args_list] == [("Hola mundo", 2), ("Hola mundo.", 4)]


def test_cache_and_draft_kwargs(engine):
    engine._model_family = "translategemma"
    engine._prompt_cache_template = [{"state": [1]}]
    engine._suffix_tokens = [8]
    engine._tokenizer.encode.return_value = [7]
    with patch("mlx_lm.stream_generate", side_effect=lambda *a, **kw: iter(responses())) as gen:
        engine.translate("hello")
        assert gen.call_args.kwargs["prompt"] == [7, 8]
        assert gen.call_args.kwargs["prompt_cache"] == engine._prompt_cache_template
        assert gen.call_args.kwargs["prompt_cache"] is not engine._prompt_cache_template
        engine._draft_model = object()
        engine._num_draft_tokens = 2
        engine.translate_streaming("hello")
        assert gen.call_args.kwargs["draft_model"] is engine._draft_model
        assert gen.call_args.kwargs["num_draft_tokens"] == 2
        assert "prompt_cache" not in gen.call_args.kwargs


def test_zero_throughput_and_empty_stream(engine):
    with patch("mlx_lm.stream_generate", return_value=iter(responses(prompt_tps=0))):
        assert engine.translate("hello").prefill_ms is None
    with patch("mlx_lm.stream_generate", return_value=iter([])):
        result = engine.translate("hello")
    assert result.text == ""
    assert result.ttft_ms is None
    assert result.decode_ms is None
    assert result.generated_tokens is None
    assert result.finish_reason is None
    assert result.draft_accept_rate is None


def test_invalid_batch(engine):
    with pytest.raises(ValueError, match="batch_size"):
        engine.translate_streaming("hello", batch_size=0)


@pytest.mark.parametrize("streaming", [False, True])
def test_pipeline_telemetry_and_exports(monkeypatch, tmp_path, streaming):
    import dry_run_ab as d

    monkeypatch.setattr(d, "MODEL_FAMILY", "gemma4")
    monkeypatch.setattr(d, "_last_gen_stats", {})
    monkeypatch.setattr(d, "CSV_PATH", str(tmp_path / "out.csv"))
    monkeypatch.setattr(d, "DIAG_PATH", str(tmp_path / "diag.jsonl"))
    tok = MagicMock()
    with (
        patch("mlx_lm.stream_generate", return_value=iter(responses())) as gen,
        patch.object(d, "_enqueue_stream_token") as enqueue,
    ):
        if streaming:
            text, latency, tps = d.translate_mlx_streaming(object(), tok, "hello", 42)
            enqueue.assert_called_once_with(("token", 42, "Hola mundo.", 3))
        else:
            text, latency, tps = d.translate_mlx(object(), tok, "hello", chunk_id=42)
    assert gen.call_args.kwargs["max_tokens"] == 64
    stats = d._last_gen_stats.pop(42)
    assert stats["gen_tokens_a"] == 4
    assert stats["prefill_ms_a"] == 50
    assert stats["finish_reason_a"] == "stop"
    assert stats["draft_accept_a"] == 0.5
    data = dict(
        chunk_id=42,
        timestamp="now",
        english="hello",
        spanish_a=text,
        spanish_b=None,
        stt_latency_ms=0,
        latency_a_ms=latency,
        latency_b_ms=0,
        e2e_latency_ms=latency,
        tps_a=tps,
        **stats,
    )
    d.init_csv()
    d.write_csv_row(data)
    d.write_diag_jsonl(data, "audio.wav")
    with open(d.CSV_PATH) as f:
        row = next(csv.DictReader(f))
    assert list(row)[-8:] == list(d._GEN_STAT_FIELDS)
    assert row["gen_tokens_a"] == "4"
    record = json.loads((tmp_path / "diag.jsonl").read_text())
    assert {key: record[key] for key in stats} == stats


def test_missing_response_metadata_is_safe():
    with patch("mlx_lm.stream_generate", return_value=iter([SimpleNamespace(text="Hola", generation_tokens=1)])):
        result = generate_translation(object(), object(), model_family="gemma4", gen_kwargs={})
    assert result.prefill_ms is None
    assert result.prompt_tokens is None
    assert result.finish_reason is None
    assert result.tokens_per_second == 0
