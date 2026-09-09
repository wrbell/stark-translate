"""Parakeet API adapters tested with fake alignments and mocked MLX only."""

import logging
import math
import sys
import zlib
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from engines.base import text_compression_ratio
from engines.parakeet_mlx_engine import ParakeetMLXEngine


def token(text, confidence=1.0, start=0.0, duration=0.1):
    return SimpleNamespace(id=1, text=text, confidence=confidence, start=start, end=start + duration)


def result(*sentences):
    aligned = [
        SimpleNamespace(text="".join(t.text for t in tokens), tokens=tokens, start=tokens[0].start, end=tokens[-1].end)
        for tokens in sentences
    ]
    return SimpleNamespace(text="".join(s.text for s in aligned), sentences=aligned)


@pytest.fixture
def api(monkeypatch):
    mx = MagicMock()
    mx.array.side_effect = lambda audio: audio
    monkeypatch.setattr(sys.modules["mlx"], "core", mx)
    monkeypatch.setitem(sys.modules, "mlx.core", mx)
    # The shared materialization helper captures mx at module import time.
    monkeypatch.setattr("engines.mlx_engine.mx", mx)
    model = MagicMock()
    model.generate.return_value = [result()]
    loader = MagicMock(return_value=model)
    mel = MagicMock()
    monkeypatch.setattr(sys.modules["parakeet_mlx"], "from_pretrained", loader)
    monkeypatch.setattr(sys.modules["parakeet_mlx.audio"], "get_logmel", mel)
    return SimpleNamespace(mx=mx, model=model, loader=loader, mel=mel)


@pytest.fixture
def engine(api):
    engine = ParakeetMLXEngine(warmup_seconds=0)
    engine.load()
    return engine


def test_load_materializes_and_warms_same_transcribe_path(api):
    engine = ParakeetMLXEngine(warmup_seconds=0.25, cache_limit_mb=32)
    engine.load()
    api.loader.assert_called_once_with(engine.model_id, dtype=api.mx.bfloat16)
    api.mx.set_cache_limit.assert_called_once_with(32 * 1024 * 1024)
    audio, config = api.mel.call_args.args
    assert audio.dtype == np.float32
    assert audio.shape == (4000,)
    assert not audio.any()
    assert config is api.model.preprocessor_config
    api.model.generate.assert_called_once_with(api.mel.return_value)
    api.mx.eval.assert_called_once_with(api.model.parameters.return_value)
    assert api.mx.method_calls[-1][0] == "synchronize"
    engine.load()
    api.loader.assert_called_once()
    assert engine.backend == "parakeet-mlx"


def test_text_metrics_segments_and_words(engine, api):
    api.model.generate.return_value = [
        result([token(" Pro", 0.25, 0.1), token("pitiation", 0.64, 0.2)], [token(" saves.", 0.9, 0.5)])
    ]
    out = engine.transcribe(np.ones(160, dtype=np.float64), language="es", word_timestamps=True)
    assert out.text == "Propitiation saves."
    expected = (math.log(0.25) + math.log(0.64) + math.log(0.9)) / 3
    assert out.avg_logprob == pytest.approx(expected)
    assert out.confidence == pytest.approx(max(0, 1 + expected))
    assert out.no_speech_prob == 0.0
    assert out.compression_ratio == text_compression_ratio(out.text)
    assert out.latency_ms >= 0
    assert len(out.segments) == 2
    first = out.segments[0]
    assert set(first) == {"avg_logprob", "no_speech_prob", "compression_ratio", "start", "end", "text", "words"}
    assert first["avg_logprob"] == pytest.approx((math.log(0.25) + math.log(0.64)) / 2)
    assert first["no_speech_prob"] == 0.0
    assert first["compression_ratio"] == text_compression_ratio("Propitiation")
    assert first["start"] == 0.1
    assert first["end"] == pytest.approx(0.3)
    word = first["words"][0]
    assert word["word"] == "Propitiation"
    assert word["probability"] == pytest.approx(0.4)
    assert word["start"] == 0.1
    assert word["end"] == pytest.approx(0.3)
    assert out.low_confidence_words == [word]
    assert api.mel.call_args.args[0].dtype == np.float32


@pytest.mark.parametrize("probability,expected", [(1.0, 1.0), (0.9, 1 + math.log(0.9)), (0, 0.0)])
def test_confidence_clipping_and_log_floor(engine, api, probability, expected):
    api.model.generate.return_value = [result([token("Grace", probability)])]
    out = engine.transcribe(np.zeros(32, dtype=np.float32))
    assert out.confidence == pytest.approx(expected)
    assert out.avg_logprob == pytest.approx(math.log(max(probability, 1e-6)))
    assert "words" not in out.segments[0]
    assert out.no_speech_prob == 0.0


@pytest.mark.parametrize("amplitude,expected", [(0.0, None), (0.007, None), (0.009, 1.0)])
def test_no_tokens_speech_proxy(engine, amplitude, expected):
    out = engine.transcribe(np.full(160, amplitude, dtype=np.float32))
    assert out.no_speech_prob == expected
    assert out.avg_logprob is None
    assert out.confidence is None
    assert out.text == ""
    assert out.compression_ratio == 0
    assert out.segments == []
    assert out.low_confidence_words == []


def test_empty_audio_proxy(engine):
    assert engine.transcribe(np.array([], dtype=np.float32)).no_speech_prob is None


@pytest.mark.parametrize("text", ["", "grace " * 100, "Jesús salva."])
def test_compression_uses_utf8_bytes(text):
    raw = text.encode("utf-8")
    assert text_compression_ratio(text) == len(raw) / len(zlib.compress(raw))


def test_prompt_warning_once(engine, caplog):
    with caplog.at_level(logging.WARNING):
        for prompt in (None, "atonement", "grace"):
            engine.transcribe(np.zeros(10, dtype=np.float32), initial_prompt=prompt)
    assert [r.message for r in caplog.records] == [
        "Parakeet has no prompt biasing; theological terms rely on correct_stt_output"
    ]


@pytest.mark.parametrize(
    "overrides,expected", [({}, ((64, 96), 2)), ({"context_size": (8, 16), "depth": 3}, ((8, 16), 3))]
)
def test_stream_delegates_and_closes_once(api, overrides, expected):
    engine = ParakeetMLXEngine(warmup_seconds=0, stream_context=(64, 96), stream_depth=2)
    engine.load()
    context = api.model.transcribe_stream.return_value
    sp = context.__enter__.return_value
    sp.result = result([token(" Hello"), token(" world")])
    sp.finalized_tokens = [token(" Hello")]
    sp.draft_tokens = [token(" world")]
    stream = engine.open_stream(**overrides)
    with stream:
        for length in (16, 32):
            stream.add_audio(np.zeros(length, dtype=np.float64))
        assert stream.text == " Hello world"
        assert stream.finalized_text == "Hello"
        assert stream.tokens == sp.finalized_tokens + sp.draft_tokens
    api.model.transcribe_stream.assert_called_once_with(context_size=expected[0], depth=expected[1])
    context.__enter__.assert_called_once()
    assert [c.args[0].shape for c in sp.add_audio.call_args_list] == [(16,), (32,)]
    assert all(c.args[0].dtype == np.float32 for c in sp.add_audio.call_args_list)
    stream.close()
    context.__exit__.assert_called_once_with(None, None, None)
    with pytest.raises(RuntimeError, match="closed"):
        stream.add_audio(np.zeros(1, dtype=np.float32))


def test_stream_closes_on_exception(engine, api):
    with pytest.raises(ValueError, match="test error"), engine.open_stream():
        raise ValueError("test error")
    args = api.model.transcribe_stream.return_value.__exit__.call_args.args
    assert args[0] is ValueError


def test_unloaded_guards_and_failed_warmup(api):
    engine = ParakeetMLXEngine()
    with pytest.raises(RuntimeError, match="load"):
        engine.transcribe(np.zeros(16, dtype=np.float32))
    with pytest.raises(RuntimeError, match="load"):
        engine.open_stream()
    api.model.generate.side_effect = ValueError("warmup failed")
    with pytest.raises(ValueError, match="warmup failed"):
        engine.load()
    assert engine._model is None


def test_unload(engine, api):
    engine.unload()
    assert engine._model is None
    api.mx.clear_cache.assert_called_once()
    with pytest.raises(RuntimeError, match="load"):
        engine.open_stream()


@pytest.mark.parametrize("audio", [np.zeros((2, 16), dtype=np.float32), np.zeros(16, dtype=np.int16)])
def test_rejects_non_mono_or_integer_audio(engine, audio):
    with pytest.raises(ValueError, match="Parakeet requires"):
        engine.transcribe(audio)
