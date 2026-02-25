"""Tests for engine utility methods."""

from engines.base import STTResult


class TestShouldFallback:
    """Tests for MLXWhisperEngine._should_fallback."""

    def _make_engine(self, fallback_threshold=-1.2, hallucination_threshold=2.4):
        from engines.mlx_engine import MLXWhisperEngine

        engine = MLXWhisperEngine.__new__(MLXWhisperEngine)
        engine._fallback_threshold = fallback_threshold
        engine._hallucination_threshold = hallucination_threshold
        return engine

    def test_good_result_no_fallback(self):
        engine = self._make_engine()
        result = STTResult(text="hello", latency_ms=100, avg_logprob=-0.3, compression_ratio=1.5)
        assert engine._should_fallback(result) is False

    def test_low_logprob_triggers_fallback(self):
        engine = self._make_engine()
        result = STTResult(text="hello", latency_ms=100, avg_logprob=-1.5, compression_ratio=1.5)
        assert engine._should_fallback(result) is True

    def test_high_compression_triggers_fallback(self):
        engine = self._make_engine()
        result = STTResult(text="hello", latency_ms=100, avg_logprob=-0.3, compression_ratio=3.0)
        assert engine._should_fallback(result) is True

    def test_both_bad_triggers_fallback(self):
        engine = self._make_engine()
        result = STTResult(text="hello", latency_ms=100, avg_logprob=-1.5, compression_ratio=3.0)
        assert engine._should_fallback(result) is True

    def test_none_values_no_fallback(self):
        engine = self._make_engine()
        result = STTResult(text="hello", latency_ms=100, avg_logprob=None, compression_ratio=None)
        assert engine._should_fallback(result) is False

    def test_exactly_at_logprob_threshold_no_fallback(self):
        engine = self._make_engine()
        result = STTResult(text="hello", latency_ms=100, avg_logprob=-1.2, compression_ratio=1.5)
        assert engine._should_fallback(result) is False

    def test_exactly_at_compression_threshold_no_fallback(self):
        engine = self._make_engine()
        result = STTResult(text="hello", latency_ms=100, avg_logprob=-0.3, compression_ratio=2.4)
        assert engine._should_fallback(result) is False

    def test_custom_thresholds(self):
        engine = self._make_engine(fallback_threshold=-0.5, hallucination_threshold=1.8)
        # avg_logprob=-0.6 is below custom threshold of -0.5
        result = STTResult(text="hello", latency_ms=100, avg_logprob=-0.6, compression_ratio=1.5)
        assert engine._should_fallback(result) is True

    def test_custom_compression_threshold(self):
        engine = self._make_engine(fallback_threshold=-0.5, hallucination_threshold=1.8)
        # compression_ratio=2.0 exceeds custom threshold of 1.8
        result = STTResult(text="hello", latency_ms=100, avg_logprob=-0.3, compression_ratio=2.0)
        assert engine._should_fallback(result) is True

    def test_none_logprob_with_bad_compression(self):
        engine = self._make_engine()
        result = STTResult(text="hello", latency_ms=100, avg_logprob=None, compression_ratio=3.0)
        assert engine._should_fallback(result) is True

    def test_bad_logprob_with_none_compression(self):
        engine = self._make_engine()
        result = STTResult(text="hello", latency_ms=100, avg_logprob=-1.5, compression_ratio=None)
        assert engine._should_fallback(result) is True


class TestPickBest:
    """Tests for MLXWhisperEngine._pick_best."""

    def _pick_best(self, primary, retry):
        from engines.mlx_engine import MLXWhisperEngine

        return MLXWhisperEngine._pick_best(primary, retry)

    def test_retry_better_logprob_returns_retry(self):
        primary = STTResult(text="primary", latency_ms=100, avg_logprob=-0.8)
        retry = STTResult(text="retry", latency_ms=120, avg_logprob=-0.3)
        best, label = self._pick_best(primary, retry)
        assert best is retry
        assert label == "retry"

    def test_primary_better_logprob_returns_primary(self):
        primary = STTResult(text="primary", latency_ms=100, avg_logprob=-0.3)
        retry = STTResult(text="retry", latency_ms=120, avg_logprob=-0.8)
        best, label = self._pick_best(primary, retry)
        assert best is primary
        assert label == "original"

    def test_equal_logprobs_returns_primary(self):
        primary = STTResult(text="primary", latency_ms=100, avg_logprob=-0.5)
        retry = STTResult(text="retry", latency_ms=120, avg_logprob=-0.5)
        best, label = self._pick_best(primary, retry)
        assert best is primary
        assert label == "original"

    def test_primary_none_logprob_returns_retry(self):
        primary = STTResult(text="primary", latency_ms=100, avg_logprob=None)
        retry = STTResult(text="retry", latency_ms=120, avg_logprob=-0.5)
        best, label = self._pick_best(primary, retry)
        assert best is retry
        assert label == "retry"

    def test_retry_none_logprob_returns_primary(self):
        primary = STTResult(text="primary", latency_ms=100, avg_logprob=-0.5)
        retry = STTResult(text="retry", latency_ms=120, avg_logprob=None)
        best, label = self._pick_best(primary, retry)
        assert best is primary
        assert label == "original"

    def test_both_none_logprob_returns_primary(self):
        primary = STTResult(text="primary", latency_ms=100, avg_logprob=None)
        retry = STTResult(text="retry", latency_ms=120, avg_logprob=None)
        best, label = self._pick_best(primary, retry)
        assert best is primary
        assert label == "original"


class TestMarianEngineInit:
    """Tests for MarianEngine constructor and property methods."""

    def _make_engine(self, model_id="Helsinki-NLP/opus-mt-en-es", device="auto"):
        from engines.mlx_engine import MarianEngine

        return MarianEngine(model_id=model_id, device=device)

    def test_default_constructor_values(self):
        engine = self._make_engine()
        assert engine._model_id_str == "Helsinki-NLP/opus-mt-en-es"
        assert engine._requested_device == "auto"
        assert engine._device is None
        assert engine._model is None
        assert engine._tokenizer is None
        assert engine._loaded is False

    def test_custom_model_id(self):
        engine = self._make_engine(model_id="Helsinki-NLP/opus-mt-en-fr")
        assert engine._model_id_str == "Helsinki-NLP/opus-mt-en-fr"

    def test_custom_device(self):
        engine = self._make_engine(device="cpu")
        assert engine._requested_device == "cpu"

    def test_unload_resets_state(self):
        engine = self._make_engine()
        # Simulate loaded state
        engine._model = "fake_model"
        engine._tokenizer = "fake_tokenizer"
        engine._loaded = True
        engine.unload()
        assert engine._model is None
        assert engine._tokenizer is None
        assert engine._loaded is False

    def test_model_id_property(self):
        engine = self._make_engine(model_id="Helsinki-NLP/opus-mt-en-es")
        assert engine.model_id == "Helsinki-NLP/opus-mt-en-es"

    def test_backend_property_defaults_to_cpu(self):
        engine = self._make_engine()
        # _device is None after construction
        assert engine.backend == "cpu"

    def test_backend_property_returns_device_when_set(self):
        engine = self._make_engine()
        engine._device = "mps"
        assert engine.backend == "mps"

    def test_backend_property_returns_cuda_when_set(self):
        engine = self._make_engine()
        engine._device = "cuda"
        assert engine.backend == "cuda"
