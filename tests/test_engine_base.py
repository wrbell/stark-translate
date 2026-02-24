"""Tests for engines/base.py — dataclasses and ABC contracts."""

import numpy as np
import pytest

from engines.base import STTEngine, STTResult, TranslationEngine, TranslationResult


class TestSTTResult:
    def test_required_fields(self):
        r = STTResult(text="hello", latency_ms=100.0)
        assert r.text == "hello"
        assert r.latency_ms == 100.0

    def test_optional_defaults(self):
        r = STTResult(text="", latency_ms=0.0)
        assert r.confidence is None
        assert r.avg_logprob is None
        assert r.compression_ratio is None
        assert r.segments == []
        assert r.low_confidence_words == []
        assert r.used_fallback is False

    def test_all_fields(self):
        r = STTResult(
            text="test",
            latency_ms=50.0,
            confidence=0.95,
            avg_logprob=-0.3,
            compression_ratio=1.5,
            segments=[{"text": "test"}],
            low_confidence_words=["test"],
            used_fallback=True,
        )
        assert r.confidence == 0.95
        assert r.used_fallback is True
        assert len(r.segments) == 1
        assert len(r.low_confidence_words) == 1


class TestTranslationResult:
    def test_required_fields(self):
        r = TranslationResult(text="hola", latency_ms=200.0)
        assert r.text == "hola"
        assert r.latency_ms == 200.0

    def test_optional_defaults(self):
        r = TranslationResult(text="", latency_ms=0.0)
        assert r.tokens_per_second == 0.0
        assert r.qe_score is None

    def test_all_fields(self):
        r = TranslationResult(text="hola", latency_ms=100.0, tokens_per_second=50.0, qe_score=0.9)
        assert r.tokens_per_second == 50.0
        assert r.qe_score == 0.9


class TestSTTEngineABC:
    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            STTEngine()

    def test_subclass_must_implement_all(self):
        class PartialEngine(STTEngine):
            def load(self):
                pass

        with pytest.raises(TypeError):
            PartialEngine()

    def test_complete_subclass(self):
        class ConcreteSTT(STTEngine):
            def load(self):
                pass

            def transcribe(self, audio: np.ndarray, **kwargs) -> STTResult:
                return STTResult(text="test", latency_ms=0.0)

            def unload(self):
                pass

            @property
            def model_id(self) -> str:
                return "test-model"

            @property
            def backend(self) -> str:
                return "cpu"

        engine = ConcreteSTT()
        assert engine.model_id == "test-model"
        assert engine.backend == "cpu"


class TestTranslationEngineABC:
    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            TranslationEngine()

    def test_complete_subclass(self):
        class ConcreteTranslation(TranslationEngine):
            def load(self):
                pass

            def translate(self, text: str, **kwargs) -> TranslationResult:
                return TranslationResult(text="hola", latency_ms=0.0)

            def unload(self):
                pass

            @property
            def model_id(self) -> str:
                return "test-model"

            @property
            def backend(self) -> str:
                return "cpu"

        engine = ConcreteTranslation()
        assert engine.model_id == "test-model"
