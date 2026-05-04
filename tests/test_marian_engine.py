"""Tests for the v2026.8 MarianMT engine refactor.

Covers:
  - Constructor / property contract for MarianHFEngine and MarianCT2Engine.
  - Backend property string format (used by dry_run_ab.py for the CSV/JSONL
    diagnostic field).
  - unload() resets state.

Loading + translation are NOT exercised here — those depend on the actual
HF/CT2 models which ``conftest.py`` mocks out for CI. The bench harness +
``scripts/convert_marian_ct2.py``'s sanity gate cover the live path.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class TestMarianHFEngineContract:
    """MarianHFEngine constructor and property methods."""

    def _make(self, **overrides):
        from engines.marian_hf_engine import MarianHFEngine

        defaults: dict = {"model_id": "Helsinki-NLP/opus-mt-en-es", "device": "auto"}
        defaults.update(overrides)
        return MarianHFEngine(**defaults)

    def test_default_constructor_state(self):
        engine = self._make()
        assert engine._model_id_str == "Helsinki-NLP/opus-mt-en-es"
        assert engine._requested_device == "auto"
        assert engine._device is None
        assert engine._model is None
        assert engine._tokenizer is None
        assert engine._loaded is False
        assert engine._max_new_tokens == 128
        assert engine._warmup_passes == 2

    def test_warmup_passes_clamped_to_zero(self):
        engine = self._make(warmup_passes=-5)
        assert engine._warmup_passes == 0

    def test_backend_disambiguates_from_ct2(self):
        engine = self._make(device="cpu")
        engine._device = "cpu"
        assert engine.backend == "hf-cpu"

    def test_backend_returns_hf_cuda_when_set(self):
        engine = self._make()
        engine._device = "cuda"
        assert engine.backend == "hf-cuda"

    def test_backend_falls_back_to_cpu_when_unloaded(self):
        engine = self._make()
        # _device is None pre-load
        assert engine.backend == "hf-cpu"

    def test_unload_resets_state(self):
        engine = self._make()
        engine._model = "fake"
        engine._tokenizer = "fake"
        engine._loaded = True
        engine.unload()
        assert engine._model is None
        assert engine._tokenizer is None
        assert engine._loaded is False

    def test_translate_before_load_raises(self):
        engine = self._make()
        with pytest.raises(RuntimeError, match="Engine not loaded"):
            engine.translate("Hello")


class TestMarianCT2EngineContract:
    """MarianCT2Engine constructor and property methods (no real CT2 load)."""

    def _make(self, **overrides):
        from engines.cuda_engine import MarianCT2Engine

        defaults: dict = {
            "model_dir": "/fake/path/to/ct2",
            "source_lang": "en",
            "target_lang": "es",
            "device": "cuda",
            "compute_type": "int8_float16",
        }
        defaults.update(overrides)
        return MarianCT2Engine(**defaults)

    def test_default_constructor_state(self):
        engine = self._make()
        assert engine._model_dir == "/fake/path/to/ct2"
        assert engine._source_lang == "en"
        assert engine._target_lang == "es"
        assert engine._compute_type == "int8_float16"
        assert engine._max_new_tokens == 128
        assert engine._warmup_passes == 2
        assert engine._translator is None
        assert engine._tokenizer is None
        assert engine._loaded is False

    def test_backend_format(self):
        engine = self._make(compute_type="int8")
        assert engine.backend == "ct2-int8"
        engine = self._make(compute_type="int8_float16")
        assert engine.backend == "ct2-int8_float16"
        engine = self._make(compute_type="float16")
        assert engine.backend == "ct2-float16"

    def test_model_id_returns_dir_path(self):
        engine = self._make(model_dir="/some/dir")
        assert engine.model_id == "/some/dir"

    def test_translate_before_load_raises(self):
        engine = self._make()
        with pytest.raises(RuntimeError, match="Engine not loaded"):
            engine.translate("Hello")

    def test_unload_resets_state(self):
        engine = self._make()
        engine._translator = "fake"
        engine._tokenizer = "fake"
        engine._loaded = True
        engine.unload()
        assert engine._translator is None
        assert engine._tokenizer is None
        assert engine._loaded is False

    def test_warmup_passes_clamped_to_zero(self):
        engine = self._make(warmup_passes=-3)
        assert engine._warmup_passes == 0


class TestSharedPyTorchLock:
    """The HF Marian path and Silero VAD must share a single _pytorch_lock.

    Pre-v2026.8 dry_run_ab.py and engines/mlx_engine.py each instantiated their
    own threading.Lock(), which meant concurrent VAD + Marian-HF calls were not
    actually serialized. Regression guard.
    """

    def test_lock_is_shared_singleton(self):
        from engines._locks import _pytorch_lock as lock_a
        from engines._locks import _pytorch_lock as lock_b

        # Same module-level object on every import.
        assert lock_a is lock_b

    def test_marian_hf_uses_shared_lock(self):
        # MarianHFEngine acquires the lock from engines._locks, not from a
        # local threading.Lock(). Verifies the import wiring in marian_hf_engine.
        import engines._locks as locks_mod
        import engines.marian_hf_engine as hf_mod

        # The engine module imports the same lock object at module load time;
        # check that no module-level threading.Lock is shadowing it.
        assert hf_mod._pytorch_lock is locks_mod._pytorch_lock


class TestFactoryMarianRouting:
    """Exercise the marian routing branch in engines.factory.create_translation_engine.

    See also tests/test_engine_factory.py::TestCreateTranslationEngine for the
    fixture-driven CT2-vs-HF dispatch tests; the cases here cover the edge
    cases specific to v2026.8 (compute_type forwarding, kwarg propagation).
    """

    def test_compute_type_forwarded_to_ct2(self, tmp_path, monkeypatch):
        from engines import factory

        active = tmp_path / "en-es" / "active"
        active.mkdir(parents=True)
        (active / "model.bin").write_bytes(b"fake")
        monkeypatch.setattr(factory, "_MARIAN_CT2_ROOT", tmp_path)

        mock_engine = MagicMock()
        with patch("engines.cuda_engine.MarianCT2Engine", return_value=mock_engine) as mock_cls:
            factory.create_translation_engine(
                engine_type="marian",
                backend="cuda",
                compute_type="float16",
                max_new_tokens=64,
                warmup_passes=3,
            )
            kwargs = mock_cls.call_args.kwargs
            assert kwargs["compute_type"] == "float16"
            assert kwargs["max_new_tokens"] == 64
            assert kwargs["warmup_passes"] == 3
            assert kwargs["source_lang"] == "en"
            assert kwargs["target_lang"] == "es"

    def test_es_en_direction_resolves(self, tmp_path, monkeypatch):
        from engines import factory

        active = tmp_path / "es-en" / "active"
        active.mkdir(parents=True)
        (active / "model.bin").write_bytes(b"fake")
        monkeypatch.setattr(factory, "_MARIAN_CT2_ROOT", tmp_path)

        mock_engine = MagicMock()
        with patch("engines.cuda_engine.MarianCT2Engine", return_value=mock_engine) as mock_cls:
            factory.create_translation_engine(
                engine_type="marian",
                backend="cuda",
                source_lang="es",
                target_lang="en",
            )
            kwargs = mock_cls.call_args.kwargs
            assert "es-en" in kwargs["model_dir"]

    def test_unsupported_direction_falls_back_to_hf(self, tmp_path, monkeypatch):
        """Hindi etc. — no CT2 adapter; auto-resolver returns None; HF takes over."""
        from engines import factory

        monkeypatch.setattr(factory, "_MARIAN_CT2_ROOT", tmp_path)
        mock_engine = MagicMock()
        with patch("engines.marian_hf_engine.MarianHFEngine", return_value=mock_engine) as mock_cls:
            factory.create_translation_engine(
                engine_type="marian",
                source_lang="en",
                target_lang="hi",
                model_id="Helsinki-NLP/opus-mt-en-hi",
            )
            mock_cls.assert_called_once()
