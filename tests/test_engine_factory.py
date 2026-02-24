"""Tests for engines/factory.py — backend detection and factory routing."""

import sys
from unittest.mock import MagicMock, patch

import pytest

from engines.factory import _detect_backend, create_stt_engine, create_translation_engine


class TestDetectBackend:
    def test_mlx_available(self):
        mock_mlx = MagicMock()
        with patch.dict(sys.modules, {"mlx": mock_mlx, "mlx.core": mock_mlx}):
            assert _detect_backend() == "mlx"

    def test_cuda_available(self):
        """When mlx is not importable but torch.cuda.is_available() is True."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True

        modules = {"mlx": None, "mlx.core": None, "torch": mock_torch}
        with (
            patch.dict(sys.modules, modules),
            patch("builtins.__import__", side_effect=_make_import_side_effect(block={"mlx.core", "mlx"})),
        ):
            assert _detect_backend() == "cuda"

    def test_cpu_fallback(self):
        """When neither mlx nor torch is available."""
        with patch("builtins.__import__", side_effect=_make_import_side_effect(block={"mlx.core", "mlx", "torch"})):
            assert _detect_backend() == "cpu"

    def test_cuda_not_available(self):
        """When torch is importable but CUDA is not available."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False

        with (
            patch("builtins.__import__", side_effect=_make_import_side_effect(block={"mlx.core", "mlx"})),
            patch.dict(sys.modules, {"torch": mock_torch}),
        ):
            result = _detect_backend()
            assert result == "cpu"


class TestCreateSTTEngine:
    def test_invalid_backend_raises(self):
        with pytest.raises(ValueError, match="Unsupported STT backend"):
            create_stt_engine(backend="tpu")

    def test_mlx_backend_imports_correctly(self):
        mock_engine = MagicMock()
        with patch("engines.mlx_engine.MLXWhisperEngine", return_value=mock_engine) as mock_cls:
            result = create_stt_engine(backend="mlx")
            mock_cls.assert_called_once()
            assert result == mock_engine

    def test_cuda_backend_imports_correctly(self):
        mock_engine = MagicMock()
        mock_module = MagicMock()
        mock_module.FasterWhisperEngine = MagicMock(return_value=mock_engine)
        with patch.dict(sys.modules, {"engines.cuda_engine": mock_module}):
            result = create_stt_engine(backend="cuda")
            mock_module.FasterWhisperEngine.assert_called_once()
            assert result == mock_engine

    def test_threshold_kwargs_only_passed_when_set(self):
        mock_engine = MagicMock()
        with patch("engines.mlx_engine.MLXWhisperEngine", return_value=mock_engine) as mock_cls:
            create_stt_engine(backend="mlx", fallback_threshold=-1.5)
            _, kwargs = mock_cls.call_args
            assert kwargs["fallback_threshold"] == -1.5
            assert "hallucination_threshold" not in kwargs
            assert "fallback_on_low_conf" not in kwargs

    def test_custom_model_id(self):
        mock_engine = MagicMock()
        with patch("engines.mlx_engine.MLXWhisperEngine", return_value=mock_engine) as mock_cls:
            create_stt_engine(backend="mlx", model_id="custom/model")
            _, kwargs = mock_cls.call_args
            assert kwargs["model_id"] == "custom/model"


class TestCreateTranslationEngine:
    def test_invalid_backend_raises(self):
        with pytest.raises(ValueError, match="Unsupported translation backend"):
            create_translation_engine(backend="tpu")

    def test_marian_engine_type(self):
        mock_engine = MagicMock()
        with patch("engines.mlx_engine.MarianEngine", return_value=mock_engine) as mock_cls:
            result = create_translation_engine(engine_type="marian")
            mock_cls.assert_called_once()
            assert result == mock_engine

    def test_gemma_mlx_backend(self):
        mock_engine = MagicMock()
        with patch("engines.mlx_engine.MLXGemmaEngine", return_value=mock_engine) as mock_cls:
            result = create_translation_engine(backend="mlx", engine_type="gemma")
            mock_cls.assert_called_once()
            assert result == mock_engine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_real_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__


def _make_import_side_effect(block: set):
    """Create an __import__ side effect that blocks specific modules."""

    def _side_effect(name, *args, **kwargs):
        if name in block:
            raise ImportError(f"Mocked: {name} not available")
        return _real_import(name, *args, **kwargs)

    return _side_effect
