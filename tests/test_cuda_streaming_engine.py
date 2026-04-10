"""Tests for CUDA streaming engine, VRAM tier detection, and prompt cache.

All tests use mocked CUDA/PyTorch — runs in CI without GPU.
"""

from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# detect_vram_tier
# ---------------------------------------------------------------------------


class TestDetectVramTier:
    """Test VRAM tier detection logic."""

    def test_24gb_returns_full_ab(self):
        from engines.cuda_engine import detect_vram_tier

        mock_props = MagicMock()
        mock_props.total_mem = 24 * 1024 * 1024 * 1024  # 24 GB
        mock_props.name = "NVIDIA GeForce RTX 3090 Ti"

        with (
            patch("engines.cuda_engine.TORCH_AVAILABLE", True),
            patch("engines.cuda_engine.torch") as mock_torch,
        ):
            mock_torch.cuda.is_available.return_value = True
            mock_torch.cuda.get_device_properties.return_value = mock_props
            tier, vram = detect_vram_tier()

        assert tier == "full_ab"
        assert vram == 24 * 1024  # MB

    def test_16gb_returns_full_ab(self):
        from engines.cuda_engine import detect_vram_tier

        mock_props = MagicMock()
        mock_props.total_mem = 16 * 1024 * 1024 * 1024
        mock_props.name = "NVIDIA RTX A2000 Ada"

        with (
            patch("engines.cuda_engine.TORCH_AVAILABLE", True),
            patch("engines.cuda_engine.torch") as mock_torch,
        ):
            mock_torch.cuda.is_available.return_value = True
            mock_torch.cuda.get_device_properties.return_value = mock_props
            tier, _ = detect_vram_tier()

        assert tier == "full_ab"

    def test_8gb_returns_4b_only(self):
        from engines.cuda_engine import detect_vram_tier

        mock_props = MagicMock()
        mock_props.total_mem = 8 * 1024 * 1024 * 1024
        mock_props.name = "NVIDIA GeForce RTX 3070 Ti"

        with (
            patch("engines.cuda_engine.TORCH_AVAILABLE", True),
            patch("engines.cuda_engine.torch") as mock_torch,
        ):
            mock_torch.cuda.is_available.return_value = True
            mock_torch.cuda.get_device_properties.return_value = mock_props
            tier, _ = detect_vram_tier()

        assert tier == "4b_only"

    def test_4gb_returns_marian(self):
        from engines.cuda_engine import detect_vram_tier

        mock_props = MagicMock()
        mock_props.total_mem = 4 * 1024 * 1024 * 1024
        mock_props.name = "NVIDIA GeForce GTX 1650"

        with (
            patch("engines.cuda_engine.TORCH_AVAILABLE", True),
            patch("engines.cuda_engine.torch") as mock_torch,
        ):
            mock_torch.cuda.is_available.return_value = True
            mock_torch.cuda.get_device_properties.return_value = mock_props
            tier, _ = detect_vram_tier()

        assert tier == "marian"

    def test_no_cuda_returns_none(self):
        from engines.cuda_engine import detect_vram_tier

        with (
            patch("engines.cuda_engine.TORCH_AVAILABLE", True),
            patch("engines.cuda_engine.torch") as mock_torch,
        ):
            mock_torch.cuda.is_available.return_value = False
            tier, vram = detect_vram_tier()

        assert tier == "none"
        assert vram == 0

    def test_no_torch_returns_none(self):
        from engines.cuda_engine import detect_vram_tier

        with patch("engines.cuda_engine.TORCH_AVAILABLE", False):
            tier, vram = detect_vram_tier()

        assert tier == "none"
        assert vram == 0


# ---------------------------------------------------------------------------
# _clone_past_key_values
# ---------------------------------------------------------------------------


class TestClonePastKeyValues:
    """Test KV cache cloning utility."""

    def test_none_returns_none(self):
        from engines.cuda_engine import _clone_past_key_values

        assert _clone_past_key_values(None) is None

    def test_clones_nested_tuples(self):
        from engines.cuda_engine import _clone_past_key_values

        # Simulate past_key_values structure: tuple of (key, value) per layer
        mock_key = MagicMock()
        mock_key.clone.return_value = MagicMock()
        mock_val = MagicMock()
        mock_val.clone.return_value = MagicMock()

        # Remove key_cache attr so it goes through the tuple path
        pkv = ((mock_key, mock_val),)
        del pkv[0][0].key_cache  # ensure no DynamicCache detection

        cloned = _clone_past_key_values(pkv)

        assert len(cloned) == 1
        assert len(cloned[0]) == 2
        mock_key.clone.assert_called_once()
        mock_val.clone.assert_called_once()

    def test_dynamic_cache_uses_deepcopy(self):
        from engines.cuda_engine import _clone_past_key_values

        mock_cache = MagicMock()
        mock_cache.key_cache = [MagicMock()]  # has key_cache attr → DynamicCache

        result = _clone_past_key_values(mock_cache)
        # copy.deepcopy should have been called (MagicMock.__deepcopy__)
        assert result is not mock_cache


# ---------------------------------------------------------------------------
# CUDAGemmaStreamingEngine
# ---------------------------------------------------------------------------


class TestCUDAGemmaStreamingEngine:
    """Test the enhanced CUDA translation engine (mocked)."""

    def test_init_requires_torch(self):
        from engines.cuda_engine import CUDAGemmaStreamingEngine

        with patch("engines.cuda_engine.TORCH_AVAILABLE", False):
            with pytest.raises(RuntimeError, match="PyTorch is not installed"):
                CUDAGemmaStreamingEngine()

    def test_init_requires_cuda(self):
        from engines.cuda_engine import CUDAGemmaStreamingEngine

        with (
            patch("engines.cuda_engine.TORCH_AVAILABLE", True),
            patch("engines.cuda_engine.torch") as mock_torch,
        ):
            mock_torch.cuda.is_available.return_value = False
            with pytest.raises(RuntimeError, match="CUDA is not available"):
                CUDAGemmaStreamingEngine()

    def test_translate_before_load_raises(self):
        from engines.cuda_engine import CUDAGemmaStreamingEngine

        with (
            patch("engines.cuda_engine.TORCH_AVAILABLE", True),
            patch("engines.cuda_engine.torch") as mock_torch,
        ):
            mock_torch.cuda.is_available.return_value = True
            engine = CUDAGemmaStreamingEngine()

        with pytest.raises(RuntimeError, match="not loaded"):
            engine.translate("hello")

    def test_properties(self):
        from engines.cuda_engine import CUDAGemmaStreamingEngine

        with (
            patch("engines.cuda_engine.TORCH_AVAILABLE", True),
            patch("engines.cuda_engine.torch") as mock_torch,
        ):
            mock_torch.cuda.is_available.return_value = True
            engine = CUDAGemmaStreamingEngine(model_id="test/model")

        assert engine.model_id == "test/model"
        assert engine.backend == "cuda"

    def test_unload_clears_state(self):
        from engines.cuda_engine import CUDAGemmaStreamingEngine

        with (
            patch("engines.cuda_engine.TORCH_AVAILABLE", True),
            patch("engines.cuda_engine.torch") as mock_torch,
        ):
            mock_torch.cuda.is_available.return_value = True
            engine = CUDAGemmaStreamingEngine()
            engine._model = MagicMock()
            engine._tokenizer = MagicMock()
            engine._prompt_cache_pkv = MagicMock()
            engine._loaded = True

            engine.unload()

        assert engine._model is None
        assert engine._tokenizer is None
        assert engine._prompt_cache_pkv is None
        assert not engine._loaded

    def test_streaming_batch_size_default(self):
        from engines.cuda_engine import CUDAGemmaStreamingEngine

        with (
            patch("engines.cuda_engine.TORCH_AVAILABLE", True),
            patch("engines.cuda_engine.torch") as mock_torch,
        ):
            mock_torch.cuda.is_available.return_value = True
            engine = CUDAGemmaStreamingEngine()

        assert engine._streaming_batch_size == 3

    def test_streaming_batch_size_custom(self):
        from engines.cuda_engine import CUDAGemmaStreamingEngine

        with (
            patch("engines.cuda_engine.TORCH_AVAILABLE", True),
            patch("engines.cuda_engine.torch") as mock_torch,
        ):
            mock_torch.cuda.is_available.return_value = True
            engine = CUDAGemmaStreamingEngine(streaming_batch_size=5)

        assert engine._streaming_batch_size == 5


# ---------------------------------------------------------------------------
# Factory integration
# ---------------------------------------------------------------------------


class TestFactoryStreamingParam:
    """Test that the factory routes to CUDAGemmaStreamingEngine."""

    def test_cuda_streaming_true_returns_streaming_engine(self):
        from engines.factory import create_translation_engine

        with patch("engines.factory._detect_backend", return_value="cuda"):
            with patch("engines.cuda_engine.CUDAGemmaStreamingEngine") as mock_cls:
                mock_cls.return_value = MagicMock()
                engine = create_translation_engine(backend="cuda", streaming=True)
                mock_cls.assert_called_once()

    def test_cuda_streaming_false_returns_basic_engine(self):
        from engines.factory import create_translation_engine

        with patch("engines.factory._detect_backend", return_value="cuda"):
            with patch("engines.cuda_engine.CUDAGemmaEngine") as mock_cls:
                mock_cls.return_value = MagicMock()
                engine = create_translation_engine(backend="cuda", streaming=False)
                mock_cls.assert_called_once()

    def test_assistant_model_forwarded(self):
        from engines.factory import create_translation_engine

        with patch("engines.factory._detect_backend", return_value="cuda"):
            with patch("engines.cuda_engine.CUDAGemmaStreamingEngine") as mock_cls:
                mock_cls.return_value = MagicMock()
                create_translation_engine(
                    backend="cuda",
                    streaming=True,
                    assistant_model_id="google/translategemma-4b-it",
                )
                _, kwargs = mock_cls.call_args
                assert kwargs["assistant_model_id"] == "google/translategemma-4b-it"


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


class TestCUDASettings:
    """Test that CUDASettings is properly nested."""

    def test_cuda_settings_accessible(self):
        from settings import PipelineSettings

        s = PipelineSettings()
        assert s.cuda.vram_tier == "auto"
        assert s.cuda.use_prompt_cache is True
        assert s.cuda.use_speculative is True
        assert s.cuda.pipeline_workers == 2
        assert s.cuda.streaming_batch_size == 3
        assert s.cuda.compute_type == "int8"

    def test_cuda_model_12b_in_translation(self):
        from settings import PipelineSettings

        s = PipelineSettings()
        assert s.translation.cuda_model_12b == "google/translategemma-12b-it"
