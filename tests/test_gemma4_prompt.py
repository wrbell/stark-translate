"""Tests for Gemma 4 instruct prompt formatting in translation engines.

Validates that model_family='gemma4' produces plain-text instruct prompts
instead of TranslateGemma's structured source_lang_code/target_lang_code format.
"""

from unittest.mock import MagicMock, patch


class TestCUDAGemma4Prompt:
    """Test Gemma 4 prompt format in CUDAGemmaStreamingEngine."""

    def test_model_family_default_is_translategemma(self):
        from engines.cuda_engine import CUDAGemmaStreamingEngine

        with (
            patch("engines.cuda_engine.TORCH_AVAILABLE", True),
            patch("engines.cuda_engine.torch") as mock_torch,
        ):
            mock_torch.cuda.is_available.return_value = True
            engine = CUDAGemmaStreamingEngine()
        assert engine._model_family == "translategemma"

    def test_model_family_gemma4(self):
        from engines.cuda_engine import CUDAGemmaStreamingEngine

        with (
            patch("engines.cuda_engine.TORCH_AVAILABLE", True),
            patch("engines.cuda_engine.torch") as mock_torch,
        ):
            mock_torch.cuda.is_available.return_value = True
            engine = CUDAGemmaStreamingEngine(model_family="gemma4")
        assert engine._model_family == "gemma4"

    def test_translategemma_prompt_has_lang_codes(self):
        from engines.cuda_engine import CUDAGemmaStreamingEngine

        with (
            patch("engines.cuda_engine.TORCH_AVAILABLE", True),
            patch("engines.cuda_engine.torch") as mock_torch,
        ):
            mock_torch.cuda.is_available.return_value = True
            engine = CUDAGemmaStreamingEngine(model_family="translategemma")
            engine._tokenizer = MagicMock()
            engine._tokenizer.apply_chat_template.return_value = MagicMock()
            engine._build_full_prompt("Hello world", "en", "es")

            call_args = engine._tokenizer.apply_chat_template.call_args
            messages = call_args[0][0]
            content = messages[0]["content"]
            assert isinstance(content, list)
            assert content[0]["source_lang_code"] == "en"
            assert content[0]["target_lang_code"] == "es"

    def test_gemma4_prompt_is_plain_text(self):
        from engines.cuda_engine import CUDAGemmaStreamingEngine

        with (
            patch("engines.cuda_engine.TORCH_AVAILABLE", True),
            patch("engines.cuda_engine.torch") as mock_torch,
        ):
            mock_torch.cuda.is_available.return_value = True
            engine = CUDAGemmaStreamingEngine(model_family="gemma4")
            engine._tokenizer = MagicMock()
            engine._tokenizer.apply_chat_template.return_value = MagicMock()
            engine._build_full_prompt("Hello world", "en", "es")

            call_args = engine._tokenizer.apply_chat_template.call_args
            messages = call_args[0][0]
            content = messages[0]["content"]
            assert isinstance(content, str)
            assert "English" in content
            assert "Spanish" in content
            assert "Hello world" in content


class TestMLXGemma4Prompt:
    """Test Gemma 4 prompt format in MLXGemmaEngine."""

    def test_model_family_default_is_translategemma(self):
        from engines.mlx_engine import MLXGemmaEngine

        with patch("engines.mlx_engine.MLX_AVAILABLE", True):
            engine = MLXGemmaEngine()
        assert engine._model_family == "translategemma"

    def test_model_family_gemma4(self):
        from engines.mlx_engine import MLXGemmaEngine

        with patch("engines.mlx_engine.MLX_AVAILABLE", True):
            engine = MLXGemmaEngine(model_family="gemma4")
        assert engine._model_family == "gemma4"


class TestGemma4PreambleStripping:
    """Test that Gemma 4 preamble text is stripped from output."""

    def test_strips_preamble(self):
        from engines.translation_prompts import clean_translation

        result = "Here is the translation:\nLa gracia de Dios es suficiente.<end_of_turn>"
        clean = clean_translation(result, model_family="gemma4")
        assert clean == "La gracia de Dios es suficiente."

    def test_no_strip_when_no_preamble(self):
        from engines.translation_prompts import clean_translation

        result = "La gracia de Dios es suficiente.<end_of_turn>"
        clean = clean_translation(result, model_family="gemma4")
        assert clean == "La gracia de Dios es suficiente."


class TestModelFamilySettings:
    """Test model_family setting in TranslationSettings."""

    def test_default_is_translategemma(self):
        from settings import TranslationSettings

        ts = TranslationSettings()
        assert ts.model_family == "translategemma"

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("STARK_TRANSLATE_MODEL_FAMILY", "gemma4")
        from settings import TranslationSettings

        ts = TranslationSettings()
        assert ts.model_family == "gemma4"
