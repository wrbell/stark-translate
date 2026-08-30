"""Tests for shared translation prompt helpers (MLX / CUDA / llama.cpp parity)."""

from engines.translation_prompts import (
    LANG_NAMES,
    build_chat_messages,
    clean_translation,
    dynamic_max_tokens,
    gemma4_chat_messages,
    gemma4_user_content,
    lang_display_name,
    llamacpp_user_content,
    translategemma_chat_messages,
)


class TestLangNames:
    def test_known_codes(self):
        assert lang_display_name("en") == "English"
        assert lang_display_name("es") == "Spanish"
        assert set(LANG_NAMES) >= {"en", "es", "hi", "zh"}

    def test_unknown_falls_back(self):
        assert lang_display_name("pt") == "pt"


class TestGemma4Prompt:
    def test_user_content_en_es(self):
        content = gemma4_user_content("God so loved the world", "en", "es")
        assert content == (
            "Translate the following English text to Spanish. "
            "Output only the translation, nothing else.\n\n"
            "God so loved the world"
        )

    def test_chat_messages_shape(self):
        messages = gemma4_chat_messages("Hello", "en", "es")
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert isinstance(messages[0]["content"], str)
        assert "Hello" in messages[0]["content"]
        assert "English" in messages[0]["content"]
        assert "Spanish" in messages[0]["content"]


class TestTranslateGemmaPrompt:
    def test_structured_lang_codes(self):
        messages = translategemma_chat_messages("Hello", "en", "es")
        content = messages[0]["content"]
        assert isinstance(content, list)
        assert content[0]["source_lang_code"] == "en"
        assert content[0]["target_lang_code"] == "es"
        assert content[0]["text"] == "Hello"


class TestBuildChatMessagesDispatch:
    def test_gemma4(self):
        msgs = build_chat_messages("x", model_family="gemma4")
        assert isinstance(msgs[0]["content"], str)

    def test_translategemma_default(self):
        msgs = build_chat_messages("x")
        assert isinstance(msgs[0]["content"], list)


class TestLlamaCppUserContent:
    def test_gemma4_matches_mlx_cuda(self):
        text = "In the beginning was the Word"
        mlx_cuda = gemma4_user_content(text, "en", "es")
        llama = llamacpp_user_content(text, source_lang="en", target_lang="es", model_family="gemma4")
        assert mlx_cuda == llama

    def test_translategemma_fallback(self):
        content = llamacpp_user_content("Hi", model_family="translategemma")
        assert content == "Translate from en to es: Hi"


class TestCleanTranslation:
    def test_strips_eot_and_preamble_gemma4(self):
        raw = "Here is the translation:\nLa gracia de Dios.<end_of_turn>"
        assert clean_translation(raw, model_family="gemma4") == "La gracia de Dios."

    def test_strips_eot_only_translategemma(self):
        raw = "Here is the translation:\nHola.<end_of_turn>"
        # TG path does not strip preamble (model rarely emits it)
        assert clean_translation(raw, model_family="translategemma") == "Here is the translation:\nHola."

    def test_no_preamble(self):
        raw = "Hola mundo.<end_of_turn>"
        assert clean_translation(raw, model_family="gemma4") == "Hola mundo."


class TestDynamicMaxTokens:
    def test_floor(self):
        assert dynamic_max_tokens("Hi") == 64

    def test_scales(self):
        text = " ".join(["word"] * 50)
        assert dynamic_max_tokens(text) == 150

    def test_custom_ratio(self):
        text = " ".join(["word"] * 20)
        assert dynamic_max_tokens(text, ratio=1.8, floor=32) == 36


class TestCrossEnginePromptParity:
    """Engines must emit identical Gemma 4 user strings for the same inputs."""

    def test_mlx_cuda_llamacpp_identical(self):
        from unittest.mock import MagicMock, patch

        from engines.cuda_engine import CUDAGemmaStreamingEngine
        from engines.llamacpp_engine import LlamaCppEngine
        from engines.mlx_engine import MLXGemmaEngine

        text = "Blessed are the peacemakers"
        expected = gemma4_user_content(text, "en", "es")

        # MLX path via build_chat_messages
        mlx_msgs = build_chat_messages(text, model_family="gemma4")
        assert mlx_msgs[0]["content"] == expected

        # llama.cpp path
        llama_content = llamacpp_user_content(text, model_family="gemma4")
        assert llama_content == expected

        # CUDA streaming engine uses build_chat_messages in _build_full_prompt
        with (
            patch("engines.cuda_engine.TORCH_AVAILABLE", True),
            patch("engines.cuda_engine.torch") as mock_torch,
        ):
            mock_torch.cuda.is_available.return_value = True
            engine = CUDAGemmaStreamingEngine(model_family="gemma4")
            engine._tokenizer = MagicMock()
            engine._tokenizer.apply_chat_template.return_value = MagicMock()
            engine._tokenizer.apply_chat_template.return_value.to = MagicMock(return_value="ids")
            engine._build_full_prompt(text, "en", "es")
            call_args = engine._tokenizer.apply_chat_template.call_args
            messages = call_args[0][0]
            assert messages[0]["content"] == expected

        with patch("engines.mlx_engine.MLX_AVAILABLE", True):
            mlx = MLXGemmaEngine(model_family="gemma4")
            assert mlx._model_family == "gemma4"

        llama = LlamaCppEngine(model_family="gemma4")
        assert llama._model_family == "gemma4"
