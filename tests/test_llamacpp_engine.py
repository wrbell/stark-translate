"""Tests for llama.cpp translation engine.

All tests mock HTTP calls — runs in CI without llama-server.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from engines.llamacpp_engine import LlamaCppEngine


class TestLlamaCppEngineConstruction:
    """Test engine construction and configuration."""

    def test_defaults(self):
        engine = LlamaCppEngine()
        assert engine._server_url == "http://127.0.0.1:8090"
        assert engine._model_family == "gemma4"
        assert engine.backend == "llamacpp"

    def test_custom_url(self):
        engine = LlamaCppEngine(server_url="http://localhost:9999")
        assert engine._server_url == "http://localhost:9999"

    def test_trailing_slash_stripped(self):
        engine = LlamaCppEngine(server_url="http://localhost:8090/")
        assert engine._server_url == "http://localhost:8090"

    def test_translategemma_family(self):
        engine = LlamaCppEngine(model_family="translategemma")
        assert engine._model_family == "translategemma"


class TestLlamaCppEngineLoad:
    """Test health check during load."""

    def test_load_success(self):
        engine = LlamaCppEngine()
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({"status": "ok"}).encode()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)

        mock_props = MagicMock()
        mock_props.read.return_value = json.dumps({"default_generation_settings": {"model": "test-model"}}).encode()
        mock_props.__enter__ = lambda s: s
        mock_props.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", side_effect=[mock_response, mock_props]):
            engine.load()
        assert engine._loaded is True
        assert engine.model_id == "test-model"

    def test_load_server_down(self):
        import urllib.error

        engine = LlamaCppEngine()
        with (
            patch("urllib.request.urlopen", side_effect=urllib.error.URLError("Connection refused")),
            pytest.raises(RuntimeError, match="Cannot connect"),
        ):
            engine.load()

    def test_load_unhealthy(self):
        engine = LlamaCppEngine()
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({"status": "loading"}).encode()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)

        with (
            patch("urllib.request.urlopen", return_value=mock_response),
            pytest.raises(RuntimeError, match="loading"),
        ):
            engine.load()


class TestLlamaCppEngineTranslate:
    """Test translation via HTTP API."""

    def test_translate_before_load_raises(self):
        engine = LlamaCppEngine()
        with pytest.raises(RuntimeError, match="not loaded"):
            engine.translate("Hello")

    def test_translate_gemma4_prompt(self):
        engine = LlamaCppEngine(model_family="gemma4")
        engine._loaded = True

        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(
            {
                "choices": [{"message": {"content": "Hola mundo"}}],
                "usage": {"completion_tokens": 5},
            }
        ).encode()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_response) as mock_open:
            result = engine.translate("Hello world", source_lang="en", target_lang="es")

        assert result.text == "Hola mundo"
        assert result.latency_ms > 0

        # Verify prompt format
        call_args = mock_open.call_args[0][0]
        body = json.loads(call_args.data)
        content = body["messages"][0]["content"]
        assert "English" in content
        assert "Spanish" in content
        assert "Hello world" in content

    def test_translate_strips_preamble(self):
        engine = LlamaCppEngine(model_family="gemma4")
        engine._loaded = True

        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(
            {
                "choices": [{"message": {"content": "Here is the translation:\nHola mundo"}}],
                "usage": {"completion_tokens": 8},
            }
        ).encode()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_response):
            result = engine.translate("Hello world")

        assert result.text == "Hola mundo"

    def test_translate_empty_response(self):
        engine = LlamaCppEngine()
        engine._loaded = True

        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({"choices": []}).encode()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_response):
            result = engine.translate("Hello")

        assert "no response" in result.text


class TestFactoryLlamaCpp:
    """Test factory routing to LlamaCppEngine."""

    def test_engine_type_llamacpp(self):
        from engines.factory import create_translation_engine

        engine = create_translation_engine(engine_type="llamacpp")
        assert type(engine).__name__ == "LlamaCppEngine"
        assert engine._server_url == "http://127.0.0.1:8090"

    def test_custom_server_url(self):
        from engines.factory import create_translation_engine

        engine = create_translation_engine(
            engine_type="llamacpp",
            model_id="http://localhost:9999",
        )
        assert engine._server_url == "http://localhost:9999"
