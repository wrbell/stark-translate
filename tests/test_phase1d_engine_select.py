"""Phase 1D — verify CUDA engine selection between llama.cpp and HF NF4."""

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def reset_settings():
    """Snapshot settings.cuda fields modified by tests; restore after."""
    from settings import settings

    saved = (
        settings.cuda.engine,
        settings.cuda.llamacpp_url,
        settings.cuda.llamacpp_url_b,
    )
    yield settings
    (
        settings.cuda.engine,
        settings.cuda.llamacpp_url,
        settings.cuda.llamacpp_url_b,
    ) = saved


class TestResolveCudaEngineChoice:
    """`_resolve_cuda_engine_choice` honors settings + probe."""

    def test_forced_hf(self, reset_settings):
        from dry_run_ab import _resolve_cuda_engine_choice

        reset_settings.cuda.engine = "hf"
        assert _resolve_cuda_engine_choice() == "hf"

    def test_forced_llamacpp(self, reset_settings):
        from dry_run_ab import _resolve_cuda_engine_choice

        reset_settings.cuda.engine = "llamacpp"
        assert _resolve_cuda_engine_choice() == "llamacpp"

    def test_auto_with_server_reachable(self, reset_settings):
        from dry_run_ab import _resolve_cuda_engine_choice

        reset_settings.cuda.engine = "auto"
        with patch("dry_run_ab._probe_llamacpp_server", return_value=True):
            assert _resolve_cuda_engine_choice() == "llamacpp"

    def test_auto_with_server_unreachable_falls_back_to_hf(self, reset_settings):
        from dry_run_ab import _resolve_cuda_engine_choice

        reset_settings.cuda.engine = "auto"
        with patch("dry_run_ab._probe_llamacpp_server", return_value=False):
            assert _resolve_cuda_engine_choice() == "hf"


class TestLoadCudaTranslationModelsLlamaCppPath:
    """When engine resolves to llamacpp, ``LlamaCppEngine`` is instantiated."""

    def test_primary_only(self, reset_settings):
        import dry_run_ab

        reset_settings.cuda.engine = "llamacpp"

        fake_engine = MagicMock()
        fake_engine.model_id = "gemma-4-e4b-it-q4km.gguf"
        fake_engine.load = MagicMock()

        with patch("engines.llamacpp_engine.LlamaCppEngine", return_value=fake_engine) as cls:
            gemma_a, tok_a, gemma_b, tok_b = dry_run_ab.load_cuda_translation_models(load_gemma=True, load_b=False)

        cls.assert_called_once()
        # Primary engine constructed at the configured URL
        kwargs = cls.call_args.kwargs
        assert kwargs["server_url"] == reset_settings.cuda.llamacpp_url
        assert kwargs["model_family"] == "gemma4"
        # Primary returned, secondary None, tokenizers None
        assert gemma_a is fake_engine
        assert tok_a is None
        assert gemma_b is None
        assert tok_b is None
        fake_engine.load.assert_called_once()

    def test_ab_loads_two_engines_at_two_urls(self, reset_settings):
        import dry_run_ab

        reset_settings.cuda.engine = "llamacpp"
        primary = reset_settings.cuda.llamacpp_url
        secondary = reset_settings.cuda.llamacpp_url_b

        engine_a = MagicMock(model_id="gemma-4-e4b-it-q4km.gguf")
        engine_b = MagicMock(model_id="gemma-4-e2b-it-q4km.gguf")
        constructed = []

        def _construct(server_url, model_family):
            constructed.append(server_url)
            return engine_a if server_url == primary else engine_b

        with patch("engines.llamacpp_engine.LlamaCppEngine", side_effect=_construct):
            gemma_a, _, gemma_b, _ = dry_run_ab.load_cuda_translation_models(load_gemma=True, load_b=True)

        assert constructed == [primary, secondary]
        assert gemma_a is engine_a
        assert gemma_b is engine_b


class TestLoadCudaTranslationModelsHfPath:
    """When engine resolves to hf, ``CUDAGemmaStreamingEngine`` is used."""

    def test_falls_back_to_streaming_engine(self, reset_settings):
        import dry_run_ab

        reset_settings.cuda.engine = "hf"

        fake_streaming = MagicMock()
        fake_streaming.load = MagicMock()
        fake_streaming._tokenizer = MagicMock()

        with (
            patch("engines.cuda_engine.CUDAGemmaStreamingEngine", return_value=fake_streaming) as cls,
            patch("engines.cuda_engine.detect_vram_tier", return_value=("4b_only", 12000)),
            patch("engines.llamacpp_engine.LlamaCppEngine") as llama_cls,
        ):
            gemma_a, tok_a, gemma_b, tok_b = dry_run_ab.load_cuda_translation_models(load_gemma=True, load_b=False)

        cls.assert_called_once()
        llama_cls.assert_not_called()  # crucial: HF path must not touch llama.cpp
        assert gemma_a is fake_streaming
        assert tok_a is fake_streaming._tokenizer
        assert gemma_b is None
        assert tok_b is None

    def test_marian_tier_returns_none(self, reset_settings):
        import dry_run_ab

        reset_settings.cuda.engine = "hf"

        with (
            patch("engines.cuda_engine.detect_vram_tier", return_value=("marian", 4000)),
            patch("engines.cuda_engine.CUDAGemmaStreamingEngine") as cls,
            patch("engines.llamacpp_engine.LlamaCppEngine") as llama_cls,
        ):
            result = dry_run_ab.load_cuda_translation_models(load_gemma=True, load_b=False)

        cls.assert_not_called()
        llama_cls.assert_not_called()
        assert result == (None, None, None, None)


class TestProbeLlamaCppServer:
    """`_probe_llamacpp_server` returns True/False without raising."""

    def test_returns_true_when_load_succeeds(self):
        from dry_run_ab import _probe_llamacpp_server

        with patch("engines.llamacpp_engine.LlamaCppEngine") as cls:
            instance = cls.return_value
            instance.load = MagicMock()
            instance.unload = MagicMock()
            assert _probe_llamacpp_server("http://127.0.0.1:8090") is True

    def test_returns_false_when_load_raises(self):
        from dry_run_ab import _probe_llamacpp_server

        with patch("engines.llamacpp_engine.LlamaCppEngine") as cls:
            instance = cls.return_value
            instance.load = MagicMock(side_effect=RuntimeError("connection refused"))
            assert _probe_llamacpp_server("http://127.0.0.1:8090") is False
