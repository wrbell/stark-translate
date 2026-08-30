"""Tests for MLX TurboQuant KV cache integration.

All tests mock MLX dependencies — runs in CI without Apple Silicon.
"""

from unittest.mock import MagicMock, patch


class TestMLXGemmaTurboQuant:
    """Test TurboQuant KV cache integration in MLXGemmaEngine."""

    def test_turboquant_disabled_by_default(self):
        from engines.mlx_engine import MLXGemmaEngine

        with patch("engines.mlx_engine.MLX_AVAILABLE", True):
            engine = MLXGemmaEngine()
        assert engine._use_turboquant is False

    def test_turboquant_enabled_via_constructor(self):
        from engines.mlx_engine import MLXGemmaEngine

        with patch("engines.mlx_engine.MLX_AVAILABLE", True):
            engine = MLXGemmaEngine(use_turboquant=True, turboquant_key_bits=3, turboquant_val_bits=4)
        assert engine._use_turboquant is True
        assert engine._turboquant_key_bits == 3
        assert engine._turboquant_val_bits == 4

    def test_turboquant_custom_bits(self):
        from engines.mlx_engine import MLXGemmaEngine

        with patch("engines.mlx_engine.MLX_AVAILABLE", True):
            engine = MLXGemmaEngine(use_turboquant=True, turboquant_key_bits=4, turboquant_val_bits=8)
        assert engine._turboquant_key_bits == 4
        assert engine._turboquant_val_bits == 8

    def test_adapter_path_stored(self):
        from engines.mlx_engine import MLXGemmaEngine

        with patch("engines.mlx_engine.MLX_AVAILABLE", True):
            engine = MLXGemmaEngine(adapter_path="/tmp/adapters/active")
        assert engine._adapter_path == "/tmp/adapters/active"

    def test_load_passes_adapter_path(self):
        """mlx_lm.load receives adapter_path when set."""
        from engines.mlx_engine import MLXGemmaEngine

        mock_tokenizer = MagicMock()
        mock_tokenizer.convert_tokens_to_ids.return_value = 106
        mock_tokenizer.eos_token_id = 1
        mock_tokenizer._eos_token_ids = {1, 106}
        mock_model = MagicMock()
        mock_load = MagicMock(return_value=(mock_model, mock_tokenizer))
        mock_mx = MagicMock()

        with (
            patch("engines.mlx_engine.MLX_AVAILABLE", True),
            patch("engines.mlx_engine.mx", mock_mx),
            patch("engines.mlx_engine.MLXGemmaEngine._build_prompt_cache", return_value=(None, None)),
            patch("mlx_lm.load", mock_load),
        ):
            engine = MLXGemmaEngine(adapter_path="/adapters/s6", use_prompt_cache=False)
            engine.load()

        mock_load.assert_called_once()
        _, kwargs = mock_load.call_args
        assert kwargs.get("adapter_path") == "/adapters/s6"

    def test_turboquant_import_error_warns(self):
        """If mlx-optiq not installed, engine loads without TurboQuant."""
        from engines.mlx_engine import MLXGemmaEngine

        with (
            patch("engines.mlx_engine.MLX_AVAILABLE", True),
            patch("engines.mlx_engine.mx") as mock_mx,
        ):
            engine = MLXGemmaEngine(use_turboquant=True)
            mock_mx.set_cache_limit.return_value = None

            with (
                patch("engines.mlx_engine.MLXGemmaEngine._build_prompt_cache", return_value=(None, None)),
                patch.dict("sys.modules", {"mlx_optiq": None}),
                patch("builtins.__import__", side_effect=ImportError("no mlx-optiq")),
            ):
                assert engine._use_turboquant is True


class TestTurboQuantSettings:
    """Test TurboQuant settings in TranslationSettings."""

    def test_turboquant_settings_exist(self):
        from settings import TranslationSettings

        ts = TranslationSettings()
        assert hasattr(ts, "turboquant")
        assert hasattr(ts, "turboquant_key_bits")
        assert hasattr(ts, "turboquant_val_bits")

    def test_turboquant_defaults(self):
        from settings import TranslationSettings

        ts = TranslationSettings()
        assert ts.turboquant is False
        assert ts.turboquant_key_bits == 3
        assert ts.turboquant_val_bits == 4

    def test_turboquant_env_override(self, monkeypatch):
        monkeypatch.setenv("STARK_TRANSLATE_TURBOQUANT", "true")
        monkeypatch.setenv("STARK_TRANSLATE_TURBOQUANT_KEY_BITS", "4")
        monkeypatch.setenv("STARK_TRANSLATE_TURBOQUANT_VAL_BITS", "8")

        from settings import TranslationSettings

        ts = TranslationSettings()
        assert ts.turboquant is True
        assert ts.turboquant_key_bits == 4
        assert ts.turboquant_val_bits == 8


class TestFactoryTurboQuant:
    """Factory forwards TurboQuant settings into MLXGemmaEngine."""

    def test_factory_injects_turboquant_from_settings(self):
        from engines.factory import create_translation_engine

        mock_engine = MagicMock()
        settings_kwargs = {
            "use_turboquant": True,
            "turboquant_key_bits": 3,
            "turboquant_val_bits": 4,
            "model_family": "translategemma",
        }

        with (
            patch("engines.mlx_engine.MLXGemmaEngine", return_value=mock_engine) as mock_cls,
            patch("engines.factory._mlx_gemma_settings_kwargs", return_value=settings_kwargs),
        ):
            result = create_translation_engine(backend="mlx")

        _, kwargs = mock_cls.call_args
        assert kwargs.get("use_turboquant") is True
        assert kwargs.get("turboquant_key_bits") == 3
        assert kwargs.get("turboquant_val_bits") == 4
        assert result == mock_engine

    def test_factory_kwargs_override_settings(self):
        from engines.factory import create_translation_engine

        mock_engine = MagicMock()
        settings_kwargs = {
            "use_turboquant": True,
            "turboquant_key_bits": 3,
            "turboquant_val_bits": 4,
            "model_family": "translategemma",
        }

        with (
            patch("engines.mlx_engine.MLXGemmaEngine", return_value=mock_engine) as mock_cls,
            patch("engines.factory._mlx_gemma_settings_kwargs", return_value=settings_kwargs),
        ):
            create_translation_engine(
                backend="mlx",
                use_turboquant=False,
                adapter_path="/a",
            )

        _, kwargs = mock_cls.call_args
        assert kwargs.get("use_turboquant") is False
        assert kwargs.get("adapter_path") == "/a"


class TestDryRunLoadMlxGemma:
    """dry_run_ab.load_mlx_gemma forwards adapter_path and TurboQuant."""

    def test_load_mlx_gemma_passes_adapter(self):
        import dry_run_ab as dra

        mock_tokenizer = MagicMock()
        mock_tokenizer.convert_tokens_to_ids.return_value = 106
        mock_tokenizer.eos_token_id = 1
        mock_tokenizer._eos_token_ids = {1, 106}
        mock_model = MagicMock()
        mock_load = MagicMock(return_value=(mock_model, mock_tokenizer))

        with (
            patch.object(dra, "USE_TURBOQUANT", False),
            patch("mlx.core.set_cache_limit"),
            patch("mlx_lm.load", mock_load),
        ):
            model, tok = dra.load_mlx_gemma("mlx-community/test", "test", adapter_path="/lora")

        assert model is mock_model
        assert tok is mock_tokenizer
        mock_load.assert_called_once_with("mlx-community/test", adapter_path="/lora")
