"""Tests for MLX Gemma 4 OptiQ settings + factory model resolution / MTS wiring."""

from unittest.mock import MagicMock, patch


class TestGemma4Settings:
    def test_defaults_are_gemma4(self):
        from settings import TranslationSettings

        ts = TranslationSettings()
        assert ts.model_family == "gemma4"
        assert ts.mlx_mts is False
        assert "OptiQ" in ts.mlx_model_gemma4_e4b
        assert "OptiQ" in ts.mlx_model_gemma4_e2b
        assert "assistant" in ts.mlx_drafter_gemma4

    def test_env_overrides(self, monkeypatch):
        monkeypatch.setenv("STARK_TRANSLATE_MODEL_FAMILY", "gemma4")
        monkeypatch.setenv("STARK_TRANSLATE_MLX_MTS", "true")
        monkeypatch.setenv(
            "STARK_TRANSLATE_MLX_MODEL_GEMMA4_E4B",
            "mlx-community/custom-e4b",
        )
        from settings import TranslationSettings

        ts = TranslationSettings()
        assert ts.model_family == "gemma4"
        assert ts.mlx_mts is True
        assert ts.mlx_model_gemma4_e4b == "mlx-community/custom-e4b"


class TestResolveMlxTranslationModelId:
    def test_explicit_wins(self):
        from engines.factory import resolve_mlx_translation_model_id

        assert resolve_mlx_translation_model_id("my/model") == "my/model"

    def test_gemma4_e4b_default(self):
        from engines.factory import resolve_mlx_translation_model_id

        mid = resolve_mlx_translation_model_id(model_family="gemma4", size="e4b")
        assert "e4b" in mid.lower() or "OptiQ" in mid

    def test_gemma4_e2b(self):
        from engines.factory import resolve_mlx_translation_model_id

        mid = resolve_mlx_translation_model_id(model_family="gemma4", size="e2b")
        assert "e2b" in mid.lower()

    def test_translategemma_default(self):
        from engines.factory import resolve_mlx_translation_model_id

        mid = resolve_mlx_translation_model_id(model_family="translategemma")
        assert "translategemma" in mid


class TestFactoryGemma4Mts:
    def test_factory_resolves_gemma4_model(self):
        from engines.factory import create_translation_engine

        mock_engine = MagicMock()
        with (
            patch("engines.mlx_engine.MLX_AVAILABLE", True),
            patch("engines.mlx_engine.MLXGemmaEngine", return_value=mock_engine) as ctor,
            patch(
                "engines.factory._mlx_gemma_settings_kwargs",
                return_value={"model_family": "gemma4", "num_draft_tokens": 1},
            ),
        ):
            create_translation_engine(backend="mlx", model_family="gemma4")
        kwargs = ctor.call_args.kwargs
        assert kwargs.get("model_family") == "gemma4"
        assert "OptiQ" in kwargs.get("model_id", "") or "gemma-4" in kwargs.get("model_id", "")

    def test_mts_injects_drafter(self):
        import settings as settings_mod
        from engines.factory import _mlx_gemma_settings_kwargs

        mock_ts = MagicMock()
        mock_ts.turboquant = False
        mock_ts.turboquant_key_bits = 3
        mock_ts.turboquant_val_bits = 4
        mock_ts.model_family = "gemma4"
        mock_ts.num_draft_tokens = 3
        mock_ts.mlx_mts = True
        mock_ts.mlx_drafter_gemma4 = "mlx-community/gemma-4-e4b-it-assistant-bf16"

        mock_root = MagicMock()
        mock_root.translation = mock_ts

        with patch.object(settings_mod, "settings", mock_root):
            kwargs = _mlx_gemma_settings_kwargs()

        assert kwargs.get("model_family") == "gemma4"
        assert kwargs.get("draft_model_id") == "mlx-community/gemma-4-e4b-it-assistant-bf16"
        assert kwargs.get("num_draft_tokens") == 1


class TestMLXGemmaEngineDraftCtor:
    def test_draft_model_id_stored(self):
        from engines.mlx_engine import MLXGemmaEngine

        with patch("engines.mlx_engine.MLX_AVAILABLE", True):
            engine = MLXGemmaEngine(
                model_family="gemma4",
                draft_model_id="mlx-community/gemma-4-e4b-it-assistant-bf16",
                num_draft_tokens=1,
            )
        assert engine._draft_model_id.endswith("assistant-bf16")
        assert engine._num_draft_tokens == 1
        assert engine._draft_model is None  # not loaded yet
