"""Tests for tools/health_check.py — MLX and CUDA backends (mocked)."""

from unittest.mock import MagicMock, patch

import pytest


class TestHealthCheckMLX:
    def test_load_mlx_passes_adapter_path(self):
        from tools import health_check as hc

        mock_tokenizer = MagicMock()
        mock_tokenizer.convert_tokens_to_ids.return_value = 106
        mock_tokenizer.eos_token_id = 1
        mock_tokenizer._eos_token_ids = {1, 106}
        mock_model = MagicMock()
        mock_load = MagicMock(return_value=(mock_model, mock_tokenizer))

        with (
            patch("mlx.core.set_cache_limit"),
            patch("mlx_lm.load", mock_load),
            patch("os.path.exists", return_value=True),
        ):
            model, tok, translate_fn = hc._load_mlx(
                "mlx-community/translategemma-4b-it-4bit",
                "/adapters/active",
                "translategemma",
            )

        assert model is mock_model
        assert tok is mock_tokenizer
        assert callable(translate_fn)
        mock_load.assert_called_once_with(
            "mlx-community/translategemma-4b-it-4bit",
            adapter_path="/adapters/active",
        )

    def test_run_health_check_mlx_mocked(self):
        from tools import health_check as hc

        mock_model = MagicMock()
        mock_tok = MagicMock()

        def _fake_translate(_m, _t, text, **_kwargs):
            # Return a Spanish-ish string that includes expected substrings for canary 1
            # We patch canary_sentences to a single controlled test instead.
            return "la propiciación de Cristo"

        with (
            patch.object(hc, "_load_mlx", return_value=(mock_model, mock_tok, _fake_translate)),
            patch.object(
                hc,
                "canary_sentences",
                return_value=[
                    {
                        "en": "the propitiation of Christ",
                        "expected_substrings": ["propiciación"],
                    }
                ],
            ),
        ):
            result = hc.run_health_check(
                base_model="mlx-community/x",
                adapter_dir=None,
                max_latency=5.0,
                n_canaries=1,
                backend="mlx",
            )

        assert result["backend"] == "mlx"
        assert result["all_pass"] is True
        assert result["passed"] == 1

    def test_run_health_check_cuda_uses_bnb_loader(self):
        from tools import health_check as hc

        mock_model = MagicMock()
        mock_tok = MagicMock()

        with (
            patch.object(hc, "_load_cuda", return_value=(mock_model, mock_tok, lambda *a, **k: "gracia")) as mock_cuda,
            patch.object(
                hc,
                "canary_sentences",
                return_value=[{"en": "grace", "expected_substrings": ["gracia"]}],
            ),
        ):
            result = hc.run_health_check(
                base_model="google/translategemma-4b-it",
                adapter_dir="/a",
                max_latency=5.0,
                n_canaries=1,
                backend="cuda",
            )

        mock_cuda.assert_called_once()
        assert result["backend"] == "cuda"
        assert result["all_pass"] is True

    def test_invalid_backend_raises(self):
        from tools import health_check as hc

        with pytest.raises(ValueError, match="Unsupported backend"):
            hc.run_health_check(
                base_model="x",
                adapter_dir=None,
                max_latency=5.0,
                backend="tpu",
            )
