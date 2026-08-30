"""Tests for MLX Mac pipeline parallelization (CUDA-parity overlap).

CI-friendly: no Metal / Apple Silicon required.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestPipelinePoolMaxWorkers:
    def test_mlx_matches_cuda(self):
        from dry_run_ab import pipeline_pool_max_workers

        assert pipeline_pool_max_workers("mlx") == 2
        assert pipeline_pool_max_workers("cuda") == 2

    def test_cpu_stays_serialized(self):
        from dry_run_ab import pipeline_pool_max_workers

        assert pipeline_pool_max_workers("cpu") == 1
        assert pipeline_pool_max_workers("auto") == 1
        assert pipeline_pool_max_workers("unknown") == 1


class TestMaterializeMlxModel:
    def test_noop_when_mlx_unavailable(self):
        import engines.mlx_engine as me

        with patch.object(me, "mx", None):
            me.materialize_mlx_model(MagicMock())  # must not raise

    def test_noop_when_model_is_none(self):
        from engines.mlx_engine import materialize_mlx_model

        materialize_mlx_model(None)  # must not raise

    def test_evals_parameters_when_present(self):
        import engines.mlx_engine as me

        fake_mx = MagicMock()
        model = MagicMock()
        params = MagicMock()
        model.parameters.return_value = params
        # Ensure hasattr(model, "parameters") is True via MagicMock default

        with patch.object(me, "mx", fake_mx):
            me.materialize_mlx_model(model)

        fake_mx.eval.assert_called_once_with(params)

    def test_falls_back_to_synchronize(self):
        import engines.mlx_engine as me

        fake_mx = MagicMock()
        # Model without parameters()/trainable_parameters()
        model = object()

        with patch.object(me, "mx", fake_mx):
            me.materialize_mlx_model(model)

        fake_mx.synchronize.assert_called_once()


class TestMlxDependencyPins:
    """Ensure packaging requires mlx-lm with thread-local generation stream."""

    def test_pyproject_mlx_lm_min_version(self):
        text = Path("pyproject.toml").read_text()
        assert "mlx-lm>=0.31.3" in text
        assert "mlx>=0.31.2" in text

    def test_requirements_mac_mlx_lm_min_version(self):
        text = Path("requirements-mac.txt").read_text()
        assert "mlx-lm>=0.31.3" in text or "mlx-lm==0.31." in text


class TestMultiprocessSettingCopy:
    def test_description_mentions_in_process_default(self):
        from settings import PipelineSettings

        field = PipelineSettings.model_fields["multiprocess"]
        desc = field.description or ""
        assert "0.31.2" in desc or "in-process" in desc.lower() or "escape" in desc.lower()
