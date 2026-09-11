"""Environment validation for the opt-in partial reuse arm."""

import pytest

from tools.latency_experiments import LatencyExperiments


@pytest.mark.parametrize("value", ["0", "300", "600"])
def test_partial_reuse_configuration_is_bounded_and_serialized(value):
    assert LatencyExperiments.from_env({}).partial_reuse_ms == 0
    config = LatencyExperiments.from_env({"STARK_EXPERIMENT_PARTIAL_REUSE_MS": value})
    assert config.partial_reuse_ms == int(value)
    assert config.as_dict()["partial_reuse_ms"] == int(value)


@pytest.mark.parametrize("value", ["601", "-1", "300.0", "true"])
def test_partial_reuse_configuration_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="STARK_EXPERIMENT_PARTIAL_REUSE_MS"):
        LatencyExperiments.from_env({"STARK_EXPERIMENT_PARTIAL_REUSE_MS": value})
