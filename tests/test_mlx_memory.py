"""Working-set wiring uses only mocked Metal APIs."""

import sys
from unittest.mock import MagicMock

import pytest

from engines import mlx_memory


@pytest.fixture
def mx(monkeypatch):
    mock = MagicMock()
    monkeypatch.setitem(sys.modules, "mlx.core", mock)
    monkeypatch.setattr(sys.modules["mlx"], "core", mock)
    monkeypatch.delenv("STARK_MLX_WIRED_LIMIT", raising=False)
    monkeypatch.setattr(mlx_memory, "_warning_logged", False)
    return mock


@pytest.mark.parametrize("value", [8 * 1024**3, str(8 * 1024**3)])
def test_recommended_limit_is_applied_as_integer(mx, value):
    mx.device_info.return_value = {"max_recommended_working_set_size": value}
    log = MagicMock()
    assert mlx_memory.apply_wired_limit(log) == int(value)
    mx.set_wired_limit.assert_called_once_with(int(value))
    log.info.assert_called_once_with("MLX wired limit set to %d MiB", 8192)


@pytest.mark.parametrize("disabled", ["0", "false", " OFF "])
def test_disabled_leaves_metal_default(mx, monkeypatch, disabled):
    monkeypatch.setenv("STARK_MLX_WIRED_LIMIT", disabled)
    log = MagicMock()
    assert mlx_memory.apply_wired_limit(log) is None
    mx.device_info.assert_not_called()
    mx.set_wired_limit.assert_not_called()
    log.info.assert_called_once_with("MLX wired limit left at the Metal default")


@pytest.mark.parametrize("failure", ["device_info", "set_wired_limit", "missing", "invalid", "mock"])
def test_failure_is_nonfatal_and_warns_once(mx, failure):
    if failure != "mock":
        mx.device_info.return_value = {"max_recommended_working_set_size": 1024**3}
    if failure in {"device_info", "set_wired_limit"}:
        getattr(mx, failure).side_effect = RuntimeError("unsupported")
    elif failure == "missing":
        mx.device_info.return_value = {}
    elif failure == "invalid":
        mx.device_info.return_value = {"max_recommended_working_set_size": "not an int"}
    log = MagicMock()
    assert mlx_memory.apply_wired_limit(log) is None
    assert mlx_memory.apply_wired_limit(log) is None
    log.warning.assert_called_once()
    if failure != "set_wired_limit":
        mx.set_wired_limit.assert_not_called()
