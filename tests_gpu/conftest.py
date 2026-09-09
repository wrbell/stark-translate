"""GPU-only tests (real MLX models). Not collected by CI.

Run explicitly on Apple Silicon:
    STARK_RUN_GPU_TESTS=1 stt_env/bin/python -m pytest tests_gpu -q

Unlike ``tests/conftest.py`` this file does NOT mock ``mlx``/``mlx_lm``.
"""

from __future__ import annotations

import os

import pytest


def pytest_collection_modifyitems(config, items):
    if os.environ.get("STARK_RUN_GPU_TESTS") == "1":
        return
    skip = pytest.mark.skip(reason="set STARK_RUN_GPU_TESTS=1 to run real-model GPU tests")
    for item in items:
        item.add_marker(skip)
