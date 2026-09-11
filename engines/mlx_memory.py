"""Best-effort Metal working-set wiring, applied once when an engine loads."""

import logging
import os

_logger = logging.getLogger(__name__)
_warning_logged = False


def apply_wired_limit(logger=None) -> int | None:
    """Keep mlx-lm's per-generation restore at the recommended working set."""
    global _warning_logged
    logger = logger if logger is not None else _logger
    if os.environ.get("STARK_MLX_WIRED_LIMIT", "1").strip().lower() in {"0", "false", "off"}:
        logger.info("MLX wired limit left at the Metal default")
        return None
    try:
        import mlx.core as mx

        value = mx.device_info()["max_recommended_working_set_size"]
        # Mock/unsupported API values can implement int() while not being sizes.
        if not isinstance(value, (int, float, str)):
            raise TypeError("Metal recommended working set is not a numeric size")
        limit = int(value)
        mx.set_wired_limit(limit)
        logger.info("MLX wired limit set to %d MiB", limit // (1024 * 1024))
        return limit
    except Exception as exc:
        if not _warning_logged:
            _warning_logged = True
            logger.warning("MLX wired limit unavailable; keeping Metal default: %s", exc)
        return None
