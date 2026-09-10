"""Fail a primary-model benchmark if startup silently chose another model.

``fallback_on_low_conf=False`` disables per-call retries, not MLX startup
fallback. Benchmark callers therefore check identity after loading and before
warmup/measurement. This validates the engine's public model identity, not file
digests or quality; complete artifact provenance remains the harness's concern.
"""

from contextlib import suppress


def load_primary_model(engine, requested_model_id: str) -> dict:
    """Load the requested primary or fail visibly, releasing a rejected model."""
    try:
        engine.load()
        actual = engine.model_id
        if not isinstance(actual, str) or actual != requested_model_id:
            raise RuntimeError(
                f"Benchmark primary-model identity mismatch: requested {requested_model_id!r}, "
                f"loaded {actual!r}. This arm is ineligible; no primary-model measurements were taken."
            )
    except BaseException:
        with suppress(Exception):
            engine.unload()
        raise
    return {"requested_model_id": requested_model_id, "actual_model_id": actual, "primary_identity_verified": True}
