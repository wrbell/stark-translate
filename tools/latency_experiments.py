"""Explicit, opt-in controls for the overnight latency research path.

Kept separate from deployment profiles: no experimental default is promoted by
selecting an inference backend. Values are validated before session startup.
``STARK_EXPERIMENT_SERIAL_FINALS`` waits for the preceding translation before
requesting final STT; disabled by default to preserve pipeline overlap.
``STARK_EXPERIMENT_PARTIAL_RECHECK_TRANSLATION`` suppresses queued partial STT
when translation is active at worker admission; false by default (true/false/1/0).
``STARK_EXPERIMENT_PARTIAL_REUSE_MS`` reuses emitted partial STT for silence
finals within the capture-sample gap threshold; 0 disables reuse (maximum 600 ms).
``STARK_EXPERIMENT_PARTIAL_REUSE_KEEP_CONFIDENCE`` lets a reused final keep the
partial's STT confidence so the conservative Marian route stays available; false by default.
``STARK_EXPERIMENT_MLX_CACHE_MB`` applies consistently to STT and translation,
including spawned workers; its default remains 256 MiB.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class LatencyExperiments:
    latest_partial: bool = False
    async_captions: bool = False
    marian_memo: int = 0
    first_preview_s: float = 0.0
    pause_preview_ms: float = 0.0
    speculate_pause_ms: float = 0.0
    clause_preview_s: float = 0.0
    vad_worker: bool = False
    incremental_stt: str = "off"
    rolling_window_s: float = 4.0
    mlx_cache_mb: int = 256
    gemma_prefix_cache: bool = False
    trace: bool = False
    trace_capacity: int = 8192
    early_clause_s: float = 0.0
    early_clause_pause_ms: float = 0.0
    partial_deadline_margin_ms: float = 0.0
    draft_model_id: str = ""
    draft_tokens: int = 0
    partial_recheck_translation: bool = False  # Recheck translation activity on the partial worker.
    serial_finals: bool = False  # Serialize final STT after the preceding translation.
    partial_reuse_ms: int = 0  # Reuse the last emitted partial for nearby silence finals; 0 is off.
    partial_reuse_keep_confidence: bool = (
        False  # Reused finals keep the partial's STT confidence (Marian route stays possible).
    )

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> LatencyExperiments:
        env = os.environ if env is None else env
        defaults = cls()
        values: dict[str, Any] = {}
        for name, default in asdict(defaults).items():
            key = "STARK_EXPERIMENT_" + name.upper()
            raw = env.get(key)
            if raw is None:
                continue
            try:
                if isinstance(default, bool):
                    if raw.lower() not in {"true", "false", "1", "0"}:
                        raise ValueError("expected true/false")
                    value = raw.lower() in {"true", "1"}
                else:
                    value = type(default)(raw)
                values[name] = value
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid {key}: {raw!r}") from exc
        result = cls(**values)
        bounds = {
            "draft_tokens": (0, 4),
            "marian_memo": (0, 4096),
            "first_preview_s": (0, 2),
            "pause_preview_ms": (0, 2000),
            "speculate_pause_ms": (0, 2000),
            "clause_preview_s": (0, 8),
            "rolling_window_s": (1, 8),
            "mlx_cache_mb": (0, 1024),
            "trace_capacity": (1024, 262144),
            "early_clause_s": (0, 8),
            "early_clause_pause_ms": (0, 500),
            "partial_deadline_margin_ms": (0, 1000),
            "partial_reuse_ms": (0, 600),
        }
        for name, (low, high) in bounds.items():
            if not low <= getattr(result, name) <= high:
                raise ValueError(f"STARK_EXPERIMENT_{name.upper()} must be between {low} and {high}")
        if bool(result.draft_model_id) != bool(result.draft_tokens):
            raise ValueError("STARK_EXPERIMENT_DRAFT_MODEL_ID and DRAFT_TOKENS must both be set or both be empty/zero")
        if result.incremental_stt not in {"off", "rolling", "stream"}:
            raise ValueError("STARK_EXPERIMENT_INCREMENTAL_STT must be off, rolling, or stream")
        if bool(result.early_clause_s) != bool(result.early_clause_pause_ms):
            raise ValueError("Early clause duration and pause must both be enabled or both be zero")
        if result.early_clause_s and result.early_clause_s < 0.7:
            raise ValueError("Early clause duration must preserve the 0.7s minimum final length")
        return result

    def as_dict(self) -> dict:
        return asdict(self)
