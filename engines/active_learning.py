"""Active learning logger for STT fallback events.

Logs every fallback event (primary model -> retry with fallback model) as a
JSONL record.  These records feed the active learning feedback loop described
in CLAUDE.md Layer 6 -- flagged low-confidence segments are prioritised for
human review and subsequent fine-tuning cycles.

Usage:
    from engines.active_learning import log_fallback_event
    log_fallback_event(original_result, retry_result, audio_hash="abc123")
"""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from engines.base import STTResult

logger = logging.getLogger(__name__)

# Default output path relative to project root
_DEFAULT_OUTPUT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "metrics",
    "fallback_events.jsonl",
)


def log_fallback_event(
    original: STTResult,
    retry: STTResult,
    audio_hash: str,
    output_path: Optional[str] = None,
    *,
    chosen: str = "retry",
    primary_model: str = "",
    fallback_model: str = "",
) -> None:
    """Append a JSONL record documenting a fallback transcription event.

    Args:
        original:       STTResult from the primary (Turbo) model.
        retry:          STTResult from the fallback model.
        audio_hash:     Hash or identifier for the audio chunk (for dedup
                        and linking back to WAV files in stark_data/).
        output_path:    Path to the JSONL file.  Defaults to
                        ``metrics/fallback_events.jsonl`` in the project root.
        chosen:         Which result was selected: ``"original"`` or ``"retry"``.
        primary_model:  Model ID of the primary engine (for logging).
        fallback_model: Model ID of the fallback engine (for logging).
    """
    if output_path is None:
        output_path = _DEFAULT_OUTPUT_PATH

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "audio_hash": audio_hash,
        "primary_model": primary_model,
        "fallback_model": fallback_model,
        "original_text": original.text,
        "retry_text": retry.text,
        "original_confidence": original.confidence,
        "retry_confidence": retry.confidence,
        "original_avg_logprob": original.avg_logprob,
        "retry_avg_logprob": retry.avg_logprob,
        "original_compression_ratio": original.compression_ratio,
        "retry_compression_ratio": retry.compression_ratio,
        "original_latency_ms": round(original.latency_ms, 1),
        "retry_latency_ms": round(retry.latency_ms, 1),
        "which_was_chosen": chosen,
    }

    try:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.debug("Fallback event logged to %s", output_path)
    except OSError as exc:
        logger.warning("Failed to write fallback event log: %s", exc)
