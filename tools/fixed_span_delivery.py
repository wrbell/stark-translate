"""Segmentation-independent server delivery measurements for uninterrupted 1x replays.

Anchors are frozen source spans, never recomputed from a candidate's segmentation.
This measures completion of audio-span captions, not semantic/reference-word recall.
Original schema-2 chunk latency and endpoint counts remain separate measures.
"""

from __future__ import annotations

import math
import statistics

from tools.source_coverage import merged


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("Missing or nonfinite timing/position")
    return value


def fixed_span_delivery(anchors: list[dict], finals: list[dict], *, sample_rate: int) -> dict:
    """Require caller-verified same input hash, uninterrupted replay and frozen anchors.

    Replay-origin offsets must agree within 2ms across emitted final records. This
    rejects pause/resume clock jumps; no cross-process perf_counter comparison.
    """
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0:
        raise ValueError("Invalid sample rate")
    offsets, records = [], []
    for row in finals:
        if row.get("timing_schema_version") != 2 or row.get("timing_source") != "replay_realtime":
            raise ValueError("Only schema-2 real-time replay finals are comparable")
        if row.get("sample_rate") != sample_rate:
            raise ValueError("Mixed source sample rates")
        start, end = (_number(row.get(key)) for key in ("sample_start", "sample_end"))
        if start < 0 or end <= start:
            raise ValueError("Invalid final span")
        stages = row.get("timing_stages_ms") or {}
        captured, ready = (_number(stages.get(key)) for key in ("captured_start", "final_ready"))
        if ready < captured:
            raise ValueError("Final precedes capture")
        offsets.append(captured - start * 1000 / sample_rate)
        records.append({"start": start, "end": end, "ready": ready, "padding": row.get("padding_samples", 0)})
    if not offsets:
        return {"schema_version": 1, "status": "no_finals", "anchors": [], "p50_ms": None, "p95_ms": None}
    if max(offsets) - min(offsets) > 2:
        raise ValueError("Replay clock discontinuity: pause/resume or inconsistent source mapping")
    origin = statistics.median(offsets)
    observed = [(r["start"], r["end"]) for r in records]
    duplicate_samples = sum(b - a for a, b in observed) - sum(b - a for a, b in merged(observed))
    if duplicate_samples:
        raise ValueError("Overlapping final source spans")
    results = []
    for anchor in anchors:
        start, end, speech_end = (_number(anchor.get(k)) for k in ("sample_start", "sample_end", "speech_end_sample"))
        if not 0 <= start < speech_end <= end:
            raise ValueError("Invalid frozen source anchor")
        eligible = anchor.get("endpoint_reason") not in {"eof", "pause", "stop"} and not anchor.get(
            "padding_samples", 0
        )
        pieces, completed_at = [], None
        for row in sorted(records, key=lambda r: r["ready"]):
            if row["padding"]:
                continue  # synthetic tail cannot supply a gain target
            left, right = max(start, row["start"]), min(end, row["end"])
            if right <= left:
                continue
            pieces.append((left, right))
            if sum(b - a for a, b in merged(pieces)) == end - start:
                completed_at = row["ready"] - origin
                break
        results.append(
            {
                "anchor_id": anchor["id"],
                "sample_start": start,
                "sample_end": end,
                "speech_end_sample": speech_end,
                "eligible": eligible,
                "covered": completed_at is not None,
                "speech_end_to_span_final_ms": (
                    completed_at - speech_end * 1000 / sample_rate if eligible and completed_at is not None else None
                ),
            }
        )
    values = sorted(r["speech_end_to_span_final_ms"] for r in results if r["speech_end_to_span_final_ms"] is not None)
    # Match NumPy's ordinary linear percentile without importing an inference stack.
    position = (len(values) - 1) * 0.95
    lower = int(position) if values else 0
    p95 = (
        values[lower] + (values[min(lower + 1, len(values) - 1)] - values[lower]) * (position - lower)
        if values
        else None
    )
    return {
        "schema_version": 1,
        "metric": "fixed_source_span_server_delivery",
        "status": (
            "no_eligible_anchors"
            if not any(r["eligible"] for r in results)
            else "complete"
            if all(r["covered"] for r in results if r["eligible"])
            else "missing_source"
        ),
        "replay_origin_at_ms": origin,
        "anchors": results,
        "n": len(values),
        "p50_ms": statistics.median(values) if values else None,
        "p95_ms": p95,
        "quality_scope": "Audio-span completion only; source/translation references scored separately",
    }
