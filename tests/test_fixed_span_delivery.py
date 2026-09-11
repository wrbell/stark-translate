import pytest

from tools.fixed_span_delivery import fixed_span_delivery


def final(start, end, ready, *, offset=10000, padding=0):
    return {
        "timing_schema_version": 2,
        "timing_source": "replay_realtime",
        "sample_start": start,
        "sample_end": end,
        "sample_rate": 1000,
        "padding_samples": padding,
        "timing_stages_ms": {"captured_start": start + offset, "final_ready": ready + offset},
    }


ANCHOR = {
    "id": "fixed",
    "sample_start": 0,
    "sample_end": 8000,
    "speech_end_sample": 7900,
    "endpoint_reason": "smart_cut",
}


def test_changed_segmentation_uses_same_full_span_completion():
    baseline = fixed_span_delivery([ANCHOR], [final(0, 8000, 10000)], sample_rate=1000)
    candidate = fixed_span_delivery([ANCHOR], [final(0, 4000, 4500), final(4000, 8000, 9000)], sample_rate=1000)
    assert baseline["p50_ms"] == 2100
    assert candidate["p50_ms"] == 1100  # not average(500, 1000)
    assert candidate["status"] == "complete"


def test_faster_fragment_cannot_hide_missing_source():
    result = fixed_span_delivery([ANCHOR], [final(0, 4000, 4100)], sample_rate=1000)
    assert result["status"] == "missing_source"
    assert result["p50_ms"] is None


def test_padding_never_supplies_a_gain_target():
    result = fixed_span_delivery([ANCHOR], [final(0, 8000, 8500, padding=512)], sample_rate=1000)
    assert result["status"] == "missing_source"


def test_clock_jumps_and_duplicate_source_rejected():
    with pytest.raises(ValueError, match="discontinuity"):
        fixed_span_delivery([ANCHOR], [final(0, 4000, 5000), final(4000, 8000, 9000, offset=11000)], sample_rate=1000)
    with pytest.raises(ValueError, match="Overlapping"):
        fixed_span_delivery([ANCHOR], [final(0, 5000, 6000), final(4000, 8000, 9000)], sample_rate=1000)


def test_relative_clocks_from_other_sessions_do_not_change_metric():
    a = fixed_span_delivery([ANCHOR], [final(0, 8000, 9000, offset=500000)], sample_rate=1000)
    b = fixed_span_delivery([ANCHOR], [final(0, 8000, 9000, offset=100)], sample_rate=1000)
    assert a["p50_ms"] == b["p50_ms"] == 1100


def test_no_eligible_anchors_is_not_a_completed_comparison():
    result = fixed_span_delivery([{**ANCHOR, "endpoint_reason": "eof"}], [final(0, 8000, 9000)], sample_rate=1000)
    assert result["status"] == "no_eligible_anchors"
    assert result["n"] == 0


def test_final_cannot_be_ready_before_its_source_arrives():
    with pytest.raises(ValueError, match="complete source capture"):
        fixed_span_delivery([ANCHOR], [final(0, 8000, 7000)], sample_rate=1000)


def test_frozen_vad_mask_allows_omitted_silence_but_not_omitted_speech():
    anchor = {**ANCHOR, "required_intervals": [[0, 3000], [4000, 7900]]}
    records = [final(0, 3000, 3500), final(4000, 7900, 8500)]
    result = fixed_span_delivery([anchor], records, sample_rate=1000)
    assert result["status"] == "complete"
    assert result["metric"] == "fixed_vad_positive_source_server_delivery"
    assert fixed_span_delivery([ANCHOR], records, sample_rate=1000)["status"] == "missing_source"
    records[1]["sample_start"] = 4100
    records[1]["timing_stages_ms"]["captured_start"] = 14100
    assert fixed_span_delivery([anchor], records, sample_rate=1000)["status"] == "missing_source"


def test_p95_is_explicit_nearest_rank_for_tail_guard():
    anchors = [
        {
            "id": str(i),
            "sample_start": i * 2000,
            "sample_end": i * 2000 + 1000,
            "speech_end_sample": i * 2000 + 1000,
            "endpoint_reason": "silence",
        }
        for i in range(2)
    ]
    result = fixed_span_delivery(anchors, [final(0, 1000, 1500), final(2000, 3000, 4130)], sample_rate=1000)
    assert result["p95_ms"] == 1130
    assert "nearest-rank" in result["percentile_method"]


def test_endpoint_categories_remain_separate_and_eof_has_no_latency():
    anchors = [
        {
            "id": str(i),
            "sample_start": i * 2000,
            "sample_end": i * 2000 + 1000,
            "speech_end_sample": i * 2000 + 1000,
            "endpoint_reason": reason,
        }
        for i, reason in enumerate(("silence", "max_duration", "eof"))
    ]
    result = fixed_span_delivery(
        anchors,
        [final(0, 1000, 1500), final(2000, 3000, 4130), final(4000, 5000, 8000)],
        sample_rate=1000,
    )
    categories = result["by_opening_endpoint_reason"]
    assert categories["silence"]["p50_ms"] == 500
    assert categories["max_duration"]["p50_ms"] == 1130
    assert categories["eof"] == {"anchors": 1, "eligible": 0, "n": 0, "p50_ms": None, "p95_ms": None}
    assert result["n"] == 2
