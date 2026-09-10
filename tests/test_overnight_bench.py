"""Measurement gates must not confuse cadence, boundaries, or missing coverage."""

import csv
import json

import pytest

from tools.overnight_bench import compare_outputs, coverage_missing, inspect_run, ranges, schedule


def test_coverage_union_accounts_for_overlap_and_internal_holes():
    baseline = ranges(
        [
            {"sample_start": 0, "sample_end": 100, "sample_rate": 10},
            {"sample_start": 50, "sample_end": 150, "sample_rate": 10},
        ]
    )
    candidate = ranges(
        [
            {"sample_start": 0, "sample_end": 50, "sample_rate": 10},
            {"sample_start": 100, "sample_end": 150, "sample_rate": 10},
        ]
    )
    assert baseline == [(0, 15)]
    assert coverage_missing(baseline, candidate) == 5
    assert coverage_missing(candidate, baseline) == 0


def test_read_production_records_keeps_endpoint_and_preview_definitions_separate(tmp_path):
    session = "measurement_en"
    columns = [
        "chunk_id",
        "sample_start",
        "sample_end",
        "sample_rate",
        "timing_schema_version",
        "endpoint_reason",
        "timing_source",
        "speech_end_to_final_ms",
    ]
    with (tmp_path / f"ab_metrics_{session}.csv").open("w", newline="") as source:
        writer = csv.DictWriter(source, fieldnames=columns)
        writer.writeheader()
        writer.writerows(
            [
                dict(zip(columns, [1, 0, 100, 10, 2, "silence", "replay_realtime", 800], strict=True)),
                dict(zip(columns, [2, 100, 200, 10, 2, "smart_cut", "replay_realtime", 3200], strict=True)),
                dict(zip(columns, [3, 200, 300, 10, "", "silence", "unknown", ""], strict=True)),
            ]
        )
    partials = [
        {"utterance_id": 1, "text_es": "", "speech_start_to_partial_ms": 100, "emitted_at_ms": 100},
        {"utterance_id": 1, "text_es": "Dios", "speech_start_to_partial_ms": 500, "emitted_at_ms": 500},
        {"utterance_id": 1, "text_es": "Dios ama", "speech_start_to_partial_ms": 1100, "emitted_at_ms": 1100},
        {"utterance_id": 2, "text_es": "Gracia", "speech_start_to_partial_ms": 600, "emitted_at_ms": 9000},
    ]
    (tmp_path / f"partials_{session}.jsonl").write_text("\n".join(json.dumps(row) for row in partials))
    (tmp_path / f"diagnostics_{session}.jsonl").write_text(
        json.dumps({"event": "session_summary", "chunks_completed": 3})
    )
    observed = inspect_run({"session_id": session}, tmp_path)
    assert observed["first_preview_samples_ms"] == [500, 600]
    assert observed["within_utterance_gap_samples_ms"] == [600]
    assert observed["endpoint_samples_ms"] == {"silence|replay_realtime": [800], "smart_cut|replay_realtime": [3200]}
    assert observed["session_summary"]["chunks_completed"] == 3
    assert observed["browser"]["coverage"]["received_final_chunks"] == 0
    assert observed["browser"]["coverage"]["missing_final_chunks"] == ["1", "2", "3"]


def test_changed_boundaries_are_unmatched_not_claimed_equal():
    row = {
        "sample_start": 0,
        "sample_end": 100,
        "sample_rate": 10,
        "endpoint_reason": "silence",
        "english": "God loves us.",
        "spanish_a": "Dios nos ama.",
    }
    baseline = {"finals": [row], "coverage_intervals_s": [(0, 10)]}
    candidate = {"finals": [{**row, "sample_end": 80}], "coverage_intervals_s": [(0, 8)]}
    result = compare_outputs(baseline, candidate)
    assert result["matched_final_bounds"] == 0
    assert result["baseline_unmatched"] == 1
    assert result["missing_source_s"] == 2
    assert result["quality_review"].startswith("pending")


def test_schedule_alternates_models_configurations_and_brackets_with_anchors():
    spec = {
        "experiments": [{"name": name} for name in ("baseline", "first_preview", "prefix")],
        "clips": [{"id": "sermon"}],
    }
    planned = list(schedule(spec, 3))
    assert len(planned) == 24
    assert [(config["name"], size) for _, config, _, size in planned[:8]] == [
        ("baseline", "e4b"),
        ("baseline", "e2b"),
        ("first_preview", "e4b"),
        ("first_preview", "e2b"),
        ("prefix", "e4b"),
        ("prefix", "e2b"),
        ("baseline_anchor", "e4b"),
        ("baseline_anchor", "e2b"),
    ]
    assert [(config["name"], size) for _, config, _, size in planned[8:12]] == [
        ("baseline", "e2b"),
        ("baseline", "e4b"),
        ("prefix", "e2b"),
        ("prefix", "e4b"),
    ]
    with pytest.raises(ValueError, match="first baseline"):
        list(schedule({**spec, "experiments": [{"name": "prefix"}]}, 3))
