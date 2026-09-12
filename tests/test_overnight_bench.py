"""Measurement gates must not confuse cadence, boundaries, or missing coverage."""

import csv
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tools import overnight_bench as bench
from tools.overnight_bench import compare_outputs, coverage_missing, inspect_run, preview_browsers, ranges, schedule


def test_visible_preview_join_rejects_unknown_duplicate_and_hidden_events():
    partials = [
        {"event_id": "s:partial:1", "utterance_id": 1, "text_es": "Dios", "preview_revision": 1},
        {"event_id": "s:partial:2", "utterance_id": 1, "text_es": "Dios ama", "preview_revision": 2},
    ]
    for row in partials:
        row.update(session_id="s", timing_schema_version=2, timing_source="replay_realtime")
    ack = {
        "event": "caption_rendered",
        "event_id": "s:partial:1",
        "session_id": "s",
        "visible": True,
        "stage": "partial",
        "client_id": "audience",
        "speech_start_to_preview_ack_upper_bound_ms": 650,
        "receive_to_render_ms": 20,
        "timing_schema_version": 2,
        "timing_source": "replay_realtime",
        "preview_revision": 1,
        "utterance_id": 1,
    }
    result = preview_browsers(
        partials,
        [
            ack,
            ack,
            {**ack, "event_id": "unknown"},
            {**ack, "event_id": "s:partial:2", "visible": False},
            {**ack, "session_id": "other"},
        ],
        "s",
    )
    assert result["cohorts"][0]["visible_preview_events"] == 1
    assert result["cohorts"][0]["first_preview_ack_upper_bound_ms"] == {"n": 1, "p50": 650, "p95": 650}
    assert result["emitted_translated_preview_events"] == 2


def test_final_only_browser_has_explicitly_missing_preview_coverage():
    partial = {"session_id": "s", "event_id": "s:p:1", "text_es": "Dios", "timing_source": "replay_realtime"}
    ack = {
        "session_id": "s",
        "event": "caption_rendered",
        "visible": True,
        "stage": "complete",
        "timing_schema_version": 2,
        "client_id": "audience",
    }
    result = preview_browsers([partial], [ack], "s")
    assert result["client_count"] == 1 and not result["available"]
    assert result["cohorts"][0]["first_preview_ack_upper_bound_ms"]["n"] == 0
    assert result["cohorts"][0]["missing_preview_event_ids"] == ["s:p:1"]


def test_first_stream_acks_use_client_chunk_identity_without_emitted_previews():
    ack = {
        "session_id": "s",
        "event": "caption_rendered",
        "event_id": "s:stream:1",
        "chunk_id": 1,
        "visible": True,
        "stage": "first_stream",
        "timing_schema_version": 2,
        "timing_source": "replay_realtime",
        "caption_delivery_mode": "queued",
        "client_id": "audience",
        "speech_end_to_ack_upper_bound_ms": 1100,
    }
    result = preview_browsers(
        [],
        [
            ack,
            ack,
            {**ack, "event_id": "s:other", "chunk_id": "1", "speech_end_to_ack_upper_bound_ms": 9999},
            {**ack, "client_id": "obs", "speech_end_to_ack_upper_bound_ms": 1200},
            {**ack, "event_id": "s:complete", "stage": "complete", "speech_end_to_ack_upper_bound_ms": 1800},
            {**ack, "client_id": "hidden", "visible": False},
            {**ack, "session_id": "old"},
        ],
        "s",
    )
    assert result["client_count"] == 2
    assert result["emitted_translated_preview_events"] == 0
    assert [row["first_visible_ms"] for row in result["cohorts"]] == [
        {"n": 1, "p50": 1100, "p95": 1100},
        {"n": 1, "p50": 1200, "p95": 1200},
    ]
    assert all(row["first_preview_ack_upper_bound_ms"]["n"] == 0 for row in result["cohorts"])


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
    (tmp_path / f"partials_{session}.jsonl").write_text("\n".join(json.dumps(row) for row in reversed(partials)))
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


def test_configuration_is_standard_tts_off_and_independent_of_ambient_operator_state(monkeypatch):
    monkeypatch.setenv("STARK_PROFILE", "lite-cpu")
    monkeypatch.setenv("STARK_SESSION_KIND", "synthetic")
    monkeypatch.setenv("STARK_TTS_ENABLED", "true")
    monkeypatch.setenv("STARK_TRANSLATE_MARIAN_BACKEND", "hf")
    monkeypatch.setenv("STARK_EXPERIMENT_INCREMENTAL_STT", "stream")
    args, env, expected = bench.configuration(
        {}, {"env": {"STARK_EXPERIMENT_FIRST_PREVIEW_S": "0.35"}}, {"lang": "en", "provenance": "church_replay"}, "e4b"
    )
    assert bench.argument_value(args, "--profile") == "standard" and "--no-tts" in args
    assert bench.argument_value(args, "--replay-wait-client-seconds") == "15"
    assert "STARK_PROFILE" not in env and "STARK_TTS_ENABLED" not in env
    assert "STARK_TRANSLATE_MARIAN_BACKEND" not in env
    assert env["STARK_SESSION_KIND"] == "replay" and env["HF_HUB_OFFLINE"] == "1"
    assert expected["first_preview_s"] == 0.35 and expected["incremental_stt"] == "off"
    with pytest.raises(ValueError, match="unknown"):
        bench.configuration({}, {"env": {"STARK_EXPERIMENT_FRIST_PREVIEW_S": "0.3"}}, {"lang": "en"}, "e4b")
    with pytest.raises(ValueError, match="standard"):
        bench.configuration({}, {"arguments": ["--profile", "lite-cpu"]}, {"lang": "en"}, "e4b")


def completed_fixture(tmp_path, session, expected, clip):
    metrics = tmp_path / "metrics"
    metrics.mkdir(exist_ok=True)
    row = {
        "chunk_id": "1",
        "utterance_id": "1",
        "sample_start": "0",
        "sample_end": "16000",
        "sample_rate": "16000",
        "speech_end_sample": "16000",
        "endpoint_reason": "silence",
        "timing_source": "replay_realtime",
        "timing_schema_version": "2",
        "speech_end_to_final_ms": "800",
        "english": "God loves us.",
        "spanish_a": "Dios nos ama.",
    }
    with (metrics / f"ab_metrics_{session}.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    diagnostics = metrics / f"diagnostics_{session}.jsonl"
    barrier = {
        "requested_seconds": 15,
        "actual_wait_seconds": 0.25,
        "status": "connected",
        "connected_clients": 1,
        "visibility_confirmed": False,
    }
    diagnostics.write_text(
        json.dumps(
            {
                "event": "session_summary",
                "chunks_completed": 1,
                "latency_experiment_configuration": expected,
                "replay_client_wait": barrier,
            }
        )
        + "\n"
    )
    return {
        "session_id": session,
        "returncode": 0,
        "session_lifecycle": {
            "status": "completed",
            "session_id": session,
            "schema_version": 1,
            "exit_code": 0,
            "pipeline_sha256": "code",
            "diagnostics": {"sha256": bench.fingerprint(diagnostics), "size_bytes": diagnostics.stat().st_size},
        },
        "session_metadata": {
            "session_id": session,
            "session_kind": "replay",
            "input_audio_sha256": clip["sha256"],
            "source_lang": clip["lang"],
            "replay_speed": 1,
            "replay_client_wait": barrier,
            "profile": {"name": "standard"},
            "backend": "mlx",
            "latency_experiment_configuration": expected,
        },
    }


def test_completed_word_alone_cannot_hide_changed_diagnostics(tmp_path):
    clip = {"lang": "en", "sha256": "input"}
    result = completed_fixture(tmp_path, "s", {}, clip)
    observed = inspect_run(result, tmp_path / "metrics")
    source = {"all_code_sha256": {"dry_run_ab.py": "code"}}
    assert bench.completion_errors(result, observed, tmp_path / "metrics", {}, clip, source) == []
    (tmp_path / "metrics/diagnostics_s.jsonl").write_text("changed")
    assert "unchanged" in bench.completion_errors(result, observed, tmp_path / "metrics", {}, clip, source)[0]


def test_runner_records_source_change_and_stops_before_more_inference(tmp_path, monkeypatch):
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"fixture input")
    clip = {
        "id": "clip",
        "path": "audio.wav",
        "sha256": bench.fingerprint(audio),
        "lang": "en",
        "provenance": "church_replay",
    }
    spec = {"clips": [clip], "sizes": ["e4b"], "experiments": [{"name": "baseline"}]}
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec))
    source = {"all_code_sha256": {"dry_run_ab.py": "code"}, "versions": {}}
    monkeypatch.setattr(bench, "ROOT", tmp_path)
    monkeypatch.setattr(
        bench, "source_snapshot", Mock(side_effect=[source, source, {**source, "versions": {"mlx": "changed"}}])
    )

    def replay(clip, wav, session, *args, **kwargs):
        expected = bench.LatencyExperiments.from_env(kwargs["env"]).as_dict()
        return completed_fixture(tmp_path, session, expected, clip)

    child = Mock(side_effect=replay)
    monkeypatch.setattr(bench, "run_replay", child)
    destination = tmp_path / "runs"
    with pytest.raises(RuntimeError, match="Stopping"):
        bench.run(
            SimpleNamespace(
                spec=path,
                output=destination,
                tag="smoke",
                repeats=1,
                ws_port=8765,
                http_port=8080,
                timeout=240,
                continue_on_error=False,
            )
        )
    child.assert_called_once()
    result = json.loads((destination / "smoke_baseline_e4b_r0_clip_en.json").read_text())
    assert not result["completion_validation"]["valid"]
    assert "during inference" in result["error"]


def test_report_excludes_failed_fast_samples_instead_of_rewarding_them(tmp_path):
    inputs = tmp_path / "runs"
    inputs.mkdir()
    for repeat, failed, latency in [(0, False, 1000), (1, True, 10)]:
        item = {
            "session_id": f"s{repeat}",
            "experiment": "baseline",
            "repeat": repeat,
            "size": "e4b",
            "clip_id": "clip",
            "completion_validation": {"valid": not failed},
            "observed": {
                "finals": [],
                "coverage_intervals_s": [],
                "endpoint_samples_ms": {"silence|replay_realtime": [latency]},
                "first_preview_samples_ms": [latency],
                "within_utterance_gap_samples_ms": [],
                "browser": {},
                "preview_browser": {
                    "cohorts": [
                        {
                            "client_id": "audience",
                            "timing_source": "replay_realtime",
                            "caption_delivery_mode": "queued",
                            "first_visible_ms": bench.stats([latency - 100]),
                        }
                    ]
                },
            },
        }
        if failed:
            item["error"] = "recording failed"
        (inputs / f"{repeat}.json").write_text(json.dumps(item))
    bench.report(SimpleNamespace(input=inputs, output=tmp_path / "report"))
    result = json.loads((tmp_path / "report/comparison.json").read_text())
    assert result["eligible_runs"] == 1 and len(result["excluded_sessions"]) == 1
    group = result["groups"][0]
    assert group["endpoints"]["silence|replay_realtime"] == {"n": 1, "p50": 1000, "p95": 1000}
    assert group["first_preview_ms"]["p50"] == 1000 and group["failures"] == 1
    markdown = (tmp_path / "report/README.md").read_text()
    assert "First visible p50 / p95 ms" in markdown
    assert "| s0 | audience | replay_realtime | queued | 900.0 / 900.0 | 1 |" in markdown
    assert "| s1 | audience" not in markdown


@pytest.mark.parametrize(
    "change",
    [
        None,
        {"status": "timed_out"},
        {"requested_seconds": 0},
        {"connected_clients": 0},
        {"actual_wait_seconds": float("nan")},
        {"visibility_confirmed": True},
    ],
)
def test_benchmark_requires_resolved_client_barrier_evidence(tmp_path, change):
    clip = {"lang": "en", "sha256": "input"}
    result = completed_fixture(tmp_path, "s", {}, clip)
    observed = inspect_run(result, tmp_path / "metrics")
    if change is None:
        result["session_metadata"].pop("replay_client_wait")
    else:
        result["session_metadata"]["replay_client_wait"].update(change)
    errors = bench.completion_errors(
        result, observed, tmp_path / "metrics", {}, clip, {"all_code_sha256": {"dry_run_ab.py": "code"}}
    )
    assert any("client barrier" in message for message in errors)
