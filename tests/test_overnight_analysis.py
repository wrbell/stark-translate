"""Synthetic evidence checks for the read-only overnight report; no model imports."""

import csv
import json

import pytest

from tools import overnight_analysis as analysis


def fixture_matrix(tmp_path, *, repeats=3, omit=(), changes=None):
    metrics, inputs = tmp_path / "metrics", tmp_path / "metrics" / "screen"
    inputs.mkdir(parents=True)
    provenance = {
        "tag": "screen",
        "repeats": repeats,
        "spec": {
            "sizes": ["e4b"],
            "clips": [{"id": "clip", "lang": "en"}],
            "experiments": [{"name": "baseline"}, {"name": "candidate"}],
        },
    }
    (inputs / "provenance.json").write_text(json.dumps(provenance))
    paths = {}
    for repeat in range(repeats):
        for config in ("baseline", "candidate", "baseline_anchor"):
            if (repeat, config) in omit:
                continue
            session = f"screen_{config}_e4b_r{repeat}_clip_en"
            delay = 700 if config == "candidate" else 1000
            first = 400 if config == "candidate" else 600
            row = {
                "chunk_id": "1",
                "utterance_id": "1",
                "sample_start": "0",
                "sample_end": "16000",
                "sample_rate": "16000",
                "speech_end_sample": "16000",
                "timing_source": "replay_realtime",
                "timing_schema_version": "2",
                "endpoint_reason": "silence",
                "source_lang": "en",
                "target_lang": "es",
                "english": "God loves us.",
                "spanish_a": "Dios nos ama.",
                "speech_end_to_final_ms": str(delay),
            }
            partials = []
            for revision, offset in [(1, first), (2, first + 300)]:
                partials.append(
                    {
                        "session_id": session,
                        "event_id": f"{session}:p:{revision}",
                        "utterance_id": 1,
                        "timing_schema_version": 2,
                        "timing_source": "replay_realtime",
                        "caption_delivery_mode": "awaited",
                        "preview_revision": revision,
                        "captured_start_at_ms": 1000,
                        "emitted_at_ms": 1000 + offset,
                        "speech_start_to_partial_ms": offset,
                        "processed_audio_s": 0.5,
                        "source_lang": "en",
                        "target_lang": "es",
                        "text_en": "God" if revision == 1 else "God loves",
                        "text_es": "Dios" if revision == 1 else "Dios ama",
                    }
                )
            acks = [
                {
                    **p,
                    "event": "caption_rendered",
                    "stage": "partial",
                    "visible": True,
                    "client_id": "browser",
                    "speech_start_to_preview_ack_upper_bound_ms": p["speech_start_to_partial_ms"] + 10,
                    "receive_to_render_ms": 5,
                }
                for p in partials
            ]
            acks.append(
                {
                    "session_id": session,
                    "event": "caption_rendered",
                    "stage": "complete",
                    "visible": True,
                    "client_id": "browser",
                    "event_id": f"{session}:f:1",
                    "chunk_id": 1,
                    "sample_start": 0,
                    "sample_end": 16000,
                    "sample_rate": 16000,
                    "speech_end_sample": 16000,
                    "timing_schema_version": 2,
                    "timing_source": "replay_realtime",
                    "speech_end_to_ack_upper_bound_ms": delay + 10,
                    "receive_to_render_ms": 5,
                }
            )
            config_fields = {}
            summary = {
                "event": "session_summary",
                "latency_experiment_configuration": config_fields,
                "latency_experiment_counters": {"partial_emitted": 2},
                "latency_trace": {
                    "enabled": True,
                    "total_events": 10,
                    "discarded_old_events": 3,
                    "events": [{"event": "audio_dequeued", "capture_age_ms": 7, "queue_depth": 0}],
                },
            }
            diag = [
                {"chunk_id": 1, "event_id": f"{session}:f:1", "timing_stages_ms": {"final_ready": 1200 + delay}},
                summary,
            ]
            lifecycle = {
                "status": "completed",
                "exit_code": 0,
                "models": {"stt": {"revision": "p"}, "translation_a": {"revision": "g"}},
                "memory": {"peak_rss_bytes": 1000000, "peak_metal_bytes": 2000000},
                "persistence": {"pending": 0, "failed": 0, "ok": True},
            }
            data = {
                "session_id": session,
                "experiment": config,
                "repeat": repeat,
                "size": "e4b",
                "clip_id": "clip",
                "language": "en",
                "source_cohort_sha256": "frozen",
                "returncode": 0,
                "completion_validation": {"valid": True},
                "requested_experiment_configuration": config_fields,
                "clip": {"sha256": "audio"},
                "session_metadata": {
                    "session_id": session,
                    "source_lang": "en",
                    "input_audio_sha256": "audio",
                    "stt_backend": "parakeet-mlx",
                    "profile": {"name": "standard"},
                    "latency_experiment_configuration": config_fields,
                },
                "session_lifecycle": lifecycle,
                "observed": {"finals": [row], "partials": partials, "session_summary": summary},
            }
            if changes:
                changes(config, repeat, data, row, partials, acks, diag)
            final_path = metrics / f"ab_metrics_{session}.csv"
            with final_path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(row))
                writer.writeheader()
                writer.writerow(row)
            for prefix, records in (("partials", partials), ("diagnostics", diag), ("display_metrics", acks)):
                (metrics / f"{prefix}_{session}.jsonl").write_text(
                    "\n".join(json.dumps(value) for value in records) + "\n"
                )
            diagnostic_path = metrics / f"diagnostics_{session}.jsonl"
            lifecycle["diagnostics"] = {
                "sha256": analysis.sha(diagnostic_path),
                "size_bytes": diagnostic_path.stat().st_size,
            }
            path = inputs / f"{session}.json"
            path.write_text(json.dumps(data))
            paths[(repeat, config)] = path
    return inputs, metrics, paths


def test_complete_candidate_is_only_worth_confirming_and_signed_gains_survive(tmp_path):
    inputs, metrics, _ = fixture_matrix(tmp_path)
    report = analysis.analyze(inputs, metrics)
    assert report["matrix_status"] == "complete" and report["valid_runs"] == 9
    arm = report["arms"][0]
    assert arm["status"] == "worth_confirming"
    assert not arm["production_promotion"] and not arm["p95_promotion_evidence"]
    final = arm["pooled_matched_metrics"]["versus_opening"]["final:silence|replay_realtime"]
    assert final["paired_delta_ms"]["p50"] == -300 and final["before"]["n"] == 3
    assert (
        arm["pooled_matched_metrics"]["opening_to_closing_drift"]["final:silence|replay_realtime"]["paired_delta_ms"][
            "p50"
        ]
        == 0
    )
    session = report["sessions"][0]
    assert session["previews"]["total_decoded_audio_seconds"] is None
    assert session["previews"]["emitted_preview_processed_audio_seconds_proxy"] == 1
    assert session["trace"]["discarded_old_events"] == 3
    assert session["counters"]["partial_emitted"] == 2  # one latest summary, not doubled
    assert session["browsers"]["browser"]["first_preview_coverage"] == 1


def test_missing_anchor_is_unassessable_not_zero_gain_or_zero_loss(tmp_path):
    inputs, metrics, _ = fixture_matrix(tmp_path, omit={(1, "baseline_anchor")})
    report = analysis.analyze(inputs, metrics)
    assert report["matrix_status"] == "incomplete" and len(report["missing_sessions"]) == 1
    arm = report["arms"][0]
    assert arm["status"] == "not_selected"
    pair = arm["comparisons"][1]["versus_closing"]
    assert pair["status"] == "unassessable" and "missing_source_s" not in pair


def test_source_cohorts_never_pool_or_gain_comparisons(tmp_path):
    def change(config, repeat, data, *_):
        if config == "candidate":
            data["source_cohort_sha256"] = "different frozen source"

    inputs, metrics, _ = fixture_matrix(tmp_path, changes=change)
    report = analysis.analyze(inputs, metrics)
    assert report["arms"][0]["comparisons"][0]["versus_opening"]["status"] == "unassessable"
    assert report["arms"][0]["status"] == "not_selected"


def test_closing_anchor_prevents_credit_for_system_wide_speedup(tmp_path):
    def change(config, repeat, data, row, partials, acks, diag):
        if config == "baseline_anchor":
            row["speech_end_to_final_ms"] = "600"
            diag[0]["timing_stages_ms"]["final_ready"] = 1800
            for partial in partials:
                partial["speech_start_to_partial_ms"] -= 300
                partial["emitted_at_ms"] -= 300
            for ack in acks:
                if ack["stage"] == "partial":
                    ack["speech_start_to_preview_ack_upper_bound_ms"] -= 300
                else:
                    ack["speech_end_to_ack_upper_bound_ms"] = 610

    inputs, metrics, _ = fixture_matrix(tmp_path, changes=change)
    report = analysis.analyze(inputs, metrics)
    arm = report["arms"][0]
    assert arm["status"] == "not_selected"
    assert (
        arm["pooled_matched_metrics"]["opening_to_closing_drift"]["final:silence|replay_realtime"]["paired_delta_ms"][
            "p50"
        ]
        == -400
    )


def test_changed_final_bound_cannot_game_latency_and_changed_text_is_preserved(tmp_path):
    def change(config, repeat, data, row, *_):
        if config == "candidate" and repeat == 1:
            row["sample_end"] = "8000"
            row["speech_end_sample"] = "8000"
        elif config == "candidate":
            row["spanish_a"] = "Output requiring actual review"

    inputs, metrics, _ = fixture_matrix(tmp_path, changes=change)
    report = analysis.analyze(inputs, metrics)
    arm = report["arms"][0]
    assert arm["status"] == "not_selected"
    assert arm["comparisons"][1]["versus_opening"]["missing_source_s"] == 0.5
    changed = arm["comparisons"][0]["versus_opening"]["changed_matched_outputs"][0]
    assert changed["after"]["spanish_a"] == "Output requiring actual review"


def test_missing_first_preview_cannot_disappear_from_fast_median(tmp_path):
    def change(config, repeat, data, row, partials, acks, diag):
        if config == "candidate":
            acks[:] = [ack for ack in acks if ack["stage"] == "complete"]

    inputs, metrics, _ = fixture_matrix(tmp_path, changes=change)
    report = analysis.analyze(inputs, metrics)
    candidate = next(row for row in report["sessions"] if row["experiment"] == "candidate")
    assert candidate["browsers"]["browser"]["missing_first_preview_utterances"] == ["1"]
    assert candidate["browsers"]["browser"]["first_preview_ack_ms"]["p50"] is None
    assert report["arms"][0]["status"] == "not_selected"


def test_missing_intermediate_ack_does_not_equal_missing_first_preview(tmp_path):
    def change(config, repeat, data, row, partials, acks, diag):
        if config == "candidate":
            acks[:] = [ack for ack in acks if ack.get("preview_revision") != 2]

    inputs, metrics, _ = fixture_matrix(tmp_path, changes=change)
    report = analysis.analyze(inputs, metrics)
    assert report["arms"][0]["status"] == "worth_confirming"
    candidate = next(row for row in report["sessions"] if row["experiment"] == "candidate")
    assert len(candidate["browsers"]["browser"]["missing_preview_events"]) == 1


def test_duplicate_hidden_stale_and_wrong_revision_acks_do_not_count(tmp_path):
    def change(config, repeat, data, row, partials, acks, diag):
        acks.extend([dict(acks[0]), {**acks[0], "session_id": "other"}, {**acks[1], "visible": False}])
        if config == "candidate":
            acks[0]["preview_revision"] = 999
            acks[1]["visible"] = False
            acks[:] = acks[:3]

    inputs, metrics, _ = fixture_matrix(tmp_path, changes=change)
    report = analysis.analyze(inputs, metrics)
    baseline = next(row for row in report["sessions"] if row["experiment"] == "baseline")
    assert baseline["browsers"]["browser"]["visible_preview_events"] == 2
    candidate = next(row for row in report["sessions"] if row["experiment"] == "candidate")
    assert candidate["browsers"]["browser"]["visible_preview_events"] == 0
    assert report["arms"][0]["status"] == "not_selected"


def test_failed_arm_keeps_evidence_and_blocks_selection(tmp_path):
    def change(config, repeat, data, *_):
        if config == "candidate" and repeat == 2:
            data["returncode"] = 1
            data["error"] = "translation failed"

    inputs, metrics, _ = fixture_matrix(tmp_path, changes=change)
    report = analysis.analyze(inputs, metrics)
    assert report["recorded_runs"] == 9 and report["valid_runs"] == 8
    assert report["arms"][0]["status"] == "not_selected"
    assert any("translation failed" in row["validation_errors"] for row in report["sessions"])


def test_post_run_source_records_are_hash_checked(tmp_path):
    inputs, metrics, paths = fixture_matrix(tmp_path)
    session = json.loads(paths[(0, "candidate")].read_text())["session_id"]
    path = metrics / f"diagnostics_{session}.jsonl"
    path.write_text(path.read_text() + "{}\n")
    report = analysis.analyze(inputs, metrics)
    assert report["valid_runs"] == 8
    assert any("Diagnostic hash differs" in error for row in report["sessions"] for error in row["validation_errors"])


def test_unexercised_memo_does_not_get_selected_on_incidental_speedup(tmp_path):
    def change(config, repeat, data, *_):
        if config == "candidate":
            data["session_metadata"]["latency_experiment_configuration"]["marian_memo"] = 128

    inputs, metrics, _ = fixture_matrix(tmp_path, changes=change)
    report = analysis.analyze(inputs, metrics)
    assert report["arms"][0]["status"] == "not_selected"
    assert report["arms"][0]["execution_evidence"][0]["unexercised_flags"] == ["marian_memo"]


def test_percentiles_signed_deltas_and_lexical_append_semantics():
    assert analysis.distribution([2204, 2044]) == {"n": 2, "p50": 2124, "p95": 2204, "max": 2204}
    assert analysis.distribution([-200, 0, 50], signed=True)["n"] == 3
    assert analysis.distribution([None, True, float("nan")])["n"] == 0
    assert analysis.lexical_change("Dios ama", "Dios ama a todos")["retracted_suffix_tokens"] == 0
    assert analysis.lexical_change("Dios ama", "Dios nos ama")["retracted_suffix_tokens"] == 1


def test_repeated_summaries_use_only_latest_and_trace_stays_explicit(tmp_path):
    def change(config, repeat, data, row, partials, acks, diag):
        diag.insert(0, {"event": "session_summary", "latency_experiment_counters": {"partial_emitted": 999}})

    inputs, metrics, _ = fixture_matrix(tmp_path, changes=change)
    report = analysis.analyze(inputs, metrics)
    assert all(row["counters"]["partial_emitted"] == 2 for row in report["sessions"])


def test_multiple_browsers_require_explicit_selection(tmp_path):
    def change(config, repeat, data, row, partials, acks, diag):
        acks.extend([{**ack, "client_id": "second"} for ack in acks])

    inputs, metrics, _ = fixture_matrix(tmp_path, changes=change)
    assert analysis.analyze(inputs, metrics)["arms"][0]["status"] == "not_selected"
    selected = analysis.analyze(inputs, metrics, client_id="browser")["arms"][0]
    assert selected["status"] == "worth_confirming"
    assert not any(metric.endswith(":second") for metric in selected["target_metrics"])


def test_tail_regression_cannot_hide_behind_two_fast_repetitions(tmp_path):
    def change(config, repeat, data, row, *_):
        if config == "candidate" and repeat == 2:
            row["speech_end_to_final_ms"] = "1200"

    inputs, metrics, _ = fixture_matrix(tmp_path, changes=change)
    arm = analysis.analyze(inputs, metrics)["arms"][0]
    assert arm["status"] == "not_selected"
    assert any("tail/memory" in reason for reason in arm["reasons"])


def test_missing_final_time_is_invalid_even_with_fast_visible_previews(tmp_path):
    def change(config, repeat, data, row, *_):
        if config == "candidate":
            row["speech_end_to_final_ms"] = ""

    inputs, metrics, _ = fixture_matrix(tmp_path, changes=change)
    report = analysis.analyze(inputs, metrics)
    assert report["valid_runs"] == 6
    assert report["arms"][0]["status"] == "not_selected"


def test_markdown_has_all_endpoint_controls_without_unescaped_pipes(tmp_path):
    inputs, metrics, _ = fixture_matrix(tmp_path)
    text = analysis.markdown(analysis.analyze(inputs, metrics))
    assert "versus_opening" in text and "versus_closing" in text and "opening_to_closing_drift" in text
    assert "final:silence / replay_realtime" in text
    assert "final:silence|replay_realtime" not in text


def test_negative_and_out_of_bounds_speech_positions_are_not_matched():
    row = {"sample_start": 20, "sample_end": 30, "sample_rate": 10, "speech_end_sample": 10}
    assert analysis.bounds(row) is None


def test_slower_visible_finals_block_fast_server_only_gain(tmp_path):
    def change(config, repeat, data, row, partials, acks, diag):
        if config == "candidate":
            acks[-1]["speech_end_to_ack_upper_bound_ms"] = 1210

    inputs, metrics, _ = fixture_matrix(tmp_path, changes=change)
    arm = analysis.analyze(inputs, metrics)["arms"][0]
    assert arm["status"] == "not_selected"
    assert any("final_visible:" in reason for reason in arm["reasons"])


@pytest.mark.parametrize("final_delay,expected_after", [(910, 0), (710, 1), (809.9, 0)])
def test_server_late_preview_and_per_client_ack_order_are_distinct(tmp_path, final_delay, expected_after):
    def change(config, repeat, data, row, partials, acks, diag):
        diag[0]["timing_stages_ms"]["speech_end"] = 1200
        if config != "candidate":
            return
        partials[-1]["emitted_at_ms"] = 2000  # final readiness 1900
        partials[-1]["speech_start_to_partial_ms"] = 1000
        acks[1]["speech_start_to_preview_ack_upper_bound_ms"] = 1010  # receipt 2010
        acks[-1]["speech_end_to_ack_upper_bound_ms"] = final_delay
        acks.extend([{**ack, "client_id": "second"} for ack in acks])
        acks[-1]["speech_end_to_ack_upper_bound_ms"] = 1200  # other browser's final receipt 2400

    inputs, metrics, _ = fixture_matrix(tmp_path, changes=change)
    report = analysis.analyze(inputs, metrics, client_id="browser")
    row = next(item for item in report["sessions"] if item["experiment"] == "candidate")
    assert row["previews"]["utterances"]["1"]["preview_after_final_ready"]
    assert len(row["browsers"]["browser"]["preview_acks_after_final_ack"]) == expected_after
    assert row["browsers"]["second"]["preview_acks_after_final_ack"] == []
    assert row["browsers"]["browser"]["preview_ack_order_unassessable_events"] == []
    assert report["arms"][0]["status"] == "not_selected"  # no automatic gate relaxation
    assert any("browser-order review required" in reason for reason in report["arms"][0]["reasons"])
    assert "does not establish stale repaint" in analysis.markdown(report)


def test_missing_final_clock_is_unassessable_ack_order(tmp_path):
    inputs, metrics, _ = fixture_matrix(tmp_path)
    report = analysis.analyze(inputs, metrics)
    browser = report["sessions"][0]["browsers"]["browser"]
    assert browser["preview_acks_after_final_ack"] == []
    assert len(browser["preview_ack_order_unassessable_events"]) == 2


def test_preview_ack_after_final_requires_review_even_if_server_emitted_earlier(tmp_path):
    def change(config, repeat, data, row, partials, acks, diag):
        diag[0]["timing_stages_ms"]["speech_end"] = 1200
        if config == "candidate":
            acks[1]["speech_start_to_preview_ack_upper_bound_ms"] = 1010

    inputs, metrics, _ = fixture_matrix(tmp_path, changes=change)
    report = analysis.analyze(inputs, metrics)
    row = next(item for item in report["sessions"] if item["experiment"] == "candidate")
    assert not row["previews"]["utterances"]["1"]["preview_after_final_ready"]
    assert len(row["browsers"]["browser"]["preview_acks_after_final_ack"]) == 1
    assert report["arms"][0]["status"] == "not_selected"
    assert any("preview ACK after final ACK" in reason for reason in report["arms"][0]["reasons"])


def test_cli_refuses_report_inside_evidence_before_reading_or_writing(tmp_path, monkeypatch):
    inputs = tmp_path / "metrics" / "screen"
    output = inputs / "report"
    monkeypatch.setattr("sys.argv", ["overnight_analysis", "--input", str(inputs), "--output", str(output)])
    with pytest.raises(SystemExit) as error:
        analysis.main()
    assert error.value.code == 2 and not output.exists()


def test_source_only_event_does_not_count_as_first_translated_preview(tmp_path):
    def change(config, repeat, data, row, partials, acks, diag):
        if config == "candidate":
            partials[0]["text_es"] = ""

    inputs, metrics, _ = fixture_matrix(tmp_path, changes=change)
    report = analysis.analyze(inputs, metrics)
    candidate = next(row for row in report["sessions"] if row["experiment"] == "candidate")
    assert candidate["previews"]["first_by_utterance_ms"] == {"1": 700}
    assert candidate["browsers"]["browser"]["first_by_utterance_ms"] == {"1": 710}
    assert candidate["browsers"]["browser"]["visible_preview_events"] == 1
