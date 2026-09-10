import copy
import json
from types import SimpleNamespace

import pytest

from tools.mac_followup_latency import configuration, report, score_pair


def run(ready):
    row = {
        "timing_schema_version": 2,
        "timing_source": "replay_realtime",
        "sample_rate": 1000,
        "sample_start": 0,
        "sample_end": 8000,
        "speech_end_sample": 7900,
        "padding_samples": 0,
        "endpoint_reason": "smart_cut",
        "timing_stages_ms": {"captured_start": 10000, "final_ready": 10000 + ready},
        "stt_queue_wait_ms": 0,
        "translation_queue_wait_ms": 0,
        "generation_lock_wait_ms_a": 0,
        "final_translation_route": "gemma",
    }
    return {
        "source_identity": "fixed",
        "clip": {"sha256": "same"},
        "diagnostic_finals": [row],
        "session_lifecycle": {"memory": {"peak_rss_bytes": 1000000000, "peak_metal_bytes": 2000000000}},
        "observed": {
            "partials": [],
            "coverage_intervals_s": [(0, 8)],
            "session_summary": {
                "final_queue_pressure": {
                    "bookkeeping_truncated": False,
                    "terminal_outstanding": 0,
                    "put_failed": 0,
                    "unmatched_dequeues": 0,
                    "max_pending": 1,
                    "max_wait_ms": 1,
                    "dequeued": 4,
                    "first_window_wait_ms": [1, 1],
                    "last_window_wait_ms": [1, 1],
                },
                "source_coverage": {
                    "observed": [
                        {"start": 0, "end": 7900, "vad_positive": True},
                        {"start": 7900, "end": 8000, "vad_positive": False},
                    ]
                },
            },
        },
    }


def test_candidate_must_beat_both_controls():
    result = score_pair(run(10000), run(9200), run(9250))
    assert result["status"] == "rejected"
    assert "median_gain_below_gate" in result["reasons"]
    assert score_pair(run(10000), run(9200), run(10000))["status"] == "latency_candidate"


@pytest.mark.parametrize("marian_arms", [(0, 1, 2), (1,), (0, 2)])
def test_marian_generation_lock_is_explicitly_not_applicable(marian_arms):
    runs = [run(10000), run(9000), run(10000)]
    for arm in marian_arms:
        runs[arm]["diagnostic_finals"][0].update(final_translation_route="marian", generation_lock_wait_ms_a=None)
    result = score_pair(*runs)
    assert result["status"] == "latency_candidate"
    locks = [r for r in result["queue_comparisons"] if r["field"] == "generation_lock_wait_ms_a"]
    assert len(locks) == 2
    for control_arm, comparison in zip((0, 2), locks):
        assert comparison["status"] == "not_applicable_no_gemma_in_one_or_both_runs"
        for prefix, arm in (("control", control_arm), ("candidate", 1)):
            marian = arm in marian_arms
            assert comparison[prefix + "_route_counts"] == {
                "gemma": int(not marian),
                "marian": int(marian),
                "unknown": 0,
            }
            assert comparison[prefix + "_sample_count"] == int(not marian)
            assert comparison[prefix + "_p95_ms"] == (None if marian else 0)


@pytest.mark.parametrize("value", [None, float("nan"), -1, True])
def test_gemma_generation_lock_still_requires_a_finite_measurement(value):
    candidate = run(9000)
    candidate["diagnostic_finals"][0]["generation_lock_wait_ms_a"] = value
    result = score_pair(run(10000), candidate, run(10000))
    assert result["status"] == "rejected"
    assert "missing_stage_queue_evidence" in result["reasons"]
    locks = [r for r in result["queue_comparisons"] if r["field"] == "generation_lock_wait_ms_a"]
    assert all(r["status"] == "invalid_evidence" and r["candidate_sample_count"] == 0 for r in locks)


def test_marian_zero_is_not_a_measured_generation_lock_wait():
    candidate = run(9000)
    candidate["diagnostic_finals"][0]["final_translation_route"] = "marian"
    assert "invalid_marian_generation_lock_evidence" in score_pair(run(10000), candidate, run(10000))["reasons"]


def test_missing_route_cannot_exempt_an_unknown_final_from_lock_checks():
    candidate = run(9000)
    del candidate["diagnostic_finals"][0]["final_translation_route"]
    assert "missing_or_invalid_final_translation_route" in score_pair(run(10000), candidate, run(10000))["reasons"]


def test_gemma_lock_tail_is_compared_after_excluding_marian_finals():
    runs = [run(10000), run(9000), run(10000)]
    for item in runs:
        first = item["diagnostic_finals"][0]
        first.update(sample_end=4000, speech_end_sample=3900, generation_lock_wait_ms_a=10)
        second = copy.deepcopy(first)
        second.update(sample_start=4000, sample_end=8000, speech_end_sample=7900)
        second["timing_stages_ms"]["captured_start"] += 4000
        second.update(final_translation_route="marian", generation_lock_wait_ms_a=None)
        item["diagnostic_finals"].append(second)
    runs[1]["diagnostic_finals"][0]["generation_lock_wait_ms_a"] = 150
    result = score_pair(*runs)
    assert result["status"] == "rejected"
    assert result["reasons"] == ["stage_queue_tail_regression"]
    locks = [r for r in result["queue_comparisons"] if r["field"] == "generation_lock_wait_ms_a"]
    assert all(r["status"] == "compared_gemma_calls" for r in locks)
    assert all(r["control_sample_count"] == r["candidate_sample_count"] == 1 for r in locks)
    assert all(r["control_p95_ms"] == 10 and r["candidate_p95_ms"] == 150 for r in locks)


def test_incomplete_source_cannot_qualify_with_fast_finals():
    candidate = run(9000)
    candidate["completion_errors"] = ["Source accounting incomplete"]
    assert "failed_or_incomplete_run" in score_pair(run(10000), candidate, run(10000))["reasons"]


@pytest.mark.parametrize("strict_control", [0, 2], ids=["opening", "closing"])
def test_disjoint_queue_growth_must_pass_both_controls(strict_control):
    runs = [run(10000), run(9000), run(10000)]
    for item in runs:
        pressure = item["observed"]["session_summary"]["final_queue_pressure"]
        # With only six dequeues the helper's retained first/last windows
        # overlap completely. Compare disjoint halves, not their whole means.
        pressure.update(
            dequeued=6,
            max_wait_ms=1000,
            first_window_wait_ms=[0, 0, 0, 1000, 1000, 1000],
            last_window_wait_ms=[0, 0, 0, 1000, 1000, 1000],
        )
    steady = runs[strict_control]["observed"]["session_summary"]["final_queue_pressure"]
    steady.update(
        first_window_wait_ms=[1000, 500, 500, 500, 500, 500],
        last_window_wait_ms=[1000, 500, 500, 500, 500, 500],
    )
    result = score_pair(*runs)
    assert result["status"] == "rejected"
    assert result["reasons"] == ["final_queue_sustained_growth"]
    trends = [r for r in result["queue_comparisons"] if r["field"] == "disjoint_queue_wait_growth_ms"]
    assert len(trends) == 2
    assert all(r["candidate"] == 1000 for r in trends)
    assert sorted(r["control"] for r in trends) == pytest.approx([-500 / 3, 1000])


def test_candidate_cannot_hide_source_from_closing_control():
    closing = run(10000)
    closing["observed"]["coverage_intervals_s"] = [(0, 9)]
    closing["diagnostic_finals"][0].update(sample_end=9000, speech_end_sample=8900)
    closing["observed"]["session_summary"]["source_coverage"]["observed"] = [
        {"start": 0, "end": 8900, "vad_positive": True}
    ]
    assert "final_source_coverage_loss" in score_pair(run(10000), run(9000), closing)["reasons"]


def test_mixed_runtime_or_audio_rejected():
    candidate = copy.deepcopy(run(9000))
    candidate["source_identity"] = "changed"
    with pytest.raises(ValueError, match="identities"):
        score_pair(run(10000), candidate, run(10000))


def test_lite_and_spanish_controls_are_explicit(monkeypatch):
    monkeypatch.setenv("STARK_PROFILE", "lite-cuda-8gb")
    args, env, _ = configuration({"profile": "lite-cpu"}, {"partial_interval": 0.9}, {"lang": "es"}, "e2b")
    assert args[args.index("--profile") + 1] == "lite-cpu"
    assert args[args.index("--stt-backend") + 1] == "faster-whisper"
    assert "STARK_PROFILE" not in env
    args, _, _ = configuration({}, {}, {"lang": "es"}, "e4b")
    assert args[args.index("--stt-backend") + 1] == "mlx"


def test_unknown_experiment_setting_rejected_before_launch():
    with pytest.raises(ValueError, match="explicit"):
        configuration({}, {"env": {"STARK_EXPERIMENT_TYPO": "1"}}, {"lang": "en"}, "e4b")


def test_report_retains_missing_candidates_and_rejects_duplicate_controls(tmp_path):
    spec = {
        "experiments": [{"name": "baseline"}, {"name": "fast"}],
        "clips": [{"id": "clip", "lang": "en"}],
        "sizes": ["e4b"],
    }
    (tmp_path / "provenance.json").write_text(
        json.dumps({"spec": spec, "repeats": 1, "tag": "test", "source_identity": "fixed"})
    )
    baseline = {
        **run(10000),
        "experiment": "baseline",
        "repeat": 0,
        "size": "e4b",
        "clip_id": "clip",
        "session_id": "test_baseline_e4b_r0_clip_en",
    }
    baseline["observed"]["final_count"] = 1
    for name in ("original", "duplicate"):
        (tmp_path / f"{name}.json").write_text(json.dumps(baseline))
    output = tmp_path / "report.json"
    report(SimpleNamespace(input=tmp_path, output=output))
    result = json.loads(output.read_text())
    assert not result["inventory_complete"]
    assert result["expected_runs"] == 3
    assert any("found 2" in error for error in result["inventory_errors"])
    assert len(result["pairs"]) == 1
    assert result["pairs"][0]["status"] == "invalid_comparison"


def test_report_rejects_shared_wrong_audio_against_frozen_provenance(tmp_path):
    expected_hash, wrong_hash = "a" * 64, "b" * 64
    spec = {
        "experiments": [{"name": "baseline"}, {"name": "fast"}],
        "clips": [{"id": "clip", "lang": "en", "sha256": expected_hash}],
        "sizes": ["e4b"],
    }
    (tmp_path / "provenance.json").write_text(
        json.dumps({"spec": spec, "repeats": 1, "tag": "test", "source_identity": "fixed"})
    )
    records = []
    for name, ready in (("baseline", 10000), ("fast", 9000), ("baseline_anchor", 10000)):
        item = {
            **run(ready),
            "experiment": name,
            "repeat": 0,
            "size": "e4b",
            "clip_id": "clip",
            "session_id": f"test_{name}_e4b_r0_clip_en",
            "clip": {"sha256": wrong_hash},
        }
        item["observed"]["final_count"] = 1
        records.append(item)
        (tmp_path / f"{name}.json").write_text(json.dumps(item))
    # Pairwise equality alone accepts the three matching, but wrong, inputs.
    assert score_pair(*records)["status"] == "latency_candidate"
    output = tmp_path / "report.json"
    report(SimpleNamespace(input=tmp_path, output=output))
    result = json.loads(output.read_text())
    assert result["inventory_complete"] is False
    assert len(result["inventory_errors"]) == 3
    assert all("Run identity mismatch" in error for error in result["inventory_errors"])
    assert result["pairs"][0]["status"] == "invalid_inventory"
    assert not any(pair["status"] == "latency_candidate" for pair in result["pairs"])
