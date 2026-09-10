import copy

import pytest

from tools.mac_followup_latency import configuration, score_pair


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
    }
    return {
        "source_identity": "fixed",
        "clip": {"sha256": "same"},
        "diagnostic_finals": [row],
        "observed": {"partials": [], "coverage_intervals_s": [(0, 8)]},
    }


def test_candidate_must_beat_both_controls():
    result = score_pair(run(10000), run(9200), run(9250))
    assert result["status"] == "rejected"
    assert "median_gain_below_gate" in result["reasons"]
    assert score_pair(run(10000), run(9200), run(10000))["status"] == "latency_candidate"


def test_incomplete_source_cannot_qualify_with_fast_finals():
    candidate = run(9000)
    candidate["completion_errors"] = ["Source accounting incomplete"]
    assert "failed_or_incomplete_run" in score_pair(run(10000), candidate, run(10000))["reasons"]


def test_candidate_cannot_hide_source_from_closing_control():
    closing = run(10000)
    closing["observed"]["coverage_intervals_s"] = [(0, 9)]
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
