import pytest

from tools.mac_followup_selection import build_spec, select


def report(*names):
    return {
        "inventory_complete": True,
        "inventory_errors": [],
        "pairs": [
            {
                "experiment": name,
                "size": "e2b",
                "clip_id": "en",
                "repeat": repeat,
                "status": "latency_candidate" if repeat < 2 else "rejected",
                "reasons": [] if repeat < 2 else ["median_gain_below_gate"],
                "opening": {"p50_ms": 1000},
                "closing": {"p50_ms": 1100},
                "candidate": {"p50_ms": 700},
                "p95_claim_eligible": False,
            }
            for name in names
            for repeat in range(3)
        ],
    }


def spec():
    return {
        "id": "screen",
        "clips": [{"id": "en"}, {"id": "es"}],
        "sizes": ["e2b", "e4b"],
        "experiments": [
            {"name": "baseline", "env": {}},
            {"name": "cadence", "partial_interval": 0.9, "env": {}},
            {"name": "deadline", "env": {"STARK_EXPERIMENT_PARTIAL_DEADLINE_MARGIN_MS": "100"}},
        ],
    }


def test_two_of_three_does_not_pool_small_p95_samples():
    result = select(report("cadence"))
    assert result["qualified_count"] == 1
    assert result["arms"][0]["passed_repeats"] == [0, 1]
    assert not result["arms"][0]["p95_claim_eligible"]
    assert not result["defaults_changed"]


@pytest.mark.parametrize("failure", ["missing", "duplicate", "inventory", "reason", "one_pass"])
def test_invalid_or_incomplete_pairs_never_qualify(failure):
    data = report("cadence")
    if failure == "missing":
        data["pairs"].pop()
    elif failure == "duplicate":
        data["pairs"][2]["repeat"] = 1
    elif failure == "inventory":
        data["inventory_errors"] = ["wrong source"]
    elif failure == "reason":
        data["pairs"][0]["reasons"] = ["failed_or_incomplete_run"]
    else:
        data["pairs"][0]["status"] = "rejected"
    assert select(data)["qualified_count"] == 0


def test_empty_selection_is_valid_and_never_invents_an_arm():
    result = select(report())
    assert result["inventory_complete"] and result["qualified_count"] == 0
    with pytest.raises(ValueError):
        build_spec(spec(), result, clip_id="en", size="e2b", names=["cadence"])


def test_combination_preserves_exact_scope_and_independent_settings():
    source = spec()
    result = build_spec(
        source,
        select(report("cadence", "deadline")),
        clip_id="en",
        size="e2b",
        names=["cadence", "deadline"],
        combine=True,
    )
    assert result["clips"] == [{"id": "en"}]
    assert result["sizes"] == ["e2b"]
    assert result["experiments"][1]["partial_interval"] == 0.9
    assert result["experiments"][1]["env"]["STARK_EXPERIMENT_PARTIAL_DEADLINE_MARGIN_MS"] == "100"
    assert len(source["experiments"]) == 3


@pytest.mark.parametrize("clip,size", [("es", "e2b"), ("en", "e4b")])
def test_success_cannot_be_transferred_to_another_scope(clip, size):
    with pytest.raises(ValueError):
        build_spec(spec(), select(report("cadence")), clip_id=clip, size=size, names=["cadence"])


def test_conflicting_candidates_cannot_be_combined():
    source = spec()
    source["experiments"][2]["partial_interval"] = 1.2
    with pytest.raises(ValueError, match="Conflicting"):
        build_spec(
            source,
            select(report("cadence", "deadline")),
            clip_id="en",
            size="e2b",
            names=["cadence", "deadline"],
            combine=True,
        )


@pytest.mark.parametrize(
    "reason",
    [
        "failed_or_incomplete_run",
        "production_caption_wer_regression",
        "tail_regression",
        "missing_fixed_source_span",
        "missing_median",
    ],
)
def test_third_repeat_must_pass_every_nonlatency_guard(reason):
    data = report("cadence")
    data["pairs"][2]["reasons"] = [reason]
    result = select(data)
    assert result["qualified_count"] == 0
    assert not result["arms"][0]["all_repeats_nonlatency_guards_passed"]


@pytest.mark.parametrize("status", ["invalid_comparison", "invalid_inventory", "unknown"])
def test_third_invalid_status_is_not_a_permitted_median_miss(status):
    data = report("cadence")
    data["pairs"][2]["status"] = status
    assert select(data)["qualified_count"] == 0


def test_median_miss_with_additional_guard_failure_rejects():
    data = report("cadence")
    data["pairs"][2]["reasons"].append("preview_source_coverage_loss")
    assert select(data)["qualified_count"] == 0


def test_cli_records_exact_input_report_hash(tmp_path, monkeypatch):
    import hashlib
    import json
    import sys

    from tools.mac_followup_selection import main

    source = tmp_path / "report.json"
    output = tmp_path / "selection.json"
    source.write_text(json.dumps(report("cadence")))
    monkeypatch.setattr(sys, "argv", ["selection", "--report", str(source), "--output", str(output)])
    main()
    result = json.loads(output.read_text())
    assert result["source_report"] == {
        "path": str(source.resolve()),
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "size_bytes": source.stat().st_size,
    }
    assert result["qualified_count"] == 1
