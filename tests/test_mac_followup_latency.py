import copy
import json
from types import SimpleNamespace

import pytest

from tools.mac_followup_latency import (
    caption_quality,
    configuration,
    preview_comparisons,
    reference_contract,
    report,
    runtime_contract,
    runtime_errors,
    score_pair,
)


def runtime_fixture():
    # Actual pilot field names; no model import/load or filesystem resolution.
    stt = dict(
        requested_id="mlx-community/parakeet-tdt-0.6b-v3",
        resolved_path="/models/parakeet",
        resolved_revision="a" * 40,
        config_sha256="b" * 64,
    )
    gemma = dict(
        requested_id="mlx-community/gemma-4-e4b-it-OptiQ-4bit",
        resolved_path="/models/gemma",
        resolved_revision="c" * 40,
        config_sha256="d" * 64,
    )
    contract = dict(
        schema_version=1,
        profile={"name": "standard", "backend": "mlx"},
        stt_backend="parakeet-mlx",
        stt=stt,
        translation_a=gemma,
        source_lang="en",
        pipeline_gain=1.0,
        vad={"backend": "torch", "partial_interval": 0.6},
        translation={"routing_policy": "legacy", "terminology_prompt": "none"},
    )
    metadata = dict(
        profile=contract["profile"],
        stt_backend=contract["stt_backend"],
        source_lang="en",
        backend="mlx",
        model_a=gemma["requested_id"],
        vad={**contract["vad"], "artifact": {"sha256": "e" * 64}},
        translation=contract["translation"],
        stt_settings={"backend": "parakeet-mlx", "whisper_model": "mlx-community/whisper-large-v3-turbo"},
    )
    models = dict(
        stt=copy.deepcopy(stt),
        translation_a=copy.deepcopy(gemma),
        marian={
            "resolved_path": "/models/marian",
            "config_sha256": "f" * 64,
            "model_bin_sha256": "1" * 64,
            "export_manifest": {"model_bin_hash_matches": True},
        },
    )
    return copy.deepcopy(contract), copy.deepcopy(metadata), models


def run(ready):
    contract, metadata, models = runtime_fixture()
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
        "mic_gain": 1.0,
        "source_lang": "en",
        "target_lang": "es",
        "english": "Hello",
        "spanish_gemma": "Hola",
    }
    return {
        "source_identity": "fixed",
        "clip": {"sha256": "same"},
        "diagnostic_finals": [row],
        "requested_runtime": contract,
        "reference_contract": {
            "status": "available",
            "clip_sha256": "same",
            "source_lang": "en",
            "reference_ids": ["source"],
            "source_reference": "Hello",
            "translation_reference": "Hola",
            "terms": [],
            "term_scope": "Unique reference types",
        },
        "session_metadata": metadata,
        "session_lifecycle": {
            "models": models,
            "memory": {"peak_rss_bytes": 1000000000, "peak_metal_bytes": 2000000000},
        },
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


def previews(*, delay=1500, gap=1000):
    return [
        dict(
            sample_start=0,
            sample_end=end,
            sample_rate=1000,
            utterance_id=7,
            target_lang="es",
            text_es="Vista previa",
            captured_start_at_ms=10000,
            captured_end_at_ms=10000 + end,
            emitted_at_ms=10000 + delay + i * gap,
            speech_start_to_partial_ms=delay + i * gap,
        )
        for i, end in enumerate((1000, 2000))
    ]


@pytest.mark.parametrize("strict_control", [0, 2])
def test_preview_first_delay_and_gap_must_pass_both_controls(strict_control):
    runs = [run(10000), run(9000), run(10000)]
    for item in runs:
        item["observed"]["partials"] = previews(delay=11500, gap=10000)
    runs[strict_control]["observed"]["partials"] = previews()
    result = score_pair(*runs)
    assert result["status"] == "rejected"
    assert {
        "first_preview_tail_regression",
        "update_gap_tail_regression",
        "matched_first_preview_tail_regression",
        "matched_update_gap_tail_regression",
    } <= set(result["reasons"])
    evidence = result["preview_responsiveness"][0][0]
    assert evidence["control"]["n"] == evidence["candidate"]["n"] == 1
    assert evidence["p95_claim_eligible"] is False


def test_preview_zero_baseline_is_explicitly_not_applicable():
    control, candidate = run(10000), run(9000)
    candidate["observed"]["partials"] = previews()
    evidence, errors = preview_comparisons(control, candidate)
    assert not errors
    assert all(row["status"] == "not_applicable_control_has_no_samples" for row in evidence)
    assert all(row["control"] == {"n": 0, "p50": None, "p95": None} for row in evidence)


def test_actual_pilot_preview_duration_rounding_is_supported():
    item = run(10000)
    row = previews()[0]
    row.update(
        sample_start=1536,
        sample_end=89088,
        sample_rate=48000,
        captured_start_at_ms=10484.017,
        captured_end_at_ms=12308.017,
        emitted_at_ms=12546.673,
        speech_start_to_partial_ms=2062.7,
    )
    item["observed"]["partials"] = [row]
    comparisons, errors = preview_comparisons(item, item)
    assert not errors
    assert comparisons[0]["control"] == {"n": 1, "p50": 2062.7, "p95": 2062.7}


def test_missing_preview_timing_and_update_samples_fail_closed():
    control, candidate = run(10000), run(9000)
    control["observed"]["partials"] = previews()
    candidate["observed"]["partials"] = previews()[:1]
    assert "missing_preview_responsiveness" in preview_comparisons(control, candidate)[1]
    candidate["observed"]["partials"][0].pop("captured_end_at_ms")
    result = score_pair(control, candidate, control)
    assert "missing_preview_responsiveness" in result["reasons"]


def test_exact_source_preview_tail_cannot_hide_in_pooled_improvement():
    control, candidate = run(10000), run(9000)
    control["observed"]["partials"] = previews()
    candidate["observed"]["partials"] = previews(delay=1800)
    # A separate slow control utterance makes pooled p95 look favorable.
    extra = {**previews(delay=15000)[0], "utterance_id": 8, "sample_start": 8000, "sample_end": 9000}
    control["observed"]["partials"].append(extra)
    evidence, errors = preview_comparisons(control, candidate)
    assert evidence[0]["candidate"]["p95"] < evidence[0]["control"]["p95"]
    assert "matched_first_preview_tail_regression" in errors


@pytest.mark.parametrize("change", ["fallback", "backend", "gemma", "vad", "cadence"])
def test_actual_loader_and_configuration_must_match_frozen_intent(change):
    item = run(9000)
    metadata, models = item["session_metadata"], item["session_lifecycle"]["models"]
    if change == "fallback":
        models["stt"].update(requested_id="/models/distil", resolved_path="/models/distil")
    elif change == "backend":
        metadata["stt_backend"] = "mlx"
    elif change == "gemma":
        metadata["model_a"] = "wrong"
    elif change == "vad":
        metadata["vad"]["backend"] = "onnx"
    else:
        metadata["vad"]["partial_interval"] = 1.2
    assert runtime_errors(item, item["requested_runtime"])
    assert "missing_or_mismatched_runtime_identity" in score_pair(run(10000), item, run(10000))["reasons"]


def test_resolved_mlx_string_primary_is_valid_but_missing_intent_is_not():
    item = run(9000)
    item["session_lifecycle"]["models"]["stt"]["requested_id"] = item["requested_runtime"]["stt"]["resolved_path"]
    assert runtime_errors(item, item["requested_runtime"]) == []
    assert runtime_errors(item, None) == ["Missing frozen runtime intent"]


@pytest.mark.parametrize("field", ["marian", "vad", "translation"])
def test_unchanged_artifact_and_prompt_settings_must_match_controls(field):
    candidate = run(9000)
    if field == "marian":
        candidate["session_lifecycle"]["models"]["marian"]["model_bin_sha256"] = "2" * 64
    elif field == "vad":
        candidate["session_metadata"]["vad"]["artifact"]["sha256"] = "3" * 64
    else:
        candidate["session_metadata"]["translation"]["marian_intra_threads"] = 8
    result = score_pair(run(10000), candidate, run(10000))
    assert "unchanged_runtime_artifact_or_settings_mismatch" in result["reasons"]


def test_intentional_spanish_backend_override_gets_its_own_primary(monkeypatch):
    from tools import mac_followup_latency as tool

    monkeypatch.setattr(tool, "_model_binding", lambda model_id, **kwargs: {"requested_id": model_id})
    baseline = runtime_contract({}, {}, {"lang": "es"}, "e4b")
    candidate = runtime_contract({}, {"stt_backend": "parakeet-mlx"}, {"lang": "es"}, "e4b")
    assert baseline["stt_backend"] == "mlx"
    assert baseline["stt"]["requested_id"] == "mlx-community/whisper-large-v3-turbo"
    assert candidate["stt_backend"] == "parakeet-mlx"
    assert candidate["stt"]["requested_id"] == "mlx-community/parakeet-tdt-0.6b-v3"
    assert candidate["translation_a"] == baseline["translation_a"]


def test_declared_stt_override_and_cadence_remain_comparable():
    runs = [run(10000), run(9000), run(10000)]
    candidate = runs[1]
    candidate["requested_runtime"]["stt_backend"] = candidate["session_metadata"]["stt_backend"] = "mlx"
    candidate["session_metadata"]["stt_settings"]["backend"] = "mlx"
    candidate["requested_runtime"]["vad"]["partial_interval"] = candidate["session_metadata"]["vad"][
        "partial_interval"
    ] = 0.9
    assert score_pair(*runs)["status"] == "latency_candidate"


def test_derived_normalized_audio_must_keep_pipeline_gain_one():
    candidate = run(9000)
    candidate["diagnostic_finals"][0]["mic_gain"] = 10.0
    assert "missing_or_mismatched_runtime_identity" in score_pair(run(10000), candidate, run(10000))["reasons"]


def test_production_caption_quality_joins_source_order_after_segmentation():
    item = run(9000)
    contract = {
        **item["reference_contract"],
        "source_reference": "Jesus Christ is Lord.",
        "translation_reference": "Jesucristo es Señor.",
        "terms": ["Jesus Christ", "Lord"],
    }
    first = {
        **item["diagnostic_finals"][0],
        "sample_start": 0,
        "sample_end": 4000,
        "english": "Jesus Christ",
        "spanish_gemma": "Jesucristo",
    }
    second = {**first, "sample_start": 4000, "sample_end": 8000, "english": "is Lord.", "spanish_gemma": "es Señor."}
    item["diagnostic_finals"] = [second, first]
    result = caption_quality(item, contract)
    assert result["wer"] == 0 and result["chrf"] == 100
    assert result["source_hypothesis"] == "Jesus Christ is Lord."
    assert result["term_opportunities"] == 2 and result["term_recall"] == 1
    assert not result["human_approved_locally"] and not result["training_eligible"]


@pytest.mark.parametrize("strict_control", [0, 2])
def test_caption_wer_and_term_recall_must_pass_both_controls(strict_control):
    runs = [run(10000), run(9000), run(10000)]
    for item in runs:
        item["reference_contract"].update(source_reference="Jesus Christ is Lord.", terms=["Jesus Christ"])
        item["diagnostic_finals"][0]["english"] = "He is Lord."
    runs[strict_control]["diagnostic_finals"][0]["english"] = "Jesus Christ is Lord."
    result = score_pair(*runs)
    assert {"production_caption_wer_regression", "production_caption_term_recall_regression"} <= set(result["reasons"])
    assert result["caption_reference_quality"]["candidate"]["term_opportunities"] == 1


def test_chrf_is_descriptive_and_zero_terms_are_not_zero_recall():
    runs = [run(10000), run(9000), run(10000)]
    runs[1]["diagnostic_finals"][0]["spanish_gemma"] = "Otra expresión"
    result = score_pair(*runs)
    assert result["status"] == "latency_candidate"
    quality = result["caption_reference_quality"]
    assert quality["candidate"]["term_recall"] is None
    assert quality["candidate"]["term_opportunities"] == 0
    assert quality["chrf_delta_vs_opening"] < 0
    assert "descriptive" in quality["chrf_gate"]


def test_missing_historical_references_are_unavailable_not_zero_quality():
    contract = reference_contract({"sha256": "same", "lang": "en"})
    assert contract["status"] == "unavailable"
    candidate = run(9000)
    candidate["reference_contract"] = contract
    result = score_pair(run(10000), candidate, run(10000))
    assert "caption_reference_quality_unavailable" in result["reasons"]
    assert result["caption_reference_quality"]["candidate"]["wer"] is None


@pytest.mark.parametrize("fault", ["missing", "empty", "duplicate", "backward"])
def test_malformed_public_references_cannot_qualify(fault):
    clip = dict(
        sha256="same",
        lang="es",
        sample_rate=16000,
        provenance="fleurs_development_read_speech_concatenation",
        spans=[
            {
                "id": "one",
                "sample_start": 0,
                "sample_end": 16000,
                "reference": "Hola.",
                "translation_reference": "Hello.",
            }
        ],
    )
    if fault == "missing":
        clip.pop("spans")
    elif fault == "empty":
        clip["spans"][0]["translation_reference"] = "  "
    elif fault == "duplicate":
        clip["spans"].append(copy.deepcopy(clip["spans"][0]))
    else:
        clip["spans"][0]["sample_start"] = 17000
    assert reference_contract(clip)["status"] == "invalid"


def test_reference_terms_are_language_specific_unique_types_and_hash_bound(tmp_path, monkeypatch):
    from tools import mac_followup_latency as tool

    path = tmp_path / "bible_data/glossary/tier2_master.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"Jesus Christ": "Jesucristo", "Lord": "Señor"}))
    monkeypatch.setattr(tool, "ROOT", tmp_path)
    clip = dict(
        sha256="same",
        lang="es",
        sample_rate=16000,
        spans=[
            {
                "id": "one",
                "sample_start": 0,
                "sample_end": 16000,
                "reference": "Jesucristo, Señor.",
                "translation_reference": "Jesus Christ, Lord.",
            },
            {
                "id": "two",
                "sample_start": 32000,
                "sample_end": 48000,
                "reference": "Jesucristo.",
                "translation_reference": "Jesus Christ.",
            },
        ],
    )
    contract = reference_contract(clip)
    assert contract["status"] == "available"
    assert contract["terms"] == ["Jesucristo", "Señor"]
    assert contract["reference_ids"] == ["one", "two"]
    assert len(contract["glossary_sha256"]) == 64


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
        item["reference_contract"]["clip_sha256"] = wrong_hash
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
