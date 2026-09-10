import json
from types import SimpleNamespace

import pytest

from tools import mac_evaluation as evaluation


def test_unreviewed_predictions_never_count_as_references(tmp_path):
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"immutable test audio")
    row = {
        "id": "test",
        "path": "clip.wav",
        "sha256": evaluation.digest(audio),
        "lang": "es",
        "draft_transcript": "Predicted speech",
        "reference_text": "Reviewed speech",
        "reference_status": "pending",
        "provenance": "natural_speech",
    }
    manifest = {"usage": "evaluation_only", "utterances": [row]}
    assert evaluation.validate_manifest(manifest, tmp_path)["approved_natural_utterances"]["es"] == 0
    row["reference_status"] = "approved"
    assert evaluation.validate_manifest(manifest, tmp_path)["approved_natural_utterances"]["es"] == 1
    row["provenance"] = "synthetic_piper"
    assert evaluation.validate_manifest(manifest, tmp_path)["approved_natural_utterances"]["es"] == 0
    audio.write_bytes(b"changed audio")
    with pytest.raises(ValueError, match="Audio changed"):
        evaluation.validate_manifest(manifest, tmp_path)


def test_annotations_preserve_original_manifest_and_audio_identity():
    original = {"utterances": [{"id": "test", "sha256": "frozen", "reference_text": None}]}
    updated = evaluation.apply_annotations(
        original, [{"id": "test", "reference_text": "Grace", "reference_status": "approved"}]
    )
    assert original["utterances"][0]["reference_text"] is None
    assert updated["utterances"][0]["sha256"] == "frozen"
    with pytest.raises(ValueError, match="cannot change"):
        evaluation.apply_annotations(original, [{"id": "test", "sha256": "changed"}])
    with pytest.raises(ValueError, match="need a transcript"):
        evaluation.apply_annotations(original, [{"id": "test", "reference_status": "approved"}])


def test_paired_schedule_alternates_model_order_for_each_clip():
    clips = [{"id": "en"}, {"id": "es"}]
    schedule = evaluation.replay_schedule(clips, ["e4b", "e2b"], 3)
    assert len(schedule) == 12
    assert [size for repeat, clip, size in schedule if clip["id"] == "en"] == ["e4b", "e2b", "e2b", "e4b", "e4b", "e2b"]
    with pytest.raises(ValueError):
        evaluation.replay_schedule(clips, ["e4b"], 0)


def test_report_separates_legacy_synthetic_and_endpoint_metrics(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluation, "ROOT", tmp_path)
    manifest = tmp_path / "manifest.json"
    frozen = {"usage": "evaluation_only", "utterances": [], "replays": []}
    inputs = tmp_path / "inputs"
    metrics = tmp_path / "metrics"
    inputs.mkdir()
    metrics.mkdir()
    for i, (schema, provenance, endpoint, delay) in enumerate(
        [
            ("2", "church_replay", "silence", "600"),
            ("2", "synthetic_piper", "silence", "100"),
            ("2", "church_replay", "hard_cut", "2500"),
            ("", "church_replay", "", ""),
        ]
    ):
        audio = tmp_path / f"clip{i}.wav"
        audio.write_bytes(f"audio{i}".encode())
        clip = {
            "id": f"clip{i}",
            "lang": "en",
            "path": audio.name,
            "sha256": evaluation.digest(audio),
            "provenance": provenance,
        }
        frozen["replays"].append(clip)
        evaluation.write_json(
            inputs / f"run{i}.json",
            {
                "experiment": "baseline",
                "size": "e4b",
                "language": "en",
                "provenance": provenance,
                "session_id": str(i),
                "returncode": 0,
                "replay_speed": 1,
                "clip": clip,
                "command": [
                    "python",
                    "dry_run_ab.py",
                    "--audio-file",
                    str(audio),
                    "--session-id",
                    str(i),
                    "--lang",
                    "en",
                    "--gemma4-size",
                    "e4b",
                ],
            },
        )
        (metrics / f"ab_metrics_{i}.csv").write_text(
            "timing_schema_version,finalization_reason,speech_end_to_final_ms,e2e_latency_ms\n"
            f"{schema},{endpoint},{delay},300\n"
        )
    evaluation.write_json(manifest, frozen)
    result = evaluation.report_results(inputs, tmp_path / "report", manifest)
    assert len(result["replays"]) == 4
    legacy = next(r for r in result["replays"] if r["schema"] == "legacy")
    assert legacy["metrics"]["speech_end_to_final_ms"]["p50"] is None
    assert json.loads((tmp_path / "report/comparison.json").read_text())["human_review"] == "pending"


@pytest.fixture
def replay_case(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluation, "ROOT", tmp_path)
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"frozen")
    clip = {
        "id": "clip",
        "path": audio.name,
        "sha256": evaluation.digest(audio),
        "lang": "en",
        "provenance": "church_replay",
    }
    manifest = tmp_path / "manifest.json"
    evaluation.write_json(
        manifest, {"usage": "evaluation_only", "replays": [clip], "utterances": [], "translations": []}
    )
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    metrics = tmp_path / "metrics"
    metrics.mkdir()
    run = {
        "experiment": "baseline",
        "size": "e4b",
        "language": "en",
        "provenance": "church_replay",
        "session_id": "session",
        "clip_id": "clip",
        "clip": clip,
        "returncode": 0,
        "replay_speed": 1,
        "manifest_sha256": "older-reference-only-manifest",
        "environment": {"source_sha256": {"dry_run_ab.py": "original"}},
        "command": [
            "python",
            "dry_run_ab.py",
            "--audio-file",
            str(audio),
            "--session-id",
            "session",
            "--lang",
            "en",
            "--gemma4-size",
            "e4b",
        ],
    }
    evaluation.write_json(inputs / "replay.json", run)
    (metrics / "ab_metrics_session.csv").write_text(
        "timing_schema_version,endpoint_reason,speech_end_to_final_ms\n2,silence,600\n"
    )
    return manifest, inputs, metrics, run


def test_report_reuses_identical_audio_but_excludes_changed_manifest_inputs(replay_case, tmp_path):
    manifest, inputs, _, run = replay_case
    bad = {**run, "session_id": "different", "clip": {**run["clip"], "sha256": "changed"}}
    evaluation.write_json(inputs / "bad.json", bad)
    evaluation.write_json(inputs / "stt_wrong.json", {"completed": True, "manifest_sha256": "wrong", "rows": []})
    result = evaluation.report_results(inputs, tmp_path / "report", manifest)
    assert result["replays"][0]["metrics"]["speech_end_to_final_ms"]["n"] == 1
    assert result["stt"] == [] and len(result["excluded_runs"]) == 2


def test_report_sorts_partials_and_deduplicates_only_valid_v2_visible_acks(replay_case, tmp_path):
    manifest, inputs, metrics, _ = replay_case
    partials = [
        {"utterance_id": 1, "emitted_at_ms": 300, "speech_start_to_partial_ms": 300},
        {"utterance_id": 1, "emitted_at_ms": 100, "speech_start_to_partial_ms": 100},
        {"utterance_id": 2, "emitted_at_ms": 200, "speech_start_to_partial_ms": 150},
        {"speech_start_to_partial_ms": 999, "captured_end_to_partial_ms": -1},
    ]
    (metrics / "partials_session.jsonl").write_text(
        "".join(json.dumps({"timing_schema_version": 2, **r}) + "\n" for r in partials)
    )
    ack = {
        "event": "caption_rendered",
        "event_id": "a",
        "client_id": "c",
        "stage": "complete",
        "visible": True,
        "timing_schema_version": 2,
        "receive_to_render_ms": 20,
        "speech_end_to_ack_upper_bound_ms": 800,
    }
    (metrics / "display_metrics_session.jsonl").write_text(
        "".join(
            json.dumps(r) + "\n"
            for r in [
                ack,
                ack,
                {**ack, "event_id": "b", "timing_schema_version": 1},
                {**ack, "event_id": "c", "visible": False},
            ]
        )
    )
    result = evaluation.report_results(inputs, tmp_path / "report", manifest)
    events = result["caption_events"][0]["metrics"]
    assert events["first_partial_ms"] == {"n": 2, "p50": 125, "p95": 150}
    assert events["partial_update_gap_ms"] == {"n": 2, "p50": 100, "p95": 100}
    assert events["receive_to_render_ms"]["n"] == 1


def test_report_never_pools_different_pipeline_sources_or_duplicate_sessions(replay_case, tmp_path):
    manifest, inputs, metrics, run = replay_case
    evaluation.write_json(inputs / "duplicate.json", run)
    second = {**run, "session_id": "second", "environment": {"source_sha256": {"dry_run_ab.py": "changed"}}}
    second["command"] = ["second" if x == "session" else x for x in run["command"]]
    evaluation.write_json(inputs / "second.json", second)
    (metrics / "ab_metrics_second.csv").write_text(
        "timing_schema_version,endpoint_reason,speech_end_to_final_ms\n2,silence,900\n"
    )
    result = evaluation.report_results(inputs, tmp_path / "report", manifest)
    assert len(result["replays"]) == 2
    assert {r["metrics"]["speech_end_to_final_ms"]["p50"] for r in result["replays"]} == {600, 900}
    assert len(result["excluded_runs"]) == 1


def test_comparison_rejects_managed_arg_overrides_and_failed_resume(replay_case, tmp_path, monkeypatch):
    manifest, inputs, _, _ = replay_case
    args = SimpleNamespace(
        manifest=manifest,
        output=inputs,
        language=None,
        sizes=["e4b"],
        runs=1,
        tag="test",
        experiment="baseline",
        pipeline_args=["--replay-speed", "2"],
    )
    with pytest.raises(ValueError, match="comparison-managed"):
        evaluation.run_replays(args)
    args.pipeline_args = []
    evaluation.write_json(inputs / "test_baseline_e4b_r0_clip_en.json", {"returncode": 1, "error": "failed"})
    with pytest.raises(ValueError, match="failed/incomplete"):
        evaluation.run_replays(args)


def test_worker_start_failure_has_durable_report(replay_case, tmp_path, monkeypatch):
    from tools import replay_bench

    manifest, _, _, _ = replay_case

    def failed(*args, **kwargs):
        raise OSError("cannot execute")

    monkeypatch.setattr(replay_bench, "run_child", failed)
    destination = tmp_path / "quality_failed.json"
    result = evaluation._run_evaluation_worker(["missing-worker"], destination, manifest)
    report = json.loads(destination.read_text())
    assert result.returncode != 0 and report["completed"] is False
    assert "cannot execute" in report["error"] and report["command"] == ["missing-worker"]


def test_quality_input_coverage_and_runtime_cohorts_gate_model_deltas(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluation, "ROOT", tmp_path)
    item = {
        "id": "q",
        "source": "Grace",
        "source_lang": "en",
        "target_lang": "es",
        "reference": None,
        "required_terms": [],
    }
    manifest = tmp_path / "manifest.json"
    evaluation.write_json(manifest, {"usage": "evaluation_only", "translations": [item], "utterances": []})
    inputs = tmp_path / "inputs"
    for size, code in [("e4b", "one"), ("e2b", "two")]:
        evaluation.write_json(
            inputs / f"quality_{size}.json",
            {
                "manifest_sha256": evaluation.digest(manifest),
                "size": size,
                "policy": "none",
                "completed": True,
                "environment": {"source_sha256": {"engines/mlx_engine.py": code}},
                "rows": [{**item, "runs": [{"text": "Gracia", "latency_ms": 100}], "canary_pass": None}],
            },
        )
    result = evaluation.report_results(inputs, tmp_path / "report", manifest)
    assert len(result["quality"]) == 2 and result["quality_deltas"] == []
    assert (tmp_path / "report/blind_review.jsonl").read_text() == ""
    bad = json.loads((inputs / "quality_e2b.json").read_text())
    bad["rows"][0]["source"] = "Changed model input"
    evaluation.write_json(inputs / "quality_e2b.json", bad)
    result = evaluation.report_results(inputs, tmp_path / "report", manifest)
    assert len(result["quality"]) == 1 and result["excluded_runs"]


def test_temporary_corpus_export_does_not_write_project_holdout(tmp_path, monkeypatch):
    from training.prepare_bible_corpus import export_training_jsonl

    monkeypatch.chdir(tmp_path)
    real_holdout = tmp_path / "bible_data/holdout/verse_pairs_test.jsonl"
    real_holdout.parent.mkdir(parents=True)
    real_holdout.write_text("do not change\n")
    export_training_jsonl(
        [{"en": "Grace", "es": "Gracia", "verse_id": "01001001"}],
        str(tmp_path / "temporary/aligned/pairs.jsonl"),
        holdout_ratio=1,
    )
    assert real_holdout.read_text() == "do not change\n"
    assert (tmp_path / "temporary/holdout/verse_pairs_test.jsonl").exists()


def test_reference_repair_joins_book_chapter_verse_and_keeps_inputs(tmp_path):
    import sqlite3

    for name, book_id, text in (("KJV", 1, "The grace of God"), ("SpaRV", 12, "La gracia de Dios")):
        with sqlite3.connect(tmp_path / f"{name}.db") as connection:
            connection.execute(f"CREATE TABLE {name}_books (id INTEGER, name TEXT)")
            connection.execute(
                f"CREATE TABLE {name}_verses (id INTEGER, book_id INTEGER, chapter INTEGER, verse INTEGER, text TEXT)"
            )
            connection.execute(f"INSERT INTO {name}_books VALUES (?, ?)", (book_id, "John"))
            connection.execute(f"INSERT INTO {name}_verses VALUES (?,?,?,?,?)", (book_id * 100, book_id, 1, 1, text))
    row = {
        "id": "test",
        "source": "The grace of God",
        "source_lang": "en",
        "target_lang": "es",
        "reference": "wrong verse",
        "domain": "public_domain_verse",
    }
    manifest = {"translations": [row]}
    fixed = evaluation.realign_references(manifest, tmp_path)
    assert row["reference"] == "wrong verse"
    assert fixed["translations"][0]["reference"] == "La gracia de Dios"
    assert fixed["translations"][0]["source"] == row["source"]


def test_rescoring_preserves_every_prediction_and_measurement(tmp_path):
    source = tmp_path / "source.json"
    manifest = tmp_path / "manifest.json"
    row = {
        "id": "verse",
        "source": "Grace",
        "source_lang": "en",
        "target_lang": "es",
        "reference": "wrong",
        "runs": [{"text": "Gracia", "latency_ms": 234.5}],
    }
    evaluation.write_json(source, {"manifest_sha256": "old", "rows": [row]})
    evaluation.write_json(manifest, {"translations": [{**row, "reference": "Gracia"}]})
    destination = tmp_path / "rescored.json"
    evaluation.rescore_quality(source, destination, manifest)
    result = json.loads(destination.read_text())
    assert result["rows"][0]["runs"] == row["runs"]
    assert result["rows"][0]["reference"] == "Gracia"
    assert json.loads(source.read_text())["rows"][0]["reference"] == "wrong"


def test_resume_accepts_reference_only_repair_and_numeric_old_speed(replay_case, monkeypatch):
    manifest, inputs, metrics, run = replay_case
    tag = "test_baseline_e4b_r0_clip_en"
    run["session_id"] = tag
    run["replay_speed"] = 1.0
    run["command"] = [
        "python",
        "dry_run_ab.py",
        "--audio-file",
        str(evaluation.ROOT / "clip.wav"),
        "--session-id",
        tag,
        "--lang",
        "en",
        "--http-port",
        "9000",
        "--model-family",
        "gemma4",
        "--gemma4-size",
        "e4b",
        "--no-mts",
        "--stt-backend",
        "parakeet-mlx",
        "--partial-interval",
        "0.6",
    ]
    run["experiment_settings"] = {
        k: v for k, v in evaluation.os.environ.items() if k.startswith(("STARK_VAD_", "STARK_TRANSLATE_"))
    }
    evaluation.write_json(inputs / f"{tag}.json", run)
    (metrics / f"ab_metrics_{tag}.csv").write_text("chunk_id\n1\n")
    monkeypatch.setattr(evaluation, "environment", lambda: run["environment"])
    args = SimpleNamespace(
        manifest=manifest,
        output=inputs,
        language=None,
        sizes=["e4b"],
        runs=1,
        tag="test",
        experiment="baseline",
        pipeline_args=[],
    )
    evaluation.run_replays(args)
    assert json.loads((inputs / f"{tag}.json").read_text())["manifest_sha256"] == "older-reference-only-manifest"


@pytest.mark.parametrize(("name", "control"), [("mlx", "forced"), ("parakeet-mlx", "auto")])
def test_stt_workers_run_every_english_and_spanish_candidate(tmp_path, monkeypatch, name, control):
    import sys

    import numpy as np

    from tools import stt_roundtrip_compare

    monkeypatch.setattr(evaluation, "ROOT", tmp_path)
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"fixture")
    rows = [
        {
            "id": str(n),
            "path": "audio.wav",
            "sha256": evaluation.digest(audio),
            "lang": "en" if n % 2 else "es",
            "provenance": "unconfirmed_session",
            "reference_status": "pending",
        }
        for n in range(61)
    ]
    manifest = tmp_path / "manifest.json"
    evaluation.write_json(manifest, {"usage": "evaluation_only", "utterances": rows})
    calls = []
    engine = SimpleNamespace(
        model_id="test-model",
        load=lambda: None,
        unload=lambda: None,
        transcribe=lambda audio, language: calls.append(language) or SimpleNamespace(text="grace", latency_ms=12),
    )
    monkeypatch.setattr(stt_roundtrip_compare, "make_engine", lambda _: engine)
    monkeypatch.setattr(
        evaluation, "_resolved_model", lambda _: {"resolved_model": "/snapshot", "model_revision": "revision"}
    )
    monkeypatch.setattr(evaluation, "environment", lambda: {})
    monkeypatch.setattr(sys.modules["scipy.io"].wavfile, "read", lambda _: (16000, np.zeros(80, dtype=np.float32)))
    monkeypatch.setitem(sys.modules, "scipy.signal", SimpleNamespace(resample_poly=lambda audio, *args: audio))
    monkeypatch.setattr(sys.modules["mlx"].core, "get_peak_memory", lambda: 0)
    output = tmp_path / f"stt_{name}.json"
    evaluation.stt_worker(SimpleNamespace(manifest=manifest, engine=name, runs=1, output=output))
    result = json.loads(output.read_text())
    assert len(result["rows"]) == 61 and calls == [r["lang"] for r in rows]
    assert result["language_control"] == control and result["model_revision"] == "revision"
    assert evaluation._stt_inputs_match(result, json.loads(manifest.read_text()))
    result["rows"].pop()
    assert not evaluation._stt_inputs_match(result, json.loads(manifest.read_text()))


def test_ack_coverage_counts_unique_final_chunks_across_clients(replay_case, tmp_path):
    manifest, inputs, metrics, _ = replay_case
    (metrics / "ab_metrics_session.csv").write_text("chunk_id,timing_schema_version\n1,2\n2,2\n3,2\n")
    base = {
        "event": "caption_rendered",
        "stage": "complete",
        "visible": True,
        "timing_schema_version": 2,
        "client_id": "a",
        "event_id": "first",
        "chunk_id": 1,
        "receive_to_render_ms": 3,
    }
    rows = [
        base,
        {**base, "client_id": "b"},
        {**base, "chunk_id": 2, "event_id": "second"},
        {**base, "chunk_id": 3, "event_id": "third", "visible": False},
    ]
    (metrics / "display_metrics_session.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    coverage = evaluation.report_results(inputs, tmp_path / "report", manifest)["caption_events"][0][
        "final_ack_coverage"
    ]
    assert coverage["received_final_chunks"] == 2 and coverage["finalized_chunks"] == 3
    assert coverage["sessions"][0]["missing_final_chunks"] == ["3"]


def test_experiment_counters_use_latest_session_summary(replay_case, tmp_path):
    manifest, inputs, metrics, _ = replay_case
    records = [
        {"event": "session_summary", "latency_experiment_counters": {"warmup_requested": 3, "warmup_executed": 1}},
        {"event": "chunk_diagnostic", "latency_experiment_counters": {"warmup_requested": 99}},
        {"event": "session_summary", "latency_experiment_counters": {"warmup_requested": 4, "warmup_executed": 0}},
    ]
    (metrics / "diagnostics_session.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    events = evaluation.report_results(inputs, tmp_path / "report", manifest)["caption_events"][0]
    assert events["latency_experiment_counters"] == {"warmup_requested": 4, "warmup_executed": 0}
    assert set(events["counter_sessions"]) == {"session"}


def test_startup_pipeline_hash_overrides_later_source_snapshot(replay_case, tmp_path):
    manifest, inputs, metrics, run = replay_case
    lifecycle = {"schema_version": 1, "session_id": "session", "pipeline_sha256": "a" * 64}
    evaluation.write_json(metrics / "session_lifecycle_session.json", lifecycle)
    original_hash = evaluation._runtime_cohort({**run, "session_lifecycle": lifecycle})
    changed = {**run, "environment": {"source_sha256": {"dry_run_ab.py": "changed after startup"}}}
    assert evaluation._runtime_cohort({**changed, "session_lifecycle": lifecycle}) == original_hash
    assert (
        evaluation._runtime_cohort({**changed, "session_lifecycle": {**lifecycle, "session_id": "other"}})
        != original_hash
    )
    result = evaluation.report_results(inputs, tmp_path / "report", manifest)
    assert result["source_cohorts"][original_hash]["pipeline_hash_basis"] == "startup_source_file"
    assert result["source_cohorts"][original_hash]["startup_pipeline_sha256"] == "a" * 64
    assert "session_lifecycle" not in json.loads((inputs / "replay.json").read_text())


@pytest.fixture
def paired_review_report(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluation, "ROOT", tmp_path)
    item = {
        "id": "q",
        "source": "Grace",
        "source_lang": "en",
        "target_lang": "es",
        "reference": None,
        "required_terms": [],
    }
    manifest = tmp_path / "manifest.json"
    evaluation.write_json(manifest, {"usage": "evaluation_only", "translations": [item], "utterances": []})
    inputs, output = tmp_path / "inputs", tmp_path / "report"
    for size, text in [("e4b", "Gracia"), ("e2b", "La gracia")]:
        evaluation.write_json(
            inputs / f"quality_{size}.json",
            {
                "manifest_sha256": evaluation.digest(manifest),
                "size": size,
                "policy": "none",
                "completed": True,
                "environment": {"source_sha256": {"engines/mlx_engine.py": "same-code"}},
                "rows": [{**item, "runs": [{"text": text, "latency_ms": 100}], "canary_pass": None}],
            },
        )
    evaluation.report_results(inputs, output, manifest)
    return inputs, output, manifest


def test_report_regeneration_preserves_human_ratings_and_notes(paired_review_report):
    inputs, output, manifest = paired_review_report
    path = output / "blind_review.jsonl"
    pair = json.loads(path.read_text())
    pair.update(
        meaning_error_A=False,
        meaning_error_B=0,
        terminology_preference="B",
        reviewed=True,
        reviewer_notes="Checked the theological term in context.",
    )
    path.write_text(json.dumps(pair) + "\n")
    evaluation.report_results(inputs, output, manifest)
    assert json.loads(path.read_text()) == pair


@pytest.mark.parametrize("change", ["output", "remove", "source", "reference"])
def test_changed_reviewed_pairs_leave_every_report_artifact_untouched(paired_review_report, change):
    inputs, output, manifest_path = paired_review_report
    path = output / "blind_review.jsonl"
    pair = json.loads(path.read_text())
    # A false rating is still saved work even before the row is marked reviewed.
    pair.update(meaning_error_A=False, reviewed=False)
    path.write_text(json.dumps(pair) + "\n")
    before = {p.name: p.read_bytes() for p in output.iterdir()}
    selected = inputs / "quality_e4b.json"
    if change == "remove":
        selected.unlink()
    elif change == "output":
        run = json.loads(selected.read_text())
        run["rows"][0]["runs"][0]["text"] = "Changed model answer"
        evaluation.write_json(selected, run)
    else:
        manifest = json.loads(manifest_path.read_text())
        manifest["translations"][0][change] = "Changed comparison context"
        evaluation.write_json(manifest_path, manifest)
        for source in inputs.glob("quality_*.json"):
            run = json.loads(source.read_text())
            run["manifest_sha256"] = evaluation.digest(manifest_path)
            run["rows"][0][change] = "Changed comparison context"
            evaluation.write_json(source, run)
    with pytest.raises(ValueError, match=r"Reviewed pair.*fresh report directory"):
        evaluation.report_results(inputs, output, manifest_path)
    assert {p.name: p.read_bytes() for p in output.iterdir()} == before


def test_malformed_existing_review_is_not_silently_discarded(paired_review_report):
    inputs, output, manifest = paired_review_report
    (output / "blind_review.jsonl").write_text("{unfinished human edit\n")
    before = {p.name: p.read_bytes() for p in output.iterdir()}
    with pytest.raises(ValueError, match="Invalid existing blind review"):
        evaluation.report_results(inputs, output, manifest)
    assert {p.name: p.read_bytes() for p in output.iterdir()} == before
