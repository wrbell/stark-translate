import json

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
    evaluation.write_json(manifest, {"usage": "evaluation_only", "utterances": []})
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
        evaluation.write_json(
            inputs / f"run{i}.json",
            {"experiment": "baseline", "size": "e4b", "language": "en", "provenance": provenance, "session_id": str(i)},
        )
        (metrics / f"ab_metrics_{i}.csv").write_text(
            "timing_schema_version,finalization_reason,speech_end_to_final_ms,e2e_latency_ms\n"
            f"{schema},{endpoint},{delay},300\n"
        )
    result = evaluation.report_results(inputs, tmp_path / "report", manifest)
    assert len(result["replays"]) == 4
    legacy = next(r for r in result["replays"] if r["schema"] == "legacy")
    assert legacy["metrics"]["speech_end_to_final_ms"]["p50"] is None
    assert json.loads((tmp_path / "report/comparison.json").read_text())["human_review"] == "pending"


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
