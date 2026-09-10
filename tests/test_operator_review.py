"""Review -> portable export -> real merge contracts; no model/audio-device loads."""

import argparse
import csv
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from operator_app.pipeline_manager import PipelineRunner, SessionStatus, get_runner
from operator_app.review import router
from tools.merge_corrections import merge_translation, merge_whisper
from tools.prepare_finetune_data import cmd_export_translation, cmd_export_whisper, load_diagnostics
from tools.review_data import ReviewConflict, ReviewStore, atomic_jsonl, normalize_record, read_jsonl


@pytest.fixture
def review(tmp_path):
    store = ReviewStore(tmp_path)
    session = "20260909_120000_en"
    audio = tmp_path / "stark_data" / "live_sessions" / session / "chunk_0001.wav"
    audio.parent.mkdir(parents=True)
    audio.write_bytes(b"RIFFtest-wave")
    row = {
        "session": session,
        "chunk_id": 1,
        "english": "God loves you.",
        "spanish_gemma": "Dios te ama.",
        "source_lang": "en",
        "session_kind": "live",
        "audio_path": str(audio.relative_to(tmp_path)),
        "review_priority": 3,
        "qe_a": 0.9,
    }
    atomic_jsonl(store.diagnostics_path(session), [row])
    return store, session, audio, row


def approved(**overrides):
    result = {
        "expected_revision": 0,
        "source_lang": "en",
        "corrected_source_text": "God loves you.",
        "corrected_translation_text": "Dios te ama.",
        "transcript_approved": True,
        "translation_approved": True,
        "excluded": False,
        "review_note": "Checked.",
    }
    result.update(overrides)
    return result


def test_live_review_revisions_and_original_diagnostics_unchanged(review):
    store, session, audio, _ = review
    original = store.diagnostics_path(session).read_bytes()
    first = store.save(session, 1, approved(transcript_approved=False, translation_approved=False))
    assert first["revision"] == 1 and first["review_note"] == "Checked."
    assert store.records(session)[0]["revision"] == 1
    with pytest.raises(ReviewConflict):
        store.save(session, 1, approved())
    second = store.save(session, 1, approved(expected_revision=1))
    assert second["revision"] == 2
    assert len(read_jsonl(store.sidecar_path(session))) == 2
    assert store.diagnostics_path(session).read_bytes() == original
    assert audio.read_bytes() == b"RIFFtest-wave"


def test_only_complete_final_records_are_reviewable(review):
    store, session, _, row = review
    with store.diagnostics_path(session).open("a") as stream:
        stream.write(json.dumps({**row, "chunk_id": 2}))  # unfinished writer record
    assert [r["chunk_id"] for r in store.records(session)] == [1]
    with pytest.raises(FileNotFoundError):
        store.save(session, 2, approved())


@pytest.mark.parametrize("bad", ["../../secret.wav", "/tmp/outside.wav"])
def test_audio_rejects_escape(review, bad):
    store, session, _, row = review
    atomic_jsonl(store.diagnostics_path(session), [{**row, "audio_path": bad}])
    with pytest.raises(ValueError):
        store.audio_path(session, 1)
    assert not store.public_record(store.record(session, 1))["audio_available"]


def test_audio_rejects_symlink_escape(review, tmp_path):
    store, session, audio, _ = review
    outside = tmp_path / "outside.wav"
    outside.write_bytes(b"outside")
    audio.unlink()
    audio.symlink_to(outside)
    with pytest.raises(ValueError):
        store.audio_path(session, 1)


def test_missing_audio_allows_translation_but_not_stt_export(review):
    store, session, audio, _ = review
    audio.unlink()
    saved = store.save(session, 1, approved())
    assert not saved["audio_available"]
    result = store.export(session)
    assert result["stt_samples"] == {"en": 0, "es": 0}
    assert result["translation_pairs"] == 1


def test_portable_bundle_roundtrip_and_idempotent_merges(review, tmp_path):
    store, session, audio, _ = review
    store.save(session, 1, approved())
    result = store.export(session)
    assert store.export(session)["bundle_id"] == result["bundle_id"]
    moved = tmp_path / "moved"
    with zipfile.ZipFile(result["archive"]) as archive:
        archive.extractall(moved)
    audio.unlink()
    train_dir = tmp_path / "train"
    first = merge_whisper(moved, train_dir)
    assert first["added"] == 1
    assert merge_whisper(moved, train_dir)["added"] == 0
    assert merge_whisper(moved, train_dir)["updated"] == 0
    rows = read_jsonl(train_dir / "metadata.jsonl")
    assert len(rows) == 1 and (train_dir / rows[0]["file_name"]).read_bytes() == b"RIFFtest-wave"
    pairs = moved / "translation" / "train.jsonl"
    target = tmp_path / "pairs_train.jsonl"
    assert merge_translation(pairs, target)["added"] == 1
    assert merge_translation(pairs, target)["added"] == 0
    assert read_jsonl(target)[0]["en"] == "God loves you."


def test_independent_approvals_exclusion_and_spanish_direction(review):
    store, session, _, row = review
    row.update(source_lang="es", english="Dios te ama.", spanish_gemma="God loves you.")
    atomic_jsonl(store.diagnostics_path(session), [row])
    store.save(
        session,
        1,
        approved(
            source_lang="es",
            corrected_source_text="Dios te ama.",
            corrected_translation_text="God loves you.",
            translation_approved=False,
        ),
    )
    first = store.export(session)
    assert first["stt_samples"] == {"en": 0, "es": 1} and first["translation_pairs"] == 0
    store.save(
        session,
        1,
        approved(
            expected_revision=1,
            source_lang="es",
            corrected_source_text="Dios te ama.",
            corrected_translation_text="God loves you.",
        ),
    )
    result = store.export(session)
    directory = Path(result["archive"]).with_suffix("")
    pair = read_jsonl(directory / "translation" / "train.jsonl")[0]
    assert (pair["en"], pair["es"]) == ("God loves you.", "Dios te ama.")
    spanish_dir = directory / "whisper" / "train" / "es"
    with pytest.raises(ValueError, match="language"):
        merge_whisper(spanish_dir, store.root / "train")
    assert merge_whisper(spanish_dir, store.root / "spanish_train", language="es")["added"] == 1
    store.save(session, 1, approved(expected_revision=2, excluded=True))
    with pytest.raises(ValueError, match="Approve"):
        store.export(session)


@pytest.mark.parametrize("kind", ["replay", "unknown", "synthetic"])
def test_non_live_sessions_cannot_train(review, kind):
    store, session, _, row = review
    atomic_jsonl(store.diagnostics_path(session), [{**row, "session_kind": kind}])
    store.save(session, 1, approved())
    with pytest.raises(ValueError, match="provenance"):
        store.export(session)
    result = store.export(session, split="eval")
    with pytest.raises(ValueError, match="Evaluation"):
        merge_whisper(Path(result["archive"]).with_suffix(""), store.root / "train")


def test_session_cannot_cross_eval_train_split(review):
    store, session, _, _ = review
    store.save(session, 1, approved())
    result = store.export(session, split="eval")
    with pytest.raises(ReviewConflict):
        store.export(session, split="train")
    path = Path(result["archive"]).with_suffix("") / "translation" / "eval.jsonl"
    with pytest.raises(ValueError, match="evaluation"):
        merge_translation(path, store.root / "train.jsonl")


def test_legacy_language_resolution_and_unknown_requires_choice():
    assert normalize_record({"session": "legacy_es", "english": "Hola"})["source_lang"] == "es"
    assert normalize_record({"session": "20260301_113532", "english": "Hello"})["source_lang"] is None


def test_api_live_save_audio_and_completed_export(review):
    store, session, audio, _ = review
    runner = PipelineRunner(store.root)
    runner._status = SessionStatus("running", session_id=session)
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_runner] = lambda: runner
    with TestClient(app) as client:
        assert client.get("/api/review/sessions").json()["sessions"][0]["active"]
        assert client.get(f"/api/review/{session}/segments").json()["total"] == 1
        endpoint = f"/api/review/{session}/segments/1"
        assert client.get(endpoint + "/audio").content == audio.read_bytes()
        assert client.put(endpoint, json=approved()).status_code == 200
        assert client.put(endpoint, json=approved()).status_code == 409
        assert client.post(f"/api/review/{session}/export", json={}).status_code == 409
        runner._status.state = "idle"
        exported = client.post(f"/api/review/{session}/export", json={})
        assert exported.status_code == 200
        assert client.get(exported.json()["download_url"]).status_code == 200


def test_existing_cli_exports_read_sidecars_and_real_merger(review, tmp_path):
    store, session, _, _ = review
    store.save(session, 1, approved(corrected_source_text="God loves every person."))
    assert load_diagnostics([store.diagnostics_path(session)])[0]["corrected_english"] == "God loves every person."
    output = tmp_path / "whisper"
    args = argparse.Namespace(
        session=[session],
        metrics_dir=str(store.metrics),
        output=str(output),
        accent="live",
        eval_ratio=0,
        source_lang=None,
    )
    cmd_export_whisper(args)
    assert merge_whisper(output, tmp_path / "train")["added"] == 1
    pairs = tmp_path / "pairs.jsonl"
    args.output = str(pairs)
    args.min_qe = 0.6
    cmd_export_translation(args)
    result = subprocess.run(
        [
            sys.executable,
            "tools/merge_corrections.py",
            "translation",
            "--corrections",
            str(pairs),
            "--train-jsonl",
            str(tmp_path / "pair_train.jsonl"),
        ],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[1],
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["added"] == 1


def test_long_valid_session_export_is_downloadable(review):
    store, _, _, row = review
    session = "s" * 160
    atomic_jsonl(store.diagnostics_path(session), [{**row, "session": session, "audio_path": None}])
    store.save(session, 1, approved())
    runner = PipelineRunner(store.root)
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_runner] = lambda: runner
    with TestClient(app) as client:
        response = client.post(f"/api/review/{session}/export", json={})
        assert response.status_code == 200
        assert client.get(response.json()["download_url"]).status_code == 200
        assert client.get("/api/review/exports/malformed").status_code == 400


@pytest.mark.parametrize("provenance", ["replay", "synthetic", "is_eval", "registry"])
def test_legacy_cli_preserves_evaluation_provenance(review, tmp_path, provenance):
    store, session, _, row = review
    if provenance in ("replay", "synthetic"):
        row["session_kind"] = provenance
    elif provenance == "is_eval":
        row["is_eval"] = True
    else:
        atomic_jsonl(store.corrections / "export_registry.jsonl", [{"session": session, "split": "eval"}])
    atomic_jsonl(store.diagnostics_path(session), [row])
    store.save(session, 1, approved())
    output = tmp_path / "whisper"
    args = argparse.Namespace(
        session=[session],
        metrics_dir=str(store.metrics),
        output=str(output),
        accent="live",
        eval_ratio=0,
        source_lang=None,
        min_qe=0.6,
    )
    cmd_export_whisper(args)
    with (output / "train" / "metadata.csv").open() as stream:
        assert list(csv.DictReader(stream)) == []
    with (output / "eval" / "metadata.csv").open() as stream:
        assert len(list(csv.DictReader(stream))) == 1
    args.output = str(tmp_path / "pairs.jsonl")
    cmd_export_translation(args)
    assert read_jsonl(Path(args.output))[0]["split"] == "eval"
    with pytest.raises(ValueError, match="evaluation"):
        merge_translation(Path(args.output), tmp_path / "train.jsonl")


def test_generic_spanish_correction_pair_direction_and_replay_rejection(tmp_path):
    source = tmp_path / "pairs.jsonl"
    row = {"source_text": "Dios te ama.", "target_text": "God loves you.", "source_lang": "es"}
    atomic_jsonl(source, [row])
    target = tmp_path / "train.jsonl"
    merge_translation(source, target)
    assert read_jsonl(target)[0]["en"] == "God loves you."
    atomic_jsonl(source, [{**row, "split": "train", "session_kind": "replay"}])
    with pytest.raises(ValueError, match="replay"):
        merge_translation(source, target)


def test_multiple_sessions_same_chunk_id_do_not_collide(review, tmp_path):
    store, session, _, row = review
    second = "20260909_130000_en"
    second_audio = tmp_path / "stark_data" / "live_sessions" / second / "chunk_0001.wav"
    second_audio.parent.mkdir(parents=True)
    second_audio.write_bytes(b"second-wave")
    atomic_jsonl(store.diagnostics_path(second), [{**row, "session": second, "audio_path": str(second_audio)}])
    for sid in [session, second]:
        store.save(sid, 1, approved())
        result = store.export(sid)
        merge_whisper(Path(result["archive"]).with_suffix(""), tmp_path / "train")
    rows = read_jsonl(tmp_path / "train" / "metadata.jsonl")
    assert len(rows) == 2 and rows[0]["file_name"] != rows[1]["file_name"]


def test_unknown_language_requires_explicit_choice_for_approval(review):
    store, session, _, row = review
    unknown = "legacy_session"
    atomic_jsonl(store.diagnostics_path(unknown), [{**row, "session": unknown, "source_lang": None}])
    with pytest.raises(ValueError, match="language"):
        store.save(unknown, 1, approved(source_lang=None))
    assert store.save(unknown, 1, approved(source_lang="es"))["source_lang"] == "es"


def test_translation_approval_alone_does_not_promote_unreviewed_transcript(review):
    store, session, _, _ = review
    store.save(session, 1, approved(transcript_approved=False))
    with pytest.raises(ValueError, match="No exportable"):
        store.export(session)


def test_csv_training_corpus_repeat_merge_is_idempotent(review, tmp_path):
    store, session, _, _ = review
    store.save(session, 1, approved())
    bundle = Path(store.export(session)["archive"]).with_suffix("")
    train = tmp_path / "train"
    train.mkdir()
    (train / "metadata.csv").write_text("file_name,transcription,source_lang\n")
    assert merge_whisper(bundle, train)["added"] == 1
    again = merge_whisper(bundle, train)
    assert again["added"] == 0 and again["updated"] == 0
    assert not (train / "metadata.jsonl").exists()


def test_concurrent_review_processes_cannot_lose_revisions(review):
    store, session, _, _ = review
    script = """import json, sys
from pathlib import Path
from tools.review_data import ReviewStore, ReviewConflict
try:
    ReviewStore(Path(sys.argv[1])).save(sys.argv[2], 1, json.loads(sys.argv[3]))
except ReviewConflict:
    raise SystemExit(9)
"""
    args = [sys.executable, "-c", script, str(store.root), session, json.dumps(approved())]
    processes = [
        subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=Path(__file__).resolve().parents[1])
        for _ in range(2)
    ]
    codes = []
    for process in processes:
        process.communicate(timeout=10)
        codes.append(process.returncode)
    assert sorted(codes) == [0, 9]
    assert store.record(session, 1)["revision"] == 1
