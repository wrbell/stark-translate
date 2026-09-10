"""Interrupted review exports recover without rewriting source material."""

import json
import shutil
import zipfile
from pathlib import Path

import pytest

from tools import review_data
from tools.review_data import ReviewConflict, ReviewStore, atomic_jsonl, read_jsonl
from tools.session_lifecycle import finish_session, start_session


def snapshot(paths):
    return {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in paths}


@pytest.fixture
def completed_review(tmp_path):
    store = ReviewStore(tmp_path)
    session = "recovery_session_en"
    run = start_session(tmp_path, session)
    audio = tmp_path / "stark_data" / "live_sessions" / session / "chunk_0001.wav"
    audio.parent.mkdir(parents=True)
    audio.write_bytes(b"RIFF-original-audio")
    diagnostics = store.diagnostics_path(session)
    atomic_jsonl(
        diagnostics,
        [{"chunk_id": 1, "source_lang": "en", "session_kind": "live", "audio_path": str(audio), "english": "Grace."}],
    )
    store.save(
        session,
        1,
        {
            "expected_revision": 0,
            "source_lang": "en",
            "corrected_source_text": "Grace.",
            "corrected_translation_text": "Gracia.",
            "transcript_approved": True,
            "translation_approved": True,
        },
    )
    finish_session(tmp_path, session, run_id=run["run_id"])
    originals = snapshot([audio, diagnostics, store.sidecar_path(session)])
    yield store, session
    assert snapshot(originals) == originals


def bundle_paths(result):
    archive = Path(result["archive"])
    return archive.with_suffix(""), archive


def test_valid_repeat_leaves_archive_directory_and_registry_unchanged(completed_review):
    store, session = completed_review
    result = store.export(session)
    directory, archive = bundle_paths(result)
    registry = store.corrections / "export_registry.jsonl"
    before = snapshot([archive, registry, *[p for p in directory.rglob("*") if p.is_file()]])
    assert store.export(session) == result
    assert snapshot(before) == before


@pytest.mark.parametrize("damage", ["truncated", "empty_zip"])
def test_invalid_archive_is_rebuilt_from_unchanged_bundle(completed_review, damage):
    store, session = completed_review
    result = store.export(session)
    directory, archive = bundle_paths(result)
    before = snapshot([p for p in directory.rglob("*") if p.is_file()])
    if damage == "truncated":
        archive.write_bytes(archive.read_bytes()[:40])
    else:
        with zipfile.ZipFile(archive, "w"):
            pass
    assert store.export(session) == result
    with zipfile.ZipFile(archive) as bundle:
        assert bundle.testzip() is None
        assert json.loads(bundle.read("manifest.json"))["samples"] == result["samples"]
        assert bundle.read(f"whisper/train/en/{session}__1.wav") == b"RIFF-original-audio"
    assert snapshot(before) == before
    assert len(read_jsonl(store.corrections / "export_registry.jsonl")) == 1


def test_interrupted_zip_never_publishes_partial_archive(completed_review, monkeypatch):
    store, session = completed_review
    original_write = zipfile.ZipFile.write

    def interrupted(bundle, *args, **kwargs):
        original_write(bundle, *args, **kwargs)
        raise OSError("interrupted ZIP write")

    with monkeypatch.context() as patch:
        patch.setattr(zipfile.ZipFile, "write", interrupted)
        with pytest.raises(OSError, match="interrupted ZIP"):
            store.export(session)
    exports = store.corrections / "exports"
    assert not list(exports.glob("*.zip"))
    assert not list(exports.glob(".export-zip-*"))
    assert not (store.corrections / "export_registry.jsonl").exists()
    # The published directory reserves the split even before registry repair.
    with pytest.raises(ReviewConflict, match="different dataset split"):
        store.export(session, split="eval")
    result = store.export(session)
    with zipfile.ZipFile(result["archive"]) as bundle:
        assert bundle.testzip() is None


def test_registry_interruption_repairs_assignment_without_republishing_valid_zip(completed_review, monkeypatch):
    store, session = completed_review
    original_write = review_data.atomic_jsonl

    def interrupted(path, rows):
        if path.name == "export_registry.jsonl":
            raise OSError("interrupted registry write")
        return original_write(path, rows)

    with monkeypatch.context() as patch:
        patch.setattr(review_data, "atomic_jsonl", interrupted)
        with pytest.raises(OSError, match="interrupted registry"):
            store.export(session)
    archive = next((store.corrections / "exports").glob("*.zip"))
    before = snapshot([archive])
    assert not (store.corrections / "export_registry.jsonl").exists()
    result = store.export(session)
    assert snapshot(before) == before
    registry = read_jsonl(store.corrections / "export_registry.jsonl")
    assert registry == [{"session": session, "split": "train", "bundle_id": result["bundle_id"]}]
    assert store.export(session) == result
    assert read_jsonl(store.corrections / "export_registry.jsonl") == registry


@pytest.mark.parametrize("existing_split", ["train", "eval"])
@pytest.mark.parametrize("retained", ["directory", "archive"])
def test_bundle_reserves_split_even_without_registry(completed_review, existing_split, retained):
    store, session = completed_review
    result = store.export(session, split=existing_split)
    directory, archive = bundle_paths(result)
    (store.corrections / "export_registry.jsonl").unlink()
    if retained == "directory":
        archive.unlink()
    else:
        shutil.rmtree(directory)
        archive.write_bytes(b"interrupted old ZIP")
    other = "eval" if existing_split == "train" else "train"
    with pytest.raises(ReviewConflict, match="different dataset split"):
        store.export(session, split=other)
    assert not (store.corrections / "export_registry.jsonl").exists()


def test_conflicting_manifest_fails_closed(completed_review):
    store, session = completed_review
    result = store.export(session)
    directory, archive = bundle_paths(result)
    path = directory / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["split"] = "eval"
    path.write_text(json.dumps(manifest))
    before = snapshot([path, archive])
    with pytest.raises(ReviewConflict, match="manifest conflicts"):
        store.export(session)
    assert snapshot(before) == before


def test_missing_directory_rebuild_keeps_valid_archive_immutable(completed_review):
    store, session = completed_review
    result = store.export(session)
    directory, archive = bundle_paths(result)
    before = snapshot([archive])
    shutil.rmtree(directory)
    assert store.export(session) == result
    assert snapshot(before) == before


@pytest.mark.parametrize("damage", ["audio", "metadata"])
def test_damaged_directory_cannot_replace_valid_archive(completed_review, damage):
    store, session = completed_review
    result = store.export(session)
    directory, archive = bundle_paths(result)
    before = snapshot([archive, store.corrections / "export_registry.jsonl"])
    language = directory / "whisper" / "train" / "en"
    if damage == "audio":
        (language / f"{session}__1.wav").write_bytes(b"truncated-copy")
    else:
        path = language / "metadata.jsonl"
        rows = read_jsonl(path)
        rows[0]["audio_sha256"] = "wrong"
        atomic_jsonl(path, rows)
    with pytest.raises(ReviewConflict, match=r"damaged|provenance"):
        store.export(session)
    assert snapshot(before) == before
