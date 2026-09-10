"""Portable review archives respect independent approval and exclusion boundaries."""

import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from operator_app.pipeline_manager import PipelineRunner, get_runner
from operator_app.review import router
from tools.review_data import ReviewConflict, ReviewStore, atomic_jsonl, read_jsonl, validate_export_download
from tools.session_lifecycle import finish_session, start_session


@pytest.fixture
def private_review(tmp_path):
    store = ReviewStore(tmp_path)
    session = "privacy_session_en"
    run = start_session(tmp_path, session)
    records = []
    for chunk in range(1, 7):
        audio = tmp_path / "stark_data" / "live_sessions" / session / f"chunk_{chunk:04d}.wav"
        audio.parent.mkdir(parents=True, exist_ok=True)
        audio.write_bytes(f"RIFF-audio-{chunk}".encode())
        records.append(
            {
                "chunk_id": chunk,
                "source_lang": "en",
                "session_kind": "live",
                "audio_path": str(audio) if chunk != 6 else None,
                "english": f"original-prediction-{chunk}",
                "spanish_gemma": f"original-translation-{chunk}",
            }
        )
    atomic_jsonl(store.diagnostics_path(session), records)
    for chunk in range(1, 7):
        store.save(
            session,
            chunk,
            {
                "expected_revision": 0,
                "source_lang": "en",
                "corrected_source_text": f"review-source-{chunk}",
                "corrected_translation_text": f"review-translation-{chunk}",
                # 1: both, 2: STT only, 3: excluded, 4: draft, 5: translation only,
                # 6: transcript only without retained audio (no exportable sample).
                "transcript_approved": chunk in (1, 2, 3, 6),
                "translation_approved": chunk in (1, 3, 5),
                "excluded": chunk == 3,
                "review_note": f"private-review-note-{chunk}",
            },
        )
    finish_session(tmp_path, session, run_id=run["run_id"])
    originals = {
        p: (p.read_bytes(), p.stat().st_mtime_ns)
        for p in [
            store.sidecar_path(session),
            store.diagnostics_path(session),
            *list((tmp_path / "stark_data" / "live_sessions" / session).glob("*.wav")),
        ]
    }
    yield store, session
    assert {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in originals} == originals


def contents(path):
    with zipfile.ZipFile(path) as archive:
        return {name: archive.read(name) for name in archive.namelist()}


def write_zip(path, data):
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, value in data.items():
            archive.writestr(name, value)


def client_for(store):
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_runner] = lambda: PipelineRunner(store.root)
    return TestClient(app)


def test_zip_contains_only_selected_approved_text_and_audio(private_review):
    store, session = private_review
    result = store.export(session)
    data = contents(result["archive"])
    raw = b"\n".join(data.values())
    assert result["schema_version"] == 2
    assert [row["sample_id"] for row in result["samples"]] == [f"{session}__1", f"{session}__2"]
    assert result["stt_samples"] == {"en": 2, "es": 0} and result["translation_pairs"] == 1
    sidecar = [json.loads(row) for row in data["corrections.jsonl"].splitlines()]
    assert [row["chunk_id"] for row in sidecar] == [1, 2]
    assert sidecar[0]["corrected_translation_text"] == "review-translation-1"
    assert "corrected_translation_text" not in sidecar[1]
    for private in ["review-translation-2", "private-review-note", "original-prediction", "original-translation"]:
        assert private.encode() not in raw
    for chunk in range(3, 7):
        assert f"review-source-{chunk}".encode() not in raw
        assert f"review-translation-{chunk}".encode() not in raw
        assert f"RIFF-audio-{chunk}".encode() not in raw
    assert store.export(session) == result
    validate_export_download(Path(result["archive"]), result["bundle_id"])
    with client_for(store) as client:
        response = client.get(f"/api/review/exports/{result['bundle_id']}")
        assert response.status_code == 200 and response.content == Path(result["archive"]).read_bytes()


@pytest.mark.parametrize("leak", ["excluded_row", "unapproved_translation", "review_note", "extra_file"])
def test_repeat_rejects_overbroad_cached_directory_without_republishing(private_review, leak):
    store, session = private_review
    result = store.export(session)
    archive = Path(result["archive"])
    before = archive.read_bytes()
    directory = archive.with_suffix("")
    sidecar = read_jsonl(directory / "corrections.jsonl")
    if leak == "excluded_row":
        sidecar.append(read_jsonl(store.sidecar_path(session))[2])
    elif leak == "unapproved_translation":
        sidecar[1]["corrected_translation_text"] = "private-unapproved-target"
    elif leak == "review_note":
        sidecar[0]["review_note"] = "private-review-note"
    else:
        (directory / "unapproved-drafts.jsonl").write_text("private draft")
    atomic_jsonl(directory / "corrections.jsonl", sidecar)
    with pytest.raises(ReviewConflict, match="privacy policy"):
        store.export(session)
    assert archive.read_bytes() == before


@pytest.mark.parametrize("leak", ["excluded_row", "unapproved_translation", "review_note", "extra_file", "pair_text"])
def test_download_rejects_overbroad_zip_even_with_existing_url(private_review, leak):
    store, session = private_review
    result = store.export(session)
    data = contents(result["archive"])
    sidecar = [json.loads(line) for line in data["corrections.jsonl"].splitlines()]
    if leak == "excluded_row":
        sidecar.append(read_jsonl(store.sidecar_path(session))[2])
    elif leak == "unapproved_translation":
        sidecar[1]["corrected_translation_text"] = "private-unapproved-target"
    elif leak == "review_note":
        sidecar[0]["review_note"] = "private-review-note"
    elif leak == "extra_file":
        data["private-drafts.jsonl"] = b"private draft"
    else:
        pairs = json.loads(data["translation/train.jsonl"])
        pairs["es"] = "unapproved replacement"
        data["translation/train.jsonl"] = json.dumps(pairs).encode()
    data["corrections.jsonl"] = b"\n".join(json.dumps(row).encode() for row in sidecar)
    write_zip(result["archive"], data)
    with client_for(store) as client:
        response = client.get(f"/api/review/exports/{result['bundle_id']}")
    assert response.status_code == 409 and "re-export" in response.json()["detail"]


def test_legacy_zip_url_blocked_and_export_uses_new_identity(private_review):
    store, session = private_review
    result = store.export(session)
    archive = Path(result["archive"])
    data = contents(archive)
    manifest = json.loads(data["manifest.json"])
    manifest["schema_version"] = 1
    legacy_digest = hashlib.sha256(json.dumps(["train", manifest["samples"]], sort_keys=True).encode()).hexdigest()[:16]
    legacy_id = f"{session}-train-{legacy_digest}"
    legacy_archive = archive.with_name(f"{legacy_id}.zip")
    data["manifest.json"] = json.dumps(manifest).encode()
    data["corrections.jsonl"] = store.sidecar_path(session).read_bytes()
    write_zip(legacy_archive, data)
    legacy_original = legacy_archive.read_bytes()
    shutil.rmtree(archive.with_suffix(""))
    archive.unlink()
    with client_for(store) as client:
        blocked = client.get(f"/api/review/exports/{legacy_id}")
        assert blocked.status_code == 409 and "re-export" in blocked.json()["detail"]
        fresh = client.post(f"/api/review/{session}/export", json={})
        assert fresh.status_code == 200
        assert fresh.json()["bundle_id"] != legacy_id
        assert client.get(fresh.json()["download_url"]).status_code == 200
    assert legacy_archive.read_bytes() == legacy_original


@pytest.mark.parametrize("field", ["corrected_translation_text", "review_note", "excluded"])
def test_download_enforces_projection_even_with_recomputed_identity(private_review, field):
    store, session = private_review
    result = store.export(session)
    data = contents(result["archive"])
    corrections = [json.loads(line) for line in data["corrections.jsonl"].splitlines()]
    corrections[1][field] = True if field == "excluded" else "private unapproved content"
    digest = hashlib.sha256(
        json.dumps([2, "train", result["samples"], corrections], sort_keys=True).encode()
    ).hexdigest()[:16]
    bundle_id = f"{session}-train-{digest}"
    data["corrections.jsonl"] = b"\n".join(json.dumps(row).encode() for row in corrections)
    path = Path(result["archive"]).with_name(f"{bundle_id}.zip")
    write_zip(path, data)
    with client_for(store) as client:
        response = client.get(f"/api/review/exports/{bundle_id}")
    assert response.status_code == 409 and "privacy policy" in response.json()["detail"]
