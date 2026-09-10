"""Review finalized chunks during or after a session, without changing live captions."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from operator_app.pipeline_manager import PipelineRunner, get_runner
from tools.review_data import ReviewConflict, ReviewStore, session_key

router = APIRouter(prefix="/api/review", tags=["review"])
_ACTIVE = {"starting", "running", "paused", "stopping"}


class CorrectionRequest(BaseModel):
    expected_revision: int = Field(ge=0)
    corrected_source_text: str = Field(max_length=30000)
    corrected_translation_text: str = Field(max_length=30000)
    transcript_approved: bool = False
    translation_approved: bool = False
    excluded: bool = False
    review_note: str = Field(default="", max_length=5000)
    source_lang: Literal["en", "es"] | None = None


class ExportRequest(BaseModel):
    split: Literal["train", "eval"] = "train"


def _store(runner: PipelineRunner) -> ReviewStore:
    return ReviewStore(runner._project_root)


def _error(exc: Exception) -> HTTPException:
    code = 409 if isinstance(exc, ReviewConflict) else 404 if isinstance(exc, FileNotFoundError) else 400
    return HTTPException(status_code=code, detail=str(exc))


@router.get("/sessions")
def review_sessions(runner: PipelineRunner = Depends(get_runner)) -> dict:
    snap = runner.status()
    active = snap.session_id if snap.state in _ACTIVE else None
    return {"sessions": _store(runner).sessions(active)}


@router.get("/{session}/segments")
def review_segments(
    session: str,
    pending_only: bool = True,
    flagged_only: bool = True,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=200),
    runner: PipelineRunner = Depends(get_runner),
) -> dict:
    store = _store(runner)
    try:
        records = store.records(session)
        # Context stays in chronological order even when the review queue is prioritized.
        contexts = {
            r["chunk_id"]: {
                "previous": records[i - 1]["source_text"] if i else "",
                "next": records[i + 1]["source_text"] if i + 1 < len(records) else "",
            }
            for i, r in enumerate(records)
        }
        if pending_only:
            records = [
                r for r in records if not r["excluded"] and not (r["transcript_approved"] and r["translation_approved"])
            ]
        if flagged_only:
            records = [r for r in records if (r.get("review_priority") or 0) >= 1]
        records.sort(key=lambda r: (-(r.get("review_priority") or 0), r["chunk_id"]))
        return {
            "session": session,
            "total": len(records),
            "offset": offset,
            "segments": [
                {**store.public_record(r), "context": contexts[r["chunk_id"]]} for r in records[offset : offset + limit]
            ],
        }
    except (FileNotFoundError, ValueError) as exc:
        raise _error(exc) from exc


@router.get("/{session}/segments/{chunk}/audio")
def review_audio(session: str, chunk: int, runner: PipelineRunner = Depends(get_runner)) -> FileResponse:
    try:
        path = _store(runner).audio_path(session, chunk)
        return FileResponse(path, media_type="audio/wav", headers={"Cache-Control": "no-store"})
    except (FileNotFoundError, ValueError) as exc:
        raise _error(exc) from exc


@router.put("/{session}/segments/{chunk}")
def save_review(
    session: str,
    chunk: int,
    req: CorrectionRequest,
    runner: PipelineRunner = Depends(get_runner),
) -> dict:
    try:
        return _store(runner).save(session, chunk, req.model_dump())
    except (FileNotFoundError, ValueError) as exc:
        raise _error(exc) from exc


@router.post("/{session}/export")
def export_review(session: str, req: ExportRequest, runner: PipelineRunner = Depends(get_runner)) -> dict:
    snap = runner.status()
    if snap.session_id == session and snap.state in _ACTIVE:
        raise HTTPException(status_code=409, detail="Finish this session before exporting; review drafts are saved")
    try:
        result = _store(runner).export(session, split=req.split)
        result.pop("archive", None)
        result["download_url"] = f"/api/review/exports/{result['bundle_id']}"
        return result
    except (FileNotFoundError, ValueError) as exc:
        raise _error(exc) from exc


@router.get("/exports/{bundle_id}")
def download_review(bundle_id: str, runner: PipelineRunner = Depends(get_runner)) -> FileResponse:
    try:
        # A bundle adds its split and hash to a valid session ID, so its maximum
        # length is greater than a session ID's maximum length.
        session, split, digest = bundle_id.rsplit("-", 2)
        session_key(session)
        if split not in {"train", "eval"} or len(digest) != 16 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("Invalid export ID")
        directory = _store(runner).corrections / "exports"
        path = (directory / f"{bundle_id}.zip").resolve()
        if not path.is_relative_to(directory.resolve()) or not path.is_file():
            raise FileNotFoundError("Export not found")
        return FileResponse(path, media_type="application/zip", filename=path.name)
    except (FileNotFoundError, ValueError) as exc:
        raise _error(exc) from exc
