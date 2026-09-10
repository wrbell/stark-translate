"""Scoped support exports and explicit cleanup of regenerable completed-session logs."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import threading
import time
import uuid
import zipfile
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from operator_app.pipeline_manager import PipelineRunner, get_runner
from tools.pipeline_health import read_health
from tools.session_lifecycle import _path, _read, require_completed

router = APIRouter(prefix="/api", tags=["support"])
_previews = {}
_lock = threading.Lock()
MAX_BUNDLE_BYTES = 100 * 1024 * 1024


def _scoped(root, path):
    root, path = root.resolve(), Path(path)
    if path.is_symlink() or not path.resolve().is_relative_to(root):
        raise ValueError("Path must stay inside this installation")
    return path


def _digest(path):
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _remember(root, kind, payload):
    token = uuid.uuid4().hex
    with _lock:
        now = time.time()
        for key in list(_previews):
            if now - _previews[key]["at"] > 600:
                del _previews[key]
        if len(_previews) >= 100:
            raise ValueError("Too many pending previews; wait before trying again")
        _previews[token] = {"root": str(root.resolve()), "kind": kind, "payload": payload, "at": now}
    return token


def _take(root, kind, token):
    with _lock:
        item = _previews.get(token)
        if not item or item["root"] != str(root.resolve()) or item["kind"] != kind or time.time() - item["at"] > 600:
            raise ValueError("Preview expired or belongs to another installation; preview again")
        del _previews[token]
        return item["payload"]


class SupportRequest(BaseModel):
    session_id: str
    include_text: bool = False
    include_audio: bool = False


class PreviewRequest(BaseModel):
    preview_id: str


class CleanupRequest(BaseModel):
    session_ids: list[str] = Field(min_length=1, max_length=100)


def storage_snapshot(root):
    disk = shutil.disk_usage(root)
    metrics = _scoped(root, root / "metrics")
    sessions = []
    if metrics.is_dir():
        for marker in sorted(metrics.glob("session_lifecycle_*.json"), reverse=True)[:200]:
            if marker.is_symlink():
                continue
            session = marker.name[len("session_lifecycle_") : -5]
            data = _read(root, session)
            logs = [
                p
                for p in metrics.glob(f"session_{session}.log*")
                if re.fullmatch(re.escape(f"session_{session}.log") + r"(?:\.[1-5])?", p.name) and not p.is_symlink()
            ]
            sessions.append(
                {
                    "session_id": session,
                    "status": data.get("status", "unknown"),
                    "cleanup_bytes": sum(p.stat().st_size for p in logs if p.is_file()),
                    "originals_preserved": True,
                }
            )
    return {
        "free_bytes": disk.free,
        "used_bytes": disk.used,
        "total_bytes": disk.total,
        "low_space": disk.free < 1024**3,
        "sessions": sessions,
        "cleanup_scope": "Completed-session operational logs only; original audio, diagnostics, corrections and exports are preserved",
    }


def preview_support(root, req):
    _path(root, req.session_id, "health", "json")  # validate session before any glob
    lifecycle = _read(root, req.session_id)
    health = read_health(root, req.session_id)
    # Explicit allowlist: no raw errors, environment, usernames, paths or captions.
    metadata = {
        "schema_version": 1,
        "session_id": req.session_id,
        "lifecycle": {
            k: lifecycle.get(k)
            for k in (
                "status",
                "exit_code",
                "started_at",
                "ended_at",
                "pipeline_sha256",
                "git_sha",
                "memory",
                "persistence",
            )
        },
        "health": {
            k: health.get(k)
            for k in ("phase", "stale", "age_s", "queues", "error_count", "errors", "persistence", "publish_failures")
        },
        "models": {
            role: {
                key: value
                for key, value in item.items()
                if key
                in {"manifest_revision", "resolved_revision", "revision_source", "model_bin_sha256", "quantization"}
            }
            for role, item in lifecycle.get("models", {}).items()
            if isinstance(item, dict)
        },
        "storage": {k: v for k, v in storage_snapshot(root).items() if k != "sessions"},
    }
    files = []
    if req.include_text:
        for kind, suffix in (("diagnostics", "jsonl"), ("ab_metrics", "csv"), ("partials", "jsonl")):
            path = _path(root, req.session_id, kind, suffix)
            if path.is_file():
                files.append(_scoped(root, path))
    if req.include_audio:
        directory = _scoped(root, root / "stark_data" / "live_sessions" / req.session_id)
        if directory.is_dir():
            files.extend(_scoped(root, p) for p in sorted(directory.glob("chunk_*.wav")) if p.is_file())
    entries = [{"path": str(p.relative_to(root)), "bytes": p.stat().st_size, "sha256": _digest(p)} for p in files]
    size = sum(item["bytes"] for item in entries)
    if size > MAX_BUNDLE_BYTES:
        raise ValueError("Support attachments exceed 100 MiB; export metadata only or select a shorter session")
    payload = {
        "metadata": metadata,
        "files": entries,
        "session_id": req.session_id,
        "include_text": req.include_text,
        "include_audio": req.include_audio,
    }
    token = _remember(root, "support", payload)
    return {
        "preview_id": token,
        "files": ["metadata.json", *[e["path"] for e in entries]],
        "bytes": size,
        "privacy": {
            "text_included": req.include_text,
            "audio_included": req.include_audio,
            "message": "Metadata excludes transcript, raw log messages, environment and local model paths. Attachments contain private session content when selected.",
        },
    }


def export_support(root, token):
    payload = _take(root, "support", token)
    folder = _scoped(root, root / "metrics" / "support")
    folder.mkdir(parents=True, exist_ok=True)
    bundle_id = uuid.uuid4().hex
    target = folder / f"{bundle_id}.zip"
    temporary = folder / f".{bundle_id}.tmp"
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("metadata.json", json.dumps(payload["metadata"], indent=2))
            for entry in payload["files"]:
                path = _scoped(root, root / entry["path"])
                if path.stat().st_size != entry["bytes"] or _digest(path) != entry["sha256"]:
                    raise ValueError("Session content changed after preview; preview again")
                archive.write(path, entry["path"])
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return {"bundle_id": bundle_id, "download_url": f"/api/support/download/{bundle_id}"}


def cleanup_preview(root, sessions):
    entries = []
    for session in dict.fromkeys(sessions):
        require_completed(root, session)
        for number in range(6):
            suffix = "log" + (f".{number}" if number else "")
            path = _path(root, session, "session", suffix)
            if path.is_file():
                _scoped(root, path)
                entries.append(
                    {"path": str(path.relative_to(root)), "sha256": _digest(path), "bytes": path.stat().st_size}
                )
    token = _remember(root, "cleanup", {"files": entries, "sessions": sessions})
    return {
        "preview_id": token,
        "files": [e["path"] for e in entries],
        "bytes": sum(e["bytes"] for e in entries),
        "originals_preserved": True,
    }


def cleanup(root, token):
    payload = _take(root, "cleanup", token)
    for session in payload["sessions"]:
        require_completed(root, session)
    for entry in payload["files"]:
        path = _scoped(root, root / entry["path"])
        if not path.is_file() or _digest(path) != entry["sha256"]:
            raise ValueError("Logs changed after preview; preview again")
    removed = 0
    for entry in payload["files"]:
        _scoped(root, root / entry["path"]).unlink()
        removed += entry["bytes"]
    return {"removed_bytes": removed, "originals_preserved": True}


def _call(fn, *args):
    try:
        return fn(*args)
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/storage")
def api_storage(runner: PipelineRunner = Depends(get_runner)):
    return _call(storage_snapshot, runner._project_root)


@router.post("/support/preview")
def api_support_preview(req: SupportRequest, runner: PipelineRunner = Depends(get_runner)):
    return _call(preview_support, runner._project_root, req)


@router.post("/support/export")
def api_support_export(req: PreviewRequest, runner: PipelineRunner = Depends(get_runner)):
    return _call(export_support, runner._project_root, req.preview_id)


@router.get("/support/download/{bundle_id}")
def api_support_download(bundle_id: str, runner: PipelineRunner = Depends(get_runner)):
    if not re.fullmatch(r"[0-9a-f]{32}", bundle_id):
        raise HTTPException(status_code=404, detail="Support bundle not found")
    path = _scoped(runner._project_root, runner._project_root / "metrics" / "support" / f"{bundle_id}.zip")
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Support bundle not found")
    return FileResponse(path, media_type="application/zip", filename=f"support-{bundle_id}.zip")


@router.post("/storage/cleanup/preview")
def api_cleanup_preview(req: CleanupRequest, runner: PipelineRunner = Depends(get_runner)):
    return _call(cleanup_preview, runner._project_root, req.session_ids)


@router.post("/storage/cleanup")
def api_cleanup(req: PreviewRequest, runner: PipelineRunner = Depends(get_runner)):
    return _call(cleanup, runner._project_root, req.preview_id)
