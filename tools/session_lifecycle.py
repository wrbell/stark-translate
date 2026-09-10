"""Explicit pipeline completion evidence for review exports, without model imports."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import socket
import sys
import tempfile
import uuid
from datetime import UTC, datetime
from pathlib import Path


class SessionNotComplete(ValueError):
    """Session completion has not been proved, so only review drafts are allowed."""


def _path(root: Path, session: str, kind: str, suffix: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,159}", session) or ".." in session:
        raise ValueError("Invalid session ID")
    root = root.resolve()
    metrics = root / "metrics"
    path = metrics / f"{kind}_{session}.{suffix}"
    if not metrics.resolve().is_relative_to(root) or path.resolve().parent != metrics.resolve():
        raise ValueError("Session lifecycle files must stay inside project metrics")
    return path


def _write(path: Path, data: dict, *, exclusive: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive:
        # Claim the ID atomically; concurrent CLI/operator starts must not share
        # diagnostics, audio paths, or previously approved corrections.
        try:
            with path.open("x", encoding="utf-8") as stream:
                json.dump(data, stream, indent=2)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
        except FileExistsError as exc:
            raise SessionNotComplete("This session ID already exists; choose a new session ID") from exc
        return
    fd, name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(data, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def _digest(path: Path) -> dict:
    hasher = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            size += len(block)
            hasher.update(block)
    return {"size_bytes": size, "sha256": hasher.hexdigest()}


def _read(root: Path, session: str) -> dict:
    path = _path(root, session, "session_lifecycle", "json")
    try:
        data = json.loads(path.read_text())
    except (FileNotFoundError, ValueError, UnicodeDecodeError):
        return {}
    if not isinstance(data, dict) or data.get("session_id") != session or data.get("schema_version") != 1:
        return {}
    return data


def _alive(pid: object) -> bool:
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def completion_metadata(model_ids: dict[str, str | None], root: Path) -> dict:
    """Read process peak counters and local model provenance after inference ends.

    No inference package is imported here. RSS/Metal peaks include model loading
    and cover this process only, so multiprocessing worker peaks remain unknown.
    """
    memory: dict = {"scope": "pipeline_process_lifetime", "peak_rss_bytes": None, "peak_metal_bytes": None}
    try:
        import resource

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        memory["peak_rss_bytes"] = int(rss if sys.platform == "darwin" else rss * 1024)
    except (ImportError, OSError, ValueError):
        pass
    mlx = sys.modules.get("mlx.core")
    get_peak = getattr(mlx, "get_peak_memory", None)
    if callable(get_peak):
        try:
            memory["peak_metal_bytes"] = int(get_peak())
        except Exception as exc:
            memory["metal_counter_error"] = type(exc).__name__

    from engines.model_paths import load_model_manifest, resolve_model_path

    try:
        manifest = load_model_manifest(root).get("models", {})
    except (OSError, ValueError):
        manifest = {}
    models = {}
    for role, model_id in model_ids.items():
        if not model_id:
            continue
        entry: dict = next(
            (v for k, v in manifest.items() if model_id in {k, v.get("repo_id"), *v.get("aliases", [])}), {}
        )
        item = {"requested_id": model_id, "manifest_revision": entry.get("revision"), "resolved_revision": None}
        try:
            resolved = resolve_model_path(model_id, project_root=root, local_only=True)
            item["resolved_path"] = resolved
            item["revision_source"] = "unknown"
            if resolved:
                path = Path(resolved)
                if path.parent.name == "snapshots" and re.fullmatch(r"[0-9a-f]{40}", path.name):
                    item["resolved_revision"] = path.name
                    item["revision_source"] = "hf_snapshot_path"
                marker = path / ".installed"
                if marker.is_file():
                    installed = json.loads(marker.read_text())
                    if (
                        isinstance(installed, dict)
                        and installed.get("repo_id") == entry.get("repo_id")
                        and installed.get("revision")
                    ):
                        item["resolved_revision"] = installed["revision"]
                        item["revision_source"] = "setup_install_marker"
                config = path / "config.json"
                if config.is_file():
                    item["config_sha256"] = _digest(config)["sha256"]
                # CT2 exports do not necessarily retain their HF source revision.
                # Hash the loaded local weights after inference instead of treating
                # the manifest's declared hash as proof of the file contents.
                model_bin = path / "model.bin"
                if model_bin.is_file():
                    weights = _digest(model_bin)
                    item["model_bin_sha256"] = weights["sha256"]
                    item["model_bin_size_bytes"] = weights["size_bytes"]
                    export_manifest = path / "export_manifest.json"
                    if export_manifest.is_file():
                        exported = json.loads(export_manifest.read_text())
                        if not isinstance(exported, dict):
                            raise ValueError("Model export manifest must be an object")
                        item["export_manifest"] = {
                            "sha256": _digest(export_manifest)["sha256"],
                            "source_model_id": exported.get("model_id"),
                            "source_revision": exported.get("source_revision"),
                            "ct2_quantization": exported.get("ct2_quantization"),
                            "direction": exported.get("direction"),
                            "declared_model_bin_sha256": exported.get("model_bin_sha256"),
                            "model_bin_hash_matches": exported.get("model_bin_sha256") == weights["sha256"],
                        }
        except (OSError, ValueError, TypeError) as exc:
            item["resolution_error"] = type(exc).__name__
        models[role] = item
    return {"memory": memory, "models": models}


def start_session(root: Path, session: str, *, git_sha: str | None = None, pipeline_sha256: str | None = None) -> dict:
    """Record ownership before model loading. This does not touch predictions/audio."""
    marker = _path(root, session, "session_lifecycle", "json")
    if marker.exists() or _path(root, session, "diagnostics", "jsonl").exists():
        raise SessionNotComplete(
            "This session ID already exists; choose a new session ID to preserve its recordings and reviews"
        )
    data = {
        "schema_version": 1,
        "session_id": session,
        "run_id": uuid.uuid4().hex,
        "status": "running",
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "started_at": datetime.now(UTC).isoformat(),
        "git_sha": git_sha,
        "pipeline_sha256": pipeline_sha256,
    }
    _write(marker, data, exclusive=True)
    return data


def finish_session(
    root: Path,
    session: str,
    *,
    run_id: str,
    status: str = "completed",
    exit_code: int = 0,
    model_ids: dict[str, str | None] | None = None,
) -> dict:
    """Call only after all diagnostics writers have drained; failure never enables export."""
    if status not in {"completed", "failed"} or (status == "completed" and exit_code != 0):
        raise ValueError("A completed session requires exit code zero")
    data = _read(root, session)
    if data.get("run_id") != run_id or data.get("status") != "running":
        raise SessionNotComplete("Session lifecycle ownership changed before completion")
    data.update(status=status, exit_code=exit_code, ended_at=datetime.now(UTC).isoformat())
    data.update(completion_metadata(model_ids or {}, root))
    diagnostics = _path(root, session, "diagnostics", "jsonl")
    if status == "completed" and diagnostics.is_file():
        data["diagnostics"] = _digest(diagnostics)
    _write(_path(root, session, "session_lifecycle", "json"), data)
    return data


def session_status(root: Path, session: str) -> dict:
    data = _read(root, session)
    status = data.get("status", "unknown")
    active = status == "running"
    reason = "Completion is unknown. Review drafts are saved; verify a successful run and migrate its completion evidence before export."
    if status == "running":
        reason = "Finish this session before exporting; review drafts are saved"
        if data.get("hostname") == socket.gethostname() and not _alive(data.get("pid")):
            status, active = "failed", False
            reason = "The pipeline exited without a completion marker; review drafts are saved but export is blocked"
    elif status == "failed":
        reason = "This session ended abnormally; review drafts are saved but export is blocked"
    elif status == "completed":
        diagnostics = _path(root, session, "diagnostics", "jsonl")
        if data.get("exit_code") == 0 and diagnostics.is_file() and data.get("diagnostics") == _digest(diagnostics):
            return {"status": "completed", "active": False, "exportable": True, "reason": ""}
        status = "unknown"
        reason = "Diagnostics changed or are missing after completion; verify the session before exporting"
    return {"status": status, "active": active, "exportable": False, "reason": reason}


def require_completed(root: Path, session: str) -> None:
    status = session_status(root, session)
    if not status["exportable"]:
        raise SessionNotComplete(status["reason"])


def request_graceful_stop(task) -> bool:
    """Cancel only the main task so its finally block can drain worker tasks."""
    if task is None or task.done():
        return False
    task.get_loop().call_soon_threadsafe(task.cancel)
    return True


def migrate_completion(root: Path, session: str, report_path: Path) -> dict:
    """Explicitly recover a legacy replay from its successful subprocess report."""
    prior = _read(root, session)
    if prior or _path(root, session, "session_lifecycle", "json").exists():
        raise SessionNotComplete("A lifecycle marker already exists; migration only accepts legacy sessions")
    report = json.loads(report_path.read_text())
    if not isinstance(report, dict):
        raise SessionNotComplete("Migration requires a successful subprocess report object")
    command = report.get("command", [])
    valid_command = isinstance(command, list) and all(isinstance(part, str) for part in command)
    valid_command = valid_command and any(Path(part).name == "dry_run_ab.py" for part in command)
    if valid_command:
        try:
            valid_command = command[command.index("--session-id") + 1] == session
        except (ValueError, IndexError):
            valid_command = False
    if (
        report.get("session_id") != session
        or type(report.get("returncode")) is not int
        or report["returncode"] != 0
        or "error" in report
        or report.get("timed_out")
        or not valid_command
    ):
        raise SessionNotComplete("Migration requires a successful subprocess report for this exact replay session")
    diagnostics = _path(root, session, "diagnostics", "jsonl")
    if not diagnostics.is_file():
        raise SessionNotComplete("Session diagnostics are missing")
    data = {
        "schema_version": 1,
        "session_id": session,
        "run_id": uuid.uuid4().hex,
        "status": "completed",
        "exit_code": 0,
        "ended_at": datetime.now(UTC).isoformat(),
        "completion_source": "successful_subprocess_report_migration",
        "subprocess_report": {"path": str(report_path.resolve()), **_digest(report_path)},
        "diagnostics": _digest(diagnostics),
        "git_sha": report.get("session_metadata", {}).get("git_sha"),
    }
    _write(_path(root, session, "session_lifecycle", "json"), data)
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--session", required=True)
    parser.add_argument("--migrate-successful-replay", type=Path, metavar="REPORT_JSON")
    args = parser.parse_args()
    if args.migrate_successful_replay:
        migrate_completion(args.root, args.session, args.migrate_successful_replay)
    print(json.dumps(session_status(args.root, args.session), indent=2))


if __name__ == "__main__":
    main()
