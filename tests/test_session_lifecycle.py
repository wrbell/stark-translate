"""Completion evidence must come from a drained pipeline, never operator idleness."""

import ast
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.session_lifecycle import (
    SessionNotComplete,
    finish_session,
    migrate_completion,
    require_completed,
    session_status,
    start_session,
)


def _diagnostics(root: Path, session: str = "example_en") -> Path:
    path = root / "metrics" / f"diagnostics_{session}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"chunk_id":1,"english":"Hello"}\n')
    return path


def test_unknown_running_completed_and_modified_diagnostics(tmp_path):
    session = "example_en"
    path = _diagnostics(tmp_path, session)
    assert session_status(tmp_path, session)["status"] == "unknown"
    with pytest.raises(SessionNotComplete, match="unknown"):
        require_completed(tmp_path, session)
    with pytest.raises(SessionNotComplete, match="already exists"):
        start_session(tmp_path, session)
    path.unlink()
    run = start_session(tmp_path, session, git_sha="abc")
    path = _diagnostics(tmp_path, session)
    assert run["pid"] == os.getpid() and run["git_sha"] == "abc"
    assert session_status(tmp_path, session)["active"]
    with pytest.raises(SessionNotComplete, match="already exists"):
        start_session(tmp_path, session)
    with pytest.raises(SessionNotComplete, match="Finish"):
        require_completed(tmp_path, session)
    finish_session(tmp_path, session, run_id=run["run_id"])
    require_completed(tmp_path, session)
    with pytest.raises(SessionNotComplete, match="already exists"):
        start_session(tmp_path, session)
    path.write_text(path.read_text() + '{"chunk_id":2}\n')
    with pytest.raises(SessionNotComplete, match="changed"):
        require_completed(tmp_path, session)


def test_failure_dead_process_and_wrong_run_cannot_complete(tmp_path, monkeypatch):
    run = start_session(tmp_path, "example_en")
    with pytest.raises(SessionNotComplete, match="ownership"):
        finish_session(tmp_path, "example_en", run_id="wrong")
    monkeypatch.setattr("tools.session_lifecycle._alive", lambda pid: False)
    assert session_status(tmp_path, "example_en")["status"] == "failed"
    with pytest.raises(ValueError, match="zero"):
        finish_session(tmp_path, "example_en", run_id=run["run_id"], exit_code=2)
    finish_session(tmp_path, "example_en", run_id=run["run_id"], status="failed", exit_code=2)
    with pytest.raises(SessionNotComplete, match="abnormally"):
        require_completed(tmp_path, "example_en")


@pytest.mark.parametrize("session", ["../../escape", "../secret", "bad/path", "bad..id"])
def test_invalid_lifecycle_paths_rejected(tmp_path, session):
    with pytest.raises(ValueError, match="Invalid session"):
        start_session(tmp_path, session)


def test_lifecycle_and_diagnostics_symlink_escapes_rejected(tmp_path):
    root = tmp_path / "project"
    (root / "metrics").mkdir(parents=True)
    outside = tmp_path / "secret"
    outside.write_text("outside")
    marker = root / "metrics" / "session_lifecycle_example_en.json"
    marker.symlink_to(outside)
    with pytest.raises(ValueError, match="inside"):
        start_session(root, "example_en")
    assert outside.read_text() == "outside"


def test_explicit_migration_requires_matching_successful_subprocess(tmp_path):
    _diagnostics(tmp_path)
    path = tmp_path / "replay_report.json"
    report = {
        "session_id": "example_en",
        "command": ["python", "dry_run_ab.py", "--session-id", "example_en"],
        "returncode": 0,
    }
    for changed in ({"returncode": 1}, {"session_id": "other"}, {"command": ["python"]}, {"error": "failed"}):
        path.write_text(json.dumps({**report, **changed}))
        with pytest.raises(SessionNotComplete, match="successful subprocess"):
            migrate_completion(tmp_path, "example_en", path)
    path.write_text(json.dumps(report))
    marker = migrate_completion(tmp_path, "example_en", path)
    assert marker["completion_source"] == "successful_subprocess_report_migration"
    require_completed(tmp_path, "example_en")
    with pytest.raises(SessionNotComplete, match="already exists"):
        migrate_completion(tmp_path, "example_en", path)


@pytest.mark.parametrize("fail", [False, True])
def test_pipeline_completion_waits_for_io_and_blocks_abnormal_exit(tmp_path, monkeypatch, fail):
    """Execute only main's lifecycle wrapper, excluding all model/audio imports."""
    source = Path(__file__).resolve().parents[1] / "dry_run_ab.py"
    tree = ast.parse(source.read_text())
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    start = next(
        i for i, node in enumerate(main.body) if isinstance(node, ast.Try) and "asyncio.run(" in ast.unparse(node)
    )
    wrapper = ast.Module(body=main.body[start:], type_ignores=[])
    pool = ThreadPoolExecutor(max_workers=1)
    monkeypatch.chdir(tmp_path)

    def pipeline(args):
        pool.submit(_diagnostics, tmp_path)
        if fail:
            raise RuntimeError("pipeline failed")

    namespace = {
        "__file__": str(source),
        "subprocess": SimpleNamespace(check_output=lambda *args, **kwargs: "abc", CalledProcessError=RuntimeError),
        "asyncio": SimpleNamespace(run=lambda result: result),
        "main_async": pipeline,
        "args": None,
        "SESSION_ID": "example_en",
        "_clean_session_shutdown": False,
        "_io_pool": pool,
        "sys": __import__("sys"),
        "print_summary": lambda: None,
        "completed": False,
        "lifecycle_root": tmp_path,
        "lifecycle": start_session(tmp_path, "example_en"),
        "finish_session": finish_session,
    }
    if fail:
        with pytest.raises(RuntimeError, match="pipeline failed"):
            exec(compile(wrapper, str(source), "exec"), namespace)
        assert session_status(tmp_path, "example_en")["status"] == "failed"
    else:
        exec(compile(wrapper, str(source), "exec"), namespace)
        require_completed(tmp_path, "example_en")
    assert pool._shutdown


@pytest.mark.parametrize("external_running", [False, True])
def test_idle_operator_blocks_unknown_or_external_session_export_but_allows_review(tmp_path, external_running):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from operator_app.pipeline_manager import get_runner
    from operator_app.review import router

    session = "example_en"
    if external_running:
        start_session(tmp_path, session)
    _diagnostics(tmp_path, session)
    app = FastAPI()
    app.include_router(router)
    runner = SimpleNamespace(_project_root=tmp_path, status=lambda: SimpleNamespace(state="idle", session_id=None))
    app.dependency_overrides[get_runner] = lambda: runner
    client = TestClient(app)
    state = client.get("/api/review/sessions").json()["sessions"][0]
    assert state["status"] == ("running" if external_running else "unknown")
    response = client.post(f"/api/review/{session}/export", json={"split": "train"})
    assert response.status_code == 409
    saved = client.put(
        f"/api/review/{session}/segments/1",
        json={
            "expected_revision": 0,
            "source_lang": "en",
            "corrected_source_text": "Hello",
            "corrected_translation_text": "Hola",
        },
    )
    assert saved.status_code == 200
