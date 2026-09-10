"""Completion evidence must come from a drained pipeline, never operator idleness."""

import ast
import asyncio
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.session_lifecycle import (
    SessionNotComplete,
    completion_metadata,
    finish_session,
    migrate_completion,
    request_graceful_stop,
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
        "session_metadata": {"git_sha": "abc123"},
    }
    for changed in (
        {"returncode": 1},
        {"session_id": "other"},
        {"command": ["python"]},
        {"error": "failed"},
        {"error": ""},
        {"timed_out": True},
    ):
        path.write_text(json.dumps({**report, **changed}))
        with pytest.raises(SessionNotComplete, match="successful subprocess"):
            migrate_completion(tmp_path, "example_en", path)
    path.write_text(json.dumps(report))
    marker = migrate_completion(tmp_path, "example_en", path)
    assert marker["completion_source"] == "successful_subprocess_report_migration"
    assert marker["git_sha"] == "abc123"
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
        "asyncio": SimpleNamespace(run=lambda result: result, CancelledError=asyncio.CancelledError),
        "main_async": pipeline,
        "args": None,
        "SESSION_ID": "example_en",
        "_clean_session_shutdown": False,
        "_session_model_ids": {},
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


def test_completion_records_peak_counters_without_importing_mlx(tmp_path, monkeypatch):
    import resource
    import sys

    monkeypatch.setitem(sys.modules, "mlx.core", SimpleNamespace(get_peak_memory=lambda: 123456))
    monkeypatch.setattr(resource, "getrusage", lambda who: SimpleNamespace(ru_maxrss=256))
    monkeypatch.setattr(sys, "platform", "darwin")
    result = completion_metadata({}, tmp_path)
    assert result["memory"]["peak_rss_bytes"] == 256
    assert result["memory"]["peak_metal_bytes"] == 123456
    monkeypatch.delitem(sys.modules, "mlx.core")
    monkeypatch.setattr(sys, "platform", "linux")
    result = completion_metadata({}, tmp_path)
    assert result["memory"]["peak_rss_bytes"] == 256 * 1024
    assert result["memory"]["peak_metal_bytes"] is None
    assert "mlx.core" not in sys.modules


def test_model_metadata_distinguishes_resolved_and_manifest_revisions(tmp_path, monkeypatch):
    revision = "a" * 40
    snapshot = tmp_path / "snapshots" / revision
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}")
    monkeypatch.setattr("engines.model_paths.resolve_model_path", lambda *args, **kwargs: str(snapshot))
    monkeypatch.setattr(
        "engines.model_paths.load_model_manifest",
        lambda root: {"models": {"model": {"repo_id": "org/repo", "revision": "b" * 40}}},
    )
    result = completion_metadata({"translation_a": "org/repo"}, tmp_path)["models"]["translation_a"]
    assert result["resolved_revision"] == revision
    assert result["manifest_revision"] == "b" * 40
    assert result["revision_source"] == "hf_snapshot_path"
    assert len(result["config_sha256"]) == 64


@pytest.mark.parametrize("declared_hash_matches", [False, True])
def test_ct2_metadata_hashes_actual_weights_without_inventing_source_revision(tmp_path, declared_hash_matches):
    import hashlib

    model_dir = tmp_path / "adapters" / "marian_ct2" / "en-es" / "active"
    model_dir.mkdir(parents=True)
    weights = b"tiny mocked CT2 weights"
    actual_hash = hashlib.sha256(weights).hexdigest()
    (model_dir / "model.bin").write_bytes(weights)
    (model_dir / "export_manifest.json").write_text(
        json.dumps(
            {
                "model_id": "Helsinki-NLP/opus-mt-en-es",
                "ct2_quantization": "int8",
                "direction": "en-es",
                "model_bin_sha256": actual_hash if declared_hash_matches else "0" * 64,
            }
        )
    )
    result = completion_metadata({"translation_b": str(model_dir)}, tmp_path)["models"]["translation_b"]
    assert result["model_bin_sha256"] == actual_hash
    assert result["model_bin_size_bytes"] == len(weights)
    assert result["resolved_revision"] is None
    assert result["revision_source"] == "unknown"
    assert result["export_manifest"]["source_model_id"] == "Helsinki-NLP/opus-mt-en-es"
    assert result["export_manifest"]["ct2_quantization"] == "int8"
    assert result["export_manifest"]["model_bin_hash_matches"] is declared_hash_matches


@pytest.mark.parametrize("forced_stop", [False, True])
@pytest.mark.parametrize("stop_signal", [2, 15])
def test_actual_stop_handler_drains_pipeline_and_only_then_marks_complete(
    tmp_path, monkeypatch, forced_stop, stop_signal
):
    """Run the production signal handler/finally blocks with inert model stubs."""
    import sys

    source = Path(__file__).resolve().parents[1] / "dry_run_ab.py"
    tree = ast.parse(source.read_text())
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    main_async = next(
        node for node in tree.body if isinstance(node, ast.AsyncFunctionDef) and node.name == "main_async"
    )
    signal_handler = next(
        node for node in main.body if isinstance(node, ast.FunctionDef) and node.name == "signal_handler"
    )
    shutdown = next(
        node for node in main_async.body if isinstance(node, ast.Try) and "await audio_loop()" in ast.unparse(node)
    )
    wrapper_start = next(
        i for i, node in enumerate(main.body) if isinstance(node, ast.Try) and "asyncio.run(" in ast.unparse(node)
    )
    cleanup = ast.AsyncFunctionDef(
        name="cleanup",
        args=ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]),
        body=[ast.Global(names=["_clean_session_shutdown"]), shutdown],
        decorator_list=[],
    )
    pool = ThreadPoolExecutor(max_workers=1)
    monkeypatch.chdir(tmp_path)
    events = []
    registered_signals = {}

    async def done():
        pass

    async def audio_loop():
        asyncio.get_running_loop().call_soon(registered_signals[stop_signal], stop_signal, None)
        await asyncio.Event().wait()

    def summary():
        events.append("summary")
        pool.submit(_diagnostics, tmp_path)

    async def pipeline(args):
        namespace["_session_main_task"] = asyncio.current_task()
        queue = asyncio.Queue()
        namespace["_pipeline_chunk_queue"] = queue

        async def coordinator():
            assert await queue.get() is None
            if forced_stop:
                raise asyncio.CancelledError("forced stop while draining")
            events.append("coordinator_drained")

        namespace["pipeline_task"] = asyncio.create_task(coordinator())
        namespace["rolling_task"] = asyncio.create_task(asyncio.sleep(30))
        namespace["stream_task"] = asyncio.create_task(asyncio.sleep(30))
        await namespace["cleanup"]()

    namespace = {
        "asyncio": asyncio,
        "sys": sys,
        "audio_loop": audio_loop,
        "_session_stop_requested": False,
        "_clean_session_shutdown": False,
        "_session_main_task": None,
        "request_graceful_stop": request_graceful_stop,
        "_stream_token_queue": None,
        "speaker_task": None,
        "_marian_engine": None,
        "_pytorch_pool": SimpleNamespace(shutdown=lambda **kwargs: None),
        "_tts_pool": None,
        "tts_engine": None,
        "MULTIPROCESS": False,
        "stop_diarize_daemon": lambda: None,
        "print_summary": summary,
        "ws_server": SimpleNamespace(close=lambda: None, wait_closed=done),
        "tts_ws_server": None,
        "main_async": pipeline,
        "args": None,
        "completed": False,
        "_io_pool": pool,
        "_session_model_ids": {},
        "SESSION_ID": "example_en",
        "lifecycle_root": tmp_path,
        "lifecycle": start_session(tmp_path, "example_en"),
        "finish_session": finish_session,
        "signal": SimpleNamespace(
            SIGINT=2, SIGTERM=15, signal=lambda number, handler: registered_signals.update({number: handler})
        ),
    }
    definitions = ast.fix_missing_locations(ast.Module(body=[signal_handler, cleanup], type_ignores=[]))
    exec(compile(definitions, str(source), "exec"), namespace)
    registrations = [
        node
        for node in main.body
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and ast.unparse(node.value.func) == "signal.signal"
    ]
    exec(compile(ast.Module(body=registrations, type_ignores=[]), str(source), "exec"), namespace)
    exec(compile(ast.Module(body=main.body[wrapper_start:], type_ignores=[]), str(source), "exec"), namespace)
    if forced_stop:
        assert session_status(tmp_path, "example_en")["status"] == "failed"
        assert "summary" not in events
    else:
        require_completed(tmp_path, "example_en")
        assert events == ["coordinator_drained", "summary"]


def test_sigkill_leaves_no_completed_export_evidence(tmp_path):
    import subprocess
    import sys

    code = (
        "from pathlib import Path; from tools.session_lifecycle import start_session; import time; "
        "start_session(Path(__import__('sys').argv[1]), 'example_en'); print('ready', flush=True); time.sleep(30)"
    )
    process = subprocess.Popen([sys.executable, "-c", code, str(tmp_path)], stdout=subprocess.PIPE, text=True)
    try:
        assert process.stdout.readline().strip() == "ready"
        process.kill()
        process.wait(timeout=5)
        assert session_status(tmp_path, "example_en")["status"] == "failed"
        with pytest.raises(SessionNotComplete):
            require_completed(tmp_path, "example_en")
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
