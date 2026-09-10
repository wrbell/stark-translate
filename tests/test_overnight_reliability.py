"""Real I/O and child-process failures, without models or audio devices."""

import json
import logging
import sys
import threading
import time
import zipfile

import pytest

from operator_app.support import SupportRequest, cleanup, cleanup_preview, export_support, preview_support
from operator_app.work_lease import WorkBusyError, WorkLease
from tools.operational_logging import AsyncOperationalHandler, configure_log
from tools.persistence import PersistenceExecutor
from tools.pipeline_health import PipelineHealth, read_health, send_control
from tools.session_lifecycle import finish_session, require_completed, start_session


def complete(root, session="service_en"):
    lifecycle = start_session(root, session)
    (root / "metrics" / f"diagnostics_{session}.jsonl").write_text('{"chunk_id":1}\n')
    finish_session(root, session, run_id=lifecycle["run_id"])


def until(predicate, timeout=3):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("Condition did not become true")


def test_failed_write_cannot_enable_export(tmp_path):
    session = start_session(tmp_path, "service_en")
    pool = PersistenceExecutor(max_workers=1)
    pool.submit((tmp_path / "metrics" / "diagnostics_service_en.jsonl").write_text, '{"chunk_id":1}\n')
    pool.submit((tmp_path / "missing" / "audio.wav").write_bytes, b"audio")
    pool.shutdown(wait=True)
    evidence = pool.snapshot()
    assert evidence["completed"] == 1 and evidence["failed"] == 1 and evidence["pending"] == 0
    result = finish_session(tmp_path, "service_en", run_id=session["run_id"], persistence=evidence)
    assert result["status"] == "failed"
    with pytest.raises(ValueError, match="abnormally"):
        require_completed(tmp_path, "service_en")


def test_writer_overload_is_bounded_and_visible():
    release = threading.Event()
    pool = PersistenceExecutor(max_workers=1, max_pending=1)
    try:
        pool.submit(release.wait)
        refused = pool.submit(lambda: "must not run")
        with pytest.raises(RuntimeError, match="queue is full"):
            refused.result()
        assert pool.snapshot()["pending"] == 1
        assert pool.snapshot()["failures"][-1]["code"] == "queue_full"
    finally:
        release.set()
        pool.shutdown(wait=True)


def test_health_uses_real_capture_age_and_acknowledges_controls(tmp_path):
    health = PipelineHealth(tmp_path, "service_en")
    health.start(lambda: {"queues": {"finals": 3}, "clients": 2}, lambda op: setattr(health, "paused", op == "pause"))
    try:
        health.input(0.125)
        health.caption({"english": "Hello", "spanish_a": "Hola", "chunk_id": 1})
        health.error("stt", "RuntimeError")
        sequence = send_control(tmp_path, "service_en", "pause")
        until(lambda: read_health(tmp_path, "service_en").get("control_sequence") == sequence)
        snap = read_health(tmp_path, "service_en")
        assert snap["phase"] == "paused" and snap["queues"]["finals"] == 3
        assert snap["input_level"] == 0.125 and snap["error_count"] == 1
        assert snap["captions"][-1]["spanish_a"] == "Hola"
    finally:
        health.close("completed")
    assert read_health(tmp_path, "service_en", stale_after=-1)["stale"]


def test_work_reservation_excludes_duplicate_summary_and_start():
    lease = WorkLease()
    token = lease.acquire("summary", "one")
    with pytest.raises(WorkBusyError):
        lease.acquire("live session")
    lease.release("wrong-owner")
    with pytest.raises(WorkBusyError):
        lease.acquire("summary")
    lease.release(token)
    with lease.reserve("audio test"):
        assert lease.snapshot()["kind"] == "audio test"
    assert lease.snapshot() is None


def test_support_default_omits_speech_paths_and_secrets(tmp_path):
    complete(tmp_path)
    (tmp_path / "metrics" / "diagnostics_service_en.jsonl").write_text(
        '{"english":"PRIVATE SPEECH", "token":"SECRET"}\n'
    )
    req = SupportRequest(session_id="service_en")
    preview = preview_support(tmp_path, req)
    assert preview["files"] == ["metadata.json"]
    result = export_support(tmp_path, preview["preview_id"])
    with zipfile.ZipFile(tmp_path / "metrics" / "support" / f"{result['bundle_id']}.zip") as archive:
        assert archive.namelist() == ["metadata.json"]
        data = archive.read("metadata.json").decode()
        assert "PRIVATE SPEECH" not in data and "SECRET" not in data and str(tmp_path) not in data


def test_support_explicit_text_preview_detects_changed_files(tmp_path):
    complete(tmp_path)
    preview = preview_support(tmp_path, SupportRequest(session_id="service_en", include_text=True))
    assert any(name.endswith(".jsonl") for name in preview["files"])
    (tmp_path / "metrics" / "diagnostics_service_en.jsonl").write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        export_support(tmp_path, preview["preview_id"])
    assert not list((tmp_path / "metrics" / "support").glob("*.zip"))


def test_cleanup_requires_completion_and_preserves_originals(tmp_path):
    complete(tmp_path)
    original = tmp_path / "metrics" / "diagnostics_service_en.jsonl"
    log = tmp_path / "metrics" / "session_service_en.log"
    log.write_text("temporary log")
    preview = cleanup_preview(tmp_path, ["service_en"])
    result = cleanup(tmp_path, preview["preview_id"])
    assert result["removed_bytes"] == len("temporary log")
    assert original.is_file() and not log.exists()
    start_session(tmp_path, "active_en")
    with pytest.raises(ValueError, match="Finish"):
        cleanup_preview(tmp_path, ["active_en"])


def test_support_rejects_cross_root_and_symlink(tmp_path):
    complete(tmp_path)
    preview = preview_support(tmp_path, SupportRequest(session_id="service_en"))
    other = tmp_path / "other"
    other.mkdir()
    with pytest.raises(ValueError, match="another installation"):
        export_support(other, preview["preview_id"])
    audio = tmp_path / "stark_data" / "live_sessions" / "service_en"
    audio.mkdir(parents=True)
    (audio / "chunk_0001.wav").symlink_to(tmp_path / "metrics" / "diagnostics_service_en.jsonl")
    with pytest.raises(ValueError, match="symlinks"):
        preview_support(tmp_path, SupportRequest(session_id="service_en", include_audio=True))


def test_async_log_rotates_redacts_and_does_not_duplicate_handlers(tmp_path):
    logger = logging.getLogger("reliability-rotation")
    logger.propagate = False
    handler = configure_log(logger, tmp_path / "operator.log")
    assert configure_log(logger, tmp_path / "operator.log") is handler
    handler._sink.maxBytes = 250
    for _ in range(6):
        logger.warning("token=SECRET sample error")
    handler.close()
    logger.removeHandler(handler)
    files = list(tmp_path.glob("operator.log*"))
    assert 1 < len(files) <= 6
    for path in files:
        text = path.read_text()
        assert "SECRET" not in text
        for line in text.splitlines():
            assert json.loads(line)["schema_version"] == 1


def test_log_backpressure_counts_dropped_events(tmp_path):
    handler = AsyncOperationalHandler(tmp_path / "log", capacity=1)
    # Make a real slow writer; logging calls must remain nonblocking.
    gate = threading.Event()
    handler._sink.handle = lambda record: gate.wait(2)
    record = logging.LogRecord("test", logging.INFO, "", 0, "event", (), None)
    try:
        for _ in range(100):
            handler.emit(record)
        assert handler.dropped > 0 and handler.queue.qsize() <= 1
    finally:
        gate.set()
        handler.close()


def test_no_native_audio_device_open_in_probe_timeout(tmp_path):
    from tools.isolated_audio import AudioCaptureError, probe_audio

    with pytest.raises(AudioCaptureError, match="did not respond"):
        probe_audio("probe", duration_s=0.1, argv=[sys.executable, "-c", "import time; time.sleep(60)"])


def test_capture_pipe_preserves_adc_clock_and_sample_gaps(tmp_path):
    from tools.isolated_audio import IsolatedInputStream

    script = tmp_path / "fake_capture.py"
    script.write_text("""import json, struct, sys, time
for start, dropped in [(0, 0), (960, 480)]:
    received = time.perf_counter()
    header = json.dumps(dict(frames=480, channels=1, inputBufferAdcTime=90.0,
        currentTime=90.01, received=received, sample_start=start, dropped=dropped, status='')).encode()
    sys.stdout.buffer.write(struct.pack('!I', len(header)) + header + bytes(480*4))
    sys.stdout.buffer.flush()
time.sleep(60)
""")
    received = []
    with IsolatedInputStream(
        callback=lambda audio, frames, stamp, status: received.append((stamp, status)),
        samplerate=48000,
        channels=1,
        dtype="float32",
        blocksize=480,
        device=None,
        argv=[sys.executable, str(script)],
    ) as stream:
        until(lambda: len(received) == 2)
        assert received[0][0].source == "portaudio_adc"
        assert received[1][0].sample_start == 960
        assert received[1][1] == "capture_overflow:480"
        assert time.perf_counter() - received[1][0].end < 1
        pid = stream._proc.pid
    import psutil

    assert not psutil.pid_exists(pid)


def test_capture_no_samples_times_out_without_blocking_enter():
    from tools.isolated_audio import IsolatedInputStream

    started = time.monotonic()
    with IsolatedInputStream(
        callback=lambda *args: None,
        samplerate=48000,
        channels=1,
        dtype="float32",
        blocksize=480,
        device=None,
        startup_timeout=0.2,
        argv=[sys.executable, "-c", "import time; time.sleep(60)"],
    ) as stream:
        assert time.monotonic() - started < 0.15
        until(lambda: stream.finished.is_set())
        assert "no samples" in str(stream.error)


def test_required_health_failure_cannot_age_out_of_error_ring(tmp_path):
    health = PipelineHealth(tmp_path, "service_en")
    health.error("translation", "failed")
    for _ in range(100):
        health.error("audio", "initial_permission_error")
    assert len(health.snapshot()["errors"]) == 32
    assert health.snapshot()["recording"]["required_failures"] == 1


def test_parent_capture_handoff_has_one_bounded_pending_drain():
    from types import SimpleNamespace

    from tools.capture_handoff import CaptureHandoff

    callbacks, consumed, drops = [], [], []
    loop = SimpleNamespace(call_soon_threadsafe=callbacks.append, call_later=lambda delay, fn: callbacks.append(fn))
    with CaptureHandoff(loop, consumed.append, lambda: True, lambda: drops.append(1), capacity=4) as handoff:
        for sample in range(100):
            handoff.put(sample)
        assert len(callbacks) == 1 and handoff.qsize() == 4 and len(drops) == 96
        callbacks.pop()()
        assert consumed == [96, 97, 98, 99]


def test_file_capture_handoff_backpressures_and_close_unblocks_producer():
    from types import SimpleNamespace

    from tools.capture_handoff import CaptureHandoff

    loop = SimpleNamespace(call_soon_threadsafe=lambda fn: None)
    handoff = CaptureHandoff(loop, lambda item: None, lambda: False, lambda: None, capacity=1, wait_for_space=True)
    handoff.put(1)
    thread = threading.Thread(target=lambda: handoff.put(2))
    thread.start()
    try:
        assert thread.is_alive() and handoff.qsize() == 1
    finally:
        handoff.__exit__()
        thread.join(timeout=1)
    assert not thread.is_alive() and handoff.dropped == 1


def test_forced_runner_stop_cleans_child_group_and_reports_incomplete(tmp_path):
    from operator_app.pipeline_manager import PipelineRunner, SessionConfig

    (tmp_path / "metrics").mkdir()
    child_pid_path = tmp_path / "child.pid"
    (tmp_path / "dry_run_ab.py").write_text(f"""import subprocess, sys, time, signal
from pathlib import Path
signal.signal(signal.SIGTERM, signal.SIG_IGN)
child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
Path({str(child_pid_path)!r}).write_text(str(child.pid))
while True: time.sleep(1)
""")
    runner = PipelineRunner(tmp_path)
    try:
        runner.start(SessionConfig())
        until(child_pid_path.exists)
        child_pid = int(child_pid_path.read_text())
        result = runner.stop(timeout_s=0.5)
        assert result.outcome in {"interrupted", "failed"}
        assert result.last_event != "stopped cleanly"
        import psutil

        until(lambda: not psutil.pid_exists(child_pid) or psutil.Process(child_pid).status() == psutil.STATUS_ZOMBIE)
    finally:
        runner.stop(timeout_s=1)


def test_fresh_preflight_blocks_start_even_if_browser_previously_passed(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from operator_app.main import app
    from operator_app.pipeline_manager import PipelineRunner, get_runner

    runner = PipelineRunner(tmp_path)
    app.dependency_overrides[get_runner] = lambda: runner
    monkeypatch.setattr(
        "operator_app.main.run_all_checks", lambda **kw: {"ok": False, "checks": [{"name": "Models", "status": "fail"}]}
    )
    try:
        with TestClient(app) as client:
            result = client.post("/api/session/start", json={})
        assert result.status_code == 422
        assert result.json()["detail"]["code"] == "preflight_failed"
        assert runner._proc is None and runner._lease.snapshot() is None
    finally:
        app.dependency_overrides.clear()
        runner.stop(timeout_s=1)


def test_summary_owns_work_until_cancelled_child_is_reaped(tmp_path):
    from operator_app.features import SummaryTaskRunner
    from operator_app.pipeline_manager import PipelineRunner, SessionConfig

    script = tmp_path / "features" / "summarize_sermon.py"
    script.parent.mkdir()
    script.write_text("import time; time.sleep(60)")
    summary = SummaryTaskRunner(tmp_path)
    task = summary.submit(str(tmp_path / "source.csv"))
    live = PipelineRunner(tmp_path)
    try:
        until(lambda: task.task_id in summary._processes)
        child = summary._processes[task.task_id]
        with pytest.raises(WorkBusyError):
            live.start(SessionConfig())
        with pytest.raises(WorkBusyError):
            summary.submit(str(tmp_path / "source.csv"))
        summary.cancel(task.task_id)
        until(lambda: summary._lease.snapshot() is None)
        assert child.poll() is not None and task.state == "error"
    finally:
        summary.cancel(task.task_id)
        live.stop(timeout_s=1)


def test_source_provenance_rejects_installed_package_ancestor_checkout(tmp_path):
    import hashlib
    import subprocess

    from tools.session_lifecycle import source_provenance

    tracked = tmp_path / "dry_run_ab.py"
    tracked.write_text("# checkout runtime\n")
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "add", "dry_run_ab.py"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    checkout = source_provenance(tracked)
    assert checkout["git_sha"] and len(checkout["git_sha"]) == 40
    installed = tmp_path / ".cache" / "venv" / "site-packages" / "dry_run_ab.py"
    installed.parent.mkdir(parents=True)
    installed.write_text("# installed runtime from an older wheel\n")
    evidence = source_provenance(installed)
    assert evidence["git_sha"] is None
    assert evidence["pipeline_sha256"] == hashlib.sha256(installed.read_bytes()).hexdigest()
    assert evidence["pipeline_sha256"] != checkout["pipeline_sha256"]


def test_explicit_model_path_retains_setup_marker_revision(tmp_path, monkeypatch):
    from tools.session_lifecycle import completion_metadata

    model = tmp_path / "model"
    model.mkdir()
    (model / ".installed").write_text(json.dumps({"repo_id": "org/pinned-model", "revision": "a" * 40}))
    monkeypatch.setattr("engines.model_paths.load_model_manifest", lambda root: {"models": {}})
    monkeypatch.setattr("engines.model_paths.resolve_model_path", lambda *args, **kwargs: str(model))
    evidence = completion_metadata({"stt": str(model)}, tmp_path)["models"]["stt"]
    assert evidence["resolved_revision"] == "a" * 40
    assert evidence["source_repo_id"] == "org/pinned-model"
    assert evidence["revision_source"] == "setup_install_marker"


def test_capabilities_separate_supported_controls_from_readiness(monkeypatch):
    from fastapi.testclient import TestClient

    from operator_app.main import app
    from settings import settings

    monkeypatch.setattr(settings.server, "http_port", 18080)
    monkeypatch.setattr(settings.server, "ws_port", 18765)
    # No lifespan needed for a read-only API; never enumerate or open devices.
    result = TestClient(app).get("/api/capabilities")
    assert result.status_code == 200
    caps = result.json()
    assert "standard" in caps["profiles"] and caps["preflight_required"]
    assert caps["audio_tests"] and caps["audio_tests_require_idle"]
    assert not caps["audio_devices_validated"]
    assert caps["audience_urls"]["audience"] == "http://localhost:18080/displays/audience_display.html?port=18765"


def test_capture_handoff_recovers_after_consumer_error():
    from types import SimpleNamespace

    from tools.capture_handoff import CaptureHandoff

    callbacks, consumed, failures = [], [], []

    def consume(value):
        if value == "bad":
            raise ValueError("bad frame")
        consumed.append(value)

    loop = SimpleNamespace(call_soon_threadsafe=callbacks.append, call_later=lambda delay, fn: callbacks.append(fn))
    with CaptureHandoff(loop, consume, lambda: True, lambda: failures.append(1)) as handoff:
        handoff.put("bad")
        handoff.put("good")
        callbacks.pop()()
        assert failures == [1] and consumed == ["good"] and handoff.qsize() == 0


def test_paused_session_reports_final_health_outcome(tmp_path):
    health = PipelineHealth(tmp_path, "service_en")
    health.paused = True
    health.close("interrupted")
    assert read_health(tmp_path, "service_en")["phase"] == "interrupted"


def test_operator_uses_same_effective_display_ports_as_capabilities(tmp_path, monkeypatch):
    from operator_app.pipeline_manager import PipelineRunner, SessionConfig
    from settings import settings

    monkeypatch.setattr(settings.server, "http_port", 18080)
    monkeypatch.setattr(settings.server, "ws_port", 18765)
    argv = PipelineRunner(tmp_path)._build_argv(SessionConfig(), "service_en")
    assert argv[argv.index("--http-port") + 1] == "18080"
    assert argv[argv.index("--ws-port") + 1] == "18765"
    monkeypatch.setenv("STARK_PROFILE", "lite-cpu")
    argv = PipelineRunner(tmp_path)._build_argv(SessionConfig(profile="standard"), "service_en")
    assert argv[argv.index("--profile") + 1] == "standard"


def test_support_rejects_parent_symlink_inside_same_installation(tmp_path):
    complete(tmp_path)
    (tmp_path / "inside").mkdir()
    (tmp_path / "stark_data").symlink_to(tmp_path / "inside", target_is_directory=True)
    with pytest.raises(ValueError, match="symlinks"):
        preview_support(tmp_path, SupportRequest(session_id="service_en", include_audio=True))


@pytest.mark.parametrize(
    "target_name",
    [
        "diagnostics_service_en.jsonl",
        "session_lifecycle_service_en.json",
        "ab_metrics_service_en.csv",
        "summary_other_session.json",
    ],
)
def test_summary_web_output_cannot_replace_recording_evidence(tmp_path, monkeypatch, target_name):
    from unittest.mock import Mock

    from fastapi.testclient import TestClient

    from operator_app.main import app
    from operator_app.pipeline_manager import PipelineRunner, get_runner

    folder = tmp_path / "metrics"
    folder.mkdir()
    source = folder / "ab_metrics_service_en.csv"
    source.write_text("chunk_id,english\n1,Hello\n")
    target = folder / target_name
    if target != source:
        target.write_text("original evidence")
    original = target.read_bytes()
    summary = Mock()
    monkeypatch.setattr("operator_app.main.get_summary_runner", lambda **kwargs: summary)
    app.dependency_overrides[get_runner] = lambda: PipelineRunner(tmp_path)
    try:
        response = TestClient(app).post(
            "/api/features/summary", json={"csv_path": str(source), "output_path": str(target)}
        )
        assert response.status_code == 400
        summary.submit.assert_not_called()
        assert target.read_bytes() == original
    finally:
        app.dependency_overrides.pop(get_runner)


def test_summary_default_output_remains_compatible(tmp_path, monkeypatch):
    from unittest.mock import Mock

    from fastapi.testclient import TestClient

    from operator_app.features import SummaryTask
    from operator_app.main import app
    from operator_app.pipeline_manager import PipelineRunner, get_runner

    source = tmp_path / "metrics" / "ab_metrics_service_en.csv"
    source.parent.mkdir()
    source.write_text("chunk_id,english\n1,Hello\n")
    output = source.parent / "summary_service_en.json"
    summary = Mock()
    summary.submit.return_value = SummaryTask(task_id="task", csv_path=str(source), output_path=str(output))
    monkeypatch.setattr("operator_app.main.get_summary_runner", lambda **kwargs: summary)
    app.dependency_overrides[get_runner] = lambda: PipelineRunner(tmp_path)
    try:
        for body in ({"csv_path": str(source)}, {"csv_path": str(source), "output_path": str(output)}):
            response = TestClient(app).post("/api/features/summary", json=body)
            assert response.status_code == 200
            summary.submit.assert_called_with(csv_path=str(source), output_path=str(output))
    finally:
        app.dependency_overrides.pop(get_runner)


@pytest.mark.parametrize("route,body", [("lang_flip", {}), ("vad", {"threshold": 0.6}), ("fallback", {"engine": "hf"})])
def test_restarts_validate_before_stopping_working_session(tmp_path, monkeypatch, route, body):
    from unittest.mock import Mock

    from fastapi.testclient import TestClient

    from operator_app.main import app
    from operator_app.pipeline_manager import SessionConfig, SessionStatus, get_runner

    runner = Mock()
    runner._project_root = tmp_path
    runner.status.return_value = SessionStatus(state="running", config=SessionConfig().__dict__.copy())
    checks = Mock(return_value={"ok": False, "checks": [{"name": "Models", "status": "fail"}]})
    monkeypatch.setattr("operator_app.main.run_all_checks", checks)
    app.dependency_overrides[get_runner] = lambda: runner
    try:
        response = TestClient(app).post(f"/api/control/{route}", json=body)
        assert response.status_code == 422 and response.json()["detail"]["code"] == "preflight_failed"
        runner.restart_with.assert_not_called()
        runner.stop.assert_not_called()
        assert runner.status.return_value.state == "running"
        assert checks.call_args.kwargs["lang"] == ("es" if route == "lang_flip" else "en")
    finally:
        app.dependency_overrides.pop(get_runner)


def test_restart_cannot_claim_engine_rejected_by_effective_profile(tmp_path, monkeypatch):
    from unittest.mock import Mock

    from fastapi.testclient import TestClient

    from operator_app.main import app
    from operator_app.pipeline_manager import SessionConfig, SessionStatus, get_runner

    runner = Mock()
    runner._project_root = tmp_path
    runner.status.return_value = SessionStatus(
        state="running",
        config=SessionConfig(profile="lite-cpu-quality", backend="cpu", engine="llamacpp").__dict__.copy(),
    )

    def preflight(cfg, root):
        cfg.engine = "llamacpp"  # selected profile's canonical policy
        return {"ok": True, "checks": []}

    monkeypatch.setattr("operator_app.main._preflight_config", preflight)
    app.dependency_overrides[get_runner] = lambda: runner
    try:
        response = TestClient(app).post("/api/control/fallback", json={"engine": "hf"})
        assert response.status_code == 422 and response.json()["detail"]["code"] == "profile_conflict"
        runner.restart_with.assert_not_called()
    finally:
        app.dependency_overrides.pop(get_runner)
