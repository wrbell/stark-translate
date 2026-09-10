"""Endurance monitor contracts using fake processes and a tiny model-free child."""

import hashlib
import json
import socket
import subprocess
import sys
import threading
import time
import wave
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import psutil
import pytest

from tools import endurance_monitor as monitor


class FakeProcess:
    def __init__(self, pid, created, *, children=(), rss=100, cpu=0):
        self.pid, self.created, self.descendants = pid, created, list(children)
        self.rss, self.cpu, self.denied, self.state = rss, cpu, False, "running"
        self.args = ["python", "dry_run_ab.py", "--session-id", "session"]

    def create_time(self):
        return self.created

    def cmdline(self):
        return self.args

    def children(self, recursive):
        assert recursive
        return self.descendants

    def status(self):
        return self.state

    def memory_info(self):
        if self.denied:
            raise psutil.AccessDenied(self.pid)
        return SimpleNamespace(rss=self.rss)

    def cpu_times(self):
        return SimpleNamespace(user=self.cpu, system=0)


class FakePS:
    Error, NoSuchProcess = psutil.Error, psutil.NoSuchProcess
    STATUS_DEAD, STATUS_ZOMBIE = psutil.STATUS_DEAD, psutil.STATUS_ZOMBIE

    def __init__(self, *processes):
        self.processes = {p.pid: p for p in processes}

    def Process(self, pid):
        if pid not in self.processes:
            raise self.NoSuchProcess(pid)
        return self.processes[pid]


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))


def lifecycle(root, **updates):
    data = {
        "schema_version": 1,
        "session_id": "session",
        "run_id": "run",
        "pid": 42,
        "hostname": socket.gethostname(),
        "status": "running",
        "started_at": datetime.fromtimestamp(101, UTC).isoformat(),
    }
    data.update(updates)
    write_json(root / "metrics/session_lifecycle_session.json", data)
    return data


def row(chunk=1, start=0, end=100, **updates):
    data = {
        "session_id": "session",
        "chunk_id": chunk,
        "utterance_id": chunk,
        "sample_start": start,
        "sample_end": end,
        "sample_rate": 100,
        "input_audio_sha256": "source",
        "timing_schema_version": 2,
        "source_lang": "en",
        "target_lang": "es",
        "timing_source": "replay_realtime",
        "endpoint_reason": "silence",
        "speech_end_to_final_ms": 200,
        "timing_stages_ms": {"final_ready": 1000 + chunk * 1000},
    }
    data.update(updates)
    return data


def artifacts(root, *, finals=None, partials=(), completed=True, persistence=None):
    finals = [row()] if finals is None else finals
    metrics = root / "metrics"
    metrics.mkdir(parents=True, exist_ok=True)
    diagnostics = metrics / "diagnostics_session.jsonl"
    diagnostics.write_text("".join(json.dumps(item) + "\n" for item in finals))
    (metrics / "partials_session.jsonl").write_text("".join(json.dumps(item) + "\n" for item in partials))
    (metrics / "ab_metrics_session.csv").write_text(
        "chunk_id,timestamp\n" + "".join(f"{item['chunk_id']},now\n" for item in finals if not item.get("event"))
    )
    wav_path = root / "source.wav"
    with wave.open(str(wav_path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(100)
        wav.writeframes(b"\x00\x00" * 1000)
    write_json(
        metrics / "session_metadata_session.json",
        {
            "session_id": "session",
            "input_audio_path": str(wav_path),
            "input_audio_sha256": "source",
            "session_kind": "replay",
            "audio_source": "file",
        },
    )
    lifecycle(
        root,
        status="completed" if completed else "running",
        exit_code=0 if completed else None,
        diagnostics={
            "sha256": hashlib.sha256(diagnostics.read_bytes()).hexdigest(),
            "size_bytes": diagnostics.stat().st_size,
        },
        persistence=persistence if persistence is not None else {"ok": True, "pending": 0, "failed": 0},
        memory={"scope": "pipeline_process_lifetime", "peak_rss_bytes": 12345},
    )


@pytest.fixture
def identity(tmp_path):
    lifecycle(tmp_path)
    return monitor.attach(tmp_path, "session", 42, FakePS(FakeProcess(42, 100)))


@pytest.mark.parametrize(
    "updates", [{"pid": 43}, {"hostname": "remote"}, {"run_id": ""}, {"status": "completed"}, {"schema_version": 2}]
)
def test_attach_rejects_wrong_lifecycle(tmp_path, updates):
    lifecycle(tmp_path, **updates)
    with pytest.raises(monitor.OwnershipError):
        monitor.attach(tmp_path, "session", 42, FakePS(FakeProcess(42, 100)))


def test_attach_rejects_reused_pid_or_unrelated_command(tmp_path):
    lifecycle(tmp_path)
    with pytest.raises(monitor.OwnershipError, match="reuse"):
        monitor.attach(tmp_path, "session", 42, FakePS(FakeProcess(42, 110)))
    process = FakeProcess(42, 100)
    process.args = ["python", "dry_run_ab.py", "--session-id", "other"]
    with pytest.raises(monitor.OwnershipError, match="exact session"):
        monitor.attach(tmp_path, "session", 42, FakePS(process))
    process.args = ["python", "/installed/dry_run_ab.py", "--session-id=session"]
    assert monitor.attach(tmp_path, "session", 42, FakePS(process))["pid"] == 42


def test_tree_counts_owned_children_retains_reparented_and_never_reads_foreign(identity):
    child = FakeProcess(50, 105, rss=300)
    parent = FakeProcess(42, 100, children=[child], rss=200)
    unrelated = FakeProcess(77, 105, rss=99999)
    ps = FakePS(parent, child, unrelated)
    sampler = monitor.ProcessTreeSampler(identity, ps)
    first = sampler.sample(10)
    assert first["rss_sum_bytes"] == 500 and first["process_count"] == 2
    assert first["cpu_percent_sum"] is None
    parent.cpu, child.cpu = 2, 3
    second = sampler.sample(15)
    assert second["cpu_percent_sum"] == 100
    del ps.processes[42]
    orphan = sampler.sample(20)
    assert orphan["rss_sum_bytes"] == 300
    assert orphan["pipeline_alive"] is False
    assert orphan["events"][0]["exit_code"] is None
    # PID reuse must not add an unrelated replacement's RSS.
    ps.processes[50] = FakeProcess(50, 500, rss=99999)
    gone = sampler.sample(25)
    assert gone["rss_sum_bytes"] == 0
    assert any(event["event"] == "pid_reused" for event in gone["events"])


def test_access_denied_stays_unknown_and_does_not_double_cpu_interval(identity):
    process = FakeProcess(42, 100)
    sampler = monitor.ProcessTreeSampler(identity, FakePS(process))
    sampler.sample(10)
    process.denied = True
    missing = sampler.sample(15)
    assert missing["rss_sum_bytes"] is None and missing["pipeline_alive"] is None
    assert missing["errors"][0]["error"] == "AccessDenied"
    process.denied, process.cpu = False, 6
    assert sampler.sample(20)["cpu_percent_sum"] is None
    process.cpu = 7
    assert sampler.sample(25)["cpu_percent_sum"] == 20


def test_zombie_exit_is_observed_without_claiming_exit_code(identity):
    process = FakeProcess(42, 100)
    sampler = monitor.ProcessTreeSampler(identity, FakePS(process))
    sampler.sample(10)
    process.state = psutil.STATUS_ZOMBIE
    result = sampler.sample(15)
    assert result["process_count"] == 1 and result["rss_sum_bytes"] == 0
    assert result["running_process_count"] == 0 and result["exited_unreaped_count"] == 1
    assert result["pipeline_alive"] is False
    assert result["processes"][0]["exited_unreaped"] is True
    assert result["events"][0] == {
        "event": "exit_observed",
        "pid": 42,
        "status": psutil.STATUS_ZOMBIE,
        "exit_code": None,
    }


def test_health_redacts_caption_text_and_retains_unknowns(tmp_path):
    write_json(
        tmp_path / "metrics/health_session.json",
        {
            "schema_version": 1,
            "session_id": "session",
            "updated_at": 100,
            "phase": "ready",
            "captions": [{"chunk_id": 1, "stage": "complete", "english": "PRIVATE WORDS"}],
            "queues": {"audio": 3},
            "recording": {"required_failures": 1},
        },
    )
    health = monitor.health_snapshot(tmp_path, "session", 105)
    assert health["stale"] is True and health["age_s"] == 5
    assert "PRIVATE" not in json.dumps(health)
    assert "capture" not in health
    assert monitor.health_snapshot(tmp_path, "other", 105) == {"available": False, "stale": True}


def test_source_union_padding_overlap_and_unclassified_gaps():
    coverage = monitor.source_coverage(
        [row(start=100, end=300, padding_samples=100), row(start=200, end=400), row(start=700, end=800)],
        10,
        [{"start_s": 4, "end_s": 7, "label": "hymn (human annotation)"}],
    )
    assert coverage["covered_seconds"] == 4 and coverage["fraction"] == 0.4
    assert coverage["intervals_seconds"] == [[1, 4], [7, 8]]
    assert [gap["duration_s"] for gap in coverage["gaps"]] == [1, 3, 2]
    assert all(gap["classification"] == "unknown" for gap in coverage["gaps"])
    assert coverage["gaps"][1]["external_annotations"][0]["label"].startswith("hymn")
    unknown = monitor.source_coverage([{}, row(start=-1), row(end=1200)], 10)
    assert unknown["unknown_bounds_rows"] == 1 and unknown["invalid_bounds_rows"] == 2


def test_artifacts_completion_and_small_sample_percentiles(tmp_path, identity):
    finals = [
        row(1, start=100, end=200, speech_end_to_final_ms=2204),
        row(2, start=300, end=400, speech_end_to_final_ms=2044),
    ]
    partial1 = row(1, emitted_at_ms=1700, speech_start_to_partial_ms=700)
    partial2 = row(1, emitted_at_ms=1800, speech_start_to_partial_ms=800)
    artifacts(tmp_path, finals=finals, partials=[partial1, partial2])
    result = monitor.artifact_summary(tmp_path, "session", identity)
    assert result["completion"]["verified"] is True
    cohort = result["cohorts"][0]
    assert cohort["final_metrics_by_endpoint"]["silence"] == {"n": 2, "p50": 2124, "p95": 2204, "max": 2204}
    assert cohort["first_partial_speech_start_ms"]["n"] == 1
    assert cohort["first_partial_speech_start_ms"]["p50"] == 700
    assert cohort["server_update_times_ms"] == [1700, 1800, 2000, 3000]
    assert result["pipeline_lifetime_memory"]["scope"] == "pipeline_process_lifetime"
    assert "rss_sum_bytes" not in result["pipeline_lifetime_memory"]


@pytest.mark.parametrize(
    "persistence", [{"ok": False, "pending": 0, "failed": 1}, {"ok": True, "pending": 1, "failed": 0}, {}]
)
def test_completion_requires_required_persistence(tmp_path, identity, persistence):
    artifacts(tmp_path, persistence=persistence)
    assert monitor.artifact_summary(tmp_path, "session", identity)["completion"]["verified"] is False


def test_no_pooling_across_schema_clock_source_and_unknown_bounds(tmp_path, identity):
    artifacts(
        tmp_path,
        finals=[
            row(1),
            row(2, timing_source="replay_nonrealtime"),
            row(3, input_audio_sha256="other"),
            row(4, timing_schema_version=1),
        ],
    )
    result = monitor.artifact_summary(tmp_path, "session", identity)
    assert len(result["cohorts"]) == 4
    other = next(c for c in result["cohorts"] if c["identity"]["input_audio_sha256"] == "other")
    assert other["final_source_coverage"]["source_duration_seconds"] is None


def test_latest_summary_not_summed_and_incomplete_json_is_reported(tmp_path, identity):
    artifacts(
        tmp_path,
        finals=[
            row(),
            {"session": "session", "event": "session_summary", "chunks_attempted": 2},
            {"session": "session", "event": "session_summary", "chunks_attempted": 3},
        ],
    )
    path = tmp_path / "metrics/diagnostics_session.jsonl"
    with path.open("a") as stream:
        stream.write('{"unfinished":')
    result = monitor.artifact_summary(tmp_path, "session", identity)
    assert result["latest_session_summary"]["chunks_attempted"] == 3
    assert result["read_issues"]["malformed_rows"] == 1
    assert result["completion"]["diagnostics_hash_matches"] is False
    assert result["completion"]["verified"] is False


def test_monitor_stop_is_partial_preserves_pipeline_and_never_overwrites(tmp_path):
    artifacts(tmp_path, completed=False)
    ps = FakePS(FakeProcess(42, 100))
    stop = threading.Event()
    stop.set()
    output = tmp_path / "reports/one"
    before = {p: p.read_bytes() for p in (tmp_path / "metrics").iterdir()}
    result = monitor.monitor(tmp_path, "session", 42, output, stop_event=stop, ps=ps)
    assert result["monitor_status"] == "partial" and result["stop_reason"] == "monitor_stop_requested"
    assert result["measurements"]["rss_sum_bytes"]["max"] == 100
    assert before == {p: p.read_bytes() for p in before}
    assert result == json.loads((output / "report.json").read_text())
    with pytest.raises(FileExistsError):
        monitor.monitor(tmp_path, "session", 42, output, stop_event=stop, ps=ps)
    with pytest.raises(ValueError, match="separate"):
        monitor.monitor(tmp_path, "session", 42, tmp_path / "metrics/monitor", stop_event=stop, ps=ps)


def test_deadline_and_stop_file_control_only_monitor(tmp_path):
    artifacts(tmp_path, completed=False)
    ps = FakePS(FakeProcess(42, 100))
    result = monitor.monitor(tmp_path, "session", 42, tmp_path / "deadline", duration=0.05, interval=0.1, ps=ps)
    assert result["stop_reason"] == "deadline" and result["monitor_status"] == "partial"
    stop = tmp_path / "stop-monitor"
    stop.touch()
    result = monitor.monitor(tmp_path, "session", 42, tmp_path / "stopfile", stop_file=stop, ps=ps)
    assert result["stop_reason"] == "monitor_stop_requested" and stop.exists()


def test_marker_ownership_change_stops_sampling_with_error(tmp_path, monkeypatch):
    artifacts(tmp_path, completed=False)
    original = monitor.attach

    def change(*args):
        identity = original(*args)
        lifecycle(tmp_path, run_id="replacement")
        return identity

    monkeypatch.setattr(monitor, "attach", change)
    result = monitor.monitor(tmp_path, "session", 42, tmp_path / "out", ps=FakePS(FakeProcess(42, 100)))
    assert result["monitor_status"] == "error"
    assert result["monitor_error"]["type"] == "OwnershipError"
    assert result["measurements"]["sample_count"] == 0


def test_real_model_free_process_tree_exits_under_its_own_control(tmp_path):
    """No inference: child allocates a tiny buffer, self-exits, and publishes fixture completion."""
    try:
        psutil.Process().children(recursive=True)
    except (psutil.Error, OSError) as exc:
        pytest.skip(f"OS denies process-tree enumeration: {type(exc).__name__}")
    script = tmp_path / "dry_run_ab.py"
    script.write_text("""import datetime, hashlib, json, os, pathlib, socket, subprocess, sys, time
root = pathlib.Path(__file__).parent
metrics = root / "metrics"
metrics.mkdir()
child = subprocess.Popen([sys.executable, "-c", "import time; b=bytearray(1024*1024); time.sleep(.6)"])
data = {"schema_version": 1, "session_id": "session", "run_id": "real", "pid": os.getpid(), "hostname": socket.gethostname(), "status": "running", "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat()}
path = metrics / "session_lifecycle_session.json"
path.write_text(json.dumps(data))
time.sleep(.8)
child.wait(timeout=3)
(metrics / "diagnostics_session.jsonl").write_text("")
(metrics / "partials_session.jsonl").write_text("")
(metrics / "ab_metrics_session.csv").write_text("chunk_id\\n")
data.update(status="completed", exit_code=0, persistence={"ok": True, "pending": 0, "failed": 0}, diagnostics={"size_bytes": 0, "sha256": hashlib.sha256(b"").hexdigest()})
tmp = path.with_suffix(".tmp")
tmp.write_text(json.dumps(data))
tmp.replace(path)
""")
    process = subprocess.Popen([sys.executable, str(script), "--session-id", "session"])
    try:
        deadline = time.monotonic() + 3
        while not (tmp_path / "metrics/session_lifecycle_session.json").exists():
            assert time.monotonic() < deadline
            time.sleep(0.01)
        # The fixture owner reaps its own child, independently of the monitor.
        reaper = threading.Thread(target=process.wait, kwargs={"timeout": 3})
        reaper.start()
        result = monitor.monitor(
            tmp_path, "session", process.pid, tmp_path / "observed", duration=3, interval=0.1, exit_grace=1
        )
        reaper.join(timeout=3)
        assert process.returncode == 0
        assert result["monitor_status"] == "completed"
        samples = [json.loads(line) for line in (tmp_path / "observed/samples.jsonl").read_text().splitlines()]
        assert any(s["tree"]["process_count"] == 2 for s in samples)
        for sample in samples:
            if sample["tree"]["counters_complete"]:
                assert sample["tree"]["rss_sum_bytes"] == sum(p["rss_bytes"] for p in sample["tree"]["processes"])
        assert any(
            e["event"] in {"exit_observed", "process_disappeared_or_unreadable"}
            for e in result["measurements"]["events"]
        )
    finally:
        if process.poll() is None:
            process.terminate()  # Test owns only its fixture; monitor has no control calls.
            process.wait(timeout=3)


def test_import_has_no_model_modules_in_fresh_interpreter():
    code = "import sys; import tools.endurance_monitor; assert not any(x in sys.modules for x in ['torch','mlx','transformers','numpy','sounddevice'])"
    subprocess.run([sys.executable, "-c", code], check=True, cwd=Path(__file__).resolve().parents[1], timeout=5)


def test_os_tree_permission_failure_retains_readable_rows_but_not_total(identity):
    process = FakeProcess(42, 100)

    def denied(recursive):
        raise PermissionError("OS denied process enumeration")

    process.children = denied
    sampler = monitor.ProcessTreeSampler(identity, FakePS(process))
    result = sampler.sample(10)
    assert result["processes"][0]["rss_bytes"] == 100
    assert result["rss_sum_bytes"] is None
    assert result["pipeline_alive"] is None
    assert result["counters_complete"] is False
    assert result["errors"][0]["error"] == "PermissionError"


def test_completed_recording_with_zombie_remains_cleanup_unverified(tmp_path, monkeypatch):
    artifacts(tmp_path, completed=False)
    original = monitor.attach

    def complete_after_attach(*args):
        identity = original(*args)
        artifacts(tmp_path, completed=True)
        return identity

    monkeypatch.setattr(monitor, "attach", complete_after_attach)
    process = FakeProcess(42, 100)
    process.state = psutil.STATUS_ZOMBIE
    report = monitor.monitor(tmp_path, "session", 42, tmp_path / "out", exit_grace=0, ps=FakePS(process))
    assert report["artifacts"]["completion"]["verified"] is True
    assert report["monitor_status"] == "partial"
    assert report["measurements"]["no_running_processes_observed"] is True
    assert report["measurements"]["process_cleanup_observed"] is False
    assert report["measurements"]["exited_unreaped_count"]["max"] == 1


def test_null_legacy_timing_dict_remains_unknown(tmp_path, identity):
    artifacts(tmp_path, finals=[row(timing_stages_ms=None)])
    report = monitor.artifact_summary(tmp_path, "session", identity)
    assert report["cohorts"][0]["server_update_times_ms"] == []
    assert report["cohorts"][0]["server_update_gap_ms"]["p50"] is None
