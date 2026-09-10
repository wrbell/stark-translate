"""Lightweight native stdout drain, rotation, privacy and retention contracts."""

import io
import json
import logging
import os
import subprocess
import sys
import threading

from stark_translate.profiles import resolve_profile
from tools.llama_runtime import ManagedLlamaServer
from tools.operational_logging import AsyncOperationalHandler, prune_completed_logs


def test_native_pipe_keeps_draining_when_disk_writer_is_stalled(tmp_path, monkeypatch):
    owner = ManagedLlamaServer(resolve_profile("lite-cpu-quality"), log_path=tmp_path / "llama_live_en.log")
    handler = AsyncOperationalHandler(owner.log_path, capacity=2, private=True, prune_backups=False)
    entered, release = threading.Event(), threading.Event()
    seen = []

    def stalled(record):
        seen.append(record)
        entered.set()
        release.wait(5)

    monkeypatch.setattr(handler._sink, "handle", stalled)
    child = subprocess.Popen(
        [sys.executable, "-c", "import os; os.write(1,b'x'*1048576); os.write(1,b'\\ntail')"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=0,
    )
    reader = threading.Thread(target=owner._capture_output, args=(child.stdout, handler), daemon=True)
    try:
        reader.start()
        assert entered.wait(2)
        # Child emits much more than pipe capacity while the disk writer is stuck.
        assert child.wait(timeout=3) == 0
        reader.join(timeout=1)
        assert not reader.is_alive()
        assert handler.snapshot()["dropped"] > 0
        assert handler.snapshot()["queued"] <= 2
    finally:
        release.set()
        if child.poll() is None:
            child.kill()
            child.wait(timeout=2)
        reader.join(timeout=1)
        handler.close()
    assert all(len(record.getMessage().encode()) <= 4096 for record in seen)
    assert owner._read_failures == 0


def test_private_native_rotation_caps_files_and_preserves_unproven_old_backup(tmp_path):
    path = tmp_path / "llama_active_en.log"
    old = path.with_suffix(".log.5")
    old.write_text("unproven session evidence")
    os.utime(old, (1, 1))
    handler = AsyncOperationalHandler(path, max_bytes=1024, backups=5, private=True, prune_backups=False)
    assert old.exists()  # Age alone cannot prune interrupted/unknown sessions.
    for _ in range(100):
        handler.handle(logging.LogRecord("llama", logging.INFO, "", 0, "native " + "x" * 80, (), None))
    handler.close()
    files = list(tmp_path.glob("llama_active_en.log*"))
    assert len(files) == 6
    assert all(p.stat().st_size <= 1024 for p in files)
    if os.name != "nt":
        assert all(p.stat().st_mode & 0o777 == 0o600 for p in files)
    assert handler.snapshot()["write_failures"] == 0
    assert json.loads(path.read_text().splitlines()[-1])["logger"] == "llama"


def test_native_drain_records_tail_and_read_failure_without_unbounded_line(tmp_path):
    owner = ManagedLlamaServer(resolve_profile("lite-cpu-quality"), log_path=tmp_path / "llama_service_en.log")
    handler = AsyncOperationalHandler(owner.log_path, private=True, prune_backups=False)
    owner._capture_output(io.BytesIO(b"start\n" + b"z" * 12000 + b"\nlast"), handler)
    handler.close()
    messages = [json.loads(line) for line in owner.log_path.read_text().splitlines()]
    assert messages[0]["message"] == "start"
    assert messages[-1]["message"] == "last"
    assert all(row["session_id"] == "service_en" for row in messages)
    assert all(row["event"] == "native_output" for row in messages)


def test_retention_prunes_native_logs_only_for_completed_sessions(tmp_path, monkeypatch):
    metrics = tmp_path / "metrics"
    metrics.mkdir()
    for name in [
        "llama_done_en.log",
        "llama_done_en.log.5",
        "session_done_en.log",
        "llama_active_en.log.1",
        "llama_unknown_en.log",
        "llama_done_en.log.6",
        "diagnostics_done_en.jsonl",
    ]:
        path = metrics / name
        path.write_text("preserve unrelated evidence")
        os.utime(path, (1, 1))
    recent = metrics / "llama_done_en.log.2"
    recent.write_text("recent")
    monkeypatch.setattr(
        "tools.session_lifecycle.session_status", lambda root, session: {"exportable": session == "done_en"}
    )
    assert prune_completed_logs(tmp_path) == 3
    assert sorted(p.name for p in metrics.iterdir()) == sorted(
        [
            "llama_active_en.log.1",
            "llama_unknown_en.log",
            "llama_done_en.log.6",
            "diagnostics_done_en.jsonl",
            "llama_done_en.log.2",
        ]
    )
