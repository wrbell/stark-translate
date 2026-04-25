"""Pipeline lifecycle owned by the operator FastAPI app.

Phase 9.3: ``PipelineRunner`` subprocesses ``dry_run_ab.py`` directly rather
than running it in-process. Tradeoffs:

- Pro: zero refactor to ``dry_run_ab`` (3800+ LOC); crashes don't take down
  the operator UI; pause/resume via SIGSTOP/SIGCONT is one syscall;
  mid-session config changes are stop+restart and don't require hot-reload
  plumbing inside the pipeline.
- Con: per-segment metrics flow via tailing the CSV the pipeline writes
  rather than direct method calls. ~50–500 ms of latency on the live
  observability stream — acceptable for a 1 Hz dashboard.

The thread that owns the subprocess also tails ``metrics/ab_metrics_*.csv``
as rows land, parses each segment, and feeds ``MetricsCollector`` so the
9.2 sparklines actually move once a session starts.
"""

from __future__ import annotations

import csv
import logging
import os
import platform
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

_IS_WINDOWS = platform.system() == "Windows"


@dataclass
class SessionConfig:
    """Subset of dry_run_ab args the operator needs to set per session."""

    lang: str = "en"
    backend: str = "auto"
    engine: str = "auto"
    tts: bool = False
    run_ab: bool = False
    vad_threshold: float = 0.3
    mic_device: int | None = None
    mic_gain: float | None = None
    log_level: str = "INFO"
    # Phase 9.4.1: TTS output device routing
    tts_output_mode: str = "ws"  # "ws" | "wav" | "both" | "local"
    tts_device: int | None = None


@dataclass
class SessionStatus:
    """Snapshot returned by ``/api/session/status``."""

    state: str  # "idle" | "starting" | "running" | "paused" | "stopping" | "error"
    session_id: str | None = None
    started_at: str | None = None
    stopped_at: str | None = None
    error: str | None = None
    config: dict | None = None
    last_event: str | None = None
    pid: int | None = None
    csv_path: str | None = None
    log_path: str | None = None

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v is not None or k == "state"}


class PipelineRunner:
    """Owns a single ``dry_run_ab`` subprocess at a time.

    Methods are thread-safe: FastAPI worker threads can call them concurrently.
    """

    PROCESS_POLL_INTERVAL_S = 0.5
    CSV_TAIL_INTERVAL_S = 0.5
    STARTUP_GRACE_S = 3.0  # how long we wait for the CSV file to appear

    def __init__(self, project_root: Path | None = None) -> None:
        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._status = SessionStatus(state="idle")
        self._config: SessionConfig | None = None
        self._proc: subprocess.Popen | None = None
        self._project_root = project_root or Path(os.environ.get("STARK_PROJECT_ROOT", os.getcwd()))

    # -- public API -----------------------------------------------------------

    def start(self, config: SessionConfig) -> SessionStatus:
        with self._lock:
            if self._status.state in ("starting", "running", "paused"):
                raise SessionAlreadyRunningError(self._status.session_id)

            self._stop_event.clear()
            session_id = f"{datetime.now():%Y%m%d_%H%M%S}_{config.lang}"
            csv_path = str(self._project_root / "metrics" / f"ab_metrics_{session_id}.csv")
            log_path = str(self._project_root / "metrics" / f"session_{session_id}.log")
            self._config = config
            self._status = SessionStatus(
                state="starting",
                session_id=session_id,
                started_at=datetime.now().isoformat(timespec="seconds"),
                config=config.__dict__.copy(),
                last_event="subprocess launching",
                csv_path=csv_path,
                log_path=log_path,
            )
            self._thread = threading.Thread(
                target=self._run, args=(config, session_id), name=f"pipeline-{session_id}", daemon=True
            )
            self._thread.start()
            return self._snapshot()

    def stop(self, timeout_s: float = 10.0) -> SessionStatus:
        with self._lock:
            if self._status.state == "idle":
                return self._snapshot()
            self._status.state = "stopping"
            self._status.last_event = "stop requested"
            self._stop_event.set()
            proc = self._proc

        # Politely SIGTERM first; SIGKILL after grace period if needed.
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=max(2.0, timeout_s - 2.0))
            except subprocess.TimeoutExpired:
                logger.warning("subprocess did not exit on SIGTERM, killing")
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass

        if self._thread is not None:
            self._thread.join(timeout=timeout_s)

        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                self._status.state = "error"
                self._status.error = f"pipeline did not stop within {timeout_s}s"
            else:
                self._status.state = "idle"
                self._status.stopped_at = datetime.now().isoformat(timespec="seconds")
                self._status.last_event = "stopped cleanly"
                self._status.pid = None
            return self._snapshot()

    def pause(self) -> SessionStatus:
        """SIGSTOP the subprocess. No-op on Windows."""
        with self._lock:
            if self._status.state != "running":
                raise InvalidStateError(f"cannot pause from state={self._status.state}")
            proc = self._proc
            if _IS_WINDOWS:
                self._status.last_event = "pause not supported on Windows"
                return self._snapshot()
            if proc is not None and proc.poll() is None:
                try:
                    os.kill(proc.pid, signal.SIGSTOP)
                    self._status.state = "paused"
                    self._status.last_event = "SIGSTOP sent"
                except ProcessLookupError:
                    self._status.last_event = "process gone before pause"
            return self._snapshot()

    def resume(self) -> SessionStatus:
        with self._lock:
            if self._status.state != "paused":
                raise InvalidStateError(f"cannot resume from state={self._status.state}")
            proc = self._proc
            if _IS_WINDOWS:
                self._status.last_event = "resume not supported on Windows"
                return self._snapshot()
            if proc is not None and proc.poll() is None:
                try:
                    os.kill(proc.pid, signal.SIGCONT)
                    self._status.state = "running"
                    self._status.last_event = "SIGCONT sent"
                except ProcessLookupError:
                    self._status.last_event = "process gone before resume"
            return self._snapshot()

    def restart_with(self, config: SessionConfig) -> SessionStatus:
        """Stop the current session and start a new one with the new config."""
        if self._status.state != "idle":
            self.stop()
        return self.start(config)

    def status(self) -> SessionStatus:
        with self._lock:
            return self._snapshot()

    def _snapshot(self) -> SessionStatus:
        return SessionStatus(
            state=self._status.state,
            session_id=self._status.session_id,
            started_at=self._status.started_at,
            stopped_at=self._status.stopped_at,
            error=self._status.error,
            config=dict(self._status.config) if self._status.config else None,
            last_event=self._status.last_event,
            pid=self._status.pid,
            csv_path=self._status.csv_path,
            log_path=self._status.log_path,
        )

    # -- internals ------------------------------------------------------------

    def _build_argv(self, config: SessionConfig) -> list[str]:
        """Translate ``SessionConfig`` to a ``dry_run_ab.py`` invocation."""
        argv = [
            sys.executable,
            "-u",
            str(self._project_root / "dry_run_ab.py"),
            "--lang",
            config.lang,
            "--backend",
            config.backend,
            "--vad-threshold",
            str(config.vad_threshold),
            "--log-level",
            config.log_level,
        ]
        if config.engine != "auto":
            argv += ["--engine", config.engine]
        if config.run_ab:
            argv += ["--ab"]
        else:
            argv += ["--no-ab"]
        if config.tts:
            argv += ["--tts"]
            argv += ["--tts-output", config.tts_output_mode]
            if config.tts_device is not None:
                argv += ["--tts-device", str(config.tts_device)]
        if config.mic_device is not None:
            argv += ["--device", str(config.mic_device)]
        if config.mic_gain is not None:
            argv += ["--gain", str(config.mic_gain)]
        return argv

    def _run(self, config: SessionConfig, session_id: str) -> None:
        try:
            argv = self._build_argv(config)
            logger.info("spawning pipeline: %s", " ".join(argv))
            try:
                proc = subprocess.Popen(
                    argv,
                    cwd=str(self._project_root),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
            except OSError as exc:
                with self._lock:
                    self._status.state = "error"
                    self._status.error = f"failed to spawn: {exc}"
                    self._status.last_event = "spawn failed"
                return

            with self._lock:
                self._proc = proc
                self._status.pid = proc.pid
                self._status.state = "running"
                self._status.last_event = "subprocess running"

            # Tail the session CSV in this same thread; doubles as a poll loop
            # for the subprocess so we notice crashes and surface them.
            self._tail_metrics_csv(proc, session_id)

            return_code = proc.poll()
            with self._lock:
                if self._stop_event.is_set():
                    self._status.last_event = f"subprocess exited (rc={return_code}) after stop"
                elif return_code == 0:
                    self._status.state = "idle"
                    self._status.last_event = "subprocess exited cleanly"
                    self._status.stopped_at = datetime.now().isoformat(timespec="seconds")
                else:
                    self._status.state = "error"
                    self._status.error = f"subprocess exited rc={return_code} unexpectedly"
                    self._status.last_event = f"subprocess crashed rc={return_code}"
        except Exception as exc:
            logger.exception("pipeline session %s crashed in runner", session_id)
            with self._lock:
                self._status.state = "error"
                self._status.error = f"{type(exc).__name__}: {exc}"
                self._status.last_event = "runner thread crashed"

    def _tail_metrics_csv(self, proc: subprocess.Popen, session_id: str) -> None:
        """Tail the session's ab_metrics CSV; feed each row to MetricsCollector."""
        from operator_app.metrics import get_collector

        collector = get_collector()
        csv_path = Path(self._status.csv_path) if self._status.csv_path else None
        deadline = time.time() + self.STARTUP_GRACE_S
        f = None
        reader = None
        header: list[str] | None = None

        try:
            while not self._stop_event.is_set() and proc.poll() is None:
                if f is None and csv_path is not None and csv_path.exists():
                    f = csv_path.open("r")
                    reader = csv.reader(f)
                    try:
                        header = next(reader)
                    except StopIteration:
                        header = None

                if f is None and time.time() > deadline:
                    # CSV never appeared — pipeline still running but not emitting.
                    # Keep polling but stop logging the warning every loop.
                    pass

                if reader is not None and header is not None:
                    advanced = False
                    for row in reader:
                        advanced = True
                        if len(row) != len(header):
                            continue
                        record = dict(zip(header, row, strict=False))
                        try:
                            collector.record_segment(
                                chunk_id=int(float(record.get("chunk_id", 0) or 0)),
                                stt_ms=float(record.get("stt_ms") or 0),
                                translate_ms=float(record.get("translate_ms") or 0),
                                total_ms=float(record.get("latency_ms") or record.get("total_ms") or 0),
                                confidence=float(record.get("confidence") or 0),
                                text_len=len(str(record.get("english", "") or "")),
                            )
                        except (ValueError, TypeError):
                            collector.record_error()
                    if not advanced:
                        time.sleep(self.CSV_TAIL_INTERVAL_S)
                else:
                    time.sleep(self.PROCESS_POLL_INTERVAL_S)
        finally:
            if f is not None:
                try:
                    f.close()
                except Exception:
                    pass
            del session_id  # quiet unused-var lint


class SessionAlreadyRunningError(RuntimeError):
    def __init__(self, session_id: str | None) -> None:
        super().__init__(f"session {session_id} already running")
        self.session_id = session_id


class InvalidStateError(RuntimeError):
    """Raised when a control op (pause/resume) doesn't fit current state."""


# Module-level singleton — FastAPI handlers grab this once.
_runner: PipelineRunner | None = None
_runner_lock = threading.Lock()


def get_runner() -> PipelineRunner:
    global _runner
    with _runner_lock:
        if _runner is None:
            _runner = PipelineRunner()
        return _runner


def reset_runner_for_tests() -> None:
    """Test helper — never call from production code."""
    global _runner
    with _runner_lock:
        if _runner is not None:
            try:
                _runner.stop(timeout_s=2.0)
            except Exception:
                pass
        _runner = None
