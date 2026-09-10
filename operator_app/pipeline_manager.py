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
import math
import os
import platform
import subprocess
import sys
import threading
import time
import weakref
from dataclasses import dataclass, field
from datetime import datetime
from functools import wraps
from pathlib import Path

from operator_app.processes import cleanup_children, descendants
from operator_app.work_lease import get_work_lease
from tools.pipeline_health import read_health, send_control
from tools.session_lifecycle import session_status

logger = logging.getLogger(__name__)

_IS_WINDOWS = platform.system() == "Windows"
_instances = weakref.WeakSet()


def _number(record: dict, *keys: str) -> float | None:
    for key in keys:
        if record.get(key) not in (None, ""):
            try:
                value = float(record[key])
                return value if math.isfinite(value) else None
            except (TypeError, ValueError):
                return None
    return None


def parse_metrics_row(record: dict) -> dict:
    """Map real and historical CSV schemas without inventing missing measurements."""
    version = record.get("timing_schema_version") or "legacy"
    if str(version) != "legacy":
        basis = "speech_end_to_final_ms"
        total = _number(record, basis)
    elif "e2e_latency_ms" in record:
        basis = "legacy_e2e_latency_ms"
        total = _number(record, "e2e_latency_ms")
    else:
        basis = "legacy_total_ms"
        total = _number(record, "latency_ms", "total_ms")
    return {
        "chunk_id": int(record["chunk_id"]),
        "stt_ms": _number(record, "stt_latency_ms", "stt_ms"),
        "translate_ms": _number(record, "latency_a_ms", "translate_ms"),
        "total_ms": total,
        "confidence": _number(record, "stt_confidence", "confidence"),
        "text_len": len(str(record.get("english") or "")),
        "timing_schema_version": str(version),
        "latency_basis": basis,
    }


@dataclass
class SessionConfig:
    """Subset of dry_run_ab args the operator needs to set per session."""

    lang: str = "en"
    profile: str = field(default_factory=lambda: os.environ.get("STARK_PROFILE", "standard"))
    record_audio: bool = True
    stt_backend: str = "auto"
    model_family: str | None = None
    gemma4_size: str | None = None
    low_vram: bool = False
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
    diarize: bool = False
    tts_device_en: int | str | None = None
    tts_device_es: int | str | None = None


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
    outcome: str | None = None
    readiness: dict | None = None
    health: dict | None = None
    work: dict | None = None
    effective_profile: dict | None = None

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v is not None or k == "state"}


def _guarded_control(method):
    """Serialize controls without letting a queued request adopt a new session."""

    @wraps(method)
    def guarded(self, *args, **kwargs):
        # Never hold the state lock while waiting for the operation lock: the
        # subprocess watcher needs it while Stop joins that watcher thread.
        with self._lock:
            generation = self._generation
        with self._control_lock:
            with self._lock:
                if self._generation != generation:
                    raise InvalidStateError("Session changed while this control was waiting; refresh before retrying")
            return method(self, *args, **kwargs)

    return guarded


class PipelineRunner:
    """Owns a single ``dry_run_ab`` subprocess at a time.

    Methods are thread-safe: FastAPI worker threads can call them concurrently.
    """

    PROCESS_POLL_INTERVAL_S = 0.5
    CSV_TAIL_INTERVAL_S = 0.5

    def __init__(self, project_root: Path | None = None) -> None:
        _instances.add(self)
        self._lock = threading.RLock()
        self._control_lock = threading.RLock()
        self._generation = 0
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._status = SessionStatus(state="idle")
        self._config: SessionConfig | None = None
        self._proc: subprocess.Popen | None = None
        self._project_root = project_root or Path(os.environ.get("STARK_PROJECT_ROOT", os.getcwd()))
        self._lease = get_work_lease(self._project_root)
        self._lease_token = None
        self._owned_children = {}
        self._log_thread = None
        self._log_handler = None

    # -- public API -----------------------------------------------------------

    def start(self, config: SessionConfig) -> SessionStatus:
        with self._control_lock, self._lock:
            if (
                self._status.state in ("starting", "running", "paused", "stopping")
                or (self._thread is not None and self._thread.is_alive())
                or (self._proc is not None and self._proc.poll() is None)
            ):
                raise SessionAlreadyRunningError(self._status.session_id)

            self._lease_token = self._lease.acquire("live session")
            self._generation += 1
            self._owned_children = {}
            self._stop_event.clear()
            self._proc = None
            session_id = f"{datetime.now():%Y%m%d_%H%M%S_%f}_{config.lang}"
            csv_path = str(self._project_root / "metrics" / f"ab_metrics_{session_id}.csv")
            log_path = str(self._project_root / "metrics" / f"session_{session_id}.log")
            self._config = config
            from operator_app.metrics import get_collector

            get_collector().reset_session(session_id)
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
            try:
                self._thread.start()
            except BaseException:
                self._lease.release(self._lease_token)
                self._lease_token = None
                self._status.state, self._status.outcome = "error", "failed"
                raise
            if config.diarize:
                try:
                    from operator_app.features import get_diarize_watcher

                    jsonl = self._project_root / "metrics" / f"diarization_{session_id}.jsonl"
                    get_diarize_watcher(jsonl_path=jsonl, csv_path=csv_path)
                except Exception:
                    logger.warning("failed to bind live diarization watcher", exc_info=True)
            return self._snapshot()

    @_guarded_control
    def stop(self, timeout_s: float = 10.0) -> SessionStatus:
        with self._lock:
            if self._status.state == "idle":
                return self._snapshot()
            self._status.state = "stopping"
            self._status.last_event = "stop requested"
            self._stop_event.set()
            proc = self._proc

        forced = False
        if proc is not None and proc.poll() is None:
            self._owned_children.update(descendants(proc.pid))
            # Cooperative commands work on Windows and let asyncio drain active
            # translations. A process still loading may not have a command loop.
            health = read_health(self._project_root, self._status.session_id)
            send_control(self._project_root, self._status.session_id, "stop")
            if not health.get("stale") and health.get("phase") in {"ready", "paused", "listening", "input_error"}:
                try:
                    proc.wait(timeout=max(0.2, timeout_s - 2))
                except subprocess.TimeoutExpired:
                    pass
            if proc.poll() is None:
                try:
                    proc.terminate()
                    proc.wait(timeout=min(2.0, max(0.2, timeout_s / 2)))
                except ProcessLookupError:
                    pass
                except subprocess.TimeoutExpired:
                    forced = True
                    proc.kill()
                    proc.wait(timeout=2.0)
            cleanup_children(proc.pid, self._owned_children)

        if self._thread is not None:
            self._thread.join(timeout=timeout_s)

        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                self._status.state = "error"
                self._status.error = f"pipeline did not stop within {timeout_s}s"
            else:
                self._status.state = "idle"
                self._status.stopped_at = datetime.now().isoformat(timespec="seconds")
                lifecycle = (
                    session_status(self._project_root, self._status.session_id) if self._status.session_id else {}
                )
                self._status.outcome = "completed" if lifecycle.get("exportable") else "interrupted"
                if lifecycle.get("status") == "failed" and not forced:
                    self._status.outcome = "failed"
                if self._status.error:
                    self._status.outcome = "failed"
                self._status.last_event = (
                    "Session saved and completed"
                    if self._status.outcome == "completed"
                    else "Session stopped; recording is incomplete"
                    if self._status.outcome == "interrupted"
                    else "Session stopped with errors; review is available, export is blocked"
                )
                self._status.pid = None
            return self._snapshot()

    @_guarded_control
    def pause(self) -> SessionStatus:
        with self._lock:
            if self._status.state != "running":
                raise InvalidStateError(f"cannot pause from state={self._status.state}")
            send_control(self._project_root, self._status.session_id, "pause")
            self._status.last_event = "Pause requested; waiting for pipeline acknowledgment"
            return self._snapshot()

    @_guarded_control
    def resume(self) -> SessionStatus:
        with self._lock:
            if self._status.state != "paused":
                raise InvalidStateError(f"cannot resume from state={self._status.state}")
            send_control(self._project_root, self._status.session_id, "resume")
            self._status.last_event = "Resume requested; waiting for pipeline acknowledgment"
            return self._snapshot()

    @_guarded_control
    def restart_with(self, config: SessionConfig) -> SessionStatus:
        """Stop the current session and start a new one with the new config."""
        if self._status.state != "idle":
            self.stop()
        return self.start(config)

    def status(self) -> SessionStatus:
        with self._lock:
            return self._snapshot()

    def _snapshot(self) -> SessionStatus:
        health = read_health(self._project_root, self._status.session_id) if self._status.session_id else {}
        phase = health.get("phase", "idle")
        stale = health.get("stale", False)
        if self._log_handler is not None:
            health["operational_logging"] = self._log_handler.snapshot()
        readiness = {
            "phase": phase,
            "ready": phase == "ready" and not stale,
            "reason": "Pipeline health is unavailable" if stale else phase.replace("_", " "),
            "updated_at": health.get("updated_at"),
            "age_s": health.get("age_s"),
            "stale": stale,
        }
        if not stale and self._status.state in {"starting", "running", "paused"}:
            if phase == "paused":
                self._status.state = "paused"
            elif phase == "ready":
                self._status.state = "running"
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
            outcome=self._status.outcome,
            readiness=readiness,
            health=health,
            work=self._lease.snapshot(),
            effective_profile={"name": self._config.profile} if self._config else None,
        )

    # -- internals ------------------------------------------------------------

    def _build_argv(self, config: SessionConfig, session_id: str | None = None) -> list[str]:
        """Translate ``SessionConfig`` to a ``dry_run_ab.py`` invocation."""
        script = self._project_root / "dry_run_ab.py"
        if not script.is_file():
            script = Path(__file__).resolve().parent.parent / "dry_run_ab.py"
        argv = [
            sys.executable,
            "-u",
            str(script),
            "--lang",
            config.lang,
            "--backend",
            config.backend,
            "--vad-threshold",
            str(config.vad_threshold),
            "--log-level",
            config.log_level,
        ]
        from settings import settings

        argv += ["--http-port", str(settings.server.http_port), "--ws-port", str(settings.server.ws_port)]
        if session_id is not None:
            argv += ["--session-id", session_id]
        for option_name in ("stt_backend", "model_family", "gemma4_size"):
            value = getattr(config, option_name)
            if value and value != "auto":
                argv += ["--" + option_name.replace("_", "-"), str(value)]
        if config.low_vram:
            argv.append("--low-vram")
        # Explicit selection must override STARK_PROFILE inherited by the child.
        argv += ["--profile", config.profile]
        if not config.record_audio:
            argv += ["--no-record-audio"]
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
            for language in ("en", "es"):
                device = getattr(config, f"tts_device_{language}")
                if device is not None:
                    argv += [f"--tts-device-{language}", str(device)]
        if config.mic_device is not None:
            argv += ["--device", str(config.mic_device)]
        if config.mic_gain is not None:
            argv += ["--gain", str(config.mic_gain)]
        if config.diarize:
            argv += ["--diarize"]
        return argv

    def _run(self, config: SessionConfig, session_id: str) -> None:
        try:
            argv = self._build_argv(config, session_id)
            logger.info("spawning pipeline: %s", " ".join(argv))
            try:
                log_path = Path(self._status.log_path)
                log_path.parent.mkdir(parents=True, exist_ok=True)
                # Stop must either cancel the launch or see the registered child.
                with self._lock:
                    if self._stop_event.is_set():
                        return
                    # Capture failures before the pipeline's logger is initialized.
                    env = dict(os.environ, STARK_OPERATOR_CAPTURE="1")
                    proc = subprocess.Popen(
                        argv,
                        cwd=str(self._project_root),
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        errors="replace",
                        start_new_session=True,
                        env=env,
                    )
                    self._proc = proc
                    self._status.pid = proc.pid
                    from tools.operational_logging import AsyncOperationalHandler

                    self._log_handler = AsyncOperationalHandler(log_path)

                    def capture_output():
                        try:
                            while line := proc.stdout.readline(4096):
                                record = logging.LogRecord("pipeline", logging.INFO, "", 0, line.rstrip(), (), None)
                                record.session_id = session_id
                                record.event = "pipeline_output"
                                self._log_handler.handle(record)
                        finally:
                            proc.stdout.close()

                    self._log_thread = threading.Thread(target=capture_output, name="pipeline-output", daemon=True)
                    self._log_thread.start()
                    self._proc = proc
                    self._status.pid = proc.pid
                    self._status.last_event = "loading models; waiting for pipeline metrics header"
            except OSError as exc:
                with self._lock:
                    self._status.state = "error"
                    self._status.error = f"failed to spawn: {exc}"
                    self._status.last_event = "spawn failed"
                return

            # Tail the session CSV in this same thread; doubles as a poll loop
            # for the subprocess so we notice crashes and surface them.
            self._tail_metrics_csv(proc, session_id)

            return_code = proc.poll()
            with self._lock:
                if self._stop_event.is_set():
                    self._status.last_event = f"subprocess exited (rc={return_code}) after stop"
                elif return_code == 0:
                    self._status.state = "idle"
                    lifecycle = session_status(self._project_root, session_id)
                    completed = lifecycle.get("exportable")
                    self._status.outcome = (
                        "completed"
                        if completed
                        else ("failed" if lifecycle.get("status") == "failed" else "interrupted")
                    )
                    self._status.last_event = (
                        "Session completed" if completed else "Process exited without complete recording evidence"
                    )
                    self._status.stopped_at = datetime.now().isoformat(timespec="seconds")
                else:
                    self._status.state = "error"
                    self._status.outcome = "failed"
                    self._status.error = f"subprocess exited rc={return_code} unexpectedly"
                    try:
                        with log_path.open("rb") as output:
                            output.seek(max(0, log_path.stat().st_size - 1500))
                            detail = output.read().decode("utf-8", errors="replace").strip()
                        if detail:
                            self._status.error += f"\n{detail}"
                    except OSError:
                        pass
                    self._status.last_event = f"subprocess crashed rc={return_code}"
        except Exception as exc:
            logger.exception("pipeline session %s crashed in runner", session_id)
            with self._lock:
                self._status.state = "error"
                self._status.error = f"{type(exc).__name__}: {exc}"
                self._status.last_event = "runner thread crashed"

        finally:
            if self._proc is not None:
                if self._proc.poll() is None:
                    self._proc.kill()
                    self._proc.wait(timeout=2)
                cleanup_children(self._proc.pid, self._owned_children)
            if self._log_thread is not None:
                self._log_thread.join(timeout=2)
            if self._log_handler is not None:
                self._log_handler.close()
            if self._lease_token is not None:
                self._lease.release(self._lease_token)
                self._lease_token = None

    def _tail_metrics_csv(self, proc: subprocess.Popen, session_id: str) -> None:
        """Tail the session's ab_metrics CSV; feed each row to MetricsCollector."""
        from operator_app.metrics import get_collector

        collector = get_collector()
        csv_path = Path(self._status.csv_path) if self._status.csv_path else None
        f = None
        reader = None
        header: list[str] | None = None

        try:
            while proc.poll() is None:
                self._owned_children.update(descendants(proc.pid))
                collector.record_health(read_health(self._project_root, session_id))
                if f is None and csv_path is not None and csv_path.exists():
                    f = csv_path.open("r")
                    reader = csv.reader(f)
                if reader is not None and header is None:
                    # init_csv() runs after model loading. File creation alone is
                    # insufficient: the producer may not have flushed its header.
                    offset = f.tell()
                    line = f.readline()
                    if not line.endswith("\n"):
                        f.seek(offset)
                    else:
                        candidate = next(csv.reader([line]))
                        if "chunk_id" in candidate and any(
                            key in candidate for key in ("stt_latency_ms", "stt_ms", "e2e_latency_ms", "total_ms")
                        ):
                            header = candidate
                            with self._lock:
                                if self._status.state == "starting":
                                    self._status.last_event = "Models loaded; waiting for audio readiness"

                if reader is not None and header is not None:
                    advanced = False
                    for row in reader:
                        advanced = True
                        if len(row) != len(header):
                            continue
                        record = dict(zip(header, row, strict=False))
                        try:
                            collector.record_segment(**parse_metrics_row(record))
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
        runners = list(_instances)
        _runner = None
    for runner in runners:
        runner.stop(timeout_s=2.0)
