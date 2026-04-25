"""Pipeline lifecycle owned by the operator FastAPI app.

The current ``dry_run_ab.main_async`` runs the pipeline as a one-shot CLI
process. Phase 9.1 wraps it in a ``PipelineRunner`` so the FastAPI app can
start/stop sessions in-process from REST endpoints without subprocessing.

Threading model: the pipeline runs in a single background thread that owns
its own asyncio event loop. The FastAPI handlers communicate via thread-safe
status queries and a stop flag — they never touch the pipeline loop directly.

Subprocess isolation can come later (Phase 9.5 if needed for crash recovery);
for now, in-process keeps the live-metrics path zero-copy.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


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


@dataclass
class SessionStatus:
    """Snapshot returned by ``/api/session/status``."""

    state: str  # "idle" | "starting" | "running" | "stopping" | "error"
    session_id: str | None = None
    started_at: str | None = None
    stopped_at: str | None = None
    error: str | None = None
    config: dict | None = None
    last_event: str | None = None

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v is not None or k == "state"}


class PipelineRunner:
    """Owns a single ``dry_run_ab`` session at a time.

    Methods are thread-safe: FastAPI worker threads can call them concurrently.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._status = SessionStatus(state="idle")
        self._config: SessionConfig | None = None

    # -- public API -----------------------------------------------------------

    def start(self, config: SessionConfig) -> SessionStatus:
        """Start a new session. Returns immediately; check ``status()`` for state."""
        with self._lock:
            if self._status.state in ("starting", "running"):
                raise SessionAlreadyRunningError(self._status.session_id)

            self._stop_event.clear()
            session_id = f"{datetime.now():%Y%m%d_%H%M%S}_{config.lang}"
            self._config = config
            self._status = SessionStatus(
                state="starting",
                session_id=session_id,
                started_at=datetime.now().isoformat(timespec="seconds"),
                config=config.__dict__.copy(),
                last_event="thread launched",
            )
            self._thread = threading.Thread(
                target=self._run,
                args=(config, session_id),
                name=f"pipeline-{session_id}",
                daemon=True,
            )
            self._thread.start()
            return self._snapshot()

    def stop(self, timeout_s: float = 10.0) -> SessionStatus:
        """Signal the running session to stop and wait briefly for shutdown."""
        with self._lock:
            if self._status.state == "idle":
                return self._snapshot()
            self._status.state = "stopping"
            self._status.last_event = "stop requested"
            self._stop_event.set()

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
            return self._snapshot()

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
        )

    # -- internals ------------------------------------------------------------

    def _run(self, config: SessionConfig, session_id: str) -> None:
        """Pipeline thread entry point."""
        try:
            with self._lock:
                self._status.state = "running"
                self._status.last_event = "pipeline thread running"

            # Phase 9.1 placeholder: simulate a session by sleeping until stop.
            # Phase 9.3+ will replace this with the actual dry_run_ab.main_async
            # invocation, refactored to accept a stop-event hook.
            #
            # We deliberately avoid importing dry_run_ab at module load time
            # because it pulls in heavy ML deps that block FastAPI startup.
            # Real wiring lives behind ``_drive_dry_run`` in 9.3.
            self._drive_pipeline_stub(config, session_id)

            with self._lock:
                if self._status.state == "running":
                    self._status.last_event = "pipeline thread exited normally"
        except Exception as exc:
            logger.exception("pipeline session %s crashed", session_id)
            with self._lock:
                self._status.state = "error"
                self._status.error = f"{type(exc).__name__}: {exc}"
                self._status.last_event = "pipeline crashed"

    def _drive_pipeline_stub(self, config: SessionConfig, session_id: str) -> None:
        """Phase 9.1 placeholder: poll the stop event until told to halt.

        Real pipeline wiring is deferred to Phase 9.3 (mid-session controls)
        because it requires refactoring ``dry_run_ab.main_async`` to accept a
        stop hook and surface live events. This stub lets 9.1 ship with
        end-to-end exercise of the FastAPI lifecycle without dragging the
        pipeline refactor into the same PR.
        """
        del config, session_id  # used by future _drive_dry_run
        while not self._stop_event.wait(timeout=0.5):
            pass


class SessionAlreadyRunningError(RuntimeError):
    def __init__(self, session_id: str | None) -> None:
        super().__init__(f"session {session_id} already running")
        self.session_id = session_id


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
