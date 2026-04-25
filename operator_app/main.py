"""FastAPI control plane (Phase 9.1+).

Run with:
    uvicorn operator_app.main:app --host 0.0.0.0 --port 9000

The HTML/JS frontend is served from ``displays/operator/`` at ``/operator/``.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import logging.handlers
import os
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from operator_app.audio import get_watcher
from operator_app.features import get_summary_runner, get_verse_watcher
from operator_app.metrics import get_collector, healthz_snapshot
from operator_app.pipeline_manager import (
    InvalidStateError,
    PipelineRunner,
    SessionAlreadyRunningError,
    SessionConfig,
    get_runner,
)
from operator_app.preflight import run_all_checks

logger = logging.getLogger(__name__)


PROJECT_ROOT = Path(os.environ.get("STARK_PROJECT_ROOT", os.getcwd()))


def _configure_logging() -> None:
    """Wire a rotating file handler so multi-day sessions don't grow unbounded.

    Honors STARK_OPERATOR_LOG_DIR if set; otherwise writes under metrics/.
    Rotates at 100 MiB, keeps 5 backups. The console handler stays at the
    root logger's existing level (defaults to WARNING).
    """
    log_dir = Path(os.environ.get("STARK_OPERATOR_LOG_DIR", str(PROJECT_ROOT / "metrics")))
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        return
    handler = logging.handlers.RotatingFileHandler(
        log_dir / "operator.log",
        maxBytes=100 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    handler.setLevel(logging.INFO)
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    )
    root = logging.getLogger()
    if not any(getattr(h, "baseFilename", "").endswith("operator.log") for h in root.handlers):
        root.addHandler(handler)
    if root.level > logging.INFO:
        root.setLevel(logging.INFO)


_configure_logging()


@contextlib.asynccontextmanager
async def _lifespan(app: FastAPI):
    """Graceful startup + shutdown.

    On shutdown we stop any running pipeline subprocess so SIGTERM/SIGINT
    (the systemd ``Restart=always`` flow) doesn't leave orphaned children
    or partially-flushed CSVs.
    """
    logger.info("operator app starting up")
    yield
    logger.info("operator app shutting down — stopping pipeline if running")
    try:
        runner = get_runner()
        if runner.status().state != "idle":
            runner.stop(timeout_s=10.0)
    except Exception as exc:
        logger.warning("graceful pipeline stop failed: %s", exc)


app = FastAPI(
    title="stark-translate operator",
    version="0.1.0",
    description="Live pipeline control plane (Phase 9).",
    lifespan=_lifespan,
)


# -- request models -----------------------------------------------------------


class StartRequest(BaseModel):
    """Subset of ``SessionConfig`` the frontend exposes."""

    lang: str = Field(default="en", pattern="^(en|es)$")
    backend: str = Field(default="auto", pattern="^(auto|mlx|cuda|cpu)$")
    engine: str = Field(default="auto", pattern="^(auto|llamacpp|hf)$")
    tts: bool = False
    run_ab: bool = False
    vad_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    mic_device: int | None = None
    mic_gain: float | None = None
    log_level: str = Field(default="INFO", pattern="^(DEBUG|INFO|WARNING|ERROR)$")


# -- endpoints ----------------------------------------------------------------


@app.get("/healthz")
def healthz() -> dict:
    """Liveness + light-weight resource snapshot for external probes."""
    return healthz_snapshot()


@app.get("/api/preflight")
def api_preflight() -> dict:
    """Run all preflight checks. Cheap enough to poll every few seconds."""
    return run_all_checks(project_root=PROJECT_ROOT)


@app.get("/api/devices")
def api_devices() -> dict:
    """Enumerate input + output audio devices, plus the change_seq counter.

    Frontend reads ``change_seq`` from the metrics WS frame and re-fetches
    this endpoint when it bumps; that's the USB-hotplug toast trigger.
    """
    listing = get_watcher().force_poll()
    body = listing.to_dict()
    body["change_seq"] = get_watcher().snapshot()["change_seq"]
    if listing.error:
        return JSONResponse(status_code=503, content=body)
    return body


@app.get("/api/session/status")
def api_session_status(runner: PipelineRunner = Depends(get_runner)) -> dict:
    return runner.status().to_dict()


@app.post("/api/session/start")
def api_session_start(req: StartRequest, runner: PipelineRunner = Depends(get_runner)) -> dict:
    cfg = SessionConfig(**req.model_dump())
    try:
        snap = runner.start(cfg)
    except SessionAlreadyRunningError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return snap.to_dict()


@app.post("/api/session/stop")
def api_session_stop(runner: PipelineRunner = Depends(get_runner)) -> dict:
    snap = runner.stop()
    return snap.to_dict()


# -- mid-session controls (Phase 9.3) -----------------------------------------


class VadRequest(BaseModel):
    threshold: float = Field(ge=0.0, le=1.0)


class FallbackRequest(BaseModel):
    """Switch the live engine. Restarts the subprocess with new args."""

    engine: str = Field(pattern="^(auto|llamacpp|hf)$")


@app.post("/api/control/pause")
def api_control_pause(runner: PipelineRunner = Depends(get_runner)) -> dict:
    try:
        snap = runner.pause()
    except InvalidStateError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return snap.to_dict()


@app.post("/api/control/resume")
def api_control_resume(runner: PipelineRunner = Depends(get_runner)) -> dict:
    try:
        snap = runner.resume()
    except InvalidStateError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return snap.to_dict()


@app.post("/api/control/lang_flip")
def api_control_lang_flip(runner: PipelineRunner = Depends(get_runner)) -> dict:
    """Flip EN↔ES. Stop+restart with the inverted lang field."""
    snap = runner.status()
    if snap.state == "idle" or snap.config is None:
        raise HTTPException(status_code=409, detail="no active session to flip")
    cfg = SessionConfig(**snap.config)
    cfg.lang = "es" if cfg.lang == "en" else "en"
    return runner.restart_with(cfg).to_dict()


@app.post("/api/control/vad")
def api_control_vad(req: VadRequest, runner: PipelineRunner = Depends(get_runner)) -> dict:
    """Update VAD threshold. Stop+restart with new threshold."""
    snap = runner.status()
    if snap.state == "idle" or snap.config is None:
        raise HTTPException(status_code=409, detail="no active session to retune")
    cfg = SessionConfig(**snap.config)
    cfg.vad_threshold = req.threshold
    return runner.restart_with(cfg).to_dict()


@app.post("/api/control/fallback")
def api_control_fallback(req: FallbackRequest, runner: PipelineRunner = Depends(get_runner)) -> dict:
    """Emergency engine swap (e.g. llamacpp → hf if llama-server crashed)."""
    snap = runner.status()
    if snap.state == "idle" or snap.config is None:
        raise HTTPException(status_code=409, detail="no active session to swap")
    cfg = SessionConfig(**snap.config)
    cfg.engine = req.engine
    return runner.restart_with(cfg).to_dict()


# -- features (Phase 9.6) -----------------------------------------------------


class SummaryRequest(BaseModel):
    csv_path: str | None = None
    output_path: str | None = None


@app.get("/api/features/verses")
def api_features_verses(
    since_chunk: int | None = None,
    runner: PipelineRunner = Depends(get_runner),
) -> dict:
    """Verse references found in the live transcript so far.

    Binds the watcher lazily to whichever session CSV is current. When no
    session is active and no historical watcher exists, returns an empty
    list rather than 404.
    """
    snap = runner.status()
    csv_path = snap.csv_path
    if csv_path:
        watcher = get_verse_watcher(csv_path=csv_path, project_root=PROJECT_ROOT)
    else:
        watcher = get_verse_watcher()
    if watcher is None:
        return {"highlights": [], "since_chunk": since_chunk}
    highlights = watcher.snapshot(since_chunk=since_chunk)
    return {"highlights": highlights, "since_chunk": since_chunk}


@app.post("/api/features/summary")
def api_features_summary(
    req: SummaryRequest,
    runner: PipelineRunner = Depends(get_runner),
) -> dict:
    """Trigger the post-session summary subprocess.

    Defaults csv_path to the current session's CSV (whether running or
    just-finished) and output_path to a sibling JSON. Returns immediately
    with a task_id; poll ``GET /api/features/summary/{id}``.
    """
    snap = runner.status()
    csv_path = req.csv_path or snap.csv_path
    if not csv_path:
        raise HTTPException(status_code=400, detail="no csv_path available — pass one or start a session first")
    if not Path(csv_path).exists():
        raise HTTPException(status_code=404, detail=f"csv_path does not exist: {csv_path}")
    task = get_summary_runner(project_root=PROJECT_ROOT).submit(csv_path=csv_path, output_path=req.output_path)
    return task.to_dict()


@app.get("/api/features/summary/{task_id}")
def api_features_summary_status(task_id: str) -> dict:
    runner = get_summary_runner(project_root=PROJECT_ROOT)
    task = runner.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"task {task_id} not found")
    return task.to_dict()


@app.get("/api/metrics")
def api_metrics() -> dict:
    """Pull-mode metrics snapshot. Same shape as /ws/control frames."""
    return get_collector().snapshot()


@app.websocket("/ws/control")
async def ws_control(websocket: WebSocket) -> None:
    """Push live metrics frames to the operator UI at ~1 Hz.

    Frame shape mirrors ``MetricsCollector.snapshot()``. Frontend renders
    sparklines from ``resources.vram_mib_recent`` and the latency aggregates.
    """
    await websocket.accept()
    collector = get_collector()
    try:
        while True:
            await websocket.send_text(json.dumps(collector.snapshot()))
            await asyncio.sleep(1.0)
    except WebSocketDisconnect:
        return
    except Exception as exc:
        logger.warning("/ws/control closed unexpectedly: %s", exc)
        try:
            await websocket.close()
        except Exception:
            pass


# -- static frontend ----------------------------------------------------------


_operator_static = PROJECT_ROOT / "displays" / "operator"
if _operator_static.exists():
    app.mount("/operator", StaticFiles(directory=str(_operator_static), html=True), name="operator")
else:
    logger.warning("displays/operator not found at %s — frontend will 404", _operator_static)
