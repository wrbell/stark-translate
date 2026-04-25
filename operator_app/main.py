"""FastAPI control plane (Phase 9.1).

Run with:
    uvicorn operator_app.main:app --host 0.0.0.0 --port 9000

The HTML/JS frontend is served from ``displays/operator/`` at ``/operator/``.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from operator_app.pipeline_manager import (
    PipelineRunner,
    SessionAlreadyRunningError,
    SessionConfig,
    get_runner,
)
from operator_app.preflight import run_all_checks

logger = logging.getLogger(__name__)


PROJECT_ROOT = Path(os.environ.get("STARK_PROJECT_ROOT", os.getcwd()))


app = FastAPI(
    title="stark-translate operator",
    version="0.1.0",
    description="Live pipeline control plane (Phase 9.1).",
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
    """Cheap liveness probe — does not touch the pipeline."""
    return {"status": "ok", "service": "stark-translate-operator"}


@app.get("/api/preflight")
def api_preflight() -> dict:
    """Run all preflight checks. Cheap enough to poll every few seconds."""
    return run_all_checks(project_root=PROJECT_ROOT)


@app.get("/api/devices")
def api_devices() -> dict:
    """Enumerate input audio devices via sounddevice."""
    try:
        import sounddevice as sd

        raw = sd.query_devices()
    except Exception as exc:
        return JSONResponse(status_code=503, content={"error": f"sounddevice unavailable: {exc}"})

    devices = []
    for idx, d in enumerate(raw):
        if d.get("max_input_channels", 0) > 0:
            devices.append(
                {
                    "index": idx,
                    "name": d.get("name", "?"),
                    "channels": d.get("max_input_channels", 0),
                    "default_sample_rate": d.get("default_samplerate", 0),
                }
            )
    return {"inputs": devices}


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


# -- static frontend ----------------------------------------------------------


_operator_static = PROJECT_ROOT / "displays" / "operator"
if _operator_static.exists():
    app.mount("/operator", StaticFiles(directory=str(_operator_static), html=True), name="operator")
else:
    logger.warning("displays/operator not found at %s — frontend will 404", _operator_static)
