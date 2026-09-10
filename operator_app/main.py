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

from fastapi import Depends, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from engines.audio_devices import list_output_devices
from operator_app.audio import get_watcher
from operator_app.audio_ingest import get_bus as get_audio_bus
from operator_app.audio_ingest import handle_audio_ingest, handle_audio_subscribe
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
from operator_app.review import router as review_router
from operator_app.work_lease import WorkBusyError

logger = logging.getLogger(__name__)


PROJECT_ROOT = Path(os.environ.get("STARK_PROJECT_ROOT", os.getcwd()))


def _configure_logging() -> None:
    from tools.operational_logging import configure_log, prune_completed_logs

    log_dir = Path(os.environ.get("STARK_OPERATOR_LOG_DIR", str(PROJECT_ROOT / "metrics")))
    configure_log(logging.getLogger(), log_dir / "operator.log")
    prune_completed_logs(PROJECT_ROOT)


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
    get_summary_runner(project_root=get_runner()._project_root).close()


app = FastAPI(
    title="stark-translate operator",
    version="0.1.0",
    description="Live pipeline control plane (Phase 9).",
    lifespan=_lifespan,
)
app.include_router(review_router)
from operator_app.support import router as support_router

app.include_router(support_router)
from operator_app.audio_tests import router as audio_test_router

app.include_router(audio_test_router)


# -- request models -----------------------------------------------------------


class StartRequest(BaseModel):
    """Subset of ``SessionConfig`` the frontend exposes."""

    lang: str = Field(default="en", pattern="^(en|es)$")
    profile: str = Field(
        default_factory=lambda: os.environ.get("STARK_PROFILE", "standard"),
        pattern="^(standard|lite-cpu|lite-cuda-8gb|lite-cpu-quality)$",
    )
    record_audio: bool = True
    stt_backend: str = Field(default="auto", pattern="^(auto|mlx|parakeet-mlx|faster-whisper|hf|parakeet-nemo)$")
    model_family: str | None = Field(default=None, pattern="^(gemma4|translategemma)$")
    gemma4_size: str | None = Field(default=None, pattern="^(e2b|e4b)$")
    low_vram: bool = False
    backend: str = Field(default="auto", pattern="^(auto|mlx|cuda|cpu)$")
    engine: str = Field(default="auto", pattern="^(auto|llamacpp|hf)$")
    tts: bool = False
    run_ab: bool = False
    vad_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    mic_device: int | None = None
    mic_gain: float | None = None
    log_level: str = Field(default="INFO", pattern="^(DEBUG|INFO|WARNING|ERROR)$")
    # Phase 9.4.1: TTS output device routing
    tts_output_mode: str = Field(default="ws", pattern="^(ws|wav|both|local)$")
    tts_device: int | None = None
    diarize: bool = False
    tts_device_en: int | str | None = None
    tts_device_es: int | str | None = None


# -- endpoints ----------------------------------------------------------------


@app.get("/healthz")
def healthz() -> dict:
    """Liveness + light-weight resource snapshot for external probes."""
    return healthz_snapshot()


@app.get("/api/capabilities")
def api_capabilities(request: Request) -> dict:
    """Supported controls, not a claim that models or physical devices are ready."""
    from settings import settings

    try:
        from stark_translate.profiles import PROFILE_NAMES

        profiles = list(PROFILE_NAMES)
    except ImportError:
        profiles = ["standard"]
    host = request.url.hostname or "localhost"
    if ":" in host:
        host = "[" + host + "]"
    http_port, ws_port = settings.server.http_port, settings.server.ws_port
    base = f"{request.url.scheme}://{host}:{http_port}"
    return {
        "profiles": profiles,
        "default_profile": os.environ.get("STARK_PROFILE", "standard"),
        "preflight_required": True,
        "audio_tests": True,
        "audio_tests_require_idle": True,
        "audio_devices_validated": False,
        "support": True,
        "storage": True,
        "display_ports": {"http": http_port, "websocket": ws_port},
        "audience_urls": {
            name: f"{base}/displays/{filename}?port={ws_port}"
            for name, filename in {
                "audience": "audience_display.html",
                "church": "church_display.html",
                "mobile": "mobile_display.html",
                "obs": "obs_overlay.html",
            }.items()
        },
    }


def _resolve_session_config(cfg):
    try:
        from stark_translate.profiles import session_overrides
    except ImportError:
        if cfg.profile != "standard":
            raise HTTPException(
                status_code=422,
                detail={"code": "profile_unavailable", "message": "Install the selected runtime profile"},
            ) from None
        return {}
    try:
        values = session_overrides(cfg.profile, cfg.backend)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    for name, value in values.items():
        if hasattr(cfg, name):
            setattr(cfg, name, value)
    return {"profile": cfg.profile}


def _preflight_config(cfg, root):
    profile_args = _resolve_session_config(cfg)
    return run_all_checks(
        project_root=root,
        backend=cfg.backend,
        lang=cfg.lang,
        tts=cfg.tts,
        diarize=cfg.diarize,
        input_device=cfg.mic_device,
        stt_backend=cfg.stt_backend,
        model_family=cfg.model_family or "gemma4",
        gemma4_size=cfg.gemma4_size or "e4b",
        **profile_args,
    )


@app.get("/api/preflight")
def api_preflight(
    backend: str = "auto",
    lang: str = "en",
    tts: bool = False,
    diarize: bool = False,
    input_device: int | None = None,
    profile: str | None = None,
    runner: PipelineRunner = Depends(get_runner),
) -> dict:
    cfg = SessionConfig(
        backend=backend,
        lang=lang,
        tts=tts,
        diarize=diarize,
        mic_device=input_device,
        profile=profile or os.environ.get("STARK_PROFILE", "standard"),
    )
    checks = _preflight_config(cfg, runner._project_root)
    checks["effective_profile"] = cfg.__dict__.copy()
    return checks


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


@app.get("/api/audio/output-devices")
def api_output_devices() -> dict:
    """Output devices with the current system default marked."""
    try:
        return {"outputs": list_output_devices()}
    except Exception as exc:
        return JSONResponse(status_code=503, content={"outputs": [], "error": str(exc)})


@app.get("/api/session/status")
def api_session_status(runner: PipelineRunner = Depends(get_runner)) -> dict:
    return runner.status().to_dict()


@app.post("/api/session/start")
def api_session_start(req: StartRequest, runner: PipelineRunner = Depends(get_runner)) -> dict:
    cfg = SessionConfig(**req.model_dump())
    if runner.status().state in {"starting", "running", "paused", "stopping"}:
        raise HTTPException(status_code=409, detail="A session is already running")
    checks = _preflight_config(cfg, runner._project_root)
    if not checks["ok"]:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "preflight_failed",
                "message": "The selected configuration is not ready. Resolve the failed checks before starting.",
                "checks": checks["checks"],
            },
        )
    try:
        snap = runner.start(cfg)
    except WorkBusyError as exc:
        raise HTTPException(
            status_code=409, detail={"code": "work_busy", "message": str(exc), "work": exc.work}
        ) from exc
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
    watcher.force_scan()
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
    if snap.state in ("starting", "running", "paused", "stopping"):
        raise HTTPException(status_code=409, detail="Stop the live session before generating a summary")
    csv_path = req.csv_path or snap.csv_path
    if not csv_path:
        raise HTTPException(status_code=400, detail="no csv_path available — pass one or start a session first")
    if not Path(csv_path).exists():
        raise HTTPException(status_code=404, detail=f"csv_path does not exist: {csv_path}")
    root = runner._project_root.resolve()
    source = Path(csv_path).resolve()
    output = Path(req.output_path).resolve() if req.output_path else None
    if source.parent != root / "metrics" or (output and output.parent != root / "metrics"):
        raise HTTPException(
            status_code=400, detail="Summary paths must stay inside this installation's metrics directory"
        )
    try:
        task = get_summary_runner(project_root=root).submit(
            csv_path=str(source), output_path=str(output) if output else None
        )
    except WorkBusyError as exc:
        raise HTTPException(
            status_code=409, detail={"code": "work_busy", "message": str(exc), "work": exc.work}
        ) from exc
    return task.to_dict()


@app.post("/api/features/summary/{task_id}/cancel")
def cancel_summary(task_id: str, runner: PipelineRunner = Depends(get_runner)):
    try:
        return get_summary_runner(project_root=runner._project_root).cancel(task_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Summary task not found") from exc


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


@app.get("/metrics", response_class=PlainTextResponse)
def metrics_prometheus() -> str:
    """Prometheus exposition format. Wraps the same snapshot the WS frame uses."""
    return _render_prometheus(get_collector().snapshot())


def _render_prometheus(snap: dict) -> str:
    """Translate the operator metrics snapshot into Prometheus text format.

    Kept inline (no prometheus_client dep) — the schema is small and stable,
    and we don't want to pull a 300 KB lib for ~10 gauges.
    """
    lines: list[str] = []

    def gauge(name: str, help_text: str, value: float) -> None:
        lines.append(f"# HELP {name} {help_text}")
        lines.append(f"# TYPE {name} gauge")
        lines.append(f"{name} {value}")

    def counter(name: str, help_text: str, value: float) -> None:
        lines.append(f"# HELP {name} {help_text}")
        lines.append(f"# TYPE {name} counter")
        lines.append(f"{name} {value}")

    gauge("stark_uptime_seconds", "Operator process uptime in seconds.", float(snap.get("uptime_s") or 0))
    gauge("stark_queue_depth", "Pending inference jobs queued at last sample.", float(snap.get("queue_depth") or 0))
    counter("stark_errors_total", "Total pipeline errors since process start.", float(snap.get("error_count") or 0))

    res = snap.get("resources", {}) or {}
    gauge("stark_vram_mib", "Current GPU 0 VRAM usage in MiB (nvidia-smi).", float(res.get("vram_mib_current") or 0))
    gauge(
        "stark_cpu_percent",
        "Whole-system CPU percent (psutil, last interval).",
        float(res.get("cpu_percent_current") or 0),
    )

    lat = snap.get("latency", {}) or {}
    gauge(
        "stark_latency_total_ms_p50",
        "Median end-to-end segment latency (recent ring).",
        float(lat.get("total_ms_p50") or 0),
    )
    gauge(
        "stark_latency_total_ms_p95",
        "p95 end-to-end segment latency (recent ring).",
        float(lat.get("total_ms_p95") or 0),
    )
    gauge("stark_latency_stt_ms_p50", "Median STT latency (recent ring).", float(lat.get("stt_ms_p50") or 0))
    gauge(
        "stark_latency_translate_ms_p50",
        "Median translate latency (recent ring).",
        float(lat.get("translate_ms_p50") or 0),
    )
    gauge("stark_confidence_mean", "Mean confidence over recent segments.", float(lat.get("confidence_mean") or 0))
    gauge("stark_segments_recent", "Number of segments in the recent ring buffer.", float(lat.get("n") or 0))

    audio = snap.get("audio", {}) or {}
    counter(
        "stark_audio_device_change_seq",
        "Monotonic counter of audio device topology changes (USB hotplug etc.).",
        float(audio.get("change_seq") or 0),
    )

    return "\n".join(lines) + "\n"


@app.get("/api/audio_ingest")
def api_audio_ingest_status() -> dict:
    """Stats on the /ws/audio/ingest endpoint — useful for the operator UI
    to show whether the audio-bridge container is live."""
    return get_audio_bus().snapshot()


@app.websocket("/ws/audio/ingest")
async def ws_audio_ingest(websocket: WebSocket) -> None:
    """Receive PCM audio frames from a remote audio-bridge container.

    See operator_app.audio_ingest for the wire protocol. The pipeline
    subprocess pulls from the same AudioBus when STARK_AUDIO_SOURCE=ws.
    """
    await handle_audio_ingest(websocket)


@app.websocket("/ws/audio/subscribe")
async def ws_audio_subscribe(websocket: WebSocket) -> None:
    """Stream PCM frames to the pipeline subprocess (Phase 9.4.2).

    Used when STARK_AUDIO_SOURCE=ws — the pipeline reads frames from this
    endpoint instead of opening sd.InputStream locally.
    """
    await handle_audio_subscribe(websocket)


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
if not _operator_static.is_dir():
    _operator_static = Path(__file__).resolve().parent.parent / "displays" / "operator"
if _operator_static.exists():
    app.mount("/operator", StaticFiles(directory=str(_operator_static), html=True), name="operator")
else:
    logger.warning("displays/operator not found at %s — frontend will 404", _operator_static)
