"""Explicit idle-only device checks; native calls run in disposable processes."""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from operator_app.pipeline_manager import PipelineRunner, get_runner
from operator_app.work_lease import WorkBusyError, get_work_lease
from tools.isolated_audio import AudioCaptureError, probe_audio

router = APIRouter(prefix="/api/audio", tags=["audio"])


class InputProbeRequest(BaseModel):
    device: int | str | None = None
    duration_s: float = Field(default=2, ge=0.2, le=5)


class OutputProbeRequest(BaseModel):
    device: int | str | None = None
    duration_s: float = Field(default=0.4, ge=0.1, le=1)


def _probe(runner, req, mode):
    try:
        with get_work_lease(runner._project_root).reserve("audio test"):
            return probe_audio(mode, req.device, req.duration_s)
    except WorkBusyError as exc:
        raise HTTPException(status_code=409, detail={"code": "work_busy", "message": str(exc)}) from exc
    except AudioCaptureError as exc:
        raise HTTPException(status_code=422, detail={"code": "audio_unavailable", "message": str(exc)}) from exc


@router.post("/test-input")
def test_input(req: InputProbeRequest, runner: PipelineRunner = Depends(get_runner)):
    return _probe(runner, req, "probe")


@router.post("/test-output")
def test_output(req: OutputProbeRequest, runner: PipelineRunner = Depends(get_runner)):
    return _probe(runner, req, "output")
