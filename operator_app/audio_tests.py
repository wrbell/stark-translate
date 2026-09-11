"""Explicit idle-only device checks; native calls run in disposable processes."""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from operator_app.pipeline_manager import PipelineRunner, get_runner
from operator_app.work_lease import WorkBusyError, get_work_lease
from tools.isolated_audio import AudioCaptureError, probe_audio

router = APIRouter(prefix="/api/audio", tags=["audio"])


class InputProbeRequest(BaseModel):
    device: int | str | None = None
    device_name: str | None = Field(default=None, min_length=1, max_length=512)
    device_host_api: str | None = Field(default=None, min_length=1, max_length=512)
    duration_s: float = Field(default=2, ge=0.2, le=5)


class OutputProbeRequest(BaseModel):
    device: int | str | None = None
    duration_s: float = Field(default=0.4, ge=0.1, le=1)


def _probe(runner, req, mode):
    try:
        with get_work_lease(runner._project_root).reserve("audio test"):
            identity = {}
            if mode == "probe" and (req.device_name is not None or req.device_host_api is not None):
                identity = {"device_name": req.device_name, "device_host_api": req.device_host_api}
            return probe_audio(mode, req.device, req.duration_s, **identity)
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
