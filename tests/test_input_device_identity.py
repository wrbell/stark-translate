"""Microphone identities survive process-local PortAudio index changes."""

import ast
import io
import json
import struct
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.input_devices import resolve_input_device


def native_devices(rows):
    return SimpleNamespace(
        query_devices=lambda: rows,
        query_hostapis=lambda index: {"name": ["Core Audio", "Other API"][index]},
    )


def mic(name="MacBook Pro Microphone", host=0, inputs=1):
    return {"name": name, "hostapi": host, "max_input_channels": inputs}


def test_exact_identity_re_resolves_index_when_continuity_input_appears():
    first = resolve_input_device(0, sd=native_devices([mic(), mic("Speakers", inputs=0)]))
    assert first == {"index": 0, "name": "MacBook Pro Microphone", "host_api": "Core Audio"}
    current = native_devices([mic("WR17.1 Microphone"), mic(), mic("Speakers", inputs=0)])
    assert resolve_input_device(0, name=first["name"], host_api=first["host_api"], sd=current)["index"] == 1
    # Integer-only callers keep the explicitly requested process-local index.
    assert resolve_input_device(0, sd=current)["name"] == "WR17.1 Microphone"


def test_duplicate_names_require_matching_host_and_reject_ambiguous_same_host():
    devices = native_devices([mic(host=0), mic(host=1)])
    with pytest.raises(ValueError, match="ambiguous"):
        resolve_input_device(name="MacBook Pro Microphone", sd=devices)
    assert resolve_input_device(name="MacBook Pro Microphone", host_api="Other API", sd=devices)["index"] == 1
    with pytest.raises(ValueError, match="ambiguous"):
        resolve_input_device(name="MacBook Pro Microphone", host_api="Core Audio", sd=native_devices([mic(), mic()]))


@pytest.mark.parametrize("rows", [[mic("WR17.1 Microphone")], [mic(inputs=0)], [mic(host=1)]])
def test_missing_selected_identity_never_uses_old_index_or_default(rows):
    with pytest.raises(ValueError, match="unavailable"):
        resolve_input_device(0, name="MacBook Pro Microphone", host_api="Core Audio", sd=native_devices(rows))


@pytest.mark.parametrize("index", [True, -1, 2, 0.5])
def test_invalid_legacy_index_rejected(index):
    with pytest.raises(ValueError, match="index"):
        resolve_input_device(index, sd=native_devices([mic()]))


def test_automatic_and_invalid_host_only_do_not_enumerate():
    assert resolve_input_device(sd=object()) is None
    with pytest.raises(ValueError, match="requires a device name"):
        resolve_input_device(host_api="Core Audio", sd=object())


def test_named_cli_input_bypasses_default_mic_probe_before_stream_open():
    # Evaluate the real pipeline's branch without importing its inference stack.
    tree = ast.parse((Path(__file__).resolve().parents[1] / "dry_run_ab.py").read_text())
    conditions = [
        node.test
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and any(
            isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == "detect_macbook_mic"
            for stmt in node.body
            for call in ast.walk(stmt)
        )
    ]
    assert len(conditions) == 1
    expression = compile(ast.Expression(conditions[0]), "input-auto-detect", "eval")
    context = dict(MIC_DEVICE=None, MIC_DEVICE_NAME=None, MIC_DEVICE_HOST_API=None, os=SimpleNamespace(environ={}))
    assert eval(expression, {"__builtins__": {}}, context)
    for selected in (
        {"MIC_DEVICE": 0},
        {"MIC_DEVICE_NAME": "MacBook Pro Microphone"},
        {"MIC_DEVICE_HOST_API": "Core Audio"},
    ):
        assert not eval(expression, {"__builtins__": {}}, context | selected)


def test_preflight_binds_identity_and_pipeline_carries_it_through_child_options(tmp_path, monkeypatch):
    from operator_app import main, preflight
    from operator_app.pipeline_manager import PipelineRunner, SessionConfig
    from tools.audio_bridge_client import open_audio_stream

    monkeypatch.setitem(sys.modules, "sounddevice", native_devices([mic(), mic("Speakers", inputs=0)]))
    selected = preflight.check_microphone(0)
    assert selected["status"] == "pass"
    monkeypatch.setattr(main, "_preflight_config", lambda cfg, root: {"ok": True, "checks": [selected]})
    config = SessionConfig(mic_device=0)
    runner = PipelineRunner(project_root=tmp_path)
    main._require_preflight(config, runner)
    argv = runner._build_argv(config)
    assert argv[argv.index("--device-name") + 1] == "MacBook Pro Microphone"
    assert argv[argv.index("--device-host-api") + 1] == "Core Audio"
    assert argv[argv.index("--device") + 1] == "0"
    stream = open_audio_stream(
        callback=lambda *args: None,
        samplerate=48000,
        channels=1,
        dtype="float32",
        blocksize=1536,
        device=config.mic_device,
        device_name=config.mic_device_name,
        device_host_api=config.mic_host_api,
    )
    assert json.loads(stream.argv[-1])["device_name"] == "MacBook Pro Microphone"
    assert json.loads(stream.argv[-1])["device_host_api"] == "Core Audio"
    # Loss of the selected identity blocks preflight despite another input.
    monkeypatch.setitem(sys.modules, "sounddevice", native_devices([mic("WR17.1 Microphone")]))
    failed = preflight.check_microphone(0, name=config.mic_device_name, host_api=config.mic_host_api)
    assert failed["status"] == "fail" and "unavailable" in failed["detail"]


def test_capture_worker_opens_current_index_and_persists_actual_identity(monkeypatch):
    import numpy as np

    from tools import capture_worker
    from tools.isolated_audio import IsolatedInputStream

    native = native_devices([mic("WR17.1 Microphone"), mic()])
    opened, wire = [], io.BytesIO()

    class NativeStream:
        def __init__(self, callback, **options):
            opened.append(options)
            self.callback = callback

        def __enter__(self):
            stamp = SimpleNamespace(inputBufferAdcTime=1, currentTime=1.032)
            self.callback(np.zeros((1536, 1), dtype="float32"), 1536, stamp, "")

        def __exit__(self, *exc):
            return False

    native.InputStream = NativeStream
    monkeypatch.setitem(sys.modules, "sounddevice", native)
    options = dict(
        device=0,
        device_name="MacBook Pro Microphone",
        device_host_api="Core Audio",
        samplerate=48000,
        channels=1,
        dtype="float32",
        blocksize=1536,
    )
    monkeypatch.setattr(sys, "argv", ["capture_worker", json.dumps(options)])
    write = capture_worker.write_frame

    class Finished(Exception):
        pass

    def write_one(stream, metadata, payload):
        write(wire, metadata, payload)
        raise Finished

    monkeypatch.setattr(capture_worker, "write_frame", write_one)
    with pytest.raises(Finished):
        capture_worker.main()
    assert opened[0]["device"] == 1
    assert "device_name" not in opened[0] and "device_host_api" not in opened[0]
    wire.seek(0)
    header_size = struct.unpack("!I", wire.read(4))[0]
    identity = json.loads(wire.read(header_size))["input_device"]
    assert identity == {"index": 1, "name": "MacBook Pro Microphone", "host_api": "Core Audio"}
    stream = IsolatedInputStream(callback=lambda *args: None, **options)
    wire.seek(0)
    stream._started = time.monotonic()
    stream._proc = SimpleNamespace(stdout=wire)
    stream._read()
    assert stream.capture_snapshot()["opened_device"] == identity


def test_missing_worker_identity_fails_before_native_open(monkeypatch):
    from tools import capture_worker

    native = native_devices([mic("WR17.1 Microphone")])
    native.InputStream = lambda **kw: pytest.fail("Must not open any substitute input")
    monkeypatch.setitem(sys.modules, "sounddevice", native)
    monkeypatch.setattr(
        sys, "argv", ["capture_worker", json.dumps({"device": 0, "device_name": "MacBook Pro Microphone"})]
    )
    with pytest.raises(ValueError, match="unavailable"):
        capture_worker.main()


def test_input_probe_api_passes_identity_without_changing_output_probe(tmp_path, monkeypatch):
    from operator_app import audio_tests

    calls = []
    monkeypatch.setattr(audio_tests, "probe_audio", lambda *args, **kw: calls.append((args, kw)) or {"ok": True})
    runner = SimpleNamespace(_project_root=tmp_path)
    audio_tests._probe(
        runner,
        audio_tests.InputProbeRequest(device=0, device_name="MacBook Pro Microphone", device_host_api="Core Audio"),
        "probe",
    )
    assert calls[-1] == (("probe", 0, 2), {"device_name": "MacBook Pro Microphone", "device_host_api": "Core Audio"})
    audio_tests._probe(runner, audio_tests.OutputProbeRequest(device=2), "output")
    assert calls[-1] == (("output", 2, 0.4), {})


def test_input_host_lookup_failure_is_reported_without_killing_device_watcher(monkeypatch):
    from operator_app.audio import list_devices

    native = native_devices([mic()])
    native.query_hostapis = lambda index: (_ for _ in ()).throw(RuntimeError("device disconnected"))
    monkeypatch.setitem(sys.modules, "sounddevice", native)
    listing = list_devices()
    assert listing.error and "identity unavailable" in listing.error
    assert not listing.inputs
