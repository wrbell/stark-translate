"""Multi-channel TTS routing with no real audio devices or Piper models."""

from unittest.mock import Mock, call, patch

import numpy as np
import pytest

from engines.audio_devices import OutputDeviceResolver, list_output_devices, resolve_output_device
from engines.base import TTSResult
from engines.mlx_engine import PiperTTSEngine


@pytest.fixture
def devices(monkeypatch):
    import sounddevice as sd

    listing = [
        {"name": "Microphone", "max_output_channels": 0},
        {"name": "MacBook Pro Speakers", "max_output_channels": 2},
        {"name": "BlackHole 2ch", "max_output_channels": 2},
        {"name": "BlackHole 16ch", "max_output_channels": 16},
    ]
    monkeypatch.setattr(sd, "query_devices", Mock(return_value=listing))
    monkeypatch.setattr(sd.default, "device", [0, 1])
    return listing


def test_output_listing(devices):
    assert list_output_devices() == [
        {"index": 1, "name": "MacBook Pro Speakers", "channels": 2, "default": True},
        {"index": 2, "name": "BlackHole 2ch", "channels": 2, "default": False},
        {"index": 3, "name": "BlackHole 16ch", "channels": 16, "default": False},
    ]


@pytest.mark.parametrize("spec,expected", [(1, 1), (2, 2), ("SPEAKERS", 1), ("blackHOLE", 2), (None, None)])
def test_resolve_output_device(devices, spec, expected):
    assert resolve_output_device(spec) == expected


@pytest.mark.parametrize("spec", [0, -1, 99, "Microphone", "missing", "", " "])
def test_reject_invalid_output(devices, spec):
    with pytest.raises(ValueError, match="No output device"):
        resolve_output_device(spec)


def test_default_resolution_needs_no_enumeration():
    with patch("sounddevice.query_devices", side_effect=AssertionError("unexpected enumeration")):
        assert resolve_output_device(None) is None
        assert OutputDeviceResolver().resolve(None) is None


@pytest.fixture
def tts_pipeline(devices, monkeypatch):
    import dry_run_ab as pipeline

    monkeypatch.setattr(pipeline, "_tts_device_resolver", OutputDeviceResolver())
    monkeypatch.setattr(pipeline.settings.tts, "output_devices", {"en": "speakers", "es": "BlackHole 2ch"})
    monkeypatch.setattr(pipeline.settings.tts, "output_device", 3)
    monkeypatch.setattr(pipeline, "_tts_chunk_counter", 0)
    engine = PiperTTSEngine()
    result = TTSResult(audio=np.zeros(32, dtype=np.float32), sample_rate=22050, latency_ms=1, text="Hello")
    monkeypatch.setattr(engine, "synthesize", Mock(return_value=result))
    monkeypatch.setattr(engine, "play", Mock())
    return pipeline, engine, result


def test_run_tts_routes_languages_and_logs_once(tts_pipeline, caplog):
    pipeline, engine, result = tts_pipeline
    with caplog.at_level("INFO", logger="engines.audio_devices"):
        for language in ("en", "es", "en", "hi"):
            pipeline._run_tts(engine, "Hello", language, 1, "local", None)
    assert engine.play.call_args_list == [
        call(result.audio, result.sample_rate, device=device) for device in (1, 2, 1, 3)
    ]
    assert caplog.text.count("TTS en output: MacBook Pro Speakers") == 1
    assert "TTS es output: BlackHole 2ch" in caplog.text


def test_explicit_none_overrides_fallback(tts_pipeline, monkeypatch):
    pipeline, engine, result = tts_pipeline
    monkeypatch.setattr(pipeline.settings.tts, "output_devices", {"en": None})
    pipeline._run_tts(engine, "Hello", "en", 1, "local", None)
    engine.play.assert_called_once_with(result.audio, result.sample_rate, device=None)


def test_hotplug_re_resolves_and_invalidates_other_routes(tts_pipeline, devices):
    import sounddevice as sd

    pipeline, engine, result = tts_pipeline
    resolver = pipeline._tts_device_resolver
    assert resolver.resolve("speakers") == 1
    assert resolver.resolve("BlackHole 2ch") == 2
    sd.query_devices.reset_mock()
    assert resolver.resolve("BlackHole 2ch") == 2
    sd.query_devices.assert_not_called()
    devices[1], devices[2] = devices[2], devices[1]
    engine.play.side_effect = [sd.PortAudioError("USB renumbered"), None]
    pipeline._run_tts(engine, "Hola", "es", 1, "local", None)
    assert engine.play.call_args_list == [
        call(result.audio, result.sample_rate, device=2),
        call(result.audio, result.sample_rate, device=1),
    ]
    assert resolver.resolve("speakers") == 2


@pytest.mark.parametrize("default_fails", [False, True])
def test_run_tts_falls_back_after_one_retry(tts_pipeline, caplog, default_fails):
    import sounddevice as sd

    pipeline, engine, result = tts_pipeline
    failure = sd.PortAudioError("unplugged")
    engine.play.side_effect = [failure, failure, failure if default_fails else None]
    pipeline._run_tts(engine, "Hola", "es", 1, "local", None)
    assert engine.play.call_args_list == [
        call(result.audio, result.sample_rate, device=device) for device in (2, 2, None)
    ]
    assert "falling back to system default" in caplog.text
    if default_fails:
        assert "skipping playback" in caplog.text


def test_missing_route_uses_default(tts_pipeline, devices, caplog):
    pipeline, engine, result = tts_pipeline
    devices.clear()
    pipeline._run_tts(engine, "Hola", "es", 1, "local", None)
    engine.play.assert_called_once_with(result.audio, result.sample_rate, device=None)
    assert "falling back to system default" in caplog.text


def test_real_engine_propagates_portaudio_to_resolver(devices):
    import sounddevice as sd

    audio = np.zeros(32, dtype=np.float32)
    with patch("sounddevice.play", side_effect=[sd.PortAudioError("gone"), sd.PortAudioError("gone"), None]) as play:
        OutputDeviceResolver().play(PiperTTSEngine().play, audio, 22050, language="en", spec=1)
    assert [c.kwargs["device"] for c in play.call_args_list] == [1, 1, None]


def test_settings_env_map(monkeypatch):
    from settings import TTSSettings

    monkeypatch.setenv("STARK_TTS_OUTPUT_DEVICES", '{"en":"MacBook Pro Speakers","es":2,"hi":null}')
    monkeypatch.setenv("STARK_TTS_OUTPUT_DEVICE", "3")
    config = TTSSettings()
    assert config.output_devices == {"en": "MacBook Pro Speakers", "es": 2, "hi": None}
    assert config.output_device == 3


def test_operator_argv_preserves_device_names_and_zero(tmp_path):
    from operator_app.pipeline_manager import PipelineRunner, SessionConfig

    config = SessionConfig(
        tts=True, tts_output_mode="local", tts_device=3, tts_device_en=0, tts_device_es="BlackHole 2ch"
    )
    runner = PipelineRunner(project_root=tmp_path)
    argv = runner._build_argv(config)
    for flag, value in [("--tts-device", "3"), ("--tts-device-en", "0"), ("--tts-device-es", "BlackHole 2ch")]:
        assert argv[argv.index(flag) + 1] == value
    config.tts = False
    assert "--tts-device-en" not in runner._build_argv(config)
    assert "--tts-device-es" not in runner._build_argv(config)


def test_output_devices_endpoint(devices):
    from fastapi.testclient import TestClient

    from operator_app.main import app

    client = TestClient(app)
    response = client.get("/api/audio/output-devices")
    assert response.status_code == 200
    assert response.json() == {"outputs": list_output_devices()}
    with patch("sounddevice.query_devices", side_effect=OSError("unavailable")):
        response = client.get("/api/audio/output-devices")
    assert response.status_code == 503
    assert response.json()["outputs"] == []


def test_start_request_passes_routes_to_session_config(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from operator_app.main import app, get_runner
    from operator_app.pipeline_manager import SessionStatus

    runner = Mock()
    runner._project_root = tmp_path
    runner.status.return_value = SessionStatus(state="idle")
    runner.start.return_value = SessionStatus(state="starting")
    monkeypatch.setattr("operator_app.main.run_all_checks", lambda **kwargs: {"ok": True, "checks": []})
    app.dependency_overrides[get_runner] = lambda: runner
    try:
        response = TestClient(app).post(
            "/api/session/start",
            json={"tts": True, "tts_output_mode": "local", "tts_device_en": 0, "tts_device_es": "BlackHole 2ch"},
        )
    finally:
        app.dependency_overrides.pop(get_runner)
    assert response.status_code == 200
    config = runner.start.call_args.args[0]
    assert config.tts_device_en == 0
    assert config.tts_device_es == "BlackHole 2ch"


def test_cli_parses_indices_and_names_without_starting_pipeline(monkeypatch):
    import argparse

    import dry_run_ab as pipeline

    parse_args = argparse.ArgumentParser.parse_args
    parsed = None

    class Parsed(Exception):
        pass

    def capture_args(parser):
        nonlocal parsed
        parsed = parse_args(
            parser,
            [
                "--tts",
                "--tts-output",
                "local",
                "--tts-device",
                "3",
                "--tts-device-en",
                "0",
                "--tts-device-es",
                "BlackHole 2ch",
            ],
        )
        raise Parsed

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", capture_args)
    with pytest.raises(Parsed):
        pipeline.main()
    assert parsed.tts_device == 3
    assert parsed.tts_device_en == 0
    assert parsed.tts_device_es == "BlackHole 2ch"


@pytest.mark.parametrize("mode", ["ws", "wav", "both"])
def test_nonlocal_modes_do_not_resolve_devices(tts_pipeline, mode, tmp_path, monkeypatch):
    pipeline, engine, _ = tts_pipeline
    monkeypatch.setattr(pipeline, "AUDIO_DIR", str(tmp_path))
    monkeypatch.setattr(pipeline, "broadcast_tts_audio", Mock())
    with (
        patch("sounddevice.query_devices", side_effect=AssertionError("unexpected device access")),
        patch("asyncio.run_coroutine_threadsafe") as broadcast,
    ):
        pipeline._run_tts(engine, "Hello", "en", 1, mode, None)
    engine.play.assert_not_called()
    assert broadcast.call_count == (1 if mode in ("ws", "both") else 0)
