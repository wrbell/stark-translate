"""Phase 9.4.1 — TTS output device selection tests."""

from __future__ import annotations

from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def _reset():
    from operator_app import audio, metrics, pipeline_manager

    metrics.reset_collector_for_tests()
    pipeline_manager.reset_runner_for_tests()
    audio.reset_watcher_for_tests()
    yield
    metrics.reset_collector_for_tests()
    pipeline_manager.reset_runner_for_tests()
    audio.reset_watcher_for_tests()


@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    from operator_app.main import app

    return TestClient(app)


# -- PiperTTSEngine.play -----------------------------------------------------


class TestPiperTTSEnginePlay:
    def test_play_calls_sounddevice_with_device(self):
        import numpy as np

        from engines.mlx_engine import PiperTTSEngine

        engine = PiperTTSEngine()
        audio = np.zeros(16000, dtype=np.float32)
        with patch("sounddevice.play") as sd_play:
            engine.play(audio, sample_rate=22050, device=3)
        sd_play.assert_called_once_with(audio, samplerate=22050, device=3)

    def test_play_with_no_device_passes_none(self):
        import numpy as np

        from engines.mlx_engine import PiperTTSEngine

        engine = PiperTTSEngine()
        audio = np.zeros(16000, dtype=np.float32)
        with patch("sounddevice.play") as sd_play:
            engine.play(audio, sample_rate=22050)
        sd_play.assert_called_once_with(audio, samplerate=22050, device=None)

    def test_play_swallows_sounddevice_errors(self):
        """If sounddevice raises, play() logs but doesn't propagate.

        Live TTS playback should never crash the pipeline thread.
        """
        import numpy as np

        from engines.mlx_engine import PiperTTSEngine

        engine = PiperTTSEngine()
        audio = np.zeros(16000, dtype=np.float32)
        with patch("sounddevice.play", side_effect=OSError("device busy")):
            engine.play(audio, sample_rate=22050, device=3)  # must not raise


# -- argv builder forwards tts flags -----------------------------------------


class TestArgvBuilderTtsForward:
    def test_local_mode_forwards_tts_output_and_device(self, tmp_path):
        from operator_app.pipeline_manager import PipelineRunner, SessionConfig

        runner = PipelineRunner(project_root=tmp_path)
        cfg = SessionConfig(lang="en", tts=True, tts_output_mode="local", tts_device=4)
        argv = runner._build_argv(cfg)

        assert "--tts" in argv
        # --tts-output local
        idx = argv.index("--tts-output")
        assert argv[idx + 1] == "local"
        # --tts-device 4
        idx = argv.index("--tts-device")
        assert argv[idx + 1] == "4"

    def test_ws_mode_omits_device_flag(self, tmp_path):
        from operator_app.pipeline_manager import PipelineRunner, SessionConfig

        runner = PipelineRunner(project_root=tmp_path)
        cfg = SessionConfig(lang="en", tts=True, tts_output_mode="ws", tts_device=None)
        argv = runner._build_argv(cfg)
        assert "--tts" in argv
        idx = argv.index("--tts-output")
        assert argv[idx + 1] == "ws"
        assert "--tts-device" not in argv

    def test_no_tts_omits_all_tts_flags(self, tmp_path):
        from operator_app.pipeline_manager import PipelineRunner, SessionConfig

        runner = PipelineRunner(project_root=tmp_path)
        cfg = SessionConfig(lang="en", tts=False)
        argv = runner._build_argv(cfg)
        assert "--tts" not in argv
        assert "--tts-output" not in argv
        assert "--tts-device" not in argv


# -- /api/session/start accepts new fields ------------------------------------


class TestStartRequestAcceptsTtsFields:
    def test_accepts_tts_output_mode_local(self, client, tmp_path):
        # Stub dry_run_ab.py so subprocess starts (we don't actually need to
        # wait running for this test — just that 200 comes back).
        from operator_app import pipeline_manager

        (tmp_path / "metrics").mkdir()
        stub = tmp_path / "dry_run_ab.py"
        stub.write_text(
            "import time, signal, sys\n_d={}\n_s = signal.signal\n_s(signal.SIGTERM, lambda *_: sys.exit(0))\nwhile True: time.sleep(0.1)\n"
        )

        pipeline_manager._runner = pipeline_manager.PipelineRunner(project_root=tmp_path)
        try:
            resp = client.post(
                "/api/session/start",
                json={
                    "lang": "en",
                    "tts": True,
                    "tts_output_mode": "local",
                    "tts_device": 5,
                },
            )
            assert resp.status_code == 200
        finally:
            pipeline_manager.get_runner().stop(timeout_s=3)

    def test_invalid_tts_output_mode_returns_422(self, client):
        resp = client.post(
            "/api/session/start",
            json={"lang": "en", "tts": True, "tts_output_mode": "magic"},
        )
        assert resp.status_code == 422


# -- TTS settings reachability ------------------------------------------------


class TestTtsSettings:
    def test_settings_has_output_device_field(self):
        from settings import settings

        # field exists on TTSSettings; default is None
        assert hasattr(settings.tts, "output_device")
        assert settings.tts.output_device is None

    def test_settings_output_mode_includes_local(self):
        from settings import settings

        # The Literal type allows "local"
        settings.tts.output_mode = "local"
        assert settings.tts.output_mode == "local"
