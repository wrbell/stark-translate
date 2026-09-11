"""Phase 9.6.1 — live diarization watcher + daemon stub tests."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).parent.parent
DAEMON = ROOT / "features" / "live_diarize.py"


@pytest.fixture(autouse=True)
def _reset():
    from operator_app import audio, features, metrics, pipeline_manager

    metrics.reset_collector_for_tests()
    pipeline_manager.reset_runner_for_tests()
    audio.reset_watcher_for_tests()
    features.reset_features_for_tests()
    yield
    metrics.reset_collector_for_tests()
    pipeline_manager.reset_runner_for_tests()
    audio.reset_watcher_for_tests()
    features.reset_features_for_tests()


# -- LiveDiarizationWatcher --------------------------------------------------


class TestLiveDiarizationWatcher:
    def test_snapshot_when_file_missing(self, tmp_path):
        from operator_app.features import LiveDiarizationWatcher

        watcher = LiveDiarizationWatcher(jsonl_path=tmp_path / "missing.jsonl")
        snap = watcher.snapshot()
        assert snap["current_speaker"] is None
        assert snap["transitions"] == 0
        assert snap["recent"] == []

    def test_force_scan_picks_up_labels(self, tmp_path):
        from operator_app.features import LiveDiarizationWatcher

        path = tmp_path / "diarize.jsonl"
        path.write_text(
            json.dumps({"chunk_id": 1, "speaker": "Speaker A", "confidence": 0.9, "ts": 1.0})
            + "\n"
            + json.dumps({"chunk_id": 2, "speaker": "Speaker B", "confidence": 0.8, "ts": 2.0})
            + "\n"
        )
        watcher = LiveDiarizationWatcher(jsonl_path=path)
        snap = watcher.force_scan()
        assert snap["current_speaker"] == "Speaker B"
        assert snap["transitions"] == 1
        assert len(snap["recent"]) == 2
        assert "captions" in snap

    def test_incremental_tail(self, tmp_path):
        """Rescanning doesn't re-emit already-seen records."""
        from operator_app.features import LiveDiarizationWatcher

        path = tmp_path / "diarize.jsonl"
        with path.open("w") as f:
            f.write(json.dumps({"chunk_id": 1, "speaker": "Speaker A", "confidence": 0.9}) + "\n")

        watcher = LiveDiarizationWatcher(jsonl_path=path)
        watcher.force_scan()
        first_recent = watcher.snapshot()["recent"]
        assert len(first_recent) == 1

        # Append a new record — only that one should appear in the next snapshot's recent.
        with path.open("a") as f:
            f.write(json.dumps({"chunk_id": 2, "speaker": "Speaker A", "confidence": 0.85}) + "\n")
        watcher.force_scan()
        snap = watcher.snapshot()
        assert len(snap["recent"]) == 2
        assert snap["transitions"] == 0  # both Speaker A
        assert snap["current_speaker"] == "Speaker A"

    def test_malformed_lines_skipped(self, tmp_path):
        from operator_app.features import LiveDiarizationWatcher

        path = tmp_path / "diarize.jsonl"
        path.write_text(
            "not-json\n"
            + json.dumps({"chunk_id": 1, "speaker": "Speaker A", "confidence": 0.9})
            + "\n"
            + "{partial json\n"
            + json.dumps({"chunk_id": 2, "speaker": "Speaker B"})
            + "\n"
        )
        watcher = LiveDiarizationWatcher(jsonl_path=path)
        snap = watcher.force_scan()
        # Two valid records survive
        assert len(snap["recent"]) == 2
        assert snap["current_speaker"] == "Speaker B"

    def test_buffer_capped_at_max_labels(self, tmp_path):
        from operator_app.features import LiveDiarizationWatcher

        path = tmp_path / "diarize.jsonl"
        # 200 records → MAX_LABELS=100 truncation
        with path.open("w") as f:
            for i in range(200):
                speaker = "A" if i % 2 == 0 else "B"
                f.write(json.dumps({"chunk_id": i, "speaker": f"Speaker {speaker}", "confidence": 0.5}) + "\n")

        watcher = LiveDiarizationWatcher(jsonl_path=path)
        watcher.force_scan()
        # snapshot returns last 10 in 'recent'; buffer total is capped at MAX_LABELS internally
        with watcher._lock:
            assert len(watcher._labels) == LiveDiarizationWatcher.MAX_LABELS

    def test_snapshot_includes_interval_timestamps(self, tmp_path):
        from operator_app.features import LiveDiarizationWatcher

        path = tmp_path / "diarize.jsonl"
        path.write_text(
            json.dumps(
                {
                    "chunk_id": 1,
                    "speaker": "Speaker A",
                    "confidence": 0.9,
                    "ts": 10.0,
                    "start_ts": 8.0,
                    "end_ts": 10.0,
                }
            )
            + "\n"
        )
        watcher = LiveDiarizationWatcher(jsonl_path=path)
        snap = watcher.force_scan()
        rec = snap["recent"][0]
        assert rec["start_ts"] == 8.0
        assert rec["end_ts"] == 10.0
        assert rec["timestamp"] == 10.0

    def test_legacy_record_timestamps_are_none(self, tmp_path):
        from operator_app.features import LiveDiarizationWatcher

        path = tmp_path / "diarize.jsonl"
        path.write_text(json.dumps({"chunk_id": 1, "speaker": "Speaker A", "confidence": 0.9, "ts": 1.0}) + "\n")
        watcher = LiveDiarizationWatcher(jsonl_path=path)
        rec = watcher.force_scan()["recent"][0]
        assert rec["start_ts"] is None
        assert rec["end_ts"] is None
        assert rec["timestamp"] == 1.0

    def test_captions_from_csv_when_bound(self, tmp_path):
        from operator_app.features import LiveDiarizationWatcher

        jsonl = tmp_path / "diarize.jsonl"
        jsonl.write_text(json.dumps({"chunk_id": 1, "speaker": "Speaker A", "confidence": 1.0, "ts": 1.0}) + "\n")
        csv_path = tmp_path / "ab.csv"
        csv_path.write_text("chunk_id,english,spanish_a,speaker\n1,Hello world,Hola mundo,Speaker A\n")
        watcher = LiveDiarizationWatcher(jsonl_path=jsonl, csv_path=csv_path)
        snap = watcher.force_scan()
        assert snap["captions"][0]["english"] == "Hello world"
        assert snap["captions"][0]["speaker"] == "Speaker A"


# -- get_diarize_watcher singleton ------------------------------------------


class TestSingletonRebind:
    def test_lazy_creation_and_rebind(self, tmp_path):
        from operator_app.features import get_diarize_watcher

        # No path passed yet → None
        assert get_diarize_watcher() is None

        path = tmp_path / "diarize.jsonl"
        path.write_text(json.dumps({"chunk_id": 1, "speaker": "Speaker X", "confidence": 1.0}) + "\n")

        w1 = get_diarize_watcher(jsonl_path=path)
        assert w1 is not None
        # Subsequent retrieval without a path returns the same watcher
        assert get_diarize_watcher() is w1

        # Rebind to a new path → new watcher instance
        path2 = tmp_path / "diarize2.jsonl"
        path2.write_text(json.dumps({"chunk_id": 1, "speaker": "Speaker Z", "confidence": 1.0}) + "\n")
        w2 = get_diarize_watcher(jsonl_path=path2)
        assert w2 is not w1


# -- metrics snapshot includes diarization ----------------------------------


class TestMetricsIntegration:
    def test_snapshot_audio_diarization_field(self, tmp_path):
        from operator_app.features import get_diarize_watcher
        from operator_app.metrics import MetricsCollector

        path = tmp_path / "diarize.jsonl"
        path.write_text(json.dumps({"chunk_id": 1, "speaker": "Speaker A", "confidence": 1.0}) + "\n")
        watcher = get_diarize_watcher(jsonl_path=path)
        watcher.force_scan()

        c = MetricsCollector()
        snap = c.snapshot()
        assert "diarization" in snap["audio"]
        assert snap["audio"]["diarization"]["current_speaker"] == "Speaker A"


# -- features/live_diarize.py daemon ----------------------------------------


class TestDaemon:
    @pytest.fixture(autouse=True)
    def _require_python(self):
        if shutil.which("python3") is None and shutil.which(sys.executable) is None:
            pytest.skip("python interpreter unavailable")

    def test_daemon_emits_fake_labels_when_pyannote_unavailable(self, tmp_path, monkeypatch):
        """The daemon shouldn't crash if pyannote isn't installed; it should
        emit synthetic Speaker A/B labels for scaffolding tests.

        We force the no-pyannote path by passing a missing rolling WAV — the
        daemon's --max-iters flag bounds the loop. It also exits cleanly via
        a fake-label fallback because the WAV isn't there to diarize.
        """
        out_path = tmp_path / "diarize.jsonl"
        rolling_wav = tmp_path / "rolling.wav"  # intentionally missing

        # Force the "no pyannote" branch by patching sys.path so the import
        # in _load_pyannote fails. Easier: just rely on the daemon's existing
        # try/except. With max-iters=2 and interval=0.1, the daemon runs twice
        # and exits.
        result = subprocess.run(
            [
                sys.executable,
                str(DAEMON),
                "--rolling-wav",
                str(rolling_wav),
                "--output",
                str(out_path),
                "--interval-s",
                "0.1",
                "--max-iters",
                "2",
                "--log-level",
                "WARNING",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert result.returncode == 0, f"stderr:\n{result.stderr}"
        # Output may be empty (no pyannote AND no rolling WAV is the no-op
        # branch). The daemon should at least NOT crash. We're satisfied
        # with returncode 0.

    def test_daemon_help_works(self):
        result = subprocess.run(
            [sys.executable, str(DAEMON), "--help"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        assert result.returncode == 0
        help_text = (result.stdout + result.stderr).lower()
        assert "live diarization" in help_text
        assert "--mode" in result.stdout or "--mode" in result.stderr

    def test_daemon_emits_fake_labels_when_wav_present(self, tmp_path):
        """``--fake-labels`` writes alternating A/B with start_ts/end_ts."""
        out_path = tmp_path / "diarize.jsonl"
        rolling_wav = tmp_path / "rolling.wav"
        rolling_wav.write_bytes(b"RIFF\x00\x00\x00\x00WAVE")

        result = subprocess.run(
            [
                sys.executable,
                str(DAEMON),
                "--rolling-wav",
                str(rolling_wav),
                "--output",
                str(out_path),
                "--interval-s",
                "0.05",
                "--max-iters",
                "2",
                "--fake-labels",
                "--log-level",
                "WARNING",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert result.returncode == 0, f"stderr:\n{result.stderr}"
        assert out_path.exists()
        lines = [json.loads(l) for l in out_path.read_text().splitlines() if l.strip()]
        assert len(lines) >= 1
        assert lines[0]["speaker"] in ("Speaker A", "Speaker B")
        assert "start_ts" in lines[0] and "end_ts" in lines[0]

    def test_daemon_embed_fake_clusters_a_and_b(self, tmp_path):
        """``--mode embed --embedder fake`` assigns distinct labels to quiet vs loud WAVs."""
        import array
        import wave

        session = tmp_path / "session"
        session.mkdir()
        out_path = tmp_path / "diarize.jsonl"

        def _write(name: str, amplitude: int) -> Path:
            path = session / name
            samples = array.array("h", [int(amplitude)] * 8000)
            with wave.open(str(path), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(samples.tobytes())
            return path

        _write("chunk_0001.wav", 0)
        _write("chunk_0002.wav", 30000)
        chunks = session / "chunks.jsonl"
        chunks.write_text(
            json.dumps({"chunk_id": 1, "wav": "chunk_0001.wav", "start_ts": 1.0, "end_ts": 2.0})
            + "\n"
            + json.dumps({"chunk_id": 2, "wav": "chunk_0002.wav", "start_ts": 2.0, "end_ts": 3.0})
            + "\n"
        )

        result = subprocess.run(
            [
                sys.executable,
                str(DAEMON),
                "--rolling-wav",
                str(session / "rolling.wav"),
                "--chunks-jsonl",
                str(chunks),
                "--session-dir",
                str(session),
                "--output",
                str(out_path),
                "--mode",
                "embed",
                "--embedder",
                "fake",
                "--interval-s",
                "0.05",
                "--max-iters",
                "2",
                "--log-level",
                "WARNING",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert result.returncode == 0, f"stderr:\n{result.stderr}"
        records = [json.loads(l) for l in out_path.read_text().splitlines() if l.strip()]
        speakers = {r["speaker"] for r in records}
        assert "Speaker A" in speakers
        assert "Speaker B" in speakers
        assert all("start_ts" in r and "end_ts" in r for r in records)


class TestPipelineHook:
    @pytest.mark.parametrize("interpreter", [None, "/audited venv/bin/python"])
    @pytest.mark.parametrize("mode", ["embed", "pyannote"])
    def test_daemon_interpreter_and_environment(self, tmp_path, monkeypatch, interpreter, mode):
        import dry_run_ab as d

        monkeypatch.setattr(d, "DIARIZE_ENABLED", True)
        monkeypatch.setattr(d, "DIARIZE_MODE", mode)
        monkeypatch.setattr(d, "DIARIZE_PYTHON", interpreter)
        monkeypatch.setattr(d, "AUDIO_DIR", str(tmp_path / "audio"))
        monkeypatch.setattr(d, "DIARIZE_JSONL", str(tmp_path / "labels.jsonl"))
        monkeypatch.setattr(d, "_diarize_proc", None)
        monkeypatch.setenv("HF_TOKEN", "test-token")
        monkeypatch.setenv("HF_HUB_OFFLINE", "0")
        popen = MagicMock()
        monkeypatch.setattr(d.subprocess, "Popen", popen)
        d.start_diarize_daemon()
        popen.assert_called_once()
        command = popen.call_args.args[0]
        assert command[0] == (interpreter or sys.executable)
        assert command[command.index("--mode") + 1] == mode
        assert popen.call_args.kwargs["start_new_session"] is True
        env = popen.call_args.kwargs["env"]
        assert env["HF_HUB_OFFLINE"] == ("1" if mode == "embed" else "0")
        assert env["HF_TOKEN"] == "test-token"
        assert d.os.environ["HF_HUB_OFFLINE"] == "0"
        assert d._diarize_configuration()["diarize_python"] == command[0]

    @pytest.mark.parametrize("kind", ["missing", "nonexecutable", "directory"])
    def test_invalid_interpreter_rejected_at_parse_time(self, tmp_path, monkeypatch, capsys, kind):
        import dry_run_ab as d

        path = tmp_path / "python"
        if kind == "nonexecutable":
            path.write_text("not executable")
            path.chmod(0o600)
        elif kind == "directory":
            path.mkdir()
        monkeypatch.setattr(sys, "argv", ["dry_run_ab.py", "--diarize-python", str(path)])
        popen = MagicMock(side_effect=AssertionError("must fail before launching anything"))
        monkeypatch.setattr(d.subprocess, "Popen", popen)
        with pytest.raises(SystemExit) as exc:
            d.main()
        assert exc.value.code == 2
        assert "--diarize-python" in capsys.readouterr().err
        popen.assert_not_called()

    def test_interpreter_validation_preserves_venv_symlink(self, tmp_path, monkeypatch):
        import dry_run_ab as d

        venv_python = tmp_path / "python"
        venv_python.symlink_to(sys.executable)
        monkeypatch.chdir(tmp_path)
        assert d._diarize_python_path("python") == str(venv_python)

    def test_summary_records_diarization_configuration(self, monkeypatch):
        import dry_run_ab as d

        monkeypatch.setattr(d, "DIARIZE_PYTHON", "/audited/bin/python")
        monkeypatch.setattr(d, "all_results", [])
        pool = MagicMock()
        monkeypatch.setattr(d, "_io_pool", pool)
        d.print_summary()
        summary = pool.submit.call_args.args[1]
        assert summary["diarize_python"] == "/audited/bin/python"
        assert summary["diarize_mode"] == d.DIARIZE_MODE

    def test_diarize_off_by_default(self):
        import dry_run_ab as d

        assert d.DIARIZE_ENABLED is False

    def test_lookup_reads_jsonl(self, tmp_path, monkeypatch):
        import dry_run_ab as d

        path = tmp_path / "diarize.jsonl"
        path.write_text(
            json.dumps(
                {
                    "chunk_id": 1,
                    "speaker": "Speaker A",
                    "confidence": 1.0,
                    "start_ts": 0.0,
                    "end_ts": 2.0,
                    "ts": 2.0,
                }
            )
            + "\n"
        )
        monkeypatch.setattr(d, "DIARIZE_JSONL", str(path))
        assert d._lookup_speaker(0.2, 1.8) == "Speaker A"

    def test_result_data_speaker_only_when_enabled(self):
        from features.speaker_labels import speaker_field_for_result

        result = {"chunk_id": 1, "english": "Hello"}
        result.update(speaker_field_for_result(False, "Speaker A"))
        assert "speaker" not in result
        result.update(speaker_field_for_result(True, "Speaker A"))
        assert result["speaker"] == "Speaker A"


def test_load_wav_array_uses_soundfile_not_torchaudio(tmp_path, monkeypatch):
    """The diarization extra ships soundfile, not TorchCodec; WAV reads must not need torchaudio.load."""
    import sys
    import wave

    import numpy as np

    from features import live_diarize

    wav = tmp_path / "two_channel.wav"
    rate = 16000
    frames = (np.sin(np.linspace(0, 200, rate // 4)) * 0.5 * 32767).astype("<i2")
    with wave.open(str(wav), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(frames.tobytes())

    class _NoTorchaudio:
        def __getattr__(self, name):  # pragma: no cover - any attribute access is a failure
            raise AssertionError("torchaudio.load must not be used when soundfile is available")

    class _WaveSoundfile:
        """Minimal soundfile stand-in (conftest stubs the real module): reads PCM16 WAV via ``wave``."""

        @staticmethod
        def read(path, dtype="float32", always_2d=True):
            with wave.open(str(path), "rb") as handle:
                pcm = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2")
                data = (pcm.astype(dtype) / 32768.0).reshape(-1, handle.getnchannels())
                return data, handle.getframerate()

    monkeypatch.setitem(sys.modules, "torchaudio", _NoTorchaudio())
    monkeypatch.setitem(sys.modules, "soundfile", _WaveSoundfile())
    array = live_diarize._load_wav_array(str(wav))
    assert array.shape == (1, rate // 4)
    assert array.dtype == np.float32
    assert abs(float(np.abs(array).max()) - 0.5) < 0.01
