"""Tests for utility functions in dry_run_ab.py."""

import os
import platform
import re
import socket

import numpy as np


class TestGetLocalIp:
    """Tests for get_local_ip()."""

    def test_returns_string(self):
        from dry_run_ab import get_local_ip

        result = get_local_ip()
        assert isinstance(result, str)

    def test_returns_valid_ip_format(self):
        from dry_run_ab import get_local_ip

        result = get_local_ip()
        # Must match x.x.x.x where each octet is 0-255
        assert re.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", result)

    def test_returns_fallback_on_socket_failure(self, monkeypatch):
        from dry_run_ab import get_local_ip

        def _raise(*args, **kwargs):
            raise OSError("network unreachable")

        monkeypatch.setattr(socket, "socket", _raise)
        assert get_local_ip() == "127.0.0.1"


class TestGetResourceSnapshot:
    """Tests for get_resource_snapshot()."""

    def test_returns_dict_with_expected_keys(self):
        from dry_run_ab import get_resource_snapshot

        result = get_resource_snapshot()
        expected_keys = {"cpu_percent", "ram_used_gb", "ram_percent", "process_ram_gb"}
        assert expected_keys == set(result.keys())

    def test_all_values_are_numeric(self):
        from dry_run_ab import get_resource_snapshot

        result = get_resource_snapshot()
        for key, value in result.items():
            assert isinstance(value, (int, float)), f"{key} is {type(value)}, expected numeric"

    def test_ram_percent_in_valid_range(self):
        from dry_run_ab import get_resource_snapshot

        result = get_resource_snapshot()
        assert 0 <= result["ram_percent"] <= 100


class TestGetHardwareProfile:
    """Tests for get_hardware_profile()."""

    def test_returns_dict_with_expected_keys(self):
        from dry_run_ab import get_hardware_profile

        result = get_hardware_profile()
        required_keys = {
            "platform",
            "processor",
            "cpu_count_physical",
            "cpu_count_logical",
            "ram_total_gb",
            "python_version",
        }
        assert required_keys.issubset(set(result.keys()))

    def test_has_gpu_model_key(self):
        from dry_run_ab import get_hardware_profile

        result = get_hardware_profile()
        # Always present -- "unknown" on non-Mac or when system_profiler fails
        assert "gpu_model" in result

    def test_python_version_matches_platform(self):
        from dry_run_ab import get_hardware_profile

        result = get_hardware_profile()
        assert result["python_version"] == platform.python_version()


class TestSaveChunkAudio:
    """Tests for save_chunk_audio().

    The function uses module-level AUDIO_DIR (``stark_data/live_sessions/{SESSION_ID}``)
    and SAMPLE_RATE (16000).  scipy is installed locally so we mock ``scipy.io.wavfile.write``
    directly and redirect AUDIO_DIR to a temp directory.
    """

    def test_returns_expected_filename_and_creates_dir(self, monkeypatch, tmp_path):
        import sys

        import dry_run_ab

        audio_dir = str(tmp_path / "chunks")
        monkeypatch.setattr(dry_run_ab, "AUDIO_DIR", audio_dir)

        # scipy is mocked by conftest; the local import in save_chunk_audio
        # resolves via attribute traversal on sys.modules["scipy"], so
        # wav.write lands on sys.modules["scipy"].io.wavfile.write.
        wav_mock = sys.modules["scipy"].io.wavfile
        wav_mock.write.reset_mock()

        audio = np.zeros(16000, dtype=np.float32)
        result = dry_run_ab.save_chunk_audio(audio, 7)

        assert result == os.path.join(audio_dir, "chunk_0007.wav")
        # os.makedirs should have created the directory
        assert os.path.isdir(audio_dir)

    def test_calls_wav_write_with_correct_args(self, monkeypatch, tmp_path):
        import sys

        import dry_run_ab

        audio_dir = str(tmp_path / "chunks")
        monkeypatch.setattr(dry_run_ab, "AUDIO_DIR", audio_dir)

        wav_mock = sys.modules["scipy"].io.wavfile
        wav_mock.write.reset_mock()

        audio = np.ones(800, dtype=np.float32)
        dry_run_ab.save_chunk_audio(audio, 3)

        wav_mock.write.assert_called_once()
        call_args = wav_mock.write.call_args[0]
        assert call_args[0] == os.path.join(audio_dir, "chunk_0003.wav")
        assert call_args[1] == 16000
        np.testing.assert_array_equal(call_args[2], audio)
