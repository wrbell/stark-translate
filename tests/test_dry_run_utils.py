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


class TestShouldSuppress:
    """Tests for _should_suppress() hallucination filter."""

    def _suppress(self, text, confidence=None, utterance_dur=None):
        from dry_run_ab import _should_suppress

        return _should_suppress(text, confidence, utterance_dur)

    # --- Tier 1: Known hallucination phrases ---

    def test_thank_you_low_conf_suppressed(self):
        assert self._suppress("Thank you.", confidence=0.30) is not None

    def test_thank_you_high_conf_kept(self):
        """A genuine 'thank you' at high confidence should pass."""
        assert self._suppress("Thank you.", confidence=0.85) is None

    def test_thanks_for_watching_suppressed(self):
        assert self._suppress("Thanks for watching.", confidence=0.50) is not None

    def test_bye_low_conf_suppressed(self):
        assert self._suppress("Bye.", confidence=0.30) is not None

    # --- Tier 2: VAD-floor duration chunks ---

    def test_vad_floor_low_conf_suppressed(self):
        assert self._suppress("Amen.", confidence=0.40, utterance_dur=0.77) is not None

    def test_vad_floor_high_conf_kept(self):
        """Short chunk at high confidence might be real speech."""
        assert self._suppress("Amen.", confidence=0.85, utterance_dur=0.77) is None

    def test_normal_duration_low_conf_not_suppressed_by_tier2(self):
        """Normal duration chunks aren't caught by the VAD-floor rule."""
        # May still be caught by tier 3 if ultra-short + very low conf
        result = self._suppress("Amen.", confidence=0.40, utterance_dur=3.5)
        # Tier 2 won't fire (dur > 0.80), tier 3 only fires if conf < 0.30
        assert result is None

    # --- Tier 3: Ultra-short text at very low confidence ---

    def test_two_words_very_low_conf_suppressed(self):
        assert self._suppress("Sagat Ruvan.", confidence=0.06) is not None

    def test_one_word_very_low_conf_suppressed(self):
        assert self._suppress("Grabo.", confidence=0.10) is not None

    def test_two_words_moderate_conf_kept(self):
        """Short text at moderate confidence should pass."""
        assert self._suppress("I see.", confidence=0.50) is None

    # --- Tier 4: Short fragments not in whitelist ---

    def test_short_fragment_low_conf_suppressed(self):
        """'I was blanking.' — 3 words, conf=0.35, not whitelisted."""
        assert self._suppress("I was blanking.", confidence=0.35) is not None

    def test_short_fragment_high_conf_kept(self):
        """Short fragment at decent confidence should pass."""
        assert self._suppress("I was blanking.", confidence=0.65) is None

    def test_whitelisted_short_phrase_kept(self):
        """'Amen' at low conf is whitelisted — should NOT be suppressed by tier 4."""
        assert self._suppress("Amen.", confidence=0.40, utterance_dur=3.0) is None

    def test_four_words_low_conf_suppressed(self):
        """'A continuous life.' — nonsensical 3-word fragment from mic artifact."""
        assert self._suppress("A continuous life.", confidence=0.28) is not None

    # --- Legitimate speech should never be suppressed ---

    def test_normal_sentence_kept(self):
        assert self._suppress("For God so loved the world.", confidence=0.92) is None

    def test_normal_sentence_low_conf_kept(self):
        """Longer sentences even at low conf should not be suppressed."""
        assert self._suppress("And Daniel prayed to the Lord his God.", confidence=0.45) is None

    def test_five_words_low_conf_kept(self):
        """5+ words at low conf should NOT be suppressed (too risky)."""
        assert self._suppress("He really wanted to understand.", confidence=0.38) is None

    def test_empty_text_kept(self):
        assert self._suppress("", confidence=0.50) is None

    def test_none_confidence_defaults_safe(self):
        """When confidence is None, defaults to 1.0 — no suppression."""
        assert self._suppress("Thank you.", confidence=None) is None


class TestPhraseCorrections:
    """Tests for offensive/erroneous phrase corrections in correct_stt_output()."""

    def _correct(self, text):
        from dry_run_ab import correct_stt_output

        return correct_stt_output(text)

    def test_damn_you_corrected_to_daniel(self):
        """'Damn you' from misrecognized 'Daniel' — critical fix."""
        result, corrections = self._correct("Damn you, if you will, is one of my heroes.")
        assert "Daniel" in result
        assert "damn" not in result.lower()
        assert len(corrections) > 0

    def test_damn_you_standalone(self):
        result, corrections = self._correct("Damn you.")
        assert result.startswith("Daniel")

    def test_danic_corrected_to_daniel(self):
        result, corrections = self._correct("Looking down at Danic Chapter 8.")
        assert "Daniel" in result

    def test_prugot_corrected(self):
        result, corrections = self._correct("Daniel Prugot.")
        assert "Proved God" in result or "proved God" in result

    def test_legitimate_text_unchanged(self):
        text = "For God so loved the world."
        result, corrections = self._correct(text)
        assert result == text
        assert len(corrections) == 0
