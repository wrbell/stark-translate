import pytest

from tools.music_recovery import MusicSpeechRecovery
from tools.pipeline_timing import AudioFrame, CaptureStamp


def frame(number, *, rate=48000):
    return AudioFrame(
        [number] * 512,
        CaptureStamp(number * 0.032, (number + 1) * 0.032, "replay_realtime", number * 1536, (number + 1) * 1536, rate),
    )


def test_recovery_has_a_hard_frame_bound_and_retains_original_stamps_and_pcm():
    recovery = MusicSpeechRecovery(15)
    original = tuple(frame(i) for i in range(15))
    for item in original:
        recovery.append(item)
    with pytest.raises(RuntimeError, match="frame limit"):
        recovery.append(frame(15))
    retained = recovery.take()
    assert retained == original and len(recovery) == 0
    timeline = recovery.timeline(retained)
    assert timeline.first == 0 and timeline.last == pytest.approx(0.48)
    assert timeline.sample_metadata()["sample_end"] == 15 * 1536


def test_recovery_requires_resolution_before_a_capture_gap_or_rate_change():
    recovery = MusicSpeechRecovery(15)
    first = frame(1)
    recovery.append(first)
    for candidate in (frame(3), frame(2, rate=16000)):
        assert not recovery.contiguous(candidate)
        with pytest.raises(ValueError, match="contiguous source"):
            recovery.append(candidate)
    assert recovery.take() == (first,)
    recovery.append(frame(3))
    assert recovery.timeline(recovery.take()).sample_metadata()["sample_start"] == 3 * 1536
