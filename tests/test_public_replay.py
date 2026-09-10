import numpy as np
import pytest

from tools.public_replay import level_audio


def test_level_rule_is_deterministic_and_preserves_source_samples():
    source = np.array([0, -0.001, 0.001, 0], dtype=np.float32)
    original = source.copy()
    leveled, metadata = level_audio(source, "rms-0.08")
    assert metadata["output_rms"] == pytest.approx(0.08)
    assert metadata["gain"] > 1
    assert np.array_equal(source, original)
    assert np.array_equal(leveled, level_audio(source, "rms-0.08")[0])
    assert np.array_equal(source, level_audio(source, "none")[0])


def test_level_rule_limits_peaks_without_clipping_impulses():
    source = np.zeros(10000, dtype=np.float32)
    source[10] = 0.2
    result, metadata = level_audio(source, "rms-0.08")
    assert metadata["output_peak"] == pytest.approx(0.95)
    assert metadata["output_rms"] < 0.08
    assert metadata["clipping"] is False
    assert np.count_nonzero(result) == 1


@pytest.mark.parametrize("audio", [np.array([]), np.array([np.nan]), np.array([0.0])])
def test_level_rule_rejects_unusable_input(audio):
    with pytest.raises(ValueError):
        level_audio(audio, "rms-0.08")
