"""Helper-level tests for tools/benchmark_stt_engines.py.

The end-to-end benchmark loop requires a GPU and real models, so it isn't
exercised in CI. These tests cover the pure helpers: WAV decoding, text
normalization, percentile aggregation, and CLI dispatch.
"""

from __future__ import annotations

import struct
import wave
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from tools import benchmark_stt_engines as bench


def _make_pcm16_wav(path: Path, duration_s: float = 0.1, sample_rate: int = 16000) -> None:
    n_samples = int(duration_s * sample_rate)
    samples = [int(0.2 * 32767) for _ in range(n_samples)]
    with wave.open(str(path), "wb") as wav_fh:
        wav_fh.setnchannels(1)
        wav_fh.setsampwidth(2)
        wav_fh.setframerate(sample_rate)
        wav_fh.writeframes(b"".join(struct.pack("<h", s) for s in samples))


def test_load_pcm16_wav_returns_float32(tmp_path: Path):
    wav = tmp_path / "tone.wav"
    _make_pcm16_wav(wav, duration_s=0.5)
    audio, sr = bench.load_pcm16_wav(wav)
    assert sr == 16000
    assert audio.dtype.name == "float32"
    assert audio.shape == (8000,)
    assert audio.min() >= -1.0 and audio.max() <= 1.0


def test_load_pcm16_wav_rejects_non_riff(tmp_path: Path):
    bogus = tmp_path / "not.wav"
    bogus.write_bytes(b"GARBAGE")
    with pytest.raises(ValueError, match="not RIFF"):
        bench.load_pcm16_wav(bogus)


def test_normalize_text_lowercases_and_strips(tmp_path: Path):
    out = bench.normalize_text("Praise the Lord!  Hallelujah.")
    assert out.islower()
    # No trailing punctuation, no double spaces
    assert "  " not in out
    assert not out.endswith(".")


def test_percentile_handles_edge_cases():
    assert bench.percentile([], 0.95) != bench.percentile([], 0.95)  # nan
    # Single element: percentile is that element
    assert bench.percentile([42.0], 0.95) == 42.0
    # 10 elements 1..10: p50 ≈ 5 or 6, p95 = 10
    vals = [float(i) for i in range(1, 11)]
    assert bench.percentile(vals, 0.95) == 10.0
    assert bench.percentile(vals, 0.0) == 1.0


def test_variants_table_has_required_fields():
    for key, cfg in bench.VARIANTS.items():
        assert "engine" in cfg, f"{key} missing engine"
        assert cfg["engine"] in {"faster-whisper", "hf"}, f"{key} unknown engine {cfg['engine']!r}"
        if cfg["engine"] == "faster-whisper":
            assert "compute_type" in cfg
        if cfg["engine"] == "hf":
            assert "compile_mode" in cfg
            assert "warmup_seconds" in cfg


def test_list_variants_prints_all(capsys):
    bench.list_variants()
    captured = capsys.readouterr().out
    for key in bench.VARIANTS:
        assert key in captured


def test_main_list_returns_zero(capsys):
    rc = bench.main(["--list"])
    assert rc == 0


def test_main_missing_variant_returns_two():
    rc = bench.main([])
    assert rc == 2


def test_build_engine_local_ct2_missing_raises(tmp_path: Path):
    variant = bench.VARIANTS["fw_int8float16_w16"].copy()
    # Override model_id to a missing path; requires_local_ct2 should trigger.
    with pytest.raises(SystemExit, match="requires a local CT2 model"):
        bench.build_engine("fw_int8float16_w16", variant, override_model_id=str(tmp_path / "missing_ct2"))


def test_collect_hardware_info_handles_missing_nvidia_smi(monkeypatch):
    """nvidia-smi may not be present on CI; helper should still return a dict."""
    monkeypatch.setattr(
        "subprocess.check_output",
        MagicMock(side_effect=FileNotFoundError("nvidia-smi not found")),
    )
    info = bench.collect_hardware_info()
    assert "platform" in info
    assert "python" in info
    # Either no GPU keys or an error key — never both populated successfully.
    assert "gpu_query_error" in info or "gpu_name" not in info
