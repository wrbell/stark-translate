"""Tests for the new --stt-backend resolution in engines/factory.py.

Covers the matrix introduced in v2026.7:
  - stt_backend in {auto, faster-whisper, hf, mlx}
  - spec_decode legacy flag still routes to HF
  - stt_backend='faster-whisper' + spec_decode=True is rejected
  - stt_backend='hf' on CUDA without spec_decode logs a warning but proceeds
  - W16 CT2 directory autodetection in the cuda/cpu branch
"""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from engines.factory import _resolve_ct2_whisper_model, create_stt_engine


@pytest.fixture
def fake_engines(monkeypatch):
    """Replace engine classes with MagicMock so we don't load real models."""
    fake_fw = MagicMock(name="FasterWhisperEngine")
    fake_hf = MagicMock(name="HFWhisperEngine")
    fake_mlx = MagicMock(name="MLXWhisperEngine")
    monkeypatch.setattr("engines.cuda_engine.FasterWhisperEngine", fake_fw)
    monkeypatch.setattr("engines.hf_whisper_engine.HFWhisperEngine", fake_hf)
    monkeypatch.setattr("engines.mlx_engine.MLXWhisperEngine", fake_mlx)
    return {"fw": fake_fw, "hf": fake_hf, "mlx": fake_mlx}


class TestResolveCT2WhisperModel:
    def test_returns_explicit_model_id_unchanged(self):
        assert _resolve_ct2_whisper_model("custom/model") == "custom/model"

    def test_returns_off_the_shelf_when_no_local_ct2(self, monkeypatch):
        # Point the active path at a non-existent location.
        monkeypatch.setattr("engines.factory._WHISPER_CT2_ACTIVE_PATH", Path("/nonexistent/whisper_ct2/active"))
        assert _resolve_ct2_whisper_model(None) == "large-v3-turbo"

    def test_returns_local_path_when_ct2_dir_exists(self, monkeypatch, tmp_path):
        ct2_dir = tmp_path / "whisper_turbo_ct2" / "active"
        ct2_dir.mkdir(parents=True)
        (ct2_dir / "model.bin").write_bytes(b"\x00" * 1024)
        monkeypatch.setattr("engines.factory._WHISPER_CT2_ACTIVE_PATH", ct2_dir)
        assert _resolve_ct2_whisper_model(None) == str(ct2_dir)

    def test_skips_local_path_when_model_bin_missing(self, monkeypatch, tmp_path):
        ct2_dir = tmp_path / "whisper_turbo_ct2" / "active"
        ct2_dir.mkdir(parents=True)
        # No model.bin written — should fall back to off-the-shelf.
        monkeypatch.setattr("engines.factory._WHISPER_CT2_ACTIVE_PATH", ct2_dir)
        assert _resolve_ct2_whisper_model(None) == "large-v3-turbo"


class TestSTTBackendResolution:
    def test_default_routes_to_faster_whisper_on_cuda(self, fake_engines):
        create_stt_engine(backend="cuda")
        fake_engines["fw"].assert_called_once()
        fake_engines["hf"].assert_not_called()
        fake_engines["mlx"].assert_not_called()

    def test_default_routes_to_mlx_on_mlx(self, fake_engines):
        create_stt_engine(backend="mlx")
        fake_engines["mlx"].assert_called_once()
        fake_engines["fw"].assert_not_called()

    def test_explicit_hf_routes_to_hf(self, fake_engines):
        create_stt_engine(backend="cuda", stt_backend="hf")
        fake_engines["hf"].assert_called_once()
        fake_engines["fw"].assert_not_called()

    def test_spec_decode_without_draft_raises(self, fake_engines):
        # distil-v3.5 + turbo is broken (see docs/archive/v2026.5/spec_decode_research.md);
        # we no longer auto-attach a default draft. Caller must supply one.
        with pytest.raises(ValueError, match="explicit draft_model_id"):
            create_stt_engine(backend="cuda", spec_decode=True)

    def test_spec_decode_with_explicit_draft_routes_to_hf(self, fake_engines):
        create_stt_engine(
            backend="cuda",
            spec_decode=True,
            draft_model_id="openai/whisper-large-v3-turbo",
            model_id="openai/whisper-large-v3",
        )
        fake_engines["hf"].assert_called_once()
        kwargs = fake_engines["hf"].call_args.kwargs
        assert kwargs["draft_model_id"] == "openai/whisper-large-v3-turbo"

    def test_stt_backend_hf_without_spec_decode_no_draft(self, fake_engines):
        create_stt_engine(backend="cuda", stt_backend="hf")
        kwargs = fake_engines["hf"].call_args.kwargs
        assert kwargs["draft_model_id"] is None

    def test_faster_whisper_plus_spec_decode_raises(self, fake_engines):
        with pytest.raises(ValueError, match="incompatible"):
            create_stt_engine(backend="cuda", stt_backend="faster-whisper", spec_decode=True)

    def test_faster_whisper_on_mlx_raises(self, fake_engines):
        with pytest.raises(ValueError, match="requires backend"):
            create_stt_engine(backend="mlx", stt_backend="faster-whisper")

    def test_warns_on_hf_cuda_without_spec_decode(self, fake_engines, caplog):
        with caplog.at_level(logging.WARNING, logger="engines.factory"):
            create_stt_engine(backend="cuda", stt_backend="hf")
        assert any("gives up CTranslate2 perf" in r.message for r in caplog.records)

    def test_compile_mode_forwards_to_hf_engine(self, fake_engines):
        create_stt_engine(backend="cuda", stt_backend="hf", compile_mode="reduce-overhead", warmup_seconds=2)
        kwargs = fake_engines["hf"].call_args.kwargs
        assert kwargs["compile_mode"] == "reduce-overhead"
        assert kwargs["warmup_seconds"] == 2

    def test_invalid_stt_backend_raises(self, fake_engines):
        with pytest.raises(ValueError, match="Unsupported stt_backend"):
            create_stt_engine(backend="cuda", stt_backend="tensorrt")

    def test_w16_autodetection_passes_local_path(self, fake_engines, monkeypatch, tmp_path):
        ct2_dir = tmp_path / "whisper_turbo_ct2" / "active"
        ct2_dir.mkdir(parents=True)
        (ct2_dir / "model.bin").write_bytes(b"\x00")
        monkeypatch.setattr("engines.factory._WHISPER_CT2_ACTIVE_PATH", ct2_dir)
        create_stt_engine(backend="cuda")
        kwargs = fake_engines["fw"].call_args.kwargs
        assert kwargs["model_id"] == str(ct2_dir)

    def test_explicit_model_id_wins_over_autodetect(self, fake_engines, monkeypatch, tmp_path):
        ct2_dir = tmp_path / "whisper_turbo_ct2" / "active"
        ct2_dir.mkdir(parents=True)
        (ct2_dir / "model.bin").write_bytes(b"\x00")
        monkeypatch.setattr("engines.factory._WHISPER_CT2_ACTIVE_PATH", ct2_dir)
        with patch("engines.factory.logger.info"):
            create_stt_engine(backend="cuda", model_id="user/override-model")
        kwargs = fake_engines["fw"].call_args.kwargs
        assert kwargs["model_id"] == "user/override-model"
