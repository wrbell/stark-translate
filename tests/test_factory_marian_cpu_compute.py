"""Marian CT2 on non-CUDA hosts must not request int8_float16 (unsupported on CPU)."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import engines.factory as factory


def _fake_ct2_dir(tmp_path):
    d = tmp_path / "adapters" / "marian_ct2" / "en-es" / "active"
    d.mkdir(parents=True)
    (d / "model.bin").write_bytes(b"x")
    for filename in ("config.json", "vocab.json", "source.spm", "target.spm", "tokenizer_config.json"):
        (d / filename).write_text("{}")
    return d


def test_mlx_host_downgrades_int8_float16_to_int8(tmp_path, monkeypatch):
    ct2_dir = _fake_ct2_dir(tmp_path)
    monkeypatch.setattr(factory, "_MARIAN_CT2_ROOT", tmp_path / "adapters" / "marian_ct2")
    monkeypatch.setenv("STARK_MODELS_DIR", str(tmp_path / "isolated-managed-cache"))
    monkeypatch.setitem(sys.modules, "ctranslate2", MagicMock())
    with patch("engines.cuda_engine.MarianCT2Engine") as eng:
        factory.create_translation_engine(
            backend="mlx",
            engine_type="marian",
            model_id="Helsinki-NLP/opus-mt-en-es",
            compute_type="int8_float16",
            source_lang="en",
            target_lang="es",
        )
    assert eng.call_args.kwargs["compute_type"] == "int8"
    assert eng.call_args.kwargs["intra_threads"] == 4
    assert str(ct2_dir) in str(eng.call_args.kwargs.get("model_dir") or eng.call_args)


def test_cuda_host_keeps_int8_float16(tmp_path, monkeypatch):
    _fake_ct2_dir(tmp_path)
    monkeypatch.setattr(factory, "_MARIAN_CT2_ROOT", tmp_path / "adapters" / "marian_ct2")
    monkeypatch.setenv("STARK_MODELS_DIR", str(tmp_path / "isolated-managed-cache"))
    monkeypatch.setitem(sys.modules, "ctranslate2", MagicMock())
    with patch("engines.cuda_engine.MarianCT2Engine") as eng:
        factory.create_translation_engine(
            backend="cuda",
            engine_type="marian",
            model_id="Helsinki-NLP/opus-mt-en-es",
            compute_type="int8_float16",
            source_lang="en",
            target_lang="es",
        )
    assert eng.call_args.kwargs["compute_type"] == "int8_float16"
    assert eng.call_args.kwargs["intra_threads"] is None
