"""Helper-level tests for tools/benchmark_translate_engines.py.

The end-to-end GPU bench loop isn't exercised in CI. These tests cover pure
helpers, CLI dispatch, mocked engine construction, and a mocked run_variant
pass so coverage stays above the 50% gate after the v2026.8 Marian CT2 merge.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tools import benchmark_translate_engines as bench


def test_adapter_for_direction():
    assert bench._adapter_for_direction("es") == bench.CT2_ES_EN
    assert bench._adapter_for_direction("en") == bench.CT2_EN_ES
    assert bench._adapter_for_direction("fr") == bench.CT2_EN_ES  # default en-es


def test_variants_table_has_required_fields():
    for key, cfg in bench.VARIANTS.items():
        assert "engine" in cfg, f"{key} missing engine"
        assert cfg["engine"] in {"hf", "ct2"}, f"{key} unknown engine {cfg['engine']!r}"
        assert "device" in cfg
        if cfg["engine"] == "ct2":
            assert "compute_type" in cfg
            assert cfg.get("requires_local_ct2") is True
        if cfg["engine"] == "hf":
            assert "torch_dtype" in cfg


def test_percentile_handles_edge_cases():
    assert math.isnan(bench.percentile([], 0.95))
    assert bench.percentile([42.0], 0.95) == 42.0
    vals = [float(i) for i in range(1, 11)]
    assert bench.percentile(vals, 0.95) == 10.0
    assert bench.percentile(vals, 0.0) == 1.0
    assert bench.percentile(vals, 0.5) in (5.0, 6.0)


def test_chrf_pp_empty_returns_none():
    assert bench.chrf_pp("", "hola") is None
    assert bench.chrf_pp("hola", "") is None


def test_chrf_pp_without_sacrebleu(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "sacrebleu":
            raise ImportError("no sacrebleu")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    assert bench.chrf_pp("hola mundo", "hola mundo") is None


def test_chrf_pp_with_mock_sacrebleu(monkeypatch):
    fake = MagicMock()
    fake.sentence_chrf.return_value = SimpleNamespace(score=87.5)
    monkeypatch.setitem(__import__("sys").modules, "sacrebleu", fake)
    score = bench.chrf_pp("hola", "hola")
    assert score == 87.5
    fake.sentence_chrf.assert_called_once()


def test_run_cometkiwi_missing_package_returns_none():
    # unbabel-comet is not installed on CI; helper must soft-fail.
    assert bench.run_cometkiwi([{"source_text": "a", "prediction": "b"}]) is None


def test_run_cometkiwi_empty_pairs_returns_none(monkeypatch):
    fake_comet = MagicMock()
    monkeypatch.setitem(__import__("sys").modules, "comet", fake_comet)
    # Even with comet importable, empty src/hyp → None
    assert bench.run_cometkiwi([{"source_text": "", "prediction": ""}]) is None


def test_list_variants_prints_all(capsys):
    bench.list_variants()
    captured = capsys.readouterr().out
    for key in bench.VARIANTS:
        assert key in captured
    assert "needs local CT2" in captured


def test_parse_args_defaults():
    ns = bench.parse_args(["--variant", "marian_hf_fp32_cpu"])
    assert ns.variant == "marian_hf_fp32_cpu"
    assert ns.iterations == 3
    assert ns.warmup == 1
    assert ns.comet is False


def test_main_list_returns_zero(capsys):
    assert bench.main(["--list"]) == 0
    assert "Available variants" in capsys.readouterr().out


def test_main_missing_variant_returns_two():
    assert bench.main([]) == 2


def test_collect_hardware_info_handles_missing_nvidia_smi(monkeypatch):
    monkeypatch.setattr(
        "subprocess.check_output",
        MagicMock(side_effect=FileNotFoundError("nvidia-smi not found")),
    )
    info = bench.collect_hardware_info()
    assert "platform" in info
    assert "python" in info
    assert "gpu_query_error" in info or "gpu_name" not in info


def test_build_engines_ct2_missing_raises(tmp_path: Path):
    variant = bench.VARIANTS["marian_ct2_int8float16_cuda"].copy()
    with pytest.raises(SystemExit, match="needs CT2 model"):
        bench.build_engines(
            "marian_ct2_int8float16_cuda",
            variant,
            override_dir={"en": str(tmp_path / "missing"), "es": str(tmp_path / "missing2")},
        )


def test_build_engines_ct2_with_override(tmp_path: Path, monkeypatch):
    en_dir = tmp_path / "en"
    es_dir = tmp_path / "es"
    en_dir.mkdir()
    es_dir.mkdir()
    (en_dir / "model.bin").write_bytes(b"x")
    (es_dir / "model.bin").write_bytes(b"x")

    mock_cls = MagicMock(side_effect=lambda **kw: SimpleNamespace(**kw, load=MagicMock()))
    monkeypatch.setattr("engines.cuda_engine.MarianCT2Engine", mock_cls)

    engines = bench.build_engines(
        "marian_ct2_int8float16_cuda",
        bench.VARIANTS["marian_ct2_int8float16_cuda"],
        override_dir={"en": str(en_dir), "es": str(es_dir)},
    )
    assert set(engines) == {"en", "es"}
    assert mock_cls.call_count == 2


def test_build_engines_hf(monkeypatch):
    mock_cls = MagicMock(side_effect=lambda **kw: SimpleNamespace(**kw))
    monkeypatch.setattr("engines.marian_hf_engine.MarianHFEngine", mock_cls)
    engines = bench.build_engines(
        "marian_hf_fp32_cpu",
        bench.VARIANTS["marian_hf_fp32_cpu"],
        override_dir=None,
    )
    assert set(engines) == {"en", "es"}
    assert mock_cls.call_count == 2
    model_ids = {c.kwargs["model_id"] for c in mock_cls.call_args_list}
    assert model_ids == {
        "Helsinki-NLP/opus-mt-en-es",
        "Helsinki-NLP/opus-mt-es-en",
    }


def test_build_engines_unknown_engine_raises():
    with pytest.raises(ValueError, match="unknown engine"):
        bench.build_engines("bogus", {"engine": "llama"}, None)


def test_run_variant_mocked(tmp_path: Path, monkeypatch):
    """Exercise the measurement / summary / JSONL path without real models."""
    clips = [
        {
            "id": "c1",
            "source_lang": "en",
            "target_lang": "es",
            "length_tier": "short",
            "tier1_terms_present": True,
            "source_text": "Lord have mercy",
            "reference_text": "Señor ten misericordia",
            "tier1_term_expected": "misericordia",
        },
        {
            "id": "c2",
            "source_lang": "es",
            "target_lang": "en",
            "length_tier": "medium",
            "tier1_terms_present": False,
            "source_text": "Hola",
            "reference_text": "Hello",
            "tier1_term_expected": "",
        },
    ]

    fake_result = SimpleNamespace(text="Señor ten misericordia", latency_ms=12.0)
    fake_engine = MagicMock()
    fake_engine.translate.return_value = fake_result

    monkeypatch.setattr(
        bench,
        "build_engines",
        lambda *a, **k: {"en": fake_engine, "es": fake_engine},
    )
    monkeypatch.setattr(bench, "chrf_pp", lambda hyp, ref: 90.0 if hyp and ref else None)

    class FakeSampler:
        def __init__(self, interval_s=0.5):
            self.interval_s = interval_s

        def start(self):
            return None

        def stop(self):
            return {"peak_mib": 100.0}

    monkeypatch.setattr(bench, "VramSampler", FakeSampler)

    clips_jsonl = tmp_path / "clips.jsonl"
    summary = bench.run_variant(
        "marian_hf_fp32_cpu",
        bench.VARIANTS["marian_hf_fp32_cpu"],
        clips,
        iterations=1,
        warmup=0,
        clips_jsonl=clips_jsonl,
        override_dir=None,
        enable_comet=False,
    )

    assert summary["variant"] == "marian_hf_fp32_cpu"
    assert summary["canary"]["hits"] == 1
    assert summary["canary"]["total"] == 1
    assert "short" in summary["tiers"]
    assert clips_jsonl.exists()
    lines = clips_jsonl.read_text().strip().splitlines()
    assert len(lines) == 2
    rec = json.loads(lines[0])
    assert rec["clip_id"] == "c1"
    assert rec["term_hit"] is True
    fake_engine.unload.assert_called()


def test_main_runs_variant_end_to_end(tmp_path: Path, monkeypatch):
    # Place under PROJECT_ROOT so relative_to() succeeds (main() embeds a
    # repo-relative manifest path in the output JSON).
    man_dir = bench.PROJECT_ROOT / "metrics" / "_test_translate_bench"
    man_dir.mkdir(parents=True, exist_ok=True)
    man_path = man_dir / "manifest.json"
    out_path = man_dir / "out.json"
    try:
        manifest = {
            "total_clips": 1,
            "directions": {"en-es": 1, "es-en": 0},
            "tier1_clip_count": 0,
            "clips": [
                {
                    "id": "only",
                    "source_lang": "en",
                    "target_lang": "es",
                    "length_tier": "short",
                    "tier1_terms_present": False,
                    "source_text": "Hi",
                    "reference_text": "Hola",
                    "tier1_term_expected": "",
                }
            ],
        }
        man_path.write_text(json.dumps(manifest))

        monkeypatch.setattr(
            bench,
            "run_variant",
            lambda *a, **k: {
                "variant": "marian_hf_fp32_cpu",
                "config": {},
                "load_seconds": 0.1,
                "iterations": 1,
                "warmup": 0,
                "vram": {},
                "cold_start_ms": 1.0,
                "canary": {"hits": 0, "total": 0, "score": None},
                "tiers": {},
                "directions": {},
            },
        )
        monkeypatch.setattr(bench, "collect_hardware_info", lambda: {"gpu_name": "fake"})

        rc = bench.main(
            [
                "--variant",
                "marian_hf_fp32_cpu",
                "--manifest",
                str(man_path),
                "--output",
                str(out_path),
                "--iterations",
                "1",
                "--warmup",
                "0",
                "--quiet",
            ]
        )
        assert rc == 0
        doc = json.loads(out_path.read_text())
        assert doc["variant"] == "marian_hf_fp32_cpu"
        assert doc["hardware"]["gpu_name"] == "fake"
        assert doc["comet_enabled"] is False
    finally:
        for p in (man_path, out_path, man_dir / "out_clips.jsonl"):
            if p.exists():
                p.unlink()
        if man_dir.exists():
            try:
                man_dir.rmdir()
            except OSError:
                pass
