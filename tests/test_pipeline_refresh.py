"""Tests for Phase 4 corpus runner, merge_corrections, canaries, Parakeet factory."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


class TestTheologicalCanaries:
    def test_eight_canaries_including_partimiento(self):
        from training.theological_canaries import THEOLOGICAL_CANARIES, canary_sentences

        assert len(THEOLOGICAL_CANARIES) == 8
        assert len(canary_sentences()) == 8
        assert len(canary_sentences(5)) == 5
        joined = " ".join(c["en"].lower() for c in THEOLOGICAL_CANARIES)
        assert "breaking of bread" in joined
        assert any("partimiento" in s for c in THEOLOGICAL_CANARIES for s in c["expected_substrings"])


class TestPhase4Corpus:
    def test_discover_and_dry_run_status(self, tmp_path):
        from training.run_phase4_corpus import discover_wavs, main, write_status

        raw = tmp_path / "raw"
        raw.mkdir()
        (raw / "a.wav").write_bytes(b"RIFF")
        sub = raw / "gospel"
        sub.mkdir()
        (sub / "b.wav").write_bytes(b"RIFF")

        found = discover_wavs(raw)
        assert len(found) == 2

        out = tmp_path / "cleaned"
        path = write_status(
            out,
            input_dir=raw,
            discovered=2,
            completed=0,
            rejected=0,
            errors=0,
            elapsed_s=0.0,
            dry_run=True,
        )
        data = json.loads(path.read_text())
        assert data["phase"] == 4
        assert data["discovered_wavs"] == 2
        assert data["dry_run"] is True

        rc = main(["--input", str(raw), "--output", str(out), "--dry-run"])
        assert rc == 0
        assert (out / "phase4_status.json").exists()


class TestMergeCorrections:
    def test_merge_translation_dedupe(self, tmp_path):
        from tools.merge_corrections import merge_translation

        train = tmp_path / "train.jsonl"
        train.write_text('{"en": "Hello", "es": "Hola"}\n', encoding="utf-8")
        corr = tmp_path / "corr.jsonl"
        corr.write_text(
            '{"en": "Hello", "es": "Hola"}\n{"en": "Grace", "es": "Gracia"}\n',
            encoding="utf-8",
        )
        summary = merge_translation(corr, train)
        assert summary["added"] == 1
        assert summary["after"] == 2

    def test_merge_whisper_copies_wavs(self, tmp_path):
        from tools.merge_corrections import merge_whisper

        corr_dir = tmp_path / "corr"
        corr_dir.mkdir()
        (corr_dir / "clip1.wav").write_bytes(b"fake")
        (corr_dir / "metadata.jsonl").write_text(
            '{"file_name": "clip1.wav", "transcription": "Amen"}\n',
            encoding="utf-8",
        )
        train_dir = tmp_path / "train"
        train_dir.mkdir()
        summary = merge_whisper(corr_dir, train_dir)
        assert summary["added"] == 1
        assert (train_dir / "metadata.jsonl").exists()
        assert any(train_dir.glob("al_*.wav"))


class TestParakeetFactory:
    def test_create_parakeet_engine_unloaded(self):
        from engines.factory import create_stt_engine
        from engines.parakeet_engine import ParakeetEngine

        eng = create_stt_engine(backend="cuda", stt_backend="parakeet")
        assert isinstance(eng, ParakeetEngine)
        assert eng.backend == "parakeet"
        assert not eng._loaded

    def test_load_without_nemo_raises(self, monkeypatch):
        import builtins

        from engines.parakeet_engine import ParakeetEngine

        real_import = builtins.__import__

        def _blocked(name, *args, **kwargs):
            if name.startswith("nemo"):
                raise ImportError("blocked")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _blocked)
        eng = ParakeetEngine()
        with pytest.raises(ImportError, match="nemo_toolkit"):
            eng.load()


class TestDeployDryRun:
    def test_deploy_dry_run_without_versions_fails(self, tmp_path, monkeypatch):
        import tools.deploy_adapters as deploy

        monkeypatch.setattr(deploy, "PROJECT_ROOT", tmp_path)
        monkeypatch.setattr(deploy, "_load_manifest", lambda: {})
        rc = deploy.deploy(
            cycle=1,
            models=["whisper_turbo_ct2"],
            endpoints=["local"],
            dry_run=True,
            skip_health=True,
        )
        assert rc == 1
