"""Additional coverage: translation_qe, deploy_adapters, score_comet22, download."""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestTranslationQE:
    def test_length_ratio(self):
        from tools.translation_qe import qe_length_ratio

        assert qe_length_ratio("", "x") == 0.0
        assert qe_length_ratio("hello", "") == 0.0
        assert qe_length_ratio("hello", "hola!!") == 1.0  # similar length
        assert qe_length_ratio("ab", "x" * 100) == 0.1

    def test_untranslated(self):
        from tools.translation_qe import qe_untranslated

        assert qe_untranslated("x", "") == 0.0
        assert qe_untranslated("x", "gracia misericordia") == 1.0
        # Heavy English stopwords in "translation"
        assert qe_untranslated("x", "the and of that have for not with") < 0.5

    def test_tier1(self):
        from tools.translation_qe import tier1_score

        r = tier1_score("Grace and mercy", "Gracia y misericordia")
        assert 0.0 <= r["tier1"] <= 1.0
        assert "length_ratio" in r and "untranslated" in r

    def test_process_csv_tier1(self, tmp_path: Path):
        from tools import translation_qe as tq

        csv_path = tmp_path / "sess.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["english", "spanish_a", "spanish_b"])
            w.writeheader()
            w.writerow({"english": "Grace", "spanish_a": "Gracia", "spanish_b": "Favor"})
            w.writerow({"english": "Hello", "spanish_a": "Hola", "spanish_b": "Hola"})
        out = tmp_path / "qe.jsonl"
        tq.process_csv(str(csv_path), tiers=[1], output_path=str(out))
        assert out.exists()
        lines = out.read_text().strip().splitlines()
        assert len(lines) >= 1

    def test_main_tier1(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        from tools import translation_qe as tq

        csv_path = tmp_path / "m.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["english", "spanish_a"])
            w.writeheader()
            w.writerow({"english": "a", "spanish_a": "b"})
        out = tmp_path / "o.jsonl"
        monkeypatch.setattr(sys, "argv", ["translation_qe.py", str(csv_path), "--tiers", "1", "--output", str(out)])
        tq.main()
        assert out.exists()


class TestDeployAdapters:
    def test_endpoints_merge(self):
        from tools.deploy_adapters import _endpoints_from_manifest

        eps = _endpoints_from_manifest({"endpoints": {"custom": {"host": "1.2.3.4"}}})
        assert "local" in eps
        assert eps["custom"]["host"] == "1.2.3.4"

    def test_rsync_or_copy(self, tmp_path: Path):
        from tools.deploy_adapters import _rsync_or_copy

        src = tmp_path / "src"
        src.mkdir()
        (src / "a.txt").write_text("hi")
        dest = tmp_path / "dest"
        _rsync_or_copy(src, dest, dry_run=True)
        assert not dest.exists()
        _rsync_or_copy(src, dest, dry_run=False)
        assert (dest / "a.txt").read_text() == "hi"
        # overwrite existing
        (src / "a.txt").write_text("bye")
        _rsync_or_copy(src, dest, dry_run=False)
        assert (dest / "a.txt").read_text() == "bye"
        # file copy
        fsrc = tmp_path / "one.bin"
        fsrc.write_bytes(b"zz")
        fdest = tmp_path / "nested" / "one.bin"
        _rsync_or_copy(fsrc, fdest, dry_run=False)
        assert fdest.read_bytes() == b"zz"

    def test_activate_local(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        from tools import deploy_adapters as da

        monkeypatch.setattr(da, "PROJECT_ROOT", tmp_path)
        ver = tmp_path / "adapters" / "gemma_4b" / "v1"
        ver.mkdir(parents=True)
        (ver / "w.txt").write_text("w")
        manifest = {
            "gemma_4b": {
                "active": None,
                "previous": None,
                "versions": {"v1": {"path": str(ver)}, "v2": {"path": str(ver)}},
            }
        }
        da._activate_local(manifest, "gemma_4b", "v1")
        assert manifest["gemma_4b"]["active"] == "v1"
        da._activate_local(manifest, "gemma_4b", "v2")
        assert manifest["gemma_4b"]["previous"] == "v1"

    def test_main_help(self):
        from tools.deploy_adapters import main

        with pytest.raises(SystemExit) as exc:
            main(["--help"])
        assert exc.value.code == 0

    def test_deploy_dry_run_and_rollback(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        from tools import deploy_adapters as da
        import tools.manage_adapters as ma

        monkeypatch.setattr(da, "PROJECT_ROOT", tmp_path)
        monkeypatch.setattr(ma, "MANIFEST_PATH", tmp_path / "adapters" / "manifest.json")
        # deploy_adapters imports _load_manifest from manage_adapters at module level
        monkeypatch.setattr(da, "_load_manifest", ma._load_manifest)
        monkeypatch.setattr(da, "_save_manifest", ma._save_manifest)

        ver = tmp_path / "runs" / "w1"
        ver.mkdir(parents=True)
        (ver / "model.bin").write_text("x")
        manifest = {
            "whisper_turbo_ct2": {
                "active": "w1",
                "previous": None,
                "versions": {"w1": {"path": str(ver)}},
            }
        }
        ma._save_manifest(manifest)

        rc = da.deploy(
            cycle=1,
            models=["whisper_turbo_ct2"],
            endpoints=["local"],
            dry_run=True,
            skip_health=True,
        )
        assert rc == 0

        # Set previous for rollback
        m = ma._load_manifest()
        m["whisper_turbo_ct2"]["previous"] = "w1"
        m["whisper_turbo_ct2"]["active"] = "w1"
        ma._save_manifest(m)
        assert da.rollback(models=["whisper_turbo_ct2"], dry_run=True) == 0
        assert da.rollback(models=["nope_model"], dry_run=True) == 1


class TestScoreComet22Main:
    def test_full_main_mocked(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        from tools import score_comet22 as sc

        def row(hyp: str) -> str:
            return json.dumps({"en": "Grace", "es_hyp": hyp, "es_ref": "Gracia"})

        a = tmp_path / "a.jsonl"
        b = tmp_path / "b.jsonl"
        hyps = ["Gracia", "Favor", "Gracia", "X"]
        a.write_text("\n".join(row(h) for h in hyps) + "\n")
        b.write_text("\n".join(row(h) for h in ["Gracia", "Gracia", "Bien", "Y"]) + "\n")

        fake_model = MagicMock()
        fake_model.predict.side_effect = [
            MagicMock(scores=[0.8, 0.5, 0.7, 0.6]),
            MagicMock(scores=[0.9, 0.85, 0.75, 0.55]),
        ]
        fake_comet = MagicMock()
        fake_comet.download_model.return_value = "ckpt"
        fake_comet.load_from_checkpoint.return_value = fake_model
        monkeypatch.setitem(sys.modules, "comet", fake_comet)
        monkeypatch.setattr(sc, "chrf_pp", lambda h, r: 50.0)

        out = tmp_path / "out.json"
        rc = sc.main(["--a", str(a), "--b", str(b), "--a-label", "prod", "--b-label", "v1", "--out", str(out)])
        assert rc == 0
        summary = json.loads(out.read_text())
        assert summary["n"] == 4
        assert "delta" in summary


class TestDownloadRoundtripMain:
    def test_main(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys):
        from tools import download_roundtrip_texts as drt

        monkeypatch.setattr(drt, "TEXTS_DIR", tmp_path)
        monkeypatch.setattr(
            drt,
            "fetch_gutenberg",
            lambda url: (
                "When in the Course of human events "
                + ("word " * 100)
                + " There is no want of power in God "
                + ("sermon " * 100)
            ),
        )
        drt.main()
        assert (tmp_path / "declaration.txt").exists()
        assert (tmp_path / "sermon.txt").exists()
        assert "Done" in capsys.readouterr().out


class TestScoreWithComet:
    def test_score_with_comet(self, monkeypatch: pytest.MonkeyPatch):
        from tools import score_comet22 as sc

        fake_model = MagicMock()
        fake_model.predict.return_value = MagicMock(scores=[0.1, 0.2])
        fake_comet = MagicMock()
        fake_comet.download_model.return_value = "ckpt"
        fake_comet.load_from_checkpoint.return_value = fake_model
        monkeypatch.setitem(sys.modules, "comet", fake_comet)
        scores = sc.score_with_comet(
            [{"src": "a", "mt": "b", "ref": "c"}, {"src": "d", "mt": "e", "ref": "f"}],
            "m",
            8,
        )
        assert scores == [0.1, 0.2]


class TestTranslationQEBacktranslate:
    def test_backtranslate_and_tier2_no_bert(self, monkeypatch: pytest.MonkeyPatch):
        from tools import translation_qe as tq

        tok = MagicMock()
        tok.return_value = {"input_ids": [1]}
        tok.decode.return_value = "grace"
        model = MagicMock()
        model.generate.return_value = [[1, 2, 3]]
        monkeypatch.setattr(tq, "_load_backtranslation", lambda: None)
        monkeypatch.setattr(tq, "_backtranslation_tokenizer", tok)
        monkeypatch.setattr(tq, "_backtranslation_model", model)
        assert tq.backtranslate("gracia") == "grace"
        # bert-score missing → None tier2
        result = tq.tier2_score("grace", "gracia")
        assert result["tier2"] is None


class TestMineHallucinationText:
    def test_main_text_mode(self, tmp_path: Path, capsys, monkeypatch: pytest.MonkeyPatch):
        from tools import mine_hallucination_phrases as mhp

        metrics = tmp_path / "m"
        metrics.mkdir()
        (metrics / "diagnostics_x.jsonl").write_text(
            json.dumps({"english": "brand new hall phrase", "is_hallucination": True})
            + "\n"
            + json.dumps({"english": "brand new hall phrase", "is_hallucination": True})
            + "\n"
            + json.dumps({"english": "brand new hall phrase", "is_hallucination": True})
            + "\n"
        )
        monkeypatch.setattr(mhp, "_HALLUCINATION_PHRASES", {})
        assert mhp.main(["--metrics-dir", str(metrics), "--min-count", "2"]) == 0
        out = capsys.readouterr().out
        assert "Suggested" in out or "brand new" in out
