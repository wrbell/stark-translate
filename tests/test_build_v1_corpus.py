"""Tests for tools/build_v1_corpus.py helpers and build()."""

from __future__ import annotations

import json
import os
import random
import sys
from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tools import build_v1_corpus as bvc


class TestJsonl:
    def test_load_skips_bad(self, tmp_path: Path):
        p = tmp_path / "x.jsonl"
        p.write_text('{"en":"a","es":"b"}\nnot-json\n{"en":"c","es":"d"}\n\n')
        rows = bvc.load_jsonl(p)
        assert len(rows) == 2

    def test_write(self, tmp_path: Path):
        p = tmp_path / "sub" / "out.jsonl"
        bvc.write_jsonl(p, [{"en": "a", "es": "b"}])
        assert json.loads(p.read_text().splitlines()[0])["en"] == "a"


class TestToTrainingRowAndSampleBible:
    def test_to_training_row(self):
        row = bvc.to_training_row(
            {"en": "a", "es": "b", "verse_id": 1, "en_source": "bbe", "es_source": "rvr1909"},
            "bible",
            "_tagged",
        )
        assert row["en"] == "a"
        assert row["_provenance"]["source"] == "bible_tagged"
        assert row["_provenance"]["verse_id"] == 1

    def test_sample_bible(self, tmp_path: Path):
        rows = []
        for i in range(20):
            rows.append(
                {
                    "en": f"e{i}",
                    "es": f"s{i}",
                    "en_source": "bbe" if i < 14 else "kjv",
                    "verse_id": i,
                }
            )
        p = tmp_path / "train.jsonl"
        bvc.write_jsonl(p, rows)
        sampled = bvc.sample_bible(p, n_target=10, rng=random.Random(0))
        assert len(sampled) == 10
        modern = sum(1 for r in sampled if r["en_source"] == "bbe")
        assert modern >= 5


class TestSampleOpusAndAnnotate:
    def test_opus_import_missing(self, monkeypatch: pytest.MonkeyPatch):
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "datasets":
                raise ImportError("no")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        assert bvc.sample_opus(5, random.Random(0)) == []

    def test_annotate_with_glossary(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        fake = MagicMock()
        fake.load_glossary.return_value = {"grace": "gracia"}
        fake.annotate_pair.side_effect = lambda en, es, g, t: (f"<g>{en}</g>", 1)
        monkeypatch.setitem(sys.modules, "glossary_annotate", fake)
        pairs = [{"en": "grace", "es": "gracia"}, {"en": "x", "es": "y"}]
        gloss = tmp_path / "g.json"
        gloss.write_text("{}")
        tagged = bvc.annotate_with_glossary(pairs, gloss, random.Random(0), n_target=1)
        assert len(tagged) == 1
        assert tagged[0]["_glossary_tags"] == 1


class TestBuildAndMain:
    def test_bad_ratios(self):
        args = Namespace(
            ratio_bible=0.5,
            ratio_sermon=0.5,
            ratio_glossary=0.5,
            ratio_opus=0.5,
            seed=0,
        )
        with pytest.raises(SystemExit):
            bvc.build(args)

    def test_build_minimal(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        bible = tmp_path / "bible.jsonl"
        sermon = tmp_path / "sermon.jsonl"
        out = tmp_path / "out.jsonl"
        gloss = tmp_path / "gloss.json"
        gloss.write_text("{}")
        rows = [
            {
                "en": f"e{i}",
                "es": f"s{i}",
                "en_source": "bbe",
                "es_source": "rvr1909",
                "verse_id": i,
                "_kiwi_score": 0.9,
            }
            for i in range(40)
        ]
        bvc.write_jsonl(bible, rows)
        bvc.write_jsonl(sermon, rows[:20])

        fake = MagicMock()
        fake.load_glossary.return_value = {"grace": "gracia"}
        fake.annotate_pair.side_effect = lambda en, es, g, t: (en, 1)
        monkeypatch.setitem(sys.modules, "glossary_annotate", fake)
        monkeypatch.setattr(bvc, "sample_opus", lambda n, rng: [{"en": "op", "es": "op"} for _ in range(max(n, 1))])

        args = Namespace(
            seed=1,
            ratio_bible=0.3,
            ratio_sermon=0.3,
            ratio_glossary=0.3,
            ratio_opus=0.1,
            sermon_path=sermon,
            train_path=bible,
            glossary_path=gloss,
            output=out,
            total_target=20,
            label_suffix="_v1",
        )
        bvc.build(args)
        assert out.exists()
        written = bvc.load_jsonl(out)
        assert len(written) > 0
        assert out.with_suffix(".manifest.json").exists()

    def test_main(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        bible = tmp_path / "bible.jsonl"
        sermon = tmp_path / "sermon.jsonl"
        out = tmp_path / "mixed.jsonl"
        gloss = tmp_path / "g.json"
        gloss.write_text("{}")
        rows = [
            {"en": f"e{i}", "es": f"s{i}", "en_source": "web", "es_source": "platense", "verse_id": i}
            for i in range(30)
        ]
        bvc.write_jsonl(bible, rows)
        bvc.write_jsonl(sermon, rows[:15])
        monkeypatch.setitem(
            sys.modules,
            "glossary_annotate",
            MagicMock(load_glossary=MagicMock(return_value={}), annotate_pair=MagicMock(return_value=("e", 0))),
        )
        monkeypatch.setattr(bvc, "sample_opus", lambda n, rng: [])
        rc = bvc.main(
            [
                "--sermon-path",
                str(sermon),
                "--train-path",
                str(bible),
                "--glossary-path",
                str(gloss),
                "--output",
                str(out),
                "--total-target",
                "10",
                "--seed",
                "2",
            ]
        )
        assert rc == 0
        assert out.exists()
