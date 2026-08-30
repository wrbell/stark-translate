"""Tests for tools/build_preference_triples.py (mocked HTTP / Comet)."""

from __future__ import annotations

import json
import os
import sys
from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tools import build_preference_triples as bpt


class TestBuildUserPrompt:
    def test_known_langs(self):
        p = bpt.build_user_prompt("Grace", "en", "es")
        assert "English" in p and "Spanish" in p and "Grace" in p

    def test_unknown_codes(self):
        p = bpt.build_user_prompt("x", "xx", "yy")
        assert "xx" in p and "yy" in p


class TestCallLlamaServer:
    def test_parses_response(self):
        payload = {
            "choices": [{"message": {"content": "  Hola  "}}],
            "usage": {"completion_tokens": 3},
        }
        resp = MagicMock()
        resp.read.return_value = json.dumps(payload).encode()
        resp.__enter__ = MagicMock(return_value=resp)
        resp.__exit__ = MagicMock(return_value=False)
        with patch("urllib.request.urlopen", return_value=resp):
            text, ntok, lat = bpt.call_llama_server("http://localhost:8092", "prompt", 0.7, 50, 5.0)
        assert text == "Hola"
        assert ntok == 3
        assert lat >= 0


class TestCmdGenerate:
    def test_generate(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        sources = tmp_path / "src.jsonl"
        sources.write_text(
            json.dumps({"en": "Grace abounds", "verse_id": 1})
            + "\n"
            + json.dumps({"no_en": True})
            + "\n"
            + json.dumps({"en": "Mercy"})
            + "\n"
        )
        out = tmp_path / "cands.jsonl"
        monkeypatch.setattr(
            bpt,
            "call_llama_server",
            lambda *a, **k: ("traducción", 5, 12.0),
        )
        args = Namespace(
            sources=sources,
            output=out,
            candidates=3,
            temperature=0.7,
            max_tokens=50,
            timeout_s=5.0,
            server_url="http://localhost:8092",
            limit=0,
        )
        assert bpt.cmd_generate(args) == 0
        rows = [json.loads(l) for l in out.read_text().splitlines()]
        assert len(rows) == 2
        assert len(rows[0]["candidates"]) == 3
        assert rows[0]["verse_id"] == 1

    def test_generate_skips_on_failures(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        sources = tmp_path / "src.jsonl"
        sources.write_text(json.dumps({"en": "x"}) + "\n")
        out = tmp_path / "cands.jsonl"

        def boom(*a, **k):
            raise bpt.urllib.error.URLError("down")

        monkeypatch.setattr(bpt, "call_llama_server", boom)
        args = Namespace(
            sources=sources,
            output=out,
            candidates=2,
            temperature=0.7,
            max_tokens=50,
            timeout_s=1.0,
            server_url="http://x",
            limit=1,
        )
        assert bpt.cmd_generate(args) == 0
        assert out.read_text() == ""


class TestCmdScore:
    def test_score_keeps_and_drops(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        cands = tmp_path / "c.jsonl"
        rows = [
            {
                "en": "Grace",
                "candidates": [{"text": "Gracia"}, {"text": "Favor"}, {"text": "Gracia"}],
            },
            {
                "en": "Same",
                "candidates": [{"text": "Igual"}, {"text": "Igual"}],
            },
            {
                "en": "Close",
                "candidates": [{"text": "A"}, {"text": "B"}],
            },
        ]
        cands.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        out = tmp_path / "triples.jsonl"

        # Scores flattened in candidate order: row0: 0.9, 0.5, 0.9; row1: 0.8, 0.8; row2: 0.51, 0.50
        flat = [0.9, 0.5, 0.9, 0.8, 0.8, 0.51, 0.50]
        fake_model = MagicMock()
        fake_model.predict.return_value = MagicMock(scores=flat)
        fake_comet = MagicMock()
        fake_comet.download_model.return_value = "ckpt"
        fake_comet.load_from_checkpoint.return_value = fake_model
        monkeypatch.setitem(sys.modules, "comet", fake_comet)

        args = Namespace(
            candidates=cands,
            output=out,
            model="fake",
            batch_size=8,
            margin=0.05,
        )
        assert bpt.cmd_score(args) == 0
        triples = [json.loads(l) for l in out.read_text().splitlines()]
        # First kept (margin 0.4), second dropped identical, third dropped low margin
        assert len(triples) == 1
        assert triples[0]["chosen"] == "Gracia"
        assert triples[0]["rejected"] == "Favor"


class TestMain:
    def test_main_generate(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        sources = tmp_path / "s.jsonl"
        sources.write_text(json.dumps({"en": "Hi"}) + "\n")
        out = tmp_path / "o.jsonl"
        monkeypatch.setattr(bpt, "call_llama_server", lambda *a, **k: ("Hola", 1, 1.0))
        rc = bpt.main(
            [
                "generate",
                "--sources",
                str(sources),
                "--output",
                str(out),
                "--candidates",
                "2",
            ]
        )
        assert rc == 0
