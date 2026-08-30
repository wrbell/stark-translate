"""Tests for mine_hallucination, live_diarize, diarize helpers, score_comet22, download_roundtrip."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# mine_hallucination_phrases
# ---------------------------------------------------------------------------


class TestMineHallucination:
    def test_load_and_candidates(self, tmp_path: Path):
        from tools import mine_hallucination_phrases as mhp

        metrics = tmp_path / "metrics"
        metrics.mkdir()
        path = metrics / "diagnostics_2026.jsonl"
        rows = [
            {"event": "start"},
            {"english": "thank you thank you", "is_hallucination": True},
            {"english": "grace and mercy", "review_priority": 3},
            {"english": "ok speech here", "stt_confidence": 0.9},
            {"english": "subtitles", "stt_confidence": 0.4},
            "not-json",
            {"english": ""},
        ]
        with open(path, "w") as f:
            for r in rows:
                f.write((r if isinstance(r, str) else json.dumps(r)) + "\n")
        recs = mhp.load_records(metrics)
        assert all("event" not in r for r in recs)
        counts = mhp.candidate_phrases(recs)
        assert len(counts) >= 2

    def test_main_json(self, tmp_path: Path, capsys, monkeypatch: pytest.MonkeyPatch):
        from tools import mine_hallucination_phrases as mhp

        metrics = tmp_path / "m"
        metrics.mkdir()
        (metrics / "diagnostics_x.jsonl").write_text(
            json.dumps({"english": "unique weird phrase", "is_hallucination": True})
            + "\n"
            + json.dumps({"english": "unique weird phrase", "is_hallucination": True})
            + "\n"
        )
        monkeypatch.setattr(mhp, "_HALLUCINATION_PHRASES", {})
        rc = mhp.main(["--metrics-dir", str(metrics), "--min-count", "2", "--json"])
        assert rc == 0
        out = json.loads(capsys.readouterr().out)
        assert out["suggestions"]


# ---------------------------------------------------------------------------
# live_diarize
# ---------------------------------------------------------------------------


class TestLiveDiarize:
    def test_emit_fake_label(self, tmp_path: Path):
        from features import live_diarize as ld

        out = tmp_path / "d.jsonl"
        ld._emit_fake_label(out, 0)
        ld._emit_fake_label(out, 1)
        lines = [json.loads(l) for l in out.read_text().splitlines()]
        assert lines[0]["speaker"] == "Speaker A"
        assert lines[1]["speaker"] == "Speaker B"

    def test_emit_real_label(self, tmp_path: Path):
        from features import live_diarize as ld

        out = tmp_path / "d.jsonl"
        ann = MagicMock()
        ann.itertracks.return_value = [(MagicMock(), None, "SPEAKER_00")]
        run = MagicMock(return_value=ann)
        ld._emit_real_label(out, 3, run, tmp_path / "r.wav")
        data = json.loads(out.read_text().strip())
        assert data["speaker"] == "SPEAKER_00"
        assert data["chunk_id"] == 3

    def test_emit_real_label_failure(self, tmp_path: Path):
        from features import live_diarize as ld

        out = tmp_path / "d.jsonl"
        run = MagicMock(side_effect=RuntimeError("boom"))
        ld._emit_real_label(out, 0, run, tmp_path / "r.wav")
        assert not out.exists() or out.read_text() == ""

    def test_main_fake_iters(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        from features import live_diarize as ld

        monkeypatch.setattr(ld, "_load_pyannote", lambda: None)
        monkeypatch.setattr(ld.time, "sleep", lambda *_: None)
        wav = tmp_path / "rolling.wav"
        wav.write_bytes(b"RIFF")
        out = tmp_path / "out.jsonl"
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "live_diarize.py",
                "--rolling-wav",
                str(wav),
                "--output",
                str(out),
                "--interval-s",
                "0",
                "--max-iters",
                "2",
            ],
        )
        assert ld.main() == 0
        assert len(out.read_text().splitlines()) == 2


# ---------------------------------------------------------------------------
# diarize helpers
# ---------------------------------------------------------------------------


class TestDiarizeHelpers:
    def test_format_timestamp(self):
        from features.diarize import _format_timestamp

        assert _format_timestamp(0) == "00:00:00.000"
        assert _format_timestamp(3661.5).startswith("01:01:01")

    def test_write_jsonl(self, tmp_path: Path, capsys):
        from features.diarize import write_jsonl

        out = tmp_path / "sub" / "d.jsonl"
        results = [
            {"speaker": "A", "start": 0, "end": 1, "duration": 1.0, "text": "hi", "confidence": 0.9},
            {"speaker": "B", "start": 1, "end": 2, "duration": 1.0, "text": "yo", "confidence": 0.8},
        ]
        write_jsonl(results, str(out), metadata={"session": "x"})
        lines = out.read_text().splitlines()
        assert json.loads(lines[0])["_metadata"]["session"] == "x"
        assert len(lines) == 3
        assert "Speakers" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# score_comet22
# ---------------------------------------------------------------------------


class TestScoreComet22:
    def test_load_hyps(self, tmp_path: Path):
        from tools import score_comet22 as sc

        p = tmp_path / "h.jsonl"
        p.write_text(json.dumps({"en": "a", "es_hyp": "b", "es_ref": "c"}) + "\n\n")
        assert len(sc.load_hyps(p)) == 1

    def test_chrf_pp(self, monkeypatch: pytest.MonkeyPatch):
        from tools import score_comet22 as sc

        fake = MagicMock()
        fake.sentence_chrf.return_value = MagicMock(score=55.5)
        monkeypatch.setitem(sys.modules, "sacrebleu", fake)
        assert sc.chrf_pp("hola", "hola") == 55.5

    def test_main_length_mismatch(self, tmp_path: Path):
        from tools import score_comet22 as sc

        a = tmp_path / "a.jsonl"
        b = tmp_path / "b.jsonl"
        a.write_text(json.dumps({"en": "x", "es_hyp": "y", "es_ref": "z"}) + "\n")
        b.write_text("")
        with pytest.raises(SystemExit, match="row count"):
            sc.main(["--a", str(a), "--b", str(b), "--out", str(tmp_path / "o.json")])

    def test_main_src_mismatch(self, tmp_path: Path):
        from tools import score_comet22 as sc

        row = lambda en, hyp: json.dumps({"en": en, "es_hyp": hyp, "es_ref": "r"}) + "\n"
        a = tmp_path / "a.jsonl"
        b = tmp_path / "b.jsonl"
        a.write_text(row("one", "a"))
        b.write_text(row("two", "b"))
        with pytest.raises(SystemExit, match="FATAL"):
            sc.main(["--a", str(a), "--b", str(b), "--out", str(tmp_path / "o.json")])


# ---------------------------------------------------------------------------
# download_roundtrip_texts
# ---------------------------------------------------------------------------


class TestDownloadRoundtrip:
    def test_clean_declaration(self):
        from tools import download_roundtrip_texts as drt

        text = (
            "Preamble junk\nWhen in the Course of human events "
            + ("word " * 50)
            + "He has excited domestic insurrections among us. "
            + ("more " * 1400)
        )
        cleaned = drt.clean_declaration(text)
        assert "excited domestic" not in cleaned
        assert cleaned.startswith("When in the Course")
        assert len(cleaned.split()) <= 1300

    def test_clean_sermon(self):
        from tools import download_roundtrip_texts as drt

        body = "intro " * 20 + "There is no want of power in God " + ("word " * 2000)
        cleaned = drt.clean_sermon(body)
        assert "There is no want of power" in cleaned
        assert len(cleaned.split()) <= 1501

    def test_fetch_gutenberg(self, monkeypatch: pytest.MonkeyPatch):
        from tools import download_roundtrip_texts as drt

        raw = b"HEADER***START***body text here***END***FOOTER"
        fake = MagicMock()
        fake.read.return_value = raw
        fake.__enter__ = MagicMock(return_value=fake)
        fake.__exit__ = MagicMock(return_value=False)
        monkeypatch.setattr(drt, "urlopen", MagicMock(return_value=fake))
        # urlopen is imported into module from urllib.request
        with patch("tools.download_roundtrip_texts.urlopen", MagicMock(return_value=MagicMock(read=MagicMock(return_value=raw)))):
            body = drt.fetch_gutenberg("https://example.com/x.txt")
        assert "body text" in body
