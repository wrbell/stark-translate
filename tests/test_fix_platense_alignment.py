"""Tests for tools/fix_platense_alignment.py — token Jaccard + SQLite realign."""

from __future__ import annotations

import json
import os
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tools import fix_platense_alignment as fpa


class TestContentTokensAndJaccard:
    def test_drops_stopwords_and_short(self):
        toks = fpa.content_tokens("El Señor es mi pastor bueno")
        assert "señor" in toks or "senor" in toks or "pastor" in toks
        assert "el" not in toks
        assert "es" not in toks
        assert "mi" not in toks

    def test_jaccard_edges(self):
        assert fpa.jaccard(set(), {"a"}) == 0.0
        assert fpa.jaccard({"a"}, set()) == 0.0
        assert fpa.jaccard({"a", "b"}, {"a", "b"}) == 1.0
        assert fpa.jaccard({"a", "b"}, {"b", "c"}) == pytest.approx(1 / 3)


def _make_bible_db(path: Path, prefix: str, books: list[tuple[int, str]], verses: list[tuple]) -> None:
    con = sqlite3.connect(str(path))
    con.execute(f"CREATE TABLE {prefix}_books (id INTEGER PRIMARY KEY, name TEXT)")
    con.execute(
        f"CREATE TABLE {prefix}_verses (id INTEGER PRIMARY KEY, book_id INTEGER, chapter INTEGER, verse INTEGER, text TEXT)"
    )
    con.executemany(f"INSERT INTO {prefix}_books VALUES (?, ?)", books)
    con.executemany(
        f"INSERT INTO {prefix}_verses VALUES (?, ?, ?, ?, ?)",
        verses,
    )
    con.commit()
    con.close()


class TestLoadBooksVerses:
    def test_load(self, tmp_path: Path):
        db = tmp_path / "t.db"
        _make_bible_db(
            db,
            "SpaRV",
            [(1, "Genesis")],
            [(1, 1, 1, 1, "En el principio creó Dios"), (2, 1, 1, 2, "short")],
        )
        con = sqlite3.connect(str(db))
        books = fpa.load_books(con, "SpaRV")
        verses = fpa.load_verses(con, "SpaRV")
        con.close()
        assert books == {1: "Genesis"}
        # length <= 5 filtered
        assert len(verses) == 1
        assert verses[0]["text"].startswith("En el")


class TestRealign:
    def test_missing_dbs(self, tmp_path: Path):
        with pytest.raises(SystemExit):
            fpa.realign(tmp_path, tmp_path / "out.jsonl", tmp_path / "report.json")

    def test_realign_happy_path(self, tmp_path: Path):
        scroll = tmp_path / "scroll"
        scroll.mkdir()
        # Shared Genesis 1:1 with overlapping content words
        shared = "principio dios creó cielos tierra creación divina"
        _make_bible_db(
            scroll / "SpaRV.db",
            "SpaRV",
            [(1, "Genesis"), (2, "Exodo")],
            [
                (10, 1, 1, 1, shared),
                (11, 1, 1, 2, "y la tierra estaba desordenada vacía"),
            ],
        )
        _make_bible_db(
            scroll / "SpaPlatense.db",
            "SpaPlatense",
            [(1, "Genesis"), (99, "Tobit")],  # apocrypha
            [
                (100, 1, 1, 1, shared + " extra"),
                (101, 1, 1, 2, "y la tierra estaba desordenada vacía total"),
                (102, 99, 1, 1, "libro apocrifo largo texto suficiente"),
                (103, 1, 9, 9, "versiculo sin pareja canonica en rvr"),
            ],
        )
        out = tmp_path / "platense_realigned.jsonl"
        report_path = tmp_path / "report.json"
        report = fpa.realign(scroll, out, report_path, validate_threshold=0.05)
        assert out.exists()
        rows = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
        assert len(rows) >= 1
        assert all("verse_id" in r for r in rows)
        assert report["apocryphal_verses_dropped"] >= 1
        assert report_path.exists()

    def test_content_validator_drops_low_jaccard(self, tmp_path: Path):
        scroll = tmp_path / "scroll"
        scroll.mkdir()
        _make_bible_db(
            scroll / "SpaRV.db",
            "SpaRV",
            [(1, "Genesis")],
            [(10, 1, 1, 1, "principio dios cielos tierra creación")],
        )
        _make_bible_db(
            scroll / "SpaPlatense.db",
            "SpaPlatense",
            [(1, "Genesis")],
            [(100, 1, 1, 1, "zzzzzzzz yyyyyyyy xxxxxxxx wwwwwwww")],
        )
        out = tmp_path / "out.jsonl"
        report = fpa.realign(scroll, out, tmp_path / "r.json", validate_threshold=0.5)
        assert report["validation_dropped_total"] == 1
        assert report["realigned_output_verses"] == 0


class TestMain:
    def test_main_calls_realign(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        scroll = tmp_path / "scroll"
        scroll.mkdir()
        text = "principio dios creó cielos tierra creación divina"
        _make_bible_db(
            scroll / "SpaRV.db",
            "SpaRV",
            [(1, "Genesis")],
            [(1, 1, 1, 1, text)],
        )
        _make_bible_db(
            scroll / "SpaPlatense.db",
            "SpaPlatense",
            [(1, "Genesis")],
            [(1, 1, 1, 1, text)],
        )
        out = tmp_path / "out.jsonl"
        report = tmp_path / "report.json"
        rc = fpa.main(
            [
                "--scrollmapper-dir",
                str(scroll),
                "--output",
                str(out),
                "--report",
                str(report),
                "--validate-threshold",
                "0.05",
            ]
        )
        assert rc == 0
        assert out.exists()
