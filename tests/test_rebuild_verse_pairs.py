"""Tests for tools/rebuild_verse_pairs.py."""

from __future__ import annotations

import json
import os
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tools import rebuild_verse_pairs as rvp


def _make_db(path: Path, prefix: str, verses: list[tuple[int, str]]) -> None:
    con = sqlite3.connect(str(path))
    con.execute(f"CREATE TABLE {prefix}_verses (id INTEGER PRIMARY KEY, text TEXT)")
    con.executemany(f"INSERT INTO {prefix}_verses VALUES (?, ?)", verses)
    con.commit()
    con.close()


class TestLoaders:
    def test_missing_db(self, tmp_path: Path):
        assert rvp.load_sqlite_verses(tmp_path / "gone.db", "KJV") == {}

    def test_load_sqlite(self, tmp_path: Path):
        db = tmp_path / "KJV.db"
        _make_db(db, "KJV", [(1, "In the beginning"), (2, "hi")])  # "hi" filtered by LENGTH>5
        verses = rvp.load_sqlite_verses(db, "KJV")
        assert verses == {1: "In the beginning"}

    def test_load_platense(self, tmp_path: Path):
        p = tmp_path / "plat.jsonl"
        p.write_text(
            json.dumps({"verse_id": 1, "text": "En el principio"})
            + "\n"
            + json.dumps({"verse_id": 2, "text": "Y dijo"})
            + "\n"
        )
        assert rvp.load_realigned_platense(p) == {1: "En el principio", 2: "Y dijo"}


class TestBuild:
    def test_cartesian_join(self, tmp_path: Path):
        scroll = tmp_path / "scroll"
        scroll.mkdir()
        _make_db(scroll / "KJV.db", "KJV", [(1, "In the beginning God")])
        _make_db(scroll / "BBE.db", "BBE", [(1, "At the first God")])
        _make_db(scroll / "SpaRV.db", "SpaRV", [(1, "En el principio Dios")])
        plat = tmp_path / "plat.jsonl"
        plat.write_text(json.dumps({"verse_id": 1, "text": "En el comienzo Dios"}) + "\n")
        out = tmp_path / "pairs.jsonl"
        report = rvp.build(scroll, plat, out)
        assert report["total_pairs_new"] >= 2  # KJV x 2 ES + BBE x 2 ES
        rows = [json.loads(l) for l in out.read_text().splitlines()]
        assert all({"en", "es", "verse_id", "en_source", "es_source"} <= set(r) for r in rows)

    def test_no_data_exits(self, tmp_path: Path):
        plat = tmp_path / "plat.jsonl"
        plat.write_text("")
        with pytest.raises(SystemExit):
            # empty platense still loads as {}, and no EN dbs
            rvp.build(tmp_path / "empty", plat, tmp_path / "out.jsonl")


class TestMain:
    def test_main_missing_platense(self, tmp_path: Path):
        with pytest.raises(SystemExit):
            rvp.main(
                [
                    "--scrollmapper-dir",
                    str(tmp_path),
                    "--platense-realigned",
                    str(tmp_path / "missing.jsonl"),
                    "--output",
                    str(tmp_path / "out.jsonl"),
                ]
            )

    def test_main_happy(self, tmp_path: Path):
        scroll = tmp_path / "scroll"
        scroll.mkdir()
        import sqlite3

        def make(name: str, prefix: str, text: str):
            db = scroll / f"{name}.db"
            con = sqlite3.connect(str(db))
            con.execute(f"CREATE TABLE {prefix}_verses (id INTEGER PRIMARY KEY, text TEXT)")
            con.execute(f"INSERT INTO {prefix}_verses VALUES (1, ?)", (text,))
            con.commit()
            con.close()

        make("KJV", "KJV", "In the beginning God created")
        make("SpaRV", "SpaRV", "En el principio creo Dios")
        plat = tmp_path / "plat.jsonl"
        plat.write_text(json.dumps({"verse_id": 1, "text": "En el comienzo"}) + "\n")
        out = tmp_path / "pairs.jsonl"
        rc = rvp.main(
            [
                "--scrollmapper-dir",
                str(scroll),
                "--platense-realigned",
                str(plat),
                "--output",
                str(out),
            ]
        )
        assert rc == 0
        assert out.exists()
