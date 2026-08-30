"""Tests for tools/build_eval_sets.py — stratified holdout + sermon eval."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tools import build_eval_sets as bes


class TestSha256AndJsonl:
    def test_sha256(self, tmp_path: Path):
        p = tmp_path / "f.bin"
        p.write_bytes(b"abc")
        assert bes.sha256_file(p) == hashlib.sha256(b"abc").hexdigest()

    def test_load_save_roundtrip(self, tmp_path: Path):
        p = tmp_path / "rows.jsonl"
        rows = [{"en": "a", "es": "b"}, {"en": "c", "es": "d"}]
        bes.save_jsonl(p, rows)
        # blank line should be ignored on load
        with open(p, "a") as f:
            f.write("\n")
        assert bes.load_jsonl(p) == rows


class TestClassifyVerseBucket:
    def test_none_unknown(self):
        assert bes.classify_verse_bucket({}) == "unknown"

    def test_int_buckets(self):
        assert bes.classify_verse_bucket({"verse_id": 1}) == "pentateuch"
        assert bes.classify_verse_bucket({"verse_id": 6000}) == "history"
        assert bes.classify_verse_bucket({"verse_id": 14000}) == "poetry"
        assert bes.classify_verse_bucket({"verse_id": 20000}) == "prophets"
        assert bes.classify_verse_bucket({"verse_id": 24000}) == "gospels_acts"
        assert bes.classify_verse_bucket({"verse_id": 30000}) == "epistles_rev"

    def test_string_book_prefix(self):
        assert bes.classify_verse_bucket({"verse_id": "01001001"}) == "pentateuch"
        assert bes.classify_verse_bucket({"verse_id": "06001001"}) == "history"
        assert bes.classify_verse_bucket({"verse_id": "19001001"}) == "poetry"
        assert bes.classify_verse_bucket({"verse_id": "23001001"}) == "prophets"
        assert bes.classify_verse_bucket({"verse_id": "40001001"}) == "gospels_acts"
        assert bes.classify_verse_bucket({"verse_id": "45001001"}) == "epistles_rev"

    def test_bad_string(self):
        assert bes.classify_verse_bucket({"verse_id": "xx"}) == "unknown"
        assert bes.classify_verse_bucket({"verse_id": 3.14}) == "unknown"


def _write_train(path: Path, n: int = 60) -> None:
    rows = []
    # Spread across int verse_id buckets
    ids = [100, 7000, 15000, 20000, 25000, 30000]
    for i in range(n):
        rows.append(
            {
                "en": f"en{i}",
                "es": f"es{i}",
                "verse_id": ids[i % len(ids)] + i,
            }
        )
    bes.save_jsonl(path, rows)


def _patch_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, **paths: Path) -> None:
    """Point build_eval_sets paths at tmp_path (must be under _PROJECT_ROOT for relative_to)."""
    monkeypatch.setattr(bes, "_PROJECT_ROOT", tmp_path)
    for name, path in paths.items():
        monkeypatch.setattr(bes, name, path)


class TestBuildTranslationHoldout:
    def test_missing_train(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        _patch_paths(monkeypatch, tmp_path, TRAIN_PATH=tmp_path / "missing.jsonl")
        assert bes.build_translation_holdout(10, seed=1, dry_run=True) is None

    def test_too_many_requested(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        train = tmp_path / "train.jsonl"
        _write_train(train, n=5)
        _patch_paths(monkeypatch, tmp_path, TRAIN_PATH=train)
        assert bes.build_translation_holdout(100, seed=1, dry_run=True) is None

    def test_dry_run_no_writes(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        train = tmp_path / "train.jsonl"
        test = tmp_path / "test.jsonl"
        _write_train(train, n=60)
        _patch_paths(monkeypatch, tmp_path, TRAIN_PATH=train, TEST_PATH=test)
        meta = bes.build_translation_holdout(12, seed=42, dry_run=True)
        assert meta is not None
        assert meta["count"] == 12
        assert not test.exists()
        assert len(bes.load_jsonl(train)) == 60

    def test_live_split(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        train = tmp_path / "train.jsonl"
        test = tmp_path / "test.jsonl"
        _write_train(train, n=60)
        _patch_paths(monkeypatch, tmp_path, TRAIN_PATH=train, TEST_PATH=test)
        meta = bes.build_translation_holdout(12, seed=7, dry_run=False)
        assert meta is not None
        assert meta["count"] == 12
        assert test.exists()
        assert "sha256" in meta
        assert len(bes.load_jsonl(test)) == 12
        assert len(bes.load_jsonl(train)) == 48

    def test_seed_stable(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        train = tmp_path / "train.jsonl"
        test = tmp_path / "test.jsonl"
        _write_train(train, n=60)
        _patch_paths(monkeypatch, tmp_path, TRAIN_PATH=train, TEST_PATH=test)
        m1 = bes.build_translation_holdout(10, seed=99, dry_run=True)
        m2 = bes.build_translation_holdout(10, seed=99, dry_run=True)
        assert m1["count"] == m2["count"] == 10


class TestBuildSermonEval:
    def test_missing_chunks(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        _patch_paths(monkeypatch, tmp_path, SERMON_CHUNKS_PATH=tmp_path / "gone.json")
        assert bes.build_sermon_eval(dry_run=True) is None

    def test_dry_run_and_live(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        chunks = tmp_path / "chunks.json"
        out = tmp_path / "eval.json"
        chunks.write_text(
            json.dumps(
                [
                    {"id": 1, "split": "eval", "text": "a"},
                    {"id": 2, "split": "train", "text": "b"},
                    {"id": 3, "split": "eval", "text": "c"},
                ]
            )
        )
        _patch_paths(monkeypatch, tmp_path, SERMON_CHUNKS_PATH=chunks, SERMON_EVAL_PATH=out)
        meta = bes.build_sermon_eval(dry_run=True)
        assert meta is not None
        assert meta["count"] == 2
        assert not out.exists()

        meta2 = bes.build_sermon_eval(dry_run=False)
        assert out.exists()
        data = json.loads(out.read_text())
        assert len(data) == 2
        assert meta2["sha256"]


class TestSaveRegistryAndMain:
    def test_save_registry_dry_and_live(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        reg = tmp_path / "eval_registry.json"
        monkeypatch.setattr(bes, "REGISTRY_PATH", reg)
        bes.save_registry([{"name": "x"}], dry_run=True)
        assert not reg.exists()
        bes.save_registry([{"name": "x"}], dry_run=False)
        data = json.loads(reg.read_text())
        assert data["eval_sets"] == [{"name": "x"}]

    def test_main_dry_run(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        train = tmp_path / "train.jsonl"
        test = tmp_path / "test.jsonl"
        chunks = tmp_path / "chunks.json"
        eval_out = tmp_path / "sermon_eval.json"
        reg = tmp_path / "registry.json"
        _write_train(train, n=40)
        chunks.write_text(json.dumps([{"split": "eval", "t": 1}, {"split": "train", "t": 2}]))
        _patch_paths(
            monkeypatch,
            tmp_path,
            TRAIN_PATH=train,
            TEST_PATH=test,
            SERMON_CHUNKS_PATH=chunks,
            SERMON_EVAL_PATH=eval_out,
            REGISTRY_PATH=reg,
        )
        monkeypatch.setattr(
            sys,
            "argv",
            ["build_eval_sets.py", "--dry-run", "--verse-count", "6", "--seed", "1"],
        )
        assert bes.main() == 0
        assert not test.exists()
        assert not reg.exists()
