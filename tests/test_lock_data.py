"""Tests for tools/lock_data.py — SHA-256 lockfile generate/verify."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from argparse import Namespace
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tools import lock_data


class TestSha256File:
    def test_known_digest(self, tmp_path: Path):
        p = tmp_path / "a.txt"
        p.write_bytes(b"hello")
        assert lock_data.sha256_file(p) == hashlib.sha256(b"hello").hexdigest()

    def test_empty_file(self, tmp_path: Path):
        p = tmp_path / "empty.bin"
        p.write_bytes(b"")
        assert lock_data.sha256_file(p) == hashlib.sha256(b"").hexdigest()


class TestCountEntries:
    def test_jsonl(self, tmp_path: Path):
        p = tmp_path / "rows.jsonl"
        p.write_text('{"a":1}\n{"b":2}\n{"c":3}\n')
        assert lock_data.count_entries(p) == 3

    def test_json_list(self, tmp_path: Path):
        p = tmp_path / "arr.json"
        p.write_text(json.dumps([1, 2, 3, 4]))
        assert lock_data.count_entries(p) == 4

    def test_json_dict(self, tmp_path: Path):
        p = tmp_path / "obj.json"
        p.write_text(json.dumps({"x": 1, "y": 2}))
        assert lock_data.count_entries(p) == 2

    def test_json_scalar(self, tmp_path: Path):
        p = tmp_path / "n.json"
        p.write_text("42")
        assert lock_data.count_entries(p) == 1

    def test_other_suffix(self, tmp_path: Path):
        p = tmp_path / "notes.txt"
        p.write_text("hi")
        assert lock_data.count_entries(p) == 0


class TestBuildFileEntry:
    def test_missing_returns_none(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(lock_data, "_PROJECT_ROOT", tmp_path)
        assert lock_data.build_file_entry("gone", "missing.jsonl") is None

    def test_jsonl_entry(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(lock_data, "_PROJECT_ROOT", tmp_path)
        rel = "data/rows.jsonl"
        path = tmp_path / rel
        path.parent.mkdir(parents=True)
        path.write_text('{"a":1}\n{"b":2}\n')
        entry = lock_data.build_file_entry("rows", rel)
        assert entry is not None
        assert entry["path"] == rel
        assert entry["lines"] == 2
        assert entry["size_bytes"] == path.stat().st_size
        assert len(entry["sha256"]) == 64

    def test_json_entry(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(lock_data, "_PROJECT_ROOT", tmp_path)
        rel = "data/tier.json"
        path = tmp_path / rel
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"a": 1, "b": 2, "c": 3}))
        entry = lock_data.build_file_entry("tier", rel)
        assert entry is not None
        assert entry["element_count"] == 3


class TestCmdGenerateVerify:
    def test_generate_and_verify_ok(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(lock_data, "_PROJECT_ROOT", tmp_path)
        lock_path = tmp_path / "bible_data" / "data_lockfile.json"
        monkeypatch.setattr(lock_data, "LOCKFILE_PATH", lock_path)
        monkeypatch.setattr(
            lock_data,
            "REGISTERED_FILES",
            {
                "train": "bible_data/train.jsonl",
                "tier": "bible_data/tier.json",
            },
        )
        train = tmp_path / "bible_data" / "train.jsonl"
        train.parent.mkdir(parents=True)
        train.write_text('{"en":"a","es":"b"}\n')
        tier = tmp_path / "bible_data" / "tier.json"
        tier.write_text(json.dumps(["x", "y"]))

        assert lock_data.cmd_generate(Namespace()) == 0
        assert lock_path.exists()
        data = json.loads(lock_path.read_text())
        assert set(data["files"]) == {"train", "tier"}
        assert lock_data.cmd_verify(Namespace()) == 0

    def test_verify_mismatch(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(lock_data, "_PROJECT_ROOT", tmp_path)
        lock_path = tmp_path / "lock.json"
        monkeypatch.setattr(lock_data, "LOCKFILE_PATH", lock_path)
        monkeypatch.setattr(lock_data, "REGISTERED_FILES", {"train": "train.jsonl"})
        f = tmp_path / "train.jsonl"
        f.write_text("one\n")
        assert lock_data.cmd_generate(Namespace()) == 0
        f.write_text("two\n")
        assert lock_data.cmd_verify(Namespace()) == 1

    def test_verify_missing_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(lock_data, "_PROJECT_ROOT", tmp_path)
        lock_path = tmp_path / "lock.json"
        monkeypatch.setattr(lock_data, "LOCKFILE_PATH", lock_path)
        monkeypatch.setattr(lock_data, "REGISTERED_FILES", {"train": "train.jsonl"})
        f = tmp_path / "train.jsonl"
        f.write_text("x\n")
        assert lock_data.cmd_generate(Namespace()) == 0
        f.unlink()
        assert lock_data.cmd_verify(Namespace()) == 1

    def test_verify_no_lockfile(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(lock_data, "LOCKFILE_PATH", tmp_path / "missing.json")
        assert lock_data.cmd_verify(Namespace()) == 1

    def test_generate_skips_missing(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(lock_data, "_PROJECT_ROOT", tmp_path)
        lock_path = tmp_path / "lock.json"
        monkeypatch.setattr(lock_data, "LOCKFILE_PATH", lock_path)
        monkeypatch.setattr(
            lock_data,
            "REGISTERED_FILES",
            {"present": "ok.jsonl", "absent": "gone.jsonl"},
        )
        (tmp_path / "ok.jsonl").write_text("{}\n")
        assert lock_data.cmd_generate(Namespace()) == 0
        data = json.loads(lock_path.read_text())
        assert "present" in data["files"]
        assert "absent" not in data["files"]


class TestMain:
    def test_generate_via_main(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(lock_data, "_PROJECT_ROOT", tmp_path)
        lock_path = tmp_path / "lock.json"
        monkeypatch.setattr(lock_data, "LOCKFILE_PATH", lock_path)
        monkeypatch.setattr(lock_data, "REGISTERED_FILES", {"a": "a.jsonl"})
        (tmp_path / "a.jsonl").write_text("{}\n")
        monkeypatch.setattr(sys, "argv", ["lock_data.py", "generate"])
        assert lock_data.main() == 0

    def test_verify_via_main(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(lock_data, "_PROJECT_ROOT", tmp_path)
        lock_path = tmp_path / "lock.json"
        monkeypatch.setattr(lock_data, "LOCKFILE_PATH", lock_path)
        monkeypatch.setattr(lock_data, "REGISTERED_FILES", {"a": "a.jsonl"})
        (tmp_path / "a.jsonl").write_text("{}\n")
        lock_data.cmd_generate(Namespace())
        monkeypatch.setattr(sys, "argv", ["lock_data.py", "verify"])
        assert lock_data.main() == 0
