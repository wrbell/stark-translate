"""Tests for tools/manage_adapters.py — register/activate/rollback/list/export."""

from __future__ import annotations

import json
import os
import sys
from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tools import manage_adapters as ma


@pytest.fixture
def manifest_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    manifest = tmp_path / "adapters" / "manifest.json"
    monkeypatch.setattr(ma, "MANIFEST_PATH", manifest)
    monkeypatch.setattr(ma, "HEALTH_CHECK_SCRIPT", tmp_path / "health_check.py")
    return tmp_path, manifest


class TestHelpers:
    def test_ensure_model_entry(self):
        m: dict = {}
        entry = ma._ensure_model_entry(m, "gemma_4b")
        assert entry["active"] is None
        assert entry["versions"] == {}
        # existing unchanged
        entry["active"] = "v1"
        assert ma._ensure_model_entry(m, "gemma_4b")["active"] == "v1"

    def test_compute_sha256(self, tmp_path: Path):
        p = tmp_path / "w.bin"
        p.write_bytes(b"weights")
        digest = ma._compute_sha256(p)
        assert len(digest) == 64

    def test_find_safetensors(self, tmp_path: Path):
        assert ma._find_safetensors(tmp_path) is None
        top = tmp_path / "adapter_model.safetensors"
        top.write_bytes(b"x")
        assert ma._find_safetensors(tmp_path) == top

        nested_root = tmp_path / "nested"
        nested_root.mkdir()
        nested = nested_root / "checkpoint-1"
        nested.mkdir()
        # Prefer top-level when present
        assert ma._find_safetensors(tmp_path) == top
        top.unlink()
        nested_file = nested / "adapter_model.safetensors"
        nested_file.write_bytes(b"y")
        assert ma._find_safetensors(tmp_path) == nested_file

    def test_load_save_manifest(self, manifest_env):
        tmp_path, manifest = manifest_env
        assert ma._load_manifest() == {}
        ma._save_manifest({"gemma_4b": {"active": None, "previous": None, "versions": {}}})
        assert manifest.exists()
        assert "gemma_4b" in ma._load_manifest()


class TestRegister:
    def test_missing_dir(self, manifest_env):
        args = Namespace(adapter="/no/such/dir", model="gemma_4b", version=None, eval_file=None)
        assert ma.cmd_register(args) == 1

    def test_register_with_weights_and_eval(self, manifest_env):
        tmp_path, manifest = manifest_env
        adapter = tmp_path / "S8_run"
        adapter.mkdir()
        (adapter / "adapter_model.safetensors").write_bytes(b"abc")
        (adapter / "training_manifest.json").write_text(json.dumps({"epochs": 1}))
        eval_file = tmp_path / "eval.json"
        eval_file.write_text(json.dumps({"comet": 0.9}))
        args = Namespace(
            adapter=str(adapter),
            model="gemma_4b",
            version=None,
            eval_file=str(eval_file),
        )
        assert ma.cmd_register(args) == 0
        data = ma._load_manifest()
        ver = data["gemma_4b"]["versions"]["S8_run"]
        assert ver["adapter_sha256"] is not None
        assert ver["eval_metrics"]["comet"] == 0.9
        assert ver["training_manifest"]["epochs"] == 1

    def test_register_without_safetensors(self, manifest_env):
        tmp_path, _ = manifest_env
        adapter = tmp_path / "empty_ad"
        adapter.mkdir()
        args = Namespace(adapter=str(adapter), model="whisper", version="v0", eval_file=None)
        assert ma.cmd_register(args) == 0
        assert ma._load_manifest()["whisper"]["versions"]["v0"]["adapter_sha256"] is None


class TestActivateRollbackList:
    def _seed(self, tmp_path: Path):
        a = tmp_path / "v1"
        b = tmp_path / "v2"
        a.mkdir()
        b.mkdir()
        (a / "adapter_model.safetensors").write_bytes(b"1")
        (b / "adapter_model.safetensors").write_bytes(b"2")
        ma.cmd_register(Namespace(adapter=str(a), model="gemma_4b", version="v1", eval_file=None))
        ma.cmd_register(Namespace(adapter=str(b), model="gemma_4b", version="v2", eval_file=None))
        return a, b

    def test_activate_pass_and_fail(self, manifest_env, monkeypatch: pytest.MonkeyPatch):
        tmp_path, _ = manifest_env
        self._seed(tmp_path)

        monkeypatch.setattr(
            ma.subprocess,
            "run",
            MagicMock(return_value=MagicMock(returncode=0, stdout="ok", stderr="")),
        )
        args = Namespace(model="gemma_4b", version="v1", base_model=None, max_latency=None)
        assert ma.cmd_activate(args) == 0
        assert ma._load_manifest()["gemma_4b"]["active"] == "v1"

        monkeypatch.setattr(
            ma.subprocess,
            "run",
            MagicMock(return_value=MagicMock(returncode=1, stdout="", stderr="fail")),
        )
        args2 = Namespace(model="gemma_4b", version="v2", base_model=None, max_latency=2.0)
        assert ma.cmd_activate(args2) == 1
        assert ma._load_manifest()["gemma_4b"]["versions"]["v2"]["verdict"] == "FAIL"

    def test_activate_unknown_version(self, manifest_env):
        args = Namespace(model="gemma_4b", version="nope", base_model=None, max_latency=None)
        assert ma.cmd_activate(args) == 1

    def test_rollback(self, manifest_env, monkeypatch: pytest.MonkeyPatch):
        tmp_path, _ = manifest_env
        self._seed(tmp_path)
        monkeypatch.setattr(
            ma.subprocess,
            "run",
            MagicMock(return_value=MagicMock(returncode=0, stdout="", stderr="")),
        )
        ma.cmd_activate(Namespace(model="gemma_4b", version="v1", base_model=None, max_latency=None))
        ma.cmd_activate(Namespace(model="gemma_4b", version="v2", base_model=None, max_latency=None))
        assert ma.cmd_rollback(Namespace(model="gemma_4b")) == 0
        data = ma._load_manifest()["gemma_4b"]
        assert data["active"] == "v1"
        assert data["previous"] == "v2"

    def test_rollback_no_previous(self, manifest_env):
        assert ma.cmd_rollback(Namespace(model="gemma_4b")) == 1

    def test_list(self, manifest_env, monkeypatch: pytest.MonkeyPatch, capsys):
        tmp_path, _ = manifest_env
        self._seed(tmp_path)
        monkeypatch.setattr(
            ma.subprocess,
            "run",
            MagicMock(return_value=MagicMock(returncode=0, stdout="", stderr="")),
        )
        ma.cmd_activate(Namespace(model="gemma_4b", version="v1", base_model=None, max_latency=None))
        assert ma.cmd_list(Namespace(model="gemma_4b")) == 0
        out = capsys.readouterr().out
        assert "v1" in out and "ACTIVE" in out
        assert ma.cmd_list(Namespace(model=None)) == 0
        assert ma.cmd_list(Namespace(model="missing")) == 0


class TestExportAndMain:
    def test_export(self, manifest_env, monkeypatch: pytest.MonkeyPatch):
        tmp_path, _ = manifest_env
        a = tmp_path / "v1"
        a.mkdir()
        (a / "adapter_model.safetensors").write_bytes(b"1")
        ma.cmd_register(Namespace(adapter=str(a), model="gemma_4b", version="v1", eval_file=None))
        m = ma._load_manifest()
        m["gemma_4b"]["active"] = "v1"
        ma._save_manifest(m)

        monkeypatch.setattr(ma.subprocess, "run", MagicMock(return_value=MagicMock(returncode=0)))
        assert ma.cmd_export(Namespace(model="gemma_4b", target="user@host:/tmp/", dry_run=True)) == 0
        assert ma._load_manifest()["gemma_4b"]["versions"]["v1"]["deployed"] is not None

        monkeypatch.setattr(ma.subprocess, "run", MagicMock(return_value=MagicMock(returncode=2)))
        assert ma.cmd_export(Namespace(model="gemma_4b", target="x", dry_run=False)) == 1

    def test_export_no_active(self, manifest_env):
        assert ma.cmd_export(Namespace(model="gemma_4b", target="x", dry_run=False)) == 1

    def test_main_dispatch(self, manifest_env, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(sys, "argv", ["manage_adapters.py"])
        assert ma.main() == 1
        monkeypatch.setattr(sys, "argv", ["manage_adapters.py", "list"])
        assert ma.main() == 0
