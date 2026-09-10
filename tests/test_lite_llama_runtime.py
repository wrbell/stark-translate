"""Native artifact identity, session ownership and failure cleanup contracts."""

import io
import json
import subprocess
import zipfile
from unittest.mock import Mock

import pytest

from stark_translate.profiles import resolve_profile
from tools import llama_runtime as runtime


def native(tmp_path, monkeypatch, backend="cpu"):
    monkeypatch.setattr(runtime, "platform_key", lambda backend: "test-" + backend)
    root = runtime.native_root(tmp_path) / ("test-" + backend)
    root.mkdir(parents=True)
    exe = root / "llama-server"
    exe.write_bytes(b"binary")
    data = {
        "revision": runtime.REVISION,
        "platform": "test-" + backend,
        "executable": "llama-server",
        "files": {"llama-server": runtime.sha256(exe)},
    }
    (root / "installed.json").write_text(json.dumps(data))
    return root, exe, data


def test_native_identity_inventory_detects_tamper(tmp_path, monkeypatch):
    root, exe, data = native(tmp_path, monkeypatch)
    assert runtime.resolve_native("cpu", tmp_path)[0] == exe
    exe.write_bytes(b"changed")
    with pytest.raises(FileNotFoundError, match="changed"):
        runtime.resolve_native("cpu", tmp_path)
    data["executable"] = "../foreign"
    (root / "installed.json").write_text(json.dumps(data))
    with pytest.raises(FileNotFoundError):
        runtime.resolve_native("cpu", tmp_path)


def test_archive_rejects_path_traversal_before_extracting(tmp_path):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("../escaped", "bad")
    target = tmp_path / "target"
    target.mkdir()
    with pytest.raises(ValueError, match="Unsafe"):
        runtime._extract(archive, target)
    assert not (tmp_path / "escaped").exists()


def fake_server(tmp_path, monkeypatch, *, foreign=False, exit_code=None):
    _, exe, data = native(tmp_path, monkeypatch)
    model = tmp_path / "gemma.gguf"
    model.write_bytes(b"model")
    monkeypatch.setattr(runtime, "resolve_model_path", lambda *a, **k: str(model))
    monkeypatch.setattr(
        runtime,
        "load_model_manifest",
        lambda: {"models": {"gemma-4-e2b-it-q4km.gguf": {"sha256": runtime.sha256(model)}}},
    )
    process = Mock(pid=31415)
    process.poll.return_value = exit_code
    process.returncode = exit_code
    start = Mock(return_value=process)
    monkeypatch.setattr(runtime.subprocess, "Popen", start)
    sock = Mock()
    sock.__enter__ = Mock(return_value=sock)
    sock.__exit__ = Mock(return_value=False)
    sock.getsockname.return_value = ("127.0.0.1", 18090)
    monkeypatch.setattr(runtime.socket, "socket", lambda: sock)

    def get(url, **kw):
        alias = start.call_args.args[0][start.call_args.args[0].index("--alias") + 1]
        return io.BytesIO(
            json.dumps(
                {"status": "ok"} if url.endswith("health") else {"data": [{"id": "foreign" if foreign else alias}]}
            ).encode()
        )

    monkeypatch.setattr(runtime.urllib.request, "urlopen", get)
    return (
        runtime.ManagedLlamaServer(resolve_profile("lite-cpu-quality"), models_dir=tmp_path, log_path=tmp_path / "log"),
        process,
        start,
    )


def test_owned_server_checks_alias_and_cpu_has_no_gpu_offload(tmp_path, monkeypatch):
    owner, process, start = fake_server(tmp_path, monkeypatch)
    assert owner.start() == "http://127.0.0.1:18090"
    argv = start.call_args.args[0]
    assert argv[argv.index("-ngl") + 1] == "0"
    assert "--no-op-offload" in argv and "--no-kv-offload" in argv
    assert "-md" not in argv
    assert not start.call_args.kwargs.get("start_new_session", False)
    assert owner.provenance["native_revision"] == runtime.REVISION
    assert owner.provenance["model_sha256"]
    owner.stop()
    process.terminate.assert_called_once()
    process.wait.assert_called_once_with(timeout=10)
    assert owner.log is None


def test_foreign_listener_is_never_adopted_and_owned_child_is_cleaned(tmp_path, monkeypatch):
    owner, process, _ = fake_server(tmp_path, monkeypatch, foreign=True)
    with pytest.raises(RuntimeError, match="identity mismatch"):
        owner.start()
    process.terminate.assert_called_once()
    assert owner.log is None


def test_owned_child_start_failure_is_not_a_model_fallback(tmp_path, monkeypatch):
    owner, process, _ = fake_server(tmp_path, monkeypatch, exit_code=2)
    with pytest.raises(RuntimeError, match="exited 2"):
        owner.start()
    process.terminate.assert_not_called()
    assert owner.log is None


def test_timeout_kills_only_owned_child(tmp_path, monkeypatch):
    owner, process, _ = fake_server(tmp_path, monkeypatch)
    owner.timeout_s = 0
    process.wait.side_effect = [subprocess.TimeoutExpired("owned", 10), 0]
    with pytest.raises(TimeoutError):
        owner.start()
    process.kill.assert_called_once()
    assert process.wait.call_args.kwargs == {"timeout": 5}


def test_offline_native_setup_does_not_download(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "platform_key", lambda backend: "Windows-x86_64-cpu")
    download = Mock(side_effect=AssertionError("network"))
    monkeypatch.setattr("operator_app.setup._download_direct", download)
    with pytest.raises(FileNotFoundError, match="Offline native archive"):
        runtime.install_native("cpu", tmp_path, offline=True)
    download.assert_not_called()
    assert not list(runtime.native_root(tmp_path).glob(".native-*"))


def test_archive_extracts_only_validated_regular_zip_members(tmp_path):
    archive = tmp_path / "native.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("bundle/", b"")
        z.writestr("bundle/llama-server.exe", b"verified binary")
        z.writestr("bundle/runtime.dll", b"verified library")
    destination = tmp_path / "unpacked"
    destination.mkdir()
    runtime._extract(archive, destination)
    assert (destination / "bundle" / "llama-server.exe").read_bytes() == b"verified binary"
    assert (destination / "bundle" / "runtime.dll").read_bytes() == b"verified library"
