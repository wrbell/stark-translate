"""Packaged VAD contracts and opt-in real CPU checks; no Hub downloads."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from operator_app import preflight
from tools import vad_runtime

ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize("backend,filename", [("torch", "silero_vad.jit"), ("onnx", "silero_vad.onnx")])
def test_packaged_vad_uses_bundled_file_and_compatible_utils(tmp_path, monkeypatch, backend, filename):
    package = tmp_path / "package"
    weights = package / "silero_vad" / "data" / filename
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"mocked bundled VAD weights")
    monkeypatch.setattr(
        vad_runtime.importlib.metadata, "distribution", lambda name: SimpleNamespace(locate_file=lambda p: package / p)
    )
    monkeypatch.setattr(vad_runtime.importlib.metadata, "version", lambda name: "6.2.1")
    model = object()
    load = Mock(return_value=model)
    utils = tuple(object() for _ in range(5))
    monkeypatch.setitem(
        sys.modules,
        "silero_vad",
        SimpleNamespace(
            load_silero_vad=load,
            get_speech_timestamps=utils[0],
            save_audio=utils[1],
            read_audio=utils[2],
            VADIterator=utils[3],
            collect_chunks=utils[4],
        ),
    )
    result, returned_utils, artifact = vad_runtime.load_packaged_vad(backend)
    assert result is model
    assert returned_utils == utils
    load.assert_called_once_with(onnx=backend == "onnx", opset_version=16)
    assert artifact["package_version"] == "6.2.1"
    assert artifact["sha256"] == hashlib.sha256(weights.read_bytes()).hexdigest()
    assert artifact["path"] == str(weights)


def test_missing_bundled_weights_fails_without_download(tmp_path, monkeypatch):
    monkeypatch.setattr(
        vad_runtime.importlib.metadata, "distribution", lambda name: SimpleNamespace(locate_file=lambda p: tmp_path / p)
    )
    with pytest.raises(FileNotFoundError, match="missing bundled weights"):
        vad_runtime.load_packaged_vad("torch")


def test_pipeline_keeps_model_utils_interface_and_records_provenance(monkeypatch):
    tree = ast.parse((ROOT / "dry_run_ab.py").read_text())
    loader = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "load_vad")
    model, utils, artifact = object(), object(), {"package_version": "6.2.1", "sha256": "known"}
    monkeypatch.setattr(vad_runtime, "load_packaged_vad", lambda backend: (model, utils, artifact))
    namespace = {
        "settings": SimpleNamespace(vad=SimpleNamespace(backend="onnx")),
        "RUNTIME_PROFILE": SimpleNamespace(lite=False),
    }
    exec(compile(ast.Module(body=[loader], type_ignores=[]), str(ROOT / "dry_run_ab.py"), "exec"), namespace)
    assert namespace["load_vad"]() == (model, utils)
    assert namespace["_vad_provenance"] == artifact


def _versions(package):
    return {
        "torch": "2.10.0",
        "mlx": "0.32.2",
        "mlx-lm": "0.31.3",
        "mlx-optiq": "0.4.34",
        "silero-vad": "6.2.1",
        "parakeet-mlx": "0.5.2",
    }.get(package, "99.0")


def test_mac_preflight_requires_exact_silero_version(monkeypatch):
    monkeypatch.setattr(
        preflight.importlib.metadata, "version", lambda name: "6.2.2" if name == "silero-vad" else _versions(name)
    )
    result = preflight.check_dependencies("mlx")
    assert result["status"] == "fail"
    assert "silero-vad==6.2.1 (installed 6.2.2)" in result["detail"]


def test_preflight_checks_both_bundled_artifacts_without_model_imports(monkeypatch):
    monkeypatch.setattr(preflight.importlib.metadata, "version", _versions)
    probe = Mock(side_effect=[Path("packaged.jit"), FileNotFoundError("missing ONNX")])
    monkeypatch.setattr(vad_runtime, "packaged_vad_path", probe)
    result = preflight.check_dependencies("mlx")
    assert result["status"] == "fail"
    assert "bundled onnx weights" in result["detail"]
    assert [call.args[0] for call in probe.call_args_list] == ["torch", "onnx"]


@pytest.mark.skipif(os.environ.get("STARK_RUN_VAD_TESTS") != "1", reason="opt-in real CPU VAD package test")
@pytest.mark.parametrize("backend", ["torch", "onnx"])
def test_real_packaged_vad_offline_with_empty_hub(tmp_path, backend):
    # Run a fresh process to avoid the main CPU suite's torch/Silero mocks.
    # Block sockets and Torch Hub explicitly; neither path is needed for the
    # installed package's JIT/ONNX weight files and one 512-sample CPU frame.
    script = r"""
import json
import socket
import sys
from pathlib import Path

def forbidden(*args, **kwargs):
    raise AssertionError("Network/Hub access during packaged VAD load")
socket.create_connection = forbidden
socket.socket.connect = forbidden
import torch
torch.hub.load = forbidden
torch.hub.download_url_to_file = forbidden
from tools.vad_runtime import load_packaged_vad
model, utils, artifact = load_packaged_vad(sys.argv[1])
score = float(model(torch.zeros(512), 16000).item())
assert 0 <= score <= 1
assert len(utils) == 5
assert artifact["package_version"] == "6.2.1"
assert artifact["sha256"] == sys.argv[2]
model.reset_states()
print(json.dumps({"artifact": artifact, "silence_probability": score}))
"""
    expected = {
        "torch": "e1122837f4154c511485fe0b9c64455f7b929c96fbb8d79fbdb336383ebd3720",
        "onnx": "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3",
    }
    env = {
        **os.environ,
        "PYTHONPATH": str(ROOT),
        "TORCH_HOME": str(tmp_path / "torch"),
        "HF_HOME": str(tmp_path / "hf"),
        "HF_HUB_CACHE": str(tmp_path / "hf" / "hub"),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "NUMBA_THREADING_LAYER": "workqueue",
        "KMP_DUPLICATE_LIB_OK": "TRUE",
    }
    result = subprocess.run(
        [sys.executable, "-c", script, backend, expected[backend]],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    assert json.loads(result.stdout.splitlines()[-1])["artifact"]["backend"] == backend
    assert not list((tmp_path / "torch").rglob("*"))
    assert not list((tmp_path / "hf").rglob("*"))
