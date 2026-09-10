"""Use Silero's installed package weights without a Torch Hub dependency."""

from __future__ import annotations

import hashlib
import importlib.metadata
from pathlib import Path


def packaged_vad_path(backend: str = "torch") -> Path:
    """Locate bundled weights without importing torch, Silero or ONNX Runtime."""
    if backend not in {"torch", "onnx"}:
        raise ValueError(f"Unsupported VAD backend: {backend}")
    filename = "silero_vad.onnx" if backend == "onnx" else "silero_vad.jit"
    distribution = importlib.metadata.distribution("silero-vad")
    path = Path(str(distribution.locate_file(f"silero_vad/data/{filename}")))
    if not path.is_file():
        raise FileNotFoundError(f"silero-vad package is missing bundled weights: {path}")
    return path.resolve()


def packaged_vad_metadata(backend: str = "torch") -> dict:
    """Bind the actual package artifact before loading; no model imports."""
    path = packaged_vad_path(backend)
    contents = path.read_bytes()
    return {
        "source": "installed_package",
        "package": "silero-vad",
        "package_version": importlib.metadata.version("silero-vad"),
        "backend": backend,
        "opset_version": 16 if backend == "onnx" else None,
        "path": str(path),
        "sha256": hashlib.sha256(contents).hexdigest(),
        "size_bytes": len(contents),
    }


def load_packaged_vad(backend: str = "torch"):
    """Return the model, historical Hub utility tuple and artifact provenance.

    The Silero 6.2.1 ONNX loader uses CPU execution and opset 16. The live
    is_speech function calls the model directly; retain the utility tuple for
    callers still accepting the previous (model, utils) interface.
    """
    artifact = packaged_vad_metadata(backend)
    from silero_vad import (
        VADIterator,
        collect_chunks,
        get_speech_timestamps,
        load_silero_vad,
        read_audio,
        save_audio,
    )

    model = load_silero_vad(onnx=backend == "onnx", opset_version=16)
    utils = (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks)
    return model, utils, artifact
