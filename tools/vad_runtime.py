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


class NumpyOnnxVAD:
    """Silero 6.2.1 recurrent ONNX contract, with no PyTorch dependency.

    The live caller already supplies mono 512-sample frames at 16 kHz. State
    and 64-sample context follow Silero's MIT-licensed ONNX wrapper contract.
    Calls must be serialized; reset_states marks a stream discontinuity.
    """

    numpy_input = True

    def __init__(self, path: str):
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.inter_op_num_threads = options.intra_op_num_threads = 1
        self.session = ort.InferenceSession(path, sess_options=options, providers=["CPUExecutionProvider"])
        self.reset_states()

    def reset_states(self, batch_size: int = 1):
        import numpy as np

        self._state = np.zeros((2, batch_size, 128), dtype=np.float32)
        self._context = np.zeros((batch_size, 64), dtype=np.float32)

    def __call__(self, audio, sample_rate: int):
        import numpy as np

        values = np.asarray(audio, dtype=np.float32)
        if values.ndim == 1:
            values = values[None, :]
        if sample_rate != 16000 or values.ndim != 2 or values.shape[1] != 512:
            raise ValueError("Live ONNX VAD requires 512 mono samples per batch at 16000 Hz")
        if values.shape[0] != self._state.shape[1]:
            self.reset_states(values.shape[0])
        window = np.concatenate((self._context, values), axis=1)
        probability, state = self.session.run(
            None, {"input": window, "state": self._state, "sr": np.array(sample_rate, dtype=np.int64)}
        )
        self._state = state
        self._context = window[:, -64:].copy()
        return probability


def load_managed_onnx_vad():
    from engines.model_paths import load_model_manifest, resolve_model_path

    path = resolve_model_path("silero-vad-onnx", local_only=True)
    if path is None:
        raise FileNotFoundError("Pinned ONNX VAD missing; run stark-translate-lite setup")
    entry = load_model_manifest()["models"]["silero-vad-onnx"]
    actual = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if actual != entry["sha256"]:
        raise ValueError("Pinned ONNX VAD checksum mismatch; rerun setup --refresh")
    artifact = {
        "source": "managed_cache",
        "package": "silero-vad",
        "package_version": "6.2.1",
        "backend": "onnx",
        "opset_version": 16,
        "path": path,
        "sha256": actual,
        "source_revision": entry["revision"],
        "size_bytes": Path(path).stat().st_size,
    }
    return NumpyOnnxVAD(path), (), artifact
