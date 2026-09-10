"""Shared, offline model lookup for setup, preflight and inference.

Model identity stays a Hugging Face ID; resolution only selects a local copy.
No model imports or downloads happen here.
"""

from __future__ import annotations

import json
import os
import platform
from pathlib import Path
from typing import Any, Literal, overload


def default_models_dir() -> Path:
    override = os.environ.get("STARK_MODELS_DIR")
    if override:
        return Path(override).expanduser()
    if platform.system() == "Windows":
        return Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "stark-translate" / "models"
    return Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))) / "stark-translate" / "models"


def load_model_manifest(project_root: Path | None = None) -> dict[str, Any]:
    root = project_root or Path(__file__).resolve().parent.parent
    path = root / "models.lock.json"
    if not path.exists():
        path = Path(__file__).resolve().parent.parent / "models.lock.json"
    return json.loads(path.read_text())


def resolve_backend(backend: str = "auto") -> str:
    if backend != "auto":
        if backend not in {"mlx", "cuda", "cpu"}:
            raise ValueError(f"Unsupported backend: {backend}")
        return backend
    if platform.system() == "Darwin" and platform.machine() in {"arm64", "aarch64"}:
        return "mlx"
    import shutil

    return "cuda" if shutil.which("nvidia-smi") else "cpu"


def _snapshot_complete(path: Path, entry: dict[str, Any]) -> bool:
    """Reject empty directories and partial snapshots without loading weights."""
    marker = path / ".installed"
    if entry.get("type") == "hf-snapshot" and marker.is_file():
        try:
            installed = json.loads(marker.read_text())
            if not isinstance(installed, dict) or (
                installed.get("repo_id") != entry.get("repo_id")
                or installed.get("revision") != entry.get("revision", "main")
            ):
                return False
        except (OSError, ValueError):
            return False
    required = entry.get("required_files", ["config.json"])
    if not all((path / name).is_file() for name in required):
        return False
    index = path / "model.safetensors.index.json"
    if index.is_file():
        try:
            shards = set(json.loads(index.read_text())["weight_map"].values())
            if not shards or not all((path / shard).is_file() for shard in shards):
                return False
        except (OSError, ValueError, KeyError, TypeError):
            return False
    if entry.get("weights_required", True):
        return any(path.glob("*.safetensors")) or any(path.glob("*.npz")) or any(path.glob("*.bin"))
    return True


@overload
def resolve_model_path(
    model_id: str,
    *,
    models_dir: Path | None = None,
    project_root: Path | None = None,
    local_only: Literal[False] = False,
) -> str: ...


@overload
def resolve_model_path(
    model_id: str, *, models_dir: Path | None = None, project_root: Path | None = None, local_only: Literal[True]
) -> str | None: ...


def resolve_model_path(
    model_id: str,
    *,
    models_dir: Path | None = None,
    project_root: Path | None = None,
    local_only: bool = False,
) -> str | None:
    """Resolve explicit path, setup cache, project models, then HF's local cache.

    Missing models retain their repo ID for normal inference downloads; a
    preflight caller uses ``local_only=True`` to receive ``None`` instead.
    """
    explicit = Path(model_id).expanduser()
    if explicit.exists():
        return str(explicit.resolve())
    try:
        manifest = load_model_manifest(project_root)
    except (OSError, ValueError):
        manifest = {"models": {}}
    entry: dict[str, Any] = next(
        (
            v
            for k, v in manifest["models"].items()
            if model_id in {k, v.get("repo_id"), v.get("filename"), *v.get("aliases", [])}
        ),
        {},
    )
    root = project_root or Path(os.environ.get("STARK_PROJECT_ROOT", Path(__file__).resolve().parent.parent))
    for directory in (models_dir or default_models_dir(), root / "models"):
        target = directory / entry.get("subdir", entry.get("filename", model_id))
        if target.is_file() or (target.is_dir() and _snapshot_complete(target, entry)):
            return str(target.resolve())
    repo_id = entry.get("repo_id", model_id)
    if "/" in repo_id:
        hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
        hub = Path(os.environ.get("HF_HUB_CACHE", hf_home / "hub"))
        repo = hub / ("models--" + repo_id.replace("/", "--"))
        revision = entry.get("revision", "main")
        ref = repo / "refs" / revision
        if ref.is_file():
            revision = ref.read_text().strip()
        snapshot = repo / "snapshots" / revision
        if snapshot.is_dir() and _snapshot_complete(snapshot, entry):
            return str(snapshot)
    return None if local_only else model_id


def resolve_piper_voice(voice_name: str, *, models_dir: Path | None = None) -> str | None:
    """Find setup-installed voices or the pre-existing Piper cache."""
    manifest = load_model_manifest()
    for key, entry in manifest["models"].items():
        if entry.get("voice") == voice_name:
            root = resolve_model_path(key, models_dir=models_dir, local_only=True)
            if root:
                return str(Path(root) / entry["required_files"][0])
    cache = Path.home() / ".local" / "share" / "piper_tts"
    for path in [cache / f"{voice_name}.onnx", *cache.glob(f"models--*/snapshots/*/**/{voice_name}.onnx")]:
        if path.is_file() and path.with_suffix(".onnx.json").is_file():
            return str(path)
    return None
