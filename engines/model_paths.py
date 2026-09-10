"""Shared, offline model lookup for setup, preflight and inference.

Model identity stays a Hugging Face ID; resolution only selects a local copy.
No model imports or downloads happen here.
"""

from __future__ import annotations

import json
import os
import platform
import re
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
    if entry.get("type") == "derived-ct2":
        resolved = resolve_marian_ct2(entry["direction"], project_root=project_root, models_dir=models_dir)
        return resolved if resolved is not None or local_only else model_id
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


def pinned_hf_entry(model_id: str, *, project_root: Path | None = None) -> dict[str, Any]:
    """Return a registered remote source only when it has an immutable revision."""
    models = load_model_manifest(project_root).get("models", {})
    entry: dict[str, Any] = next(
        (
            value
            for key, value in models.items()
            if model_id in {key, value.get("repo_id"), value.get("voice"), *value.get("aliases", [])}
        ),
        {},
    )
    revision = entry.get("revision")
    if (
        entry.get("type") != "hf-snapshot"
        or not entry.get("repo_id")
        or not isinstance(revision, str)
        or not re.fullmatch(r"[0-9a-f]{40}", revision)
    ):
        raise ValueError(
            f"No pinned HF source for {model_id}; register its full commit in models.lock.json "
            "and run setup, or configure an explicit local model path"
        )
    return entry


def resolve_hf_model_source(
    model_id: str, *, models_dir: Path | None = None, project_root: Path | None = None
) -> tuple[str, dict[str, Any]]:
    """Select a local-only model or a manifest-pinned remote Transformers source.

    Local overrides retain their identity; a revision is never invented for them.
    Unknown remote IDs must be prepared explicitly instead of following HF main.
    """
    local = resolve_model_path(model_id, models_dir=models_dir, project_root=project_root, local_only=True)
    if local:
        return local, {"local_files_only": True}
    entry = pinned_hf_entry(model_id, project_root=project_root)
    return entry["repo_id"], {"revision": entry["revision"]}


def resolve_piper_voice(voice_name: str, *, models_dir: Path | None = None) -> str | None:
    """Find setup voices, explicit local overrides, or a pinned Piper HF snapshot."""
    explicit = Path(voice_name).expanduser()
    if explicit.is_file() and explicit.with_suffix(".onnx.json").is_file():
        return str(explicit.resolve())
    manifest = load_model_manifest()
    for key, entry in manifest["models"].items():
        if entry.get("voice") == voice_name:
            root = resolve_model_path(key, models_dir=models_dir, local_only=True)
            if root:
                return str(Path(root) / entry["required_files"][0])
            # Older standard installs used a separate HF cache. Select only the
            # registered commit, never an arbitrary snapshot from its history.
            pinned = pinned_hf_entry(voice_name)
            cache = Path.home() / ".local" / "share" / "piper_tts"
            snapshot = cache / ("models--" + pinned["repo_id"].replace("/", "--")) / "snapshots"
            model = snapshot / pinned["revision"] / entry["required_files"][0]
            if model.is_file() and model.with_suffix(".onnx.json").is_file():
                return str(model)
    # Flat files are existing user-managed voices, not HF revision evidence.
    cache = Path.home() / ".local" / "share" / "piper_tts"
    path = cache / f"{voice_name}.onnx"
    if path.is_file() and path.with_suffix(".onnx.json").is_file():
        return str(path)
    return None


MARIAN_CT2_REQUIRED_FILES = (
    "model.bin",
    "config.json",
    "vocab.json",
    "source.spm",
    "target.spm",
    "tokenizer_config.json",
)


def marian_ct2_complete(path: Path) -> bool:
    """Cheap structural check; model loading and file hashes belong to setup."""
    return path.is_dir() and all(
        (path / name).is_file() and (path / name).stat().st_size > 0 for name in MARIAN_CT2_REQUIRED_FILES
    )


def resolve_marian_ct2(
    direction: str,
    *,
    explicit_path: str | Path | None = None,
    project_root: Path | None = None,
    models_dir: Path | None = None,
    adapter_root: Path | None = None,
    managed_only: bool = False,
) -> str | None:
    """Use an explicit override, existing working adapter, then managed cache.

    Managed active.json is atomically replaced only after conversion succeeds.
    Its immutable build directory carries source revision and quantization.
    Existing adapters are preserved without inventing a source revision for them.
    """
    if direction not in {"en-es", "es-en"}:
        raise ValueError(f"Unsupported Marian direction: {direction}")
    if explicit_path is not None and not managed_only:
        explicit = Path(explicit_path).expanduser()
        if not marian_ct2_complete(explicit):
            raise ValueError(f"Explicit Marian CT2 path is incomplete: {explicit}")
        return str(explicit.resolve())
    project = project_root or Path(os.environ.get("STARK_PROJECT_ROOT", Path.cwd()))
    adapters = adapter_root if adapter_root is not None else project / "adapters" / "marian_ct2"
    if adapter_root is not None and not adapters.is_absolute():
        adapters = project / adapters
    candidate = adapters / direction / "active"
    if not managed_only and marian_ct2_complete(candidate):
        return str(candidate.resolve())
    try:
        entries = load_model_manifest(project).get("models", {})
        entry = entries[f"marian-ct2-{direction}"]
        source = entries[entry["source_model"]]
        managed = (models_dir or default_models_dir()) / entry["subdir"]
        pointer = json.loads((managed / "active.json").read_text())
        name = pointer["directory"]
        if not isinstance(name, str) or Path(name).name != name or name in {".", ".."}:
            return None
        candidate = managed / name
        if candidate.is_symlink() or candidate.resolve().parent != managed.resolve():
            return None
        exported = json.loads((candidate / "export_manifest.json").read_text())
        if not isinstance(exported, dict) or (
            exported.get("model_id") != source["repo_id"]
            or exported.get("source_revision") != source["revision"]
            or exported.get("direction") != direction
            or exported.get("ct2_quantization") != entry["quantization"]
        ):
            return None
        if marian_ct2_complete(candidate):
            return str(candidate.resolve())
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return None
