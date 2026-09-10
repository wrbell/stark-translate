"""Shared product profiles. Importable by setup and preflight without ML runtimes."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass

PROFILE_NAMES = ("standard", "lite-cpu", "lite-cpu-quality", "lite-cuda-8gb")


@dataclass(frozen=True)
class RuntimeProfile:
    name: str
    backend: str
    stt_model: str | None = None
    stt_compute_type: str | None = None
    stt_threads: int = 3
    stt_workers: int = 1
    marian_threads: int = 1
    final_engine: str | None = None
    translation_model: str | None = None
    vad_backend: str | None = None
    ram_floor_gb: int = 8
    physical_core_floor: int = 4
    vram_floor_gb: int = 0

    @property
    def lite(self) -> bool:
        return self.name != "standard"

    def to_dict(self) -> dict:
        return asdict(self)

    def model_keys(self, include: list[str] | None = None, manifest: dict | None = None) -> set[str]:
        if not self.lite:
            raise ValueError("Standard model selection uses the backend manifest")
        keys = {
            self.stt_model,
            "silero-vad-onnx",
            "marian-en-es",
            "marian-es-en",
            "marian-ct2-en-es",
            "marian-ct2-es-en",
        }
        if self.translation_model:
            keys.add(self.translation_model)
        groups = set(include or [])
        if groups - {"tts"}:
            raise ValueError("Lite profiles accept only --include tts; choose lite-cpu-quality for E2B")
        if manifest:
            keys.update(k for k, v in manifest.items() if v.get("optional_group") in groups)
        return {k for k in keys if k is not None}


def resolve_profile(name: str | None = None, backend: str = "auto") -> RuntimeProfile:
    name = name or os.environ.get("STARK_PROFILE", "standard")
    if name not in PROFILE_NAMES:
        raise ValueError(f"Unknown profile {name!r}; choose {', '.join(PROFILE_NAMES)}")
    if name == "standard":
        return RuntimeProfile(name, backend)
    selected = "cuda" if name == "lite-cuda-8gb" else "cpu"
    if backend not in ("auto", selected):
        raise ValueError(f"Profile {name} requires backend {selected}, not {backend}")
    quality = name != "lite-cpu"
    return RuntimeProfile(
        name,
        selected,
        stt_model="whisper-large-v3-turbo" if selected == "cuda" else "whisper-small",
        stt_compute_type="int8_float16" if selected == "cuda" else "int8",
        final_engine="llamacpp" if quality else "marian",
        translation_model="gemma-4-e2b-it-q4km.gguf" if quality else None,
        vad_backend="onnx",
        ram_floor_gb=16 if quality else 8,
        vram_floor_gb=8 if selected == "cuda" else 0,
    )


def apply_profile(settings, name: str | None = None, backend: str = "auto") -> RuntimeProfile:
    """Apply bounded product policy once, after CLI overrides, before model loads."""
    profile = resolve_profile(name, backend)
    settings.profile = profile.name
    if not profile.lite:
        return profile
    settings.backend = profile.backend
    settings.run_ab = False
    settings.multiprocess = False
    settings.low_vram = profile.final_engine == "marian"
    settings.stt.backend = "faster-whisper"
    settings.stt.whisper_cuda_model = profile.stt_model
    settings.stt.whisper_cuda_compute_type = profile.stt_compute_type
    settings.stt.cpu_threads = profile.stt_threads
    settings.stt.num_workers = profile.stt_workers
    settings.stt.local_files_only = True
    settings.stt.fallback_on_low_conf = False
    settings.vad.backend = "onnx"
    settings.translation.marian_backend = "ct2"
    settings.translation.marian_device = "cpu"
    settings.translation.marian_compute_type = "int8"
    settings.translation.marian_intra_threads = profile.marian_threads
    settings.translation.model_family = "gemma4"
    settings.translation.mlx_mts = False
    settings.cuda.engine = "llamacpp" if profile.final_engine == "llamacpp" else "auto"
    settings.cuda.use_speculative = False
    return profile


def session_overrides(name: str | None = None, backend: str = "auto") -> dict:
    """Public operator adapter: canonical options shown and passed to subprocess."""
    p = resolve_profile(name, backend)
    if not p.lite:
        return {"profile": p.name}
    return {
        "profile": p.name,
        "backend": p.backend,
        "stt_backend": "faster-whisper",
        "model_family": "gemma4",
        "gemma4_size": "e2b",
        "engine": "llamacpp" if p.final_engine == "llamacpp" else "auto",
        "low_vram": p.final_engine == "marian",
        "run_ab": False,
    }


def resolve_profile_model(key: str, *, models_dir=None, project_root=None):
    """Resolve registered profile artifacts; reject unversioned same-name folders."""
    import json
    from pathlib import Path

    from engines.model_paths import load_model_manifest, resolve_marian_ct2, resolve_model_path

    entry = load_model_manifest(project_root)["models"][key]
    if entry.get("type") == "derived-ct2":
        return resolve_marian_ct2(
            entry["direction"], models_dir=models_dir, project_root=project_root, managed_only=True
        )
    found = resolve_model_path(key, models_dir=models_dir, project_root=project_root, local_only=True)
    if found is None or entry.get("type") != "hf-snapshot":
        return found
    path = Path(found)
    if path.parent.name == "snapshots" and path.name == entry["revision"]:
        return found
    try:
        installed = json.loads((path / ".installed").read_text())
        if installed.get("repo_id") == entry["repo_id"] and installed.get("revision") == entry["revision"]:
            return found
    except (OSError, ValueError, AttributeError):
        pass
    return None
