"""Cheap profile-specific admission checks; no model loads or auto fallback."""

from __future__ import annotations

import importlib.metadata
import subprocess
from pathlib import Path

from engines.model_paths import resolve_marian_ct2
from stark_translate.profiles import resolve_profile, resolve_profile_model


def checks(
    profile: str,
    *,
    backend: str = "auto",
    lang: str = "en",
    models_dir: Path | None = None,
    project_root: Path | None = None,
    tts: bool = False,
    diarize: bool = False,
) -> tuple[list[dict], object]:
    import psutil
    from packaging.specifiers import SpecifierSet

    from operator_app.preflight import _check

    p = resolve_profile(profile, backend)
    requirements = {
        "faster-whisper": ">=1.2.1,<2",
        "ctranslate2": ">=4.7.1,<5",
        "onnxruntime": ">=1.20,<2",
        "transformers": ">=5.5,<6",
        "sentencepiece": ">=0.2,<0.3",
        "scipy": ">=1.13,<2",
        "sacremoses": ">=0.1",
    }
    if tts:
        requirements["piper-tts"] = ">=1.4.1,<2"
    missing = []
    for name, version in requirements.items():
        try:
            if importlib.metadata.version(name) not in SpecifierSet(version):
                missing.append(name + version)
        except importlib.metadata.PackageNotFoundError:
            missing.append(name + version)
    out = [
        _check(
            "Runtime dependencies",
            "fail" if missing else "pass",
            "Missing/incompatible: " + ", ".join(missing)
            if missing
            else "Torch-free lite runtime dependencies available",
        )
    ]
    # Feature extras cannot silently expand the advertised memory envelope.
    if diarize:
        out.append(
            _check(
                "Diarization",
                "fail",
                "Diarization is outside the lite memory profile; select standard for this feature",
            )
        )
    cores = psutil.cpu_count(logical=False) or 0
    ram = psutil.virtual_memory().total / 1024**3
    out.append(
        _check(
            "Lite hardware",
            "pass" if cores >= p.physical_core_floor and ram >= p.ram_floor_gb * 0.95 else "fail",
            f"{cores} physical cores, {ram:.1f} GiB RAM; {p.name} floor {p.physical_core_floor} cores / {p.ram_floor_gb} GiB. Performance targets require certification on this hardware.",
        )
    )
    names = [p.stt_model, "silero-vad-onnx"] + ([p.translation_model] if p.translation_model else [])
    absent = [
        name for name in names if resolve_profile_model(name, project_root=project_root, models_dir=models_dir) is None
    ]
    direction = "es-en" if lang == "es" else "en-es"
    if resolve_marian_ct2(direction, project_root=project_root, models_dir=models_dir, managed_only=True) is None:
        absent.append("marian-ct2-" + direction)
    if tts:
        from engines.model_paths import resolve_piper_voice
        from settings import settings

        voice = settings.tts.voices.get("en" if lang == "es" else "es")
        if not voice or resolve_piper_voice(voice, models_dir=models_dir) is None:
            absent.append("Piper " + str(voice))
    out.append(
        _check(
            "Models",
            "fail" if absent else "pass",
            "Missing: " + ", ".join(absent) + f". Run stark-translate-lite setup --profile {p.name}"
            if absent
            else f"Pinned {p.stt_model}; CPU Marian INT8; finals {p.final_engine}; ONNX VAD",
        )
    )
    if p.final_engine == "llamacpp":
        from tools.llama_runtime import resolve_native

        try:
            executable, native = resolve_native(p.backend, models_dir, verify=False)
            out.append(
                _check(
                    "Managed llama-server",
                    "pass",
                    f"{executable} ({native['revision']}); owned child starts with the session and verifies all hashes",
                )
            )
        except FileNotFoundError as exc:
            out.append(_check("Managed llama-server", "fail", str(exc)))
    if p.backend == "cuda":
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total,compute_cap", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                check=True,
                timeout=5,
            )
            name, memory, capability = result.stdout.splitlines()[0].split(",")
            valid = float(memory) >= 7800 and float(capability) >= 7.5
            out.append(
                _check(
                    "Lite CUDA",
                    "pass" if valid else "fail",
                    f"{name.strip()}: {memory.strip()} MiB, sm_{capability.strip()}; requires 8 GB / sm_75+, CUDA 12 cuBLAS and cuDNN 9 for CT2. Actual compute support is checked at load.",
                )
            )
        except (OSError, ValueError, subprocess.SubprocessError, IndexError) as exc:
            out.append(_check("Lite CUDA", "fail", f"Cannot verify selected NVIDIA GPU: {exc}"))
    return out, p
