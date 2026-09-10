"""Pre-flight checks for an operator session.

Each check returns a ``Check`` dict with a ``name``, ``status`` (one of
``"pass" | "warn" | "fail"``), and ``detail`` string. The frontend renders
these as green / yellow / red rows; the operator can only "Start Session"
when no check is ``fail``.

Checks are intentionally cheap (no model loads, no network round-trips
beyond a single ``/health``) so the page can poll them every few seconds.
"""

from __future__ import annotations

import importlib.metadata
import json
import os
import shutil
import urllib.error
import urllib.request
from pathlib import Path
from typing import Literal

from engines.model_paths import resolve_backend, resolve_marian_ct2, resolve_model_path, resolve_piper_voice

CheckStatus = Literal["pass", "warn", "fail"]
Check = dict


def _check(name: str, status: CheckStatus, detail: str) -> Check:
    return {"name": name, "status": status, "detail": detail}


def check_gpu() -> Check:
    """Detect any usable inference accelerator (CUDA, MLX/Metal, or CPU)."""
    # CUDA via nvidia-smi (no torch import — keep startup cheap).
    if shutil.which("nvidia-smi"):
        return _check("GPU", "pass", "NVIDIA GPU detected via nvidia-smi")
    # MLX (Apple Silicon).
    try:
        import platform

        if platform.system() == "Darwin" and platform.machine() in ("arm64", "aarch64"):
            return _check("GPU", "pass", "Apple Silicon detected (MLX path)")
    except Exception:
        pass
    return _check("GPU", "warn", "No GPU detected — will run on CPU (slow but functional)")


def check_dependencies(
    backend: str, *, lang: str = "en", stt_backend: str = "auto", tts: bool = False, diarize: bool = False
) -> Check:
    """Check distribution versions without importing GPU runtimes."""
    requirements = {
        "torch": "2.6",
        "scipy": "1.13",
        "transformers": "5.5",
        "sentencepiece": "0.2",
        "silero-vad": "6.2.1",
        "onnxruntime": "1.20",
    }
    if backend == "mlx":
        requirements.update(
            {"mlx": "0.31.2", "mlx-lm": "0.31.3", "mlx-optiq": "0.4.34", "mlx-whisper": "0.4", "ctranslate2": "4.7.1"}
        )
        if stt_backend == "parakeet-mlx" or (stt_backend == "auto" and lang == "en"):
            requirements["parakeet-mlx"] = "0.5.2"
    else:
        requirements.update({"faster-whisper": "1.1", "ctranslate2": "4.5"})
    if tts:
        requirements["piper-tts"] = "1.0"
    if diarize:
        requirements.update({"speechbrain": "1.0", "torchaudio": "2.6", "soundfile": "0.12", "scikit-learn": "1.4"})
    if backend == "mlx":
        requirements.update(
            {"torch": ">=2.10,<2.11", "mlx": ">=0.32.2,<0.33", "mlx-lm": ">=0.31.3,<0.32", "mlx-optiq": ">=0.4.34,<0.5"}
        )
    if backend == "mlx":
        requirements["silero-vad"] = "==6.2.1"
    if diarize:
        requirements["torchaudio"] = ">=2.10,<2.11"
    from packaging.specifiers import SpecifierSet

    missing = []
    for package, minimum in requirements.items():
        specifier = minimum if minimum.startswith((">", "<", "=", "!", "~")) else f">={minimum}"
        try:
            installed = importlib.metadata.version(package)
            if installed not in SpecifierSet(specifier):
                missing.append(f"{package}{specifier} (installed {installed})")
        except importlib.metadata.PackageNotFoundError:
            missing.append(f"{package}{specifier}")
    if not missing:
        from tools.vad_runtime import packaged_vad_path

        for vad_backend in ("torch", "onnx"):
            try:
                packaged_vad_path(vad_backend)
            except (importlib.metadata.PackageNotFoundError, OSError) as exc:
                missing.append(f"silero-vad bundled {vad_backend} weights ({exc})")
    if missing:
        extra = backend + (",diarization" if diarize else "")
        return _check(
            "Runtime dependencies",
            "fail",
            f"Missing/incompatible: {', '.join(missing)}. Install stark-translate[{extra}] in this interpreter.",
        )
    return _check("Runtime dependencies", "pass", f"{backend} dependencies available")


def check_models(
    project_root: Path,
    *,
    backend: str = "cuda",
    lang: str = "en",
    model_family: str = "gemma4",
    gemma4_size: str = "e4b",
    stt_backend: str = "auto",
    models_dir: Path | None = None,
    tts: bool = False,
) -> Check:
    """Require the selected backend's local models; no downloads or GPU loads."""
    if backend == "mlx":
        stt = (
            "mlx-parakeet-v3"
            if stt_backend == "parakeet-mlx" or (stt_backend == "auto" and lang == "en")
            else "mlx-whisper-large-v3-turbo"
        )
        translation = f"mlx-gemma4-{gemma4_size}" if model_family == "gemma4" else "translategemma-4b-it-mlx"
        names = [stt, translation]
    else:
        names = ["whisper-large-v3-turbo", f"gemma-4-{'e2b' if backend == 'cpu' else gemma4_size}-it-q4km.gguf"]
    direction = "es-en" if lang == "es" else "en-es"
    ct2 = resolve_marian_ct2(direction, project_root=project_root, models_dir=models_dir)
    if ct2 is None:
        names.append(f"marian-ct2-{direction}" if backend == "mlx" else f"marian-{direction}")

    missing = [
        name
        for name in names
        if resolve_model_path(name, models_dir=models_dir, project_root=project_root, local_only=True) is None
    ]
    if tts:
        from settings import settings

        voice = settings.tts.voices.get("en" if lang == "es" else "es")
        if not voice or resolve_piper_voice(voice, models_dir=models_dir) is None:
            missing.append(f"Piper voice {voice}")
    if missing:
        return _check(
            "Models",
            "fail",
            f"Missing local models: {', '.join(missing)}. Run stark-translate setup --backend {backend}"
            + (" --include tts" if tts else ""),
        )
    partial = f"CT2 ({ct2})" if ct2 else "HF CPU fallback"
    return _check("Models", "pass", f"{', '.join(names)}; Marian {partial}")


def check_microphone(input_device: int | str | None = None) -> Check:
    """Enumerate input audio devices via sounddevice (or report unavailable)."""
    try:
        import sounddevice as sd

        devices = sd.query_devices()
        if input_device is not None:
            selected = sd.query_devices(input_device, "input")
            if selected.get("max_input_channels", 0) <= 0:
                return _check("Microphone", "fail", "Selected device has no input channels")
    except Exception as exc:
        return _check("Microphone", "fail", f"Audio input unavailable: {exc}")

    inputs = [d for d in devices if d.get("max_input_channels", 0) > 0]
    if not inputs:
        return _check("Microphone", "fail", "No input devices found")
    names = [d.get("name", "?") for d in inputs[:3]]
    suffix = f" (+{len(inputs) - 3} more)" if len(inputs) > 3 else ""
    return _check("Microphone", "pass", f"{len(inputs)} input device(s): {', '.join(names)}{suffix}")


def check_adapter_manifest(project_root: Path) -> Check:
    """Verify adapters/manifest.json is parseable if it exists."""
    manifest = project_root / "adapters" / "manifest.json"
    if not manifest.exists():
        return _check("Adapter manifest", "warn", "No adapters/manifest.json — running with base models")
    try:
        data = json.loads(manifest.read_text())
    except json.JSONDecodeError as exc:
        return _check("Adapter manifest", "fail", f"Invalid JSON: {exc}")

    active = []
    for model_key, slots in data.items() if isinstance(data, dict) else []:
        if isinstance(slots, dict) and "active" in slots:
            active.append(f"{model_key}={slots['active']}")
    detail = f"Active: {', '.join(active)}" if active else "Manifest present but no active adapter slots"
    return _check("Adapter manifest", "pass", detail)


def check_llamacpp_server(url: str) -> Check:
    """Probe the configured llama-server URL."""
    health_url = f"{url.rstrip('/')}/health"
    try:
        with urllib.request.urlopen(health_url, timeout=2) as resp:
            body = resp.read().decode("utf-8", errors="replace")
    except urllib.error.URLError as exc:
        return _check(
            "llama-server",
            "warn",
            f"Not reachable at {url} ({exc.reason if hasattr(exc, 'reason') else exc}) — "
            "pipeline will use HF NF4 fallback. Start with ./start_server.sh",
        )
    if '"status":"ok"' in body:
        return _check("llama-server", "pass", f"Healthy at {url}")
    return _check("llama-server", "warn", f"Reachable but unhealthy at {url}: {body[:80]}")


def run_all_checks(
    project_root: Path | None = None,
    llamacpp_url: str | None = None,
    *,
    backend: str = "auto",
    lang: str = "en",
    model_family: str = "gemma4",
    gemma4_size: str = "e4b",
    tts: bool = False,
    diarize: bool = False,
    models_dir: Path | None = None,
    input_device: int | str | None = None,
    stt_backend: str = "auto",
) -> dict:
    """Cheap backend-aware preflight shared by CLI and selected operator config."""
    if project_root is None:
        project_root = Path(os.environ.get("STARK_PROJECT_ROOT", os.getcwd()))
    backend = resolve_backend(backend)
    if llamacpp_url is None:
        try:
            from settings import settings

            llamacpp_url = settings.cuda.llamacpp_url
        except Exception:
            llamacpp_url = "http://127.0.0.1:8090"
    checks = [
        check_gpu(),
        check_dependencies(backend, lang=lang, stt_backend=stt_backend, tts=tts, diarize=diarize),
        check_models(
            project_root,
            backend=backend,
            lang=lang,
            model_family=model_family,
            gemma4_size=gemma4_size,
            stt_backend=stt_backend,
            models_dir=models_dir,
            tts=tts,
        ),
        check_microphone(input_device),
        check_adapter_manifest(project_root),
    ]
    if backend in {"cuda", "cpu"}:
        checks.append(check_llamacpp_server(llamacpp_url))
    status_counts = {"pass": 0, "warn": 0, "fail": 0}
    for check in checks:
        status_counts[check["status"]] += 1
    return {"checks": checks, "ok": status_counts["fail"] == 0, "status_counts": status_counts, "backend": backend}
