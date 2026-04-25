"""Pre-flight checks for an operator session.

Each check returns a ``Check`` dict with a ``name``, ``status`` (one of
``"pass" | "warn" | "fail"``), and ``detail`` string. The frontend renders
these as green / yellow / red rows; the operator can only "Start Session"
when no check is ``fail``.

Checks are intentionally cheap (no model loads, no network round-trips
beyond a single ``/health``) so the page can poll them every few seconds.
"""

from __future__ import annotations

import json
import os
import shutil
import urllib.error
import urllib.request
from pathlib import Path
from typing import Literal

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


def check_models(project_root: Path) -> Check:
    """Verify Gemma 4 GGUFs exist (production CUDA path) and Whisper is installable."""
    models_dir = project_root / "models"
    e2b = models_dir / "gemma-4-e2b-it-q4km.gguf"
    e4b = models_dir / "gemma-4-e4b-it-q4km.gguf"
    have = [p.name for p in (e2b, e4b) if p.exists()]
    missing = [p.name for p in (e2b, e4b) if not p.exists()]

    if not have:
        return _check(
            "Translation models",
            "warn",
            "No Gemma 4 GGUFs found — pipeline will fall back to HF NF4 (slower, more VRAM)",
        )
    if missing:
        return _check(
            "Translation models",
            "warn",
            f"Found {', '.join(have)} but missing {', '.join(missing)} — A/B mode disabled",
        )
    return _check("Translation models", "pass", f"Found {', '.join(have)}")


def check_microphone() -> Check:
    """Enumerate input audio devices via sounddevice (or report unavailable)."""
    try:
        import sounddevice as sd

        devices = sd.query_devices()
    except Exception as exc:
        return _check("Microphone", "warn", f"sounddevice unavailable: {exc}")

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


def run_all_checks(project_root: Path | None = None, llamacpp_url: str | None = None) -> dict:
    """Run every preflight check and return an aggregated payload.

    Output:
      {
        "checks": [Check, ...],
        "ok": bool,            # True if no check is "fail"
        "status_counts": {"pass": N, "warn": N, "fail": N},
      }
    """
    if project_root is None:
        project_root = Path(os.environ.get("STARK_PROJECT_ROOT", os.getcwd()))
    if llamacpp_url is None:
        try:
            from settings import settings

            llamacpp_url = settings.cuda.llamacpp_url
        except Exception:
            llamacpp_url = "http://127.0.0.1:8090"

    checks = [
        check_gpu(),
        check_models(project_root),
        check_microphone(),
        check_adapter_manifest(project_root),
        check_llamacpp_server(llamacpp_url),
    ]

    status_counts = {"pass": 0, "warn": 0, "fail": 0}
    for c in checks:
        status_counts[c["status"]] += 1

    return {
        "checks": checks,
        "ok": status_counts["fail"] == 0,
        "status_counts": status_counts,
    }
