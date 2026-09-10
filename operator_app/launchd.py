"""Explicit per-user launchd installation, generated from actual install paths."""

from __future__ import annotations

import os
import platform
import plistlib
import subprocess
import sys
from pathlib import Path

LABEL = "com.starkroad.translate"


def launchd_plist(
    project_root: Path, python: Path, *, host: str = "127.0.0.1", port: int = 9000, profile: str | None = None
) -> dict:
    from stark_translate.profiles import resolve_profile

    selected_profile = resolve_profile(profile).name
    root = project_root.resolve()
    # Do not resolve Python symlinks: a venv's interpreter path selects that venv.
    interpreter = python.expanduser().absolute()
    logs = Path(os.environ.get("STARK_OPERATOR_LOG_DIR", root / "metrics")).expanduser().absolute()
    environment = {"STARK_PROJECT_ROOT": str(root), "STARK_OPERATOR_LOG_DIR": str(logs)}
    if os.environ.get("STARK_MODELS_DIR"):
        environment["STARK_MODELS_DIR"] = os.environ["STARK_MODELS_DIR"]
    return {
        "Label": LABEL,
        "ProgramArguments": [
            str(interpreter),
            "-m",
            "operator_app.cli",
            "operator",
            "--no-browser",
            "--profile",
            selected_profile,
            "--host",
            host,
            "--port",
            str(port),
        ],
        "WorkingDirectory": str(root),
        "EnvironmentVariables": environment,
        "RunAtLoad": True,
        "KeepAlive": {"SuccessfulExit": False},
        "ThrottleInterval": 5,
        "StandardOutPath": str(logs / "launchd-stdout.log"),
        "StandardErrorPath": str(logs / "launchd-stderr.log"),
    }


def manage_launchd(
    action: str,
    *,
    project_root: Path,
    python: Path | None = None,
    output: Path | None = None,
    profile: str | None = None,
) -> int:
    """Render is portable; install/uninstall explicitly call launchctl on macOS."""
    path = output or Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist"
    if action != "render" and platform.system() != "Darwin":
        raise ValueError("launchd install/uninstall is only supported on macOS")
    if action == "uninstall":
        if path.exists():
            subprocess.run(["launchctl", "bootout", f"gui/{os.getuid()}", str(path)], check=False)
            path.unlink()
        return 0
    config = launchd_plist(project_root, python or Path(sys.executable), profile=profile)
    if action == "render" and output is None:
        print(plistlib.dumps(config).decode())
        return 0
    interpreter = Path(config["ProgramArguments"][0])
    if not interpreter.is_file():
        raise ValueError(f"Python interpreter does not exist: {interpreter}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if action == "install" and path.exists():
        subprocess.run(["launchctl", "bootout", f"gui/{os.getuid()}", str(path)], check=False)
    path.write_bytes(plistlib.dumps(config))
    if action == "install":
        Path(config["StandardOutPath"]).parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["launchctl", "bootstrap", f"gui/{os.getuid()}", str(path)], check=True)
    return 0
