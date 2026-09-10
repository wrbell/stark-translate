"""``stark-translate`` CLI entry point.

Subcommands:

    stark-translate operator [--port 9000] [--host 127.0.0.1] [--no-browser]
        Launch the FastAPI control plane and (by default) open the operator
        UI in the user's default browser.

    stark-translate setup [--models-dir PATH] [--refresh] [--allow PATTERN]
        Idempotent model bootstrap. Reads models.lock.json, downloads each
        entry to the configured cache, verifies SHA-256, writes a .installed
        sidecar. Resumable.

    stark-translate doctor [--json]
        Run /api/preflight from the CLI without launching uvicorn. Prints
        the same checks the operator UI shows and exits 0/1.

    stark-translate version
        Print the installed package version.

Run ``stark-translate --help`` for full usage.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import webbrowser
from importlib import metadata
from pathlib import Path


def _resolve_version() -> str:
    """Report this checkout's version, or installed metadata outside a checkout."""
    import tomllib

    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    if pyproject.is_file():
        config = tomllib.loads(pyproject.read_text())
        if config.get("project", {}).get("name") == "stark-translate":
            return config["project"]["version"]
    try:
        return metadata.version("stark-translate")
    except metadata.PackageNotFoundError:
        return "0.0.0+dev"


def cmd_version(_args: argparse.Namespace) -> int:
    print(_resolve_version())
    return 0


def cmd_operator(args: argparse.Namespace) -> int:
    """Launch FastAPI on the configured port and open the browser."""
    import uvicorn

    from operator_app.security import configure_operator_host

    configure_operator_host(args.host)
    # Select a usable browser address; this comparison does not bind a socket.
    browser_host = "localhost" if args.host in {"0.0.0.0", "::"} else args.host  # nosec B104
    if ":" in browser_host:
        browser_host = f"[{browser_host}]"
    url = f"http://{browser_host}:{args.port}/operator/"
    if not args.no_browser:
        # Open AFTER uvicorn binds — but uvicorn.run blocks. Use a small thread.
        import threading
        import time

        def _open_when_ready() -> None:
            time.sleep(1.5)  # give uvicorn time to bind
            try:
                webbrowser.open(url)
            except Exception:
                pass

        threading.Thread(target=_open_when_ready, daemon=True).start()

    print(f"stark-translate operator → {url}", file=sys.stderr)
    uvicorn.run(
        "operator_app.main:app",
        host=args.host,
        port=args.port,
        log_level=args.log_level.lower(),
        reload=False,
    )
    return 0


def cmd_setup(args: argparse.Namespace) -> int:
    """Bootstrap models from models.lock.json (or just check URLs with --check)."""
    if args.check:
        from operator_app.setup import check_lockfile_urls

        result = check_lockfile_urls()
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            if "error" in result:
                print(f"setup --check: {result['error']}", file=sys.stderr)
                return 2
            print(f"Lockfile URL check: {len(result['checks'])} entries")
            for c in result["checks"]:
                glyph = "✓" if c["status"] == "pass" else "✗"
                print(f"  {glyph} {c['name']:36s} {c['detail']:24s} {c['url']}")
        return 0 if result.get("ok") else 1

    from operator_app.setup import bootstrap_models

    models_dir = Path(args.models_dir) if args.models_dir else None
    return bootstrap_models(
        models_dir=models_dir,
        refresh=args.refresh,
        backend=args.backend,
        include=args.include,
        allow_patterns=args.allow if args.allow else None,
        profile=getattr(args, "profile", None),
        offline=getattr(args, "offline", False),
        build_native=getattr(args, "build_native", False),
        converter_python=getattr(args, "converter_python", None),
    )


def cmd_doctor(args: argparse.Namespace) -> int:
    """Run preflight checks from the CLI."""
    # Lazy import — preflight pulls in optional sounddevice.
    from operator_app.preflight import run_all_checks

    project_root = Path(os.environ.get("STARK_PROJECT_ROOT", os.getcwd()))
    payload = run_all_checks(
        project_root=project_root,
        backend=args.backend,
        profile=getattr(args, "profile", None),
        lang=args.lang,
        tts=args.tts,
        diarize=args.diarize,
        models_dir=Path(args.models_dir) if args.models_dir else None,
    )

    if args.json:
        print(json.dumps(payload, indent=2))
        return 0 if payload["ok"] else 1

    counts = payload["status_counts"]
    print(f"Pre-flight: {counts['pass']} pass, {counts['warn']} warn, {counts['fail']} fail")
    print()
    for check in payload["checks"]:
        glyph = {"pass": "✓", "warn": "!", "fail": "✗"}[check["status"]]
        print(f"  {glyph} {check['name']}: {check['detail']}")
    return 0 if payload["ok"] else 1


def cmd_launchd(args: argparse.Namespace) -> int:
    from operator_app.launchd import manage_launchd

    return manage_launchd(
        args.action,
        project_root=Path(args.project_root),
        python=Path(args.python) if args.python else None,
        output=Path(args.output) if args.output else None,
        profile=getattr(args, "profile", None),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="stark-translate",
        description="Live bilingual speech-to-text for Stark Road Gospel Hall.",
    )
    parser.set_defaults(func=None)
    sub = parser.add_subparsers(metavar="COMMAND")

    p_op = sub.add_parser("operator", help="Launch the FastAPI control plane + browser UI")
    p_op.add_argument(
        "--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1; remote access has no authentication)"
    )
    p_op.add_argument("--port", type=int, default=9000, help="Bind port (default: 9000)")
    p_op.add_argument("--no-browser", action="store_true", help="Don't open the browser")
    p_op.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="uvicorn log level",
    )
    p_op.set_defaults(func=cmd_operator)

    p_setup = sub.add_parser("setup", help="Download + verify models from models.lock.json")
    p_setup.add_argument(
        "--models-dir",
        default=None,
        help="Cache directory (default: ~/.cache/stark-translate/models on Unix, "
        "%%LOCALAPPDATA%%/stark-translate/models on Windows)",
    )
    p_setup.add_argument(
        "--refresh",
        action="store_true",
        help="Re-download models even if .installed sidecars match the lockfile",
    )
    p_setup.add_argument(
        "--allow",
        nargs="*",
        default=None,
        metavar="PATTERN",
        help="Glob pattern(s) for HF snapshots (passed to allow_patterns)",
    )
    p_setup.add_argument(
        "--check",
        action="store_true",
        help="HEAD every URL in models.lock.json and report reachability "
        "without downloading. Useful before cutting a release tag.",
    )
    p_setup.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON instead of human-readable output (with --check)",
    )
    p_setup.add_argument("--backend", choices=["auto", "mlx", "cuda", "cpu"], default="auto")
    p_setup.add_argument(
        "--include",
        nargs="*",
        choices=["e2b", "tts", "translategemma", "whisper-fallback", "diarization", "diarization-pyannote"],
        default=[],
    )
    p_setup.set_defaults(func=cmd_setup)

    p_doctor = sub.add_parser("doctor", help="Run preflight checks (same as operator UI)")
    p_doctor.add_argument("--json", action="store_true", help="Emit JSON instead of human-readable text")
    p_doctor.add_argument("--backend", choices=["auto", "mlx", "cuda", "cpu"], default="auto")
    p_doctor.add_argument("--lang", choices=["en", "es"], default="en")
    p_doctor.add_argument("--models-dir")
    p_doctor.add_argument("--tts", action="store_true")
    p_doctor.add_argument("--diarize", action="store_true")
    p_doctor.set_defaults(func=cmd_doctor)

    p_service = sub.add_parser("launchd", help="Explicit macOS login service install/uninstall or preview")
    p_service.add_argument("action", choices=["render", "install", "uninstall"])
    p_service.add_argument("--project-root", default=os.environ.get("STARK_PROJECT_ROOT", os.getcwd()))
    p_service.add_argument("--python", help="Absolute venv interpreter path (default: current interpreter)")
    p_service.add_argument("--output", help="Plist destination; render prints to stdout when omitted")
    p_service.set_defaults(func=cmd_launchd)

    p_ver = sub.add_parser("version", help="Print installed version")
    p_ver.set_defaults(func=cmd_version)

    from stark_translate.profiles import PROFILE_NAMES

    for command in (p_op, p_setup, p_doctor, p_service):
        command.add_argument("--profile", choices=PROFILE_NAMES, default=os.environ.get("STARK_PROFILE", "standard"))
    p_setup.add_argument("--offline", action="store_true", help="Use only verified prepared cache; never download")
    p_setup.add_argument("--build-native", action="store_true", help="Build pinned llama.cpp sm_75 on Linux CUDA")
    p_setup.add_argument("--converter-python", help="Separate interpreter with lite-build extra for Marian CT2 setup")
    args = parser.parse_args(argv)
    if hasattr(args, "profile"):
        os.environ["STARK_PROFILE"] = args.profile
    if getattr(args, "offline", False):
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    if args.func is None:
        parser.print_help(sys.stderr)
        return 2
    return args.func(args)


def lite_main(argv: list[str] | None = None) -> int:
    """Same application, explicit CPU product default (including on a Mac)."""
    os.environ.setdefault("STARK_PROFILE", "lite-cpu")
    return main(argv)


if __name__ == "__main__":
    sys.exit(main())
