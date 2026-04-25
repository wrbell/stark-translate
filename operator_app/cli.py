"""``stark-translate`` CLI entry point.

Subcommands:

    stark-translate operator [--port 9000] [--host 0.0.0.0] [--no-browser]
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
    """Return the installed package version, falling back to pyproject if dev-installed."""
    try:
        return metadata.version("stark-translate")
    except metadata.PackageNotFoundError:
        # Dev environment: read from pyproject.toml at the repo root.
        here = Path(__file__).resolve().parent.parent
        pyproject = here / "pyproject.toml"
        if pyproject.exists():
            for line in pyproject.read_text().splitlines():
                if line.startswith("version = "):
                    return line.split('"')[1]
        return "0.0.0+dev"


def cmd_version(_args: argparse.Namespace) -> int:
    print(_resolve_version())
    return 0


def cmd_operator(args: argparse.Namespace) -> int:
    """Launch FastAPI on the configured port and open the browser."""
    import uvicorn

    url = f"http://{args.host if args.host != '0.0.0.0' else 'localhost'}:{args.port}/operator/"
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
        allow_patterns=args.allow if args.allow else None,
    )


def cmd_doctor(args: argparse.Namespace) -> int:
    """Run preflight checks from the CLI."""
    # Lazy import — preflight pulls in optional sounddevice.
    from operator_app.preflight import run_all_checks

    project_root = Path(os.environ.get("STARK_PROJECT_ROOT", os.getcwd()))
    payload = run_all_checks(project_root=project_root)

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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="stark-translate",
        description="Live bilingual speech-to-text for Stark Road Gospel Hall.",
    )
    parser.set_defaults(func=None)
    sub = parser.add_subparsers(metavar="COMMAND")

    p_op = sub.add_parser("operator", help="Launch the FastAPI control plane + browser UI")
    p_op.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
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
    p_setup.set_defaults(func=cmd_setup)

    p_doctor = sub.add_parser("doctor", help="Run preflight checks (same as operator UI)")
    p_doctor.add_argument("--json", action="store_true", help="Emit JSON instead of human-readable text")
    p_doctor.set_defaults(func=cmd_doctor)

    p_ver = sub.add_parser("version", help="Print installed version")
    p_ver.set_defaults(func=cmd_version)

    args = parser.parse_args(argv)
    if args.func is None:
        parser.print_help(sys.stderr)
        return 2
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
