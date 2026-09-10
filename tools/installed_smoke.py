"""Exercise an installed wheel from outside a checkout, without model loads."""

from __future__ import annotations

import json
import sys
from importlib.metadata import version
from pathlib import Path


def main() -> None:
    from fastapi.testclient import TestClient

    import operator_app
    from operator_app.main import app
    from operator_app.setup import load_lockfile
    from training.theological_canaries import canary_sentences

    package = Path(operator_app.__file__).resolve()
    if not package.is_relative_to(Path(sys.prefix).resolve()):
        raise RuntimeError(f"Smoke imported checkout instead of installed wheel: {package}")
    runtime_root = package.parent.parent
    runtime_files = (
        "dry_run_ab.py",
        "workers.py",
        "tools/session_lifecycle.py",
        "tools/vad_runtime.py",
        "tools/marian_ct2_setup.py",
        "training/theological_canaries.py",
        "displays/audience_display.html",
        "displays/caption_telemetry.js",
        "displays/display_connection.js",
        "displays/operator/review.js",
    )
    missing = [name for name in runtime_files if not (runtime_root / name).is_file()]
    if missing:
        raise RuntimeError(f"Installed wheel is missing runtime files: {missing}")
    with TestClient(app) as client:
        responses = {path: client.get(path).status_code for path in ("/healthz", "/operator/", "/operator/review.js")}
    if not all(status == 200 for status in responses.values()):
        raise RuntimeError(f"Installed operator endpoints failed: {responses}")
    manifest = load_lockfile()
    assert "mlx-parakeet-v3" in manifest["models"]
    assert canary_sentences(1)[0]["en"]
    print(
        json.dumps(
            {
                "version": version("stark-translate"),
                "http": responses,
                "model_entries": len(manifest["models"]),
                "package": str(package),
                "runtime_files": len(runtime_files),
            }
        )
    )


if __name__ == "__main__":
    main()
