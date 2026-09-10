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

    package = Path(operator_app.__file__).resolve()
    if not package.is_relative_to(Path(sys.prefix).resolve()):
        raise RuntimeError(f"Smoke imported checkout instead of installed wheel: {package}")
    with TestClient(app) as client:
        responses = {path: client.get(path).status_code for path in ("/healthz", "/operator/")}
    if not all(status == 200 for status in responses.values()):
        raise RuntimeError(f"Installed operator endpoints failed: {responses}")
    manifest = load_lockfile()
    assert "mlx-parakeet-v3" in manifest["models"]
    print(
        json.dumps(
            {
                "version": version("stark-translate"),
                "http": responses,
                "model_entries": len(manifest["models"]),
                "package": str(package),
            }
        )
    )


if __name__ == "__main__":
    main()
