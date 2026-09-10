"""Exercise an installed wheel from outside a checkout, without model loads."""

from __future__ import annotations

import json
import sys
from importlib.metadata import version
from pathlib import Path


def check_operator_routes(app) -> dict[str, int]:
    """Exercise the actual localhost boundary, outside pytest's client fixture."""
    from fastapi.testclient import TestClient

    with TestClient(app, base_url="http://127.0.0.1", client=("127.0.0.1", 50000)) as client:
        return {
            path: client.get(path).status_code
            for path in (
                "/healthz",
                "/operator/",
                "/operator/review.js",
                "/operator/widgets/qr.js",
                "/api/capabilities",
            )
        }


def main() -> None:
    import operator_app
    from features.extract_verses import VerseExtractor
    from operator_app.main import app
    from operator_app.setup import load_lockfile
    from tools.release_artifacts import RUNTIME_REQUIRED
    from training.theological_canaries import canary_sentences

    package = Path(operator_app.__file__).resolve()
    if not package.is_relative_to(Path(sys.prefix).resolve()):
        raise RuntimeError(f"Smoke imported checkout instead of installed wheel: {package}")
    runtime_root = package.parent.parent
    runtime_files = sorted(RUNTIME_REQUIRED)
    missing = [name for name in runtime_files if not (runtime_root / name).is_file()]
    if missing:
        raise RuntimeError(f"Installed wheel is missing runtime files: {missing}")
    responses = check_operator_routes(app)
    if not all(status == 200 for status in responses.values()):
        raise RuntimeError(f"Installed operator endpoints failed: {responses}")
    manifest = load_lockfile()
    assert "mlx-parakeet-v3" in manifest["models"]
    assert canary_sentences(1)[0]["en"]
    verses = VerseExtractor()
    verses.extract_from_text("Luke twenty three and verse uh thirty two.")
    verses.extract_from_text("Philemon 50:11.")
    if [row["reference"] for row in verses.references] != ["Luke 23:32"]:
        raise RuntimeError("Installed verse parser or structural bounds failed")
    print(
        json.dumps(
            {
                "version": version("stark-translate"),
                "http": responses,
                "model_entries": len(manifest["models"]),
                "package": str(package),
                "runtime_files": len(runtime_files),
                "verse_parser": "passed",
            }
        )
    )


if __name__ == "__main__":
    main()
