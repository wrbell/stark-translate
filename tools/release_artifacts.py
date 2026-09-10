"""Build and verify complete source bundles; reject tag/version mismatches."""

from __future__ import annotations

import argparse
import json
import subprocess
import tarfile
import tomllib
import zipfile
from pathlib import Path

MAC_ROOTS = (
    "stark_translate",
    "operator_app",
    "engines",
    "tools",
    "features",
    "displays",
    "scripts",
    "launchd",
    "dry_run_ab.py",
    "workers.py",
    "settings.py",
    "models.lock.json",
    "pyproject.toml",
    "README.md",
    "LICENSE",
    "run_operator.sh",
    "bootstrap.sh",
    "start_server.sh",
    "requirements-mac.txt",
    "docs/packaging",
)
RUNTIME_REQUIRED = {
    "stark_translate/__init__.py",
    "operator_app/cli.py",
    "displays/operator/index.html",
    "engines/model_paths.py",
    "tools/audio_bridge_client.py",
    "features/live_diarize.py",
    "displays/audience_display.html",
    "dry_run_ab.py",
    "settings.py",
    "workers.py",
    "models.lock.json",
}


def validate_version(root: Path, tag: str | None = None, *, check_ref: bool = False) -> str:
    config = tomllib.loads((root / "pyproject.toml").read_text())
    version = config["project"]["version"]
    if config["tool"]["briefcase"]["version"] != version:
        raise ValueError("Briefcase version differs from project.version")
    if tag is not None and tag != f"v{version}":
        raise ValueError(f"Tag {tag!r} does not match project version v{version}")
    if tag is not None and check_ref:
        tagged = subprocess.run(
            ["git", "rev-parse", f"refs/tags/{tag}^{{commit}}"], cwd=root, text=True, capture_output=True, check=False
        )
        if tagged.returncode == 0:
            head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
            if tagged.stdout.strip() != head:
                raise ValueError(f"Tag {tag} does not point to this checkout")
    return version


def build_mac_bundle(root: Path, output_dir: Path, tag: str) -> Path:
    validate_version(root, tag)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"stark-translate-{tag}-mac.zip"
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for item in MAC_ROOTS:
            source = root / item
            if not source.exists():
                raise ValueError(f"Missing bundle input: {item}")
            for path in sorted(source.rglob("*") if source.is_dir() else [source]):
                if (
                    path.is_file()
                    and not path.is_symlink()
                    and "__pycache__" not in path.parts
                    and path.suffix != ".pyc"
                ):
                    archive.write(path, path.relative_to(root))
    verify_artifact(output)
    return output


def verify_artifact(path: Path) -> None:
    if path.name.endswith(".tar.gz"):
        with tarfile.open(path) as archive:
            names = {"/".join(name.split("/")[1:]) for name in archive.getnames()}
        required = RUNTIME_REQUIRED | {"pyproject.toml", "README.md"}
    else:
        with zipfile.ZipFile(path) as archive:
            names = set(archive.namelist())
        required = RUNTIME_REQUIRED
        if path.suffix == ".zip":
            required |= {"run_operator.sh", "bootstrap.sh", "scripts/runtime_env.sh", "pyproject.toml", "README.md"}
    missing = required - names
    if missing:
        raise ValueError(f"{path.name} missing runtime files: {', '.join(sorted(missing))}")
    if any("__pycache__" in name or name.endswith(".pyc") for name in names):
        raise ValueError(f"{path.name} includes Python cache files")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["validate", "mac", "verify"])
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--tag")
    parser.add_argument("--output-dir", type=Path, default=Path("dist"))
    parser.add_argument("artifacts", nargs="*", type=Path)
    args = parser.parse_args()
    if args.command == "verify":
        for artifact in args.artifacts:
            verify_artifact(artifact)
    elif args.command == "mac":
        print(build_mac_bundle(args.root, args.output_dir, args.tag or f"v{validate_version(args.root)}"))
    else:
        print(json.dumps({"version": validate_version(args.root, args.tag, check_ref=True)}))


if __name__ == "__main__":
    main()
