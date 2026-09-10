"""Pinned Marian source resolution and atomic CPU CT2 installation."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import uuid
from datetime import UTC, datetime
from pathlib import Path

from engines.model_paths import (
    default_models_dir,
    load_model_manifest,
    marian_ct2_complete,
    resolve_marian_ct2,
    resolve_model_path,
)

COPY_FILES = ["source.spm", "target.spm", "vocab.json", "tokenizer_config.json", "special_tokens_map.json"]


def resolve_hf_source(
    model_id: str, *, project_root: Path | None = None, models_dir: Path | None = None, revision: str | None = None
) -> tuple[Path, dict]:
    """Use the pinned shared cache; remote custom sources require a full commit."""
    explicit = Path(model_id).expanduser()
    if explicit.exists():
        return explicit.resolve(), {
            "model_id": model_id,
            "source_revision": None,
            "source_path": str(explicit.resolve()),
            "source_kind": "explicit_local",
        }
    models = load_model_manifest(project_root).get("models", {})
    entry: dict = next((v for key, v in models.items() if model_id in {key, v.get("repo_id")}), {})
    repo_id = entry.get("repo_id", model_id)
    selected_revision = revision or entry.get("revision")
    if not isinstance(selected_revision, str) or not re.fullmatch(r"[0-9a-f]{40}", selected_revision):
        raise ValueError(f"A full pinned HF commit is required for {model_id}; register it or pass --revision")
    cached = None
    if selected_revision == entry.get("revision"):
        cached = resolve_model_path(model_id, project_root=project_root, models_dir=models_dir, local_only=True)
    if cached is not None:
        candidate = Path(cached).resolve()
        pinned = candidate.parent.name == "snapshots" and candidate.name == selected_revision
        marker = candidate / ".installed"
        if marker.is_file():
            try:
                data = json.loads(marker.read_text())
                pinned = (
                    isinstance(data, dict)
                    and data.get("repo_id") == repo_id
                    and data.get("revision") == selected_revision
                )
            except (OSError, ValueError):
                pinned = False
        if not pinned:
            cached = None
    if cached is None:
        from huggingface_hub import snapshot_download

        cached = snapshot_download(repo_id=repo_id, revision=selected_revision)
    source = Path(cached).resolve()
    if not source.is_dir():
        raise ValueError(f"Pinned source is not a local snapshot: {source}")
    return source, {
        "model_id": repo_id,
        "source_revision": selected_revision,
        "source_path": str(source),
        "source_kind": "pinned_hf_snapshot",
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(data)
    return digest.hexdigest()


def convert_source(source: Path, output: Path, quantization: str, direction: str) -> dict:
    """Run conversion and one nonempty CPU smoke in the selected interpreter."""
    copies = [name for name in COPY_FILES if (source / name).is_file()]
    command = [
        sys.executable,
        "-m",
        "ctranslate2.converters.transformers",
        "--model",
        str(source),
        "--output_dir",
        str(output),
        "--quantization",
        quantization,
        "--force",
    ]
    if copies:
        command.extend(["--copy_files", *copies])
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    subprocess.run(command, check=True, env=env)
    if not marian_ct2_complete(output):
        raise ValueError("CT2 conversion did not produce the complete tokenizer/model artifact")
    script = """
import json, sys
import ctranslate2
from transformers import MarianTokenizer
path, direction = sys.argv[1:]
tokenizer = MarianTokenizer.from_pretrained(path, local_files_only=True)
engine = ctranslate2.Translator(path, device="cpu", compute_type="int8", intra_threads=2)
text = "The grace of God is sufficient." if direction == "en-es" else "La gracia de Dios es suficiente."
tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(text))
result = engine.translate_batch([tokens], max_decoding_length=64)[0].hypotheses[0]
translation = tokenizer.decode(tokenizer.convert_tokens_to_ids(result), skip_special_tokens=True)
if not translation.strip():
    raise RuntimeError("Converted Marian produced an empty smoke translation")
print(json.dumps({"device": "cpu", "nonempty": True, "input": text, "translation": translation}))
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(output), direction], check=True, env=env, capture_output=True, text=True
    )
    return {"command": command, "smoke": json.loads(result.stdout.splitlines()[-1])}


def ensure_managed_marian(
    entry: dict, *, project_root: Path, models_dir: Path | None = None, refresh: bool = False
) -> tuple[Path, bool]:
    """Preserve adapters; publish only complete validated new cache artifacts."""
    models_dir = models_dir or default_models_dir()
    direction = entry["direction"]
    existing = resolve_marian_ct2(direction, project_root=project_root, models_dir=models_dir)
    managed = models_dir / entry["subdir"]
    if existing is not None and (not refresh or not Path(existing).is_relative_to(managed.resolve())):
        return Path(existing), False
    manifest = load_model_manifest(project_root)
    source_entry = manifest["models"][entry["source_model"]]
    source, provenance = resolve_hf_source(source_entry["repo_id"], project_root=project_root, models_dir=models_dir)
    if provenance.get("source_revision") != source_entry["revision"]:
        raise ValueError("Managed Marian conversion requires its pinned HF source revision")
    managed.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".staging-", dir=managed))
    artifact_name = "build-" + uuid.uuid4().hex
    artifact = managed / artifact_name
    published = False
    pointer_tmp = managed / (".active-" + uuid.uuid4().hex + ".tmp")
    try:
        conversion = convert_source(source, staging, entry["quantization"], direction)
        if not marian_ct2_complete(staging):
            raise ValueError("Cannot publish an incomplete Marian CT2 conversion")
        weights = staging / "model.bin"
        exported = {
            "version": "1.1",
            **provenance,
            "direction": direction,
            "ct2_quantization": entry["quantization"],
            "ct2_version": importlib.metadata.version("ctranslate2"),
            "exported_at_utc": datetime.now(UTC).isoformat(),
            "model_bin_sha256": _sha256(weights),
            "model_bin_size_bytes": weights.stat().st_size,
            "files": {p.name: _sha256(p) for p in staging.iterdir() if p.is_file()},
            "converter_python": sys.executable,
            **conversion,
        }
        (staging / "export_manifest.json").write_text(json.dumps(exported, indent=2) + "\n")
        staging.rename(artifact)
        pointer_tmp.write_text(json.dumps({"directory": artifact_name, "manifest_version": manifest["version"]}) + "\n")
        os.replace(pointer_tmp, managed / "active.json")
        published = True
        return artifact.resolve(), True
    finally:
        if staging.exists():
            shutil.rmtree(staging)
        if pointer_tmp.exists():
            pointer_tmp.unlink()
        if not published and artifact.exists():
            shutil.rmtree(artifact)
