# Model setup and offline resolution

Setup, preflight and inference share `models.lock.json` and
`engines/model_paths.py`. Mac defaults are Parakeet STT for English, Whisper
Turbo for Spanish, Gemma 4 E4B OptiQ finals and Marian in both directions.

```bash
stark-translate setup --backend mlx
stark-translate setup --backend mlx --include e2b tts
stark-translate setup --backend mlx --models-dir /path/to/cache
stark-translate doctor --backend mlx --lang en
```

The CLI defaults to backend auto-detection. `required_for` selects the default
entries for that backend; `optional_group` selects profiles requested through
`--include`. Setup can reuse complete, pinned Hugging Face snapshots already on
disk. A custom setup directory must also be supplied as `STARK_MODELS_DIR` when
launching the operator or pipeline.

## Manifest

Hugging Face model entries pin a commit revision and required files. This is an
example from the current manifest:

```json
{
  "mlx-parakeet-v3": {
    "type": "hf-snapshot",
    "repo_id": "mlx-community/parakeet-tdt-0.6b-v3",
    "revision": "ed2b7e8c15f9aaa0b5772e2efb986255eaef7e15",
    "subdir": "mlx-parakeet-v3",
    "required_for": ["mlx"],
    "required_files": ["config.json", "tokenizer.model"]
  }
}
```

`hf-snapshot` entries use `huggingface_hub.snapshot_download`; optional
`allow_patterns` restrict downloads, including Piper ONNX voice/config files.
Preflight checks required files, weight presence and indexed weight shards.
`direct` entries pin a download URL, filename, expected size and SHA-256;
the checksum is verified after download. Their `.partial` file supports resume
through HTTP Range.

## Cache lookup

| Priority | Location |
|---|---|
| 1 | Explicit existing model path supplied by the caller |
| 2 | `STARK_MODELS_DIR`, or the platform's setup cache |
| 3 | Project `models/` directory |
| 4 | Local Hugging Face snapshot at the manifest revision |

The setup cache defaults to `$XDG_CACHE_HOME/stark-translate/models` on Mac/Linux
(falling back to `~/.cache/stark-translate/models`) and
`%LOCALAPPDATA%\stark-translate\models` on Windows. Hugging Face cache overrides
`HF_HOME` and `HF_HUB_CACHE` are respected. The preflight path is offline; missing
required files prevent Start.

## Installation sidecars

Snapshot setup writes `.installed` inside the model directory with `repo_id`,
`revision`, `lockfile_version` and `installed_at`. Direct downloads use
`<filename>.installed` with checksum and manifest version. Setup uses these
markers to decide whether its installed copy can be reused; a changed snapshot
revision or manifest version prompts setup to update that entry. Runtime lookup
rejects a managed snapshot whose marker identifies a different repository or
revision, and can fall back to the matching local Hugging Face snapshot.

Explicit model paths remain an intentional override. Custom adapters and
unversioned local copies should be recorded separately when evaluating quality;
completion metadata distinguishes resolved revisions from manifest revisions.

## Optional models

`--include e2b` adds the faster Gemma profile; `--include tts` adds EN/ES Piper
voices; `--include translategemma` adds the legacy translation family. Marian
CT2 conversion remains a separate step from model setup. See
[macos.md](./macos.md#models-and-offline-readiness) for the conversion workflow.

Update the model manifest whenever a pinned model revision or installation
requirement changes. App versions and manifest versions are independent.
