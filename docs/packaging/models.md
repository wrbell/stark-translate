# Models — bootstrap and lockfile

All channels (PyPI, Linux Docker, Windows MSI) read the same
`models.lock.json` at the repo root. It pins URLs, sizes, and SHA-256 for
every artifact the operator needs.

```bash
stark-translate setup                  # download missing entries
stark-translate setup --refresh        # re-download even if .installed sidecar matches
stark-translate setup --models-dir /path/to/cache
```

---

## Lockfile schema

```json
{
  "version": "2026.7.0",
  "models": {
    "gemma-4-e4b-it-q4km.gguf": {
      "type": "direct",
      "url": "https://huggingface.co/.../gemma-4-e4b-it-q4km.gguf",
      "filename": "gemma-4-e4b-it-q4km.gguf",
      "size_bytes": 5234567890,
      "sha256": "abc...",
      "required_for": ["cuda"]
    },
    "openai/whisper-large-v3-turbo": {
      "type": "hf-snapshot",
      "repo_id": "openai/whisper-large-v3-turbo",
      "revision": "main",
      "subdir": "whisper-large-v3-turbo",
      "required_for": ["cuda", "mlx", "cpu"]
    }
  }
}
```

Two entry types:

- **`direct`** — single file (typically GGUFs). Fields: `url`, `filename`,
  `size_bytes`, `sha256` (verified after download), `required_for`.
- **`hf-snapshot`** — HuggingFace snapshot (multi-file repo). Fields:
  `repo_id`, `revision`, `subdir`, optional `allow_patterns` to skip
  irrelevant files.

`required_for` is informational — the operator's `/api/preflight` only flags
missing files for the active backend.

---

## Cache layout

| Platform | Default cache root |
|---|---|
| Linux/Mac | `$XDG_CACHE_HOME/stark-translate/models`, falls back to `~/.cache/stark-translate/models` |
| Windows | `%LOCALAPPDATA%\stark-translate\models` |

Override with `STARK_MODELS_DIR` env var or `--models-dir` CLI flag. The
operator preflight reads the same path and refuses Start when required
entries are missing.

---

## `.installed` sidecars

Every successful download writes a `<filename>.installed` JSON sidecar:

```json
{
  "lockfile_version": "2026.7.0",
  "sha256": "abc...",
  "installed_at": "2026-04-25T18:42:00"
}
```

Re-runs of `stark-translate setup` check sidecar presence + matching SHA-256
to skip without re-hashing 5 GB GGUFs. Bumping the lockfile version triggers
a re-download even when sidecars are present.

---

## Resumable downloads

For `direct` entries, interrupted downloads land in `<filename>.partial`.
The next run resumes via HTTP Range from the partial size. Use case:
flaky church Wi-Fi mid-download of a 5 GB GGUF.

`hf-snapshot` resumability is whatever `huggingface_hub` provides natively,
which for v0.24+ is robust per-file resume.

---

## Cadence

| Layer | Bumps |
|---|---|
| App version (`pyproject.toml`) | Monthly CalVer (`2026.7.0`, `2026.7.1`, …) |
| Model version (`models.lock.json`) | Yearly, or when an adapter ships |
| GGUF binary churn | Pinned commit on llama.cpp side (`LLAMA_CPP_REF` in Dockerfile) |

The operator UI surfaces a "models out of date" warning when the lockfile
version on disk differs from the installed sidecar version. Run
`stark-translate setup --refresh` to clear it.

---

## See also

- [`pypi.md`](./pypi.md) — `stark-translate setup` CLI behavior
- [`linux-docker.md`](./linux-docker.md) — bind-mount layout for Docker
- [`../tools/`](../../tools/) — `tools/lock_data.py` is the primitive these
  patterns came from
