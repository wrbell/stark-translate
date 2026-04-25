# PyPI / uv install — v2026.7+

For developers, contributors, and Mac volunteers comfortable installing
Python tooling. Single-command install in <10 s on a machine that already
has `uv`.

```bash
# Linux/NVIDIA
uv tool install 'stark-translate[cuda]'

# Apple Silicon Mac
uv tool install 'stark-translate[mlx]'

# CPU-only fallback (any machine)
uv tool install 'stark-translate[cpu]'
```

Then bootstrap models and launch the operator:

```bash
stark-translate setup            # downloads ~10 GB of models, idempotent + resumable
stark-translate doctor           # /api/preflight from the CLI
stark-translate operator         # opens browser to http://localhost:9000/operator/
```

---

## Extras matrix

| Extra | Includes | Use it when |
|---|---|---|
| `cuda` | torch, transformers, ctranslate2≥4.5, faster-whisper, bitsandbytes, piper-tts | NVIDIA GPU on Linux |
| `mlx` | mlx, mlx-whisper, mlx-lm, transformers, scipy, piper-tts | Apple Silicon (M1+) |
| `cpu` | torch, transformers, ctranslate2≥4.5, faster-whisper, scipy, piper-tts | No GPU, willing to wait |
| `dev` | ruff, mypy, pytest, pytest-cov, pre-commit, bandit, vulture, httpx, build | Contributing |

`stark-translate doctor` reports which backend was selected at runtime — the
factory in `engines/factory.py` auto-detects MLX vs CUDA, with `STARK_BACKEND`
as an explicit override.

### CTranslate2 version pinning per platform

The `cuda` and `cpu` extras pin `ctranslate2>=4.5`, which requires CUDA
12.3+/cuDNN 9. If your toolchain is older:

| Toolchain | Required pin |
|---|---|
| CUDA 12 + cuDNN 9 | `ctranslate2>=4.5` (default) |
| CUDA 12 + cuDNN 8 | `ctranslate2<=4.4.0` |
| CUDA 11 + cuDNN 8 | `ctranslate2<=3.24.0` |

Override via `pip install 'stark-translate[cuda]' 'ctranslate2<=4.4.0'`.

---

## CLI reference

```text
stark-translate operator [--port N] [--no-browser]   launch FastAPI + open UI
stark-translate setup [--models-dir PATH] [--refresh] download models from lockfile
stark-translate doctor [--json]                       run preflight checks
stark-translate version                               print version
```

### `setup` behavior

- Reads `models.lock.json` from the installed package or repo root.
- Default cache: `~/.cache/stark-translate/models` (Linux/Mac),
  `%LOCALAPPDATA%\stark-translate\models` (Windows). Override with
  `STARK_MODELS_DIR` or `--models-dir`.
- Each downloaded entry gets a `.installed` sidecar JSON with the
  lockfile version and SHA-256. Re-runs skip already-installed entries
  without re-hashing.
- Direct downloads (GGUFs) resume from `.partial` sidecars on flaky
  connections — the entrypoint honors HTTP Range.
- HF snapshots (Whisper, MarianMT, Piper voices) use
  `huggingface_hub.snapshot_download`.

---

## Building from source

```bash
git clone https://github.com/wrbell/stark-translate.git
cd stark-translate
uv venv && source .venv/bin/activate
uv pip install -e '.[cuda,dev]'
pytest tests/
```

The wheel build uses `hatchling` and force-includes top-level files
(`settings.py`, `dry_run_ab.py`, `models.lock.json`, `displays/`, the shell
helpers). `python -m build` produces both wheel and sdist.

---

## CI / Trusted Publishing

`.github/workflows/pypi.yml` publishes to PyPI on `v*` tags via OIDC
Trusted Publishing — no API tokens. PyPI must have the
`wrbell/stark-translate` repo + `pypi.yml` workflow trusted for the
`stark-translate` project at <https://pypi.org/manage/account/publishing/>.

For a dry-run build without publishing: trigger `workflow_dispatch` with
`dry_run: true`.

---

## See also

- [`linux-docker.md`](./linux-docker.md) — `docker compose up` path
- [`models.md`](./models.md) — `models.lock.json` schema
- [`windows.md`](./windows.md) — MSI for non-technical Windows users
- [`macos.md`](./macos.md) — native `.app` plan (deferred to v2026.8)
