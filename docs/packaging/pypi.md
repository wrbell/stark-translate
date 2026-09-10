# Python package installation

Use a checkout, Mac source ZIP, or locally built wheel while publication of the
current version is pending. The Mac installation and native dependency steps are
in [macos.md](./macos.md).

```bash
python3.11 -m venv venv
venv/bin/python -m pip install '.[mlx]'
venv/bin/stark-translate setup --backend mlx
venv/bin/stark-translate doctor --backend mlx --lang en
venv/bin/stark-translate doctor --backend mlx --lang es
venv/bin/stark-translate operator
```

To install a built wheel, replace `.[mlx]` with the wheel path followed by
`[mlx]`. Once the matching version is published, `uv tool install 'stark-translate[mlx]'`
provides an isolated package environment. Download size
and installation time depend on the selected dependencies and local caches.

## Extras

| Extra | Runtime |
|---|---|
| `mlx` | Apple Silicon: pinned minor lines for MLX/MLX-LM/OptiQ/Parakeet/PyTorch, Whisper, CT2, Marian tokenizers, Silero, ONNX and Piper |
| `cuda` | NVIDIA: PyTorch, Transformers, faster-whisper/CT2, Silero, Marian tokenizers and Piper |
| `cpu` | CPU: PyTorch, Transformers, faster-whisper/CT2, Silero, Marian tokenizers and Piper |
| `diarization` | Optional SpeechBrain, matching torchaudio, soundfile and scikit-learn |
| `eval` | Optional SacreBLEU/chrF and WER scoring |
| `dev` | Lint, test and package-build tools |

For example, `.[mlx,eval,diarization]` installs all Mac validation dependencies.
The optional diarization runtime still requires its model and service-readiness
gate. See `pyproject.toml` for exact dependency constraints; changing CUDA/CT2
versions requires a matching supported CUDA runtime.

## CLI

```text
stark-translate operator [--port N] [--no-browser]
stark-translate setup [--backend auto|mlx|cuda|cpu] [--models-dir PATH] [--refresh]
                     [--include e2b tts translategemma]
stark-translate doctor [--backend auto|mlx|cuda|cpu] [--lang en|es] [--json]
stark-translate launchd render|install|uninstall
stark-translate version
```

`doctor` runs local preflight checks directly; an operator server need not be
running. Its selected backend/language/model/features determine the required
dependencies and local model files. Use `doctor --help` for optional TTS,
diarization and model-profile checks.

`setup` defaults to backend auto-detection and selects that backend's default
model entries. Optional profiles are added explicitly with `--include`. Both
language directions are included for Mac. [models.md](./models.md) describes
cache lookup, pinned revisions, sidecars and resumable downloads.

## Build and verify

Build a source distribution, then build the wheel from it, so missing source
files are caught before publishing. `tools/release_artifacts.py verify` checks
runtime files in wheels and required documentation/tests in source bundles.
`tools/installed_smoke.py` checks a wheel installed outside the checkout; the
full Mac procedure is in [macos.md](./macos.md#package-checks).

`.github/workflows/pypi.yml` publishes on version tags through PyPI Trusted
Publishing. The publisher must match repository `wrbell/stark-translate`,
workflow `pypi.yml`, and GitHub environment `pypi`. Publication is a separate
step from a successful local build. `workflow_dispatch` with `dry_run: true`
builds and validates without publishing.

Release guards require the tag to match both project and Briefcase versions;
an existing local tag must point at the checkout being validated. The workflows
verify archives and exercise the installed operator before publication.
