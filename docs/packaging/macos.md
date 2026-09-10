# macOS installation and readiness

The supported Mac path is Python on Apple Silicon, with Parakeet for English STT,
Whisper Turbo for Spanish, Gemma 4 E4B OptiQ for finals and Marian for partials.
A signed `.app` remains deferred. The PyPI release needs its trusted publisher
configured; use a checkout, complete Mac ZIP, or verified wheel until publication succeeds.

## Install from a checkout or Mac ZIP

Use Python 3.11 or newer and install ffmpeg/PortAudio with your system package manager.
Create an isolated environment; existing `stt_env` installations can continue unchanged.

```bash
python3.11 -m venv venv
venv/bin/python -m pip install '.[mlx]'
venv/bin/python -m operator_app.cli setup --backend mlx
venv/bin/python -m operator_app.cli doctor --backend mlx --lang en
venv/bin/python -m operator_app.cli doctor --backend mlx --lang es
VENV="$PWD/venv" ./run_operator.sh
```

`.[mlx]` includes the live runtime: PyTorch/Silero, ONNX Runtime, CT2, Marian
SentencePiece support, Parakeet, Whisper, MLX/OptiQ, and Piper. Optional extras:
`.[mlx,diarization]` adds SpeechBrain/ECAPA dependencies; `.[mlx,eval]` adds WER
and translation scoring. Mac installs constrain MLX/OptiQ/Parakeet and PyTorch
to the verified minor versions; upgrading those lines requires another replay gate. Diarization is still opt-in and its model/latency gate
must pass before use at a service.

`bootstrap.sh --skip-systemd` performs the dependency install, backend-specific
model setup and CLI preflight without requiring an already-running web server.
No macOS login service is installed unless `--install-launchd` is supplied.

## Models and offline readiness

`setup --backend mlx` fetches only Mac defaults, including both language directions.
Use `--include e2b`, `--include tts`, or `--include translategemma` for optional
profiles. E2B is a faster, separately evaluated profile; it does not replace E4B.
Piper setup downloads only the EN/ES voice and configuration files.

Both setup and inference use `engines.model_paths`: explicit existing model paths,
then `--models-dir` / `STARK_MODELS_DIR`, project `models/`, then the Hugging Face
cache. Setup-installed models are therefore usable by the inference engines.
Existing pinned HF snapshots are reused. `STARK_MODELS_DIR` must also be set when
launching if setup used a custom directory. Snapshot markers include revision and
manifest version, so changing the manifest cannot silently reuse an old revision.

`doctor` and operator preflight inspect local files/dependency versions without
loading Metal models. Missing required models or packages block Start; MLX does
not require GGUF files or a running llama-server. Preflight is a readiness check,
not a substitute for live caption, microphone-permission, and physical-output tests.

For CT2 partials, run `scripts/convert_marian_ct2.py --quantization int8` once per
direction. Until the converted adapters are present, the supported HF Marian
fallback works but is slower. The conversion is a separate operation from setup.

## Interpreter selection and launchd

`run_operator.sh` and bootstrap select `STARK_PYTHON` or explicit `VENV` first,
then activated `VIRTUAL_ENV`, then repo `stt_env`, then repo `venv`.
An invalid explicit override fails instead of silently using another interpreter.
The launcher runs `python -m uvicorn` so the pipeline child uses the same environment.

Generate and inspect the actual login-agent configuration first:

```bash
venv/bin/python -m operator_app.cli launchd render --project-root "$PWD"
venv/bin/python -m operator_app.cli launchd install --project-root "$PWD"
venv/bin/python -m operator_app.cli launchd uninstall
```

The plist records the current venv interpreter, working directory, model cache
override, and log paths. It binds localhost by default. These commands explicitly
install/remove a user login service; a plain setup or operator launch never does so.
The checked-in plist is a reference, not an installable hard-coded user configuration.

## Package checks

Build wheel and sdist with `python -m build`; verify both with
`python tools/release_artifacts.py verify dist/*.whl dist/*.tar.gz`.
Build a complete source ZIP with `python tools/release_artifacts.py mac`.
Release workflows reject tag/project/Briefcase version mismatches and include
CLI, operator UI, tools, workers, displays, model manifest and launch scripts.
Source distributions and Mac ZIPs also include the current documentation and tests.

For a Mac installation check, use a separate virtual environment and install the
built wheel with `[mlx,eval,diarization]` extras, then run `python -m pip check`.
From a directory outside the checkout, run `python -m tools.installed_smoke` using
that environment's interpreter. The smoke checks installed package provenance,
the health and operator routes, the review JavaScript, and required runtime files.
Import the selected runtime dependencies with `HF_HUB_OFFLINE=1` to verify native
library loading before a separate model/inference check. Rebuild and reinstall
the wheel after the final source/version changes so the check covers the release artifact.
