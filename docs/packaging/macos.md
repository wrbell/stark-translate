# macOS installation and readiness

The supported Mac path is Python on Apple Silicon, with Parakeet for English STT,
Whisper Turbo for Spanish, Gemma 4 E4B OptiQ for finals and Marian for partials.
A signed `.app` remains deferred. The PyPI release needs its trusted publisher
configured; use a checkout, complete Mac ZIP, or verified wheel until publication succeeds.

The current [validation packet](../evaluation/mac_followup_20260910/final-760e948/README.md)
binds source `760e948`, its completed software checks and fresh installed artifacts.
The [installed-delivery record](../evaluation/mac_followup_20260910/final-760e948/installed-delivery.md)
records isolated Standard/Lite installs, operator checks, EN/ES file smokes and
actual dependency findings. Both full-service technical gates passed;
public raw evidence is indexed in that packet; consult [implementation status](../mac_implementation_status.md)
for current service and merge decisions. The earlier
[c13 delivery record](../evaluation/mac_followup_20260910/final-c13f51f/installed-delivery.md)
preserves its failed monitor and blank-preview checks. A completed pipeline alone
does not establish successful monitoring or a certified service.

## Install from a checkout or Mac ZIP

Use Python 3.11 or newer and install ffmpeg/PortAudio with your system package manager.
Create an isolated environment; an existing `stt_env` is kept unmodified as the rollback environment.

```bash
python3.11 -m venv venv
venv/bin/python -m pip install --upgrade 'pip>=26.2' 'setuptools>=83.0.0'
venv/bin/python -m pip install -c constraints/macos-arm64-py311-runtime.txt '.[mlx]'
venv/bin/python -m operator_app.cli setup --backend mlx
venv/bin/python -m operator_app.cli doctor --backend mlx --lang en
venv/bin/python -m operator_app.cli doctor --backend mlx --lang es
printf '%s\n' "$PWD/venv/bin/python" > .stark-python   # launcher pointer; rollback: point it at stt_env/bin/python
./run_operator.sh
```

`.[mlx]` includes the live runtime: PyTorch/Silero, ONNX Runtime, CT2, Marian
SentencePiece support, Parakeet, Whisper, MLX/OptiQ, and Piper. Optional extras:
`.[mlx,diarization]` adds SpeechBrain/ECAPA dependencies; `.[mlx,eval]` adds WER
and translation scoring. The Mac runtime pins Torch 2.13.0 / TorchAudio 2.11.0,
[audited 2026-09-11](../evaluation/overnight_20260911/L6-torch213-candidate/README.md).
The [constraints file](../../constraints/macos-arm64-py311-runtime.txt) carries the
six transitive audit fixes. Mac installs also constrain MLX/OptiQ/Parakeet;
upgrading the pinned lines requires another replay gate. Diarization is still opt-in and its model/latency gate
must pass before use at a service.

The [earlier installed-dependency audit](../evaluation/overnight_security/README.md)
records two unresolved findings in the pinned Mac Torch line and the incompatible
audio-wheel upgrade attempt. The Lite runtime has a separate clean audit; neither
that result nor filtered CI certifies the full Mac dependency set. A separate
[full-application dependency candidate](../evaluation/mac_followup_20260910/torch-full-application-candidate.md)
passed installed EN/ES inference and a zero-known-finding audit of 123 third-party
distributions. On 2026-09-11 its constraints became the production Mac bounds; the working
`stt_env` is retained unmodified as the rollback environment.

`bootstrap.sh --skip-systemd` performs the dependency install, backend-specific
model setup and CLI preflight without requiring an already-running web server.
No macOS login service is installed unless `--install-launchd` is supplied.

## Models and offline readiness

`setup --backend mlx` fetches only Mac defaults, including both language directions.
Use `--include e2b`, `--include tts`, or `--include translategemma` for optional
model groups. E2B is an opt-in final model; the [fixed-reference comparison](../evaluation/mac_followup_20260910/translation-comparison.md)
found faster isolated translation with a quality tradeoff. That result does not
establish faster caption delivery or replace the E4B default.
Piper setup downloads only the EN/ES voice and configuration files. Standard setup
also accepts optional `--include whisper-fallback`, `--include diarization`
(SpeechBrain model), and `--include diarization-pyannote`. Optional runtime packages
and any gated model access are still required; [current access/validation limits](../evaluation/mac_followup_20260910/live-hf-pinning.md)
remain pending. These groups are not included in ordinary setup, and Lite retains
its `--include tts`-only policy.

Both setup and inference use `engines.model_paths`: explicit existing model paths,
then `--models-dir` / `STARK_MODELS_DIR`, project `models/`, then the Hugging Face
cache. Setup-installed models are therefore usable by the inference engines.
Existing pinned HF snapshots are reused. Setup rejects selected HF entries without
a full 40-character revision before changing model caches. MLX inference and
summary loaders use a separate pinned acquisition resolver; doctor/preflight
lookup remains offline. `STARK_MODELS_DIR` must also be set when
launching if setup used a custom directory. Snapshot markers include revision and
manifest version, so changing the manifest cannot silently reuse an old revision.

`doctor` and operator preflight inspect local files/dependency versions without
loading Metal models. Missing required models or packages block Start; MLX does
not require GGUF files or a running llama-server. Preflight is a readiness check,
not a substitute for live caption, microphone-permission, and physical-output tests.

Mac setup includes derived int8 Marian CT2 models for both directions. It reuses
complete existing `adapters/marian_ct2/<direction>/active` directories first,
without modifying them. If neither adapter nor a valid managed artifact exists,
setup converts the pinned HF source using the same interpreter, validates a CPU
smoke and atomically publishes the result under the selected model cache. Mac
preflight requires the selected direction's complete CT2 artifact. The HF engine
remains available for explicit/manual fallback, but does not satisfy this default
Mac readiness gate. See [models.md](./models.md#derived-marian-ct2-models-on-mac).

The manual converter remains available for custom adapter work:
`scripts/convert_marian_ct2.py --quantization int8` with the existing `--model-id`
and `--output` flags. Setup never runs it against your working adapter directory.

## Interpreter selection and launchd

`run_operator.sh` and bootstrap select, in order: `STARK_PYTHON`, explicit `VENV`,
the repo `.stark-python` pointer, activated `VIRTUAL_ENV`, `CONDA_PREFIX`, repo
`stt_env`, repo `venv`, then `python3`. The pointer ranks above auto-activated Conda
base. Explicit overrides ignore a present pointer with a notice on stderr.

The pointer uses the first nonblank, noncomment line, trims whitespace and CRLF,
expands a leading `~/`, and resolves relative paths against the project root.
An absent pointer preserves fallback selection; an empty pointer or one naming a
missing, non-file or non-executable interpreter fails loudly. Roll back in one line:

```bash
printf '%s\n' "$PWD/stt_env/bin/python" > .stark-python
```

An invalid selected interpreter fails instead of silently using another environment.
The launcher delegates to the installed operator CLI, which starts uvicorn; the
pipeline child uses that same selected environment.

Generate and inspect the actual login-agent configuration first:

```bash
venv/bin/python -m operator_app.cli launchd render --project-root "$PWD"
venv/bin/python -m operator_app.cli launchd install --project-root "$PWD"
venv/bin/python -m operator_app.cli launchd uninstall
```

The plist records the current venv interpreter, working directory, model cache
override, log paths, and the resolved profile. Pass `--profile lite-cpu` when
rendering/installing a Lite service (the `stark-translate-lite` alias retains its
Lite default). The generated command preserves that choice at the next login.
It binds localhost by default. These commands explicitly
install/remove a user login service; a plain setup or operator launch never does so.
The checked-in plist is a reference, not an installable hard-coded user configuration.

## Package checks

Build wheel and sdist with `python -m build`; verify both with
`python tools/release_artifacts.py verify dist/*.whl dist/*.tar.gz`.
Build a complete source ZIP with `python tools/release_artifacts.py mac`.
Release workflows reject tag/project/Briefcase version mismatches and include
CLI, operator UI, tools, workers, displays, model manifest and launch scripts.
Source distributions and Mac ZIPs also include the current documentation and tests.
Generated roundtrip audio/text directories are excluded explicitly, including
when a wheel is built from an unpacked source ZIP without Git ignore metadata.

For a Mac installation check, use a separate virtual environment and install the
built wheel with `[mlx,eval,diarization]` extras. Install `httpx>=0.28` in this
smoke environment for the ASGI test client, then run `python -m pip check`.
From a directory outside the checkout, run `python -m tools.installed_smoke` using
that environment's interpreter. The smoke checks installed package provenance,
the health and operator routes, the review JavaScript, and required runtime files.
Import the selected runtime dependencies with `HF_HUB_OFFLINE=1` to verify native
library loading before a separate model/inference check. Rebuild and reinstall
the wheel after the final source/version changes so the check covers the release artifact.
Also unpack the Mac ZIP, build its wheel offline, compare its contents with the
sdist-built wheel, and run the outside-checkout smoke after installing each.
