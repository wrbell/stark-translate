# Mac inference and validation runbook

Updated September 10, 2026. This is the current Mac-only path; the
[August runbook](mac_pipeline_refresh_20260830.md) is historical and contains
superseded flags and benchmark labels. Use [current status](mac_implementation_status.md)
for actual validation and [the backlog](backlog.md) for remaining work.

## Install and prepare

Create an isolated environment; an existing `stt_env` is kept unmodified as the
rollback environment. The promoted runtime is selected by `.stark-python`.

```bash
python3.11 -m venv venv
venv/bin/python -m pip install --upgrade 'pip>=26.2' 'setuptools>=83.0.0'
venv/bin/python -m pip install -c constraints/macos-arm64-py311-runtime.txt '.[mlx]'
venv/bin/stark-translate setup --backend mlx
venv/bin/stark-translate doctor --backend mlx --lang en
venv/bin/stark-translate doctor --backend mlx --lang es
printf '%s\n' "$PWD/venv/bin/python" > .stark-python   # launcher pointer; rollback: point it at stt_env/bin/python
./run_operator.sh
```

Setup reuses complete local/cache artifacts and supplies pinned Parakeet English,
Whisper Turbo Spanish, E4B OptiQ and both Marian CT2 directions. `--include e2b tts`
adds optional E2B and voices. A custom `--models-dir` must also be provided through
`STARK_MODELS_DIR` when launching. See [installation](packaging/macos.md).

## Operate and rehearse

Follow the [operator runbook](operator_runbook.md): Prepare → Start → Live,
Pause/Resume, Switch language, Stop, then Sessions/Review. Model loading and input
readiness are distinct. A stalled microphone must surface as an error; it must not
be mistaken for a quiet successful session. Real microphone and physical-output
checks are deferred to the next attended session.

The readiness script is an additional control check. Its explicit `STARK_PYTHON`
selection remains supported; the operator launcher also accepts `.stark-python`:

```bash
STARK_PYTHON="$PWD/venv/bin/python" ./scripts/dry_run_rehearsal.sh
```

It starts and switches real sessions only when the operator is idle. It requires
fresh audio readiness, distinct session identities and completed Stop. It does not
certify translation quality, audience rendering, a full hymn or physical devices.

## Measure EN↔ES

Standard defaults are Parakeet EN / Whisper ES → Marian previews → E4B finals.
For a controlled real-time replay:

```bash
venv/bin/python dry_run_ab.py --backend mlx --lang en --no-ab --no-tts \
  --audio-file /absolute/clip.wav --session-id unique_replay_en \
  --replay-speed 1 --replay-wait-client-seconds 15
```

Open the audience display at `http://localhost:8080/displays/audience_display.html`
so its WebSocket can connect before capture begins. Repeat separately with
`--lang es` and Spanish audio. The client wait gates connection, not visibility;
visible render ACKs still need to be verified. Use a unique session ID each run.

Compare `--gemma4-size e2b` in a separate process on identical inputs. Do not run
multiple GPU benchmarks together. Freeze source, dependency/model versions,
settings and audio hashes; alternate model order and retain failed attempts.
[Evaluation instructions](evaluation/README.md) and the
[overnight experiment plan](evaluation/overnight_experiment_plan.md) define gates.

Schema 2 measures estimated last speech frame → final payload readiness. Browser
receipt → render and speech end → ACK upper bound are separate; return-network time
is included in the latter. Legacy processing timings, hidden tabs and accelerated
replays cannot prove sub-second caption delivery. E4B remains the default.

## Review and hand off

Review supports live finalized chunks and completed sessions, revision-checked
corrections and persistent drafts. Human STT and bilingual approvals are independent.
Export completed approved data with audio/provenance; preserve training/evaluation
separation. See [Review contracts](current_architecture.md) and the operator runbook.

Lite CPU and optional CPU E2B work on a separate Torch-free runtime. RTX2070 and
native Windows execution remain hardware gates: [Lite profiles](lite_profiles.md).

WSL preprocessing, W17/domain training, adapter conversion and CUDA experiments
belong to the [WSL runbook](wsl_pipeline_refresh.md). They have not run on this Mac.
`--mts` is explicitly unavailable live; prior negative research stays archived.
Hindi/Chinese expansion is a separate R&D decision, not part of EN↔ES latency work.
