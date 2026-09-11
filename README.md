# stark-translate

> **Mac EN↔ES follow-up:** [PR #196](https://github.com/wrbell/stark-translate/pull/196) records this work.
> [EN↔ES evidence](docs/evaluation/mac_followup_20260910/README.md) records completed
> screens with no qualified arms, CPU Whisper-base rejection and silent hymn diagnostics.
> Defaults are unchanged. Current artifact, service and merge results are recorded in
> [implementation status](docs/mac_implementation_status.md).


[![Lint](https://github.com/wrbell/stark-translate/actions/workflows/lint.yml/badge.svg)](https://github.com/wrbell/stark-translate/actions/workflows/lint.yml)
[![Test](https://github.com/wrbell/stark-translate/actions/workflows/test.yml/badge.svg)](https://github.com/wrbell/stark-translate/actions/workflows/test.yml)
[![Security](https://github.com/wrbell/stark-translate/actions/workflows/security.yml/badge.svg)](https://github.com/wrbell/stark-translate/actions/workflows/security.yml)
[![codecov](https://codecov.io/gh/wrbell/stark-translate/graph/badge.svg)](https://codecov.io/gh/wrbell/stark-translate)

Fully on-device, live bilingual speech-to-text for church outreach at Stark Road Gospel Hall (Farmington Hills, MI). English/Spanish, real-time mic input, browser display. Inference uses no cloud APIs and works offline after the selected models are prepared locally.

> **Source and releases (2026-09-10):** this guide describes v2026.14 source
> (`2026.14.0.0`), with prior integration in [PR #192](https://github.com/wrbell/stark-translate/pull/192)
> and the EN↔ES follow-up in [PR #196](https://github.com/wrbell/stark-translate/pull/196).
> The last published release is **v2026.14.0.0** (tagged 2026-09-11; GitHub Release with Mac/NVIDIA/Windows
> ZIPs and the MSI). Source integration, release publication and service certification are separate;
> PyPI publication is deferred by decision (2026-09-11); the publish job stays disabled until the repository variable `PYPI_PUBLISH_ENABLED` is set.
> Contracts: [`docs/current_architecture.md`](docs/current_architecture.md) · evidence:
> [`docs/mac_implementation_status.md`](docs/mac_implementation_status.md) · remaining work:
> [`docs/backlog.json`](docs/backlog.json) (rendered as [`docs/backlog.md`](docs/backlog.md)).

The September 10 normalized follow-up completed [Standard screening](docs/evaluation/mac_followup_20260910/standard-screen-result.md)
(96 technical passes, 0/24 qualified arms), [Spanish Parakeet screening](docs/evaluation/mac_followup_20260910/spanish-parakeet-result.md)
(18 passes, 0/2) and [CPU Lite cadence screening](docs/evaluation/mac_followup_20260910/lite-cadence-result.md)
(24 passes, 0/4). Each run has six eligible anchors, so these screens support no
p95 claim. Faster medians did not satisfy the other guards. E4B finals, Spanish
Whisper and the 0.6-second cadence remain unchanged. Lite finals in this screen
use Marian CPU; E2B is only a harness label. The separate
[CPU STT comparison](docs/evaluation/mac_followup_20260910/cpu-stt-comparison.md)
completed 600 calls and rejected Whisper-base for higher WER in both languages.
The [independent Lite deadline screen](docs/evaluation/mac_followup_20260910/lite-deadline-result.md)
also completed 24 runs with 0/4 qualified arms at unchanged 0.6-second cadence.
Rejected arms cannot enter confirmation or combinations.

The [earlier September 10 overnight screen](docs/evaluation/overnight_screen_20260910/README.md) recorded 96/96 valid runs and selected 0/28 experiment/model arms. The sub-second final-delivery goal was not met on this 45-second English cohort; E4B defaults remain unchanged. Small endpoint samples, unreviewed references and the locked-native-screen/browser-DOM distinction limit this evidence.

Current [source validation](docs/evaluation/mac_followup_20260910/final-760e948/source-validation.md)
records the repaired `760e948` source. Its [delivery packet](docs/evaluation/mac_followup_20260910/final-760e948/README.md)
keeps installed-artifact and full-service acceptance separate.

The earlier c13 [source checks](docs/evaluation/mac_followup_20260910/final-c13f51f/source-validation.md),
[350-second hymn control](docs/evaluation/mac_followup_20260910/final-c13f51f/hymn-capture.md)
and [102-call text comparison](docs/evaluation/mac_followup_20260910/final-c13f51f/hymn-boundary.md)
are separate evidence. The natural control never entered music hold; supplied text
boundaries remain hypotheses. #193/#194 still require natural labels and bilingual
review. [Installed delivery](docs/evaluation/mac_followup_20260910/final-c13f51f/installed-delivery.md)
keeps artifact, pipeline, monitor and full-service acceptance separate.

EN↔ES remains the active speed priority. The installed Standard and CPU Lite
full-service runs on `752ab9a` completed with consistent retained source spans,
required writes and process cleanup. Lite was functional on this Mac, but sparse
previews and large observed tails do not support recommending it as a fast
production profile today. See the [separate endurance cohorts](docs/evaluation/overnight_endurance_20260910/README.md);
these observational runs do not establish a causal speed comparison, human quality,
physical display visibility or a default promotion.

## Architecture

```
                              Two-Pass Pipeline
                              ================

  Mic (48kHz) ──> Resample 16kHz ──> Silero VAD ──┐
                                                    │
            ┌───────────────────────────────────────┘
            │
            ├─ PARTIAL (every 0.6s of new speech, while speaker is talking)
            │    Mac: Parakeet EN / Whisper turbo ES · Marian CT2 CPU
            │    CUDA: W16 Whisper CT2 · Marian CT2                ← italic in UI
            │
            └─ FINAL (on 0.5s silence gap or 8s max utterance)
                 Same STT · Gemma 4 E4B (MLX OptiQ on Mac / llama.cpp on CUDA)
                 ← replaces partial
                 ├─ Piper TTS (--tts, optional)                    ← audio output
                 Gemma 4 E2B (--gemma4-size e2b / low-VRAM CUDA)    ← opt-in

                 Pipeline overlap: translation runs on utterance N
                 while STT runs on utterance N+1.
                                     │
                                     ▼
                  WebSocket :8765 · HTTP :8080 · Operator API :9000
                                     │
              ┌──────────┬───────────┼───────────┬───────────┐
              ▼          ▼           ▼           ▼           ▼
          Audience    A/B/C       Mobile     Operator      CSV +
          Display    Compare     Display       SPA       Diagnostics
         (projector) (operator)  (QR code)  (/operator/)   (JSONL)
```

Runs on Apple Silicon (MLX) or NVIDIA GPUs (CUDA) via `--backend auto|mlx|cuda`. Measured
latency and accuracy live with their definitions in
[`docs/evaluation/README.md`](docs/evaluation/README.md) and the dated archives
([CUDA STT](docs/archive/v2026.7/STT_BENCHMARK.md), [Marian CT2](docs/archive/v2026.8/MARIAN_BENCHMARK.md),
[Mac latency](docs/archive/v2026.13/MAC_LATENCY.md)); sub-second speech-end-to-caption is the
goal and is not yet achieved. Do not extrapolate these numbers to other GPUs.

## Quick Start

Run these commands from a checkout or extracted Mac source ZIP. Package publication
is pending; a locally built wheel or source install provides the current code.
See [Mac installation and readiness](docs/packaging/macos.md) for full setup and
optional evaluation/diarization dependencies.

```bash
# Mac (Apple Silicon)
brew install ffmpeg portaudio
python3.11 -m venv venv
venv/bin/python -m pip install --upgrade 'pip>=26.2' 'setuptools>=83.0.0'
venv/bin/python -m pip install -c constraints/macos-arm64-py311-runtime.txt '.[mlx]'
venv/bin/stark-translate setup --backend mlx    # Mac defaults, both languages; --include e2b tts translategemma
venv/bin/stark-translate doctor --backend mlx --lang en
printf '%s\n' "$PWD/venv/bin/python" > .stark-python   # launcher pointer; rollback: point it at stt_env/bin/python
./run_operator.sh              # Open /operator/ on port 9000

# NVIDIA (Linux)
python3.11 -m venv venv
venv/bin/python -m pip install --upgrade 'pip>=26.2' 'setuptools>=83.0.0'
venv/bin/python -m pip install '.[cuda]'
venv/bin/stark-translate setup --backend cuda
venv/bin/stark-translate operator

# Lite (CPU-only; Torch-free runtime, separate venv)
python3.11 -m venv .venv-lite
.venv-lite/bin/python -m pip install --upgrade 'pip>=26.2' 'setuptools>=83.0.0'
.venv-lite/bin/python -m pip install '.[lite-cpu,tts]'
python3.11 -m venv .venv-lite-build
.venv-lite-build/bin/python -m pip install --upgrade 'pip>=26.2' 'setuptools>=83.0.0'
.venv-lite-build/bin/python -m pip install '.[lite-build]'
.venv-lite/bin/stark-translate-lite setup --models-dir /absolute/lite-models \
  --converter-python /absolute/.venv-lite-build/bin/python --include tts
STARK_MODELS_DIR=/absolute/lite-models .venv-lite/bin/stark-translate-lite doctor --json
STARK_MODELS_DIR=/absolute/lite-models .venv-lite/bin/stark-translate-lite operator
```

`stark-translate-lite` defaults to the `lite-cpu` profile (Whisper small CT2 int8 + Marian
CT2 finals); `--profile lite-cpu-quality` / `lite-cuda-8gb` add Gemma 4 E2B through a
session-owned `llama-server`. Profiles never auto-upgrade; contract and evidence in
[`docs/lite_profiles.md`](docs/lite_profiles.md), including the complete RTX 2070
installation recipe with the same profile selected for setup, doctor and operator.

`setup` downloads the pinned models from `models.lock.json` and builds the Marian CT2
int8 artifacts; existing `adapters/marian_ct2/<dir>/active` directories are reused
unchanged (manual conversion: `scripts/convert_marian_ct2.py`). The legacy
`requirements-mac.txt` / `requirements-nvidia.txt` path is deprecated for inference;
`requirements-windows.txt` remains the WSL training environment
([`CLAUDE-windows.md`](./CLAUDE-windows.md)).

Key `dry_run_ab.py` flags: `--lang es` (Spanish speaker mode), `--tts` (audio output),
`--audio-file clip.wav` (replay), `--dry-run-text "test"` (no mic), `--gemma4-size e2b`,
`--model-family translategemma [--ab]`, `--vad-threshold 0.3`, `--log-level DEBUG`.

## Models

Pinned in `models.lock.json`; resolution order (explicit path → `STARK_MODELS_DIR` →
`models/` → Hugging Face cache) is shared by setup, preflight and inference.

| Role | Mac (MLX) default | CUDA default | Notes |
|------|-------------------|--------------|-------|
| VAD | Silero 6.2.1 (torch; ONNX opt-in) | same | 0.5 s silence trigger, 8 s max utterance |
| STT EN | Parakeet TDT 0.6B v3 (`parakeet-mlx`) | Whisper large-v3-turbo + W16 LoRA → CT2 int8_float16 | Mac `--stt-backend mlx` forces Whisper |
| STT ES | mlx-whisper large-v3-turbo | same CT2 model | English-only Distil fallback is rejected for Spanish; confidence thresholds in `settings.py` |
| Partial translation | Marian opus-mt en-es / es-en → CT2 int8 on CPU (HF fallback) | Marian CT2 int8_float16 on GPU | [Marian benchmark](docs/archive/v2026.8/MARIAN_BENCHMARK.md) |
| Final translation | Gemma 4 E4B OptiQ 4-bit (`--gemma4-size e2b` opt-in) | Gemma 4 E4B Q4_K_M via llama.cpp (`start_server.sh`), E2B for low VRAM | [CUDA benchmark](docs/archive/v2026.5/BENCHMARK.md); HF NF4 is legacy |
| Opt-out translation | TranslateGemma 4B / 12B 4-bit (`--model-family translategemma`, `--ab`) | — | Historical default; see [`docs/mlx_cuda_parity.md`](./docs/mlx_cuda_parity.md) |
| TTS (off) | Piper `en_US-lessac-high` / `es_MX-claude-high` (ONNX) | same | `--tts --tts-output ws|wav|both|local` |

## Displays

Browser displays served over LAN; phones connect via QR code on the audience display.
Protocol and timing semantics: [`displays/CLAUDE.md`](./displays/CLAUDE.md).

| Display | Purpose |
|---------|---------|
| `displays/operator/` | Operator SPA on `:9000/operator/`: start/stop, preflight, devices, verses, summary, Review/export |
| `audience_display.html` | Projector: EN/ES side-by-side, fading context, fullscreen, QR overlay |
| `ab_display.html` | Operator: A (default) / MarianMT / B comparison with latency stats |
| `mobile_display.html` | Phone/tablet: responsive, model toggle, Spanish-only mode |
| `church_display.html` | Simplified church layout |
| `obs_overlay.html` | Transparent overlay for OBS Studio streaming |

## Training

Fine-tuning runs on Windows/WSL (A2000 Ada 16 GB); exported artifacts transfer to the Mac.
Guide: [`training/CLAUDE.md`](./training/CLAUDE.md).

- **STT (Whisper LoRA):** W16 is the deployed CUDA adapter ([bench](docs/archive/v2026.7/STT_BENCHMARK.md)); W17 DoRA + hard-mix is scripted for the next WSL cycle. Labels come from the Deepgram Nova-3 oracle with a tiered theological glossary.
- **Translation (Gemma 4):** E2B/E4B QLoRA SFT and CPO program in [`docs/gemma4_tuning/`](docs/gemma4_tuning/overview.md); adapters so far reach parity with stock E4B and the Jacobo canary still fails (#136), so stock E4B remains the default. The earlier TranslateGemma S1–S9 sweep is historical ([`docs/archive/training/`](docs/archive/training/benchmark_training.md)).
- **Data:** ~265K verse pairs (`verse_pairs_train_v2.jsonl`, after the [Platense alignment fix](docs/platense_alignment_bug.md)), sermon pairs, SHA-256 data lockfile, stratified eval sets.

## Hardware

| Target | Config | Status |
|--------|--------|--------|
| MacBook Pro M3 Pro 18 GB | Parakeet/Whisper + Marian CT2 + Gemma 4 E4B OptiQ | Dated functional/replay evidence in [Mac evaluation](docs/evaluation/mac_followup_20260910/README.md); microphone reliability and human/device gates remain pending |
| Smaller-memory Apple Silicon | `--gemma4-size e2b` | Intended path, not validated |
| NVIDIA A2000 Ada 16 GB (WSL) | W16 Whisper CT2 + Marian CT2 + Gemma 4 E4B Q4_K_M | Benchmarked v2026.5–8 (archives above) |
| NVIDIA 6–8 GB | Gemma 4 E2B / E4B Q4_K_M via llama.cpp | Per [`docs/archive/v2026.5/BENCHMARK.md`](./docs/archive/v2026.5/BENCHMARK.md) VRAM figures; not separately certified |
| Lite CPU (x86 or Apple, ≥ 4 cores / 8 GiB) | `stark-translate-lite` → `lite-cpu` (Whisper small CT2 int8 + Marian CT2 finals), `lite-cpu-quality` adds Gemma 4 E2B via CPU llama.cpp | Implemented; isolated Mac CPU synthetic EN+ES caption/TTS smoke passed ([`docs/lite_profiles.md`](docs/lite_profiles.md)); x86 CPU performance and quality not measured |
| RTX 2070 8 GB, native Windows | `lite-cuda-8gb` (Whisper turbo CT2 int8_float16 + Marian + Gemma 4 E2B via CUDA llama.cpp) | Implemented; pinned E2B GGUF / native runtime setup verified on the Mac only; **nothing run on a 2070 yet**; MSI is a scaffold plan |

> **CUDA HF NF4 not recommended:** Gemma 4 HF NF4 keeps bf16 Per-Layer Embeddings resident;
> use llama.cpp Q4_K_M. Details in the v2026.5 benchmark.

## Testing & CI

```bash
pytest tests/ -v
ruff check . && ruff format --check .
mypy engines/ settings.py
python tools/render_backlog.py validate
python tools/render_backlog.py render --check
pytest tests/test_documentation.py -v
```

10 GitHub Actions workflow files in `.github/workflows/`: Lint, Test (3.11 + 3.12, coverage
gate in `test.yml`), Security (pip-audit), Release, Windows MSI Release, PyPI Publish
(tag/manual-triggered build; publishing deferred by decision), Docker Image (GHCR), Label PRs, Commitlint, Stale. CalVer in
`pyproject.toml`. Validated CPU suite counts live only in
[`docs/mac_implementation_status.md`](docs/mac_implementation_status.md).

## Project Structure

```
dry_run_ab.py                  Main pipeline: mic → VAD → STT → translate → display
settings.py                    Unified config (pydantic-settings, STARK_ prefix)
models.lock.json               Pinned model sources consumed by `stark-translate setup`

operator_app/                  FastAPI control plane (:9000), setup/doctor CLI, preflight, review,
                               support bundles, idle-only audio tests, lite preflight, owned-process cleanup
stark_translate/               Package entry; profiles.py (standard / lite-cpu / lite-cpu-quality / lite-cuda-8gb)
engines/                       STT + translation + TTS engine layer (see engines/CLAUDE.md)
  base.py                      ABCs and result dataclasses
  mlx_engine.py                Apple Silicon MLX Whisper + Gemma
  parakeet_mlx_engine.py       Parakeet TDT EN (Mac default)
  marian_hf_engine.py          Marian HF fallback; MarianCT2Engine lives in cuda_engine.py
  llamacpp_engine.py           CUDA via llama.cpp HTTP (production CUDA path)
  cuda_engine.py               CUDA HF engines (legacy) + Marian CT2 / faster-whisper CT2
  factory.py                   Backend detection and engine construction
start_server.sh                Launch llama-server with the default Gemma 4 GGUF

displays/                      Static browser displays + operator SPA (displays/operator/)

training/                      Windows/WSL training scripts (see training/CLAUDE.md)
  train_whisper.py             Whisper LoRA/DoRA (curriculum, --init-from)
  train_gemma4.py              Gemma 4 QLoRA SFT (used for v1/v1.1; production recipe not yet run)
  train_gemma4_cpo.py          Gemma 4 CPO preference optimization (v2-cpo)
  export_ct2.py / export_gguf.py   Whisper → CT2, Gemma → GGUF
  align_deepgram_chunks.py     Deepgram-Whisper alignment (sharded Arrow)
  mine_hard_examples.py        Hard example mining

tools/                         Evaluation, monitoring, review tooling (see tools/CLAUDE.md)
  mac_evaluation.py            Frozen Mac evaluation pipeline
  pipeline_timing.py           Schema 2 timing records
  live_caption_monitor.py      YouTube caption comparison
  translation_qe.py            3-tier translation quality estimation
  manage_adapters.py           Adapter lifecycle (register, activate, rollback)
  health_check.py              Theological canary adapter verification (8 of 18 by default)
  isolated_audio.py            PortAudio in a disposable child; no-input timeouts (mic-stall fix)
  pipeline_health.py           Low-rate health/control channel read by the operator
  llama_runtime.py             Pinned native llama.cpp install + session-owned llama-server (Lite)
  offline_hindi.py             Offline church-audio Hindi baseline (evaluation only, #138)
  render_backlog.py            Backlog validation/rendering/link check

features/                      Diarization, verse extraction, summary (wired through operator_app)
docs/                          Architecture, evaluation, backlog, dated archives
```

## Documentation

| Doc | Contents |
|-----|----------|
| [`CLAUDE.md`](./CLAUDE.md) / [`AGENTS.md`](./AGENTS.md) | Project overview and navigation (human + agent guides, same content) |
| [`docs/current_architecture.md`](./docs/current_architecture.md) | Current inference/operator contracts (v2026.14 candidate) |
| [`docs/backlog.json`](./docs/backlog.json) | Machine-readable remaining tasks (render: `tools/render_backlog.py`) |
| [`docs/mac_implementation_status.md`](./docs/mac_implementation_status.md) | Local validation evidence and open gates |
| [`docs/overnight_status.md`](./docs/overnight_status.md) | Overnight documentation deliverables and unfinished areas |
| [`docs/lite_profiles.md`](./docs/lite_profiles.md) | Lite profiles contract, pinned artifacts, CPU smoke evidence |
| [`CLAUDE-macbook.md`](./CLAUDE-macbook.md) | Mac inference environment |
| [`CLAUDE-windows.md`](./CLAUDE-windows.md) | Windows: WSL training (Part A) and native Lite inference (Part B) |
| [`engines/`](./engines/CLAUDE.md) | Engine layer — paired `AGENTS.md` in each subdirectory |
| [`training/`](./training/CLAUDE.md) | Fine-tuning and data pipeline |
| [`tools/`](./tools/CLAUDE.md) | Evaluation, QE, adapter deployment |
| [`displays/`](./displays/CLAUDE.md) | Display modes and WebSocket protocol |
| [`features/`](./features/CLAUDE.md) | Diarization, summary, verse extraction |
| [`docs/operator_runbook.md`](./docs/operator_runbook.md) | Day-of-event workflow for non-technical operators |
| [`docs/roadmap.md`](./docs/roadmap.md) | Current state, active work, archived history |

## Status

**Last published release recorded here (v2026.13):** bidirectional EN/ES inference, operator control plane,
Mac latency fixes (#180–191), Parakeet EN STT, Marian CT2 Mac path, replay harness,
TTS routing, live diarization code behind `--diarize`. See [`docs/archive/`](docs/archive/)
for version-specific benchmarks — legacy `e2e_latency_ms` is processing time, not
speech-end-to-display.

**v2026.14 source ([PR #192](https://github.com/wrbell/stark-translate/pull/192)):**
operator reliability (isolated capture, health channel, work lease, owned-process cleanup),
schema 2 timing, reproducible setup, Review/export, frozen screening, Lite profiles, opt-in
latency experiments, the lay-volunteer operator page and the offline Hindi baseline tool —
implemented in the integrated source; recorded evidence and remaining gates are in
[`docs/mac_implementation_status.md`](docs/mac_implementation_status.md) and
[`docs/lite_profiles.md`](docs/lite_profiles.md). PR #192 merged into main;
[closeout evidence](docs/evaluation/overnight_closeout_20260910/README.md) records the
merge and justified issue closures. Tag `v2026.14.0.0` and its GitHub Release/MSI were published on 2026-09-11; PyPI publication is deferred by decision (2026-09-11).

**Live microphone (2026-09-09 → 10):** the built-in-mic session stalled after model load
(no audio frames, operator showed RUNNING from the CSV header) while file replay passed. The
fix — PortAudio in a disposable child with a no-input timeout, and operator readiness from
the pipeline health channel — is implemented. Attended EN and ES microphone sessions now
receive real frames, reach ready and stop cleanly. These quiet-room runs did not establish
spoken caption quality. [Session evidence](docs/evaluation/attended_mic_20260910/README.md)
distinguishes them from visible-caption file replay. Later [synthetic EN/ES caption checks](docs/evaluation/tts_routing_20260910/README.md)
retain a failed Spanish capture-loss result.

The later [capture-loss accounting repair](docs/evaluation/mac_followup_20260910/capture-loss-accounting.md)
(`d7ed43d`) separates counted worker FIFO drops from native-overflow flags with an
unknown lost-sample count, validates bounded POSIX terminal receipts and records
shutdown-only losses. It does not certify native capture reliability or resolve
the retained Spanish failure; no native audio test was performed for this repair.

**Open gates:** spoken microphone EN/ES captions, church-specific Spanish references, bilingual review,
visible-browser timing certification, two-speaker diarization gate, physical second
output, WSL training cycle, Lite x86 CPU / RTX 2070 hardware performance — tracked
in [`docs/backlog.json`](docs/backlog.json). The laptop runbook rehearsal is complete
and #134 is closed. The original [per-language routing acceptance for #132](docs/evaluation/mac_followup_20260910/tts-routing-acceptance.md)
is met; closure follows reviewed merge, with physical output and audibility checked
separately. Hymn/quality follow-ups #193/#194 remain open.

## License

Private project. All Bible translation training data uses public domain or CC-licensed sources only.
