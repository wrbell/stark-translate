# stark-translate

[![Lint](https://github.com/wrbell/stark-translate/actions/workflows/lint.yml/badge.svg)](https://github.com/wrbell/stark-translate/actions/workflows/lint.yml)
[![Test](https://github.com/wrbell/stark-translate/actions/workflows/test.yml/badge.svg)](https://github.com/wrbell/stark-translate/actions/workflows/test.yml)
[![Security](https://github.com/wrbell/stark-translate/actions/workflows/security.yml/badge.svg)](https://github.com/wrbell/stark-translate/actions/workflows/security.yml)
[![codecov](https://codecov.io/gh/wrbell/stark-translate/graph/badge.svg)](https://codecov.io/gh/wrbell/stark-translate)

Fully on-device, live bilingual speech-to-text for Stark Road Gospel Hall (Farmington Hills, MI).
A speaker talks in English or Spanish; the laptop transcribes, translates and pushes captions to a
projector page and to phones on the local network. No cloud APIs: once the pinned models are
prepared, inference works offline.

Current release **v2026.14.0.0** (tagged 2026-09-11; GitHub Release with Mac/NVIDIA/Windows ZIPs
and the MSI; PyPI publication deferred by decision). Contracts:
[`docs/current_architecture.md`](docs/current_architecture.md) · evidence:
[`docs/mac_implementation_status.md`](docs/mac_implementation_status.md) · remaining work:
[`docs/backlog.json`](docs/backlog.json) (rendered [`docs/backlog.md`](docs/backlog.md)) · every
document: [`docs/README.md`](docs/README.md).

## Start here

| You want to… | Read |
|---|---|
| Run captions on a Sunday | [Operator runbook](docs/operator_runbook.md) |
| Set up the Mac | [macOS installation and readiness](docs/packaging/macos.md) |
| Run on a church PC or an RTX 2070 | [Lite profiles](docs/lite_profiles.md) and [`CLAUDE-windows.md`](CLAUDE-windows.md) Part B |
| Change the code | [`CLAUDE.md`](CLAUDE.md), then the guide of the directory you touch (`engines/`, `tools/`, `displays/`, `features/`, `training/`) |
| Delegate work to a coding agent | [`AGENTS.md`](AGENTS.md) (standing constraints, checks, workflow) |
| Know what is proven and what is not | [Mac implementation status](docs/mac_implementation_status.md), [evaluation index](docs/evaluation/README.md) |
| See what is left | [Backlog](docs/backlog.md) |
| Train or fine-tune models | [`training/CLAUDE.md`](training/CLAUDE.md) and [`CLAUDE-windows.md`](CLAUDE-windows.md) Part A |

## Product overview

One pipeline, one operator page, the same displays and review format everywhere. The **profile**
decides which models run and what hardware is admitted. Profiles are selected explicitly
(`--profile` or `STARK_PROFILE`) and never auto-upgrade; every row uses Silero VAD 6.2.1 (torch on
Standard, ONNX on Lite) and optional Piper TTS (`--tts`).

| Implementation | Entry point | Hardware floor | STT EN / ES | Previews | Finals | State |
|---|---|---|---|---|---|---|
| **Mac Standard** (default) | `stark-translate` / `./run_operator.sh`, profile `standard` | Apple Silicon; tested on an M3 Pro 18 GB | Parakeet TDT 0.6B v3 (MLX) / mlx-whisper large-v3-turbo | Marian opus-mt CT2 int8 on CPU | Gemma 4 E4B OptiQ 4-bit (MLX); E2B opt-in | Production path. Replay latency, endurance and quality evidence recorded; live-microphone, visible-display and bilingual-review gates open |
| **CUDA Standard** | `stark-translate --backend cuda` + `start_server.sh` | NVIDIA 16 GB; tested on an A2000 Ada | W16 Whisper large-v3-turbo LoRA → CT2 int8_float16 (both) | Marian CT2 on GPU | Gemma 4 E4B Q4_K_M via llama.cpp `b10883` | Benchmarked v2026.5–8; no recent runs (WSL box unreachable) |
| **`lite-cpu`** | `stark-translate-lite` (its default profile) | 4 cores / 8 GiB, Torch-free runtime | Whisper small CT2 int8 (both) | Marian CT2 | Marian CT2 (no Gemma) | Implemented; Mac CPU smoke and a service hour recorded; x86 unmeasured; not a fast profile today |
| **`lite-cpu-quality`** | `stark-translate-lite --profile lite-cpu-quality` | 4 cores / 16 GiB | same | Marian CT2 | Gemma 4 E2B Q4_K_M via a session-owned CPU `llama-server` | Implemented; one installed synthetic smoke; no latency gate |
| **`lite-cuda-8gb`** | `stark-translate-lite --profile lite-cuda-8gb` | 4 cores / 16 GiB / 8 GB VRAM, sm_75+ | Whisper large-v3-turbo CT2 int8_float16 on CUDA | Marian CT2 | Gemma 4 E2B Q4_K_M via CUDA `llama-server` | Implemented; nothing has run on an RTX 2070 or native Windows yet; MSI unverified |

Opt-in variants (never defaults): `--gemma4-size e2b`, `--model-family translategemma [--ab]`
(historical opt-out), `--routing-policy conservative`, `--diarize` (live speaker labels),
`--tts` (spoken translation), the `STARK_EXPERIMENT_*` latency controls (every screened arm is
closed; registry in [`docs/latency_next_experiments.md`](docs/latency_next_experiments.md)).
`--mts` (MLX drafter) is rejected before any model loads.

## How it works

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

**Policy:** fast, revisable partials; careful finals. Every 0.6 s of new speech the pipeline emits
an italic preview (Marian); when the speaker pauses for 0.5 s, or at 8 s, it runs the final STT and
a Gemma 4 translation that replaces the preview. Short, high-confidence finals may take the Marian
route instead of Gemma. Sub-second median caption delivery is the goal and is **not yet achieved**;
the measured numbers below say where the time goes.

**Glossary**

- **Partial / final** — the preview emitted while speech continues, and the caption emitted after
  the utterance ends. **Silence** finals end on the 0.5 s pause; **smart cuts** and **hard cuts**
  end long utterances early.
- **`speech_end_to_final_ms`** (schema 2) — estimated end of speech → final payload ready on the
  server. **`speech_end_to_ack_upper_bound_ms`** — the same start → a visible browser's render
  acknowledgement, including the return network hop. **First token** — speech end → first streamed
  translation token (`translation_started + ttft − speech_end`); **`first_stream`** is the browser
  acknowledgement of that first batch. Legacy `e2e_latency_ms` is processing time, not delivery.
  Definitions: [`docs/evaluation/README.md`](docs/evaluation/README.md).
- **Screen / arm / gate** — a *screen* replays fixed clips through the pipeline for a *control* and
  one or more candidate *arms*; declared *gates* (median gain, tail, previews, text, memory) decide.
  Rejected arms are never re-run or combined. Defaults change only after a passing screen and human
  review.
- **OptiQ** — the mixed-precision Gemma 4 quantization for MLX; uniform 4-bit Gemma 4 quants are broken.
- **Profile** — a bounded product configuration (`standard`, `lite-cpu`, `lite-cpu-quality`,
  `lite-cuda-8gb`). **Adapter** — a fine-tuned LoRA (Whisper W16) or Gemma delta, gated by
  `tools/health_check.py` before activation.

## Measured performance

Numbers are quoted from the dated evidence documents they link to; nothing here is recomputed.
Metric definitions are in the glossary above. Every replay screen is machine-timed and headless
(no display client, no human quality judgement) and none supports a p95 claim: the p95 columns are
reported, not gated.[^1]

### MacBook Pro M3 Pro 18 GB — Standard profile, current default code (2026-09-12)

Control arm of the [series-4 P2 screen](docs/evaluation/series4_20260912/P2-arm-screen/README.md):
two 360 s English sermon clips (A, B), three repeats each, Parakeet EN, Marian previews, Gemma 4
E4B finals.

| Metric | Clip A p50 / p95 | Clip B p50 / p95 |
|---|---|---|
| First translated token after speech end | 1165 / 1362 ms | 1095 / 1989 ms |
| Gemma-routed silence final ready | 1960 / 2399 ms | 1607 / 2508 ms |
| All silence finals ready (Gemma and Marian routes) | 1635 / 2399 ms | 1468 / 2298 ms |
| All finals ready, including smart and hard cuts | 2271 / 5406 ms | 1524 / 4293 ms |
| Marian-routed silence final ready ([2026-09-11 control](docs/evaluation/followup_20260911/X-tail-screen/README.md)) | 804.0 / 1666.9 ms | 868.8 / 984.0 ms |

The series-4 runtime fixes merged in #217 (keep-warm after the final, first stream token, Parakeet
joint decode) cut the Gemma-silence first-token median by 10.8 % (A) and 7.5 % (B) against the
previous default with byte-identical output
([identity screen](docs/evaluation/series4_20260912/P1-runtime/README.md)).

### MacBook Pro M3 Pro 18 GB — Standard profile, service length and components

| Metric | Value | Measured | Evidence |
|---|---|---|---|
| 3,640 s operator file-replay service: finals / previews | 563 / 3049 | 2026-09-11 | [series-3 endurance](docs/evaluation/series3_20260912/P1E-endurance/README.md) |
| Same service: silence finals p50 / p95 — all routes; Gemma; Marian | 1213.8 / 2236.2; 1631.3 / 2563.3; 843.7 / 1049.3 ms | 2026-09-11 | same |
| Same service: peak RSS / peak Metal | 3.38 GiB / 8.66 GiB | 2026-09-11 | same |
| Visible-browser ACK upper bound, in-file silence finals p50 / p95 (n = 401)[^2] | 1442.2 / 3033.9 ms | 2026-09-10 | [Standard service hour](docs/evaluation/overnight_endurance_20260910/README.md) |
| Isolated STT on 50 FLEURS recordings × 3: Parakeet EN WER / call p50 | 5.005 % / 220–224 ms | 2026-09-10 | [EN↔ES follow-up](docs/evaluation/mac_followup_20260910/README.md) |
| Same: Whisper turbo EN; Whisper turbo ES (default); Parakeet ES | 4.438 % / 699–739 ms; 3.016 % / 719–876 ms; 3.931 % / 224–270 ms | 2026-09-10 | same |
| Gemma 4 E4B OptiQ: isolated decode / TTFT; live decode p50 | ≈ 33 tok/s, TTFT ≈ 230–330 ms; 29.2 tok/s | 2026-09-09; 2026-09-11 | [v2026.13 Mac latency](docs/archive/v2026.13/MAC_LATENCY.md); [series-3 attribution](docs/evaluation/series3_20260912/LA-attribution/README.md) |
| Marian CT2 int8 on CPU, per 8–12-word final (offline) | 30.7 / 46.3 ms p50 / p95 | 2026-09-12 | [series-4 Marian packet](docs/evaluation/series4_20260912/P4-marian-band-packet/README.md) |
| Spanish-speaker sessions (Whisper turbo + E4B), silence final p50 range, six anchors per run[^3] | 3,120.9–4,364.8 ms (ES→EN); 2,149.9–2,894.7 ms (EN→ES) | 2026-09-10 | [Standard screen](docs/evaluation/mac_followup_20260910/standard-screen-result.md) |

### MacBook Pro M3 Pro 18 GB — opt-in Gemma 4 E2B finals

| Metric | Value | Measured | Evidence |
|---|---|---|---|
| Isolated translate p50 / p95, E2B vs E4B; canaries passed | 516.6 / 832.4 vs 839.9 / 1442.6 ms; 11/18 vs 13/18 | 2026-09-09 | [v2026.14 report](docs/evaluation/mac_v2026_14_report/README.md) |
| Matched later silence finals on a 45 s clip, E4B vs E2B p50 / p95 | 2,003 / 2,823 vs 1,516 / 1,702 ms | 2026-09-10 | [frozen screen](docs/evaluation/mac_v2026_14_screening/README.md) |
| E2B as a speculative draft for E4B, live | rejected: STT p95 511 → 2,487 ms on clip A, 12.4–12.8 GiB Metal | 2026-09-11 | [tail screen](docs/evaluation/followup_20260911/X-tail-screen/README.md) |

### This Mac's CPU — `lite-cpu` (Whisper small CT2 int8 + Marian finals)

Lite "cannot be recommended as a fast production profile today"; no Lite latency gate has passed.

| Metric | Value | Measured | Evidence |
|---|---|---|---|
| 3,640 s service hour, in-file silence finals p50 / p95 (n = 311)[^4] | 2854.6 / 11246.8 ms | 2026-09-10 | [Lite service hour](docs/evaluation/overnight_endurance_20260910/README.md) |
| Same hour: STT call p50; Marian call p50; finals that got a first preview | 1294.1 ms; 58.8 ms; 174/468 (37.18 %) | 2026-09-10 | same |
| Whisper small vs base WER (CPU CT2 int8, 600 FLEURS calls): EN; ES | 5.38 % vs 8.97 %; 5.21 % vs 11.52 % — base rejected | 2026-09-10 | [CPU STT comparison](docs/evaluation/mac_followup_20260910/cpu-stt-comparison.md) |
| Whisper small isolated call median, EN / ES (50 calls per worker, beam 5) | 1,413.7–1,493.3 / 1,491.8–1,633.6 ms | 2026-09-10 | same |
| `lite-cpu-quality` (E2B via CPU llama.cpp), one synthetic final | 3,452.8 ms speech end → ready (STT 1,251.3, translation 428.7) | 2026-09-10 | [Lite profiles](docs/lite_profiles.md) |

### NVIDIA A2000 Ada 16 GB (WSL2) — CUDA Standard, archived benchmarks

| Metric | Value | Measured | Evidence |
|---|---|---|---|
| W16 Whisper CT2 int8_float16: p50 / p95 / WER on the 41-clip bench[^5]; full-utterance fresh-eval WER | 353 / 413 ms / 11.00 %; 7.25 % | 2026-05-03 | [v2026.7 STT benchmark](docs/archive/v2026.7/STT_BENCHMARK.md) |
| Marian CT2 int8_float16: p50 / p95 / VRAM | 57 / 116 ms / 1.58 GB | 2026-05-03 | [v2026.8 Marian benchmark](docs/archive/v2026.8/MARIAN_BENCHMARK.md) |
| Gemma 4 E4B Q4_K_M via llama.cpp: p50 / p95 / decode / VRAM / canaries | 473 / 610 ms / 42.8 tok/s / 4.75 GB / 7 of 8 | 2026-05-04 | [v2026.9 Gemma phase 2](docs/archive/v2026.9/GEMMA_OPTIM_PHASE2.md) |
| Gemma 4 E2B Q4_K_M (low-VRAM): p50 / p95 / VRAM / canaries | 263 / 375 ms / 3.31 GB / 6 of 8 | 2026-05-04 | same |
| TranslateGemma 4B / 12B HF NF4 (historical): p50 / VRAM | 2381 ms / 7.22 GB; 3304 ms / 15.59 GB | 2026-04-25 | [v2026.5 benchmark](docs/archive/v2026.5/BENCHMARK.md) |

No valid Mac measurement of TranslateGemma exists: the 2026-08-30 MLX row was invalidated by the
stop-token bug (#172). Do not extrapolate any row to other hardware.

[^1]: Series-3, series-4 and follow-up screens record `p95_claim_eligible: false` (45 eligible
    Gemma-routed silence finals per arm on clip A). Their READMEs state that they certify "no human
    quality, visible delivery or production default".
[^2]: Collected on the older `752ab9a` source with the native Mac locked; browser ACKs are client
    reports, not proof of physical visibility. `first_visible_ms` (the `first_stream` ACK, #216)
    has no measurement yet because the replay harness has no display client.
[^3]: The 2026-09-10 follow-up screens have six eligible fixed-source anchors per run, below the
    100 the protocol requires for a p95 claim.
[^4]: The Standard and Lite hours "are separate functional rehearsals, not a controlled paired
    comparison".
[^5]: The bench corpus is W7-era with partial-utterance Deepgram references; the benchmark states
    its absolute WER is biased low by roughly 3–5 points and names the 7.25 % full-utterance number
    as the authoritative claim.

## Quick start

Run these from a checkout or an extracted release ZIP. The package is not on PyPI (deferred by
decision); a checkout or the release ZIP provides the current code. Full Mac setup, model
preparation and rollback: [`docs/packaging/macos.md`](docs/packaging/macos.md).

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

`setup` downloads the pinned models from `models.lock.json` and builds the Marian CT2 int8
artifacts; existing `adapters/marian_ct2/<dir>/active` directories are reused unchanged (manual
conversion: `scripts/convert_marian_ct2.py`). `bootstrap.sh --skip-systemd` performs install, setup
and preflight in one step and refuses to install into the `stt_env` rollback environment. The
legacy `requirements-*.txt` files are deprecated for inference; `requirements-windows.txt` remains
the WSL training environment.

## Common tasks

| Task | Command | Guide |
|---|---|---|
| Run a live session | `./run_operator.sh` → `http://localhost:9000/operator/` | [Operator runbook](docs/operator_runbook.md) |
| Replay a recording through the pipeline | `venv/bin/python dry_run_ab.py --audio-file clip.wav --session-id demo_en` (`--lang es` for Spanish) | [`CLAUDE-macbook.md`](CLAUDE-macbook.md) |
| Try a caption without a microphone | `venv/bin/python dry_run_ab.py --dry-run-text "For God so loved the world"` | same |
| Screen a latency experiment | `venv/bin/python -m tools.replay_bench --manifest <runs.json> --tag <tag> --configs "arm=<argv>"` then `venv/bin/python tools/tail_screen_report.py --runs <runs.jsonl> --output <report.json>` | [`tools/CLAUDE.md`](tools/CLAUDE.md), [registry](docs/latency_next_experiments.md) |
| Run the frozen Mac evaluation | `venv/bin/python tools/mac_evaluation.py validate --manifest docs/evaluation/mac_v2026_14_manifest_v2.json` | [`docs/evaluation/README.md`](docs/evaluation/README.md) |
| Gate and activate an adapter | `venv/bin/python tools/health_check.py --backend mlx --adapter DIR` then `tools/manage_adapters.py activate …` | [`tools/CLAUDE.md`](tools/CLAUDE.md), [`docs/deploy.md`](docs/deploy.md) |
| Audit the installed runtime | `scripts/audit_mac_runtime.sh --output metrics/runtime-audit-$(date +%Y%m%d-%H%M%S)` | [`docs/packaging/macos.md`](docs/packaging/macos.md) |
| Build and verify release artifacts | `python -m build && python tools/release_artifacts.py verify dist/*.whl dist/*.tar.gz` | [`docs/packaging/macos.md`](docs/packaging/macos.md) |
| Run every check CI runs | see [Testing & CI](#testing--ci) | [`CLAUDE.md`](CLAUDE.md) |

## Models

Pinned in `models.lock.json`; resolution order (explicit path → `STARK_MODELS_DIR` → `models/` →
Hugging Face cache) is shared by setup, preflight and inference. Allowed model sources are Google
Gemma 4 (and its official drafters), NVIDIA Parakeet, OpenAI Whisper, Helsinki-NLP opus-mt and
mlx-community re-quantizations of those.

| Role | Mac (MLX) default | CUDA default | Notes |
|------|-------------------|--------------|-------|
| VAD | Silero 6.2.1 (torch; ONNX opt-in) | same | 0.5 s silence trigger, 8 s max utterance, 0.6 s partial cadence |
| STT EN | Parakeet TDT 0.6B v3 (`parakeet-mlx`) | Whisper large-v3-turbo + W16 LoRA → CT2 int8_float16 | Mac `--stt-backend mlx` forces Whisper |
| STT ES | mlx-whisper large-v3-turbo | same CT2 model | The English-only Distil fallback is never used for Spanish; confidence thresholds in `settings.py` |
| Partial translation | Marian opus-mt en-es / es-en → CT2 int8 on CPU (HF fallback) | Marian CT2 int8_float16 on GPU | [Marian benchmark](docs/archive/v2026.8/MARIAN_BENCHMARK.md) |
| Final translation | Gemma 4 E4B OptiQ 4-bit (`--gemma4-size e2b` opt-in) | Gemma 4 E4B Q4_K_M via llama.cpp (`start_server.sh`), E2B for low VRAM | [CUDA benchmark](docs/archive/v2026.5/BENCHMARK.md); HF NF4 is legacy |
| Opt-out translation | TranslateGemma 4B / 12B 4-bit (`--model-family translategemma`, `--ab`) | — | Historical default; [`docs/mlx_cuda_parity.md`](docs/mlx_cuda_parity.md) |
| TTS (off) | Piper `en_US-lessac-high` / `es_MX-claude-high` (ONNX) | same | `--tts --tts-output ws\|wav\|both\|local` |

## Displays

Browser displays served over the LAN on port 8080 with captions on WebSocket 8765 (TTS audio on
8766); phones connect through the QR code on the audience display. Protocol and timing semantics:
[`displays/CLAUDE.md`](displays/CLAUDE.md).

| Display | Purpose |
|---------|---------|
| `displays/operator/` | Operator page on `:9000/operator/`: start/stop, preflight, devices, verses, summary, Review/export |
| `audience_display.html` | Projector: EN/ES side by side, fading context, fullscreen, QR overlay |
| `ab_display.html` | Operator: A (default) / Marian / B comparison with latency stats |
| `mobile_display.html` | Phone/tablet: responsive, model toggle, Spanish-only mode |
| `church_display.html` | Simplified church layout |
| `obs_overlay.html` | Transparent overlay for OBS Studio streaming |

## Training

Fine-tuning runs on Windows/WSL (A2000 Ada 16 GB); exported artifacts transfer to the Mac.
Guide: [`training/CLAUDE.md`](training/CLAUDE.md). No WSL job has run since 2026-04-30.

- **STT (Whisper LoRA):** W16 is the deployed CUDA adapter ([bench](docs/archive/v2026.7/STT_BENCHMARK.md)); W17 DoRA + hard-mix is scripted for the next WSL cycle. Labels come from the Deepgram Nova-3 oracle with a tiered theological glossary. The Mac engines load no LoRA.
- **Translation (Gemma 4):** E2B/E4B QLoRA SFT and CPO program in [`docs/gemma4_tuning/`](docs/gemma4_tuning/overview.md); adapters so far reach parity with stock E4B and the Jacobo canary still fails (#136), so stock E4B remains the default. The TranslateGemma S1–S9 sweep is historical ([`docs/archive/training/`](docs/archive/training/benchmark_training.md)).
- **Data:** ~265K verse pairs (`verse_pairs_train_v2.jsonl`, after the [Platense alignment fix](docs/platense_alignment_bug.md)), sermon pairs, hymn stanzas ([`bible_data/hymns/`](bible_data/hymns/README.md)), SHA-256 data lockfile, stratified eval sets. Public-domain Bible editions only.

## Testing & CI

```bash
ruff check . && ruff format --check .
mypy engines/ settings.py
pytest tests/ -v --cov=engines --cov=tools --cov=features --cov-fail-under=65
python tools/render_backlog.py validate && python tools/render_backlog.py render --check
python tools/render_backlog.py check-links
pytest tests/test_documentation.py -v
```

10 GitHub Actions workflow files in `.github/workflows/`: Lint (ruff, mypy, bandit, HTML Tidy on
the displays), Test (3.11 + 3.12, coverage gate 65 %), Security (pip-audit + Bandit), Release,
Windows MSI Release, PyPI Publish (build only; publishing gated on `PYPI_PUBLISH_ENABLED`), Docker
Image (GHCR), Label PRs, Commitlint, Stale. CalVer in `pyproject.toml`. Suite counts live only in
[`docs/mac_implementation_status.md`](docs/mac_implementation_status.md). On the Mac the promoted
`venv` is runtime-only; run these checks with the rollback environment's tooling
(`stt_env/bin/python -m pytest …`).

## Project structure

```
dry_run_ab.py                  Main pipeline: capture → VAD → STT → translate → displays
settings.py                    Unified config (pydantic-settings, STARK_ prefix)
models.lock.json               Pinned model sources consumed by `stark-translate setup`
.stark-python                  Launcher interpreter pointer (venv/bin/python; stt_env is the rollback)
run_operator.sh / bootstrap.sh Operator launcher; one-step install + setup + preflight
start_server.sh                CUDA: launch llama-server with the default Gemma 4 GGUF

operator_app/                  FastAPI control plane (:9000), setup/doctor/launchd CLI, preflight, review,
                               support bundles, idle-only audio tests, lite preflight, owned-process cleanup
stark_translate/               Package entry; profiles.py (standard / lite-cpu / lite-cpu-quality / lite-cuda-8gb)
engines/                       STT + translation + TTS engine layer (engines/CLAUDE.md)
  base.py, factory.py          ABCs, result dataclasses, backend detection and construction
  mlx_engine.py                Apple Silicon MLX Whisper + Gemma (streaming, warm-up, generation lock)
  parakeet_mlx_engine.py       Parakeet TDT EN (Mac default); parakeet_joint_decode.py pins its fast decode
  mlx_memory.py                Opt-in Metal wired limit (off by default)
  stt_fallback.py              English-only fallback rule; tts_engine.py Piper TTS
  cuda_engine.py, llamacpp_engine.py, marian_hf_engine.py   CUDA CT2/HF engines, llama-server client, Marian HF fallback
displays/                      Static browser displays + operator page (displays/CLAUDE.md)
features/                      Diarization, verse extraction, summary (features/CLAUDE.md)
tools/                         Evaluation, monitoring, review tooling (tools/CLAUDE.md)
  mac_evaluation.py            Frozen Mac evaluation pipeline
  replay_bench.py              Sequential real-audio replay runs for screens
  tail_screen_report.py        Offline screen gates from recorded artifacts
  pipeline_timing.py           Schema 2 timing records and render ACK tracking
  latency_experiments.py       Opt-in STARK_EXPERIMENT_* controls, validated before startup
  isolated_audio.py            PortAudio in a disposable child; no-input timeouts
  pipeline_health.py           Low-rate health/control channel read by the operator
  endurance_monitor.py         Read-only session endurance evidence
  display_server.py            Serves only the public audience display bundle
  llama_runtime.py             Pinned native llama.cpp install + session-owned llama-server (Lite)
  health_check.py, manage_adapters.py   Canary gate and adapter lifecycle
  render_backlog.py            Backlog validation/rendering/link check
training/                      Windows/WSL training scripts (training/CLAUDE.md)
scripts/                       runtime_env.sh (interpreter selection), audit_mac_runtime.sh, convert_marian_ct2.py, cuda/
docs/                          Architecture, evaluation evidence, backlog, dated archives (docs/README.md)
```

## Documentation

The full map is [`docs/README.md`](docs/README.md). The files you will reach for most:

| Doc | Contents |
|-----|----------|
| [`CLAUDE.md`](CLAUDE.md) / [`AGENTS.md`](AGENTS.md) | Developer guide / short agent guide (constraints, checks, workflow) |
| [`docs/current_architecture.md`](docs/current_architecture.md) | Inference and operator contracts for v2026.14 |
| [`docs/mac_implementation_status.md`](docs/mac_implementation_status.md) | What is implemented, what is proven, open gates |
| [`docs/backlog.json`](docs/backlog.json) | Remaining work, machine-readable (render: `tools/render_backlog.py`) |
| [`docs/evaluation/README.md`](docs/evaluation/README.md) | Metric definitions, frozen inputs, index of dated evidence |
| [`docs/latency_next_experiments.md`](docs/latency_next_experiments.md) | Closed-arm registry: every latency idea screened and its outcome |
| [`docs/lite_profiles.md`](docs/lite_profiles.md) | Lite profile contract, pinned artifacts, CPU evidence |
| [`docs/operator_runbook.md`](docs/operator_runbook.md) | Day-of-event workflow for non-technical operators |
| [`CLAUDE-macbook.md`](CLAUDE-macbook.md) / [`CLAUDE-windows.md`](CLAUDE-windows.md) | Machine guides: Mac inference; WSL training and native Lite |

## Status

v2026.14.0.0 (2026-09-11) shipped the reliability work (isolated capture, health channel, work
lease), schema 2 timing, reproducible setup, Review/export, screening harnesses, Lite profiles and
the lay-operator page. The 2026-09-11/12 boards
([overnight](docs/evaluation/overnight_20260911/STATUS.md),
[follow-up](docs/evaluation/followup_20260911/STATUS.md),
[series 3](docs/evaluation/series3_20260912/STATUS.md),
[series 4](docs/evaluation/series4_20260912/STATUS.md)) promoted the Torch 2.13 runtime with a
rollback pointer, merged output-identical runtime fixes, added the first-visible acknowledgement,
and rejected every screened latency arm; production defaults are unchanged and the sub-second goal
is not met.

Open human and device gates, tracked in [`docs/backlog.json`](docs/backlog.json): sustained live
microphone captions (#131), a natural two-speaker diarization clip (#133), hymn labels and
boundary review (#193, #194), physical second audio output, blinded bilingual review of Spanish
quality, a visible-browser timing run with a connected display, and Lite performance on x86 CPUs
and an RTX 2070.

## License

Private project. All Bible translation training data uses public domain or CC-licensed sources only.
