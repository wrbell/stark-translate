# AGENTS.md — Live Bilingual Speech-to-Text

> **Mac EN↔ES follow-up:** [PR #196](https://github.com/wrbell/stark-translate/pull/196) records this work.
> [EN↔ES evidence](docs/evaluation/mac_followup_20260910/README.md) records completed
> screens with no qualified arms, CPU Whisper-base rejection and silent hymn diagnostics.
> Defaults are unchanged. Current artifact, service and merge results are recorded in
> [implementation status](docs/mac_implementation_status.md).

> **Source and releases (2026-09-10):** this guide describes v2026.14 source
> (`2026.14.0.0`), with historical integration in [PR #192](https://github.com/wrbell/stark-translate/pull/192)
> and the EN↔ES follow-up in [PR #196](https://github.com/wrbell/stark-translate/pull/196).
> Current integration and delivery state is recorded in [implementation status](docs/mac_implementation_status.md).
> The last published release recorded here is **v2026.13**. Source integration,
> release publication and service certification are separate; PyPI/package artifacts
> and release tags remain pending by user choice. Do not recreate `stt_env`.
>
> Paired human guide: [`CLAUDE.md`](CLAUDE.md) (same content, human-facing links).
>
> Contracts: [`docs/current_architecture.md`](docs/current_architecture.md) ·
> Evidence: [`docs/mac_implementation_status.md`](docs/mac_implementation_status.md) ·
> Remaining work: [`docs/backlog.json`](docs/backlog.json) (rendered
> [`docs/backlog.md`](docs/backlog.md)) · Overnight: [`docs/overnight_status.md`](docs/overnight_status.md)

On-device live EN↔ES speech-to-text for Stark Road Gospel Hall (Farmington Hills, MI).
`--lang en` (EN→ES) · `--lang es` (ES→EN) · optional Piper TTS (`--tts`).
MLX on Apple Silicon for inference; CUDA/WSL for training. Lite CPU and native
Windows/RTX 2070 inference are equal-priority targets: the profiles are **implemented**
(`standard` default; opt-in `lite-cpu`, `lite-cpu-quality`, `lite-cuda-8gb`) and passed an
isolated Mac CPU synthetic smoke; hardware performance and certification are pending
([`docs/lite_profiles.md`](docs/lite_profiles.md)).

## Two-pass pipeline (current Mac defaults, from `settings.py` / `engines/factory.py`)

| Stage | When | STT | Translation | UI |
|-------|------|-----|-------------|-----|
| **Partial** | Every 0.6 s of new speech | Parakeet MLX (EN) / mlx-whisper large-v3-turbo (ES) | Marian CT2 int8 on CPU (HF fallback) | Italic preview |
| **Final** | 0.5 s silence or 8 s max utterance | Same | Gemma 4 E4B OptiQ (`--gemma4-size e2b` opt-in) | Replaces partial |

**Policy:** fast revisable partials; careful finals. Sub-second median caption delivery
is the goal and is **not yet achieved**; it is active Mac engineering with separate
quality/certification gates (natural references, bilingual review, visible-browser ACKs).
Schema 2 `speech_end_to_final_ms` = estimated speech end → final payload ready;
`speech_end_to_ack_upper_bound_ms` = estimated speech end → visible-browser
acknowledgement (includes return-network time); legacy `e2e_latency_ms` is archived
processing time. Definitions: [`docs/evaluation/README.md`](docs/evaluation/README.md).

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

EN↔ES remains the active speed priority. The installed Standard and CPU Lite
full-service runs on `752ab9a` completed with consistent retained source spans,
required writes and process cleanup. Lite was functional on this Mac, but sparse
previews and large observed tails do not support recommending it as a fast
production profile today. See the [separate endurance cohorts](docs/evaluation/overnight_endurance_20260910/README.md);
these observational runs do not establish a causal speed comparison, human quality,
physical display visibility or a default promotion.

**Lite profiles** (`stark_translate/profiles.py`, `stark-translate-lite`): Whisper small
CT2 int8 + Marian CT2 finals on CPU (`lite-cpu`), Gemma 4 E2B Q4_K_M via a session-owned
`llama-server` (`lite-cpu-quality`, `lite-cuda-8gb`); ONNX Silero; no A/B, drafting,
multiprocess or live diarization. Selected explicitly, never auto-upgraded.

**CUDA (v2026.8+ on A2000):** W16 Whisper CT2 + Marian CT2 + Gemma 4 E4B via llama.cpp
(`start_server.sh`, default `--no-draft`, `--mtp` opt-in, pin `b10883`). Benchmarks:
[`docs/archive/v2026.7/STT_BENCHMARK.md`](docs/archive/v2026.7/STT_BENCHMARK.md),
[`docs/archive/v2026.8/MARIAN_BENCHMARK.md`](docs/archive/v2026.8/MARIAN_BENCHMARK.md).
On the Mac, `--mts` (MLX MTP drafter, #177) is **rejected before any model loads**
(`validate_live_mts`); it stays an offline experiment.

**Live microphone (2026-09-09 → 10):** the built-in-mic session
`20260909_233204_799019_en` stalled after "Listening..." while the operator showed RUNNING
from the CSV header. The fix is **implemented**: PortAudio runs in a disposable child
(`tools/isolated_audio.py`, `tools/capture_worker.py`) with a 5 s no-input startup timeout
and 3 s idle timeout that fail the session (`AudioCaptureError`), and operator readiness
comes from `tools/pipeline_health.py` phases (`loading → listening → ready`,
`input_error`) rather than file presence. The September 10 attended EN and ES
microphone sessions both received real frames, reached ready and stopped cleanly;
EN pause/resume and language restart also passed. These quiet-room runs did not
establish spoken caption quality. Later synthetic speaker-to-microphone checks
produced captions in both languages: English completed cleanly; Spanish and a
traced retest failed because capture lost samples. The retest lost no parent
handoff frames but still lost 160 ms upstream. Keep that failure open.

Exact microphone name and host API now cross preflight, restart and capture;
the native child resolves the current index and rejects missing/ambiguous input.
A two-second native probe passed stale-index resolution and missing-name rejection,
with no saved audio or STT. Native inference workers also drain before EOF summaries
and model unloading, preserving queued TTS. These repairs do not certify sustained
microphone capture or human-heard output. See [quiet-room receipts](docs/evaluation/attended_mic_20260910/README.md),
[synthetic checks and identity probe](docs/evaluation/tts_routing_20260910/README.md),
and `mac-live-mic-stall` / `issue-131-smoke` in the backlog.

The later [capture-loss accounting repair](docs/evaluation/mac_followup_20260910/capture-loss-accounting.md)
(`d7ed43d`) separates counted worker FIFO drops from native-overflow flags with an
unknown lost-sample count, validates bounded POSIX terminal receipts and records
shutdown-only losses. It does not certify native capture reliability or resolve
the retained Spanish failure; no native audio test was performed for this repair.

The [hymn source repairs](docs/evaluation/mac_followup_20260910/hymn-source-repairs.md)
preserve existing whitespace and retain the first 14 speech frames when music-hold
recovery accepts an onset. The completed [350-second natural file control](docs/evaluation/mac_followup_20260910/final-c13f51f/hymn-capture.md)
never entered music hold; the [102-call text packet](docs/evaluation/mac_followup_20260910/final-c13f51f/hymn-boundary.md)
used supplied boundary hypotheses. Neither approves automatic singing detection or
bilingual meaning. Thresholds and defaults are unchanged; #193/#194 remain open.
Current source checks are in the [760e948 validation record](docs/evaluation/mac_followup_20260910/final-760e948/source-validation.md).
The [delivery packet](docs/evaluation/mac_followup_20260910/final-760e948/README.md) records
artifact and service acceptance separately; the hymn evidence retains its c13 source identity.

## Environment split

| Machine | Role | Guide |
|---------|------|-------|
| MacBook M3 Pro 18 GB | Inference, operator UI, displays | [`CLAUDE-macbook.md`](CLAUDE-macbook.md) |
| Windows desktop, WSL2, A2000 Ada | Preprocess, fine-tune, export, CUDA bench | [`CLAUDE-windows.md`](CLAUDE-windows.md) Part A |
| Native Windows / RTX 2070 or CPU church PC | Lite inference (`stark-translate-lite`) | [`CLAUDE-windows.md`](CLAUDE-windows.md) Part B, [`docs/lite_profiles.md`](docs/lite_profiles.md) |

Adapters: WSL → export → copy to Mac `adapters/`. Mac install/readiness:
[`docs/packaging/macos.md`](docs/packaging/macos.md).

## Six quality layers

1. Audio preprocessing (WSL) — [`training/AGENTS.md`](training/AGENTS.md)
2. Data quality assessment (WSL) — same
3. Confidence flagging (Mac) — [`engines/AGENTS.md`](engines/AGENTS.md)
4. YouTube caption comparison — [`tools/AGENTS.md`](tools/AGENTS.md)
5. Translation QE — same
6. Active learning loop — infer → review → retrain (both); Review/export is
   implemented, real correction evidence pending (#137)

## Release history (archived evidence)

Detailed notes live under [`docs/archive/`](docs/archive/) — do not duplicate
benchmark numbers in guides.

| Era | Summary | Evidence |
|-----|---------|----------|
| v2026.5–6 | llama.cpp CUDA default; operator control plane | [`v2026.5/BENCHMARK.md`](docs/archive/v2026.5/BENCHMARK.md) |
| v2026.7–8 | W16 Whisper CT2; Marian CT2 partials on CUDA | [`v2026.7/STT_BENCHMARK.md`](docs/archive/v2026.7/STT_BENCHMARK.md), [`v2026.8/MARIAN_BENCHMARK.md`](docs/archive/v2026.8/MARIAN_BENCHMARK.md) |
| v2026.9–11 | llama.cpp tuning, IQ4_XS rejected, imatrix calibration | [`v2026.9/GEMMA_OPTIM_PHASE2.md`](docs/archive/v2026.9/GEMMA_OPTIM_PHASE2.md), [`v2026.10/IQ4_XS_BENCHMARK.md`](docs/archive/v2026.10/IQ4_XS_BENCHMARK.md), [`v2026.11/IMATRIX_CALIBRATION.md`](docs/archive/v2026.11/IMATRIX_CALIBRATION.md) |
| v2026.12 | Gemma 4 OptiQ E4B Mac default; EOS bug #172 fixed | [`docs/mlx_cuda_parity.md`](docs/mlx_cuda_parity.md) |
| v2026.13 (last published release) | Mac latency fixes #180–191; Parakeet EN; Marian CT2 Mac; replay harness | [`v2026.13/MAC_LATENCY.md`](docs/archive/v2026.13/MAC_LATENCY.md) |
| v2026.14 source | Reliability (isolated capture, health, work lease), schema 2, setup, Review/export, screening, Lite profiles, latency experiments, lay operator page, offline Hindi baseline — **integration history in PR #192 and EN↔ES follow-up in PR #196; acceptance remains evidence-specific** | [`docs/mac_implementation_status.md`](docs/mac_implementation_status.md), [`docs/evaluation/README.md`](docs/evaluation/README.md), [`docs/lite_profiles.md`](docs/lite_profiles.md) |

## Subdirectory guides

| Directory | AGENTS.md | CLAUDE.md |
|-----------|-----------|-----------|
| [`engines/`](engines/AGENTS.md) | Engine ABCs, MLX thread safety, models | [`engines/CLAUDE.md`](engines/CLAUDE.md) |
| [`training/`](training/AGENTS.md) | Preprocess, LoRA/QLoRA, corpora | [`training/CLAUDE.md`](training/CLAUDE.md) |
| [`tools/`](tools/AGENTS.md) | QE, YouTube compare, adapters, evaluation | [`tools/CLAUDE.md`](tools/CLAUDE.md) |
| [`displays/`](displays/AGENTS.md) | WebSocket protocol, displays, operator SPA | [`displays/CLAUDE.md`](displays/CLAUDE.md) |
| [`features/`](features/AGENTS.md) | Diarization, summary, verses | [`features/CLAUDE.md`](features/CLAUDE.md) |

## Extension patterns

- New engine → `engines/AGENTS.md` § Adding a New Engine (details in `engines/CLAUDE.md`)
- New language → `engines/AGENTS.md` + `training/AGENTS.md` (Hindi/Chinese are pending user decisions; `tools/offline_hindi.py` is an offline evaluation baseline, not a live path)
- New display → `displays/AGENTS.md`
- Adapter deploy → `tools/AGENTS.md`
- Active learning → `tools/AGENTS.md`
- New deployment profile → `stark_translate/profiles.py` + `operator_app/lite_preflight.py` + `models.lock.json` ([`docs/lite_profiles.md`](docs/lite_profiles.md))

## CI/CD

10 GitHub Actions workflow files in `.github/workflows/`: Lint, Test (3.11 + 3.12,
coverage gate in `test.yml`), Security (pip-audit + Bandit; B615 skipped in CI —
see [`docs/evaluation/mac_v2026_14_security.md`](docs/evaluation/mac_v2026_14_security.md)),
Release, Windows MSI Release, PyPI Publish (tag-triggered; trusted publisher pending),
Docker Image (GHCR), Label PRs, Commitlint, Stale. CalVer in `pyproject.toml`.

```bash
ruff check . && ruff format --check .
mypy engines/ settings.py
pytest tests/ -v --cov=engines --cov=tools --cov=features
python tools/render_backlog.py validate && python tools/render_backlog.py render --check
python tools/render_backlog.py check-links
pytest tests/test_documentation.py -v
```

Latest recorded suite counts live only in
[`docs/mac_implementation_status.md`](docs/mac_implementation_status.md).

## Phase checklist

- [x] Phases 0–3, 5–6, 9 — infrastructure, data, first fine-tunes, operator UI
- [ ] Phase 4 — WSL full preprocess ([`docs/wsl_pipeline_refresh.md`](docs/wsl_pipeline_refresh.md))
- [ ] Phase 7–8 — Mac A/B (#135), active learning evidence (#137)
- [ ] Phase 10 — Remaining human/device gates: sustained live EN/ES microphone
      captions (#131), natural two-speaker diarization (#133), physical output,
      human audibility and unplug/replug checks. The original per-language
      [routing acceptance for #132](docs/evaluation/mac_followup_20260910/tts-routing-acceptance.md)
      is met; issue closure follows reviewed merge. These separate physical checks
      do not expand that issue's physical-or-virtual routing requirement.
      The laptop runbook rehearsal is complete and #134 is closed; see
      [closeout evidence](docs/evaluation/overnight_closeout_20260910/README.md).

Statuses, priorities, dependencies and acceptance per item: [`docs/backlog.json`](docs/backlog.json).
