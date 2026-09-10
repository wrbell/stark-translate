# CLAUDE.md — Live Bilingual Speech-to-Text

> **Release lines (2026-09-09):** **main** is v2026.13. The v2026.14 candidate
> (`2026.14.0.0`) lives on `codex/mac-reliability-roadmap` (base `5154fb9`) and is
> proposed in draft [PR #192](https://github.com/wrbell/stark-translate/pull/192) —
> open, **not merged**. Overnight worktrees (docs, lite, latency, operator-ui,
> reliability) are pending integration into that PR. PyPI/package/release tags are
> pending by user choice. Do not recreate `stt_env`.
>
> Contracts: [`docs/current_architecture.md`](docs/current_architecture.md) ·
> Evidence: [`docs/mac_implementation_status.md`](docs/mac_implementation_status.md) ·
> Remaining work: [`docs/backlog.json`](docs/backlog.json) (rendered
> [`docs/backlog.md`](docs/backlog.md)) · Tonight: [`docs/overnight_status.md`](docs/overnight_status.md)

On-device live EN↔ES speech-to-text for Stark Road Gospel Hall (Farmington Hills, MI).
`--lang en` (EN→ES) · `--lang es` (ES→EN) · optional Piper TTS (`--tts`).
MLX on Apple Silicon for inference; CUDA/WSL for training. Lite CPU and native
Windows/RTX 2070 inference are equal-priority targets whose implementation is in
progress (lite worktree) and whose certification on hardware is pending.

## Two-pass pipeline (current Mac defaults, from `settings.py` / `engines/factory.py`)

| Stage | When | STT | Translation | UI |
|-------|------|-----|-------------|-----|
| **Partial** | Every 0.6 s of new speech | Parakeet MLX (EN) / mlx-whisper large-v3-turbo (ES) | Marian CT2 int8 on CPU (HF fallback) | Italic preview |
| **Final** | 0.5 s silence or 8 s max utterance | Same | Gemma 4 E4B OptiQ (`--gemma4-size e2b` opt-in) | Replaces partial |

**Policy:** fast revisable partials; careful finals. Sub-second median caption delivery
is the goal and is **not yet achieved**; it is active Mac engineering with separate
quality/certification gates (natural references, bilingual review, visible-browser ACKs).
Schema 2 `speech_end_to_final_ms` = estimated speech end → payload ready;
`speech_end_to_ack_upper_bound_ms` includes return-network time; legacy
`e2e_latency_ms` is archived processing time. Definitions:
[`docs/evaluation/README.md`](docs/evaluation/README.md).

**CUDA (v2026.8+ on A2000):** W16 Whisper CT2 + Marian CT2 + Gemma 4 E4B via llama.cpp
(`start_server.sh`, default `--no-draft`, `--mtp` opt-in, pin `b10883`). Benchmarks:
[`docs/archive/v2026.7/STT_BENCHMARK.md`](docs/archive/v2026.7/STT_BENCHMARK.md),
[`docs/archive/v2026.8/MARIAN_BENCHMARK.md`](docs/archive/v2026.8/MARIAN_BENCHMARK.md).

**Known open bug (2026-09-09):** the built-in microphone session stalled after
"Listening..." while the operator showed RUNNING from the CSV header; file replay
passed. Live-mic and physical-device checks are deferred to tomorrow. See
`mac-live-mic-stall` in the backlog.

## Environment split

| Machine | Role | Guide |
|---------|------|-------|
| MacBook M3 Pro 18 GB | Inference, operator UI, displays | [`CLAUDE-macbook.md`](CLAUDE-macbook.md) |
| Windows WSL2 A2000 Ada | Preprocess, fine-tune, export | [`CLAUDE-windows.md`](CLAUDE-windows.md) |

Adapters: WSL → copy to Mac `adapters/`. Mac install/readiness:
[`docs/packaging/macos.md`](docs/packaging/macos.md).

## Six quality layers

1. Audio preprocessing (WSL) — [`training/CLAUDE.md`](training/CLAUDE.md)
2. Data quality assessment (WSL) — same
3. Confidence flagging (Mac) — [`engines/CLAUDE.md`](engines/CLAUDE.md)
4. YouTube caption comparison — [`tools/CLAUDE.md`](tools/CLAUDE.md)
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
| v2026.13 (main) | Mac latency fixes #180–191; Parakeet EN; Marian CT2 Mac; replay harness | [`v2026.13/MAC_LATENCY.md`](docs/archive/v2026.13/MAC_LATENCY.md) |
| v2026.14 candidate | Reliability, schema 2, setup, Review/export, screening — **local branch, PR #192 draft** | [`docs/mac_implementation_status.md`](docs/mac_implementation_status.md), [`docs/evaluation/README.md`](docs/evaluation/README.md) |

## Subdirectory guides

| Directory | CLAUDE.md | AGENTS.md |
|-----------|-----------|-----------|
| [`engines/`](engines/CLAUDE.md) | Engine ABCs, MLX thread safety, models | [`engines/AGENTS.md`](engines/AGENTS.md) |
| [`training/`](training/CLAUDE.md) | Preprocess, LoRA/QLoRA, corpora | [`training/AGENTS.md`](training/AGENTS.md) |
| [`tools/`](tools/CLAUDE.md) | QE, YouTube compare, adapters, evaluation | [`tools/AGENTS.md`](tools/AGENTS.md) |
| [`displays/`](displays/CLAUDE.md) | WebSocket protocol, displays, operator SPA | [`displays/AGENTS.md`](displays/AGENTS.md) |
| [`features/`](features/CLAUDE.md) | Diarization, summary, verses | [`features/AGENTS.md`](features/AGENTS.md) |

## Extension patterns

- New engine → `engines/CLAUDE.md` § Adding a New Engine
- New language → `engines/CLAUDE.md` + `training/CLAUDE.md` (Hindi/Chinese are pending user decisions)
- New display → `displays/CLAUDE.md`
- Adapter deploy → `tools/CLAUDE.md`
- Active learning → `tools/CLAUDE.md`

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
- [ ] Phase 10 — Human gates: live-mic smoke (#131), diarization gate (#133), physical
      second output (#132), Sunday dry-run (#134); TTS routing code and live diarization
      code are implemented, their acceptance is not certified

Statuses, priorities, dependencies and acceptance per item: [`docs/backlog.json`](docs/backlog.json).
