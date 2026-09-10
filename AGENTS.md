# AGENTS.md — Live Bilingual Speech-to-Text

> **v2026.14 candidate (local):** [`docs/mac_implementation_status.md`](docs/mac_implementation_status.md) ·
> [`docs/current_architecture.md`](docs/current_architecture.md) ·
> [`docs/backlog.json`](docs/backlog.json)
>
> **Main** is v2026.13 until root merge. Local work lives on
> `codex/mac-reliability-roadmap` (base `5154fb9`). Do not recreate `stt_env`.
> PyPI publication is pending by user choice.

On-device live EN↔ES speech-to-text for Stark Road Gospel Hall (Farmington Hills, MI).
`--lang en` (EN→ES) · `--lang es` (ES→EN) · optional Piper TTS (`--tts`).
MLX on Apple Silicon for inference; CUDA/WSL for training.

## Two-pass pipeline (current Mac defaults)

| Stage | When | STT | Translation | UI |
|-------|------|-----|-------------|-----|
| **Partial** | Every 0.6 s of new speech | Parakeet (EN) / Whisper turbo (ES) | Marian CT2 CPU (HF fallback) | Italic preview |
| **Final** | 0.5 s silence | Same | Gemma 4 E4B OptiQ (E2B opt-in) | Replaces partial |

**Policy:** fast revisable partials; careful finals. Sub-second median caption delivery
is the goal and is **not yet achieved**. Schema 2 `speech_end_to_final_ms` measures
estimated speech end → payload ready; legacy `e2e_latency_ms` is archived processing time.

**CUDA (v2026.8+ on A2000):** W16 Whisper CT2 + Marian CT2 + Gemma 4 E4B llama.cpp —
see [`docs/archive/v2026.7/STT_BENCHMARK.md`](docs/archive/v2026.7/STT_BENCHMARK.md) and
[`docs/archive/v2026.8/MARIAN_BENCHMARK.md`](docs/archive/v2026.8/MARIAN_BENCHMARK.md).

## Environment split

| Machine | Role | Guide |
|---------|------|-------|
| MacBook M3 Pro 18 GB | Inference, operator UI, displays | [`CLAUDE-macbook.md`](CLAUDE-macbook.md) |
| Windows WSL2 A2000 Ada | Preprocess, fine-tune, export | [`CLAUDE-windows.md`](CLAUDE-windows.md) |

Adapters: WSL → copy to Mac `adapters/`.

## Six quality layers

1. Audio preprocessing (WSL) — [`training/AGENTS.md`](training/AGENTS.md)
2. Data quality assessment (WSL) — same
3. Confidence flagging (Mac) — [`engines/AGENTS.md`](engines/AGENTS.md)
4. YouTube caption comparison — [`tools/AGENTS.md`](tools/AGENTS.md)
5. Translation QE — same
6. Active learning loop — infer → review → retrain (both)

## Release history (archived)

Detailed version notes live under [`docs/archive/`](docs/archive/) — do not duplicate
benchmark numbers here. Highlights:

| Era | Summary |
|-----|---------|
| v2026.5–6 | llama.cpp CUDA default; operator control plane shipped |
| v2026.7–8 | W16 Whisper CT2; Marian CT2 partials on CUDA |
| v2026.12 | Restart close-out; Gemma 4 OptiQ E4B Mac default; EOS bug #172 |
| v2026.13 | Mac latency fixes (#180–191 on main); Parakeet EN; Marian CT2 Mac; replay harness |
| v2026.14 candidate | Reliability, schema 2, setup, Review/export, screening — **local branch only** |

Open engineering debt: [`docs/backlog.md`](docs/backlog.md) (from [`backlog.json`](docs/backlog.json)).

## Subdirectory guides

| Directory | AGENTS.md | CLAUDE.md |
|-----------|-----------|-----------|
| [`engines/`](engines/AGENTS.md) | Engine ABCs, MLX thread safety, models | [`engines/CLAUDE.md`](engines/CLAUDE.md) |
| [`training/`](training/AGENTS.md) | Preprocess, LoRA/QLoRA, corpora | [`training/CLAUDE.md`](training/CLAUDE.md) |
| [`tools/`](tools/AGENTS.md) | QE, YouTube compare, adapters | [`tools/CLAUDE.md`](tools/CLAUDE.md) |
| [`displays/`](displays/AGENTS.md) | WebSocket protocol, displays | [`displays/CLAUDE.md`](displays/CLAUDE.md) |
| [`features/`](features/AGENTS.md) | Diarization, summary, verses | [`features/CLAUDE.md`](features/CLAUDE.md) |

## Extension patterns

- New engine → `engines/AGENTS.md` § Adding a New Engine
- New language → `engines/AGENTS.md` + `training/AGENTS.md`
- New display → `displays/AGENTS.md`
- Adapter deploy → `tools/AGENTS.md`
- Active learning → `tools/AGENTS.md`

## CI/CD

Seven GitHub Actions: lint, test (coverage gate in `test.yml`), security, release,
label, commitlint, stale. CalVer in `pyproject.toml`.

```bash
ruff check . && ruff format --check .
mypy engines/ settings.py
pytest tests/ -v --cov=engines --cov=tools --cov=features
python tools/render_backlog.py validate
pytest tests/test_documentation.py -v
```

## Phase checklist

- [x] Phases 0–3, 5–6, 9 — infrastructure, data, first fine-tunes, operator UI
- [ ] Phase 4 — WSL full preprocess ([`docs/wsl_pipeline_refresh.md`](docs/wsl_pipeline_refresh.md))
- [ ] Phase 7–8 — Mac A/B, active learning loop
- [ ] Phase 10 — Human gates: natural audio review, diarization gate, second output, Sunday dry-run

Remaining tasks with status, priority, and acceptance: [`docs/backlog.json`](docs/backlog.json).
