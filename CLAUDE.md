# CLAUDE.md — Live Bilingual Speech-to-Text

> **State:** v2026.14.0.0 published 2026-09-11 (`50f81c6`); defaults unchanged since. Contracts:
> [`docs/current_architecture.md`](docs/current_architecture.md) · evidence:
> [`docs/mac_implementation_status.md`](docs/mac_implementation_status.md) · remaining work:
> [`docs/backlog.json`](docs/backlog.json) (rendered [`docs/backlog.md`](docs/backlog.md)) · latest
> boards: [series 3](docs/evaluation/series3_20260912/STATUS.md), [series 4](docs/evaluation/series4_20260912/STATUS.md)
> · every document: [`docs/README.md`](docs/README.md). Short agent guide with the same constraints:
> [`AGENTS.md`](AGENTS.md).

On-device live EN↔ES speech-to-text for Stark Road Gospel Hall (Farmington Hills, MI).
`--lang en` (EN→ES) · `--lang es` (ES→EN) · optional Piper TTS (`--tts`). MLX on Apple Silicon for
inference; CUDA/WSL for training; Lite CPU and native Windows/RTX 2070 profiles are implemented
and await hardware evidence ([`docs/lite_profiles.md`](docs/lite_profiles.md)). The product
overview, measured performance and quick start are in [`README.md`](README.md).

## Start here

| Task | Guide |
|---|---|
| Change engines, prompts, model selection | [`engines/CLAUDE.md`](engines/CLAUDE.md) |
| Evaluation, screens, replay, review tooling | [`tools/CLAUDE.md`](tools/CLAUDE.md) |
| Displays, WebSocket protocol, operator page | [`displays/CLAUDE.md`](displays/CLAUDE.md) |
| Diarization, verses, summary | [`features/CLAUDE.md`](features/CLAUDE.md) |
| Training and data (WSL) | [`training/CLAUDE.md`](training/CLAUDE.md) |
| Run or debug on the Mac | [`CLAUDE-macbook.md`](CLAUDE-macbook.md) |
| WSL training box, native Lite inference | [`CLAUDE-windows.md`](CLAUDE-windows.md) |

## Two-pass pipeline (current Mac defaults, from `settings.py` / `engines/factory.py`)

| Stage | When | STT | Translation | UI |
|-------|------|-----|-------------|-----|
| **Partial** | Every 0.6 s of new speech | Parakeet MLX (EN) / mlx-whisper large-v3-turbo (ES) | Marian CT2 int8 on CPU (HF fallback) | Italic preview |
| **Final** | 0.5 s silence or 8 s max utterance | Same | Gemma 4 E4B OptiQ (`--gemma4-size e2b` opt-in); short high-confidence finals may take the Marian route | Replaces partial |

**Policy:** fast revisable partials; careful finals. Sub-second median caption delivery is the goal
and is **not yet achieved**; the screenable hypotheses on the current models are exhausted
([registry](docs/latency_next_experiments.md)), so the next step is a model or goal decision, not
another screen. Schema 2 `speech_end_to_final_ms` = estimated speech end → final payload ready;
`speech_end_to_ack_upper_bound_ms` = speech end → visible-browser acknowledgement (includes return
network); first token = `translation_started + ttft − speech_end`, acknowledged by the browser as
the `first_stream` stage; legacy `e2e_latency_ms` is archived processing time. Definitions:
[`docs/evaluation/README.md`](docs/evaluation/README.md).

## Standing constraints

These apply to every session, attended or not, and are mirrored in [`AGENTS.md`](AGENTS.md).

- **No microphone, no speakers, unattended.** Automation uses `--audio-file` replay and
  `--dry-run-text`. Live-microphone and playback checks are attended sessions Willem runs.
- **Interpreters.** `.stark-python` points at `venv/bin/python` (Torch 2.13.0, runtime only: no
  pytest/ruff/mypy). `stt_env` (Torch 2.10) is the rollback environment: never install into,
  upgrade or recreate it (`stt_env/bin/python -m pip freeze | shasum -a 256` starts `a09be842`).
  Run checks with `stt_env/bin/python -m pytest|ruff|mypy` (read-only use).
- **Models.** Google Gemma 4 (and its official drafters/QAT), NVIDIA Parakeet, OpenAI Whisper,
  Helsinki-NLP opus-mt, and mlx-community re-quantizations of those. No Chinese-origin models.
  Revisions are pinned in `models.lock.json`; no unpinned live-path downloads.
- **Defaults.** E4B OptiQ finals, 0.5 s silence, 0.6 s cadence, Parakeet EN / mlx-whisper turbo
  ES, Marian CT2 previews change only through a declared screen that passes its gates plus human
  review. Rejected arms are never re-run as confirmations or combined. Output-identical engineering
  changes merge on a paired identity screen.
- **Evidence.** Files under `docs/evaluation/**` are immutable once written; corrections are dated
  appends. Never fabricate references, labels or numbers. Numbers live in the evidence documents;
  the README's performance section quotes and links them, every other guide links only.
- **One inference process at a time** on the Mac GPU.
- **Git.** Branch or worktree per task, never on `main`; delegated agents never commit; commitlint
  header ≤ 100 chars with a lower-case subject; rebuild a branch with
  `git reset --hard origin/main && git cherry-pick …` (never `reset --soft`) and check
  `git diff --stat origin/main` before a force-push; squash-merge with auto-merge, `gh pr
  update-branch` when BEHIND, verify earlier PRs' hunks after every merge; published tags never move.
- **Guides** keep durable rules and links: one state line, no per-PR banners, no session-scoped
  statements, no dated narrative (that belongs in the boards).

## Where the evidence lives

- **Live microphone:** capture runs in a disposable PortAudio child with no-input timeouts and the
  operator's readiness comes from the health channel. Quiet-room sessions reached ready; the
  synthetic Spanish check retained upstream sample loss, so #131 stays open.
  [Quiet-room receipts](docs/evaluation/attended_mic_20260910/README.md),
  [synthetic checks](docs/evaluation/tts_routing_20260910/README.md),
  [capture-loss accounting](docs/evaluation/mac_followup_20260910/capture-loss-accounting.md).
- **Hymns / music hold:** the energy heuristic can miss singing; operators pause during
  congregational singing. #193/#194 need natural labels and bilingual review.
  [Hymn source repairs](docs/evaluation/mac_followup_20260910/hymn-source-repairs.md),
  [natural control](docs/evaluation/mac_followup_20260910/final-c13f51f/hymn-capture.md).
- **Latency program:** every arm screened since 2026-09-10 and its outcome is in
  [`docs/latency_next_experiments.md`](docs/latency_next_experiments.md); the boards above hold the
  runs. The Marian-band review packet and the P2 text differences await bilingual review.
- **Lite and CUDA:** [`docs/lite_profiles.md`](docs/lite_profiles.md),
  [`docs/cuda_latency_proposal.md`](docs/cuda_latency_proposal.md), archives under
  [`docs/archive/`](docs/archive/).

## Environment split

| Machine | Role | Guide |
|---------|------|-------|
| MacBook M3 Pro 18 GB | Inference, operator UI, displays, screens | [`CLAUDE-macbook.md`](CLAUDE-macbook.md) |
| Windows desktop, WSL2, A2000 Ada | Preprocess, fine-tune, export, CUDA bench (unreachable from the Mac; work is delivered as scripts) | [`CLAUDE-windows.md`](CLAUDE-windows.md) Part A |
| Native Windows / RTX 2070 or CPU church PC | Lite inference (`stark-translate-lite`) | [`CLAUDE-windows.md`](CLAUDE-windows.md) Part B, [`docs/lite_profiles.md`](docs/lite_profiles.md) |

Adapters: WSL → export → copy to Mac `adapters/`. Mac install/readiness:
[`docs/packaging/macos.md`](docs/packaging/macos.md).

## Six quality layers

1. Audio preprocessing (WSL) — [`training/CLAUDE.md`](training/CLAUDE.md)
2. Data quality assessment (WSL) — same
3. Confidence flagging (Mac) — [`engines/CLAUDE.md`](engines/CLAUDE.md)
4. YouTube caption comparison — [`tools/CLAUDE.md`](tools/CLAUDE.md)
5. Translation QE — same
6. Active learning loop — infer → review → retrain; Review/export is implemented, real correction
   evidence pending (#137)

## Release history (archived evidence)

Detailed notes live under [`docs/archive/`](docs/archive/); guides link, they do not quote.

| Era | Summary | Evidence |
|-----|---------|----------|
| v2026.5–6 | llama.cpp CUDA default; operator control plane | [`v2026.5/BENCHMARK.md`](docs/archive/v2026.5/BENCHMARK.md) |
| v2026.7–8 | W16 Whisper CT2; Marian CT2 partials on CUDA | [`v2026.7/STT_BENCHMARK.md`](docs/archive/v2026.7/STT_BENCHMARK.md), [`v2026.8/MARIAN_BENCHMARK.md`](docs/archive/v2026.8/MARIAN_BENCHMARK.md) |
| v2026.9–11 | llama.cpp tuning, IQ4_XS rejected, imatrix calibration | [`v2026.9/GEMMA_OPTIM_PHASE2.md`](docs/archive/v2026.9/GEMMA_OPTIM_PHASE2.md), [`v2026.10/IQ4_XS_BENCHMARK.md`](docs/archive/v2026.10/IQ4_XS_BENCHMARK.md), [`v2026.11/IMATRIX_CALIBRATION.md`](docs/archive/v2026.11/IMATRIX_CALIBRATION.md) |
| v2026.12 | Gemma 4 OptiQ E4B Mac default; EOS bug #172 fixed | [`docs/mlx_cuda_parity.md`](docs/mlx_cuda_parity.md) |
| v2026.13 | Mac latency fixes #180–191; Parakeet EN; Marian CT2 Mac; replay harness | [`v2026.13/MAC_LATENCY.md`](docs/archive/v2026.13/MAC_LATENCY.md) |
| v2026.14.0.0 (published 2026-09-11) | Reliability (isolated capture, health, work lease), schema 2, setup, Review/export, screening, Lite profiles, latency experiments, lay operator page, offline Hindi baseline (PR #192, EN↔ES follow-up PR #196) | [`docs/mac_implementation_status.md`](docs/mac_implementation_status.md), [`docs/lite_profiles.md`](docs/lite_profiles.md) |

Work on `main` after the v2026.14 tag, one board per run:

| Board | Outcome |
|---|---|
| [Overnight 2026-09-11](docs/evaluation/overnight_20260911/STATUS.md) | Stage attribution, opt-in E2B draft (rejected), diarization interpreter, B615 pinning, tag and release published |
| [Follow-up 2026-09-11](docs/evaluation/followup_20260911/STATUS.md) | Torch 2.13 runtime promoted with the `.stark-python` rollback pointer, PyPI deferred, tail screen (both arms rejected) |
| [Series 3, 2026-09-12](docs/evaluation/series3_20260912/STATUS.md) | Attribution, L-B screen (three admitted arms rejected), first-token report, coverage gate 65, launcher pointer in launchd/bootstrap, Mac runtime audit |
| [Series 4, 2026-09-12](docs/evaluation/series4_20260912/STATUS.md) | Output-identical runtime fixes merged on a paired identity screen (keep-warm after the final, first stream token, Parakeet joint decode; wired limit rejected), first-visible `first_stream` ACK, `partial_reuse_ms` arm rejected (text guard), Smart Turn v3 endpointing no-go, Marian-band review packet |

## Subdirectory guides

| Directory | CLAUDE.md | AGENTS.md |
|-----------|-----------|-----------|
| [`engines/`](engines/CLAUDE.md) | Engine ABCs, MLX thread safety, models, env vars | [`engines/AGENTS.md`](engines/AGENTS.md) |
| [`training/`](training/CLAUDE.md) | Preprocess, LoRA/QLoRA, corpora | [`training/AGENTS.md`](training/AGENTS.md) |
| [`tools/`](tools/CLAUDE.md) | Evaluation, screens, QE, review, adapters | [`tools/AGENTS.md`](tools/AGENTS.md) |
| [`displays/`](displays/CLAUDE.md) | WebSocket protocol, displays, operator page | [`displays/AGENTS.md`](displays/AGENTS.md) |
| [`features/`](features/CLAUDE.md) | Diarization, summary, verses | [`features/AGENTS.md`](features/AGENTS.md) |

## Extension patterns

- New engine → `engines/CLAUDE.md` § Adding a New Engine
- New language → `engines/CLAUDE.md` + `training/CLAUDE.md` (Hindi/Chinese are pending user decisions; `tools/offline_hindi.py` is an offline evaluation baseline, not a live path)
- New display → `displays/CLAUDE.md` § Adding a display
- Adapter deploy → `tools/CLAUDE.md` § Adapter deployment
- Active learning → `tools/CLAUDE.md` § Review and correction contracts
- New deployment profile → `stark_translate/profiles.py` + `operator_app/lite_preflight.py` + `models.lock.json` ([`docs/lite_profiles.md`](docs/lite_profiles.md))
- New latency experiment → `tools/latency_experiments.py` (opt-in, validated before startup), declared protocol before run 1, gates from `tools/tail_screen_report.py`, outcome row in `docs/latency_next_experiments.md`

## CI/CD

10 GitHub Actions workflow files in `.github/workflows/`: Lint (ruff, mypy, bandit, HTML Tidy on
`displays/*.html`), Test (3.11 + 3.12, `--cov-fail-under=65`), Security (pip-audit + Bandit; B615
skipped in CI — see [`docs/evaluation/mac_v2026_14_security.md`](docs/evaluation/mac_v2026_14_security.md)),
Release, Windows MSI Release, PyPI Publish (build only; publishing gated on `PYPI_PUBLISH_ENABLED`),
Docker Image (GHCR), Label PRs, Commitlint, Stale. CalVer in `pyproject.toml`. Required checks for
merge: `lint`, `test (3.11)`, `test (3.12)`, `security`.

```bash
ruff check . && ruff format --check .
mypy engines/ settings.py
pytest tests/ -v --cov=engines --cov=tools --cov=features --cov-fail-under=65
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
- [ ] Phase 10 — remaining human/device gates: sustained live EN/ES microphone captions (#131),
      natural two-speaker diarization (#133), hymn labels and boundary review (#193, #194),
      physical second output, human audibility and unplug/replug checks, blinded bilingual review,
      a visible-browser timing run with a connected display (`caption-delivery-goal`,
      `visible-browser-timing-run`, `mac-live-mic-stall` in the backlog). The per-language routing
      acceptance for #132 is met and the issue was closed on 2026-09-11
      ([evidence](docs/evaluation/mac_followup_20260910/tts-routing-acceptance.md)); the laptop
      runbook rehearsal is complete and #134 is closed
      ([closeout evidence](docs/evaluation/overnight_closeout_20260910/README.md)).

Statuses, priorities, dependencies and acceptance per item: [`docs/backlog.json`](docs/backlog.json).
