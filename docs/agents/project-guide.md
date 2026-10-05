# Developer guide — Live Bilingual Speech-to-Text

> Moved here from the root `CLAUDE.md` on 2026-10-04. The agent rules (standing constraints,
> commands, conventions, Clarity) are in [`AGENTS.md`](../../AGENTS.md); this guide holds the
> reference material those rules link to.

> **State:** v2026.14.0.0 published 2026-09-11 (`50f81c6`); defaults unchanged since. Contracts:
> [`docs/current_architecture.md`](../current_architecture.md) · evidence:
> [`docs/mac_implementation_status.md`](../mac_implementation_status.md) · remaining work:
> [`docs/backlog.json`](../backlog.json) (rendered [`docs/backlog.md`](../backlog.md)) · latest
> boards: [series 3](../evaluation/series3_20260912/STATUS.md), [series 4](../evaluation/series4_20260912/STATUS.md)
> · every document: [`docs/README.md`](../README.md). Agent rules:
> [`AGENTS.md`](../../AGENTS.md).

On-device live EN↔ES speech-to-text for Stark Road Gospel Hall (Farmington Hills, MI).
`--lang en` (EN→ES) · `--lang es` (ES→EN) · optional Piper TTS (`--tts`). MLX on Apple Silicon for
inference; CUDA/WSL for training; Lite CPU and native Windows/RTX 2070 profiles are implemented
and await hardware evidence ([`docs/lite_profiles.md`](../lite_profiles.md)). The product
overview, measured performance and quick start are in [`README.md`](../../README.md).

## Start here

| Task | Guide |
|---|---|
| Change engines, prompts, model selection | [`docs/agents/engines.md`](engines.md) |
| Evaluation, screens, replay, review tooling | [`docs/agents/tools.md`](tools.md) |
| Displays, WebSocket protocol, operator page | [`docs/agents/displays.md`](displays.md) |
| Diarization, verses, summary | [`docs/agents/features.md`](features.md) |
| Training and data (WSL) | [`docs/agents/training.md`](training.md) |
| Run or debug on the Mac | [`docs/agents/platform-macbook.md`](platform-macbook.md) |
| WSL training box, native Lite inference | [`docs/agents/platform-windows.md`](platform-windows.md) |

## Two-pass pipeline (current Mac defaults, from `settings.py` / `engines/factory.py`)

| Stage | When | STT | Translation | UI |
|-------|------|-----|-------------|-----|
| **Partial** | Every 0.6 s of new speech | Parakeet MLX (EN) / mlx-whisper large-v3-turbo (ES) | Marian CT2 int8 on CPU (HF fallback) | Italic preview |
| **Final** | 0.5 s silence or 8 s max utterance | Same | Gemma 4 E4B OptiQ (`--gemma4-size e2b` opt-in); short high-confidence finals may take the Marian route | Replaces partial |

**Policy:** fast revisable partials; careful finals. Sub-second median caption delivery is the goal
and is **not yet achieved**; the screenable hypotheses on the current models are exhausted
([registry](../latency_next_experiments.md)), so the next step is a model or goal decision, not
another screen. Schema 2 `speech_end_to_final_ms` = estimated speech end → final payload ready;
`speech_end_to_ack_upper_bound_ms` = speech end → visible-browser acknowledgement (includes return
network); first token = `translation_started + ttft − speech_end`, acknowledged by the browser as
the `first_stream` stage; legacy `e2e_latency_ms` is archived processing time. Definitions:
[`docs/evaluation/README.md`](../evaluation/README.md).

## Current Mac defaults (verify in `settings.py` / `engines/factory.py` before citing)

| Role | Engine / model |
|------|----------------|
| STT EN / ES | `ParakeetMLXEngine` `mlx-community/parakeet-tdt-0.6b-v3` / `MLXWhisperEngine` `mlx-community/whisper-large-v3-turbo` |
| Previews | `MarianCT2Engine` int8 on CPU (adapter dir or managed cache); `MarianHFEngine` fallback |
| Finals | `MLXGemmaEngine` `mlx-community/gemma-4-e4b-it-OptiQ-4bit`; E2B via `--gemma4-size e2b` |
| VAD / TTS | Packaged Silero 6.2.1 (`tools/vad_runtime.py`, ONNX opt-in) / Piper, off by default |
| Profile | `standard`; `lite-cpu`, `lite-cpu-quality`, `lite-cuda-8gb` via `--profile` or `STARK_PROFILE` |

`--mts` is rejected before any model loads; `STARK_EXPERIMENT_*` controls are opt-in research arms.

## Standing constraints

The standing constraints (microphone, interpreters, models, defaults, evidence, one inference
process, guides) are in [`AGENTS.md` § Standing constraints](../../AGENTS.md#standing-constraints).
The Git rules are in [`AGENTS.md` § Pull requests and commits](../../AGENTS.md#pull-requests-and-commits).

## Where the evidence lives

- **Live microphone:** capture runs in a disposable PortAudio child with no-input timeouts and the
  operator's readiness comes from the health channel. Quiet-room sessions reached ready; the
  synthetic Spanish check retained upstream sample loss, so #131 stays open.
  [Quiet-room receipts](../evaluation/attended_mic_20260910/README.md),
  [synthetic checks](../evaluation/tts_routing_20260910/README.md),
  [capture-loss accounting](../evaluation/mac_followup_20260910/capture-loss-accounting.md).
- **Hymns / music hold:** the energy heuristic can miss singing; operators pause during
  congregational singing. #193/#194 need natural labels and bilingual review.
  [Hymn source repairs](../evaluation/mac_followup_20260910/hymn-source-repairs.md),
  [natural control](../evaluation/mac_followup_20260910/final-c13f51f/hymn-capture.md).
- **Latency program:** every arm screened since 2026-09-10 and its outcome is in
  [`docs/latency_next_experiments.md`](../latency_next_experiments.md); the boards above hold the
  runs. The Marian-band review packet and the P2 text differences await bilingual review.
- **Lite and CUDA:** [`docs/lite_profiles.md`](../lite_profiles.md),
  [`docs/cuda_latency_proposal.md`](../cuda_latency_proposal.md), archives under
  [`docs/archive/`](../archive/).

## Environment split

| Machine | Role | Guide |
|---------|------|-------|
| MacBook M3 Pro 18 GB | Inference, operator UI, displays, screens | [`docs/agents/platform-macbook.md`](platform-macbook.md) |
| Windows desktop, WSL2, A2000 Ada | Preprocess, fine-tune, export, CUDA bench (unreachable from the Mac; work is delivered as scripts) | [`docs/agents/platform-windows.md`](platform-windows.md) Part A |
| Native Windows / RTX 2070 or CPU church PC | Lite inference (`stark-translate-lite`) | [`docs/agents/platform-windows.md`](platform-windows.md) Part B, [`docs/lite_profiles.md`](../lite_profiles.md) |

Adapters: WSL → export → copy to Mac `adapters/`. Mac install/readiness:
[`docs/packaging/macos.md`](../packaging/macos.md).

## Six quality layers

1. Audio preprocessing (WSL) — [`docs/agents/training.md`](training.md)
2. Data quality assessment (WSL) — same
3. Confidence flagging (Mac) — [`docs/agents/engines.md`](engines.md)
4. YouTube caption comparison — [`docs/agents/tools.md`](tools.md)
5. Translation QE — same
6. Active learning loop — infer → review → retrain; Review/export is implemented, real correction
   evidence pending (#137)

## Release history

The release table and the boards for work on `main` after the v2026.14 tag are in
[`CHANGELOG.md`](../../CHANGELOG.md).

## Subdirectory guides

| Directory | Reference guide | Rules |
|-----------|-----------|-----------|
| [`engines/`](engines.md) | Engine ABCs, MLX thread safety, models, env vars | [`engines/AGENTS.md`](../../engines/AGENTS.md) |
| [`training/`](training.md) | Preprocess, LoRA/QLoRA, corpora | [`training/AGENTS.md`](../../training/AGENTS.md) |
| [`tools/`](tools.md) | Evaluation, screens, QE, review, adapters | [`tools/AGENTS.md`](../../tools/AGENTS.md) |
| [`displays/`](displays.md) | WebSocket protocol, displays, operator page | [`displays/AGENTS.md`](../../displays/AGENTS.md) |
| [`features/`](features.md) | Diarization, summary, verses | [`features/AGENTS.md`](../../features/AGENTS.md) |

## Extension patterns

- New engine → `docs/agents/engines.md` § Adding a New Engine (ABC → file → `factory.py` branch →
  `tests/conftest.py` mock → `models.lock.json` + setup profile)
- New language → `docs/agents/engines.md` + `docs/agents/training.md` (Hindi/Chinese are pending user decisions; `tools/offline_hindi.py` is an offline evaluation baseline, not a live path)
- New display → `docs/agents/displays.md` § Adding a display
- Adapter deploy → `docs/agents/tools.md` § Adapter deployment
- Active learning → `docs/agents/tools.md` § Review and correction contracts
- New deployment profile → `stark_translate/profiles.py` + `operator_app/lite_preflight.py` + `models.lock.json` ([`docs/lite_profiles.md`](../lite_profiles.md))
- New latency experiment → `tools/latency_experiments.py` (opt-in, validated before startup), declared protocol before run 1, gates from `tools/tail_screen_report.py`, outcome row in `docs/latency_next_experiments.md`

## CI/CD

11 GitHub Actions workflow files in `.github/workflows/`: Lint (ruff, mypy, bandit, HTML Tidy on
`displays/*.html`), Test (3.11 + 3.12, `--cov-fail-under=65`), Security (pip-audit + Bandit; B615
skipped in CI — see [`docs/evaluation/mac_v2026_14_security.md`](../evaluation/mac_v2026_14_security.md)),
Release, Windows MSI Release, PyPI Publish (build only; publishing gated on `PYPI_PUBLISH_ENABLED`),
Docker Image (GHCR), Label PRs, Commitlint, Stale, Standards (`standards.yml`, inlined from
`wrbell/standards`; not a required check). CalVer in `pyproject.toml`. Required checks for
merge: `lint`, `test (3.11)`, `test (3.12)`, `security`.

```bash
ruff check . && ruff format --check .
mypy engines/ settings.py
pytest tests/ -v --cov=engines --cov=tools --cov=features --cov-fail-under=65
python tools/render_backlog.py validate && python tools/render_backlog.py render --check
python tools/render_backlog.py check-links
pytest tests/test_documentation.py -v
```

`tests/test_documentation.py` checks the workflow count, required links and forbidden stale
claims in the guides. Latest recorded suite counts live only in
[`docs/mac_implementation_status.md`](../mac_implementation_status.md).

## Phase checklist

- [x] Phases 0–3, 5–6, 9 — infrastructure, data, first fine-tunes, operator UI
- [ ] Phase 4 — WSL full preprocess ([`docs/wsl_pipeline_refresh.md`](../wsl_pipeline_refresh.md))
- [ ] Phase 7–8 — Mac A/B (#135), active learning evidence (#137)
- [ ] Phase 10 — remaining human/device gates: sustained live EN/ES microphone captions (#131),
      natural two-speaker diarization (#133), hymn labels and boundary review (#193, #194),
      physical second output, human audibility and unplug/replug checks, blinded bilingual review,
      a visible-browser timing run with a connected display (`caption-delivery-goal`,
      `visible-browser-timing-run`, `mac-live-mic-stall` in the backlog). The per-language routing
      acceptance for #132 is met and the issue was closed on 2026-09-11
      ([evidence](../evaluation/mac_followup_20260910/tts-routing-acceptance.md)); the laptop
      runbook rehearsal is complete and #134 is closed
      ([closeout evidence](../evaluation/overnight_closeout_20260910/README.md)).

Statuses, priorities, dependencies and acceptance per item: [`docs/backlog.json`](../backlog.json).

## Clarity

The Clarity rules are in [`AGENTS.md` § Clarity](../../AGENTS.md#clarity).
