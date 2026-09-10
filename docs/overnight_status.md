# Overnight documentation status — `codex/overnight-docs`

> **Worktree role:** documentation, backlog and doc-validation tooling only. No model
> loads, installs, GPU work, device access or GitHub delivery from this branch. The parent
> session owns merge, PR, issue closure, publication and all runtime evidence
> (`docs/operator_runbook.md`, `docs/mac_implementation_status.md`, `docs/evaluation/*`,
> `docs/packaging/*`, `docs/lite_profiles.md`).

**Integration state (2026-09-10):** the first two documentation passes (`a7432c6`,
`4056fe9` on the candidate branch) and every other overnight worktree — lite, latency,
operator-ui, reliability, issue-evidence — are **integrated** on
`codex/mac-reliability-roadmap` (HEAD `c5fb689` when this pass started) and are being
validated by the parent. This branch was re-based onto that HEAD for pass 3.
**Main:** v2026.13 · **Draft [PR #192](https://github.com/wrbell/stark-translate/pull/192):**
open, base `main` ← `codex/mac-reliability-roadmap`, **not merged**. Source and issue
publishing and the final merge are authorized; PyPI/package artifacts/release tags remain
pending by user choice.

## Delivered by the three documentation passes

| Artifact | State |
|----------|-------|
| [`backlog.json`](./backlog.json) (schema 2) + rendered [`backlog.md`](./backlog.md) | `status` and `certification` kept separate; integrated worktree items moved from `in_progress` to `implemented` with `certification: pending`; #131/#134/#138 acceptance aligned with the issue text; `speech_end_to_ack_upper_bound_ms` defined as estimated speech end → visible-browser ACK |
| `tools/render_backlog.py`, `tests/test_documentation.py` | `validate`, `render [--check]`, `check-links`; workflow count derived from `.github/workflows/`; stale-claim guard |
| Root `CLAUDE.md` / `AGENTS.md`, `README.md` | Integrated state, mic-stall fix implemented with retest pending, `--mts` rejected before load, Lite profiles (`standard` default; `lite-cpu`, `lite-cpu-quality`, `lite-cuda-8gb`), Lite quick start, environment split incl. native Windows Lite |
| `CLAUDE-macbook.md` | Mic-stall section rewritten as "fix implemented, live retest deferred" with the exact modules and timeouts; MTP and profile rows; troubleshooting updated |
| `CLAUDE-windows.md` | **Restructured** into Part A (WSL training, verified flags and runbook index) and Part B (native Windows / RTX 2070 Lite inference from `docs/lite_profiles.md`); the previous 1,365-line design-era version is preserved at [`archive/training/claude_windows_design_notes.md`](./archive/training/claude_windows_design_notes.md) |
| `training/CLAUDE.md` / `AGENTS.md` | **Rewritten from `argparse` definitions** at `c5fb689`: every flag and default checked; W17 `o_proj`/`out_proj` discrepancy and the SFT recipe's v1-corpus default surfaced; Gemma 4 trainers described as "ran for v1/v1.1/v2-cpo, headers still say UNTESTED" |
| `engines/`, `displays/`, `tools/` guides | Lite profile mechanics (`apply_profile`), MTP rejection, isolated capture / health channel, lay operator page endpoints, offline Hindi tool, latency-experiment modules |
| [`roadmap.md`](./roadmap.md), [`current_architecture.md`](./current_architecture.md) | Current-state block with capture/health/Lite/MTP lines; operator contract with health-derived readiness; deployment targets as implemented-vs-pending |
| [`deploy.md`](./deploy.md) | "Status vs implementation" table (what `manage_adapters.py` / `deploy_adapters.py` / `health_check.py` actually do; SIGUSR1 hot-reload and CONVERT phase not implemented); original design retained below it |
| [`wsl_pipeline_refresh.md`](./wsl_pipeline_refresh.md), [`cuda_latency_proposal.md`](./cuda_latency_proposal.md) | Nothing executed (WSL idle since 2026-04-30); PR #162 merged; `STARK_GEMMA4_VERSE` requirement; W17 module-name check; CUDA proposal scoped away from the 2070 and the Mac |
| [`gemma4_tuning/`](./gemma4_tuning/overview.md) | Status table in `overview.md`; per-phase "as-written plan / what ran" banners; `phase_e_deploy.md` banner listing which planned registry changes never landed |

## Evidence boundaries (do not over-claim)

**Implemented and locally validated** — cite [`mac_implementation_status.md`](./mac_implementation_status.md)
and [`lite_profiles.md`](./lite_profiles.md) for counts and hashes; not duplicated in guides:

- Operator reliability (isolated capture, health channel, work lease, owned-process
  cleanup), schema 2 timing, setup/resolver, Review/export, support bundles, opt-in
  latency experiments, lay operator page
- Controlled **file-replay** EN and ES caption sessions (`20260909_233546_027169_en`,
  `20260909_233823_034893_es`, `audio_source: file`)
- Lite: isolated Mac CPU install, synthetic EN+ES caption/TTS replays, E2B GGUF and
  native llama.cpp download/integrity/version verification (no inference started)

**Implemented, not yet proven live:**

- **Built-in microphone:** the 2026-09-09 stall (`20260909_233204_799019_en`) is addressed
  by `tools/isolated_audio.py` (5 s startup / 3 s idle no-input timeouts in a disposable
  PortAudio child) and `tools/pipeline_health.py` readiness in the operator. The real mic
  retest is **deferred to tomorrow** → `mac-live-mic-stall`, `issue-131-smoke`
  (`implemented`, `certification: pending`)
- **Lite on target hardware:** no x86 CPU, native Windows or RTX 2070 run; no Lite latency
  or memory gate → `lite-cpu-inference`, `rtx2070-native-validation`
- **MTP:** live `--mts` is rejected before load; nothing remains to certify on the Mac path
  (#177 stays experimental/offline)

**External or pending user input:** physical second output (#132), dry run with laptop
stand-in (#134 — the issue text itself permits the stand-in and does not require a live
mic or human walkthrough), natural Spanish references, bilingual review, correction-loop
evidence (#137), Hindi/Chinese decision (#138 — `tools/offline_hindi.py` is an offline
baseline with no live integration, run sequentially by the parent), WSL execution.

## User decisions captured

- **Publishing:** source/issue publishing and final merge authorized; PyPI, package
  artifacts and release tags pending
- **Mac defaults:** Parakeet EN, Whisper turbo ES, Marian CT2 CPU partials, Gemma 4 E4B
  OptiQ finals, 0.6 s partial cadence / 0.5 s silence — frozen for the candidate
- **Partials vs finals:** fast revisable previews; careful finals; E2B only after review
- **Targets:** Lite CPU and RTX 2070 equal priority alongside Mac; implementation and
  certification tracked separately; default profile stays `standard`
- **Tomorrow:** live microphone retest and physical-device checks

## Parent checklist (from this pass)

1. Review pass-3 commit(s) on `codex/overnight-docs`; integrate as before
2. `python tools/render_backlog.py validate && python tools/render_backlog.py render --check`
3. `python tools/render_backlog.py check-links` and `pytest tests/test_documentation.py -v`
4. After the mic retest: flip `mac-live-mic-stall` / `issue-131-smoke` certification with
   the session id; do not close #131/#132/#134 on file-replay evidence
5. Decide on `training/run_w17_curriculum.sh` `o_proj` → `out_proj` and the
   `run_gemma4_e4b_domain_sft.sh` verse default (source edits are outside this worktree)
6. Refresh [`operator_runbook.md`](./operator_runbook.md) and
   [`mac_implementation_status.md`](./mac_implementation_status.md) with final UI /
   benchmark evidence (parent-owned)

## Unfinished / out of scope for this worktree

- `docs/mac_pipeline_refresh.md` (dated 2026-08-30) and `docs/packaging/windows.md`
  (v2026.7.2 MSI scaffold plan) were link-checked but not refreshed — parent/lite ownership.
- `docs/mac_implementation_status.md`, `docs/operator_runbook.md`, `docs/evaluation/*`,
  `docs/lite_profiles.md` were read only.
- WSL install commands in `CLAUDE-windows.md` Part A remain as last executed (2026-03/04);
  nothing ran on that machine. Native Windows Lite install steps mirror
  `docs/lite_profiles.md` and have not been executed on Windows.
- Historical numbers remain only in the archives and the labeled historical documents;
  none were added to guides.
