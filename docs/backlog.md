# Remaining backlog — Stark Road Bilingual Speech-to-Text

> **Canonical machine-readable source:** [`backlog.json`](./backlog.json).
> Regenerate this file with `python tools/render_backlog.py render`.
> **Last updated:** 2026-09-09

## Integration status

- **Main release:** `v2026.13` — main at 09e4679; v2026.13 PRs #180–191 merged
- **Local candidate:** `2026.14.0.0` on `codex/mac-reliability-roadmap` (base `5154fb9`)
- **Publication:** pending by user choice

Items marked implemented or validated below exist on the local reliability branch unless noted as main-only. Do not treat them as released on main or PyPI until root integration supplies merge/tag evidence.

## Current Mac defaults

- EN STT: `parakeet-mlx` · ES STT: `mlx-whisper large-v3-turbo`
- Partials: `marian_ct2_cpu_int8_with_hf_fallback` · Finals: `gemma4_e4b_optiq`
- Silence `0.5` s · partial cadence `0.6` s · MTP `off`

See [`current_architecture.md`](./current_architecture.md) and [`mac_implementation_status.md`](./mac_implementation_status.md) for contracts and evidence.

## Pending Input Or Hardware

### `caption-delivery-goal` — Sub-second median speech-end-to-caption delivery

- **Priority:** P0 · **Machine:** mac
- **Depends on:** `natural-spanish-refs`, `bilingual-blinded-review`, `visible-browser-timing-run`
- **Sources:** `docs/mac_implementation_status.md`, `docs/evaluation/README.md`
- **Acceptance:** Schema 2 speech_end_to_final_ms and visible ACK establish median under 1000 ms on natural bilingual speech with approved references.
- **Next action:** Run frozen visible-browser replay after human reference gates; do not pool historical cohorts.

### `issue-134-sunday-dry-run` — Sunday dry-run on church hardware (#134)

- **Priority:** P0 · **Machine:** mac
- **Depends on:** `physical-second-output`, `natural-two-speaker`
- **Sources:** [134](https://github.com/wrbell/stark-translate/issues/134), `docs/operator_runbook.md`
- **Acceptance:** Written dry-run note with time-to-first-caption and non-technical operator blockers filed as issues.
- **Next action:** Schedule service rehearsal with natural bilingual speech and real audio hardware; controlled hymn rehearsal does not close this gate.

### `bilingual-blinded-review` — Blinded bilingual meaning and terminology review

- **Priority:** P1 · **Machine:** mac
- **Depends on:** `natural-spanish-refs`
- **Sources:** `docs/evaluation/mac_v2026_14_quality/comparison.md`, `docs/mac_implementation_status.md`
- **Acceptance:** Human review of meaning errors and terminology preferences completed; E2B speed tradeoff documented before any default change.
- **Next action:** Review bounded E4B vs E2B comparison outputs; do not switch default on text-only canary counts alone.

### `cuda-latency-proposal` — CUDA latency proposal execution on A2000

- **Priority:** P1 · **Machine:** wsl
- **Depends on:** none
- **Sources:** `docs/cuda_latency_proposal.md`, `scripts/cuda/`, [174](https://github.com/wrbell/stark-translate/issues/174)
- **Acceptance:** llama.cpp ≥ b10883, MTP opt-in bench, -fa retest, W16 HF fp16 and Parakeet probes recorded.
- **Next action:** Run scripts/cuda/*.sh on WSL; not executed in v2026.13 close-out.

### `issue-135-mac-ab` — Mac live A/B: W16 STT and v2-cpo Gemma (#135)

- **Priority:** P1 · **Machine:** both
- **Depends on:** `wsl-w17-export`
- **Sources:** [135](https://github.com/wrbell/stark-translate/issues/135), `docs/wsl_pipeline_refresh.md`
- **Acceptance:** Short A/B note with canary scores and ship/no-ship decision for v2-cpo; stock E4B remains default if no-ship.
- **Next action:** Transfer W16 CT2 and v2-cpo artifacts from WSL; run health_check --n-canaries 8 on Mac.

### `lite-cpu-packaging` — Lite CPU inference packaging validation

- **Priority:** P1 · **Machine:** any
- **Depends on:** none
- **Sources:** `docs/packaging/pypi.md`
- **Acceptance:** CPU-only install path documented and smoke-tested outside Mac MLX extras scope.
- **Next action:** Owned by overnight lite agent; equal priority with RTX 2070 path per user decision.

### `natural-spanish-refs` — Human-reviewed natural Spanish references (≥50 utterances)

- **Priority:** P1 · **Machine:** mac
- **Depends on:** none
- **Sources:** `docs/mac_implementation_status.md`, `docs/evaluation/README.md`
- **Acceptance:** At least 50 approved natural Spanish utterances per language direction policy; no WER claim without approval.
- **Next action:** User to supply natural Spanish recording location; 11 Spanish candidates exist unapproved.

### `natural-two-speaker` — Natural two-speaker audio and human transition labels

- **Priority:** P1 · **Machine:** mac
- **Depends on:** none
- **Sources:** `docs/mac_implementation_status.md`
- **Acceptance:** Labeled natural clip with human speaker-transition ground truth available for diarization and service rehearsal.
- **Next action:** Obtain or record church two-speaker segment; synthetic routing probes do not substitute.

### `physical-second-output` — Second physical audio output validation

- **Priority:** P1 · **Machine:** mac
- **Depends on:** none
- **Sources:** `docs/mac_implementation_status.md`
- **Acceptance:** Unplug/replug, device selection persistence, and acoustic playback verified on non-built-in output.
- **Next action:** Exercise church USB/interface path; built-in speaker rehearsal is insufficient.

### `rtx2070-native-validation` — Native Windows / RTX 2070 inference validation

- **Priority:** P1 · **Machine:** windows
- **Depends on:** none
- **Sources:** `docs/archive/research/rtx2070_feasibility.md`, `packaging/windows/README.md`
- **Acceptance:** Documented smoke on 2070-class hardware with llama.cpp E2B/E4B tier selection.
- **Next action:** Execute on target hardware; v2026.13 MSI digest verified without Windows run here.

### `visible-browser-timing-run` — Frozen visible-browser timing run

- **Priority:** P1 · **Machine:** mac
- **Depends on:** none
- **Sources:** `docs/evaluation/mac_v2026_14_screening/README.md`, `docs/evaluation/mac_v2026_14_routing/README.md`
- **Acceptance:** Non-zero visible final ACK coverage on acceptance configuration with schema 2 fields recorded.
- **Next action:** Repeat replay with audience display connected; 48-run screen and 24 routing probes had 0/348 and 0/72 ACK coverage.

### `wsl-e4b-domain-sft` — Gemma 4 E4B domain SFT → GGUF on WSL

- **Priority:** P1 · **Machine:** wsl
- **Depends on:** `wsl-phase4`
- **Sources:** `docs/wsl_pipeline_refresh.md`, `training/run_gemma4_e4b_domain_sft.sh`
- **Acceptance:** 8-canary sanity pass on exported GGUF; domain SFT artifact ready for Mac/CUDA transfer.
- **Next action:** Run after Phase 4 or parallel if corpus ready.

### `wsl-phase4` — Phase 4 full audio preprocessing on WSL

- **Priority:** P1 · **Machine:** wsl
- **Depends on:** none
- **Sources:** `docs/wsl_pipeline_refresh.md`, `training/run_phase4_preprocess.sh`
- **Acceptance:** phase4_status.json complete for sermon corpus on WSL storage.
- **Next action:** Execute runbook §1 on A2000 box when WAVs available.

### `wsl-w17-export` — W17 Whisper DoRA + hard-mix → CT2

- **Priority:** P1 · **Machine:** wsl
- **Depends on:** `wsl-phase4`
- **Sources:** `docs/wsl_pipeline_refresh.md`, `training/run_w17_curriculum.sh`
- **Acceptance:** benchmark_stt_engines.py gate: W17 WER ≤ W16 on fresh eval.
- **Next action:** Script exists; training not yet executed.

### `issue-136-jacobo-cpo` — Jacobo canary preference triples + CPO (#136)

- **Priority:** P2 · **Machine:** wsl
- **Depends on:** `wsl-e4b-domain-sft`
- **Sources:** [136](https://github.com/wrbell/stark-translate/issues/136), `docs/gemma4_tuning/v3_directions.md`
- **Acceptance:** Jacobo canary passes or documented impossibility; COMET-22 does not regress vs v2-cpo.
- **Next action:** Hand-craft 50–100 preference triples; one CPO continue from v2 on A2000.

### `pypi-publication` — PyPI trusted publisher and release tag

- **Priority:** P2 · **Machine:** any
- **Depends on:** none
- **Sources:** `docs/mac_implementation_status.md`, `.github/workflows/pypi.yml`
- **Acceptance:** New version tag published; trusted publisher mapped (owner wrbell, repo stark-translate, workflow pypi.yml, environment pypi).
- **Next action:** User explicitly left publishing pending; merge to main authorized when integration complete.

### `issue-138-hindi-zero-shot` — Hindi zero-shot baseline (#138)

- **Priority:** P3 · **Machine:** mac
- **Depends on:** none
- **Sources:** [138](https://github.com/wrbell/stark-translate/issues/138), `docs/archive/research/multi_lingual.md`
- **Acceptance:** Short baseline note on zero-shot Hindi through live pipeline and QLoRA week decision.
- **Next action:** Offline Hindi generation exists for 43 inputs; references and live Hindi/Chinese remain later.

## Experimental

### `conservative-marian-routing` — Conservative Marian partial routing

- **Priority:** P3 · **Machine:** mac
- **Depends on:** none
- **Sources:** `docs/evaluation/mac_v2026_14_routing/README.md`
- **Acceptance:** Synthetic EN/ES routing probes pass; natural Spanish and human review still required before default.
- **Next action:** Keep opt-in; 24/24 synthetic runs exited zero.

### `issue-177-mtp` — Gemma 4 MTP drafter on MLX (#177)

- **Priority:** P3 · **Machine:** mac
- **Depends on:** none
- **Sources:** [177](https://github.com/wrbell/stark-translate/issues/177), `docs/archive/v2026.13/MAC_LATENCY.md`, `engines/mlx_spec.py`
- **Acceptance:** Acceptance rate and latency win justify opt-in default; byte-identical output already proven.
- **Next action:** Keep off; probe showed ≤14% latency win at 31% acceptance; RoPE-offset suspect rejected.

## Deferred

### `security-b615-pinning` — Remaining unpinned Hugging Face download paths (B615)

- **Priority:** P2 · **Machine:** both
- **Depends on:** none
- **Sources:** `docs/evaluation/mac_v2026_14_security.md`
- **Acceptance:** Optional/fallback HF paths pin revisions or document explicit operator-only scope.
- **Next action:** 27 medium B615 findings documented; CI bandit pass excludes B615; prioritize live-path fallbacks first.

### `issue-176-multiprocess` — --multiprocess TranslateGemma prompt staleness (#176)

- **Priority:** P3 · **Machine:** mac
- **Depends on:** none
- **Sources:** [176](https://github.com/wrbell/stark-translate/issues/176), `workers.py`
- **Acceptance:** workers.py uses translation_prompts.build_chat_messages + ensure_stop_tokens for gemma4, or flag deprecated.
- **Next action:** Low priority; in-process max_workers=2 is production path.

### `macos-shortcuts` — macOS Shortcuts voice-command triggers

- **Priority:** P3 · **Machine:** mac
- **Depends on:** none
- **Sources:** `docs/roadmap.md`
- **Acceptance:** Optional operator convenience documented if useful.
- **Next action:** None scheduled.

### `multilingual-expansion` — Hindi and Chinese QLoRA expansion (Phase 8)

- **Priority:** P3 · **Machine:** both
- **Depends on:** `issue-138-hindi-zero-shot`
- **Sources:** `docs/roadmap.md`, `docs/archive/research/multi_lingual.md`
- **Acceptance:** chrF++/COMET gates and adapter switching per roadmap Phase 8.
- **Next action:** Defer until Whisper fine-tuning stabilizes and EN/ES gates close.

### `v2026-9-followups` — llama.cpp deferred optimizations (-fa, -c 2048, prompt-cache reuse)

- **Priority:** P3 · **Machine:** wsl
- **Depends on:** `cuda-latency-proposal`
- **Sources:** `docs/roadmap.md`, `docs/archive/v2026.9/GEMMA_OPTIM_PHASE2.md`
- **Acceptance:** Each experiment meets 20% latency gate without canary regression.
- **Next action:** Folded into CUDA latency proposal; -fa retest blocked on newer build.

## Implemented

### `mac-reliability-implementation` — v2026.14 Mac reliability program (local branch)

- **Priority:** P0 · **Machine:** mac
- **Depends on:** none
- **Sources:** `docs/mac_implementation_status.md`, `docs/overnight_status.md`
- **Acceptance:** Operator session identity, schema 2 timing, setup/resolver, review/export, opt-in experiments, packaging guards implemented on codex/mac-reliability-roadmap.
- **Next action:** Root owns merge, tag, and GitHub delivery; do not claim main release until integrated.

### `issue-132-tts-routing` — Multi-channel TTS routing (9.4.1 / #132)

- **Priority:** P1 · **Machine:** mac
- **Depends on:** none
- **Sources:** [132](https://github.com/wrbell/stark-translate/issues/132), `docs/roadmap.md`, `tests/test_phase9_4_1_tts_device.py`
- **Acceptance:** Per-language device map, operator persistence, hotplug retry; EN and ES routable independently.
- **Next action:** Close #132 after root merge; validate acoustic playback on second physical output separately.

### `issue-133-diarize-gate` — Live diarization gate (9.6.1 / #133)

- **Priority:** P1 · **Machine:** mac
- **Depends on:** `natural-two-speaker`
- **Sources:** [133](https://github.com/wrbell/stark-translate/issues/133), `docs/live_diarization.md`, `features/live_diarize.py`
- **Acceptance:** Code behind --diarize (default off) with speaker on finals/CSV/WebSocket; p95 finals within +50 ms vs off on two-speaker clip.
- **Next action:** Run tools/replay_bench.py gate with HF_TOKEN or SpeechBrain ECAPA and human-labeled two-speaker audio.

### `issue-137-active-learning` — Active learning operator correction loop (#137)

- **Priority:** P2 · **Machine:** both
- **Depends on:** none
- **Sources:** [137](https://github.com/wrbell/stark-translate/issues/137), `tools/merge_corrections.py`, `operator_app review UI`
- **Acceptance:** Operator corrects caption; approved pair exports to dated corrections corpus with provenance; retrain path documented.
- **Next action:** Complete one closed loop on a recorded Sunday after human approvals; merge_corrections dry run on WSL.

## Validated

### `issue-131-smoke` — Post-dormancy operator smoke (#131)

- **Priority:** P0 · **Machine:** mac
- **Depends on:** none
- **Sources:** [131](https://github.com/wrbell/stark-translate/issues/131)
- **Acceptance:** Operator UI loads; partial + final on audience display; factory prefers configured adapters.
- **Next action:** Close #131 after root merge confirms same checks on main; local controlled rehearsal passed without fabricating approvals.

### `mac-cpu-test-suite` — Final CPU test suite after VAD/CT2 setup changes

- **Priority:** P0 · **Machine:** mac
- **Depends on:** `mac-reliability-implementation`
- **Sources:** `docs/mac_implementation_status.md`, `.cache/mac-roadmap/full-tests-final.log`
- **Acceptance:** Documented pass/skip/coverage recorded in mac_implementation_status.md; gate in test.yml remains ≥50% coverage.
- **Next action:** Re-run after substantive code changes; avoid hardcoding counts in multiple guides—cite mac_implementation_status.md.

### `mac-defaults-frozen` — Retain Mac production defaults after 48-run screen

- **Priority:** P0 · **Machine:** mac
- **Depends on:** none
- **Sources:** `docs/evaluation/mac_v2026_14_screening/README.md`, `docs/mac_implementation_status.md`
- **Acceptance:** No combined experiment beats E4B + 0.5 s silence + 0.6 s cadence across both models; defaults unchanged.
- **Next action:** Keep experiments opt-in; require human review before any default change.

### `packaging-artifacts-local` — v2026.14 wheel/sdist/Mac ZIP local validation

- **Priority:** P1 · **Machine:** mac
- **Depends on:** `mac-reliability-implementation`
- **Sources:** `docs/evaluation/mac_v2026_14_installation.md`
- **Acceptance:** Outside-checkout install, launcher, and EN/ES runtime smoke documented with hashes.
- **Next action:** Publication pending; packaging detail owned by lite agent for cross-platform docs.

## Summary counts

| Status | Count |
|--------|------:|
| Pending Input Or Hardware | 17 |
| Experimental | 2 |
| Deferred | 5 |
| Implemented | 4 |
| Validated | 4 |
