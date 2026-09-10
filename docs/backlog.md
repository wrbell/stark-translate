# Remaining backlog — Stark Road Bilingual Speech-to-Text

> **Canonical machine-readable source:** [`backlog.json`](./backlog.json).
> Regenerate this file with `python tools/render_backlog.py render`.
> **Last updated:** 2026-09-09

## Integration status

- **Main release:** `v2026.13` — main at 09e4679; v2026.13 PRs #180–191 merged. main stays v2026.13 until the authorized final merge.
- **Local candidate:** `2026.14.0.0` on `codex/mac-reliability-roadmap` (base `5154fb9`)
- **Draft PR:** [PR #192](https://github.com/wrbell/stark-translate/pull/192) — OPEN DRAFT, base `main` ← `codex/mac-reliability-roadmap`; not merged. Root is integrating overnight operator, lite, latency and failure-recovery work before marking it ready.
- **Publication:** Source and issue publishing and the final merge to main are authorized by the user. PyPI publication, package artifacts and release tags remain pending by user choice.

Items marked implemented or validated exist on the local reliability branch (or in an overnight worktree awaiting integration) unless noted as main-only. Nothing below is released on main or PyPI until root supplies merge/tag evidence. Certification records whether the item's own acceptance was met; implementation alone does not close an issue.

## Status vocabulary

| Status | Meaning |
|--------|---------|
| `in_progress` | Active engineering right now; code may live uncommitted in an overnight worktree and is pending integration. |
| `pending_input_or_hardware` | Blocked on human input, approved references, or hardware/device access. |
| `experimental` | Opt-in path that stays off by default until a measured, reviewed gain exists. |
| `deferred` | Intentionally postponed; often a pending user decision. |
| `implemented` | Code exists on the local branch; the item's acceptance has not been certified with recorded evidence. |
| `validated` | Acceptance met with recorded local evidence (see mac_implementation_status.md and docs/evaluation). |

`certification` records whether the item's stated acceptance has been met (`met`, `pending`, `not_applicable`) independently of implementation status.

## Current Mac defaults

- EN STT: `parakeet-mlx (mlx-community/parakeet-tdt-0.6b-v3)` · ES STT: `mlx-whisper large-v3-turbo`
- Partials: `Marian CT2 int8 on CPU (adapters/marian_ct2 or managed cache; HF fallback)` · Finals: `Gemma 4 E4B OptiQ (mlx-community/gemma-4-e4b-it-OptiQ-4bit)`
- Silence `0.5` s · partial cadence `0.6` s · MTP `off (--mts never loads under mlx-lm 0.31.x; engines/mlx_spec.py probe only)`

See [`current_architecture.md`](./current_architecture.md) and [`mac_implementation_status.md`](./mac_implementation_status.md) for contracts and evidence.

## In Progress

### `caption-delivery-goal` — Sub-second median speech-end-to-caption delivery

- **Priority:** P0 · **Machine:** mac · **Certification:** pending
- **Depends on:** `visible-browser-timing-run`
- **Sources:** `docs/mac_implementation_status.md`, `docs/evaluation/README.md`, `docs/archive/v2026.13/MAC_LATENCY.md`
- **Acceptance:** Median schema 2 speech_end_to_final_ms under 1000 ms on the frozen real-time baseline with visible ACKs, without regressing final quality; natural-speech quality certification (references, bilingual review) is a separate gate and must not be pooled with historical cohorts.
- **Notes:** Active Mac engineering: the latency worktree committed opt-in bounded scheduling and caption-delivery instrumentation (codex/overnight-latency 1a8470c) and is drafting incremental STT/preview candidates; none of it is integrated or measured yet. Independent engineering experiments proceed on the frozen English screen without waiting for Spanish references; the 48-run screen rejected shorter silence and combined tweaks under the current pipeline.
- **Next action:** Integrate and measure the scheduling experiment on the frozen screen with a visible browser; keep every change opt-in until a matched gain is shown on both models.

### `issue-131-smoke` — Post-dormancy boot and operator smoke (#131)

- **Priority:** P0 · **Machine:** mac · **Certification:** pending
- **Depends on:** `mac-live-mic-stall`
- **Sources:** [#131](https://github.com/wrbell/stark-translate/issues/131)
- **Issue acceptance (verbatim intent):** Operator UI loads, one partial + one final appear on audience display, no uncaught errors in the session log — including one live mic utterance EN→ES and ES→EN and confirmation that the factory prefers the configured adapters.
- **Acceptance:** Live built-in-microphone EN→ES and ES→EN utterances render a partial and a final on the audience display with clean session logs; controlled file replay alone does not satisfy the issue.
- **Evidence:** Controlled file-replay EN and ES sessions rendered captions and exited 0; review draft recovery passed (2026-09-09, base 5154fb9).
- **Evidence:** Live microphone attempt stalled the same night (see mac-live-mic-stall).
- **Next action:** After the mic stall fix, run one live EN and one live ES utterance through the operator UI with the audience display connected and keep the session logs as evidence; root closes #131 only then.

### `mac-live-mic-stall` — Built-in microphone capture stalled; operator showed RUNNING without audio

- **Priority:** P0 · **Machine:** mac · **Certification:** pending
- **Depends on:** none
- **Sources:** `operator_app/pipeline_manager.py`, `tools/audio_bridge_client.py`, `docs/overnight_status.md`
- **Acceptance:** A live built-in-microphone session produces partial and final captions on the audience display; operator status reflects actual audio frames rather than the CSV header; a stalled input stream is detected and surfaced within seconds.
- **Evidence:** 2026-09-09 23:32 session 20260909_233204_799019_en (audio_source=mic): models loaded, 'Listening...' printed, audience page fetched, then no frames; session_lifecycle stayed status=running; partials file empty.
- **Evidence:** Operator UI derived RUNNING from the CSV header while the audience display stayed disconnected; a separate sounddevice sd.rec probe also stalled.
- **Evidence:** File-replay sessions 20260909_233546_027169_en and 20260909_233823_034893_es (audio_source=file) completed with exit 0 on the same build; they do not exercise the microphone path.
- **Notes:** User explicitly deferred live microphone and physical-device checks until tomorrow. The reliability worktree is drafting isolated capture/health helpers; nothing is integrated yet.
- **Next action:** Reproduce with CoreAudio permission and device enumeration logging, add a frame-arrival health check that fails Start when no audio arrives, then re-run live EN and ES microphone sessions.

### `pr-192-integration` — Integrate overnight work into draft PR #192 and finish the authorized merge

- **Priority:** P0 · **Machine:** any · **Certification:** pending
- **Depends on:** none
- **Sources:** [PR #192](https://github.com/wrbell/stark-translate/pull/192), `docs/overnight_status.md`
- **Acceptance:** PR #192 marked ready with integrated operator, lite, latency and failure-recovery changes, CI green, and root-recorded evidence; main advances from v2026.13 only at that merge.
- **Notes:** Root owns integration and GitHub delivery. Source/issue publishing and the final merge are authorized; PyPI, package artifacts and release tags stay pending.
- **Next action:** Root: reconcile the overnight worktrees (docs, lite, latency, operator-ui, reliability, issue-evidence) onto the PR branch and re-run the CPU suite before marking ready.

### `lite-cpu-inference` — Lite CPU inference profile (no GPU) — implementation

- **Priority:** P1 · **Machine:** any · **Certification:** pending
- **Depends on:** none
- **Sources:** `docs/packaging/pypi.md`, `docs/overnight_status.md`
- **Acceptance:** A documented CPU-only profile installs from the `cpu` extra, passes setup/preflight, and runs EN/ES file replay end to end on a machine without a GPU; smoke evidence recorded.
- **Notes:** The lite worktree has uncommitted profile, preflight, TTS engine and llama runtime changes plus setup/preflight edits; not integrated. Packaging documentation is owned by the lite agent. Implementation is distinct from certification on target hardware.
- **Next action:** Lite agent finishes and tests the profile; root integrates; then smoke on a CPU-only host.

### `overnight-latency-scheduling` — Opt-in bounded scheduling and caption delivery instrumentation

- **Priority:** P1 · **Machine:** mac · **Certification:** pending
- **Depends on:** none
- **Sources:** `docs/overnight_status.md`
- **Acceptance:** Merged into the PR branch behind opt-in flags with tests; a frozen-screen replay shows a matched delivery improvement on both models or the experiment is recorded as rejected.
- **Notes:** codex/overnight-latency commit 1a8470c adds tools/latency_scheduler.py, tools/caption_delivery.py, tools/latency_experiments.py, tools/latency_trace.py and tests; uncommitted incremental_stt.py / preview_candidates.py drafts exist. Not on the PR branch.
- **Next action:** Root integrates; then run tools/mac_evaluation.py experiments with a visible browser.

### `overnight-reliability` — Process supervision, work leases and isolated audio capture

- **Priority:** P1 · **Machine:** mac · **Certification:** pending
- **Depends on:** none
- **Sources:** `docs/overnight_status.md`
- **Acceptance:** Operator status derives from live pipeline health (frames, heartbeats) rather than file presence; stalled capture surfaces as a failure; tests cover the new modules.
- **Notes:** Uncommitted processes.py, work_lease.py, support.py, capture_worker.py, isolated_audio.py, pipeline_health.py, persistence.py, operational_logging.py in the reliability worktree. Directly relevant to mac-live-mic-stall; not integrated.
- **Next action:** Root integrates behind tests; verify against a real microphone tomorrow.

### `docs-refresh-remaining` — Documentation refresh — remaining areas after the overnight docs pass

- **Priority:** P2 · **Machine:** any · **Certification:** pending
- **Depends on:** `pr-192-integration`
- **Sources:** `docs/overnight_status.md`
- **Acceptance:** Every README/CLAUDE/AGENTS guide describes current behavior from source, historical numbers live only under dated archive links, and tests/test_documentation.py plus render/link checks pass.
- **Notes:** Areas the overnight docs pass did not finish are listed in docs/overnight_status.md § Unfinished; docs/operator_runbook.md and docs/packaging/* are owned elsewhere.
- **Next action:** Root audits; owners of the runbook and packaging docs refresh their files after integration.

### `overnight-operator-ui` — Operator UI caption and QR widgets

- **Priority:** P2 · **Machine:** mac · **Certification:** pending
- **Depends on:** none
- **Sources:** `docs/overnight_status.md`, `docs/operator_runbook.md`
- **Acceptance:** Widgets integrated, HTML5 Tidy clean, and the operator runbook updated with root-recorded UI evidence.
- **Notes:** Uncommitted captions.js / qr.js widgets and index/style edits in the operator-ui worktree. Runbook evidence is root-owned.
- **Next action:** Root integrates and refreshes docs/operator_runbook.md.

## Pending Input Or Hardware

### `issue-134-sunday-dry-run` — Dry-run with the operator runbook (#134)

- **Priority:** P0 · **Machine:** mac · **Certification:** pending
- **Depends on:** `issue-131-smoke`
- **Sources:** [#134](https://github.com/wrbell/stark-translate/issues/134), `docs/operator_runbook.md`
- **Issue acceptance (verbatim intent):** Walk the runbook on church hardware or a laptop stand-in; time setup → first caption; capture one full hymn plus one spoken segment; a written dry-run note exists (what worked, what broke, time-to-first-caption) and blocking UX holes have their own issues.
- **Acceptance:** Written dry-run note with setup-to-first-caption timing, one full hymn and one spoken segment captured through the live microphone, and follow-up issues filed for non-technical operator blockers. A laptop stand-in is acceptable; church hardware is not mandatory.
- **Notes:** The controlled hymn/pause/restart rehearsal used file replay and TTS; it does not provide the live-mic timing or the human walk-through the issue requires.
- **Next action:** Schedule a laptop stand-in walk-through once live mic capture works; record time-to-first-caption and file UX issues.

### `bilingual-blinded-review` — Blinded bilingual review of meaning errors and terminology

- **Priority:** P1 · **Machine:** mac · **Certification:** pending
- **Depends on:** none
- **Sources:** `docs/evaluation/mac_v2026_14_quality/comparison.md`, `docs/evaluation/README.md`
- **Acceptance:** A bilingual reviewer completes blind_review.jsonl for the E4B/E2B comparison; the E2B speed tradeoff is accepted or rejected on meaning and terminology, not canary counts alone.
- **Next action:** Hand the blind review form to a reviewer; keep review_key.jsonl separate. Do not change the default before this.

### `cuda-latency-proposal` — CUDA latency proposal execution on the A2000

- **Priority:** P1 · **Machine:** wsl · **Certification:** pending
- **Depends on:** none
- **Sources:** `docs/cuda_latency_proposal.md`, `scripts/cuda/bench_mtp.sh`, [#174](https://github.com/wrbell/stark-translate/issues/174)
- **Acceptance:** llama.cpp b10883 built, MTP opt-in bench, `-fa on` retest, W16 HF fp16 and Parakeet probes recorded with the proposal's tables filled in.
- **Notes:** start_server.sh now defaults to --no-draft with --mtp opt-in and both pins read b10883; the scripts remain header-marked unexecuted.
- **Next action:** Run scripts/cuda/*.sh on WSL.

### `issue-135-mac-ab` — Deploy W16 + v2-cpo to Mac and live A/B vs stock (#135)

- **Priority:** P1 · **Machine:** both · **Certification:** pending
- **Depends on:** none
- **Sources:** [#135](https://github.com/wrbell/stark-translate/issues/135), `docs/gemma4_tuning/v3_directions.md`, `docs/wsl_pipeline_refresh.md`
- **Issue acceptance (verbatim intent):** A short A/B note with canary scores and a ship decision. If no-ship, stock E4B stays default and this closes.
- **Acceptance:** W16 CT2 and v2-cpo artifacts copied to the Mac; live A/B (stock E4B vs v2-cpo; stock whisper-turbo vs W16) with the 8-canary health check and a written ship/no-ship note.
- **Notes:** Mac default EN STT is now Parakeet MLX; W16 is a faster-whisper CT2 artifact, so the STT half of this A/B runs the CPU faster-whisper path or compares against Parakeet explicitly. v2-cpo reached statistical parity with stock E4B on COMET-22 but still misses the Jacobo canary.
- **Next action:** Transfer artifacts from WSL; run tools/health_check.py --backend mlx --n-canaries 8 and a replay A/B.

### `natural-spanish-refs` — Human-reviewed natural Spanish references

- **Priority:** P1 · **Machine:** mac · **Certification:** pending
- **Depends on:** none
- **Sources:** `docs/mac_implementation_status.md`, `docs/evaluation/README.md`
- **Acceptance:** At least 50 approved natural utterances per language; no WER or natural-audio acceptance claim before approval.
- **Notes:** 50 English and 11 Spanish candidates exist, all unapproved. The user has no natural Spanish recording location yet; the synthetic Piper Spanish clip stays separate.
- **Next action:** User supplies a natural Spanish source; then annotate via tools/mac_evaluation.py annotate.

### `natural-two-speaker` — Natural two-speaker audio with human transition labels

- **Priority:** P1 · **Machine:** mac · **Certification:** pending
- **Depends on:** none
- **Sources:** `docs/mac_implementation_status.md`
- **Acceptance:** A natural church clip with two speakers and human-labeled speaker transitions available for the diarization gate and the service rehearsal.
- **Next action:** Record or obtain a two-speaker segment; synthetic routing probes do not substitute.

### `physical-second-output` — Second physical audio output: selection, unplug/replug, audible playback

- **Priority:** P1 · **Machine:** mac · **Certification:** pending
- **Depends on:** `issue-132-tts-routing`
- **Sources:** `docs/mac_implementation_status.md`
- **Acceptance:** EN and ES TTS routed to two distinct real outputs, device choice persists across restart, unplug/replug re-resolves the named route, and playback is audibly verified.
- **Notes:** User deferred physical-device checks until tomorrow.
- **Next action:** Exercise a USB/interface output plus the built-in speaker on the Mac; record the session and device list.

### `rtx2070-native-validation` — Native Windows / RTX 2070 inference — certification on hardware

- **Priority:** P1 · **Machine:** windows · **Certification:** pending
- **Depends on:** `lite-cpu-inference`
- **Sources:** `docs/archive/research/rtx2070_feasibility.md`, `packaging/windows/README.md`
- **Acceptance:** Setup, preflight and an EN/ES file replay complete on a 2070-class Windows machine with llama.cpp E2B/E4B tier selection recorded; MSI or source install path documented.
- **Notes:** The v2026.13 MSI digest and embedded version were verified without Windows execution.
- **Next action:** Run on the target hardware after the lite profile lands.

### `visible-browser-timing-run` — Frozen timing run with a visible browser (non-zero ACK coverage)

- **Priority:** P1 · **Machine:** mac · **Certification:** pending
- **Depends on:** none
- **Sources:** `docs/evaluation/mac_v2026_14_screening/README.md`, `docs/evaluation/mac_v2026_14_routing/README.md`
- **Acceptance:** Replay on the unlocked Mac with the audience display connected records schema 2 speech_end_to_final_ms and non-zero visible final ACK coverage for the acceptance configuration.
- **Notes:** The 48-run screen and 24 routing probes recorded 0/348 and 0/72 visible ACKs; the controlled operator rehearsal is separate evidence.
- **Next action:** Re-run the baseline replay with a visible audience tab and keep the display_metrics JSONL.

### `wsl-e4b-domain-sft` — Gemma 4 E4B domain SFT → GGUF on WSL

- **Priority:** P1 · **Machine:** wsl · **Certification:** pending
- **Depends on:** `wsl-phase4`
- **Sources:** `docs/wsl_pipeline_refresh.md`, `training/run_gemma4_e4b_domain_sft.sh`, `docs/gemma4_tuning/overview.md`
- **Acceptance:** 8-canary sanity on the exported GGUF passes and the artifact is ready for Mac/CUDA transfer.
- **Notes:** training/train_gemma4.py and export_gguf.py carry UNTESTED headers pending their first end-to-end run.
- **Next action:** Run after Phase 4, or in parallel if the corpus is ready.

### `wsl-phase4` — Phase 4 full audio preprocessing on WSL

- **Priority:** P1 · **Machine:** wsl · **Certification:** pending
- **Depends on:** none
- **Sources:** `docs/wsl_pipeline_refresh.md`, `training/run_phase4_preprocess.sh`
- **Acceptance:** `phase4_status.json` written for the sermon corpus on WSL storage.
- **Next action:** Execute runbook §1 on the A2000 box.

### `wsl-w17-export` — W17 Whisper DoRA + hard-mix → CT2

- **Priority:** P1 · **Machine:** wsl · **Certification:** pending
- **Depends on:** `wsl-phase4`
- **Sources:** `docs/wsl_pipeline_refresh.md`, `training/run_w17_curriculum.sh`
- **Acceptance:** tools/benchmark_stt_engines.py gate: W17 fresh-eval WER ≤ W16.
- **Next action:** Scripted; not yet trained.

### `issue-136-jacobo-cpo` — Jacobo canary — targeted preference triples (#136)

- **Priority:** P2 · **Machine:** wsl · **Certification:** pending
- **Depends on:** none
- **Sources:** [#136](https://github.com/wrbell/stark-translate/issues/136), `docs/gemma4_tuning/v3_directions.md`, `tools/build_preference_triples.py`
- **Issue acceptance (verbatim intent):** Jacobo canary passes (or we document why it cannot) and COMET-22 does not regress vs v2-cpo.
- **Acceptance:** 50–100 hand-crafted preference triples, one CPO continue from v2 (`training/train_gemma4_cpo.py --init-adapter`), re-scored 8-canary set and 500-verse holdout, Jacobo passing or a documented reason.
- **Notes:** v3_directions Tier 1 also proposes few-shot disambiguation in the production prompt as a cheaper first step; the opt-in `--terminology-prompt church` examples are the Mac analogue and remain opt-in after the screen.
- **Next action:** Author triples on WSL; run one CPO continue; re-score.

### `pypi-publication` — PyPI trusted publisher, package artifacts and release tag

- **Priority:** P2 · **Machine:** any · **Certification:** pending
- **Depends on:** `pr-192-integration`
- **Sources:** `docs/mac_implementation_status.md`, `.github/workflows/pypi.yml`
- **Acceptance:** Trusted publisher mapped (owner wrbell, repo stark-translate, workflow pypi.yml, environment pypi), a new version tag pushed, and the PyPI workflow green.
- **Notes:** Explicitly pending by user choice; only source/issue publishing and the final merge are authorized.
- **Next action:** User decision after the merge.

## Experimental

### `conservative-marian-routing` — Conservative Marian partial routing

- **Priority:** P3 · **Machine:** mac · **Certification:** pending
- **Depends on:** `natural-spanish-refs`
- **Sources:** `docs/evaluation/mac_v2026_14_routing/README.md`, `settings.py`
- **Acceptance:** Natural Spanish references plus human review show no quality loss when Marian handles allowlisted phrases; until then `--routing-policy conservative` stays opt-in.
- **Notes:** 24/24 synthetic EN/ES routing probes exited zero and routed as designed; they are functional checks, not quality evidence.
- **Next action:** Keep opt-in.

### `issue-177-mtp` — Gemma 4 MTP assistant drafter on MLX (#177)

- **Priority:** P3 · **Machine:** mac · **Certification:** pending
- **Depends on:** none
- **Sources:** [#177](https://github.com/wrbell/stark-translate/issues/177), `docs/archive/v2026.13/MAC_LATENCY.md`, `docs/mlx_mtp_notes.md`, `engines/mlx_spec.py`, `tools/mts_acceptance_probe.py`
- **Acceptance:** Acceptance rate and latency win on the frozen screen justify an opt-in default; output stays byte-identical to the target model.
- **Notes:** `--mts` still routes through mlx_lm and never loads (no gemma4_assistant class). The engines/mlx_spec.py probe over mlx-optiq produced byte-identical output at low acceptance with a bounded latency win; the RoPE-offset hypothesis was rejected. Numbers: see the archive link.
- **Next action:** Keep off; revisit only if upstream mlx-lm adds the assistant model class or acceptance improves.

## Deferred

### `security-b615-pinning` — Remaining unpinned Hugging Face download paths (B615)

- **Priority:** P2 · **Machine:** both · **Certification:** pending
- **Depends on:** none
- **Sources:** `docs/evaluation/mac_v2026_14_security.md`
- **Acceptance:** Optional/fallback HF paths pin revisions or are documented as operator-only, starting with live-path fallbacks (Marian HF, Piper missing-voice).
- **Notes:** The configured CI Bandit pass skips B615; the expanded scan's findings and scope limits are documented in the security note.
- **Next action:** Pin the live-path fallbacks first.

### `issue-138-hindi-zero-shot` — Hindi zero-shot baseline on church audio (#138)

- **Priority:** P3 · **Machine:** mac · **Certification:** pending
- **Depends on:** none
- **Sources:** [#138](https://github.com/wrbell/stark-translate/issues/138), `docs/evaluation/mac_v2026_14_hindi/README.md`, `docs/archive/research/multi_lingual.md`
- **Issue acceptance (verbatim intent):** A short baseline note: does zero-shot Hindi even work on church audio, and is a QLoRA week worth it this semester — via target_lang_code="hi" through the live pipeline, noting SOV partial garble, with 8 canaries and a few verse pairs.
- **Acceptance:** Hindi target through the live pipeline on church audio with a written baseline note and QLoRA go/no-go; the existing offline text-only probe does not satisfy this.
- **Notes:** Offline text Hindi generation exists for 43 English inputs on both models with no references or live integration. Hindi/Chinese timing remains a pending user decision.
- **Next action:** None until the user schedules Hindi; keep the offline probe labeled as such.

### `macos-shortcuts` — macOS Shortcuts voice-command triggers

- **Priority:** P3 · **Machine:** mac · **Certification:** not applicable
- **Depends on:** none
- **Sources:** `docs/roadmap.md`
- **Acceptance:** Optional convenience; documented only if adopted.
- **Next action:** None scheduled.

### `multilingual-expansion` — Hindi and Chinese adaptation (roadmap Phase 8)

- **Priority:** P3 · **Machine:** both · **Certification:** pending
- **Depends on:** `issue-138-hindi-zero-shot`
- **Sources:** `docs/roadmap.md`, `docs/archive/research/multi_lingual.md`
- **Acceptance:** Per the roadmap: chrF++/COMET gates, adapter switching and display updates per language.
- **Notes:** Pending user decision; EN/ES gates come first.
- **Next action:** None scheduled.

### `v2026-9-followups` — llama.cpp deferred optimizations (-fa, -c 2048, prompt-cache reuse)

- **Priority:** P3 · **Machine:** wsl · **Certification:** pending
- **Depends on:** `cuda-latency-proposal`
- **Sources:** `docs/archive/v2026.9/GEMMA_OPTIM_PHASE2.md`, [#175](https://github.com/wrbell/stark-translate/issues/175)
- **Acceptance:** Each experiment meets the proposal's latency gate without canary regression.
- **Next action:** Folded into the CUDA proposal.

## Implemented

### `mac-reliability-implementation` — v2026.14 Mac reliability program on the local branch

- **Priority:** P0 · **Machine:** mac · **Certification:** pending
- **Depends on:** none
- **Sources:** `docs/mac_implementation_status.md`, [PR #192](https://github.com/wrbell/stark-translate/pull/192)
- **Acceptance:** Merged to main through PR #192 with the human, device and visible-browser gates recorded as still open.
- **Notes:** Operator session identity, schema 2 timing, setup/resolver, Review/export, opt-in experiments and packaging guards are implemented and locally validated; integration is the remaining step.
- **Next action:** Root: finish PR #192.

### `issue-132-tts-routing` — Multi-channel TTS routing code and tests (9.4.1 / #132)

- **Priority:** P1 · **Machine:** mac · **Certification:** pending
- **Depends on:** none
- **Sources:** [#132](https://github.com/wrbell/stark-translate/issues/132), `tests/test_tts_multichannel.py`, `tests/test_phase9_4_1_tts_device.py`, `settings.py`
- **Issue acceptance (verbatim intent):** Operator can send TTS to a chosen output device; EN and ES can be routed independently. Tests cover the new engine path.
- **Acceptance:** Hardware-independent part: per-language output map (`--tts-device-en/es`, `STARK_TTS_OUTPUT_DEVICES`), operator persistence and hotplug retry covered by tests. Physical part: an operator routes EN and ES to two real outputs and hears each — tracked as `physical-second-output`.
- **Evidence:** tests/test_tts_multichannel.py and tests/test_phase9_4_1_tts_device.py pass in the recorded CPU suite (device enumeration mocked).
- **Evidence:** Built-in MacBook speaker playback calls completed in the controlled rehearsal; no second physical device was used.
- **Next action:** Keep #132 open until physical-second-output passes; then root closes both together.

### `issue-133-diarize-gate` — Live diarization on a rolling buffer (9.6.1 / #133)

- **Priority:** P1 · **Machine:** mac · **Certification:** pending
- **Depends on:** `natural-two-speaker`
- **Sources:** [#133](https://github.com/wrbell/stark-translate/issues/133), `docs/live_diarization.md`, `features/live_diarize.py`, `features/speaker_labels.py`
- **Issue acceptance (verbatim intent):** Two-speaker dry-run shows distinct speaker labels on finals without blowing p95 caption latency. Offline path still works.
- **Acceptance:** With `--diarize`, a natural two-speaker clip yields distinct speaker labels on finals, final p95 stays within +50 ms of `--diarize` off (tools/replay_bench.py), and the offline pyannote path still runs.
- **Notes:** Code is in: rolling buffer, separate daemon (`embed` or `pyannote` mode), speaker on finals/CSV/JSONL/WebSocket. The gate has not been run; it needs HF_TOKEN or SpeechBrain ECAPA plus labeled two-speaker audio.
- **Next action:** Run the replay_bench gate on a two-speaker clip once one exists; keep default off.

### `issue-137-active-learning` — Active learning — low-confidence to operator correction (#137)

- **Priority:** P2 · **Machine:** both · **Certification:** pending
- **Depends on:** none
- **Sources:** [#137](https://github.com/wrbell/stark-translate/issues/137), `operator_app/review.py`, `tools/review_data.py`, `tools/merge_corrections.py`, `tests/test_operator_review.py`, `tests/test_correction_import_safety.py`
- **Issue acceptance (verbatim intent):** An operator can correct a caption and that pair lands in a dated corrections corpus. Retrain script documented even if the first retrain is a dry run.
- **Acceptance:** A human operator corrects and approves at least one real caption from a recorded session, the approved pair exports to a dated corrections corpus with provenance, and tools/merge_corrections.py is run (dry run acceptable) on that export.
- **Notes:** Live/post-session Review, independent transcript/translation approval, revisioned drafts, portable bundles and evaluation/training separation are implemented and tested with fixtures. No human approval has been recorded; fixtures are not approved correction data.
- **Next action:** After a live-mic session, have the operator approve one real correction, export it, and run `tools/merge_corrections.py translation` (and `whisper`) against a scratch copy of the training JSONL; keep the export and merged output as evidence.

### `issue-176-multiprocess` — --multiprocess workers use shared Gemma 4 prompts and stop rules (#176)

- **Priority:** P3 · **Machine:** mac · **Certification:** pending
- **Depends on:** none
- **Sources:** [#176](https://github.com/wrbell/stark-translate/issues/176), `workers.py`, `engines/translation_prompts.py`
- **Issue acceptance (verbatim intent):** Route workers.py through engines/translation_prompts build_chat_messages + ensure_stop_tokens (skipping the prompt cache for gemma4), or deprecate --multiprocess.
- **Acceptance:** workers.translation_worker_main builds engines through MLXGemmaEngine with the parent-selected model_family, so Gemma 4 never receives TranslateGemma prompts and stop handling is shared; a live --multiprocess run is not part of the acceptance.
- **Evidence:** workers.py at base 5154fb9: translation_worker_main wraps MLXGemmaEngine(model_family=...) and only attaches the TG draft path when model_family == 'translategemma'.
- **Notes:** Escape hatch only; the in-process max_workers=2 overlap remains the production path.
- **Next action:** Root closes #176 after the PR merge, citing workers.py.

## Validated

### `mac-cpu-test-suite` — Final CPU test suite after VAD/CT2 setup changes

- **Priority:** P0 · **Machine:** mac · **Certification:** met
- **Depends on:** none
- **Sources:** `docs/mac_implementation_status.md`, `.github/workflows/test.yml`
- **Acceptance:** Recorded pass/skip/coverage in mac_implementation_status.md above the test.yml coverage gate.
- **Notes:** Counts are recorded only in mac_implementation_status.md; guides link there instead of repeating numbers.
- **Next action:** Re-run after integrating the overnight worktrees.

### `mac-defaults-frozen` — Retain Mac defaults after the 48-run English screen

- **Priority:** P0 · **Machine:** mac · **Certification:** met
- **Depends on:** none
- **Sources:** `docs/evaluation/mac_v2026_14_screening/README.md`, `docs/mac_implementation_status.md`
- **Acceptance:** No combined configuration beat E4B + 0.5 s silence + 0.6 s cadence across both models on the frozen 45-second English screen; all experiments remain opt-in.
- **Evidence:** 48/48 runs exited zero across eight configurations × two models × three pairs; no consistent both-model winner (screening README).
- **Next action:** Any default change requires a matched both-model gain plus human review.

### `packaging-artifacts-local` — v2026.14 wheel, sdist and Mac ZIP local validation

- **Priority:** P1 · **Machine:** mac · **Certification:** met
- **Depends on:** none
- **Sources:** `docs/evaluation/mac_v2026_14_installation.md`
- **Acceptance:** Outside-checkout install, launcher checks and installed EN/ES inference recorded with artifact hashes.
- **Notes:** Publication is pending; overnight integration will change the artifact and require a rebuild.
- **Next action:** Rebuild after PR #192 integration; lite agent owns cross-platform packaging prose.

## Summary counts

| Status | Count |
|--------|------:|
| In Progress | 9 |
| Pending Input Or Hardware | 14 |
| Experimental | 2 |
| Deferred | 5 |
| Implemented | 5 |
| Validated | 3 |
