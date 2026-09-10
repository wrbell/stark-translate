# Remaining backlog — Stark Road Bilingual Speech-to-Text

> **Canonical machine-readable source:** [`backlog.json`](./backlog.json).
> Regenerate this file with `python tools/render_backlog.py render`.
> **Last updated:** 2026-09-10

## Integration status

- **Main release:** `v2026.13` — main at 09e4679; v2026.13 PRs #180–191 merged. main stays v2026.13 until the authorized final merge.
- **Local candidate:** `2026.14.0.0` on `codex/mac-reliability-roadmap` (base `5154fb9; subsequent integrated runtime, UI, Lite, security and documentation commits are tracked by PR #192.`)
- **Draft PR:** [PR #192](https://github.com/wrbell/stark-translate/pull/192) — open draft, not merged; final validation and source merge are in progress.
- **Publication:** Source and issue publishing and the final merge to main are authorized by the user. PyPI publication, package artifacts and release tags remain pending by user choice.

Items marked implemented or validated exist on the local reliability branch (all overnight worktrees are integrated as of 2026-09-10) unless noted as main-only. Nothing below is released on main or PyPI until the parent supplies merge/tag evidence. Certification records whether the item's own acceptance was met; implementation alone does not close an issue.

## Status vocabulary

| Status | Meaning |
|--------|---------|
| `in_progress` | Active engineering right now; code may still be changing on the branch and its acceptance is not yet met. |
| `pending_input_or_hardware` | Blocked on human input, approved references, or hardware/device access. |
| `experimental` | Opt-in path that stays off by default until a measured, reviewed gain exists. |
| `deferred` | Intentionally postponed; often a pending user decision. |
| `implemented` | Code exists on the local branch; the item's acceptance has not been certified with recorded evidence. |
| `validated` | Acceptance met with recorded local evidence (see mac_implementation_status.md and docs/evaluation). |

`certification` records whether the item's stated acceptance has been met (`met`, `pending`, `not_applicable`) independently of implementation status.

## Current Mac defaults

- EN STT: `parakeet-mlx (mlx-community/parakeet-tdt-0.6b-v3)` · ES STT: `mlx-whisper large-v3-turbo`
- Partials: `Marian CT2 int8 on CPU (adapters/marian_ct2 or managed cache; HF fallback)` · Finals: `Gemma 4 E4B OptiQ (mlx-community/gemma-4-e4b-it-OptiQ-4bit)`
- Silence `0.5` s · partial cadence `0.6` s · MTP `off (#177); a live --mts / STARK_TRANSLATE_MLX_MTS request is rejected before any model loads (dry_run_ab.validate_live_mts); engines/mlx_spec.py offline probe only`

See [`current_architecture.md`](./current_architecture.md) and [`mac_implementation_status.md`](./mac_implementation_status.md) for contracts and evidence.

## In Progress

### `caption-delivery-goal` — Sub-second median speech-end-to-caption delivery

- **Priority:** P0 · **Machine:** mac · **Certification:** pending
- **Depends on:** `visible-browser-timing-run`
- **Sources:** `docs/mac_implementation_status.md`, `docs/evaluation/README.md`, `docs/archive/v2026.13/MAC_LATENCY.md`
- **Acceptance:** Median schema 2 speech_end_to_final_ms under 1000 ms on the frozen real-time baseline with visible ACKs, without regressing final quality; natural-speech quality certification (references, bilingual review) is a separate gate and must not be pooled with historical cohorts.
- **Notes:** Active Mac engineering: opt-in latency experiments, bounded scheduling and caption-delivery instrumentation are integrated on the candidate branch (overnight-latency-scheduling) but not yet measured with a visible browser. Independent engineering experiments proceed on the frozen English screen without waiting for Spanish references; the 48-run screen rejected shorter silence and combined tweaks under the current pipeline.
- **Next action:** Measure the integrated experiments on the frozen screen with a visible browser (tools/overnight_bench.py); keep every change opt-in until a matched gain is shown on both models.

### `issue-134-sunday-dry-run` — Dry-run with the operator runbook (#134)

- **Priority:** P0 · **Machine:** mac · **Certification:** pending
- **Depends on:** none
- **Sources:** [#134](https://github.com/wrbell/stark-translate/issues/134), `docs/operator_runbook.md`
- **Issue acceptance (verbatim intent):** Walk the runbook on church hardware or a laptop stand-in; time setup → first caption; capture one full hymn plus one spoken segment; a written dry-run note exists (what worked, what broke, time-to-first-caption) and blocking UX holes have their own issues.
- **Acceptance:** Per the issue text: walk the runbook on church hardware or a laptop stand-in (stand-in explicitly permitted); time setup → first caption; capture one full hymn plus one spoken segment; write the dry-run note (what worked, what broke, time-to-first-caption) and file follow-up issues for blocking UX holes. The issue does not add a live-microphone or human-walkthrough requirement beyond that text.
- **Notes:** The laptop-stand-in recorded rehearsal is independent of the deferred live-microphone gate. Earlier short mixed clips did not include a full hymn; a complete service replay and setup-to-first-caption note remain required.
- **Next action:** Complete the laptop-stand-in full recorded hymn/spoken rehearsal, record setup-to-first-caption timing, and publish the UX note. Live microphone is a separate tomorrow gate.

### `pr-192-integration` — Validate the integrated candidate branch and finish the authorized merge of draft PR #192

- **Priority:** P0 · **Machine:** any · **Certification:** pending
- **Depends on:** none
- **Sources:** [PR #192](https://github.com/wrbell/stark-translate/pull/192), `docs/overnight_status.md`
- **Acceptance:** PR #192 marked ready with integrated operator, lite, latency and failure-recovery changes, CI green, and root-recorded evidence; main advances from v2026.13 only at that merge.
- **Notes:** All overnight worktrees (docs, lite, latency, operator-ui, reliability, issue-evidence) are integrated on codex/mac-reliability-roadmap as of 2026-09-10; the parent is validating and collecting operator, benchmark, installation and release evidence. Source/issue publishing and the final merge are authorized; PyPI, package artifacts and release tags stay pending.
- **Next action:** Parent: finish validation and evidence refresh, run the CPU suite, mark PR #192 ready and merge; then tag/publish only when the user approves.

### `packaging-artifacts-local` — v2026.14 wheel, sdist and Mac ZIP local validation

- **Priority:** P1 · **Machine:** mac · **Certification:** pending
- **Depends on:** none
- **Sources:** `docs/evaluation/mac_v2026_14_installation.md`
- **Acceptance:** Outside-checkout install, launcher checks and installed EN/ES inference recorded with artifact hashes.
- **Notes:** Earlier artifact/source hashes remain valid historical evidence, but the integrated overnight candidate requires a new build and installed validation.
- **Next action:** Build and verify final wheel/sdist/Mac ZIP after runtime integration, install outside checkout and record hashes and EN/ES inference separately.

### `visible-browser-timing-run` — Frozen timing run with a visible browser (non-zero ACK coverage)

- **Priority:** P1 · **Machine:** mac · **Certification:** pending
- **Depends on:** none
- **Sources:** `docs/evaluation/mac_v2026_14_screening/README.md`, `docs/evaluation/mac_v2026_14_routing/README.md`
- **Acceptance:** Replay on the unlocked Mac with the audience display connected records schema 2 speech_end_to_final_ms and non-zero visible final ACK coverage for the acceptance configuration.
- **Notes:** The 48-run screen and 24 routing probes recorded 0/348 and 0/72 visible ACKs; the controlled operator rehearsal is separate evidence.
- **Next action:** Complete the frozen 96-run visible-audience matrix and retain actual ACKs/coverage; use a separate report cohort from earlier no-browser screens.

### `docs-refresh-remaining` — Documentation refresh — remaining areas after the overnight docs pass

- **Priority:** P2 · **Machine:** any · **Certification:** pending
- **Depends on:** `pr-192-integration`
- **Sources:** `docs/overnight_status.md`
- **Acceptance:** Every README/CLAUDE/AGENTS guide describes current behavior from source, historical numbers live only under dated archive links, and tests/test_documentation.py plus render/link checks pass.
- **Notes:** All root/subdirectory guides, platform runbooks, Mac refresh and Windows packaging status are refreshed. Final benchmark/endurance/artifact/CI evidence and merge/issue status remain to update.
- **Next action:** Finish evidence and PR/main-state refresh, then run render/link/documentation checks.

## Pending Input Or Hardware

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
- **Acceptance:** Copy W16 CT2 and v2-cpo artifacts to the Mac; perform the original live stock/adapted STT and translation A/B, at least the issue's five-canary health check, terminology audit and a written ship/no-ship note. The current eight/eighteen-item sets are additional coverage, not a rewritten original criterion.
- **Notes:** Mac default EN STT is now Parakeet MLX; W16 is a faster-whisper CT2 artifact, so the STT half of this A/B runs the CPU faster-whisper path or compares against Parakeet explicitly. v2-cpo reached statistical parity with stock E4B on COMET-22 but still misses the Jacobo canary.
- **Next action:** Transfer artifacts from WSL; run tools/health_check.py --backend mlx --n-canaries 8 and a replay A/B.

### `mac-torch-security-migration` — Resolve pinned Mac Torch dependency advisories

- **Priority:** P1 · **Machine:** mac · **Certification:** pending
- **Depends on:** none
- **Sources:** `docs/evaluation/overnight_security/README.md`, `pyproject.toml`
- **Acceptance:** A compatible patched Mac Torch/audio dependency set passes installed imports, VAD and real EN/ES inference, with an explicit full installed audit result. No incompatible forced install or changed frozen benchmark environment.
- **Notes:** Fresh isolated Mac audit retains two Torch 2.10 findings. A Torch 2.13 upgrade attempt failed before mutation because a matching Mac torchaudio 2.13 wheel was unavailable. Working stt_env is preserved.
- **Next action:** Validate a supported patched Torch/audio pairing or isolate optional diarization when available; keep the current audit limitation explicit.

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
- **Depends on:** none
- **Sources:** `docs/mac_implementation_status.md`
- **Acceptance:** EN and ES TTS routed to two distinct real outputs, device choice persists across restart, unplug/replug re-resolves the named route, and playback is audibly verified.
- **Notes:** User deferred physical-device checks until tomorrow.
- **Next action:** Exercise a USB/interface output plus the built-in speaker on the Mac; record the session and device list.

### `rtx2070-native-validation` — Native Windows / RTX 2070 inference — certification on hardware

- **Priority:** P1 · **Machine:** windows · **Certification:** pending
- **Depends on:** `lite-cpu-inference`
- **Sources:** `stark_translate/profiles.py`, `docs/lite_profiles.md`, `docs/archive/research/rtx2070_feasibility.md`, `packaging/windows/README.md`
- **Acceptance:** Setup, preflight and an EN/ES replay complete on a 2070-class native Windows machine with the lite-cuda-8gb profile (Whisper turbo CT2 int8_float16 + Marian + Gemma 4 E2B via CUDA llama-server), tier selection recorded; MSI or source install path documented.
- **Notes:** lite-cuda-8gb is implemented with pinned Windows CUDA 12.4 llama.cpp archives (tools/llama_runtime.py, b10883) and an sm_75 native build option (setup --build-native, Linux); the v2026.13 MSI digest was verified without Windows execution; the MSI remains a scaffold plan. Nothing has run on a 2070 or native Windows.
- **Next action:** Run setup/doctor/operator with --profile lite-cuda-8gb on the target hardware; record results in docs/lite_profiles.md.

### `windows-msi-bootstrap` — Prove and repair native Windows MSI first-launch bootstrap

- **Priority:** P1 · **Machine:** windows · **Certification:** pending
- **Depends on:** `pypi-publication`
- **Sources:** `docs/packaging/windows.md`, `packaging/windows/README.md`, `.github/workflows/release-win.yml`
- **Acceptance:** A clean Windows account installs the MSI, obtains matching runtime/profile dependencies and models, starts the operator from Start Menu, produces EN/ES captions and relaunches offline; uninstall and signing state are recorded.
- **Notes:** Current MSI bytes/ProductVersion were inspected on Mac. Reference PyApp TOML is not consumed by workflow; automatic extras/args/updater behavior remains unverified.
- **Next action:** Execute the bootstrap chain on Windows after a matching package is available; fix and recheck any entry-point/profile wiring failures.

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
- **Sources:** [#177](https://github.com/wrbell/stark-translate/issues/177), `docs/archive/v2026.13/MAC_LATENCY.md`, `docs/mlx_mtp_notes.md`, `engines/mlx_spec.py`, `tools/mts_acceptance_probe.py`, `dry_run_ab.py`
- **Acceptance:** Acceptance rate and latency win on the frozen screen justify an opt-in default; output stays byte-identical to the target model.
- **Notes:** Live `--mts` (or STARK_TRANSLATE_MLX_MTS) is rejected by dry_run_ab.validate_live_mts before any model loads — the flag never routes to mlx_lm anymore; `--no-mts` is the explicit off. The engines/mlx_spec.py offline probe over mlx-optiq produced byte-identical output at low acceptance with a bounded latency win; the RoPE-offset hypothesis was rejected. Numbers: see the archive link.
- **Next action:** Keep off; revisit only if upstream mlx-lm adds the assistant model class or acceptance improves.

## Deferred

### `security-b615-pinning` — Remaining unpinned Hugging Face download paths (B615)

- **Priority:** P2 · **Machine:** both · **Certification:** pending
- **Depends on:** none
- **Sources:** `docs/evaluation/mac_v2026_14_security.md`
- **Acceptance:** Optional/fallback HF paths pin revisions or are documented as operator-only, starting with live-path fallbacks (Marian HF, Piper missing-voice).
- **Notes:** The configured CI Bandit pass skips B615; the expanded scan's findings and scope limits are documented in the security note.
- **Next action:** Pin the live-path fallbacks first.

### `wsl-training-recipe-checks` — Repair unexecuted W17 projection and domain-corpus recipe assumptions

- **Priority:** P2 · **Machine:** wsl · **Certification:** pending
- **Depends on:** `wsl-phase4`
- **Sources:** `training/run_w17_curriculum.sh`, `training/run_gemma4_e4b_domain_sft.sh`, `training/CLAUDE.md`
- **Acceptance:** W17 uses real Whisper projection module names and compatible initialization shape/rank; domain SFT resolves the intended versioned corpus explicitly and rejects missing configured inputs before training. A dry-run command/data inspection is recorded on WSL.
- **Notes:** Current W17 shell names o_proj whereas Whisper uses out_proj. Domain SFT recipe can select legacy default verse corpus; guides currently require explicit reviewed paths. No WSL execution occurred on Mac.
- **Next action:** Correct and inspect the recipes alongside the real W16 adapter and prepared WSL corpus before the next training run.

### `issue-138-hindi-zero-shot` — Hindi zero-shot baseline on church audio (#138)

- **Priority:** P3 · **Machine:** mac · **Certification:** pending
- **Depends on:** none
- **Sources:** [#138](https://github.com/wrbell/stark-translate/issues/138), `tools/offline_hindi.py`, `docs/evaluation/mac_v2026_14_hindi/README.md`, `docs/archive/research/multi_lingual.md`, `docs/evaluation/overnight_hindi/README.md`
- **Issue acceptance (verbatim intent):** A short baseline note: does zero-shot Hindi even work on church audio, and is a QLoRA week worth it this semester — via target_lang_code="hi" through the live pipeline, noting SOV partial garble, with 8 canaries and a few verse pairs.
- **Acceptance:** Written baseline note on church audio (does zero-shot Hindi work; is a QLoRA week worth it) with a QLoRA go/no-go. The issue asks for the live pipeline; the offline audio tool provides the baseline measurement only — a live Hindi target is not integrated.
- **Notes:** The separate offline church-audio R&D baseline is complete on E4B/E2B with every output archived. There are no Hindi references or human quality review and no live Hindi integration. User reaffirmed EN↔ES as the speed priority; no further Hindi work is scheduled.
- **Next action:** Await a later R&D decision and human Hindi review; keep live Hindi integration and QLoRA separate from EN↔ES optimization.

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

### `issue-131-smoke` — Post-dormancy boot and operator smoke (#131)

- **Priority:** P0 · **Machine:** mac · **Certification:** pending
- **Depends on:** `mac-live-mic-stall`
- **Sources:** [#131](https://github.com/wrbell/stark-translate/issues/131)
- **Issue acceptance (verbatim intent):** Operator UI loads, one partial + one final appear on audience display, no uncaught errors in the session log — including one live mic utterance EN→ES and ES→EN and confirmation that the factory prefers the configured adapters.
- **Acceptance:** Live built-in-microphone EN→ES and ES→EN utterances render a partial and a final on the audience display with clean session logs; controlled file replay alone does not satisfy the issue.
- **Evidence:** Controlled file-replay EN and ES sessions rendered captions and exited 0; review draft recovery passed (2026-09-09, base 5154fb9).
- **Evidence:** Live microphone attempt stalled the same night (see mac-live-mic-stall); the capture/readiness fix is integrated but untested against a real microphone.
- **Notes:** The original issue explicitly asks to confirm W16 CT2 preference. The factory retains adapter preference for the configured faster-whisper path; Mac English auto now selects Parakeet by deliberate policy. Document this distinction during the live EN/ES retest instead of claiming W16 is the Mac auto default.
- **Next action:** Run one live EN and one live ES built-in-mic utterance through the operator UI with the audience display connected and keep the session logs as evidence; the issue requires live mic — file replay does not close it.

### `mac-live-mic-stall` — Built-in microphone capture stalled; operator showed RUNNING without audio

- **Priority:** P0 · **Machine:** mac · **Certification:** pending
- **Depends on:** none
- **Sources:** `tools/isolated_audio.py`, `tools/capture_worker.py`, `tools/pipeline_health.py`, `operator_app/pipeline_manager.py`, `docs/current_architecture.md`
- **Acceptance:** A live built-in-microphone session produces partial and final captions on the audience display; operator status reflects actual audio frames rather than the CSV header; a stalled input stream is detected and surfaced within seconds.
- **Evidence:** 2026-09-09 23:32 session 20260909_233204_799019_en (audio_source=mic): models loaded, 'Listening...' printed, audience page fetched, then no frames; session_lifecycle stayed status=running; partials file empty.
- **Evidence:** Operator UI derived RUNNING from the CSV header while the audience display stayed disconnected; a separate sounddevice sd.rec probe also stalled.
- **Evidence:** File-replay sessions 20260909_233546_027169_en and 20260909_233823_034893_es (audio_source=file) completed with exit 0 on the same build; they do not exercise the microphone path.
- **Evidence:** Fix integrated 2026-09-10: tools/isolated_audio.py runs PortAudio in a disposable child (tools/capture_worker.py) and raises AudioCaptureError after 5 s without first samples or 3 s idle; tools/pipeline_health.py phases loading → listening → ready feed operator readiness (stale after 3 s); unit tests cover the modules. No live microphone session has been run against the fix.
- **Notes:** Fix implemented and integrated (reliability worktree → candidate branch); the real built-in-microphone retest and physical-device checks are deferred to tomorrow at the user's request.
- **Next action:** Tomorrow: run a live built-in-mic session through the operator UI with the audience display connected; confirm readiness flips only on real frames and that an unplugged/blocked input surfaces as input_error within seconds; record the session id here.

### `mac-reliability-implementation` — v2026.14 Mac reliability program on the local branch

- **Priority:** P0 · **Machine:** mac · **Certification:** pending
- **Depends on:** none
- **Sources:** `docs/mac_implementation_status.md`, [PR #192](https://github.com/wrbell/stark-translate/pull/192)
- **Acceptance:** Merged to main through PR #192 with the human, device and visible-browser gates recorded as still open.
- **Notes:** Operator session identity, schema 2 timing, setup/resolver, Review/export, opt-in experiments and packaging guards are implemented and locally validated; the overnight worktrees are integrated; parent validation, evidence refresh and the merge remain.
- **Next action:** Parent: finish validation and merge PR #192.

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

### `lite-cpu-inference` — Lite CPU inference profiles (no GPU) — implementation

- **Priority:** P1 · **Machine:** any · **Certification:** pending
- **Depends on:** none
- **Sources:** `stark_translate/profiles.py`, `operator_app/lite_preflight.py`, `tools/llama_runtime.py`, `docs/lite_profiles.md`, `docs/evaluation/lite_cpu_smoke_20260910.json`, `docs/evaluation/lite_cpu_quality_preparation_20260910.json`
- **Acceptance:** A documented CPU-only profile installs from the `lite-cpu` extra, passes setup/preflight, and runs EN/ES replay end to end on a machine without a GPU; smoke evidence recorded. Performance on an x86 CPU host and natural-speech quality are certified separately.
- **Evidence:** 2026-09-10: isolated Mac CPU install of the Torch-free lite-cpu extra; stark-translate-lite setup/doctor; synthetic EN and ES caption + TTS replays completed (docs/evaluation/lite_cpu_smoke_20260910.json).
- **Evidence:** 2026-09-10: lite-cpu-quality preparation — pinned Gemma 4 E2B Q4_K_M GGUF and native llama.cpp b10883 downloaded and hash/version-verified; no E2B inference run (docs/evaluation/lite_cpu_quality_preparation_20260910.json).
- **Notes:** Profiles standard (default), lite-cpu, lite-cpu-quality and lite-cuda-8gb are integrated (stark_translate/profiles.py, --profile on operator/setup/doctor, STARK_PROFILE, lite preflight admission floors, session-owned llama-server). Evidence so far is synthetic Mac CPU smoke only; no x86 CPU host, no natural speech, no latency or memory gate.
- **Next action:** Smoke and time the lite-cpu and lite-cpu-quality profiles on an actual CPU-only x86 host with natural EN/ES audio; record in docs/lite_profiles.md.

### `overnight-latency-scheduling` — Opt-in bounded scheduling and caption delivery instrumentation

- **Priority:** P1 · **Machine:** mac · **Certification:** pending
- **Depends on:** none
- **Sources:** `tools/latency_experiments.py`, `tools/latency_scheduler.py`, `tools/caption_delivery.py`, `tools/overnight_bench.py`, `docs/current_architecture.md`
- **Acceptance:** Merged into the PR branch behind opt-in flags with tests; a frozen-screen replay shows a matched delivery improvement on both models or the experiment is recorded as rejected.
- **Notes:** Integrated on the candidate branch: tools/latency_scheduler.py, tools/caption_delivery.py, tools/latency_experiments.py (provisional previews, exact fixed-prefix cache, bounded allocator, pause speculation; validated before startup), tools/latency_trace.py, tools/overnight_bench.py and tests. All experiments stay opt-in; no matched delivery improvement has been measured yet.
- **Next action:** Parent: run the experiment matrix with tools/overnight_bench.py and a visible audience browser; record accept/reject per experiment.

### `overnight-reliability` — Process supervision, work leases and isolated audio capture

- **Priority:** P1 · **Machine:** mac · **Certification:** pending
- **Depends on:** none
- **Sources:** `operator_app/processes.py`, `operator_app/work_lease.py`, `operator_app/support.py`, `tools/capture_worker.py`, `tools/isolated_audio.py`, `tools/pipeline_health.py`
- **Acceptance:** Operator status derives from live pipeline health (frames, heartbeats) rather than file presence; stalled capture surfaces as a failure; tests cover the new modules.
- **Notes:** Integrated on the candidate branch with tests: processes.py (owned-process cleanup), work_lease.py, support.py (support bundles), capture_worker.py / isolated_audio.py (disposable PortAudio child, no-input timeouts), pipeline_health.py (phase/readiness channel consumed by the operator), tools/persistence.py, tools/operational_logging.py. Operator readiness now derives from health rather than the CSV header. Behaviour against a real microphone is unverified until tomorrow's retest (mac-live-mic-stall).
- **Next action:** Verify against the real built-in microphone tomorrow; then flip certification with the session id.

### `issue-137-active-learning` — Active learning — low-confidence to operator correction (#137)

- **Priority:** P2 · **Machine:** both · **Certification:** pending
- **Depends on:** none
- **Sources:** [#137](https://github.com/wrbell/stark-translate/issues/137), `operator_app/review.py`, `tools/review_data.py`, `tools/merge_corrections.py`, `tests/test_operator_review.py`, `tests/test_correction_import_safety.py`
- **Issue acceptance (verbatim intent):** An operator can correct a caption and that pair lands in a dated corrections corpus. Retrain script documented even if the first retrain is a dry run.
- **Acceptance:** An operator approves a correction from a recorded Sunday session, the pair reaches a dated corpus with provenance, and the documented correction → merge → smoke-retrain workflow is exercised. The first retrain may be a dry run; a merger dry run alone is not the retrain step.
- **Notes:** Live/post-session Review, independent transcript/translation approval, revisioned drafts, portable bundles and evaluation/training separation are implemented and tested with fixtures. No human approval has been recorded; fixtures are not approved correction data.
- **Next action:** After human approval of a recorded Sunday correction, export and merge against a scratch training corpus, then execute/document the smoke-retrain dry run. Live microphone capture is not an extra prerequisite for reviewing a recorded session.

### `overnight-operator-ui` — Operator UI caption and QR widgets

- **Priority:** P2 · **Machine:** mac · **Certification:** pending
- **Depends on:** none
- **Sources:** `displays/operator/widgets/captions.js`, `displays/operator/widgets/qr.js`, `displays/operator/widgets/sparkline.js`, `docs/operator_runbook.md`
- **Acceptance:** Widgets integrated, HTML5 Tidy clean, and the operator runbook updated with root-recorded UI evidence.
- **Notes:** Actual integrated EN↔ES browser sessions, review draft reload, faithful summaries and private support download are recorded in evaluation/overnight_operator_rehearsal.md. QR oracle/decoder checks pass. Final HTML/check evidence still needs refresh.
- **Next action:** Run final HTML/UI checks on the integrated source; keep live microphone and physical audio testing in their separate gates.

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
- **Next action:** Parent re-runs the suite on the integrated branch and refreshes the recorded counts in mac_implementation_status.md.

### `mac-defaults-frozen` — Retain Mac defaults after the 48-run English screen

- **Priority:** P0 · **Machine:** mac · **Certification:** met
- **Depends on:** none
- **Sources:** `docs/evaluation/mac_v2026_14_screening/README.md`, `docs/mac_implementation_status.md`
- **Acceptance:** No combined configuration beat E4B + 0.5 s silence + 0.6 s cadence across both models on the frozen 45-second English screen; all experiments remain opt-in.
- **Evidence:** 48/48 runs exited zero across eight configurations × two models × three pairs; no consistent both-model winner (screening README).
- **Next action:** Any default change requires a matched both-model gain plus human review.

## Summary counts

| Status | Count |
|--------|------:|
| In Progress | 6 |
| Pending Input Or Hardware | 14 |
| Experimental | 2 |
| Deferred | 6 |
| Implemented | 11 |
| Validated | 2 |
