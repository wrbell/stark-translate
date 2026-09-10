# Remaining backlog — Stark Road Bilingual Speech-to-Text

> **Canonical machine-readable source:** [`backlog.json`](./backlog.json).
> Regenerate this file with `python tools/render_backlog.py render`.
> **Last updated:** 2026-09-10

## Integration status

- **Main release:** `v2026.13` — PR #192 merged into main at 3e935fe39b96e7b0aa62a74711307f2b3e31a18c on 2026-09-10T11:57:22Z, from reviewed source ab66ad2929e61171e4ab9c7c77685ba1ac988577. v2026.13 remains the last published release; no release/tag upload followed this source merge.
- **Local candidate:** `2026.14.0.0` on `main` (base `PR #192 merged at 3e935fe39b96e7b0aa62a74711307f2b3e31a18c from reviewed ab66ad2. The 752ab9a endurance cohort, five-module 84832fb operator cleanup and ab66ad2 bootstrap delivery retain separate exact-source evidence.`)
- **Draft PR:** [PR #192](https://github.com/wrbell/stark-translate/pull/192) — merged into main at 3e935fe39b96e7b0aa62a74711307f2b3e31a18c on 2026-09-10T11:57:22Z; exact-head Python 3.11/3.12 CI passed 2,398 tests with four skips each. Merge and closure receipts: docs/evaluation/overnight_closeout_20260910/README.md.
- **Publication:** PR #192 source merge and justified issue closures are complete. Local package artifacts are mechanically validated; PyPI/GHCR/release uploads and release tags remain pending by user choice.

Implemented/validated source from PR #192 is merged into main. Source integration, local artifact validation, public distribution and service certification remain separate. Certification records each item’s actual acceptance; source merge alone does not close human/device gates.

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
- **Sources:** `docs/mac_implementation_status.md`, `docs/evaluation/README.md`, `docs/archive/v2026.13/MAC_LATENCY.md`, `docs/evaluation/overnight_screen_20260910/README.md`, `docs/latency_next_experiments.md`
- **Acceptance:** Median schema 2 speech_end_to_final_ms under 1000 ms on the frozen real-time baseline with visible ACKs, without regressing final quality; natural-speech quality certification (references, bilingual review) is a separate gate and must not be pooled with historical cohorts.
- **Notes:** The completed 96-run, 672-final English screen selected 0/28 experiment/model arms and did not achieve the sub-second final goal on this workload. E4B defaults remain unchanged. All 588 candidate final comparisons against each control set retained text; quality remains unreviewed. Endpoint counts are small and control drift is material.
- **Next action:** Pursue distinct endpoint/deadline/readback hypotheses from latency_next_experiments.md after endurance; do not schedule ordinary confirmations or combine these rejected arms. Keep natural references and physical display certification separate.

### `mac-live-mic-stall` — Built-in microphone capture stalled; operator showed RUNNING without audio

- **Priority:** P0 · **Machine:** mac · **Certification:** pending
- **Depends on:** none
- **Sources:** `tools/isolated_audio.py`, `tools/capture_worker.py`, `tools/pipeline_health.py`, `operator_app/pipeline_manager.py`, `docs/current_architecture.md`, `docs/evaluation/attended_mic_20260910/README.md`, `docs/evaluation/tts_routing_20260910/README.md`
- **Acceptance:** A live built-in-microphone session produces partial and final captions on the audience display; operator status reflects actual audio frames rather than the CSV header; a stalled input stream is detected and surfaced within seconds.
- **Evidence:** 2026-09-09 23:32 session 20260909_233204_799019_en (audio_source=mic): models loaded, 'Listening...' printed, audience page fetched, then no frames; session_lifecycle stayed status=running; partials file empty.
- **Evidence:** Operator UI derived RUNNING from the CSV header while the audience display stayed disconnected; a separate sounddevice sd.rec probe also stalled.
- **Evidence:** File-replay sessions 20260909_233546_027169_en and 20260909_233823_034893_es (audio_source=file) completed with exit 0 on the same build; they do not exercise the microphone path.
- **Evidence:** Capture/readiness fix integrated September 10: isolated PortAudio child, bounded startup/idle/backpressure failures and actual frame readiness. Subsequent real-device sessions and failures are retained in attended_mic_20260910 and tts_routing_20260910.
- **Notes:** Built-in quiet-room EN/ES sessions reached ready and stopped cleanly on September 10. Synthetic acoustic EN produced partials/final and completed. Original Spanish acoustic capture failed; traced a8511ee retest eliminated parent-handoff drops but retained 7,680 upstream dropped samples (160 ms), so lossless capture is not certified. Retest included incidental speech; shared evidence is structural only. A per-process device-index mismatch also exposed a selection defect now being repaired.
- **Next action:** Finish identity-based input selection and resolve the measured upstream capture loss; preserve all failed lifecycles. Controlled file runs and client-reported visible ACKs do not certify live microphone or physical display quality.

### `packaging-artifacts-local` — v2026.14 wheel, sdist and Mac ZIP local validation

- **Priority:** P1 · **Machine:** mac · **Certification:** pending
- **Depends on:** none
- **Sources:** `docs/evaluation/mac_v2026_14_installation.md`, `docs/evaluation/overnight_artifact_validation_20260910.json`, `docs/evaluation/overnight_endurance_20260910/README.md`
- **Acceptance:** Outside-checkout install, launcher checks and installed EN/ES inference recorded with artifact hashes.
- **Notes:** Frozen 752 wheel/sdist/Mac ZIP mechanical validation passed: 152 runtime members match source; ZIP/sdist rebuilt wheels match canonical, and both isolated installs passed outside-checkout HTTP/profile/version/launchd checks without inference. Wheel 7477574d25c91739b6a88ca142a35bf36258599a66671b8dbb32237d1fa852b5 then ran separate completed Standard and CPU Lite natural-service hours. This does not establish human quality or fast CPU production. The validator-only false failure and old b65 receipt remain preserved. Docker main pushes build without uploading; public distributions remain pending.
- **Next action:** Retain exact artifact/runtime receipts and completed functional evidence, verify final workflow/source checks, and publish only after the separately required release authorization.

### `visible-browser-timing-run` — Frozen timing run with a visible browser (non-zero ACK coverage)

- **Priority:** P1 · **Machine:** mac · **Certification:** pending
- **Depends on:** none
- **Sources:** `docs/evaluation/mac_v2026_14_screening/README.md`, `docs/evaluation/mac_v2026_14_routing/README.md`, `docs/evaluation/overnight_screen_20260910/README.md`, `docs/evaluation/overnight_endurance_20260910/README.md`
- **Acceptance:** Replay on the unlocked Mac with the audience display connected records schema 2 speech_end_to_final_ms and non-zero visible final ACK coverage for the acceptance configuration.
- **Notes:** The separate 96-run screen has per-session browser-DOM ACK evidence. Repaired Standard also has one matched document-visible audience connection acknowledging 563/563 finals and 2,813 nonempty translated previews. The native Mac was observed locked; physical screen visibility was not certified. These observations do not satisfy this item’s existing unlocked-Mac criterion. Earlier no-browser screens remain separate cohorts. Lite also has 468/468 final and271 translated-preview ACKs, but first-preview coverage is174/468. Physical visibility remains uncertified; this is not a human-quality result.
- **Next action:** Retain the completed matrix, and verify the separate unlocked/attended physical display gate when the Mac is available. Do not rerun rejected arms as ordinary confirmations.

### `docs-refresh-remaining` — Documentation refresh — remaining areas after the overnight docs pass

- **Priority:** P2 · **Machine:** any · **Certification:** pending
- **Depends on:** `pr-192-integration`
- **Sources:** `docs/overnight_status.md`, `docs/evaluation/overnight_endurance_20260910/README.md`
- **Acceptance:** Every README/CLAUDE/AGENTS guide describes current behavior from source, historical numbers live only under dated archive links, and tests/test_documentation.py plus render/link checks pass.
- **Notes:** Current guides, exact 752 artifact/test receipts, canonical backlog and explicit public endurance files are refreshed. Standard/Lite completed evidence remains separate from original failed data and 96-run screen. The raw archive is verified and linked. Hymn follow-ups #193/#194 are linked. Final-head CI passed; current source/issue states now reference actual closeout receipts. Historical pre-merge snapshots remain unchanged.
- **Next action:** Keep current guides synchronized with actual states; preserve historical receipts and separate device/human/publication gates.

### `security-b615-pinning` — Remaining unpinned Hugging Face download paths (B615)

- **Priority:** P2 · **Machine:** both · **Certification:** pending
- **Depends on:** none
- **Sources:** `docs/evaluation/mac_v2026_14_security.md`, `docs/evaluation/security_feasibility_20260910/README.md`
- **Acceptance:** Optional/fallback HF paths pin revisions or are documented as operator-only, starting with live-path fallbacks (Marian HF, Piper missing-voice).
- **Notes:** Live Marian HF and missing Piper voice resolution now use registered pinned revisions, with local explicit paths supported and unknown remote fallback rejected. Static B615 still reports calls using validated keyword arguments; no suppression was added. Broader optional/training call-site inventory remains outside the validated live-path subset.
- **Next action:** Retain live-path pinning evidence; reconcile remaining optional and training download sites individually before claiming all B615 scope complete.

## Pending Input Or Hardware

### `bilingual-blinded-review` — Blinded bilingual review of meaning errors and terminology

- **Priority:** P1 · **Machine:** mac · **Certification:** pending
- **Depends on:** none
- **Sources:** `docs/evaluation/mac_v2026_14_quality/comparison.md`, `docs/evaluation/README.md`, `docs/evaluation/mac_followup_20260910/bilingual-review-preparation.json`, `docs/evaluation/mac_followup_20260910/translation-comparison.md`
- **Acceptance:** A bilingual reviewer completes blind_review.jsonl for the E4B/E2B comparison; the E2B speed tradeoff is accepted or rejected on meaning and terminology, not canary counts alone.
- **Notes:** A model-blinded development packet contains 354 cases (118 source/canary items × three repeats); all approval fields remain false. The private key is separate and excluded from the reviewer ZIP. Isolated E4B/E2B latency, reference chrF, changed outputs and all 18 canaries are retained separately without claiming a single quality percentage.
- **Next action:** Give only the prepared reviewer ZIP to bilingual reviewers; withhold the private key and model-labeled report until annotations are locked. No model-default change is authorized by automated scores alone.

### `cuda-latency-proposal` — CUDA latency proposal execution on the A2000

- **Priority:** P1 · **Machine:** wsl · **Certification:** pending
- **Depends on:** none
- **Sources:** `docs/cuda_latency_proposal.md`, `scripts/cuda/bench_mtp.sh`, [#174](https://github.com/wrbell/stark-translate/issues/174)
- **Acceptance:** llama.cpp b10883 built, MTP opt-in bench, `-fa on` retest, W16 HF fp16 and Parakeet probes recorded with the proposal's tables filled in.
- **Notes:** start_server.sh now defaults to --no-draft with --mtp opt-in and both pins read b10883; the scripts remain header-marked unexecuted.
- **Next action:** Run scripts/cuda/*.sh on WSL.

### `hymn-capture-suppression` — Reduce unwanted hymn captions without losing short spoken replies (#193)

- **Priority:** P1 · **Machine:** mac · **Certification:** pending
- **Depends on:** none
- **Sources:** [#193](https://github.com/wrbell/stark-translate/issues/193), `docs/evaluation/overnight_endurance_20260910/README.md`, [README.md](https://github.com/wrbell/stark-translate/blob/0483f81a57a3ee689b51cda70ed0b8dc85e7926d/docs/evaluation/overnight_endurance_20260910/README.md)
- **Acceptance:** On a bounded, human-labeled natural hymn→speech transition, measure unwanted captions, suppressed legitimate speech and recovery alongside unchanged controls and partial/final latency. Include quiet prayer and short valid EN/ES replies; no global short-word blacklist or inferred reference labels from generated captions.
- **Evidence:** Repaired Standard emitted hymn-context fragments including “It dies a” and “Changing uh” while process health remained normal. No music-hold event was recorded. Current energy/VAD logic is a heuristic, not a validated music classifier; absence of its event does not establish exactly why the streak threshold was not met.
- **Notes:** Open hymn-handling follow-up #193 from the completed #134 rehearsal. Attended Pause during singing and Resume before spoken prayer remains a manual option; no such pause was inserted into the repaired full-service replay. Human correction/training approval remains separate. Earlier operator UX blockers were fixed separately.
- **Next action:** Label uncertain music/speech transitions before evaluating a detector or scheduling change; retain legitimate short spoken replies and report recovery and latency.

### `hymn-translation-boundary` — Preserve title/sentence meaning in natural hymn captions (#194)

- **Priority:** P1 · **Machine:** mac · **Certification:** pending
- **Depends on:** `bilingual-blinded-review`
- **Sources:** [#194](https://github.com/wrbell/stark-translate/issues/194), `docs/evaluation/overnight_endurance_20260910/README.md`, [README.md](https://github.com/wrbell/stark-translate/blob/0483f81a57a3ee689b51cda70ed0b8dc85e7926d/docs/evaluation/overnight_endurance_20260910/README.md)
- **Acceptance:** With independently reviewed source boundaries and bilingual references, preserve the intended subject/title boundary on the retained natural hymn example without hard-coded word substitutions. Compare unchanged controls, report subject/negation/name/theological meaning errors and preview/final latency before any prompt or context promotion.
- **Evidence:** Repaired Standard session 20260910_043120_839144_en, chunk 3, source 414.592–422.592 s, installed 752ab9a: the recognized string includes “Eternity Time will soon end”, but the Spanish final says “La eternidad pronto terminará.” QE 1.0 did not flag the changed meaning. Audio punctuation and the bilingual reference remain unreviewed.
- **Notes:** Open quality follow-up #194 from the completed #134 rehearsal. Retained evidence is not an approved correction, model-default promotion or attribution of the whole error to one pipeline stage. Earlier operator UX blockers were fixed separately.
- **Next action:** Obtain independent boundary/reference review, then compare a bounded delimiter/context hypothesis with unchanged controls and meaning/latency guards.

### `issue-135-mac-ab` — Deploy W16 + v2-cpo to Mac and live A/B vs stock (#135)

- **Priority:** P1 · **Machine:** both · **Certification:** pending
- **Depends on:** none
- **Sources:** [#135](https://github.com/wrbell/stark-translate/issues/135), `docs/gemma4_tuning/v3_directions.md`, `docs/wsl_pipeline_refresh.md`
- **Issue acceptance (verbatim intent):** A short A/B note with canary scores and a ship decision. If no-ship, stock E4B stays default and this closes.
- **Acceptance:** Copy W16 CT2 and v2-cpo artifacts to the Mac; perform the original live stock/adapted STT and translation A/B, at least the issue's five-canary health check, terminology audit and a written ship/no-ship note. The current eight/eighteen-item sets are additional coverage, not a rewritten original criterion.
- **Notes:** Mac default EN STT is now Parakeet MLX; W16 is a faster-whisper CT2 artifact, so the STT half of this A/B runs the CPU faster-whisper path or compares against Parakeet explicitly. v2-cpo reached statistical parity with stock E4B on COMET-22 but still misses the Jacobo canary.
- **Next action:** Transfer artifacts from WSL; run tools/health_check.py --backend mlx --n-canaries 8 and a replay A/B.

### `natural-spanish-refs` — Human-reviewed natural Spanish references

- **Priority:** P1 · **Machine:** mac · **Certification:** pending
- **Depends on:** none
- **Sources:** `docs/mac_implementation_status.md`, `docs/evaluation/README.md`, `docs/evaluation/mac_followup_20260910/public_data/README.md`
- **Acceptance:** At least 50 approved natural utterances per language; no WER or natural-audio acceptance claim before approval.
- **Notes:** Pinned public FLEURS contains 100 natural read-speech recordings per language, with 50 development and 50 untouched confirmation, exact upstream references and hashes. These permit explicitly labeled engineering WER; they are not local human approval or church-domain references. Existing church candidates remain unapproved.
- **Next action:** Obtain bilingual approval for at least 50 natural church-relevant utterances per language; keep public engineering references and synthetic Piper evidence separate.

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
- **Notes:** The user deferred physical-device checks to the next attended session.
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
- **Sources:** [#136](https://github.com/wrbell/stark-translate/issues/136), `docs/gemma4_tuning/v3_directions.md`, `tools/build_preference_triples.py`, `training/candidates/README.md`
- **Issue acceptance (verbatim intent):** Jacobo canary passes (or we document why it cannot) and COMET-22 does not regress vs v2-cpo.
- **Acceptance:** 50–100 hand-crafted preference triples, one CPO continue from v2 (`training/train_gemma4_cpo.py --init-adapter`), re-scored 8-canary set and 500-verse holdout, Jacobo passing or a documented reason.
- **Notes:** Sixty original Jacobo/Santiago preference candidates are prepared, explicitly unapproved and excluded from training. Available exclusion files were checked; the missing versioned WSL holdout is still required. No CPO continue, COMET-22 or trained-canary success is claimed.
- **Next action:** Review candidates against the complete WSL holdouts, obtain explicit preference approval, run one CPO continue from v2 and re-score the canaries and 500-verse holdout.

### `pypi-publication` — PyPI trusted publisher, public artifact uploads and release tag

- **Priority:** P2 · **Machine:** any · **Certification:** pending
- **Depends on:** `pr-192-integration`
- **Sources:** `docs/mac_implementation_status.md`, `.github/workflows/pypi.yml`
- **Acceptance:** Trusted publisher mapped (owner wrbell, repo stark-translate, workflow pypi.yml, environment pypi), a new version tag pushed, and the PyPI workflow green.
- **Notes:** Publication is explicitly pending by user choice; source/issue publishing and final main merge are authorized. Local wheel/sdist/Mac ZIP artifacts are already mechanically validated, which does not authorize PyPI/GHCR/release uploads.
- **Next action:** User decision after the merge.

## Experimental

### `conservative-marian-routing` — Conservative Marian partial routing

- **Priority:** P3 · **Machine:** mac · **Certification:** pending
- **Depends on:** `natural-spanish-refs`
- **Sources:** `docs/evaluation/mac_v2026_14_routing/README.md`, `settings.py`
- **Acceptance:** Natural Spanish references plus human review show no quality loss when Marian handles allowlisted phrases; until then `--routing-policy conservative` stays opt-in.
- **Notes:** 24/24 synthetic EN/ES routing probes exited zero and routed as designed; they are functional checks, not quality evidence.
- **Next action:** Keep opt-in.

## Deferred

### `issue-138-hindi-zero-shot` — Hindi zero-shot baseline on church audio (#138)

- **Priority:** P3 · **Machine:** mac · **Certification:** pending
- **Depends on:** none
- **Sources:** [#138](https://github.com/wrbell/stark-translate/issues/138), `tools/offline_hindi.py`, `docs/evaluation/mac_v2026_14_hindi/README.md`, `docs/archive/research/multi_lingual.md`, `docs/evaluation/overnight_hindi/README.md`
- **Issue acceptance (verbatim intent):** A short baseline note: does zero-shot Hindi even work on church audio, and is a QLoRA week worth it this semester — via target_lang_code="hi" through the live pipeline, noting SOV partial garble, with 8 canaries and a few verse pairs.
- **Acceptance:** Written baseline note on church audio (does zero-shot Hindi work; is a QLoRA week worth it) with a QLoRA go/no-go. The issue asks for the live pipeline; the offline audio tool provides the baseline measurement only — a live Hindi target is not integrated.
- **Notes:** The separate offline church-audio R&D baseline is complete on E4B/E2B with every output archived. There are no Hindi references or human quality review and no live Hindi integration. User reaffirmed EN↔ES as the speed priority; no further Hindi work is scheduled.
- **Next action:** Await a later R&D decision and human Hindi review; keep live Hindi integration and QLoRA separate from EN↔ES optimization.

### `issue-177-mtp` — Gemma 4 MTP assistant drafter on MLX (#177)

- **Priority:** P3 · **Machine:** mac · **Certification:** pending
- **Depends on:** none
- **Sources:** [#177](https://github.com/wrbell/stark-translate/issues/177), `docs/archive/v2026.13/MAC_LATENCY.md`, `docs/mlx_mtp_notes.md`, `engines/mlx_spec.py`, `tools/mts_acceptance_probe.py`, `dry_run_ab.py`, `docs/evaluation/overnight_closeout_20260910/README.md`
- **Issue acceptance (verbatim intent):** Greedy-identical output; canary ≥7/8; medium p50 ≤0.85× post-EOS-fix E4B; drafter acceptance ≥30%. Otherwise MTP stays off; low acceptance was a timeboxed implementation investigation.
- **Acceptance:** Record the original timeboxed investigation and each promotion gate. Only greedy-identical output, canary ≥7/8, medium p50 ≤0.85× the post-EOS baseline and acceptance ≥30% together permit promotion; otherwise keep MTP off.
- **Notes:** The archived offline experiment had 33/33 greedy-identical outputs and 31.3% acceptance, but medium p50 1,339/1,393 ms (about 0.96×) failed the ≤0.85× speed gate; a ≥7/8 canary result was not established. The RoPE-offset hypothesis was rejected. Live requested/configured MTP fails before model loading; --no-mts explicitly selects the supported off path. #177 closed NOT_PLANNED at 2026-09-10T11:57:53Z as a negative/timeboxed investigation. Promotion acceptance remains unmet; implementation is deferred and certification pending.
- **Next action:** Keep MTP off. A distinct future upstream/runtime hypothesis requires new evidence and all promotion gates; closure is not an optimization certification.

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

### `issue-132-tts-routing` — Multi-channel TTS routing code and tests (9.4.1 / #132)

- **Priority:** P1 · **Machine:** mac · **Certification:** pending
- **Depends on:** none
- **Sources:** [#132](https://github.com/wrbell/stark-translate/issues/132), `tests/test_tts_multichannel.py`, `tests/test_phase9_4_1_tts_device.py`, `settings.py`, `docs/evaluation/tts_routing_20260910/README.md`
- **Issue acceptance (verbatim intent):** Operator can send TTS to a chosen output device; EN and ES can be routed independently. Tests cover the new engine path.
- **Acceptance:** Hardware-independent part: per-language output map (`--tts-device-en/es`, `STARK_TTS_OUTPUT_DEVICES`), operator persistence and hotplug retry covered by tests. Physical part: an operator routes EN and ES to two real outputs and hears each — tracked as `physical-second-output`.
- **Evidence:** tests/test_tts_multichannel.py and tests/test_phase9_4_1_tts_device.py pass in the recorded CPU suite (device enumeration mocked).
- **Evidence:** Built-in MacBook speaker playback calls completed in the controlled rehearsal; no second physical device was used.
- **Notes:** Actual operator controls persisted independent EN/ES routes across reload and restart. Production Piper and output resolver completed EN on MacBook speakers and ES on Microsoft Teams virtual output, including explicit pinned voice paths after a symlink resolver fix. Native stream completion is established; human audibility, far-end reception and physical unplug/replug remain separate pending gates. Original issue allows virtual routing.
- **Next action:** Review final integrated caption-to-selected-output wiring and tests, then close #132 if its original routing acceptance is met. Keep physical-second-output pending independently; do not add that stronger requirement to the issue.

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
- **Sources:** `stark_translate/profiles.py`, `operator_app/lite_preflight.py`, `tools/llama_runtime.py`, `docs/lite_profiles.md`, `docs/evaluation/lite_cpu_smoke_20260910.json`, `docs/evaluation/lite_cpu_quality_preparation_20260910.json`, `docs/evaluation/lite_cpu_quality_smoke_20260910.json`, `docs/evaluation/overnight_endurance_20260910/README.md`
- **Acceptance:** A documented CPU-only profile installs from the `lite-cpu` extra, passes setup/preflight, and runs EN/ES replay end to end on a machine without a GPU; smoke evidence recorded. Performance on an x86 CPU host and natural-speech quality are certified separately.
- **Evidence:** 2026-09-10: isolated Mac CPU install of the Torch-free lite-cpu extra; stark-translate-lite setup/doctor; synthetic EN and ES caption + TTS replays completed (docs/evaluation/lite_cpu_smoke_20260910.json).
- **Evidence:** 2026-09-10: installed lite-cpu-quality E2B inference completed on Mac CPU with native GPU layers disabled; output and artifact hashes are retained in docs/evaluation/lite_cpu_quality_smoke_20260910.json. This is functional smoke, not performance certification.
- **Notes:** Profile/setup/doctor/launcher integration and isolated Mac CPU synthetic EN/ES smokes are recorded. Natural English session 20260910_053518_894101_en completed on 752 with 468 finals/271 previews consistent, 1,979 writes complete and cleanup. First translated preview coverage 174/468 and large observed final tails do not support fast-production recommendation. Three selected waveform windows match; no all-source coverage, x86/RAM-floor or human-quality certification.
- **Next action:** Use the completed Mac functional result to target sparse previews and long CPU latency tails; retain quality differences. Validate native x86 CPU/Windows and hardware floors on representative hardware before certification.

### `overnight-reliability` — Process supervision, work leases and isolated audio capture

- **Priority:** P1 · **Machine:** mac · **Certification:** pending
- **Depends on:** none
- **Sources:** `operator_app/processes.py`, `operator_app/work_lease.py`, `operator_app/support.py`, `tools/capture_worker.py`, `tools/isolated_audio.py`, `tools/pipeline_health.py`
- **Acceptance:** Operator status derives from live pipeline health (frames, heartbeats) rather than file presence; stalled capture surfaces as a failure; tests cover the new modules.
- **Notes:** Integrated health/readiness, owned-process cleanup, work leases, bounded capture handoff, required-write ledger and support/logging contracts are tested. Repaired installed Standard completed all 7,594 required writes with zero pending/failed and verified pipeline/descendant cleanup. Original source-bound failure remains recorded; the 752 buffer-discard and authoritative-final guards address its discovered reliability defects with regression coverage. Real-microphone behavior still needs the next attended retest; file evidence does not certify it.
- **Next action:** Retain the repaired Standard evidence and finish the separate Lite outcome; exercise real built-in capture during the next attended session before claiming the microphone gate.

### `issue-137-active-learning` — Active learning — low-confidence to operator correction (#137)

- **Priority:** P2 · **Machine:** both · **Certification:** pending
- **Depends on:** none
- **Sources:** [#137](https://github.com/wrbell/stark-translate/issues/137), `operator_app/review.py`, `tools/review_data.py`, `tools/merge_corrections.py`, `tests/test_operator_review.py`, `tests/test_correction_import_safety.py`, `docs/evaluation/mac_followup_20260910/correction-handoff.md`
- **Issue acceptance (verbatim intent):** An operator can correct a caption and that pair lands in a dated corrections corpus. Retrain script documented even if the first retrain is a dry run.
- **Acceptance:** An operator approves a correction from a recorded Sunday session, the pair reaches a dated corpus with provenance, and the documented correction → merge → smoke-retrain workflow is exercised. The first retrain may be a dry run; a merger dry run alone is not the retrain step.
- **Notes:** Review drafts survived save/reload on a real replay while original diagnostics remained unchanged. The actual unapproved correction was rejected by scratch training/evaluation export; merger now rejects explicit unapproved, excluded, evaluation-only and protected holdout targets. Conditional correction→merge→CPU trainer-preflight commands are documented. No human-approved Sunday correction or smoke-retrain execution is fabricated.
- **Next action:** After human approval of a recorded Sunday correction, run the documented scratch export/merge and actual trainer CPU preflight with real WSL inputs, retaining dated provenance. Live mic is not an extra prerequisite for reviewing a recorded session.

### `overnight-operator-ui` — Operator UI caption and QR widgets

- **Priority:** P2 · **Machine:** mac · **Certification:** pending
- **Depends on:** none
- **Sources:** `displays/operator/widgets/captions.js`, `displays/operator/widgets/qr.js`, `displays/operator/widgets/sparkline.js`, `docs/operator_runbook.md`
- **Acceptance:** Widgets integrated, HTML5 Tidy clean, and the operator runbook updated with root-recorded UI evidence.
- **Notes:** Actual integrated browser sessions exercised EN↔ES switching, Pause/Resume/Stop, unapproved draft recovery and metadata-only support download. Repaired Standard added a full natural-service Live/hymn/prayer view, preserved draft revision 1 with both approvals false, and an actual long-summary UI result disclosing omitted middle content. No successful reload or human summary-fidelity approval is inferred. QR oracle/decoder checks and all six HTML5 Tidy checks passed; final production caption guards are covered by the frozen 752 suite.
- **Next action:** Finalize the current runbook/evidence links and blocking UX issue mapping. Keep live microphone, physical outputs and human quality in their separate gates.

### `wsl-training-recipe-checks` — Repair unexecuted W17 projection and domain-corpus recipe assumptions

- **Priority:** P2 · **Machine:** wsl · **Certification:** pending
- **Depends on:** `wsl-phase4`
- **Sources:** `training/run_w17_curriculum.sh`, `training/run_gemma4_e4b_domain_sft.sh`, `training/CLAUDE.md`, `tools/training_preflight.py`
- **Acceptance:** W17 uses real Whisper projection module names and compatible initialization shape/rank; domain SFT resolves the intended versioned corpus explicitly and rejects missing configured inputs before training. A dry-run command/data inspection is recorded on WSL.
- **Notes:** W17 now uses out_proj, explicitly validates rank/shape and source LoRA tensors, initializes newly expanded DoRA magnitudes and materializes its JSON selection into aligned audio data. Domain SFT requires explicit versioned corpora. Real CPU preflight inspects trainer/config/data/safetensors and fails on missing inputs. Mac tests and conditional WSL commands are retained; actual WSL adapter/corpus execution remains pending.
- **Next action:** Run the repaired CPU preflight on WSL with the real W16/v2 adapters, prepared corpora and complete holdouts before any GPU training.

## Validated

### `issue-134-sunday-dry-run` — Dry-run with the operator runbook (#134)

- **Priority:** P0 · **Machine:** mac · **Certification:** met
- **Depends on:** none
- **Sources:** [#134](https://github.com/wrbell/stark-translate/issues/134), `docs/operator_runbook.md`, `docs/evaluation/overnight_endurance_20260910/README.md`, [#193](https://github.com/wrbell/stark-translate/issues/193), [#194](https://github.com/wrbell/stark-translate/issues/194), [README.md](https://github.com/wrbell/stark-translate/blob/0483f81a57a3ee689b51cda70ed0b8dc85e7926d/docs/evaluation/overnight_endurance_20260910/README.md), `docs/evaluation/overnight_closeout_20260910/README.md`
- **Issue acceptance (verbatim intent):** Walk the runbook on church hardware or a laptop stand-in; time setup → first caption; capture one full hymn plus one spoken segment; a written dry-run note exists (what worked, what broke, time-to-first-caption) and blocking UX holes have their own issues.
- **Acceptance:** Per the issue text: walk the runbook on church hardware or a laptop stand-in (stand-in explicitly permitted); time setup → first caption; capture one full hymn plus one spoken segment; write the dry-run note (what worked, what broke, time-to-first-caption) and file follow-up issues for blocking UX holes. The issue does not add a live-microphone or human-walkthrough requirement beyond that text.
- **Notes:** The laptop stand-in used the installed 752ab9a Standard runtime on the uncropped 3,640.053 s natural English service. Session 20260910_043120_839144_en completed exit 0 with consistent bounds for 563 finals and 2,814 previews, all 7,594 required writes complete and observed cleanup. Actual SPA hymn→prayer chronology, prepared operator launch-to-first-server-preview 436.790088 s and actual Start-to-first-server-preview 420.492–420.762 s (including 403.1448125 s of source zeros), operator/helper steps and a post-Stop long summary are retained. First-install/download time was not measured; the summary omitted the transcript middle and is not human-reviewed. Original b65 failures stay separate. The written report and verified archive are linked; remaining hymn behavior is mapped to #193 and #194 as quality follow-ups; earlier operator UX blockers were fixed. #134 closed COMPLETED at 2026-09-10T11:57:48Z after the recorded source merge.
- **Next action:** Rehearsal acceptance is met. Continue open hymn/quality follow-ups #193/#194 and separate microphone/device/human gates without relabeling them as #134 requirements.

### `mac-cpu-test-suite` — Frozen 752ab9a CPU, GPU and prescribed static validation

- **Priority:** P0 · **Machine:** mac · **Certification:** met
- **Depends on:** none
- **Sources:** `docs/mac_implementation_status.md`, `.github/workflows/test.yml`, `docs/evaluation/overnight_validation_20260910.json`
- **Acceptance:** Recorded pass/skip/coverage in mac_implementation_status.md above the test.yml coverage gate.
- **Notes:** Frozen 752ab9a351815feee4b8cd155f732c588cb30a6c passed 2,363 CPU-suite tests with four skips and 63.80% coverage. Three separate real GPU regressions passed in 21.16 s. Ruff/format, mypy engines/settings, widened CI-scope Bandit and all six HTML5 Tidy checks passed. The earlier 2,213-test receipt is historical and must remain unchanged when the current validation companion is refreshed.
- **Next action:** Bind current documentation to the 752 validation receipt. Later docs/workflow tests and final-head CI must have their own identities; repeat runtime checks only when subsequent changes warrant them.

### `mac-defaults-frozen` — Retain Mac defaults after separate 48-run and 96-run English screens

- **Priority:** P0 · **Machine:** mac · **Certification:** met
- **Depends on:** none
- **Sources:** `docs/evaluation/mac_v2026_14_screening/README.md`, `docs/mac_implementation_status.md`, `docs/evaluation/overnight_screen_20260910/README.md`
- **Acceptance:** Recorded screening decisions retain E4B, 0.5 s silence and 0.6 s cadence unless a matched both-model improvement and the separate quality gate justify a change. Rejected arms do not justify an unmeasured combined configuration.
- **Evidence:** Historical v2026.14 screen: 48/48 runs across eight configurations × two models × three pairs; no consistent both-model winner and no combined configuration recommended.
- **Evidence:** Separate September 10 screen: 96/96 valid runs, 672 finals and 0/28 selected experiment/model arms. Defaults remain unchanged; these distinct cohorts are not pooled.
- **Next action:** Retain both negative reports and unchanged defaults. Any new experiment needs a distinct hypothesis; a default change still requires matched gains and human quality review.

### `mac-reliability-implementation` — v2026.14 Mac reliability source integrated in main

- **Priority:** P0 · **Machine:** mac · **Certification:** met
- **Depends on:** none
- **Sources:** `docs/mac_implementation_status.md`, [PR #192](https://github.com/wrbell/stark-translate/pull/192), `docs/evaluation/overnight_endurance_20260910/README.md`, `docs/evaluation/overnight_closeout_20260910/README.md`
- **Acceptance:** Merged to main through PR #192 with the human, device and visible-browser gates recorded as still open.
- **Notes:** Operator/session reliability, schema 2 timing, setup, safe Review/export and packaging are integrated. 752 CPU/GPU/static checks passed; the 96-run screen selected no arms. Repaired Standard and CPU Lite completed their independent natural-English runs with consistent retained spans and durable writes/cleanup; selected waveforms match and the old failure is preserved. Docker publication guard is implemented. Evidence and #193/#194 links are recorded; source merged at 3e935fe39b96e7b0aa62a74711307f2b3e31a18c; device/human gates remain separate.
- **Next action:** Source integration acceptance is met; continue actual live-mic, independent outputs and human-quality validation separately.

### `pr-192-integration` — Integrated source merged through PR #192

- **Priority:** P0 · **Machine:** any · **Certification:** met
- **Depends on:** none
- **Sources:** [PR #192](https://github.com/wrbell/stark-translate/pull/192), `docs/overnight_status.md`, `docs/evaluation/overnight_screen_20260910/README.md`, `docs/evaluation/overnight_endurance_20260910/README.md`, `docs/evaluation/overnight_closeout_20260910/README.md`
- **Acceptance:** PR #192 marked ready with integrated operator, Lite, latency and failure-recovery changes, CI green on the final head, and root-recorded evidence; main advances only at the authorized merge, without publishing a release.
- **Notes:** Actual merge 3e935fe39b96e7b0aa62a74711307f2b3e31a18c at 2026-09-10T11:57:22Z; final reviewed head ab66ad2929e61171e4ab9c7c77685ba1ac988577 passed 2,398 tests with four skips on each Python 3.11/3.12 CI job. Bootstrap delivery verifies executable ZIP bootstrap and 152 unchanged runtime members versus 84832fb. Earlier 752 endurance, 848 cleanup checks and negative 96-run screen remain separate evidence. No package publication or device/human certification.
- **Next action:** None for source integration. Continue remaining latency, human/device and hardware work; release publication needs the separate user decision.

### `mac-torch-security-migration` — Resolve pinned Mac Torch dependency advisories

- **Priority:** P1 · **Machine:** mac · **Certification:** met
- **Depends on:** none
- **Sources:** `docs/evaluation/overnight_security/README.md`, `pyproject.toml`, `docs/evaluation/mac_followup_20260910/torch-full-application-candidate.md`
- **Acceptance:** A compatible patched Mac Torch/audio dependency set passes installed imports, VAD and real EN/ES inference, with an explicit full installed audit result. No incompatible forced install or changed frozen benchmark environment.
- **Notes:** A fresh Mac arm64 Python 3.11 full-application candidate with Torch 2.13/TorchAudio 2.11 passed installed native imports, bundled VAD, pip check, selected-extra metadata checks and actual normalized EN/ES replay: all 19 validation checks per language. Full installed audit: zero known findings across 123 third-party distributions, unpublished first-party explicitly skipped. Working stt_env and production bounds are unchanged. This meets compatible-candidate feasibility acceptance, not final-source production migration or performance certification.
- **Next action:** If promoting the candidate later, rebuild final source into a fresh environment with the complete audited constraints, revalidate installed artifacts and matched-source performance, then switch the launcher with rollback. Do not upgrade working stt_env in place.

### `overnight-latency-scheduling` — Opt-in bounded scheduling and caption delivery instrumentation

- **Priority:** P1 · **Machine:** mac · **Certification:** met
- **Depends on:** none
- **Sources:** `tools/latency_experiments.py`, `tools/latency_scheduler.py`, `tools/caption_delivery.py`, `tools/overnight_bench.py`, `docs/current_architecture.md`, `docs/evaluation/overnight_screen_20260910/README.md`
- **Acceptance:** Merged into the PR branch behind opt-in flags with tests; a frozen-screen replay shows a matched delivery improvement on both models or the experiment is recorded as rejected.
- **Notes:** Opt-in implementation and tests are integrated. The complete 96-run screen recorded the negative outcome: 0/28 experiment/model arms selected; all remain opt-in, with no ordinary confirmation or combination justified. This meets this item's acceptance alternative of recording rejected experiments, not the separate sub-second or quality gates.
- **Next action:** Retain negative evidence and unchanged defaults. Any follow-up needs a new explicit hypothesis and separate evidence.

### `issue-176-multiprocess` — --multiprocess workers use shared Gemma 4 prompts and stop rules (#176)

- **Priority:** P3 · **Machine:** mac · **Certification:** met
- **Depends on:** none
- **Sources:** [#176](https://github.com/wrbell/stark-translate/issues/176), `workers.py`, `engines/translation_prompts.py`, `tests/test_worker_translation_contract.py`, `docs/issue_closure_audit.md`, `docs/evaluation/overnight_closeout_20260910/README.md`
- **Issue acceptance (verbatim intent):** Route workers.py through engines/translation_prompts build_chat_messages + ensure_stop_tokens (skipping the prompt cache for gemma4), or deprecate --multiprocess.
- **Acceptance:** workers.translation_worker_main builds engines through MLXGemmaEngine with the parent-selected model_family, so Gemma 4 never receives TranslateGemma prompts and stop handling is shared; a live --multiprocess run is not part of the acceptance.
- **Evidence:** Integrated workers.translation_worker_main delegates to MLXGemmaEngine with the parent-selected model family, direction and adapters; Gemma 4 skips the incompatible TranslateGemma prompt cache.
- **Evidence:** Real pipe-loop contract tests cover both model families, EN/ES directions, the six-value A/B response and cleanup with mocked inference; included in the frozen 752ab9a CPU suite.
- **Notes:** #176 closed COMPLETED at 2026-09-10T11:57:50Z after source merge: original shared-engine option (a) is implemented and contract-tested. Multiprocess remains optional with no new performance endorsement.
- **Next action:** Maintain shared prompt/stop contracts; no new multiprocess performance promotion.

## Summary counts

| Status | Count |
|--------|------:|
| In Progress | 6 |
| Pending Input Or Hardware | 15 |
| Experimental | 1 |
| Deferred | 5 |
| Implemented | 8 |
| Validated | 8 |
