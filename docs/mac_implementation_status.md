# Mac implementation status — September 10 candidate

The integrated candidate is on `codex/mac-reliability-roadmap`, proposed in
[PR #192](https://github.com/wrbell/stark-translate/pull/192). Final validation and
the authorized source merge remain in progress. Package publication and release
tags are pending by user choice. The [September 9 snapshot](mac_implementation_status_20260909.md)
preserves earlier counts and artifact identities; those do not certify later changes.

EN↔ES is the production and latency priority. Mac defaults remain Parakeet English,
Whisper Turbo Spanish, Marian CT2 CPU previews and Gemma 4 E4B OptiQ finals,
0.5-second silence and 0.6-second partial cadence. E2B and new scheduling experiments
are opt-in. MTP is rejected before live model loading. Hindi is a separate completed
offline R&D baseline, with no further work in the current speed program.

The [completed 96-run screen](evaluation/overnight_screen_20260910/README.md) selected **0/28 experiment/model arms**. The sub-second final-delivery goal was not met on this 45-second English cohort. No ordinary historical confirmation or combination of these arms is justified; E4B remains the default. New engineering hypotheses are tracked separately.

## Implemented

- Operator-owned session identity, production CSV parsing and unavailable values.
  Readiness requires fresh health and observed audio, not a process or CSV header.
- Isolated bounded microphone capture, input-stall reporting, persistence failure
  reporting and owned-process cleanup. Pause/Stop drain work; file Resume preserves
  consumed audio positions instead of replaying or skipping prefetched samples.
- Additive timing schema 2 through partials, carryover and finalization. Silence,
  smart/hard cuts, EOF, Pause and Stop remain distinct. Actual visible render ACKs
  never block inference; speech-end-to-ACK includes return-network time.
- Shared MLX generation, prompt, stop, warmup, streaming and telemetry contracts,
  including model-family prompts in the multiprocess worker.
- Prepare, Live, Sessions, Help and Advanced operator views; actual readiness,
  language restart, review, faithful short-session excerpts, support and storage.
  QR codes have independent encoder and decoder verification.
- Separate revision-checked corrections, persistent drafts, independent STT and
  bilingual approvals, explicit languages, portable audio/provenance and idempotent
  imports. Incomplete sessions and missing audio cannot silently enter STT training.
- Bounded structured operational logs, a separate required-write persistence ledger,
  work leases and metadata-only support defaults. Optional logs/audio/transcripts
  require explicit selection; native llama.cpp children have bounded private logs.
- Shared-code Lite profiles: CPU Whisper small/Marian, optional CPU E2B quality,
  and RTX2070 E2B. CPU runtime is Torch-free; conversion uses a separate interpreter.
- Backend-aware pinned setup/resolution, explicit/active environment selection,
  explicit generated launchd install/uninstall, complete runtime artifact checks
  and release version/tag identity checks.

## Evidence and active validation

The [integrated browser rehearsal](evaluation/overnight_operator_rehearsal.md)
records actual EN→ES→EN captions, John 3:16, Pause/Resume, Stop, draft reload,
faithful bilingual excerpts and a metadata-only support download. All three sessions
completed with zero required persistence failures. The longer English session had
7/8 final ACKs, below the 95% delivery gate; short EN/ES sessions had 1/1 each.
No human transcript or translation approval was fabricated.

[Lite evidence](lite_profiles.md) includes isolated installed CPU EN/ES caption and
Piper WAV smokes, actual installed CPU E2B translation, pinned native/model hashes,
clean imports and an [installed dependency audit](evaluation/lite_installer_security_20260910.json).
These Mac CPU functional tests do not certify x86 or RTX2070. E2B pipeline RSS
excludes its native child and cannot certify combined memory.

The [runtime-freeze artifact check](evaluation/overnight_artifact_validation_20260910.json)
verified the wheel, sdist and Mac ZIP at `b65e6e0`. All 149 runtime members match
source, and wheels rebuilt from the sdist/ZIP match the canonical wheel. Both isolated
installations passed five real HTTP routes, version/profile/launchd-render checks
and `pip check`, without loading inference libraries. Final documentation archives
and actual installed inference remain separate checks.

The [final local validation record](evaluation/overnight_validation_20260910.json)
contains 2,213 passing CPU-suite tests, four skips and 63.52% coverage; the three
real GPU regressions also passed. These checks validate the integrated source
before this documentation refresh; they do not establish a latency or quality gate.
Final CI, source review and the authorized merge remain separate.

The [96-run English screen](evaluation/overnight_screen_20260910/README.md) is
complete on frozen source `911f4ae`: 96/96 valid runs, 672 finals, and 0/28 selected
experiment/model arms. It retains 384 smart cuts, 96 hard cuts, 96 recorded-silence
endings and 96 endings assisted by virtual EOF padding. Every candidate final text
matched its control in 588 comparisons against opening controls and 588 against
closing controls; these repeated comparisons are not independent utterances or
reference-quality scores. The independent audit passed 24,324 consistency checks.

The native Mac reported a locked screen while the audience DOM reported `visible`
and sent ACKs; lock onset and physical display visibility were not observed.
Package activity from 05:51:37 to 05:53:18.380938 UTC overlapped three sessions,
which remain included and annotated. The narrow workload, tiny endpoint cohorts,
unreviewed references and control drift prevent a promotion claim. The
[first-preview E4B note](evaluation/overnight_screen_20260910/first_preview_e4b.md)
explains why a large pooled improvement against closing controls does not justify
an ordinary confirmation run.

At the recorded 06:49:36 UTC handoff on September 10, the standard full-service
endurance replay had started; completion, persistence and memory results were
pending. CPU Lite endurance follows and had not started. No full-hour gate is
marked passed from a successful launch.

Previous [48-run screening](evaluation/mac_v2026_14_screening/README.md),
[24 bilingual routing probes](evaluation/mac_v2026_14_routing/README.md) and
[translation quality comparison](evaluation/mac_v2026_14_quality/comparison.md)
remain separate cohorts. Negative silence, warmup and scheduling results are
retained. Legacy processing times and isolated generation speed are not caption
delivery measurements. The sub-second median goal remains unachieved.

## Remaining gates

- Complete and review the standard and CPU Lite endurance rehearsals. The 96-run
  screen justifies no ordinary confirmations or combined configuration; future
  profiling needs a distinct hypothesis. Keep any Spanish probes in a separate
  cohort, and require quality review before a default change.
- Retest live microphone and physical outputs tomorrow, as requested. Natural
  Spanish, two-speaker audio, bilingual review and approved corrections remain
  external dependencies. Predicted text does not count as a human reference.
- Execute native Windows/RTX2070, representative x86 CPU and WSL training/CUDA
  gates on their target hardware; portable reviewed data remains the handoff.
- Finish the integrated checks, installed inference, evidence/docs refresh and PR
  merge. Source privacy fixes and isolated artifact checks have passed their focused
  regressions; the [Mac dependency assessment](evaluation/overnight_security/README.md)
  retains two unresolved Torch advisories. Preserve working
  `stt_env`, original holdouts and the frozen benchmark dependencies.
- Leave PyPI and release publication pending. Published tags have not moved.
  Previous v2026.13 MSI verification and obsolete-asset cleanup remain in the
  September 9 snapshot; native Windows installation remains untested.

The [backlog](backlog.md) and [issue acceptance audit](issue_closure_audit.md)
keep implementation and certification separate. #134 permits a laptop stand-in
with a complete recorded hymn, spoken segment and written timing/UX note; live
microphone and physical-device requirements belong to their own gates.
