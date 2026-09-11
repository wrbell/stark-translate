# Mac implementation status — September 10 source and evidence

> **Mac EN↔ES follow-up:** [PR #196](https://github.com/wrbell/stark-translate/pull/196) records this work.
> The [experiment screens](evaluation/mac_followup_20260910/README.md) selected no
> default changes. Current [source checks](evaluation/mac_followup_20260910/final-760e948/source-validation.md)
> bind `760e948`; the silent hymn diagnostics retain their earlier c13 identity.
> [Installed delivery](evaluation/mac_followup_20260910/final-760e948/installed-delivery.md)
> records artifact and service acceptance separately. Integration status is recorded separately.


> **Follow-up, September 10:** [Current EN/ES work](evaluation/mac_followup_20260910/README.md)
> adds pinned public speech comparisons, isolated dependency remediation and
> [real microphone capture/readiness checks](evaluation/attended_mic_20260910/README.md).
> The quiet-room check produced no spoken captions; that gate remains pending.
> Older experiment and endurance measurements below retain their original source identities.

This document describes v2026.14 source (`2026.14.0.0`), with prior integration in
[PR #192](https://github.com/wrbell/stark-translate/pull/192) and the EN↔ES follow-up in
[PR #196](https://github.com/wrbell/stark-translate/pull/196). The last published release is v2026.14.0.0 (tagged 2026-09-11 on `50f81c6`; see the [overnight 2026-09-11 status](evaluation/overnight_20260911/STATUS.md)).
PR #192 merged into main at `3e935fe39b96e7b0aa62a74711307f2b3e31a18c` on 2026-09-10T11:57:22Z; [closeout evidence](evaluation/overnight_closeout_20260910/README.md) retains actual merge/closure records, final ab66ad2 CI and bootstrap ZIP evidence. Source integration remains separate from the acceptance evidence below.
PyPI publication is deferred by decision (2026-09-11; the publish job is gated on `PYPI_PUBLISH_ENABLED`); the GitHub Release, MSI and tag are published. The [September 9 snapshot](mac_implementation_status_20260909.md)
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

PR #196 merged into main at `ea4af9a7efc87cd6fc8c86787d15c5610ab1ddd6` on 2026-09-11T01:00:13Z, from reviewed
`1b723bd87f4a2f4adecd081a281ded26cafbb0bf`. Required CI and all three automatic Cursor reviews passed.
Issue #132 subsequently closed COMPLETED at 2026-09-11T01:00:33Z; the other eight issues
remain open with their original gates. [Actual integration receipts](evaluation/mac_followup_20260910/integration-closeout/README.md)
keep this source merge separate from publication and human/device certification.

## Evidence and active validation

The current [760e948 source-validation record](evaluation/mac_followup_20260910/final-760e948/source-validation.md)
retains the completed local CPU, static, text-only GPU and pre-commit checks,
plus exact-source CI. It preserves the original wrapper bookkeeping failure
separately from the successful checks. [Current installed delivery](evaluation/mac_followup_20260910/final-760e948/installed-delivery.md)
has its own artifact and full-service gates.

The historical [c13f51f source validation](evaluation/mac_followup_20260910/final-c13f51f/source-validation.md)
records 2,837 local passing tests, four skips, 20 subtests and 66.62% coverage;
three actual text-only MLX checks, 11 prescribed static commands and seven isolated
pre-commit hooks also passed. Exact-head GitHub Python 3.11/3.12 jobs each passed
2,835 tests, six skips and 20 subtests with 66.49% coverage. Their coverage-comment
format annotation is a separate reporting defect, addressed by selecting the
action's JSON input. Later monitor and documentation checks retain their own identities.

The final installed c13 wheel, sdist and Mac ZIP passed build/member checks,
outside-checkout installation for both Standard and CPU Lite, four read-only
operator launches and all four EN↔ES file smokes. The Standard full-service
pipeline completed all 3,946 writes and 563 finals without cleanup intervention.
Its monitor rejected the valid 91.2 MB diagnostics at a 64 MiB read limit;
a separate bounded-reader report recovered those artifacts while retaining the
original failure. The unchanged terminal validator then passed 46/47 checks and
failed on one blank translated preview out of 2,562. The [producer repair](evaluation/mac_followup_20260910/final-c13f51f/empty-preview-repair.md)
preserves the prior caption and logs that rejection explicitly; 21 focused
production-coroutine checks passed. Fresh repaired-source
Standard/Lite rehearsals on `760e948` passed their separate terminal gates,
with 5,533/5,533 required writes across all six file sessions. Standard silence-final
median was 1,458.7 ms; CPU Lite was 3,297.7 ms with sparse previews. The sub-second
goal remains unmet; c13 is not relabeled as fully passed.
See the [current installed delivery](evaluation/mac_followup_20260910/final-760e948/installed-delivery.md)
and the separate [c13 failure record](evaluation/mac_followup_20260910/final-c13f51f/installed-delivery.md).

The [350-second hymn control](evaluation/mac_followup_20260910/final-c13f51f/hymn-capture.md)
completed without entering music hold. The [102-call boundary comparison](evaluation/mac_followup_20260910/final-c13f51f/hymn-boundary.md)
retains raw title/subject errors and supplied delimiter hypotheses. Neither
establishes human acoustic labels or bilingual approval; #193/#194 remain open.
The [loader continuity note](evaluation/mac_followup_20260910/final-c13f51f/model-source-continuity.md)
binds the selected source review without clearing optional/offline residuals.


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

The [final integration validation](evaluation/overnight_final_validation_20260910.md)
records source `84832fb`: 2,384 CPU tests passed, four skipped, 63.80% coverage
(91.56 s), prescribed static checks and mechanical artifact checks passed. Remote
CI also passed its executed checks on that exact head; its legacy audit was skipped.
A real interpreter-exit crash exposed unfinished native audio-watcher work. The
cleanup fix changed five operator modules; 138/143 code/resource wheel members,
including the pipeline and every engine module, remain byte-identical to 752.
Six actual native enumeration/rapid-exit checks passed with no surviving polling
workers or model imports. They saw zero input/output devices and opened no audio
stream, so this is not a hardware or microphone test. No new inference hour was
run after this cleanup change. The report preserves the failed intermediate receipt
and the older 752 evidence below. This linked `84832fb` report remains a pre-merge snapshot; the actual merge and closures are in [closeout evidence](evaluation/overnight_closeout_20260910/README.md).

The [runtime-freeze artifact check](evaluation/overnight_artifact_validation_20260910.json)
verified the wheel, sdist and Mac ZIP at frozen source `752ab9a`. All 152 runtime
members match that source, and both rebuilt wheels are byte-identical to canonical
wheel `7477574d…`. Isolated Mac and Lite installations passed five real HTTP routes and an installed verse-parser
assertion, plus version/profile/launchd-render checks
and `pip check`. Dependency versions stayed unchanged and inference libraries were
not loaded. The [older b65e6e0 receipt](evaluation/overnight_artifact_validation_b65e6e0_20260910.json)
is preserved for the original hour. Mechanical package checks do not transfer an
old inference result to changed runtime bytes or certify a new full-hour run.

The [752 local validation record](evaluation/overnight_validation_20260910.json)
records 2,363 passing CPU-suite tests, four skips and 63.80% coverage at frozen
source `752ab9a` (89.97 s); all three real GPU regressions passed (21.16 s).
Pre-commit and the [prescribed static checks](evaluation/overnight_static_validation_752ab9a_20260910.json)
also passed: Ruff/format, mypy engines/settings, the configured Bandit scan and
HTML5 Tidy on six pages. Bandit uses the documented CI scope/exclusions; this is
not a new dependency advisory audit. The [earlier local check record](evaluation/overnight_validation_early_20260910.json)
is retained with its original source-binding limitation. These checks establish
no latency, human-quality or endurance gate; remote CI and source integration are
recorded separately.

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

The earlier Standard hour completed required writes and cleanup but failed
source-bound validation (23/549 final spans and 97/2,720 preview spans); it remains
a separate failed cohort. The fresh full-service Standard session
`20260910_043120_839144_en`, installed from frozen source `752ab9a`, completed at
09:32:38.589810 UTC with exit 0. All 563 final spans and retained WAV headers,
and all 2,814 preview spans, are consistent; all 7,594 writes completed without
failure and process cleanup was observed. One document-visible ACK connection
matched all 563 finals. These checks do not establish human quality, physical
screen visibility or complete source-speech coverage.

CPU Lite session `20260910_053518_894101_en`, using the same `752ab9a` wheel,
completed at 10:36:47.110148 UTC with exit 0. All 468 final spans/WAV headers and
271 preview spans agree; all 1,979 writes completed with zero pending/failed, and
cleanup was observed. One matched document-visible connection acknowledged all
468 finals and 271 translated previews, but only 174/468 finalized utterances had
a first translated preview. Lite is functionally exercised on this Mac; its sparse
previews and large observed latency tails do not support recommending it as a fast
production profile today. Hardware and human-quality certification remain pending.

[The retained endurance report](evaluation/overnight_endurance_20260910/README.md)
keeps the old failed Standard, repaired Standard and Lite cohorts separate. Their
concurrent lightweight activities are disclosed; observed Standard/Lite speed
is not a causal paired comparison. Selected waveform reconstruction matched three
windows in each repaired cohort, while reproducing the old chunk 141 deficit.
That is supporting regression evidence, not exact EOF or all-source coverage.
Lite's last caption was followed by 162.549 seconds of unclassified source; its
closing-hymn context must not be described as a quiet tail. The prepared 350-second
slice was unused.

Previous [48-run screening](evaluation/mac_v2026_14_screening/README.md),
[24 bilingual routing probes](evaluation/mac_v2026_14_routing/README.md) and
[translation quality comparison](evaluation/mac_v2026_14_quality/comparison.md)
remain separate cohorts. Negative silence, warmup and scheduling results are
retained. Legacy processing times and isolated generation speed are not caption
delivery measurements. The sub-second median goal remains unachieved.

## Remaining gates

- Retain and review the completed Standard/Lite endurance evidence and the original
  failed cohort, including sparse Lite previews and unresolved quality. The 96-run
  screen justifies no ordinary confirmations or combined configuration; future
  profiling needs a distinct hypothesis. Keep any Spanish probes in a separate
  cohort, and require quality review before a default change.
- Retest live microphone and physical outputs at the next attended session. Natural
  Spanish, two-speaker audio, bilingual review and approved corrections remain
  external dependencies. Predicted text does not count as a human reference.
- Execute native Windows/RTX2070, representative x86 CPU and WSL training/CUDA
  gates on their target hardware; portable reviewed data remains the handoff.
- Preserve the recorded integration checks, installed inference, evidence and actual
  PR merge/closure receipts. Source privacy fixes and isolated artifact checks have
  passed; the [Mac dependency assessment](evaluation/overnight_security/README.md)
  retains two unresolved Torch advisories. Preserve working
  `stt_env`, original holdouts and the frozen benchmark dependencies.
- Leave PyPI and release publication pending. Published tags have not moved.
  Previous v2026.13 MSI verification and obsolete-asset cleanup remain in the
  September 9 snapshot; native Windows installation remains untested.

The [backlog](backlog.md) and [issue acceptance audit](issue_closure_audit.md)
keep implementation and certification separate. #134 permits a laptop stand-in
with a complete recorded hymn, spoken segment and written timing/UX note; live
microphone and physical-device requirements belong to their own gates.

The earlier reviewed `ab66ad2` head passed 2,398 tests with four skips on both Python 3.11 and 3.12 CI. Its Mac ZIP contains the patched executable bootstrap; all 152 packaged runtime members match `84832fb`. No additional inference or hardware validation is implied. #134/#176 acceptance is met after their actual completed closures; #177 is closed NOT_PLANNED with implementation deferred and promotion certification pending.
