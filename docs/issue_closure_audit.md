# Issue acceptance and evidence matrix

Assessment date: 2026-09-10. Acceptance follows the original linked issue bodies.
“Ready after merge” means closure is supported once the relevant source is merged;
it does not assert a GitHub issue state or a completed merge. Current delivery
status lives in [Mac implementation status](mac_implementation_status.md) and the
[backlog](backlog.md). PR #192 remains an open draft; #134, #176 and #177 are OPEN.
The frozen `752ab9a` checks are green, but final evidence/workflow edits still need
their own final-head check before the authorized source merge. No release publication
is implied.

Frozen `752ab9a351815feee4b8cd155f732c588cb30a6c` passed **2,363 CPU tests, four skips
and 63.80% coverage**, plus three GPU regressions in 21.16 seconds and the prescribed
static checks. Its mechanically validated wheel, sdist and Mac ZIP have 152 matching
runtime members. These checks are separate from the [96-run screen](evaluation/overnight_screen_20260910/README.md),
which selected **0/28** experiment/model arms, and from service rehearsal acceptance.

| Issue | Original acceptance | Current evidence | Closure readiness / remaining work |
|---|---|---|---|
| [#131 — startup smoke](https://github.com/wrbell/stark-translate/issues/131) | Fresh setup on Mac and/or CUDA; green preflight; text smoke plus live microphone EN→ES and ES→EN; W16 CT2 preference. Audience partial/final captions without uncaught errors. | [Installed Mac/Lite evidence](lite_profiles.md), [operator rehearsal](evaluation/overnight_operator_rehearsal.md), setup/preflight and factory tests. File captions work; isolated capture and health-derived readiness address the microphone stall. | **Pending:** actual microphone retest in both directions with audience/log evidence, plus the applicable W16 preference check. File replay does not satisfy the microphone item. |
| [#132 — TTS routing](https://github.com/wrbell/stark-translate/issues/132) | Output-device support, local playback and multi-channel/virtual-cable routing; operator selects real devices and routes EN/ES independently; engine tests. | [Runbook routing workflow](operator_runbook.md), [TTS engine](../engines/tts_engine.py), [routing tests](../tests/test_tts_multichannel.py), installed Piper WAV/playback evidence in [Lite profiles](lite_profiles.md). | **Pending:** observe independent EN/ES routing to selected physical or virtual outputs. A WAV or built-in-speaker smoke alone does not demonstrate independent routing. |
| [#133 — live diarization](https://github.com/wrbell/stark-translate/issues/133) | Rolling live diarization labels final captions in operator/audience displays; off by default; two-speaker rehearsal preserves caption p95 and the offline path. | [Implementation and test plan](live_diarization.md), [live daemon](../features/live_diarize.py), opt-in pipeline integration and automated tests. | **Pending:** natural two-speaker rehearsal, distinct observed final labels and caption-p95 comparison. Fixtures do not establish the gate. |
| [#134 — runbook rehearsal](https://github.com/wrbell/stark-translate/issues/134) | Runbook walk on church hardware **or a laptop stand-in**; setup→first-caption timing and operator-only/helper-needed steps; **one full hymn and one spoken segment**; written results and issues for blocking UX holes. | The installed `752ab9a` Standard laptop rehearsal completed the uncropped 3,640.053-second service. Actual SPA hymn→prayer observations, Start timing, operator/helper steps, an unapproved draft and post-Stop summary are retained. All 563 final and 2,814 preview bounds are consistent; 7,594 required writes completed and owned-process cleanup was observed. The earlier b65 span failure is preserved separately. | **OPEN; evidence and follow-ups mapped:** the written note and verified archive are available; operator UX blockers were fixed, and remaining hymn/quality work is tracked in [#193](https://github.com/wrbell/stark-translate/issues/193) and [#194](https://github.com/wrbell/stark-translate/issues/194). Final closure action is still pending. The completed Standard workload supplies the full-service rehearsal; the separate completed Lite hour is not an added #134 prerequisite. Microphone, church hardware, unlocked physical display, exact singing boundaries and lyric review are not additional requirements. |
| [#135 — W16/v2-cpo Mac A/B](https://github.com/wrbell/stark-translate/issues/135) | Transfer W16 CT2/v2-cpo; live Mac stock E4B/v2-cpo and stock Turbo/W16 comparisons; five-canary health/term audit; explicit ship/no-ship note. | [WSL handoff](wsl_pipeline_refresh.md), [health tool](../tools/health_check.py) and [tuning results](gemma4_tuning/v3_directions.md). Stock E4B/E2B latency screens are a different comparison. | **Pending:** specified adapter A/B, canary scores and ship decision. Documented no-ship may close this issue; merely retaining stock E4B does not. |
| [#136 — Jacobo preferences](https://github.com/wrbell/stark-translate/issues/136) | 50–100 preference triples; CPO continuation from v2; rescore eight canaries and 500 verses. Jacobo passes, or inability is documented, without COMET-22 regression versus v2-cpo. | [Tuning directions](gemma4_tuning/v3_directions.md) and [CPO trainer](../training/train_gemma4_cpo.py). Prompt experiments are not the requested training cycle. | **Pending:** triples, continuation and comparison evidence. No new Jacobo success or COMET non-regression is claimed. |
| [#137 — correction loop](https://github.com/wrbell/stark-translate/issues/137) | Flag uncertainty; write training-shaped corrections; recorded-Sunday correction→merge→smoke-retrain loop. A human-corrected pair reaches a dated corpus; retraining is documented, with its first execution allowed to be a dry run. | [Review workflow](operator_runbook.md), [review/export store](../tools/review_data.py), [merge tool](../tools/merge_corrections.py) and contract tests cover drafts, independent approvals, revisions, provenance, audio and idempotence. [Earlier rehearsal](evaluation/overnight_operator_rehearsal.md) and repaired Standard saved **unapproved** drafts. Standard correction revision 1 retained both approvals false; original metric/diagnostic prefixes remained unchanged. | **Pending:** real human correction/approval, dated export, merge and documented retrain smoke/dry run. Generated text, summary output and draft notes are not approved pairs. Replay/training provenance restrictions remain enforced. |
| [#138 — Hindi baseline](https://github.com/wrbell/stark-translate/issues/138) | Live Hindi target, SOV partial-behavior note, eight canaries plus verse pairs, church-audio baseline and QLoRA go/no-go judgment. | Separate [offline church-audio baseline](evaluation/overnight_hindi/README.md): 41 English utterances from 195 seconds plus eight text probes; each model generated all 147 outputs. [Reproduction guide](offline_hindi_baseline.md). | **Pending; separate R&D:** live integration, reviewed Hindi/verse references and the QLoRA judgment remain absent. No further Hindi work is scheduled in the EN↔ES latency program. |
| [#176 — worker prompts](https://github.com/wrbell/stark-translate/issues/176) | Share model-family chat/stop rules and skip the incompatible Gemma 4 prompt cache, or deprecate multiprocess. | [Workers](../workers.py) use `MLXGemmaEngine` with the parent's model family and shared [prompt/stop helpers](../engines/translation_prompts.py). [Pipe tests](../tests/test_worker_translation_contract.py) cover both model families, EN/ES and the A/B contract. | **OPEN; ready after merge:** shared-engine option implemented and tested. Multiprocess remains optional; closure is not a performance endorsement. |
| [#177 — MLX MTP](https://github.com/wrbell/stark-translate/issues/177) | Greedy-equivalent output, canary ≥7/8, medium p50 ≤0.85× post-EOS E4B, acceptance ≥30%; otherwise MTP stays off. | [Recorded experiment](archive/v2026.13/MAC_LATENCY.md): 33/33 identical outputs, 31.3% acceptance, medium 1339/1393 ms (≈0.96×), missing the speed gate. [Live validation](../dry_run_ab.py) rejects requested/configured MTP before loading; [tests](../tests/test_worker_translation_contract.py) verify rejection. | **OPEN; ready after merge as a negative experiment:** keep MTP off and state the speed gate failed. The ≥7/8 canary promotion gate is not established. Fixed-prefix caching is a separate experiment, not MTP evidence. |

The [endurance monitor](../tools/endurance_monitor.py) supplies sampled process-tree
memory, queues, declared source bounds and terminal recording/cleanup evidence.
The three service cohorts remain separate:

- Original Standard `20260910_024936_300302_en` used the b65 wheel. It exited 0 and
  drained required writes, but **23/549 final spans and 97/2,720 preview spans failed**
  the source-bound audit. Those originals stay unchanged. The prepared 350-second
  slice was **not run**.
- Repaired Standard `20260910_043120_839144_en` used the installed 752 wheel, SHA-256
  `7477574d25c91739b6a88ca142a35bf36258599a66671b8dbb32237d1fa852b5`.
  It completed at 09:32:38.589810 UTC with exit 0: **563 consistent final spans,
  2,814 consistent preview spans and 7,594/7,594 required writes**, zero pending/failed,
  and observed pipeline/descendant cleanup. All 563 saved WAV-header durations agree;
  three selected waveform windows also matched their declared source sequence. This does not establish unselected or all-source coverage. Its one matched document-visible
  audience connection acknowledged all 563 finals and 2,813 nonempty translated
  previews. The native Mac was locked; these ACKs do not certify physical visibility.
- CPU Lite `20260910_053518_894101_en` completed on the same installed 752 wheel:
  **468 consistent final spans/WAV headers, 271 preview spans, all 1,979 writes**,
  exit 0 and cleanup. One matched connection ACKed all finals and previews, but
  first previews covered only 174/468 final utterances. Three selected waveform
  windows matched. Lite's sparse previews and large observed tails do not support
  recommending it as a fast production profile today. Keep profiles separate;
  this is no causal speed, natural-quality or hardware certification.

The [written endurance report](evaluation/overnight_endurance_20260910/README.md)
records Standard's prepared operator launch→first-server-preview interval of
**436.790088 seconds**, excluding unmeasured one-time installation/download.
Actual SPA Start→first-server-preview was **420.492–420.762 seconds**.
The selected input begins with **403.1448125 seconds of digital zeros**; first nonzero
PCM is not necessarily speech onset. This wall-clock interval is not caption latency
or isolated model-loading time, and first-install/download duration was not measured.
The retained natural-service hymn→prayer chronology supports the original input
workload without claiming all-source or semantic coverage. The last final window
ends at 3479.360 seconds, leaving 160.693 seconds unclassified; no exact reader EOF
record establishes what that remaining interval contains.

The post-Stop Standard summary task finished, and its actual UI disclosed that it
used the transcript beginning/end and omitted the middle. Successful generation is
not a human fidelity approval. Hymn fragments and the time→eternity translation
meaning error are mapped to [#193](https://github.com/wrbell/stark-translate/issues/193) and [#194](https://github.com/wrbell/stark-translate/issues/194) as remaining hymn/quality follow-ups. The
[verified archive](evaluation/overnight_endurance_20260910/raw/artifact-manifest.json)
and [immutable written evidence](https://github.com/wrbell/stark-translate/blob/0483f81a57a3ee689b51cda70ed0b8dc85e7926d/docs/evaluation/overnight_endurance_20260910/README.md) are available. Earlier operator UX blockers were fixed; no issue closure or merge is claimed.

Human, device, training and target-hardware gates remain pending until their own
evidence is recorded. Completing the laptop #134 rehearsal does not close #131,
#132, #133, #135 or #137, promote an experimental model/default, or publish a release.
