# Independent Standard96 review

The zero-qualified result is supported. No reporter/selector correctness blocker was found in this bounded review. Preserve the negatives; they do not justify changing defaults or weakening the frozen gates.

## Evidence and independent checks

- Report: `.cache/mac-en-es-closeout/standard-screen-normalized-v2-report.json`, SHA-256 `67bfa237f21f04440b807c6a5da5e1049e5c4aea32f35d6a162d658c8719b48e`; selection references those exact bytes.
- Frozen runner: clean `eddb0adb4741307680079802c47f33cb21d56aec`, source identity `e22411047861089b919965328dcc24ba9e06b170e4c0fe28b280a81436a20047`. Current reporter and selector have no diff from this revision.
- The retained spec equals `docs/evaluation/mac_followup_20260910/protocol/standard-screen.json`. Independent filename reconstruction finds exactly 96 expected runs: two languages, two models, three repetitions, opening/closing controls and six experimental settings.
- All 72 comparisons are rejected; all 24 arms have three unique repeats; none has an all-guard passing repeat. Independent stdlib arithmetic checked 216 fixed-anchor distributions, the latency/memory/preview/WER and preview-tail reasons in all 72 comparisons, and all 24 selector decisions. See `checks.json`. This was analysis of retained JSON, not new runtime testing.
- Read production `score_pair`, preview comparators, runtime/inventory checks and `select`; their behavior agrees with the published protocol. In particular, the unsuccessful third repeat may fail only the median-gain gate, and all controls must be respected. Final p95 eligibility is false everywhere: only six frozen eligible anchors per run. Nearest-rank tails still serve the declared screening guard; they do not certify p95 performance.

No full raw-cohort reanalysis, source/model rehash, audio read, compression, inference or device operation was performed. Three representative raw JSONs were inspected directly: EN E2B r0 opening control, EN E2B r0 early 4s/160ms, and ES E2B r0 early 4s/240ms. The full report's completion/identity results supply the remaining-run scope; this review is not a second exhaustive physical-trace audit.

## Negatives that matter

**Half of the candidate repeats pass the median-only gate (36/72).** Do not summarize this as “nothing was faster.” None preserves every other required property.

| Arm / language | Fixed-source p50, opening → candidate → closing (ms), repeats 0 / 1 / 2 | Decision |
|---|---|---|
| E2B early 4s/160ms, EN | 1502.3 → 1326.1 → 1918.0 / 1658.9 → 1273.3 → 1498.2 / 1530.1 → 1324.4 → 2449.5 | Median gain passes all three. Opening preview coverage loses 3.77–3.85% in all three; r0 also fails RSS. No confirmation qualification. |
| E2B early 2s/160ms, EN | 1502.3 → 1299.6 → 1918.0 / 1658.9 → 1081.3 → 1498.2 / 1530.1 → 965.0 → 2449.5 | Median gain passes all three, but WER rises from 6.86% to 8.82% and preview coverage fails. The one sub-second number is not goal completion. |
| E2B early 2s/160ms, ES | 2426.2 → 1507.9 → 2633.7 / 2378.9 → 1520.9 → 2356.1 / 2368.4 → 1521.8 → 3422.9 | Median gain passes, but opening preview loss is 30.61%, 34.95%, 34.95%, with first-preview regression in all repeats. |

For the closest EN E2B early 4s/160ms r0 result, actual missing preview intervals are `[1551360,1580544)` and `[1920000,1949184)` at 48 kHz: 58,368 samples / 1.216 seconds. Of these, 46,080 samples / 0.960 seconds are opening-control VAD-positive speech. The candidate's finals cover that speech. Therefore this is a real loss of translated preview coverage, **not missing final speech or merely omitted trailing silence**. Candidate RSS peaks at 3,615,096,832 bytes versus opening 2,904,358,912 bytes; the 710,737,920-byte increase exceeds the declared allowance. This observation does not establish a memory leak or its cause.

The ES E2B early 4s/240ms r0 `missing_preview_responsiveness` reason is also real: one emitted translated preview supplies zero within-utterance update gaps. The actual queue reaches three pending requests and 12,255.4 ms maximum wait; observed final STT takes 2,730–6,490 ms. The run completes with 12/12 physical STT calls represented, 15/15 required writes, zero terminal queue work and zero trace truncation. Completed lifecycle does not erase the responsiveness failure. No evidence here identifies a lost canceled wrapper, malformed metric or stale source as its cause.

## Scope and next steps

1. Archive and report this cohort unchanged, including per-arm/repeat guard reasons and opening-endpoint partitions. Do not launch confirmation or combinations from these rejected Standard arms. The selector's no-qualified artifact is the correct next-stage decision.
2. Continue the already declared Spanish STT and CPU Lite independent screens, separately scoped. Their different engines and cadence hypotheses cannot be inferred from this Standard result.
3. If revisiting early finalization later, start a **new development hypothesis** about preserving translated preview coverage and tail behavior. The closest EN E2B arm is useful diagnosis material, not a selected winner. Inspect finalization/cancellation scheduling and actual missing-preview intervals before changing code; these results alone do not prove a correctness defect that requires repair.
4. Retain deadline 100/250ms arms as negatives. Trace-led investigation of unusually slow Spanish STT/queue intervals may inform a new scheduling experiment, but these observations do not prove GPU/GIL/memory causation. Do not rerun the outlier merely to replace an unfavorable result.
5. These five-recording-per-language, normalized public development clips are read speech. Every reference glossary denominator in this screen is zero; theological-term recall is unavailable. WER is concatenated production-final WER after corrections, not independent STT-engine WER; chrF is descriptive and human meaning review remains pending. Controls reused within a repetition are not independent extra observations.
6. Fixed-source server delivery is not browser ACK latency. The inspected raw runs have zero browser clients. Retain the distinction from full buffered-span diagnostics, omitted VAD-negative silence, physical visibility, live capture, TTS, church speech, human approval and untouched confirmation. No source/default promotion or broader service certification follows.

The user ban on microphone capture and output-device playback remains in force for this session. The review and proposed next stages require no device operations.
