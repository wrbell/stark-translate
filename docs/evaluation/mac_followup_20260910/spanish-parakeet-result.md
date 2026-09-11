# Spanish Parakeet screen

All 18 real-time file replays completed with passing technical integrity on
`eddb0adb4741307680079802c47f33cb21d56aec`. **Neither of the two Spanish Parakeet
arms qualifies for confirmation.** Parakeet improved the median-only measure in
five of six paired comparisons, but every repeat failed another declared guard.
The Spanish Whisper default and Gemma E4B default remain unchanged.

The [frozen protocol](protocol/spanish-parakeet-screen.json) uses the same five
normalized public Spanish development recordings, 47.6 seconds total, for each
run. Opening and closing controls use mlx-whisper large-v3-turbo; the candidate
uses Parakeet MLX v3. E4B and E2B finals are evaluated separately across three
repeats. The Standard profile, Marian partial translation, 0.6-second partial
cadence, fixed gain 1 and the other settings are preserved. Runs explicitly use
file input and `--no-tts`; this is not a microphone, speaker or browser-visibility
test. The [full Standard screen](standard-screen-result.md) is a separate cohort.

## Median gains and rejected tradeoffs

The metric is server delivery of each opening control's frozen VAD-positive
source mask, timed from its estimated speech end. It is distinct from isolated
STT time, the full-buffer diagnostic and browser acknowledgment. Each row below
is a separate opening/candidate/closing comparison; values are milliseconds and
are not pooled across repeats or model sizes.

| Final model | Repeat | Opening p50 | Parakeet p50 | Closing p50 | Median-only gate |
|---|---:|---:|---:|---:|---|
| E2B | 0 | 3,194.7 | 1,692.5 | 3,020.7 | Pass |
| E2B | 1 | 2,640.8 | 1,911.2 | 2,389.4 | Pass |
| E2B | 2 | 2,284.1 | 1,801.5 | 2,752.3 | Pass |
| E4B | 0 | 5,444.1 | 3,915.7 | 3,384.9 | Fail against closing control |
| E4B | 1 | 3,882.6 | 2,467.5 | 3,483.8 | Pass |
| E4B | 2 | 3,224.4 | 2,103.9 | 3,239.7 | Pass |

The median-only gate requires at least 15% or 150 ms improvement against both
controls. Qualification also requires every repeat's other guards, so the
median improvements alone cannot select either arm.

| Final model | Repeat | Declared rejection reasons |
|---|---:|---|
| E2B | 0 | Matched first-preview tail regression |
| E2B | 1 | Final queue wait, matched update-gap tail and overall update-gap tail regressions |
| E2B | 2 | Matched and overall update-gap tail regressions |
| E4B | 0 | Median gain below gate; RSS, generation-lock queue tail, matched update-gap and overall update-gap regressions |
| E4B | 1 | Preview source coverage loss; final queue wait and disjoint-window growth; matched and overall update-gap regressions |
| E4B | 2 | Final queue wait; matched and overall update-gap regressions |

For E2B repeat 0, the one exactly matched first-preview observation takes
1,489.4 ms with Parakeet versus 1,294.1 ms in the closing control. The 195.3 ms
increase exceeds the unchanged max(100 ms, 5%) allowance. This is a small-sample
screening guard, not a population p95 estimate. E2B repeats 1 and 2 also worsen
update-gap tails, so this single observation is not the only reason the arm
cannot qualify.

For E4B repeat 1, candidate previews miss 2.70% of the source covered by either
control's previews, exceeding the two-percentage-point allowance. Final delivery
still covers the required source. Maximum final queue wait reaches 2,369.3 ms
versus 274.6 / 53.2 ms in the opening/closing controls. Disjoint first/last queue
windows grow by 786.9 ms versus 81.4 / 13.6 ms; these are short-screen observations,
not a long-service trend. Terminal queue work reaches zero.

E4B repeat 0 also raises process-lifetime RSS by about 384.2 MiB against its
closing control, above the 329.0 MiB allowance. The measurement does not establish
a memory leak or its cause. Its generation-lock queue tail is 153.5 ms against
34.1 ms in that control, exceeding the 100 ms allowance.

The final-delivery nearest-rank tails are lower than both controls in all six
comparisons; the tail failures above concern preview responsiveness and queueing.
Every run has only six eligible fixed-source anchors. The frozen screening guards
still apply, but **no p95 performance claim is supported**: at least 100 eligible
observations are required in each individual opening, candidate and closing run.
Repeats, endpoints, languages and models must not be pooled to reach that count.
None of the candidate medians is below one second.

## Model identity and quality scope

The report records no requested/actual runtime-identity errors in any of its six
triplets. Representative raw-cell checks confirm the controls actually loaded
`mlx-community/whisper-large-v3-turbo` at
`a4aaeec0636e6fef84abdcbe3544cb2bf7e9f6fb`, and the candidate loaded
`mlx-community/parakeet-tdt-0.6b-v3` at
`ed2b7e8c15f9aaa0b5772e2efb986255eaef7e15`. Requested/resolved paths, config hashes,
backend and Spanish source language agree; no primary startup substitution is
recorded. Gemma, Marian, VAD and the remaining runtime settings retain their
declared bindings. These are recorded local artifact identities, not a new
weight-download or full-weight audit.

All six comparisons use the same exact public reference contract. Production
caption WER is 1/80 words (1.25%) for Parakeet and 2/80 (2.50%) for each Whisper
control, so the declared WER guard passes. This includes segmentation and pipeline
corrections; it is not isolated engine WER. There are zero glossary opportunities,
so terminology recall is unavailable. Translation chrF is slightly lower with
Parakeet in both model scopes; it is descriptive and has no promotion threshold.
These five public read-speech recordings have no local bilingual approval and do
not certify church terminology or meaning quality.

## Evidence and decision

The [report](spanish-parakeet-normalized-v1/report.json), SHA-256
`2f26333f2532e144594916027ad3cefd094021c3d9cdb79f7fb4f621869beeff`.
The [selection](spanish-parakeet-normalized-v1/selection.json) records zero qualifiers. Independent retained-data review
reproduced that selection, recomputed all 18 per-run anchor distributions from the
reported anchors, and checked representative raw opening/candidate/closing cells
for runtime identity, references, preview responsiveness and source coverage.
This is not another model run or an exhaustive revalidation of every raw trace.
The [evidence index](spanish-parakeet-normalized-v1/evidence-index.json) binds
all 18 original results and their ten terminal artifacts per run in the compressed
raw archive. The [source index](spanish-parakeet-normalized-v1/source-index.json)
binds the frozen code, protocol and archiver. Original payloads are unchanged.

No confirmation or combination is authorized by either rejected arm, and the
earlier pending-state confirmation notes remain retained separately. A future
Spanish STT/admission revision needs a new declared experiment that preserves
preview coverage and responsiveness. This result does not authorize weaker guards,
replacement of unfavorable repeats or a production default change. Research
traces used capacity 131,072; ordinary tracing remains off by default. Human
quality, visible-browser timing and sub-second caption-delivery gates remain open.
