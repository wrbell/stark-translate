# CPU Lite cadence screen

All 24 real-time file replays completed with passing technical integrity on
`eddb0adb4741307680079802c47f33cb21d56aec`. **None of the four language/cadence
arms qualifies for confirmation.** The 1.2-second cadence improves the final
delivery median in all six comparisons, but every candidate comparison loses too
much preview coverage. The default 0.6-second cadence remains unchanged.

The [frozen protocol](protocol/lite-screen.json) compares 0.9- and 1.2-second
partial intervals with opening and closing 0.6-second controls, repeated three
times separately for English and Spanish. Controls are shared between cadence
candidates: 12 paired comparisons comprise 24 distinct runs. Each language uses
five normalized public development recordings: 46.94 seconds EN, 47.6 seconds ES.
Runs explicitly use file input, fixed pipeline gain 1 and `--no-tts`.

**These are CPU Lite runs with Whisper-small CT2 int8 and Marian CT2 int8
translation, including finals.** They use ONNX Silero, three STT threads, one STT
worker and one Marian thread. The `e2b` size and `gemma4` family strings retained
by the generic harness/configuration are labels; no Gemma model or generation
route was used. This cohort does not measure the Gemma CPU quality profile.

## Measured tradeoffs

The metric is server delivery of the opening control's frozen VAD-positive source
mask, timed from its estimated speech end. Values below are separate per-run
medians in milliseconds, ordered by repeats 0/1/2. They are not pooled. The loss
column gives the range of six individual comparisons against opening/closing
preview coverage, not loss of capture or final source coverage.

| Language | Candidate cadence | Candidate p50, repeats 0 / 1 / 2 | Median-only gate | Control preview source missing |
|---|---:|---|---:|---:|
| EN | 0.9 s | 2,597.4 / 2,861.8 / 5,342.5 | 0/3 | 39.47–84.74% |
| EN | 1.2 s | 1,943.5 / 1,971.0 / 1,944.4 | 3/3 | 4.76–33.33% |
| ES | 0.9 s | 2,999.7 / 3,039.8 / 3,474.9 | 0/3 | 9.47–34.21% |
| ES | 1.2 s | 2,531.8 / 2,493.7 / 2,892.5 | 3/3 | 12.00–38.38% |

The median-only gate requires at least 15% or 150 ms improvement against both
controls. Selection additionally requires every repeat's other guards to pass.
All 12 comparisons exceed the 2% preview-source-loss allowance. Nine fail a
first-preview tail guard and eight fail an update-gap tail guard. Five fail the
final-delivery tail guard; six fail disjoint final-queue growth, five fail maximum
final-queue wait, and two fail a stage-queue tail guard. These counts overlap.

For English 0.9-second repeat 2, previews fall to **1**, versus **7 / 6** in the
opening/closing controls. Unique preview-covered source falls to 2.784 seconds
from 18.240 / 12.768 seconds. No candidate update gap or exactly matched
first-preview observation exists; missing responsiveness is not scored as zero.
Maximum final-input queue wait rises to 353.60 ms from 0.10 / 0.69 ms. All three
runs still finish six finals with complete recorded source accounting and no
recorded capture loss. A preview regression must not be described as lost input.

For Spanish 1.2-second repeat 1, the median improves to 2,493.7 ms from
2,979.3 / 2,831.2 ms, but preview coverage loses 28.00% / 38.38%. The candidate's
first-preview nearest-rank tail is 5,520.5 ms versus 4,078.3 ms in the opening
control; its update-gap tail is 2,727.1 ms versus 1,970.5 / 1,962.7 ms. These exceed
the unchanged max(100 ms, 5%) allowance. Fewer partial requests therefore do not
establish a better caption experience, despite faster final medians.

Every run has **six eligible fixed-source anchors**, below the required 100
observations per individual run for a p95 claim. The predeclared small-sample
screening guards still apply. Repeats, languages, endpoint types and shared
controls cannot be pooled to qualify p95. No candidate median is sub-second.

## Identity, quality and decision

The report records no requested/actual runtime-identity errors. Representative
raw triplets in both languages confirm only STT and Marian model lifecycle roles,
six Marian final routes per run and no Gemma requests. Whisper-small uses the
recorded Systran revision `536b0662742c02347bc0e980a01041f333bce120`; the local
Marian exports bind EN→ES source revision
`5bc4493d463cf000c1f0b50f8d56886a392ed4ab` and ES→EN revision
`c96e2c5399ebfae4fc43d9669556b9afa74bb69d`. These are retained artifact identities,
not a new full-weight audit.

Production caption WER is unchanged across controls and candidates: 11/102 words
(10.78%) EN and 0/80 ES. Translation chrF is also unchanged, 54.68 EN→ES and 63.44
ES→EN, and is descriptive. These measures include segmentation and pipeline
corrections. The public references are evaluation-only and lack local bilingual
approval; zero glossary opportunities leave terminology recall unavailable.
Unchanged short-clip scores do not certify church meaning quality.

The [report](lite-cadence-normalized-v1/report.json), SHA-256
`852c44a45a94b3b4147e2b59fbccd5cb6a86b3ddb98b4c7f22cb91768178b17f`, and
[selection](lite-cadence-normalized-v1/selection.json) retain all rejections.
The [run summary](lite-cadence-normalized-v1/run-summary.json) records 24 completed
runs, six finals each and complete recorded source accounting. Independent
retained-data review checked the report binding, per-run anchor distributions
and representative raw triplets; it did not rerun inference or verify compressed
archive contents. The [evidence index](lite-cadence-normalized-v1/evidence-index.json)
and [source index](lite-cadence-normalized-v1/source-index.json) retain the original
payload and source bindings.

No rejected cadence arm can enter confirmation or a cadence/deadline combination.
Independent CPU Lite deadline screens retain 0.6-second cadence and require their
own same-language guards; Standard results cannot qualify a Lite arm. Research
traces used capacity 131,072. This screen supplies no microphone, speaker,
visible-browser or human-quality validation; sub-second delivery remains
unachieved.
