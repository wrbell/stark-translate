# CPU Lite independent deadline screen — September 10, 2026

All **24 real-time file replays passed technical integrity; none of the four
language/margin arms qualifies for confirmation**. Some final-delivery medians
improve, but every candidate repeat fails another required guard. CPU Lite keeps
Whisper-small, Marian finals and the default 0.6-second partial cadence; deadline
admission remains opt-in. No confirmation or cadence/deadline combination follows
from these results.

The separate [English](lite-deadline-normalized-v2/en/protocol.json) and
[Spanish](lite-deadline-normalized-v2/es/protocol.json) protocols compare 100 ms
and 250 ms partial-admission margins with opening and closing controls, repeated
three times. Each language has 12 distinct runs and six paired comparisons;
controls are shared between candidates. Every arm retains 0.6-second cadence.
Each input combines five normalized public development recordings, lasting
46.94 seconds EN and 47.6 seconds ES. Commands explicitly use file input,
`--gain 1` and `--no-tts`.

All runs use frozen source `eddb0adb4741307680079802c47f33cb21d56aec` and the actual
`lite-cpu` route: Whisper-small CT2 int8, beam 1, three STT threads, one STT worker,
Marian CT2 int8 partials **and finals**, one Marian thread and ONNX Silero. The
generic harness's `e2b` label does not mean Gemma ran. Requested and actual runtime
bindings agree in all 24 raw results.

## Delivery and responsiveness

The metric is server delivery of the opening control's fixed VAD-positive source
mask, timed from its estimated speech end. Candidate values below are individual
run medians in milliseconds, ordered by repeats 0/1/2. The median-only gate is
at least 150 ms or 15% improvement against **both** controls; qualification also
requires that gain in at least two repeats and every repeat's other guards.

| Language | Margin | Candidate p50, repeats 0 / 1 / 2 | Median-only passes | Actual deferrals, repeats 0 / 1 / 2 |
|---|---:|---|---:|---|
| EN | 100 ms | 2,946.1 / 3,075.8 / 2,574.2 | 1/3 | 0 / 0 / 0 |
| EN | 250 ms | 2,350.4 / 3,065.4 / 2,674.5 | 2/3 | 0 / 3 / 3 |
| ES | 100 ms | 2,998.7 / 3,317.6 / 3,455.4 | 1/3 | 0 / 0 / 0 |
| ES | 250 ms | 3,063.0 / 3,264.1 / 2,793.9 | 2/3 | 0 / 0 / 0 |

English's six comparisons all fail the STT stage-queue tail guard; five fail the
final-delivery tail guard, two lose preview source coverage, two fail memory and
one fails the overall first-preview tail guard. Spanish has four final-delivery
tail failures, four disjoint final-input-queue growth failures, three preview
coverage failures, three stage-queue tail failures and two maximum final-input
queue-wait failures. Its first-preview, matched-first-preview, update-gap and
memory guards also reject individual repeats. These counts overlap.

For English 250 ms repeat 2, the median improves to 2,674.5 ms from
3,030.6 / 2,976.4 ms, but its final-delivery tail is 10,763.1 ms against the closing
control's 8,851.7 ms. Peak RSS is 1,879.8 MiB versus 1,521.5 / 1,546.0 MiB,
exceeding the max(256 MiB, 10%) allowance. Six candidate previews cover 12.768
seconds in total, compared with five closing-control previews covering 11.552
seconds, yet they miss **0.608 seconds (5.26%) of that control's source spans**.
More previews or greater total coverage cannot establish coverage of the same
source. The allowance is 2%.

Spanish 250 ms repeat 1 similarly has six previews versus five in its opening
control, but misses 0.608 seconds (5.26%) of the opening preview source. Its
maximum final-input queue wait is lower, 864.0 ms versus 1,574.7 / 1,742.3 ms;
the separate disjoint queue-growth metric still regresses against the opening
control: +287.9 ms versus −209.9 ms. Final-input waits are distinct from the STT
stage-queue diagnostic.

Controls also vary. Closing-minus-opening median drift ranges from −9.63% to
+19.08% EN and −4.22% to +13.04% ES. These observations remain in the report;
they do not waive the requirement to pass against both controls.

Every run has **six eligible source anchors**, below the required **100 per
individual run** for a p95 claim. Small-sample screening tail guards still apply;
their nearest-rank values do not establish population p95 performance. Repeats,
languages and shared controls are not pooled. No candidate median is sub-second.

The session-local predictor needs three observations in a two-second audio-length
bin before estimating partial runtime. Cold bins admit work. Immediately before
partial STT, the worker checks the predicted finish plus margin against the
utterance's eight-second maximum-duration deadline. Running inference is not
preempted.

Only English 250 ms repeats 1 and 2 exercised actual deferral. All Spanish
candidates recorded finite predictions as well as cold admissions, but skipped
no partials. Median differences in runs with zero deferrals cannot be credited
to omitted late partial work. Research traces were enabled with capacity 131,072
and **zero discarded events in every run**; the ordinary default remains off.

## Integrity, quality and retained evidence

All raw runs finish six Marian finals, pass recorded source/EOF accounting,
complete required persistence, drain final input queues and report no runtime
identity errors or recorded capture loss. Preview-source regressions therefore
must not be described as missing captured input. This is file replay evidence,
with no microphone, speaker, physical-browser or human-quality certification.

Caption WER is identical across controls and candidates: 11/102 words (10.78%) EN
and 0/80 ES. Translation chrF is also unchanged, 54.68 EN→ES and 63.44 ES→EN.
There are no glossary opportunities. These short development references remain
evaluation-only and lack local bilingual approval. The separate
[50-item-per-language CPU model comparison](cpu-stt-comparison.md) uses beam 5
and four threads; it does not supply production latency or approval for these
beam-1, three-thread runs. It rejected both base-model alternatives, and the
[cadence screen](lite-cadence-result.md) rejected all four cadence arms.

The [English report](lite-deadline-normalized-v2/en/report.json) has SHA-256
`c3394548708908ed6f93a3369af16f836534aae8b764e490c2d7f27aaf2bc3af`; the
[Spanish report](lite-deadline-normalized-v2/es/report.json) has SHA-256
`e19fe8f0b1883e13de3168094920563291b7b39dbda2b286e9d3770472858cfc`.
The separate [English](lite-deadline-normalized-v2/en/selection.json) and
[Spanish](lite-deadline-normalized-v2/es/selection.json) selections retain all
rejections. Independent review reproduced all 12 paired scores and both selections
from all 24 raw JSON results, including per-run anchor counts, caption quality,
source accounting and physical admission traces. It made no inference calls.

The [English evidence index](lite-deadline-normalized-v2/en/evidence-index.json),
[Spanish evidence index](lite-deadline-normalized-v2/es/evidence-index.json) and
[source index](lite-deadline-normalized-v2/source-index.json) retain raw payload,
terminal-artifact, coordinator, frozen-source and independent-review bindings.
Public report bytes match the reviewed originals. This independent review did
not reopen compressed archives; archive readback is a separate closeout check.
The prior failed recovery and cadence cohorts remain separate. There is no
qualified deadline, cadence or smaller-STT arm to combine or confirm.
