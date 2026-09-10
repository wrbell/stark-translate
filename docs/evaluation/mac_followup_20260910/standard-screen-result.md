# Standard EN/ES endpoint and deadline screen

All 96 real-time file replays completed with passing technical integrity on
`eddb0adb4741307680079802c47f33cb21d56aec`. **None of the 24 model/language arms
qualified for confirmation.** Defaults remain unchanged. Faster individual
results are retained alongside the properties they worsened.

The [frozen protocol](protocol/standard-screen.json) uses five normalized public
development recordings per language, E4B and E2B, three alternating-order repeats,
and opening/closing controls. Six settings independently test early clause cuts
(2/4 seconds with 160/240 ms pauses) and partial-STT deadline margins (100/250 ms).
This complete v2 cohort is separate from the
[aborted v1 and v4 shutdown pilot](normalized-replay-integrity-closeout/README.md).

## Measurement and controls

The primary metric is server delivery of each opening control's fixed
VAD-positive source mask, measured from its estimated speech end. Changing
segmentation cannot shorten that required mask. It is distinct from the original
full-buffer diagnostic, isolated translation time and browser acknowledgment.
No microphone, speaker or browser-visibility test is part of this screen.

| Model | Direction | Opening/closing per-run fixed-source p50 range |
|---|---|---:|
| E4B | EN→ES | 2,149.9–2,894.7 ms |
| E4B | ES→EN | 3,120.9–4,364.8 ms |
| E2B | EN→ES | 1,498.2–2,449.5 ms |
| E2B | ES→EN | 2,356.1–3,422.9 ms |

These ranges describe six control runs per cell, not a pooled median or a
comparison with earlier sermon clips. Each run has only six eligible frozen
anchors: nearest-rank tails enforce the predeclared screening guard but do not
support a p95 performance claim. Repeats, endpoints and languages are never pooled
to reach the required 100 eligible observations per individual run.

## What improved, and what failed

Half of the 72 candidate/control comparisons passed the median-only requirement
(at least 15% or 150 ms against both controls). None passed every guard; all 24
arms therefore fail the required three-repeat selection. No confirmation or
combination is scheduled from this Standard screen.

| Example | Fixed-source candidate p50 across three repeats | Reason it cannot qualify |
|---|---:|---|
| EN E2B, early 4 s / 160 ms | 1,326.1 / 1,273.3 / 1,324.4 ms | Median gain passes each repeat, but opening-control preview coverage loses 3.77–3.85%; repeat 0 also exceeds the RSS guard. |
| EN E2B, early 2 s / 160 ms | 1,299.6 / 1,081.3 / 965.0 ms | WER rises from 6.86% to 8.82%, with preview coverage and responsiveness failures. One sub-second result does not meet the delivery goal. |
| ES E2B, early 2 s / 160 ms | 1,507.9 / 1,520.9 / 1,521.8 ms | Opening-control preview coverage loses 30.61–34.95%, with first-preview regression in every repeat. |

For the closest EN E2B 4 s / 160 ms example, repeat 0 omits 1.216 seconds of
preview source, including 0.960 seconds of opening-control VAD-positive speech.
Its final captions cover that speech. This is an observed preview tradeoff, not
missing final speech or only omitted trailing silence. RSS rises by about
678 MiB against its opening control, exceeding the frozen allowance; this alone
does not establish a leak or its cause.

An ES E2B 4 s / 240 ms repeat emits only one translated preview and reaches
12,255 ms maximum final queue wait. Its physical STT calls and required writes
still drain completely. Update-gap evidence is correctly unavailable when there
is only one preview; process completion does not establish responsiveness.

The independent review reproduced all 72 pair decisions and 24 arm selections,
checked 216 anchor distributions and inspected three representative raw runs.
It found no reporter/selector defect that justifies replacing these negatives.
This was retained-data analysis, not a second model execution or exhaustive raw
trace revalidation.

## Evidence and next steps

The [report](standard-screen-normalized-v2/report.json), SHA-256
`67bfa237f21f04440b807c6a5da5e1049e5c4aea32f35d6a162d658c8719b48e`.
The [selection](standard-screen-normalized-v2/selection.json) records zero qualifiers.
The [evidence index](standard-screen-normalized-v2/evidence-index.json) binds all
96 original results and their ten terminal artifacts per run in the compressed
raw archive. The [source index](standard-screen-normalized-v2/source-index.json)
binds the frozen code, protocol and archiver. Original payloads are unchanged.

The separate [Spanish Parakeet screen](spanish-parakeet-result.md) also rejected
its candidates; CPU Lite follow-ups retain their own declared hypotheses. Any later early-cut revision must explicitly test preservation of
preview coverage and tail behavior. This result does not authorize weaker guards,
repeating an unfavorable run to replace it, or changing production defaults.

Public read-speech references are not church or bilingual approval. Production
caption WER includes final segmentation and corrections; it is not isolated STT
WER. These ten selected recordings contain zero glossary opportunities, so term
recall is unavailable. Human meaning review, untouched confirmation for any
future qualified candidate and the sub-second caption-delivery goal remain
unmet.
