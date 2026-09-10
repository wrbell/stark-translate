# Why first-preview E4B does not warrant ordinary confirmation

The completed screen selected **0/28 experiment/model arms**. `first_preview/E4B`
is one of those negative results. Starting the first partial at 0.35 seconds did
not produce a consistent gain against both controls, and other endpoint/tail
guards failed. E4B defaults remain unchanged; no confirmation or combination is
scheduled from this arm.

The [exact paired evidence](screen-decision.json) is bound to the original
[full analysis](artifact-manifest.json). All figures below are milliseconds.
Each hard-cut entry has **n=1**; each first-visible paired-median entry has **n=7**.
Paired-median deltas are computed from matched utterance differences, not by
subtracting independently computed medians.

| Repeat | Hard-cut opening | Candidate | Closing | Candidate − opening / closing | First-visible paired-median Δ, opening / closing |
|---|---:|---:|---:|---:|---:|
| r0 | 2,725.7 | 2,844.7 | 2,586.8 | +119.0 / +257.9 | +51.4 / +94.7 |
| r1 | 1,941.4 | 2,003.4 | 2,579.9 | +62.0 / −576.5 | −316.4 / −804.9 |
| r2 | 2,634.9 | 1,979.5 | 2,645.1 | −655.4 / −665.6 | −102.9 / −773.1 |

Only r2 improves hard-cut timing against both controls. In r1, the closing control
is 638.5 ms slower than the opening control: the candidate appears faster against
the closing control while remaining 62 ms slower than the opening control.
The pooled improvement is therefore not a consistent paired win. The stronger
closing-control preview deltas also need this drift context; they do not override
failed repeat and tail guards.

All 588 candidate final comparisons against opening controls, and all 588 against
closing controls, retained identical English transcript and Spanish translation
text. These repeated matches are not reference-quality approval. Early E2B
previews repeatedly changed `Currently` / `Actualmente` to `Current.` / `Corriente.`
in all three repetitions before the identical final, showing a real provisional
wording tradeoff.

The matrix contains 670 matched final ACKs out of 672; both missing ACKs belong to
`async_captions/E4B`. No observed preview ACK follows its final ACK. Those facts do
not establish what remained visible on a physical display: the Mac was locked
while the browser DOM reported visible. Recorded silence and EOF-padding-assisted
finals remain separate analytical endpoints. Package-window overlap is annotated
in [provenance](provenance.json), with every affected run retained.

Standard and CPU Lite endurance are separate functional/stability cohorts. New
[endpoint/deadline/readback hypotheses](../../latency_next_experiments.md) require
new evidence; this note does not authorize a default change or reopen rejected
arms as ordinary confirmations.
