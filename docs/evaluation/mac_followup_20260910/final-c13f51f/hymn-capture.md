# Natural English hymn-region control: technical diagnostic

The unchanged Standard control completed its 350-second file diagnostic on
`c13f51f1346b1581079b3457e34cb4d5fd0c2565`. Independent retained-data checks
confirm complete source accounting and normal inference/write cleanup.
**Music hold never activated**, so this run does not validate natural speech
recovery from a hold and does not close issue #193. No acoustic labels or
translation references were approved.

Session `hymn193_20260910_191530_7ced9384_en` ran on September 10, 2026,
19:15:30–19:21:44 UTC. Its input was the existing parent 400–750-second slice,
SHA-256 `7937eb4f0fc2b2c072add522d1b8ebcb65520cf197ff62c47c712a46f29f577e`.
The wrapper reverified the parent and crop hashes before and after execution.
The independent review read retained records only, without rereading audio or
model weights. Raw input/output text and audio are not copied into this note.

The run used English Parakeet revision
`ed2b7e8c15f9aaa0b5772e2efb986255eaef7e15`, E4B OptiQ revision
`6be2eaf50f08b88b5dadca0f4a02c0a0880e7bd3`, existing Marian CT2 int8 and the
unchanged legacy routing policy. Thirteen finals requested Gemma; five used
the existing Marian final route. Gain remained 1, VAD threshold 0.3, RMS
threshold 0.15, nominal holdoff 5 seconds, silence threshold 0.5 seconds and
partial cadence 0.6 seconds. File input, no TTS and no audio recording were
explicit. There was no microphone or output-device activity.

## What the records establish

All 22 independent structural checks passed. The retained trace has 44,686
events with no truncation, including exactly 11,000 matched dequeue/VAD pairs.
Their source spans cover all 16,800,000 callback-rate samples at 48 kHz exactly
once: 350 source seconds. A separate 96,000 padding samples represent the two
virtual EOF seconds. Each processed frame has 512 samples at 16 kHz.
Mapping to parent 16 kHz samples is `6,400,000 + replay_sample / 3`; this is
time correspondence, not an assertion of resampled waveform equality.

| Recorded disposition | Source seconds | Interpretation |
|---|---:|---|
| Machine non-speech observation | 272.32 | VAD/buffering decision, not an acoustic label |
| Final-ready spans, 18 outputs | 67.84 | Exact final-ready/diagnostic/CSV/physical-STT joins |
| Empty-STT final rejection, 4 spans | 8.72 | Classified rejection; does not establish absence of real speech |
| Short-silence discard, 2 spans | 1.12 | Existing minimum-length policy; no human validity judgment |

The latter three rows exactly account for the 77.68 seconds admitted to the
speech buffer. Both the observed-source union and final-disposition union
cover the full source without gaps, overlaps, unknown bounds or unfinished
dispositions. The 18 finals ended at 10 silence boundaries, four smart cuts
and four hard cuts. Their English and translated text matched the CSV using
the real diagnostic `spanish_gemma` → CSV `spanish_a` mapping; text is omitted
from this report.

All 84 partial and 22 final physical STT calls have matching finishes and source
bounds; all 84 physical partial-result records were retained. Sixty-three
previews were persisted. Physical work and persisted previews are different
counts: empty results, cancelled publication and existing filters can prevent
publication after inference. Nine partial and four final empty-STT records
are explicit; the review does not invent a disposition for every unpublished
preview. All 114 required write jobs completed, none failed or remained
pending, and the owned process exited without forced cleanup.

## Why the existing hold stayed inactive

The [current production heuristic](https://github.com/wrbell/stark-translate/blob/c13f51f1346b1581079b3457e34cb4d5fd0c2565/dry_run_ab.py#L4875)
requires 156 consecutive 32 ms frames with VAD false **and** RMS strictly above
0.15. Its observed maximum was only 48 frames, or 1.536 seconds, at parent
603.104–604.640 seconds. Entry streaks reset 724 times on low-RMS VAD-negative
frames and 14 times on VAD-positive frames. RMS reached 0.317, but isolated
high levels do not satisfy the sustained 4.992-second conjunction.

There were no hold entries, exits, tentative recovery spans, recovered onset
events or short-burst suppression while held. The accepted-onset repair from
`26f854c` remains covered by controlled tests; this natural control did not
exercise it. VAD booleans and RMS do not establish which intervals contain
singing, legitimate quiet speech, overlap or silence. Raw VAD probabilities
were not recorded.

Arithmetic on the **fixed observed decisions**, without running another model,
narrows the proposed follow-up: RMS 0.15 with nominal 2-second holdoff still
has no qualifying streak (48 observed versus 62 required frames). RMS 0.08
with the original holdoff has a maximum 172-frame streak; its first eligible
entry frame ends at parent 605.216 seconds. This is a feasibility calculation,
not a candidate replay, subsequent hold/recovery prediction or recommendation.
Actual VAD state can change after a candidate enters and later leaves hold.

## Remaining acceptance

Independent listening must first label the prepared natural search regions,
including uncertain boundaries, legitimate short/quiet replies and spoken
onsets. Then the bounded opt-in threshold comparisons can measure unwanted
captions against speech lost, followed by continuous transition confirmation.
Natural Spanish examples and bilingual review remain separate inputs.

This run had zero WebSocket connections. The existing
[broadcast path](https://github.com/wrbell/stark-translate/blob/c13f51f1346b1581079b3457e34cb4d5fd0c2565/dry_run_ab.py#L3978)
skipped audience sends, so persisted captions do not establish visible delivery,
browser acknowledgments or latency certification. Recording was disabled;
saved-WAV integrity is inapplicable. This control makes no acoustic recall,
meaning-quality, live-microphone reliability, speaker-output, p95 or default
promotion claim.
