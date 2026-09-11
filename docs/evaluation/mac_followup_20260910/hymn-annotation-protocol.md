# Hymn suppression annotation and experiment preparation

[Issue #193](https://github.com/wrbell/stark-translate/issues/193) remains open.
A source-linked annotation packet and bounded opt-in protocol are prepared
locally; no natural labels, model results or new device validation were produced
by this preparation. No thresholds or model defaults changed. The separate
[source repairs](hymn-source-repairs.md) preserve accepted speech onset without
changing short-speech acceptance.

## Retained input and uncertain regions

The packet refers to the same natural English source as the
[installed Standard rehearsal](../overnight_endurance_20260910/fresh-standard-early/README.md),
session `20260910_043120_839144_en`, source `752ab9a`. Its previously verified
parent SHA-256 is
`8bec0f104dd883fd001f2a4b12ff9b454217c344f233458868feda07a0e83f53`.
Header-only inspection confirms mono PCM16, 16 kHz, 58,240,848 samples.
The existing lossless 400–750 s slice has recorded SHA-256
`7937eb4f0fc2b2c072add522d1b8ebcb65520cf197ff62c47c712a46f29f577e`.
Neither audio hash was recomputed during preparation; reverify before execution.

Eight coarse search regions cover source 400–780 s, with 21 historical caption
locators retained as IDs, bounds and text hashes. All acoustic/approval fields
remain uncertain or unset. The existing crop stops before the later closing
reply and next announcement; those regions require the parent recording.

| Search region | Parent source seconds | Meaning of locator |
|---|---:|---|
| Opening and possible singing onset | 400–430.112 | Announcement/title chronology only; acoustic boundaries unknown |
| Fragment cluster | 430.112–462 | Actual generated fragments locate a problem; they are not reference speech |
| Intervening context | 462–700 | Caption gaps do not establish music or silence |
| Possible spoken transition | 700–733.248 | Search around the later prayer chronology |
| Prayer and closing-reply search | 733.248–770.112 | Quiet speech and individual reply duration require listening |
| Next announcement | 770.112–780 | Check preservation of subsequent spoken material |

Annotation coordinates are original **16 kHz** samples. Historical pipeline
stamps are **48 kHz** transport samples; their rate must be retained when mapping
to parent positions. Crop offsets and virtual EOF padding must be accounted
separately. No audio, generated transcript, private correction note or incidental
room speech is copied into this public document.

The mixed-input Spanish microphone retest remains excluded: recording was off,
no source audio was saved, and upstream samples were lost. Its
[structural evidence](../tts_routing_20260910/README.md) cannot supply natural hymn,
quiet-prayer or Spanish-reference labels.

## Annotation and policy contracts

An independent listener should first label music, sung vocals, spoken speech,
overlap, language, quiet speech and uncertain intervals without seeing generated
captions. Whether a region should produce captions is a separate policy field.
Spoken onset/end estimates need sample bounds and uncertainty; individual valid
short replies must be located within larger chunks. Then compare predictions,
retaining disagreement, revisions and reviewer timestamps. Acoustic annotation
does not approve transcription, translation or training export.

The current entry heuristic requires 156 consecutive 32 ms frames with VAD false
and RMS strictly above 0.15: **4.992 s**, from nominal 5 s. Recovery requires
15 speech-positive frames: **480 ms**, from nominal 0.5 s. VAD probability must
be strictly above 0.3. Commit `26f854c` retains the first 14 accepted frames
previously discarded, preserving **448 ms** of onset. Unaccepted short bursts
remain suppressed; the minimum final duration stays 0.7 s. Pending/recovered
source states alone do not establish completed captions or acoustic recall.

## Bounded silent experiments

The smallest natural diagnostic is one unchanged E4B, real-time file replay of
the existing 350-second crop, with TTS explicitly disabled, recording disabled,
gain 1, a fresh session and a 600-second owned-process bound. Source/runtime/model
identities and hashes must be frozen first. Model work runs serially. The user's
current public-place restriction still prohibits microphone capture and all
speaker/output playback; listening and device work remain deferred.

Commit `4b0144d` adds the engineering fields to the existing opt-in `vad_complete`
event: actual boolean decision and already-computed RMS, source/processed sample
coordinates, thresholds, postclassification streak counters and hold state before
transition. It adds neither a VAD call nor an event. Raw VAD probability remains
unavailable. VAD elapsed time retains its call/worker-wait scope; the event
timestamp now follows classification bookkeeping. Recovery/source-contiguity
outcomes, rather than pre-transition counters alone, establish whether pending
frames were accepted. A controlled real-loop regression was added; the full
remote Python 3.11/3.12 suites and lint passed for `4b0144d`
([CI run](https://github.com/wrbell/stark-translate/actions/runs/34502608307)).
Natural diagnostic execution and acoustic labels remain pending.

After that diagnostic, compare one candidate at a time against unchanged
controls: RMS threshold 0.08 with holdoff 5 s, or holdoff 2 s with RMS 0.15.
These are exploratory values, not recommendations. Keep recovery/short-speech
policy, VAD threshold, cadence, gain, STT, Marian and prompts fixed. Do not add
short-word blacklists. Repeated VAD-positive singing can defeat both candidates;
retain that negative result rather than forcing a pass.

An economical screen uses unchanged 400–470 s and 700–780 s slices, both models,
control/candidate, three paired repeats: 24 serial runs per candidate. These
short slices are proposed, not yet extracted or hashed. Starting at 700 s resets
state, so a promising candidate still needs continuous 400–780 s confirmation.
Alternate model and arm order; never pool historical runtime results with these
paired controls.

Report unwanted partials/finals, independently labeled legitimate speech lost,
onset retained, short/quiet reply outcomes, first-partial delay, update gaps and
schema-2 final timing with endpoint counts. Distinguish unobserved capture gaps
from suppression, recovered buffers from delivery, and uncertain labels from
correctness denominators. Browser acknowledgements and physical visibility are
separate gates. Fewer fragments alone cannot qualify a change that loses valid
spoken material.

The local packet lives under
`.cache/mac-en-es-closeout/hymn-capture-preparation/`; its manifest, blank
annotations, locator index, generator and full protocol remain preparation.
It now also contains a runnable, source-reviewed one-control wrapper using the
actual replay supervisor and a 600-second owned-child bound. Tiny stdlib fixtures
cover source/VAD coordinates, EOF padding, suppression/recovery dispositions,
physical STT drain and fail-closed file/no-TTS guards. The wrapper has not run
the natural recording. It records engineering decisions and raw local evidence;
saved-audio waveform validation, human quality and audience delivery are outside
this recording-disabled diagnostic's claims.
Missing natural Spanish/short-reply inputs and independent acoustic/bilingual
review stay explicit completion dependencies. None requires CUDA, and none is
fulfilled by generated labels or this document.
