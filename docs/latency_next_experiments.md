# EN↔ES latency: experiments after the overnight screen

**2026-09-10 update:** the normalized [Standard screen](evaluation/mac_followup_20260910/standard-screen-result.md)
completed 96 runs with 0/24 model/language arms qualified; the
[Spanish Parakeet screen](evaluation/mac_followup_20260910/spanish-parakeet-result.md)
completed 18 runs with 0/2 arms qualified; the
[CPU Lite cadence screen](evaluation/mac_followup_20260910/lite-cadence-result.md)
completed 24 runs with 0/4 language/cadence arms qualified. Their retained results
do not authorize confirmation or combinations of the rejected arms. Gemma E4B,
Spanish Whisper and the 0.6-second partial cadence remain unchanged.

The earlier [overnight screen](evaluation/overnight_screen_20260910/README.md)
tested 15 configurations on E4B and E2B: 96/96 valid runs and 0/28 selected
experiment/model arms. That historical cohort and its stage observations below
remain separate from the normalized [EN↔ES follow-up](evaluation/mac_followup_20260910/README.md).
Public Spanish read-speech references are available for engineering comparisons;
church references and bilingual review remain required for quality certification. The [completed Standard and Lite endurance audits](evaluation/overnight_endurance_20260910/README.md)
confirm consistent retained spans and durable completion on `752ab9a`. Lite's
sparse translated previews and large observed tails do not support a fast-production
recommendation. These functional runs do not promote a screen arm, establish a
causal speed improvement or satisfy quality/hardware certification.

## Historical stage observations

On frozen source `911f4ae`, the first two opening-baseline smart-cut finals wait
about 3,085 and 2,829 ms between the selected earlier speech boundary and the VAD
submission decision. Both model sizes show this delay. A faster translation model
cannot remove time spent before transcription starts.

STT also has a variable tail. E4B opening-baseline chunk 6 takes 1,947 ms inside
STT, versus 552 ms for identical audio bounds in the closing control. In the
latest-partial arm, chunk 2 queues for 1,370 ms before a 128-ms STT call, but chunk
7 still takes 2,310 ms inside STT with negligible queueing. These are individual
observations from repetition 0, not a selection result or an estimate of average
improvement. Full comparisons must retain every repetition and both controls.

There is no Python inference mutex in [Parakeet](../engines/parakeet_mlx_engine.py)
to remove. [Gemma's lock](../engines/mlx_generation_lock.py) protects its translation
model. The inspected final STT calls start after the previous final translation
ends, so these records do not establish that overlap as the cause. Untraced warmup
and concurrently executing partial work remain possible contributors. STT-reported
latency matches wrapper call wall time closely; returning from the worker to the
event loop contributes at most 2.8 ms in the inspected records.

## CPU Lite follow-up

The completed CPU hour already used beam size 1, three STT threads with one
worker, one Marian thread and a 0.6-second partial interval. Its recorded STT
admission waits and roughly 1.3–1.5-second STT-call medians dominate; Marian's
silence-final translation median was 58.8 ms. A bounded partial-admission/cadence
trial should preserve translated-preview coverage, final-tail latency and meaning
while measuring any gain. Greedy decoding is already enabled, and adding E2B
does not address the CPU STT queue. The completed
[cadence comparison](evaluation/mac_followup_20260910/lite-cadence-result.md)
rejected both slower intervals because preview coverage and responsiveness did
not meet the guards, even where final medians improved. Preview coverage loss
does not imply capture or final-source loss.

Independent CPU Lite deadline screens remain pending at the unchanged
0.6-second cadence; no rejected cadence arm enters a combination. The separate
CPU Whisper small/base quality recovery also remains pending, using 50 public
development recordings per language in three repeats. Its isolated STT scores
cannot establish live caption latency or qualify a production model change.

## First improve attribution

The follow-up now records a common monotonic clock for capture, VAD, worker
admission, physical STT calls and final delivery. Source coverage and actual
EOF accounting accompany the trace; native inference drains before terminal
summaries. The separate sampled Parakeet profile records exact model inputs,
encoder/decoder work and scalar readback waits with an alternating unprofiled
control. Its observed overhead is retained in the linked report.

The later [capture loss accounting repair](evaluation/mac_followup_20260910/capture-loss-accounting.md)
on `d7ed43d` separates measured worker FIFO loss from PortAudio overflow with an
unknown sample count and reconciles losses at shutdown. It is implemented;
no native retest or sustained capture certification is established by that repair.

Historical [LatencyTrace](../tools/latency_trace.py) and session timing used
separate relative clock origins. Do not retrospectively join them without a
recorded offset; their existing within-clock durations remain usable. New
schema-2 traces expose their explicit common origin.
STT call wall time includes internal runtime waits; it is not a kernel-only timer.
MLX [synchronize](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.synchronize.html)
uses the current default stream when no stream is specified, so a call alone does
not prove every model's GPU work has finished. Measure profiling overhead itself
and keep it out of production timing gates.

## Ranked hypotheses and completed screening

Standard v2 completed the six independent clause/deadline settings plus opening
and closing controls for both models and languages. Its
[selection and rejected tradeoffs](evaluation/mac_followup_20260910/standard-screen-result.md)
apply to the first two hypotheses below. The failed v1 technical cohort and
three-run shutdown-repair pilot remain separate. Any later revision needs a new
declared experiment; all quality-changing behavior stays opt-in.


1. **Commit an eligible clause boundary sooner.** Opt-in early final commitment
   was exercised in Standard v2; none of its tested arms qualified. Ordinary
   smart cuts still wait until the eight-second limit to choose an earlier pause.
   A revised early-cut experiment must retain candidate-boundary timing and
   resumed speech, and preserve preview coverage as well as finals. Require
   full-recording source coverage,
   aligned transcript/translation review, and better visible latency. Producing
   more short captions is not itself a quality or speed win. This applies to both
   language directions and must preserve theological phrases across boundaries.

2. **Admit partial STT against an approaching final deadline.** Latest-only
   queuing cannot interrupt an expensive partial already executing. Measure
   physical partial starts/finishes, request age, audio length, and prediction
   error before deferring a full-prefix partial likely to outlast an imminent
   forced cut. Keep the ordinary 0.6-second cadence outside this bounded guard.
   Reject gains that sacrifice first-preview coverage or update-gap tails.
   The tested Standard deadline arms did not qualify. Independent CPU Lite
   deadline measurements remain pending and must pass their own guards; Standard
   results cannot qualify a Lite arm.

3. **Reduce scalar synchronization in Parakeet TDT decoding.** The exercised
   Parakeet package reads token, confidence and duration scalars separately in
   its decoder loop. If profiling proves those readbacks dominate, compare joint
   evaluation of those outputs or a compiled pure decoder/joint step in an
   isolated dependency environment. Require identical tokens/durations and stable
   confidence near routing thresholds before paired replay testing. This is an
   EN→ES STT optimization; it does not accelerate Spanish Whisper. No confidence
   threshold changes are implied, and the working installation must stay intact.

   **Completed result:** joint scalar evaluation saved 15.13 ms (9.18%) at the
   paired median across 18 exact-output pairs, below the promotion gate. The
   compiled decoder did not improve that paired median and introduced confidence
   differences plus worse first-call tails. Both remain unintegrated; see
   [retained profiling evidence](evaluation/mac_followup_20260910/README.md).

Blind waveform padding is not justified by the current evidence. Parakeet
normalizes across time and uses full-context attention, so padding can alter its
outputs. Its full inference path is not explicitly compiled; Spanish Whisper
already pads decoding windows. Shape compilation and allocator contention remain
hypotheses until measured.

No MTP or Hindi work is included in these follow-ups. The next decision should
use stage-level evidence and independently confirmed output changes, preserving
careful finals and fast, revisable previews.
