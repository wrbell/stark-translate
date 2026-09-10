# EN↔ES latency: experiments after the overnight screen

The [completed screen](evaluation/overnight_screen_20260910/README.md) tested
15 configurations on E4B and E2B: 96/96 valid runs and 0/28 selected
experiment/model arms. Ordinary
confirmations or combinations of those arms are therefore not justified. The
hypotheses below now have implemented opt-in experiments in the
[EN↔ES follow-up](evaluation/mac_followup_20260910/README.md). Their new paired
pipeline results are still pending; they do not recommend changing defaults.
Public Spanish read-speech references are available for engineering comparisons;
church references and bilingual review remain required for quality certification. The [completed Standard and Lite endurance audits](evaluation/overnight_endurance_20260910/README.md)
confirm consistent retained spans and durable completion on `752ab9a`. Lite's
sparse translated previews and large observed tails do not support a fast-production
recommendation. These functional runs do not promote a screen arm, establish a
causal speed improvement or satisfy quality/hardware certification.

## What the stage records show

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
does not address the CPU STT queue. The frozen follow-up independently screens
0.6/0.9/1.2-second cadence, then conditions deadline admission on that result.
The separate small/base CT2 quality comparison uses 50 public development
recordings per language in three repeats. Both CPU screens are queued behind
the Standard cohort; no gain is established yet.

## First improve attribution

The follow-up now records a common monotonic clock for capture, VAD, worker
admission, physical STT calls and final delivery. Source coverage and actual
EOF accounting accompany the trace; native inference drains before terminal
summaries. The separate sampled Parakeet profile records exact model inputs,
encoder/decoder work and scalar readback waits with an alternating unprofiled
control. Its observed overhead is retained in the linked report.

Historical [LatencyTrace](../tools/latency_trace.py) and session timing used
separate relative clock origins. Do not retrospectively join them without a
recorded offset; their existing within-clock durations remain usable. New
schema-2 traces expose their explicit common origin.
STT call wall time includes internal runtime waits; it is not a kernel-only timer.
MLX [synchronize](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.synchronize.html)
uses the current default stream when no stream is specified, so a call alone does
not prove every model's GPU work has finished. Measure profiling overhead itself
and keep it out of production timing gates.

## Ranked hypotheses and current execution

The Standard v2 cohort runs six independent clause/deadline arms plus opening
and closing controls for both models and languages, three repeats (96 runs).
It follows a failed v1 technical cohort and a three-run shutdown-repair pilot;
those cohorts remain separate. All quality-changing behavior stays opt-in.


1. **Commit an eligible clause boundary sooner.** Current clause mode creates
   revisable previews; ordinary smart cuts wait until the eight-second limit to
   choose an earlier pause. First record when each candidate boundary becomes
   eligible, resumed speech, and the eventual boundary. Then test early final
   commitment behind a flag. This has the largest directly observed opportunity
   and the greatest semantic risk. Require full-recording source coverage,
   aligned transcript/translation review, and better visible latency. Producing
   more short captions is not itself a quality or speed win. This applies to both
   language directions and must preserve theological phrases across boundaries.

2. **Admit partial STT against an approaching final deadline.** Latest-only
   queuing cannot interrupt an expensive partial already executing. Measure
   physical partial starts/finishes, request age, audio length, and prediction
   error before deferring a full-prefix partial likely to outlast an imminent
   forced cut. Keep the ordinary 0.6-second cadence outside this bounded guard.
   Reject gains that sacrifice first-preview coverage or update-gap tails.
   Test each STT backend separately; this is more specific than suppressing all
   partials whenever final translation is active.

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
