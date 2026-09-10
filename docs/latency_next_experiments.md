# EN↔ES latency: experiments after the overnight screen

The [completed screen](evaluation/overnight_screen_20260910/README.md) tested
15 configurations on E4B and E2B and selected no experimental arms. Ordinary
confirmations or combinations of those arms are therefore not justified. The
proposals below are subsequent research, not implemented optimizations or a
recommendation to change defaults. Natural Spanish and bilingual review remain
required for quality certification.

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

## First improve attribution

Add opt-in sampled profiling in an isolated evaluation checkout. Record task,
thread and stream identity; exact audio bounds and mel shape; preprocessing,
encoder, decoder and synchronization durations; and every warmup interval.

[LatencyTrace](../tools/latency_trace.py) and session timing currently use separate
relative clock origins. Record a common origin or their explicit offset before
joining those timelines. Their existing within-clock durations remain usable.
STT call wall time includes internal runtime waits; it is not a kernel-only timer.
MLX [synchronize](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.synchronize.html)
uses the current default stream when no stream is specified, so a call alone does
not prove every model's GPU work has finished. Measure profiling overhead itself
and keep it out of production timing gates.

## Ranked follow-ups

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

Blind waveform padding is not justified by the current evidence. Parakeet
normalizes across time and uses full-context attention, so padding can alter its
outputs. Its full inference path is not explicitly compiled; Spanish Whisper
already pads decoding windows. Shape compilation and allocator contention remain
hypotheses until measured.

No MTP or Hindi work is included in these follow-ups. The next decision should
use stage-level evidence and independently confirmed output changes, preserving
careful finals and fast, revisable previews.
