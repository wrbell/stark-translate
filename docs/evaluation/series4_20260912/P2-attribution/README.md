# P2 attribution — can the last emitted partial's STT stand in for the final's STT?

**Question.** On silence-ended finals the pipeline runs a second Parakeet call over the whole utterance (≈ 0.36–0.44 s at
p50) although a preview partial was transcribed moments earlier. How often does the last *emitted* partial already
cover the utterance end, was its result available before the VAD finalized, and is its text the same?

**Source.** The six series-3 control replays `lb0912_{A,B}_ctl_r{0,1,2}` (main `a46649e`, promoted runtime, trace on;
360 s clips A = 12/14/25 sermon at 1290 s, B = 2/8/26 sermon), diagnostics rows + embedded latency trace and the
`partials_*.jsonl` preview logs (SHA256 per run in `attribution.json`). Script: [`attribution.py`](attribution.py),
run 2026-09-11T23:50Z from `main` @ `5227a73` with `venv/bin/python`. Coordinates are capture samples at 48 kHz;
`gap_ms = (final.speech_end_sample − last_partial.sample_end) / 48`; a negative gap means the partial window already
extended past the speech end (pause previews include trailing frames). `lead_stt_ms = final_queue_submitted −
physical_stt_finished(partial)`; positive means the partial's STT result existed before the pipeline finalized.

## Result (silence finals only; full table in [`attribution.md`](attribution.md))

| cohort | silence finals | gap ≤ 100 ms | ≤ 300 ms | partial STT done before finalize | lead p50 (min) ms | text equal (normalized) | prefix | differs | final STT p50/p95 ms | reuse@100 / @300 among Gemma-routed |
|---|---|---|---|---|---|---|---|---|---|---|
| clip A pooled (3 runs) | 63 | 75.0 % | 95.0 % | 100 % | 166 (9) | 86.7 % | 3.3 % | 10.0 % | 392 / 565 | 29 / 41 of 45 |
| clip B pooled (3 runs) | 204 | 75.6 % | 99.5 % | 100 % | 346 (1) | 83.1 % | 0.5 % | 16.4 % | 360 / 447 | 101 / 128 of 129 |
| all (6 runs) | 267 | 75.5 % | 98.5 % | 100 % | 318 (1) | 83.9 % | 1.1 % | 14.9 % | 360 / 489 | 130 / 169 of 174 |

- 261 of 267 silence finals had an emitted partial; the last one ends within 100 ms of the speech end in three quarters
  of them and within 300 ms in 98.5 %. Its STT result was in hand before the final was enqueued in every case.
- Exact byte identity with the raw final STT: 75.9 %; after normalization (case, punctuation) 83.9 %; a further 1.1 %
  are prefixes. The 14.9 % that differ are listed at the end of `attribution.md`; they are mostly disfluency and
  boundary-word differences (“um”, a repeated word, a trailing word the partial did not yet see), which is why the arm
  spec carries a text guard (pooled English WER ≤ 3 % vs control, Spanish concatenated WER ≤ 3 %, every differing final
  listed for the bilingual reviewer).
- The replaced final STT costs 360 ms p50 (489 ms p95) pooled; on Gemma-routed silence finals a reuse threshold of
  100 ms would apply to 130 of 174 (75 %), 300 ms to 169 of 174 (97 %).

**Decision.** Design A (reuse the last emitted partial when its window reaches within `partial_reuse_ms` of the speech
end) is admitted as a screened, opt-in arm (`STARK_EXPERIMENT_PARTIAL_REUSE_MS`, off by default). Design B (a
speculative STT call at a fixed pause) is not built: it would add STT calls for less coverage than the partials already
give. This attribution does not certify the arm; the declared screen (`P2-arm-screen/`) does.
