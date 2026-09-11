# L-C — what the audience sees first vs what the goal measures (2026-09-11)

**Number for Willem.** On the promoted runtime, the first translated tokens of a Gemma-routed final are produced
about **1.1–1.2 s after speech end at the median** (p95 1.3 s), while the schema 2 number the goal measures
(`speech_end_to_final_ms`, payload ready) is 1.6–2.0 s for those finals. Counting Marian-routed finals at their
final-ready time (they are not streamed), the "first visible caption text after speech end" is **p50 1.09–1.19 s,
p95 1.23–1.53 s** across all silence finals. Sub-second is still not met by either measure, but the gap to the
goal is about 0.2 s on the first-token measure versus 0.6–1.0 s on the payload-ready measure. Whether the goal
should be restated in first-token terms is a product decision, not a measurement one; nothing here changes the
declared goal.

Source: the six traced control runs of `tail_screen_20260911` (`STARK_EXPERIMENT_TRACE=true`, promoted `venv`,
`main` `6f160bb`), inputs listed in [`first_token.json`](first_token.json).

| clip | Gemma-routed silence finals | first token p50 / p95 (ms) | payload ready p50 / p95 (ms) | Marian-routed finals | Marian final ready p50 / p95 | all-route first visible p50 / p95 |
|---|---:|---|---|---:|---|---|
| A (12_14_25 @ 1290 s, 360 s) | 45 | 1206.2 / 1346.1 | 1970.5 / 2424.6 | 18 | 804.0 / 1666.9 | 1193.8 / 1532.9 |
| B (2_8_26 @ 1170 s, 360 s) | 129 | 1127.0 / 1264.9 | 1618.7 / 1914.5 | 75 | 868.8 / 984.0 | 1089.9 / 1228.9 |

## Method and its bias

`first_token_ms = timing_stages_ms.translation_started + ttft_ms_a − timing_stages_ms.speech_end`, i.e. the
moment the model produced token 1 (nearest-rank percentiles, pooled over the three control repeats per clip).
This slightly **under**estimates the wire: `ttft_ms_a` starts after the prompt build and the generation lock
(`generation_lock_wait_ms_a` p95 ≈ 80 ms here), the first `translation_stream` message carries three tokens
(`STREAM_TOKEN_BATCH_SIZE = 3`), and headless replays have no client, so no stream message is actually sent. The
P2-H harness PR adds a `first_stream_token` trace record so an attended run with a connected display can measure
the wire time exactly. Browser render and visible-ACK latency are outside this number (see
`docs/evaluation/README.md`).

This is an engineering derivation on machine-timed replays; it certifies no human-visible delivery, no quality,
and changes no goal or default.
