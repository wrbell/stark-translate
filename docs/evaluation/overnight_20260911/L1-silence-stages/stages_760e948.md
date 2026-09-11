# Standard full service, installed 760e948 V6 delivery

Source: `/Users/willem/Code/vibes/SRTranslate/.cache/overnight-20260911/L1/delivery/sessions/standard-full_service_en/metrics/diagnostics_final_final-delivery-v6_standard_full_service_en.jsonl`
SHA256: `a647dfa2c026a056febafb5c2299c761094a75f7ed3e65d2109d6975136f1425`
Generated at UTC: 2026-09-11T02:13:45.398314+00:00

Nearest-rank percentiles. Durations are milliseconds except utterance_dur (seconds); gen_tokens_a is tokens and tps_a is tokens/second. Missing statistics are shown as —.

share_of_total_p50 is an approximation: stage p50 / total p50; medians do not sum. Non-millisecond scalar rows have no share. Broadcast follows final readiness and is outside the total.

Silence rows: CSV 401.0; JSONL 401.0.

## Endpoint: silence

JSONL rows: 401.0

### Gemma

| stage | n | missing | mean | p50 | p95 | share (approx.) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| silence_wait_and_vad_decision | 401.0 | 0.0 | 515.7 | 490.7 | 648.9 | 0.3 |
| submit | 401.0 | 0.0 | 1.7 | 0.2 | 2.7 | 0.0 |
| queue | 401.0 | 0.0 | 61.6 | 2.3 | 196.5 | 0.0 |
| stt_dispatch | 401.0 | 0.0 | 66.9 | 0.2 | 254.9 | 0.0 |
| stt_call | 401.0 | 0.0 | 516.8 | 395.5 | 1284.2 | 0.3 |
| to_translation | 401.0 | 0.0 | 11.0 | 0.2 | 4.0 | 0.0 |
| translation_lock_wait | 401.0 | 0.0 | 56.6 | 0.0 | 406.2 | 0.0 |
| translation_prepare | 401.0 | 0.0 | 10.8 | 0.0 | 78.3 | 0.0 |
| translation_call | 401.0 | 0.0 | 466.4 | 394.4 | 1264.8 | 0.3 |
| finalize | 401.0 | 0.0 | 1.6 | 0.5 | 2.3 | 0.0 |
| broadcast | 401.0 | 0.0 | 0.1 | 0.1 | 0.2 | 0.0 |
| total_speech_end_to_final | 401.0 | 0.0 | 1709.2 | 1458.7 | 3283.2 | 1.0 |
| stt_latency_ms | 401.0 | 0.0 | 516.0 | 395.4 | 1284.1 | 0.3 |
| prefill_ms_a | 205.0 | 196.0 | 129.2 | 120.8 | 177.5 | 0.1 |
| ttft_ms_a | 205.0 | 196.0 | 320.5 | 298.3 | 484.5 | 0.2 |
| decode_ms_a | 205.0 | 196.0 | 494.6 | 449.5 | 981.5 | 0.3 |
| gen_tokens_a | 205.0 | 196.0 | 15.0 | 14.0 | 31.0 | — |
| tps_a | 401.0 | 0.0 | 17.5 | 18.8 | 39.5 | — |
| generation_lock_wait_ms_a | 205.0 | 196.0 | 37.6 | 23.6 | 94.6 | 0.0 |
| finalization_overhead_ms | 401.0 | 0.0 | 1.6 | 0.5 | 2.3 | 0.0 |
| broadcast_ms | 401.0 | 0.0 | 0.1 | 0.1 | 0.2 | 0.0 |
| utterance_dur | 401.0 | 0.0 | 2.8 | 2.2 | 6.8 | — |

Checksum: stamp total p50 1458.7 ms; reported p50 1458.7 ms; checksum_ok: true (tolerance 0.5 ms).

### Marian

| stage | n | missing | mean | p50 | p95 | share (approx.) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| silence_wait_and_vad_decision | 0.0 | 0.0 | — | — | — | — |
| submit | 0.0 | 0.0 | — | — | — | — |
| queue | 0.0 | 0.0 | — | — | — | — |
| stt_dispatch | 0.0 | 0.0 | — | — | — | — |
| stt_call | 0.0 | 0.0 | — | — | — | — |
| to_translation | 0.0 | 0.0 | — | — | — | — |
| translation_lock_wait | 0.0 | 0.0 | — | — | — | — |
| translation_prepare | 0.0 | 0.0 | — | — | — | — |
| translation_call | 0.0 | 0.0 | — | — | — | — |
| finalize | 0.0 | 0.0 | — | — | — | — |
| broadcast | 0.0 | 0.0 | — | — | — | — |
| total_speech_end_to_final | 0.0 | 0.0 | — | — | — | — |
| stt_latency_ms | 0.0 | 0.0 | — | — | — | — |
| prefill_ms_a | 0.0 | 0.0 | — | — | — | — |
| ttft_ms_a | 0.0 | 0.0 | — | — | — | — |
| decode_ms_a | 0.0 | 0.0 | — | — | — | — |
| gen_tokens_a | 0.0 | 0.0 | — | — | — | — |
| tps_a | 0.0 | 0.0 | — | — | — | — |
| generation_lock_wait_ms_a | 0.0 | 0.0 | — | — | — | — |
| finalization_overhead_ms | 0.0 | 0.0 | — | — | — | — |
| broadcast_ms | 0.0 | 0.0 | — | — | — | — |
| utterance_dur | 0.0 | 0.0 | — | — | — | — |

Checksum: stamp total p50 — ms; reported p50 — ms; checksum_ok: null (tolerance 0.5 ms).

### Ranked Gemma stages (p50 descending)

Total is included as a reference; scalar rows are excluded from this ranking.

| stage | p50 (ms) |
| --- | ---: |
| total_speech_end_to_final | 1458.7 |
| silence_wait_and_vad_decision | 490.7 |
| stt_call | 395.5 |
| translation_call | 394.4 |
| queue | 2.3 |
| finalize | 0.5 |
| to_translation | 0.2 |
| submit | 0.2 |
| stt_dispatch | 0.2 |
| broadcast | 0.1 |
| translation_prepare | 0.0 |
| translation_lock_wait | 0.0 |

## Endpoint: hard_cut

JSONL rows: 38.0

### Gemma

| stage | n | missing | mean | p50 | p95 | share (approx.) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| silence_wait_and_vad_decision | 38.0 | 0.0 | 9.2 | 11.0 | 12.4 | 0.0 |
| submit | 38.0 | 0.0 | 0.3 | 0.1 | 1.3 | 0.0 |
| queue | 38.0 | 0.0 | 22.9 | 5.4 | 23.8 | 0.0 |
| stt_dispatch | 38.0 | 0.0 | 0.2 | 0.2 | 0.6 | 0.0 |
| stt_call | 38.0 | 0.0 | 655.5 | 585.9 | 1172.4 | 0.3 |
| to_translation | 38.0 | 0.0 | 1.8 | 0.3 | 3.7 | 0.0 |
| translation_lock_wait | 38.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| translation_prepare | 38.0 | 0.0 | 22.7 | 0.0 | 104.3 | 0.0 |
| translation_call | 38.0 | 0.0 | 1536.5 | 1511.1 | 2414.6 | 0.7 |
| finalize | 38.0 | 0.0 | 2.4 | 2.3 | 4.2 | 0.0 |
| broadcast | 38.0 | 0.0 | 0.1 | 0.1 | 0.1 | 0.0 |
| total_speech_end_to_final | 38.0 | 0.0 | 2251.6 | 2244.6 | 2940.4 | 1.0 |
| stt_latency_ms | 38.0 | 0.0 | 655.4 | 585.8 | 1172.3 | 0.3 |
| prefill_ms_a | 37.0 | 1.0 | 154.4 | 158.9 | 189.7 | 0.1 |
| ttft_ms_a | 37.0 | 1.0 | 391.4 | 324.6 | 1057.0 | 0.1 |
| decode_ms_a | 37.0 | 1.0 | 1171.5 | 1176.0 | 1501.1 | 0.5 |
| gen_tokens_a | 37.0 | 1.0 | 35.7 | 35.0 | 43.0 | — |
| tps_a | 38.0 | 0.0 | 30.4 | 31.3 | 33.3 | — |
| generation_lock_wait_ms_a | 37.0 | 1.0 | 5.9 | 0.0 | 49.8 | 0.0 |
| finalization_overhead_ms | 38.0 | 0.0 | 2.4 | 2.3 | 4.2 | 0.0 |
| broadcast_ms | 38.0 | 0.0 | 0.1 | 0.1 | 0.1 | 0.0 |
| utterance_dur | 38.0 | 0.0 | 8.0 | 8.0 | 8.0 | — |

Checksum: stamp total p50 2244.6 ms; reported p50 2244.6 ms; checksum_ok: true (tolerance 0.5 ms).

### Marian

| stage | n | missing | mean | p50 | p95 | share (approx.) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| silence_wait_and_vad_decision | 0.0 | 0.0 | — | — | — | — |
| submit | 0.0 | 0.0 | — | — | — | — |
| queue | 0.0 | 0.0 | — | — | — | — |
| stt_dispatch | 0.0 | 0.0 | — | — | — | — |
| stt_call | 0.0 | 0.0 | — | — | — | — |
| to_translation | 0.0 | 0.0 | — | — | — | — |
| translation_lock_wait | 0.0 | 0.0 | — | — | — | — |
| translation_prepare | 0.0 | 0.0 | — | — | — | — |
| translation_call | 0.0 | 0.0 | — | — | — | — |
| finalize | 0.0 | 0.0 | — | — | — | — |
| broadcast | 0.0 | 0.0 | — | — | — | — |
| total_speech_end_to_final | 0.0 | 0.0 | — | — | — | — |
| stt_latency_ms | 0.0 | 0.0 | — | — | — | — |
| prefill_ms_a | 0.0 | 0.0 | — | — | — | — |
| ttft_ms_a | 0.0 | 0.0 | — | — | — | — |
| decode_ms_a | 0.0 | 0.0 | — | — | — | — |
| gen_tokens_a | 0.0 | 0.0 | — | — | — | — |
| tps_a | 0.0 | 0.0 | — | — | — | — |
| generation_lock_wait_ms_a | 0.0 | 0.0 | — | — | — | — |
| finalization_overhead_ms | 0.0 | 0.0 | — | — | — | — |
| broadcast_ms | 0.0 | 0.0 | — | — | — | — |
| utterance_dur | 0.0 | 0.0 | — | — | — | — |

Checksum: stamp total p50 — ms; reported p50 — ms; checksum_ok: null (tolerance 0.5 ms).

### Ranked Gemma stages (p50 descending)

Total is included as a reference; scalar rows are excluded from this ranking.

| stage | p50 (ms) |
| --- | ---: |
| total_speech_end_to_final | 2244.6 |
| translation_call | 1511.1 |
| stt_call | 585.9 |
| silence_wait_and_vad_decision | 11.0 |
| queue | 5.4 |
| finalize | 2.3 |
| to_translation | 0.3 |
| stt_dispatch | 0.2 |
| submit | 0.1 |
| broadcast | 0.1 |
| translation_prepare | 0.0 |
| translation_lock_wait | 0.0 |

## Endpoint: smart_cut

JSONL rows: 124.0

### Gemma

| stage | n | missing | mean | p50 | p95 | share (approx.) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| silence_wait_and_vad_decision | 124.0 | 0.0 | 1915.3 | 1232.1 | 5037.0 | 0.4 |
| submit | 124.0 | 0.0 | 1.4 | 0.1 | 4.9 | 0.0 |
| queue | 124.0 | 0.0 | 4.3 | 0.1 | 2.7 | 0.0 |
| stt_dispatch | 124.0 | 0.0 | 5.5 | 0.2 | 5.6 | 0.0 |
| stt_call | 124.0 | 0.0 | 290.7 | 216.9 | 735.9 | 0.1 |
| to_translation | 124.0 | 0.0 | 9.8 | 0.4 | 3.5 | 0.0 |
| translation_lock_wait | 124.0 | 0.0 | 0.1 | 0.0 | 0.0 | 0.0 |
| translation_prepare | 124.0 | 0.0 | 24.0 | 0.0 | 98.4 | 0.0 |
| translation_call | 124.0 | 0.0 | 1665.9 | 1476.7 | 2878.6 | 0.4 |
| finalize | 124.0 | 0.0 | 4.9 | 1.7 | 14.4 | 0.0 |
| broadcast | 124.0 | 0.0 | 0.5 | 0.1 | 0.6 | 0.0 |
| total_speech_end_to_final | 124.0 | 0.0 | 3921.8 | 3351.9 | 7220.7 | 1.0 |
| stt_latency_ms | 124.0 | 0.0 | 290.4 | 216.7 | 735.8 | 0.1 |
| prefill_ms_a | 118.0 | 6.0 | 230.5 | 143.9 | 242.5 | 0.0 |
| ttft_ms_a | 118.0 | 6.0 | 836.3 | 556.1 | 1388.9 | 0.2 |
| decode_ms_a | 118.0 | 6.0 | 866.0 | 825.4 | 1449.3 | 0.2 |
| gen_tokens_a | 118.0 | 6.0 | 25.1 | 25.0 | 41.0 | — |
| tps_a | 124.0 | 0.0 | 29.1 | 30.8 | 35.7 | — |
| generation_lock_wait_ms_a | 118.0 | 6.0 | 1.4 | 0.0 | 0.1 | 0.0 |
| finalization_overhead_ms | 124.0 | 0.0 | 4.9 | 1.7 | 14.4 | 0.0 |
| broadcast_ms | 124.0 | 0.0 | 0.6 | 0.1 | 0.6 | 0.0 |
| utterance_dur | 124.0 | 0.0 | 6.1 | 6.7 | 7.8 | — |

Checksum: stamp total p50 3351.9 ms; reported p50 3351.9 ms; checksum_ok: true (tolerance 0.5 ms).

### Marian

| stage | n | missing | mean | p50 | p95 | share (approx.) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| silence_wait_and_vad_decision | 0.0 | 0.0 | — | — | — | — |
| submit | 0.0 | 0.0 | — | — | — | — |
| queue | 0.0 | 0.0 | — | — | — | — |
| stt_dispatch | 0.0 | 0.0 | — | — | — | — |
| stt_call | 0.0 | 0.0 | — | — | — | — |
| to_translation | 0.0 | 0.0 | — | — | — | — |
| translation_lock_wait | 0.0 | 0.0 | — | — | — | — |
| translation_prepare | 0.0 | 0.0 | — | — | — | — |
| translation_call | 0.0 | 0.0 | — | — | — | — |
| finalize | 0.0 | 0.0 | — | — | — | — |
| broadcast | 0.0 | 0.0 | — | — | — | — |
| total_speech_end_to_final | 0.0 | 0.0 | — | — | — | — |
| stt_latency_ms | 0.0 | 0.0 | — | — | — | — |
| prefill_ms_a | 0.0 | 0.0 | — | — | — | — |
| ttft_ms_a | 0.0 | 0.0 | — | — | — | — |
| decode_ms_a | 0.0 | 0.0 | — | — | — | — |
| gen_tokens_a | 0.0 | 0.0 | — | — | — | — |
| tps_a | 0.0 | 0.0 | — | — | — | — |
| generation_lock_wait_ms_a | 0.0 | 0.0 | — | — | — | — |
| finalization_overhead_ms | 0.0 | 0.0 | — | — | — | — |
| broadcast_ms | 0.0 | 0.0 | — | — | — | — |
| utterance_dur | 0.0 | 0.0 | — | — | — | — |

Checksum: stamp total p50 — ms; reported p50 — ms; checksum_ok: null (tolerance 0.5 ms).

### Ranked Gemma stages (p50 descending)

Total is included as a reference; scalar rows are excluded from this ranking.

| stage | p50 (ms) |
| --- | ---: |
| total_speech_end_to_final | 3351.9 |
| translation_call | 1476.7 |
| silence_wait_and_vad_decision | 1232.1 |
| stt_call | 216.9 |
| finalize | 1.7 |
| to_translation | 0.4 |
| stt_dispatch | 0.2 |
| submit | 0.1 |
| queue | 0.1 |
| broadcast | 0.1 |
| translation_prepare | 0.0 |
| translation_lock_wait | 0.0 |
