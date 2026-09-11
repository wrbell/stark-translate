# Standard full service, installed 752ab9a, session 20260910_043120_839144_en

Source: `/Users/willem/Code/vibes/SRTranslate/.cache/overnight-20260910/installed-standard-fixed-endurance/metrics/diagnostics_20260910_043120_839144_en.jsonl`
SHA256: `f9927c40ec742ad483b06b7d0ed9238ef0c665aa4b084604ebd8679138a957b8`
Generated at UTC: 2026-09-11T02:13:44.757882+00:00

Nearest-rank percentiles. Durations are milliseconds except utterance_dur (seconds); gen_tokens_a is tokens and tps_a is tokens/second. Missing statistics are shown as —.

share_of_total_p50 is an approximation: stage p50 / total p50; medians do not sum. Non-millisecond scalar rows have no share. Broadcast follows final readiness and is outside the total.

Silence rows: CSV 401.0; JSONL 401.0.

## Endpoint: silence

JSONL rows: 401.0

### Gemma

| stage | n | missing | mean | p50 | p95 | share (approx.) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| silence_wait_and_vad_decision | 401.0 | 0.0 | 507.7 | 491.1 | 575.0 | 0.3 |
| submit | 401.0 | 0.0 | 0.4 | 0.1 | 0.5 | 0.0 |
| queue | 401.0 | 0.0 | 10.5 | 1.9 | 11.8 | 0.0 |
| stt_dispatch | 401.0 | 0.0 | 6.7 | 0.1 | 22.6 | 0.0 |
| stt_call | 401.0 | 0.0 | 505.2 | 414.3 | 1218.6 | 0.3 |
| to_translation | 401.0 | 0.0 | 23.0 | 0.4 | 4.2 | 0.0 |
| translation_lock_wait | 401.0 | 0.0 | 45.2 | 0.0 | 462.8 | 0.0 |
| translation_prepare | 401.0 | 0.0 | 7.2 | 0.0 | 37.3 | 0.0 |
| translation_call | 401.0 | 0.0 | 466.6 | 381.3 | 1289.1 | 0.3 |
| finalize | 401.0 | 0.0 | 2.3 | 0.5 | 1.8 | 0.0 |
| broadcast | 401.0 | 0.0 | 3.3 | 0.8 | 4.3 | 0.0 |
| total_speech_end_to_final | 401.0 | 0.0 | 1574.8 | 1429.7 | 3019.0 | 1.0 |
| stt_latency_ms | 401.0 | 0.0 | 501.9 | 414.2 | 1196.9 | 0.3 |
| prefill_ms_a | 205.0 | 196.0 | 124.2 | 115.1 | 169.3 | 0.1 |
| ttft_ms_a | 205.0 | 196.0 | 319.9 | 299.2 | 408.6 | 0.2 |
| decode_ms_a | 205.0 | 196.0 | 504.1 | 456.4 | 1039.8 | 0.3 |
| gen_tokens_a | 205.0 | 196.0 | 15.0 | 14.0 | 31.0 | — |
| tps_a | 401.0 | 0.0 | 17.1 | 26.3 | 38.6 | — |
| generation_lock_wait_ms_a | 0.0 | 401.0 | — | — | — | — |
| finalization_overhead_ms | 401.0 | 0.0 | 2.3 | 0.5 | 1.8 | 0.0 |
| broadcast_ms | 401.0 | 0.0 | 3.3 | 0.8 | 4.3 | 0.0 |
| utterance_dur | 401.0 | 0.0 | 2.8 | 2.2 | 6.8 | — |

Checksum: stamp total p50 1429.7 ms; reported p50 1429.7 ms; checksum_ok: true (tolerance 0.5 ms).

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
| total_speech_end_to_final | 1429.7 |
| silence_wait_and_vad_decision | 491.1 |
| stt_call | 414.3 |
| translation_call | 381.3 |
| queue | 1.9 |
| broadcast | 0.8 |
| finalize | 0.5 |
| to_translation | 0.4 |
| stt_dispatch | 0.1 |
| submit | 0.1 |
| translation_prepare | 0.0 |
| translation_lock_wait | 0.0 |

## Endpoint: hard_cut

JSONL rows: 38.0

### Gemma

| stage | n | missing | mean | p50 | p95 | share (approx.) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| silence_wait_and_vad_decision | 38.0 | 0.0 | 8.4 | 8.5 | 12.5 | 0.0 |
| submit | 38.0 | 0.0 | 0.1 | 0.1 | 0.2 | 0.0 |
| queue | 38.0 | 0.0 | 37.0 | 6.2 | 130.6 | 0.0 |
| stt_dispatch | 38.0 | 0.0 | 0.2 | 0.2 | 0.5 | 0.0 |
| stt_call | 38.0 | 0.0 | 727.7 | 616.9 | 1406.4 | 0.3 |
| to_translation | 38.0 | 0.0 | 3.2 | 0.7 | 13.1 | 0.0 |
| translation_lock_wait | 38.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| translation_prepare | 38.0 | 0.0 | 32.5 | 0.1 | 129.9 | 0.0 |
| translation_call | 38.0 | 0.0 | 1578.7 | 1649.4 | 2074.7 | 0.7 |
| finalize | 38.0 | 0.0 | 1.9 | 1.7 | 3.8 | 0.0 |
| broadcast | 38.0 | 0.0 | 1.2 | 1.0 | 2.0 | 0.0 |
| total_speech_end_to_final | 38.0 | 0.0 | 2389.7 | 2325.7 | 3220.2 | 1.0 |
| stt_latency_ms | 38.0 | 0.0 | 727.6 | 616.9 | 1406.3 | 0.3 |
| prefill_ms_a | 37.0 | 1.0 | 150.7 | 154.2 | 196.5 | 0.1 |
| ttft_ms_a | 37.0 | 1.0 | 398.4 | 328.8 | 974.6 | 0.1 |
| decode_ms_a | 37.0 | 1.0 | 1209.0 | 1264.4 | 1442.9 | 0.5 |
| gen_tokens_a | 37.0 | 1.0 | 35.7 | 35.0 | 43.0 | — |
| tps_a | 38.0 | 0.0 | 29.4 | 30.2 | 32.1 | — |
| generation_lock_wait_ms_a | 0.0 | 38.0 | — | — | — | — |
| finalization_overhead_ms | 38.0 | 0.0 | 1.9 | 1.7 | 3.8 | 0.0 |
| broadcast_ms | 38.0 | 0.0 | 1.2 | 1.0 | 2.0 | 0.0 |
| utterance_dur | 38.0 | 0.0 | 8.0 | 8.0 | 8.0 | — |

Checksum: stamp total p50 2325.7 ms; reported p50 2325.7 ms; checksum_ok: true (tolerance 0.5 ms).

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
| total_speech_end_to_final | 2325.7 |
| translation_call | 1649.4 |
| stt_call | 616.9 |
| silence_wait_and_vad_decision | 8.5 |
| queue | 6.2 |
| finalize | 1.7 |
| broadcast | 1.0 |
| to_translation | 0.7 |
| stt_dispatch | 0.2 |
| submit | 0.1 |
| translation_prepare | 0.1 |
| translation_lock_wait | 0.0 |

## Endpoint: smart_cut

JSONL rows: 124.0

### Gemma

| stage | n | missing | mean | p50 | p95 | share (approx.) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| silence_wait_and_vad_decision | 124.0 | 0.0 | 1900.8 | 1229.0 | 5031.2 | 0.4 |
| submit | 124.0 | 0.0 | 1.4 | 0.1 | 1.3 | 0.0 |
| queue | 124.0 | 0.0 | 0.5 | 0.1 | 0.8 | 0.0 |
| stt_dispatch | 124.0 | 0.0 | 0.2 | 0.1 | 0.4 | 0.0 |
| stt_call | 124.0 | 0.0 | 252.9 | 217.4 | 453.0 | 0.1 |
| to_translation | 124.0 | 0.0 | 1.3 | 0.6 | 3.5 | 0.0 |
| translation_lock_wait | 124.0 | 0.0 | 0.0 | 0.0 | 0.1 | 0.0 |
| translation_prepare | 124.0 | 0.0 | 11.4 | 0.1 | 94.9 | 0.0 |
| translation_call | 124.0 | 0.0 | 1600.6 | 1577.1 | 2739.6 | 0.5 |
| finalize | 124.0 | 0.0 | 3.7 | 1.4 | 7.3 | 0.0 |
| broadcast | 124.0 | 0.0 | 7.0 | 1.3 | 35.1 | 0.0 |
| total_speech_end_to_final | 124.0 | 0.0 | 3772.8 | 3449.3 | 6451.3 | 1.0 |
| stt_latency_ms | 124.0 | 0.0 | 252.8 | 217.2 | 452.9 | 0.1 |
| prefill_ms_a | 118.0 | 6.0 | 164.4 | 143.8 | 243.1 | 0.0 |
| ttft_ms_a | 118.0 | 6.0 | 771.0 | 551.3 | 1395.7 | 0.2 |
| decode_ms_a | 118.0 | 6.0 | 867.7 | 856.9 | 1401.7 | 0.2 |
| gen_tokens_a | 118.0 | 6.0 | 25.1 | 25.0 | 41.0 | — |
| tps_a | 124.0 | 0.0 | 28.5 | 29.7 | 34.4 | — |
| generation_lock_wait_ms_a | 0.0 | 124.0 | — | — | — | — |
| finalization_overhead_ms | 124.0 | 0.0 | 3.7 | 1.4 | 7.3 | 0.0 |
| broadcast_ms | 124.0 | 0.0 | 7.0 | 1.3 | 35.1 | 0.0 |
| utterance_dur | 124.0 | 0.0 | 6.1 | 6.7 | 7.8 | — |

Checksum: stamp total p50 3449.3 ms; reported p50 3449.3 ms; checksum_ok: true (tolerance 0.5 ms).

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
| total_speech_end_to_final | 3449.3 |
| translation_call | 1577.1 |
| silence_wait_and_vad_decision | 1229.0 |
| stt_call | 217.4 |
| finalize | 1.4 |
| broadcast | 1.3 |
| to_translation | 0.6 |
| stt_dispatch | 0.1 |
| submit | 0.1 |
| queue | 0.1 |
| translation_prepare | 0.1 |
| translation_lock_wait | 0.0 |
