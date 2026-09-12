# p1_identity_screen_series4_20260912

Gates cover every declared clip; p95 claim eligibility is separate.

Nearest-rank percentiles; latency in ms. Unknown routes are counted in all-route gates only.

## Clip A

n by route covers all eligible finals; p50/p95 are all-route silence finals.

| Arm | n Gemma / Marian / unknown | Gemma silence n | Silence p50 | Silence p95 | Gemma silence p95 | G1 | G2 | G3 | G4 | G5 | G6 | G7 |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- | --- | --- | --- |
| ctl | 159 / 24 / 0 | 45 | 1661.6 | 2377.8 | 2463.7 | — | — | — | — | — | — | — |
| cand | 159 / 24 / 0 | 45 | 1618.5 | 2271.6 | 2284.7 | FAIL | PASS | PASS | PASS | PASS | FAIL | PASS |

### Identity: p1s0912_A_cand_r0 vs p1s0912_A_ctl_r0

Translation share: 1.0; English share: 1.0; aligned rows: 61; chunk-count difference: 0.

### Identity: p1s0912_A_cand_r1 vs p1s0912_A_ctl_r1

Translation share: 1.0; English share: 1.0; aligned rows: 61; chunk-count difference: 0.

### Identity: p1s0912_A_cand_r2 vs p1s0912_A_ctl_r2

Translation share: 1.0; English share: 1.0; aligned rows: 61; chunk-count difference: 0.

## Clip B

n by route covers all eligible finals; p50/p95 are all-route silence finals.

| Arm | n Gemma / Marian / unknown | Gemma silence n | Silence p50 | Silence p95 | Gemma silence p95 | G1 | G2 | G3 | G4 | G5 | G6 | G7 |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- | --- | --- | --- |
| ctl | 0 / 0 / 0 | 0 | — | — | — | — | — | — | — | — | — | — |
| cand | 0 / 0 / 0 | 0 | — | — | — | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL |

## Experiment counters

| Run | Source | Counters |
| --- | --- | --- |
| p1s0912_A_ctl_r0 | diagnostics_jsonl.session_summary | {"final_stt_waited_for_translation": 0, "partial_suppressed_after_stt": 0, "partial_suppressed_backlog": 0, "partial_suppressed_empty_translation": 0, "partial_suppressed_final_decode": 0, "partial_suppressed_final_pending": 0, "partial_suppressed_in_flight": 13, "partial_suppressed_translation_running": 0} |
| p1s0912_A_cand_r0 | diagnostics_jsonl.session_summary | {"final_stt_waited_for_translation": 0, "partial_suppressed_after_stt": 0, "partial_suppressed_backlog": 0, "partial_suppressed_empty_translation": 0, "partial_suppressed_final_decode": 0, "partial_suppressed_final_pending": 0, "partial_suppressed_in_flight": 0, "partial_suppressed_translation_running": 0} |
| p1s0912_A_cand_r1 | diagnostics_jsonl.session_summary | {"final_stt_waited_for_translation": 0, "partial_suppressed_after_stt": 0, "partial_suppressed_backlog": 0, "partial_suppressed_empty_translation": 0, "partial_suppressed_final_decode": 0, "partial_suppressed_final_pending": 0, "partial_suppressed_in_flight": 0, "partial_suppressed_translation_running": 0} |
| p1s0912_A_ctl_r1 | diagnostics_jsonl.session_summary | {"final_stt_waited_for_translation": 0, "partial_suppressed_after_stt": 0, "partial_suppressed_backlog": 0, "partial_suppressed_empty_translation": 0, "partial_suppressed_final_decode": 0, "partial_suppressed_final_pending": 0, "partial_suppressed_in_flight": 16, "partial_suppressed_translation_running": 0} |
| p1s0912_A_ctl_r2 | diagnostics_jsonl.session_summary | {"final_stt_waited_for_translation": 0, "partial_suppressed_after_stt": 0, "partial_suppressed_backlog": 0, "partial_suppressed_empty_translation": 0, "partial_suppressed_final_decode": 0, "partial_suppressed_final_pending": 0, "partial_suppressed_in_flight": 2, "partial_suppressed_translation_running": 0} |
| p1s0912_A_cand_r2 | diagnostics_jsonl.session_summary | {"final_stt_waited_for_translation": 0, "partial_suppressed_after_stt": 0, "partial_suppressed_backlog": 0, "partial_suppressed_empty_translation": 0, "partial_suppressed_final_decode": 0, "partial_suppressed_final_pending": 0, "partial_suppressed_in_flight": 0, "partial_suppressed_translation_running": 0} |

## Outcomes

ctl p95_claim_eligible: false — screen without p95 claim.

cand: REJECTED

Failing gates: A:G1, A:G6, B:G1, B:G2, B:G3, B:G4, B:G5, B:G6, B:G7.

cand p95_claim_eligible: false — screen without p95 claim.
