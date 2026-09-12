# p1_identity_screen_series4_20260912_r2

Gates cover every declared clip; p95 claim eligibility is separate.

Nearest-rank percentiles; latency in ms. Unknown routes are counted in all-route gates only.

## Clip A

n by route covers all eligible finals; p50/p95 are all-route silence finals.

| Arm | n Gemma / Marian / unknown | Gemma silence n | Silence p50 | Silence p95 | Gemma silence p95 | G1 | G2 | G3 | G4 | G5 | G6 | G7 |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- | --- | --- | --- |
| ctl | 159 / 24 / 0 | 45 | 2146.9 | 3234.2 | 3273.2 | — | — | — | — | — | — | — |
| cand | 159 / 24 / 0 | 45 | 1759.3 | 3365.2 | 3393.1 | FAIL | PASS | PASS | PASS | PASS | PASS | PASS |

### Identity: p1s0912b_A_cand_r0 vs p1s0912b_A_ctl_r0

Translation share: 1.0; English share: 1.0; aligned rows: 61; chunk-count difference: 0.

### Identity: p1s0912b_A_cand_r1 vs p1s0912b_A_ctl_r1

Translation share: 1.0; English share: 1.0; aligned rows: 61; chunk-count difference: 0.

### Identity: p1s0912b_A_cand_r2 vs p1s0912b_A_ctl_r2

Translation share: 1.0; English share: 1.0; aligned rows: 61; chunk-count difference: 0.

## Clip B

n by route covers all eligible finals; p50/p95 are all-route silence finals.

| Arm | n Gemma / Marian / unknown | Gemma silence n | Silence p50 | Silence p95 | Gemma silence p95 | G1 | G2 | G3 | G4 | G5 | G6 | G7 |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- | --- | --- | --- |
| ctl | 162 / 78 / 0 | 129 | 1589.6 | 2658.2 | 2874.9 | — | — | — | — | — | — | — |
| cand | 162 / 78 / 0 | 129 | 1491.0 | 2527.0 | 2696.0 | FAIL | PASS | PASS | PASS | PASS | PASS | PASS |

### Identity: p1s0912b_B_cand_r0 vs p1s0912b_B_ctl_r0

Translation share: 1.0; English share: 1.0; aligned rows: 80; chunk-count difference: 0.

### Identity: p1s0912b_B_cand_r1 vs p1s0912b_B_ctl_r1

Translation share: 1.0; English share: 1.0; aligned rows: 80; chunk-count difference: 0.

### Identity: p1s0912b_B_cand_r2 vs p1s0912b_B_ctl_r2

Translation share: 1.0; English share: 1.0; aligned rows: 80; chunk-count difference: 0.

## Experiment counters

| Run | Source | Counters |
| --- | --- | --- |
| p1s0912b_A_ctl_r0 | diagnostics_jsonl.session_summary | {"final_stt_waited_for_translation": 0, "partial_suppressed_after_stt": 1, "partial_suppressed_backlog": 0, "partial_suppressed_discarded_utterance": 1, "partial_suppressed_empty_translation": 0, "partial_suppressed_final_decode": 0, "partial_suppressed_final_pending": 0, "partial_suppressed_in_flight": 99, "partial_suppressed_published_final": 1, "partial_suppressed_translation_running": 0} |
| p1s0912b_A_cand_r0 | diagnostics_jsonl.session_summary | {"final_stt_waited_for_translation": 0, "partial_suppressed_after_stt": 0, "partial_suppressed_backlog": 0, "partial_suppressed_empty_translation": 0, "partial_suppressed_final_decode": 0, "partial_suppressed_final_pending": 0, "partial_suppressed_in_flight": 28, "partial_suppressed_published_final": 1, "partial_suppressed_translation_running": 0} |
| p1s0912b_A_cand_r1 | diagnostics_jsonl.session_summary | {"final_stt_waited_for_translation": 0, "partial_suppressed_after_stt": 2, "partial_suppressed_backlog": 0, "partial_suppressed_empty_translation": 0, "partial_suppressed_final_decode": 0, "partial_suppressed_final_pending": 0, "partial_suppressed_in_flight": 54, "partial_suppressed_translation_running": 0} |
| p1s0912b_A_ctl_r1 | diagnostics_jsonl.session_summary | {"final_stt_waited_for_translation": 0, "partial_suppressed_after_stt": 0, "partial_suppressed_backlog": 0, "partial_suppressed_empty_translation": 0, "partial_suppressed_final_decode": 0, "partial_suppressed_final_pending": 0, "partial_suppressed_in_flight": 64, "partial_suppressed_translation_running": 0} |
| p1s0912b_A_ctl_r2 | diagnostics_jsonl.session_summary | {"final_stt_waited_for_translation": 0, "partial_suppressed_after_stt": 0, "partial_suppressed_backlog": 0, "partial_suppressed_empty_translation": 0, "partial_suppressed_final_decode": 0, "partial_suppressed_final_pending": 0, "partial_suppressed_in_flight": 45, "partial_suppressed_translation_running": 0} |
| p1s0912b_A_cand_r2 | diagnostics_jsonl.session_summary | {"final_stt_waited_for_translation": 0, "partial_suppressed_after_stt": 0, "partial_suppressed_backlog": 0, "partial_suppressed_empty_translation": 0, "partial_suppressed_final_decode": 0, "partial_suppressed_final_pending": 0, "partial_suppressed_in_flight": 27, "partial_suppressed_translation_running": 0} |
| p1s0912b_B_ctl_r0 | diagnostics_jsonl.session_summary | {"final_stt_waited_for_translation": 0, "partial_suppressed_after_stt": 0, "partial_suppressed_backlog": 0, "partial_suppressed_empty_translation": 0, "partial_suppressed_final_decode": 0, "partial_suppressed_final_pending": 0, "partial_suppressed_in_flight": 24, "partial_suppressed_translation_running": 0} |
| p1s0912b_B_cand_r0 | diagnostics_jsonl.session_summary | {"final_stt_waited_for_translation": 0, "partial_suppressed_after_stt": 0, "partial_suppressed_backlog": 0, "partial_suppressed_empty_translation": 0, "partial_suppressed_final_decode": 0, "partial_suppressed_final_pending": 0, "partial_suppressed_in_flight": 14, "partial_suppressed_translation_running": 0} |
| p1s0912b_B_cand_r1 | diagnostics_jsonl.session_summary | {"final_stt_waited_for_translation": 0, "partial_suppressed_after_stt": 0, "partial_suppressed_backlog": 0, "partial_suppressed_empty_translation": 0, "partial_suppressed_final_decode": 0, "partial_suppressed_final_pending": 1, "partial_suppressed_in_flight": 20, "partial_suppressed_translation_running": 0} |
| p1s0912b_B_ctl_r1 | diagnostics_jsonl.session_summary | {"final_stt_waited_for_translation": 0, "partial_suppressed_after_stt": 2, "partial_suppressed_backlog": 0, "partial_suppressed_empty_translation": 0, "partial_suppressed_final_decode": 0, "partial_suppressed_final_pending": 0, "partial_suppressed_in_flight": 13, "partial_suppressed_translation_running": 0} |
| p1s0912b_B_ctl_r2 | diagnostics_jsonl.session_summary | {"final_stt_waited_for_translation": 0, "partial_suppressed_after_stt": 0, "partial_suppressed_backlog": 0, "partial_suppressed_empty_translation": 0, "partial_suppressed_final_decode": 0, "partial_suppressed_final_pending": 0, "partial_suppressed_in_flight": 16, "partial_suppressed_translation_running": 0} |
| p1s0912b_B_cand_r2 | diagnostics_jsonl.session_summary | {"final_stt_waited_for_translation": 0, "partial_suppressed_after_stt": 0, "partial_suppressed_backlog": 0, "partial_suppressed_empty_translation": 0, "partial_suppressed_final_decode": 0, "partial_suppressed_final_pending": 0, "partial_suppressed_in_flight": 3, "partial_suppressed_translation_running": 0} |

## Outcomes

ctl p95_claim_eligible: false — screen without p95 claim.

cand: REJECTED

Failing gates: A:G1, B:G1.

cand p95_claim_eligible: false — screen without p95 claim.
