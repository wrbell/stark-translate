# L1 — where the time goes in a silence-ended final caption

**Lane state:** DONE (analysis only; no inference, no default change).
**Tool:** `tools/silence_final_stages.py` (PR #199, merged 2026-09-11T02:17Z), run from source `9983f68` + that PR.
**Method:** schema 2 `timing_stages_ms` stamps per final, nearest-rank p50/p95, Gemma and Marian-routed finals kept separate, checksum against each record's own `speech_end_to_final_ms` (both cohorts: `checksum_ok = true`, 401/401 silence finals, CSV row count matches).

## Inputs

| Cohort | Source | Diagnostics | Silence finals |
|---|---|---|---|
| Repaired Standard full service, installed `752ab9a`, session `20260910_043120_839144_en` | `.cache/overnight-20260910/installed-standard-fixed-endurance/metrics/diagnostics_20260910_043120_839144_en.jsonl` (SHA256 `f9927c40…`) | 563 finals | 401 |
| V6 Standard full service, installed `760e948` | reconstructed from `docs/evaluation/mac_followup_20260910/final-760e948/raw/` (archive SHA256 `a9efcd8f…`), `delivery/sessions/standard-full_service_en/metrics/diagnostics_final_final-delivery-v6_standard_full_service_en.jsonl` | 563 finals | 401 |

Both are the same 3,640 s natural English service replayed in real time on the same Mac; they are two separate observational cohorts, not a paired comparison. All 401 silence finals in each cohort were Gemma E4B finals (the Marian-routed cohort is empty). 196 of 401 lack the Gemma `prefill/ttft/decode` scalars (the streaming final path does not populate them); the stamp-based `translation_call` stage covers all 401.

## Stage medians and p95 (ms, silence endpoint, Gemma cohort)

| stage | 752ab9a p50 | 752ab9a p95 | 760e948 p50 | 760e948 p95 |
|---|---:|---:|---:|---:|
| `silence_wait_and_vad_decision` (speech end → VAD final decision) | 491.1 | 575.0 | 490.7 | 648.9 |
| `queue` (submitted → dequeued) | 1.9 | 11.8 | 2.3 | 196.5 |
| `stt_dispatch` | 0.1 | 22.6 | 0.2 | 254.9 |
| `stt_call` (Parakeet, includes wrapper waits) | 414.3 | 1218.6 | 395.5 | 1284.2 |
| `to_translation` | 0.4 | 4.2 | 0.2 | 4.0 |
| `translation_lock_wait` | 0.0 | 462.8 | 0.0 | 406.2 |
| `translation_prepare` | 0.0 | 37.3 | 0.0 | 78.3 |
| `translation_call` (Gemma E4B) | 381.3 | 1289.1 | 394.4 | 1264.8 |
| `finalize` | 0.5 | 1.8 | 0.5 | 2.3 |
| `broadcast` (after final ready; outside total) | 0.8 | 4.3 | 0.1 | 0.2 |
| **`total_speech_end_to_final`** | **1429.7** | **3019.0** | **1458.7** | **3283.2** |
| `prefill_ms_a` (n=205) | 115.1 | 169.3 | 120.8 | 177.5 |
| `ttft_ms_a` (n=205) | 299.2 | 408.6 | 298.3 | 484.5 |
| `decode_ms_a` (n=205) | 456.4 | 1039.8 | 449.5 | 981.5 |
| `gen_tokens_a` (tokens, n=205) | 14.0 | 31.0 | 14.0 | 31.0 |
| `utterance_dur` (s) | 2.2 | 6.8 | 2.2 | 6.8 |

Full tables including smart-cut, hard-cut and EOF endpoints: `stages_752ab9a.md`, `stages_760e948.md` (JSON beside them).

## Ranked statement (both cohorts agree)

1. **Fixed silence trigger ≈ 0.49 s at the median** (34 %). This is the 0.5 s `silence_trigger` policy plus VAD bookkeeping; it is not compute. Earlier screens already rejected shorter triggers on quality/preview guards, so it is a policy floor, not an optimization target.
2. **STT call ≈ 0.40 s median, 1.2–1.3 s p95** (28 %). Parakeet on a 2.2 s median utterance; the p95 tail is the largest single contributor to the total p95. Isolated Parakeet calls on whole 47 s FLEURS recordings take ~220 ms, so per-utterance overhead (wrapper waits, Metal contention with preview work) dominates here.
3. **Gemma E4B final translation ≈ 0.38–0.39 s median, 1.26–1.29 s p95** (27 %). Prefill ~120 ms, first token ~300 ms, decode ~450 ms for 14 tokens. Decode is where a draft model (lane L2) could act; prefill/ttft it cannot.
4. **Everything else < 5 ms at the median.** Queue, dispatch, lock wait, prepare, finalize and broadcast are negligible at p50; lock wait and queue show p95 tails of 0.2–0.5 s when a preview job holds the Metal lock.

Arithmetic consequence: with the 0.5 s trigger fixed, STT + translation must fall below ~0.5 s combined at the median to reach a sub-second median. Neither a 15 % translation gain nor a 15 % STT gain alone gets there; both together (≈ −120 ms) still leave the median near 1.3 s. This is why the sub-second goal remained unmet across all screens, and why the next credible levers are the p95 tails (Metal contention between preview and final work), not the medians.

This is engineering attribution on machine-timed replays. It does not certify human quality, physical display timing or live-microphone behaviour.
