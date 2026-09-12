# P2 — `partial_reuse_ms` arm screen → **REJECTED** (both thresholds); confidence variant not screened

**Arm.** Opt-in `STARK_EXPERIMENT_PARTIAL_REUSE_MS` (PR #218): when a silence-ended final's speech end lies within the
threshold of the last emitted partial's STT window, the final reuses that partial's English instead of running a second
Parakeet call (`stt_confidence=None`, so the Gemma route is taken; cuts always run full STT; `final_stt_route` per
final in the diagnostics rows). Admitted by the [P2 attribution](../P2-attribution/README.md): the last partial reaches
within 300 ms of the speech end in 98.5 % of silence finals, its result exists before finalization in 100 %, its text
equals the batch final's in 84 %, and the replaced final STT costs 360 ms p50.

**Screen.** Protocol `partial_reuse_screen_series4_20260912` ([`protocol.json`](protocol.json), declared
2026-09-12T00:15Z, checkout `56a045b` = the arm on main `815dade` with #216/#217 recorded before run 1). Control vs
`reuse100` (100 ms) vs `reuse300` (300 ms), 3 order-balanced repeats per arm on each 360 s clip, 18/18 runs rc 0,
02:10Z → 04:02Z, `venv/bin/python`, one process at a time, trace on, HF offline. Report [`report.md`](report.md)
(`tools/tail_screen_report.py`, all inputs hashed), per-run metrics [`metrics.json`](metrics.json) (`screen_metrics.py`),
text guard [`quality_guard.json`](quality_guard.json) (`quality_guard.py`; every differing final listed),
declared-gate verdicts [`verdicts.json`](verdicts.json) (`evaluate_gates.py`; the tail reading — both screened
metrics at p95 — was fixed at 03:15Z, before clip B).

## Numbers (pooled per clip, 3 runs each)

| clip | arm | Gemma-routed silence n | first-token p50 / p95 ms | payload-ready p50 / p95 | all-silence ready p50 / p95 (n) | all finals ready p50 / p95 | Marian-routed finals | reused / silence finals per run | EN WER vs ctl | ES WER vs ctl |
|---|---|---|---|---|---|---|---|---|---|---|
| A | ctl | 45 | 1165 / 1362 | 1960 / 2399 | 1635 / 2399 (63) | 2271 / 5406 | 24 | — | — | — |
| A | reuse100 | 60 | 905 / 1569 (−22 % / **+15 %**) | 1523 / 2300 (−22 % / −4 %) | 1530 / 2338 (−6 % / −3 %) | 2201 / 5623 | 9 | 13–15 / 21 | 0.17 % | 0.9–1.1 % |
| A | reuse300 | 62 | 892 / 1689 (−24 % / **+24 %**) | 1633 / 2243 (−17 % / −7 %) | 1633 / 2243 (0 % / −7 %) | 2152 / 5496 | 7 | 17–19 / 21 | 0.34 % | 1.28 % |
| B | ctl | 129 | 1095 / 1989 | 1607 / 2508 | 1468 / 2298 (204) | 1524 / 4293 | 78 | — | — | — |
| B | reuse100 | 184 | 896 / 1551 (−18 % / −22 %) | 1411 / 2366 (**−12 %** / −6 %) | 1379 / 2160 (−6 % / −6 %) | 1449 / 4291 | 28 | 43–48 / 68 | **2.7–4.2 %** | **3.7–5.0 %** |
| B | reuse300 | 210 | 882 / 973 (−19 % / −51 %) | 1329 / 1746 (−17 % / −30 %) | 1329 / 1746 (−9 % / −24 %) | 1379 / 3835 | 6 | 68–70 / 68 (+3 finals) | **6.3–6.5 %** | **7.9–8.0 %** |

Cut-final p95 (tool G2): clip A 5511 → 6147 (reuse100, +12 %) / 6958 (reuse300, +26 %); clip B 6246 → 6109 / 6115.
Previews within 1 % of control on both clips. Memory: one reuse100 run on clip A exceeded the relative RSS rule
(4.30 vs 4.07 GiB limit); Metal peaks unchanged.

## Gates and verdict

| declared gate | reuse100 | reuse300 |
|---|---|---|
| G_median (Gemma-silence first-token AND payload-ready p50 ≥ 15 % better on both clips) | A pass; **B fail** (ready −12 %) | pass on both clips |
| G_tail (no p95 regression beyond max(5 %, 100 ms), both metrics, three cohorts) | **A fail** (first-token p95 +207 ms) | **A fail** (first-token p95 +327 ms) |
| G_preview (within 2 pp) | pass | pass |
| G_text (EN and ES WER vs control ≤ 3 %, bad_split not above control) | A pass; **B fail** | A pass; **B fail** |
| G_memory (tool G6 relative rule) | **A fail** (one run) | pass |
| tool G7 (returncode, lifecycle, chunk count) | pass | pass |

**Both arms REJECTED.** Two independent failures:

1. **The text guard on the denser clip.** Reusing the partial's transcript changes 15 % of silence finals (attribution),
   which on clip B (68 silence finals per run, 43–70 reused) becomes 2.7–6.5 % English WER and 3.7–8.0 % Spanish WER
   against control, and at 300 ms three extra finals per run. The differing finals are listed in
   `quality_guard.json` for a bilingual reviewer; they are mostly disfluency and boundary-word differences, but the
   declared guard is 3 % and it is not met.
2. **Routing side-effect and tails.** A reused final has no STT confidence, so `should_use_marian_only` never routes it
   to Marian: 15–17 finals per run on clip A (24 → 9 / 7 Marian-routed) moved from the ≈ 0.8 s Marian path to Gemma.
   The Gemma cohort's median gain (−22 %) is therefore partly a composition change — the all-silence median moved only
   −6 % / 0 % on clip A — and the extra Gemma work regressed the first-token p95 (+15 % / +24 %) and the cut-final p95
   (+12 % / +26 %) on clip A. On clip B, where Gemma finals dominate, the same arm improved every tail (payload-ready
   p95 −30 % at 300 ms); the mechanism is real when the reused finals were Gemma-bound anyway.

**Confidence-preserving variant (not screened).** `partial_reuse_keep_confidence` (branch
`codex/series4-p2-reuse-confidence` @ `9314a8d`, tests green, not merged) keeps the partial's confidence so the Marian
route stays available; its 12-run screen `partial_reuse_confidence_screen_series4_20260912`
([`variant/protocol.json`](variant/protocol.json)) was declared and started at 04:02Z, then **cancelled before its first
candidate run** ([`variant/screen.log`](variant/screen.log)): the routing change cannot repair the clip-B text-guard
failure, which comes from the reused text itself, so the variant could not pass its declared gates. It is not a rejected
arm (never screened) and stays available if a reviewer later judges the listed text differences acceptable.

**Decision.** REJECTED. The opt-in flag is merged off by default like the other screened-and-rejected experiment flags
(registry row in `docs/latency_next_experiments.md`); no default changes; this screen makes no p95 claim
(`p95_claim_eligible: false`), no human-quality and no visible-display statement. What would change the verdict is a
human judgement that the listed final-text differences are acceptable (then the variant is the arm to screen), not more
machine-timed runs.
