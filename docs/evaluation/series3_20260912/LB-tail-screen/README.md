# L-B — declared screen `tail_screen_series3_20260912` (2026-09-11 → 12)

**Outcome: all three admitted arms REJECTED on the pre-declared gates**, on both clips, primarily on G1 (Gemma-routed
silence-final p95 must fall ≥ 15 % or ≥ 300 ms): `marian_threads_2` made the tail worse on both clips;
`max_utterance_6` lowered the median but raised the tail on clip A, barely moved it on clip B, changed the
segmentation (as expected) and failed its substitute quality guard; `partial_recheck_translation` cut the clip A
Gemma p95 by 8.8 % — the right direction, below the gate — and did nothing on clip B, while costing 7–15 % of
previews. No default changes; no combination follows; these arms join the closed-arm registry.

Declaration: [`protocol.json`](protocol.json) (SHA256 `e96d927ac12f53d8…`, written 20:10Z before run 1, admission reasons from
L-A recorded inside). Runs: 24/24 rc 0 ([`runs.jsonl`](runs.jsonl), [`screen.log`](screen.log)), `main` @ `a46649e`,
promoted `venv`, two 360 s clips (A = 12_14_25 @ 1290 s, B = 2_8_26 @ 1170 s), 3 repeats per configuration with
controls at both ends and candidate order reversed in the middle repetition, `STARK_EXPERIMENT_TRACE=true` on all
runs. Every arm's setting is proven in `session_metadata` (`marian_intra_threads` 2, `max_utterance` 6.0,
`partial_recheck_translation` true; see [`quality_guard.json`](quality_guard.json)). Report:
[`report.md`](report.md) / [`report.json`](report.json) (`tools/tail_screen_report.py`, sequence-aligned G4, PR #214);
per-run replay summaries under [`runs/`](runs/). Eligible Gemma-routed silence finals per arm: 45 (A) + 114–129 (B) —
below 50 on clip A, so this remains a **screen without p95 claim**.

## Results (pooled over 3 repeats per clip; nearest-rank; ms)

| clip | arm | Gemma / Marian silence n | Gemma silence n | all-route silence p50 | all-route silence p95 | Gemma silence p95 | previews per run | `partial_suppressed_translation_running` |
|---|---|---|---:|---:|---:|---:|---|---|
| A | ctl | 159 / 24 / 0 | 45 | 1650.1 | 2477.3 | 2481.5 | 514, 516, 508 | 0, 0, 0 |
| A | marian_threads_2 | 159 / 24 / 0 | 45 | 1618.5 | 2779.1 | 2875.0 | 518, 516, 499 | 0, 0, 0 |
| A | max_utterance_6 | 183 / 48 / 0 | 36 | 1476.2 | 2684.7 | 3182.8 | 488, 487, 472 | 0, 0, 0 |
| A | partial_recheck_translation | 159 / 24 / 0 | 45 | 1592.2 | 2247.8 | 2263.0 | 438, 448, 448 | 82, 79, 78 |
| B | ctl | 162 / 78 / 0 | 129 | 1467.9 | 1843.2 | 1886.5 | 448, 443, 450 | 0, 0, 0 |
| B | marian_threads_2 | 162 / 78 / 0 | 129 | 1496.8 | 1942.8 | 2008.8 | 450, 445, 445 | 0, 0, 0 |
| B | max_utterance_6 | 171 / 96 / 0 | 114 | 1384.7 | 1795.3 | 1841.9 | 445, 444, 444 | 0, 0, 0 |
| B | partial_recheck_translation | 162 / 78 / 0 | 129 | 1469.4 | 1887.9 | 1892.1 | 413, 419, 416 | 32, 34, 31 |

Gate verdicts (`report.md` has the per-gate numbers; G1 limit = 0.85 × control or control − 300 ms):

| arm | clip A | clip B | outcome |
|---|---|---|---|
| `marian_threads_2` | G1 FAIL (2 875 vs 2 482, limit 2 182), G2 FAIL, G5 FAIL | G1 FAIL (2 009 vs 1 887, limit 1 604) | **REJECTED** |
| `max_utterance_6` | G1 FAIL (3 183 vs 2 482), G2 FAIL, G5 FAIL, G7 FAIL (chunk count +26 %) | G1 FAIL (1 842 vs 1 887, limit 1 604), G7 FAIL (+11 %) | **REJECTED** |
| `partial_recheck_translation` | G1 FAIL (2 263 vs 2 482 = −8.8 %, limit 2 182), G5 FAIL (previews −13 %) | G1 FAIL (1 892 vs 1 887), G5 FAIL (previews −7 %) | **REJECTED** |

Substitute quality guard for `max_utterance_6` (declared: no empty/duplicate finals and concatenated-Spanish WER vs
the same-repeat control ≤ +1 pt): clip A **20.3 % WER** with 2 duplicated finals in every repeat, clip B 3.9 % with 1
duplicate — **fails** on both clips (the 6 s cap re-segments and re-translates the long passages differently). For
the two non-segmenting arms the concatenated Spanish is byte-identical to control on both clips (0.0 % WER), i.e.
they changed timing only.

## Reading

- **`marian_threads_2` — REJECTED.** Halving Marian CT2's threads did not relieve the MLX host thread; the Gemma p95
  rose 16 % on clip A and 6 % on clip B and previews fell on clip A. CPU-side contention with previews is not the
  lever (or 2 threads simply slow the previews enough to hurt elsewhere).
- **`max_utterance_6` — REJECTED (policy arm).** Shorter cuts lower the all-route median (−11 % A, −6 % B) because
  each final is shorter, but the tail got worse on clip A (+28 % Gemma p95) and the guard shows the re-segmentation
  changes what is said (20 % concatenated WER on clip A, duplicated finals). Not a candidate for Willem's review.
- **`partial_recheck_translation` — REJECTED, closest to the gate.** The flag did what it should (78–82 partials per
  run suppressed on clip A, 31–34 on clip B, exactly the overlap L-A measured) and the clip A Gemma p95 fell 8.8 %
  with the median −3.5 %; on clip B, where the overlap is smaller, nothing moved. It fails the 15 %/300 ms gate on
  both clips and the preview-coverage guard. A third of decode time overlapping preview STT is therefore worth at
  most ~9 % of the tail on this hardware — the decode itself (14–35 tokens at ~30 tok/s) is the cost, not the
  contention.
- With this, every hypothesis that does not change the model, the token count or the goal's metric has been
  screened and closed. The remaining levers are outside a screen: a smaller/faster final model with a quality
  review, fewer output tokens, or restating the goal on the first-token measure (L-C).

This is a server-timed replay screen; it certifies no human quality, visible delivery or production default, and
changes none. Rejected arms are never re-run as confirmations and never enter combinations.
