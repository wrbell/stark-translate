# Compact caption screening report

**Outcome: 96/96 valid runs, 672 finals, and 0/28 selected experiment/model arms.**
The sub-second final-delivery goal was not met on this 45-second English cohort.
E4B, 0.5-second silence and 0.6-second partial cadence remain unchanged. These arms
justify no ordinary historical confirmation or combined configuration. Endpoint
samples are small and references are unreviewed; this is a bounded screen.

Final text stayed unchanged in 588 comparisons against opening controls and 588
against closing controls. That repeat agreement is not reference-quality approval;
[the first-preview E4B note](first_preview_e4b.md) explains the control drift and
provisional wording tradeoff. New hypotheses and the separately reported standard
and CPU Lite endurance work remain distinct from promotion of these arms.

The [full supplemental JSON and raw archive](artifact-manifest.json) are deterministic gzip streams, split into parts of at most 480 KiB. [File hashes](inventory.jsonl.gz) and [provenance/limits](provenance.json) describe every member. Reconstruct and verify with the stdlib-only [verifier](verify_evidence.py):

```sh
python verify_evidence.py --root . --output /tmp/stark-screen-evidence
```

Use a new output directory. This produces readable `analysis.json` and the complete raw `screen-evidence.tar.gz`; it verifies every member without extracting source files. The archive retains original absolute paths in provenance.

**Collection limits:** the native Mac was locked while the controlled browser DOM reported visible. ACKs are browser telemetry, not proof of physical screen visibility. Package activity occurred 2026-09-10 05:51:37 to 05:53:18.380938 UTC; overlapping sessions are listed in provenance and retained. Its performance effect was not measured.

[Independent raw-data audit](independent-audit.json): 24,324 checks, no discrepancies; 672 finals (96 hard_cut, 96 silence, 96 silence_replay_tail, 384 smart_cut). This verifies recorded evidence and calculation consistency, not reference quality or physical display visibility.

**COMPLETE SCREEN: 96/96 recorded; 96 valid.**

Directions actually present: EN→ES. [Full analysis and per-session evidence](artifact-manifest.json).

No production promotion is made here. WER, reference translation quality and human approval are unavailable in this screen. Natural/locked-source EN and synthetic ES are separate workloads; they must not be pooled. A frozen native runtime/source cohort is not native-device certification.

## Baselines: server payload readiness

Raw observations pooled only within each model/source/clock/control cohort; p50 / nearest-rank p95 (n), rounded milliseconds. Opening and closing controls are separate. Raw `silence` can occur after artificial EOF padding: analytical `_replay_tail` groups are separate and cannot be the sole final-latency selection target. Speech end is the last VAD-positive frame, not acoustic ground truth or browser display.

| Model / direction / source | Control (runs) | Endpoint / clock | Final p50 / p95 (n) |
|---|---|---|---|
| E2B · EN→ES · screening_45s (natural) · 4d25de1e | baseline (3) | hard_cut / replay_realtime | 1,327 / 2,410 (3) |
| E2B · EN→ES · screening_45s (natural) · 4d25de1e | baseline (3) | silence_replay_tail (virtual EOF padding) / replay_realtime | 1,361 / 1,444 (3) |
| E2B · EN→ES · screening_45s (natural) · 4d25de1e | baseline (3) | silence (real audio, no virtual padding) / replay_realtime | 1,742 / 2,122 (3) |
| E2B · EN→ES · screening_45s (natural) · 4d25de1e | baseline (3) | smart_cut / replay_realtime | 3,107 / 4,090 (12) |
| E2B · EN→ES · screening_45s (natural) · 4d25de1e | baseline_anchor (3) | hard_cut / replay_realtime | 1,333 / 1,337 (3) |
| E2B · EN→ES · screening_45s (natural) · 4d25de1e | baseline_anchor (3) | silence_replay_tail (virtual EOF padding) / replay_realtime | 1,447 / 1,450 (3) |
| E2B · EN→ES · screening_45s (natural) · 4d25de1e | baseline_anchor (3) | silence (real audio, no virtual padding) / replay_realtime | 1,761 / 2,101 (3) |
| E2B · EN→ES · screening_45s (natural) · 4d25de1e | baseline_anchor (3) | smart_cut / replay_realtime | 3,099 / 3,793 (12) |
| E4B · EN→ES · screening_45s (natural) · 54f77975 | baseline (3) | hard_cut / replay_realtime | 2,635 / 2,726 (3) |
| E4B · EN→ES · screening_45s (natural) · 54f77975 | baseline (3) | silence_replay_tail (virtual EOF padding) / replay_realtime | 1,759 / 1,953 (3) |
| E4B · EN→ES · screening_45s (natural) · 54f77975 | baseline (3) | silence (real audio, no virtual padding) / replay_realtime | 2,253 / 3,661 (3) |
| E4B · EN→ES · screening_45s (natural) · 54f77975 | baseline (3) | smart_cut / replay_realtime | 3,706 / 4,996 (12) |
| E4B · EN→ES · screening_45s (natural) · 54f77975 | baseline_anchor (3) | hard_cut / replay_realtime | 2,587 / 2,645 (3) |
| E4B · EN→ES · screening_45s (natural) · 54f77975 | baseline_anchor (3) | silence_replay_tail (virtual EOF padding) / replay_realtime | 1,806 / 1,926 (3) |
| E4B · EN→ES · screening_45s (natural) · 54f77975 | baseline_anchor (3) | silence (real audio, no virtual padding) / replay_realtime | 2,308 / 2,963 (3) |
| E4B · EN→ES · screening_45s (natural) · 54f77975 | baseline_anchor (3) | smart_cut / replay_realtime | 3,814 / 5,225 (12) |

Replay-tail inventory: E2B/baseline/4d25de1e: 18 recorded-endpoint finals, 3 virtual-tail finals (raw endpoint counts {'smart_cut': 12, 'hard_cut': 3, 'silence': 6}); E2B/baseline_anchor/4d25de1e: 18 recorded-endpoint finals, 3 virtual-tail finals (raw endpoint counts {'smart_cut': 12, 'hard_cut': 3, 'silence': 6}); E4B/baseline/54f77975: 18 recorded-endpoint finals, 3 virtual-tail finals (raw endpoint counts {'smart_cut': 12, 'hard_cut': 3, 'silence': 6}); E4B/baseline_anchor/54f77975: 18 recorded-endpoint finals, 3 virtual-tail finals (raw endpoint counts {'smart_cut': 12, 'hard_cut': 3, 'silence': 6}).
All-utterance preview observations remain available: a preview of real recorded speech is not excluded merely because its final later needs EOF padding.

| Model / cohort / control | First translated server preview, all utterances p50 / p95 (n) | Within-utterance update gap p50 / p95 (n) |
|---|---|---|
| E2B / 4d25de1e / baseline | 785 / 3,663 (21) | 623 / 730 (183) |
| E2B / 4d25de1e / baseline_anchor | 890 / 3,664 (21) | 618 / 674 (186) |
| E4B / 54f77975 / baseline | 1,997 / 4,450 (21) | 620 / 961 (171) |
| E4B / 54f77975 / baseline_anchor | 1,392 / 4,534 (21) | 620 / 1,199 (156) |

## Baseline stage decomposition

Diagnostic stage wall times, p50 / p95 (n), milliseconds. Calls include waits inside the inference runtime; endpoint decision starts at the chosen last VAD-positive frame. An earlier smart-cut boundary can therefore create a large decision delay. Stage medians must not be added to claim a median total; cached/speculative generation counters are not final-path wall times.

| Model / cohort / control | Endpoint / clock | Endpoint decision | Submit→dequeue | STT call wall | Translation call wall |
|---|---|---|---|---|---|
| E2B / 4d25de1e / baseline | hard_cut / replay_realtime | 8 / 12 (3) | 15 / 41 (3) | 525 / 1,470 (3) | 776 / 860 (3) |
| E2B / 4d25de1e / baseline | silence_replay_tail (virtual EOF padding) / replay_realtime | 491 / 492 (3) | 4 / 4 (3) | 373 / 430 (3) | 497 / 518 (3) |
| E2B / 4d25de1e / baseline | silence (real audio, no virtual padding) / replay_realtime | 492 / 493 (3) | 11 / 14 (3) | 460 / 858 (3) | 760 / 777 (3) |
| E2B / 4d25de1e / baseline | smart_cut / replay_realtime | 2,187 / 3,086 (12) | 0 / 0 (12) | 180 / 212 (12) | 682 / 879 (12) |
| E2B / 4d25de1e / baseline_anchor | hard_cut / replay_realtime | 8 / 12 (3) | 12 / 14 (3) | 549 / 551 (3) | 760 / 763 (3) |
| E2B / 4d25de1e / baseline_anchor | silence_replay_tail (virtual EOF padding) / replay_realtime | 493 / 494 (3) | 4 / 4 (3) | 434 / 446 (3) | 511 / 516 (3) |
| E2B / 4d25de1e / baseline_anchor | silence (real audio, no virtual padding) / replay_realtime | 490 / 492 (3) | 11 / 12 (3) | 460 / 830 (3) | 785 / 796 (3) |
| E2B / 4d25de1e / baseline_anchor | smart_cut / replay_realtime | 2,185 / 3,085 (12) | 0 / 0 (12) | 182 / 216 (12) | 688 / 731 (12) |
| E4B / 54f77975 / baseline | hard_cut / replay_realtime | 12 / 12 (3) | 17 / 18 (3) | 1,134 / 1,182 (3) | 1,356 / 1,519 (3) |
| E4B / 54f77975 / baseline | silence_replay_tail (virtual EOF padding) / replay_realtime | 492 / 492 (3) | 4 / 4 (3) | 473 / 632 (3) | 788 / 823 (3) |
| E4B / 54f77975 / baseline | silence (real audio, no virtual padding) / replay_realtime | 492 / 492 (3) | 11 / 14 (3) | 549 / 1,947 (3) | 1,195 / 1,211 (3) |
| E4B / 54f77975 / baseline | smart_cut / replay_realtime | 2,187 / 3,094 (12) | 0 / 0 (12) | 206 / 681 (12) | 1,148 / 1,735 (12) |
| E4B / 54f77975 / baseline_anchor | hard_cut / replay_realtime | 7 / 14 (3) | 27 / 42 (3) | 1,207 / 1,216 (3) | 1,342 / 1,380 (3) |
| E4B / 54f77975 / baseline_anchor | silence_replay_tail (virtual EOF padding) / replay_realtime | 494 / 494 (3) | 4 / 4 (3) | 503 / 583 (3) | 803 / 855 (3) |
| E4B / 54f77975 / baseline_anchor | silence (real audio, no virtual padding) / replay_realtime | 491 / 492 (3) | 12 / 13 (3) | 602 / 1,212 (3) | 1,207 / 1,246 (3) |
| E4B / 54f77975 / baseline_anchor | smart_cut / replay_realtime | 2,185 / 3,085 (12) | 0 / 4 (12) | 210 / 703 (12) | 1,638 / 1,921 (12) |

## Visible browser evidence

Only the analyzer's bound display role is aggregated; an unbound/ambiguous session contributes no browser measurement. Socket IDs are session-local: role binding does not prove one persistent physical browser, foreground lock, or pixels remaining visible. ACK timing includes the return network trip. Detailed original client IDs, timing/order exceptions and missing events remain in the linked JSON.

| Model / cohort / control | Bound sessions / connections | Final ACKs / finals | First visible p50 / p95 (n) | Final p50 / p95 (n), by endpoint |
|---|---|---|---|---|
| E2B / 4d25de1e / baseline | 3/3 / 3 | 21/21 | 894 / 3,678 (21); expected 21 utterances | hard_cut / replay_realtime: 1,340 / 2,421 (3); silence_replay_tail (virtual EOF padding) / replay_realtime: 1,377 / 1,455 (3); silence (real audio, no virtual padding) / replay_realtime: 1,754 / 2,134 (3); smart_cut / replay_realtime: 3,123 / 4,106 (12) |
| E2B / 4d25de1e / baseline_anchor | 3/3 / 3 | 21/21 | 905 / 3,675 (21); expected 21 utterances | hard_cut / replay_realtime: 1,349 / 1,354 (3); silence_replay_tail (virtual EOF padding) / replay_realtime: 1,464 / 1,465 (3); silence (real audio, no virtual padding) / replay_realtime: 1,771 / 2,111 (3); smart_cut / replay_realtime: 3,113 / 3,809 (12) |
| E4B / 54f77975 / baseline | 3/3 / 3 | 21/21 | 2,013 / 4,468 (21); expected 21 utterances | hard_cut / replay_realtime: 2,648 / 2,738 (3); silence_replay_tail (virtual EOF padding) / replay_realtime: 1,773 / 1,968 (3); silence (real audio, no virtual padding) / replay_realtime: 2,263 / 3,670 (3); smart_cut / replay_realtime: 3,719 / 5,009 (12) |
| E4B / 54f77975 / baseline_anchor | 3/3 / 3 | 21/21 | 1,402 / 4,549 (21); expected 21 utterances | hard_cut / replay_realtime: 2,598 / 2,665 (3); silence_replay_tail (virtual EOF padding) / replay_realtime: 1,819 / 1,940 (3); silence (real audio, no virtual padding) / replay_realtime: 2,325 / 2,976 (3); smart_cut / replay_realtime: 3,829 / 5,239 (12) |

## Experiment observations

Δ is the median paired candidate-control difference (negative = faster), shown versus opening / closing controls with matched n. The table reports only silence finals and first bound visible previews for compactness; every endpoint and guard remains in full analysis. Descriptive deltas never override a failed guard or incomplete matrix.

| Experiment / model / cohort | Assessment | Silence-final Δ ms (n), open / close | First-visible Δ ms (n), open / close | Guard/reason summary |
|---|---|---|---|---|
| allocation_1024 / E2B / EN→ES / screening_45s / 4d25de1e | not selected by analyzer | +21 (3) / +68 (3) | +10 (21) / +18 (21) | guard: server-final tail, visible-final tail, preview→final tail, update-gap tail; median threshold unmet |
| allocation_1024 / E4B / EN→ES / screening_45s / 54f77975 | not selected by analyzer | +767 (3) / +711 (3) | +347 (21) / +329 (21) | guard: server-final tail, visible-final tail, first-visible tail, first-server tail, preview→final tail; guard: server-final tail, visible-final tail, first-visible tail, first-server tail, update-gap tail, RSS; guard: server-final tail, visible-final tail, preview→final tail, update-gap tail; median threshold unmet |
| allocation_512 / E2B / EN→ES / screening_45s / 4d25de1e | not selected by analyzer | +17 (3) / +56 (3) | +14 (21) / +17 (21) | guard: server-final tail, visible-final tail; guard: server-final tail, visible-final tail, first-visible tail, first-server tail, preview→final tail, update-gap tail; median threshold unmet |
| allocation_512 / E4B / EN→ES / screening_45s / 54f77975 | not selected by analyzer | +562 (3) / +160 (3) | +224 (21) / -9 (21) | guard: server-final tail, visible-final tail, first-visible tail, first-server tail, preview→final tail; guard: server-final tail, visible-final tail, first-visible tail, first-server tail, update-gap tail, RSS; guard: server-final tail, visible-final tail, preview→final tail; median threshold unmet |
| async_captions / E2B / EN→ES / screening_45s / 4d25de1e | not selected by analyzer | -57 (3) / -10 (3) | +1 (21) / +7 (21) | guard: server-final tail, visible-final tail; guard: server-final tail, visible-final tail, first-visible tail, first-server tail, update-gap tail; guard: server-final tail, visible-final tail, preview→final tail, update-gap tail; median threshold unmet |
| async_captions / E4B / EN→ES / screening_45s / 54f77975 | not selected by analyzer | +122 (3) / +706 (3) | +28 (21) / -45 (21) | coverage/segmentation; guard: server-final tail, visible-final tail, first-visible tail, first-server tail, update-gap tail; median threshold unmet |
| clause_preview / E2B / EN→ES / screening_45s / 4d25de1e | not selected by analyzer | +22 (3) / +44 (3) | -2 (21) / -1 (21) | guard: server-final tail, visible-final tail, first-visible tail, preview→final tail, update-gap tail; guard: server-final tail, visible-final tail, preview→final tail, update-gap tail; median threshold unmet |
| clause_preview / E4B / EN→ES / screening_45s / 54f77975 | not selected by analyzer | -48 (3) / -104 (3) | +5 (21) / -124 (21) | guard: server-final tail, visible-final tail, first-visible tail, first-server tail, preview→final tail, update-gap tail; guard: server-final tail, visible-final tail, update-gap tail; median threshold unmet |
| first_preview / E2B / EN→ES / screening_45s / 4d25de1e | not selected by analyzer | +42 (3) / +0 (3) | -136 (21) / -113 (21) | guard: preview→final tail; guard: server-final tail, visible-final tail, preview→final tail; median threshold unmet |
| first_preview / E4B / EN→ES / screening_45s / 54f77975 | not selected by analyzer | +116 (3) / -86 (3) | -103 (21) / -515 (21) | guard: server-final tail, visible-final tail; guard: server-final tail, visible-final tail, first-visible tail, first-server tail, update-gap tail; guard: server-final tail, visible-final tail, preview→final tail; median threshold unmet |
| latest_partial / E2B / EN→ES / screening_45s / 4d25de1e | not selected by analyzer | +0 (3) / -70 (3) | +15 (21) / +75 (21) | guard: server-final tail, visible-final tail, first-visible tail, first-server tail, preview→final tail; guard: server-final tail, visible-final tail, preview→final tail; median threshold unmet |
| latest_partial / E4B / EN→ES / screening_45s / 54f77975 | not selected by analyzer | +35 (3) / -22 (3) | +44 (21) / -11 (21) | guard: server-final tail, visible-final tail, first-visible tail, first-server tail, preview→final tail, update-gap tail; guard: server-final tail, visible-final tail, first-visible tail, first-server tail, update-gap tail, RSS; guard: server-final tail, visible-final tail, preview→final tail; median threshold unmet |
| marian_memo / E2B / EN→ES / screening_45s / 4d25de1e | not selected by analyzer | -32 (3) / -17 (3) | +5 (21) / +7 (21) | guard: preview→final tail; median threshold unmet |
| marian_memo / E4B / EN→ES / screening_45s / 54f77975 | not selected by analyzer | +92 (3) / +772 (3) | +44 (21) / -26 (21) | guard: server-final tail, visible-final tail; guard: server-final tail, visible-final tail, first-visible tail, first-server tail, update-gap tail, RSS; guard: server-final tail, visible-final tail, preview→final tail; median threshold unmet |
| pause_preview / E2B / EN→ES / screening_45s / 4d25de1e | not selected by analyzer | -23 (3) / -41 (3) | -0 (21) / -1 (21) | guard: server-final tail, visible-final tail, preview→final tail; median threshold unmet |
| pause_preview / E4B / EN→ES / screening_45s / 54f77975 | not selected by analyzer | +15 (3) / -78 (3) | -27 (21) / -249 (21) | guard: preview→final tail; guard: server-final tail, visible-final tail, first-visible tail, first-server tail, preview→final tail, update-gap tail, RSS; guard: server-final tail, visible-final tail, preview→final tail, update-gap tail; median threshold unmet |
| pause_speculation / E2B / EN→ES / screening_45s / 4d25de1e | not selected by analyzer | +688 (3) / +735 (3) | +69 (21) / +72 (21) | guard: server-final tail, visible-final tail, first-visible tail, first-server tail, preview→final tail; guard: server-final tail, visible-final tail, preview→final tail; median threshold unmet |
| pause_speculation / E4B / EN→ES / screening_45s / 54f77975 | not selected by analyzer | +1,178 (3) / +1,801 (3) | +19 (21) / -111 (21) | guard: server-final tail, visible-final tail, first-visible tail, first-server tail, preview→final tail, update-gap tail; guard: server-final tail, visible-final tail, preview→final tail; median threshold unmet |
| prefix_cache / E2B / EN→ES / screening_45s / 4d25de1e | not selected by analyzer | -40 (3) / +6 (3) | +11 (21) / +11 (21) | guard: preview→final tail, update-gap tail; guard: server-final tail, visible-final tail, preview→final tail, update-gap tail; median threshold unmet |
| prefix_cache / E4B / EN→ES / screening_45s / 54f77975 | not selected by analyzer | -95 (3) / +27 (3) | +55 (21) / +29 (21) | guard: server-final tail, visible-final tail, first-visible tail, first-server tail, RSS; guard: server-final tail, visible-final tail, preview→final tail; guard: server-final tail, visible-final tail, preview→final tail, update-gap tail; median threshold unmet |
| rolling_6s / E2B / EN→ES / screening_45s / 4d25de1e | not selected by analyzer | -14 (3) / -35 (3) | +22 (21) / +20 (21) | guard: server-final tail, visible-final tail, first-visible tail, first-server tail, preview→final tail; guard: server-final tail, visible-final tail, preview→final tail; median threshold unmet |
| rolling_6s / E4B / EN→ES / screening_45s / 54f77975 | not selected by analyzer | +63 (3) / +558 (3) | +37 (21) / -76 (21) | guard: server-final tail, visible-final tail; guard: server-final tail, visible-final tail, preview→final tail; guard: server-final tail, visible-final tail, preview→final tail, update-gap tail, RSS; median threshold unmet |
| streaming_stt / E2B / EN→ES / screening_45s / 4d25de1e | not selected by analyzer | +392 (3) / +323 (3) | +157 (21) / +194 (21) | guard: server-final tail, visible-final tail, first-visible tail, first-server tail, preview→final tail; median threshold unmet |
| streaming_stt / E4B / EN→ES / screening_45s / 54f77975 | not selected by analyzer | +511 (3) / +939 (3) | +159 (21) / -142 (21) | guard: server-final tail, visible-final tail, first-visible tail, first-server tail; guard: server-final tail, visible-final tail, first-visible tail, first-server tail, preview→final tail, update-gap tail; guard: server-final tail, visible-final tail, preview→final tail; median threshold unmet |
| vad_worker_onnx / E2B / EN→ES / screening_45s / 4d25de1e | not selected by analyzer | +57 (3) / -13 (3) | +10 (21) / +7 (21) | guard: server-final tail, visible-final tail, preview→final tail, update-gap tail; median threshold unmet |
| vad_worker_onnx / E4B / EN→ES / screening_45s / 54f77975 | not selected by analyzer | -48 (3) / -104 (3) | +20 (21) / +27 (21) | guard: server-final tail, visible-final tail, first-visible tail, first-server tail, preview→final tail, update-gap tail; guard: server-final tail, visible-final tail, first-visible tail, first-server tail, update-gap tail; guard: server-final tail, visible-final tail, update-gap tail; median threshold unmet |
| vad_worker_torch / E2B / EN→ES / screening_45s / 4d25de1e | not selected by analyzer | +6 (3) / -36 (3) | +2 (21) / -3 (21) | guard: server-final tail, visible-final tail, update-gap tail; median threshold unmet |
| vad_worker_torch / E4B / EN→ES / screening_45s / 54f77975 | not selected by analyzer | +630 (3) / +575 (3) | +22 (21) / -98 (21) | guard: server-final tail, visible-final tail, update-gap tail; median threshold unmet |

## Memory and execution

RSS and Metal are separate per-process lifetime counters, never added together or described as combined process-tree RAM. Ranges are min-max session peaks in GiB (available sessions / recorded). Counters use one final summary per session; absent primary counters remain unavailable. STT calls and emitted previews do not measure total decoded audio; suppressed work can be absent from traces.

| Experiment / model / cohort (runs) | Pipeline RSS GiB | Metal GiB | Partial STT / final STT / emitted previews; reuse/drop counters |
|---|---|---|---|
| allocation_1024 / E2B / 4d25de1e (3) | 3.16-3.55 (3/3) | 6.42-6.62 (3/3) | 222/21/203 |
| allocation_1024 / E4B / 54f77975 (3) | 2.91-3.27 (3/3) | 8.34-8.53 (3/3) | 194/21/176 |
| allocation_512 / E2B / 4d25de1e (3) | 3.17-3.53 (3/3) | 6.29-6.47 (3/3) | 223/21/204 |
| allocation_512 / E4B / 54f77975 (3) | 2.95-3.75 (3/3) | 8.30-8.40 (3/3) | 199/21/182 |
| async_captions / E2B / 4d25de1e (3) | 3.33-3.43 (3/3) | 6.24-6.57 (3/3) | 219/21/201 |
| async_captions / E4B / 54f77975 (3) | 2.65-3.18 (3/3) | 8.33-8.65 (3/3) | 200/21/178 |
| baseline / E2B / 4d25de1e (3) | 3.23-3.49 (3/3) | 6.21-6.44 (3/3) | 222/21/204 |
| baseline / E4B / 54f77975 (3) | 1.85-3.38 (3/3) | 8.24-8.51 (3/3) | 209/21/192 |
| baseline_anchor / E2B / 4d25de1e (3) | 3.25-3.41 (3/3) | 6.42-6.64 (3/3) | 225/21/207 |
| baseline_anchor / E4B / 54f77975 (3) | 3.03-3.30 (3/3) | 8.29-8.49 (3/3) | 196/21/177 |
| clause_preview / E2B / 4d25de1e (3) | 3.15-3.45 (3/3) | 6.32-6.49 (3/3) | 223/21/208 |
| clause_preview / E4B / 54f77975 (3) | 2.61-3.69 (3/3) | 8.47-8.55 (3/3) | 204/21/189 |
| first_preview / E2B / 4d25de1e (3) | 3.06-3.35 (3/3) | 6.27-6.53 (3/3) | 226/21/207 |
| first_preview / E4B / 54f77975 (3) | 2.59-3.46 (3/3) | 8.40-8.50 (3/3) | 211/21/192 |
| latest_partial / E2B / 4d25de1e (3) | 3.20-3.31 (3/3) | 6.22-6.26 (3/3) | 225/21/204 |
| latest_partial / E4B / 54f77975 (3) | 3.04-3.60 (3/3) | 8.21-8.35 (3/3) | 214/21/192 |
| marian_memo / E2B / 4d25de1e (3) | 3.21-3.38 (3/3) | 6.18-6.47 (3/3) | 222/21/204; marian_memo_hit=6 |
| marian_memo / E4B / 54f77975 (3) | 2.71-2.86 (3/3) | 8.35-8.83 (3/3) | 205/21/185; marian_memo_hit=5 |
| pause_preview / E2B / 4d25de1e (3) | 3.16-3.45 (3/3) | 6.45-6.47 (3/3) | 222/21/207 |
| pause_preview / E4B / 54f77975 (3) | 2.99-3.92 (3/3) | 8.43-8.53 (3/3) | 204/21/189 |
| pause_speculation / E2B / 4d25de1e (3) | 2.43-3.37 (3/3) | 6.25-6.41 (3/3) | 243/21/204; speculation_completed=3, speculation_discarded_before_translate=11, speculation_discarded_resumed_speech=4, speculation_reused=3, speculation_started=18, speculation_suppressed_busy=3 |
| pause_speculation / E4B / 54f77975 (3) | 2.66-3.83 (3/3) | 8.36-8.42 (3/3) | 224/21/188; speculation_completed=3, speculation_discarded_before_translate=7, speculation_discarded_resumed_speech=4, speculation_reused=3, speculation_started=14, speculation_suppressed_busy=7 |
| prefix_cache / E2B / 4d25de1e (3) | 2.95-3.35 (3/3) | 6.33-6.54 (3/3) | 220/21/201 |
| prefix_cache / E4B / 54f77975 (3) | 2.23-3.27 (3/3) | 8.31-8.76 (3/3) | 201/21/183 |
| rolling_6s / E2B / 4d25de1e (3) | 3.08-3.22 (3/3) | 6.14-6.22 (3/3) | 225/21/204 |
| rolling_6s / E4B / 54f77975 (3) | 3.09-3.97 (3/3) | 8.24-8.31 (3/3) | 217/21/197 |
| streaming_stt / E2B / 4d25de1e (3) | 2.93-3.37 (3/3) | 6.69-6.80 (3/3) | 224/21/206 |
| streaming_stt / E4B / 54f77975 (3) | 2.69-3.03 (3/3) | 8.61-8.66 (3/3) | 219/21/201 |
| vad_worker_onnx / E2B / 4d25de1e (3) | 2.85-3.46 (3/3) | 6.29-6.33 (3/3) | 224/21/205 |
| vad_worker_onnx / E4B / 54f77975 (3) | 2.47-3.63 (3/3) | 8.34-8.82 (3/3) | 198/21/181 |
| vad_worker_torch / E2B / 4d25de1e (3) | 3.03-3.25 (3/3) | 6.28-6.33 (3/3) | 224/21/206 |
| vad_worker_torch / E4B / 54f77975 (3) | 2.81-3.02 (3/3) | 8.42-8.60 (3/3) | 205/21/184 |

## Opening-to-closing control drift

Each anchor pair is shown once. Positive paired Δ means the closing control was slower; this is observed drift, not an experiment gain.

| Model / cohort / opening session | Silence-final Δ p50 ms (n) | First-server-preview Δ p50 ms (n) |
|---|---|---|
| E2B / 4d25de1e / overnight_screen_baseline_e2b_r0_screening_45s_en | -22 (1) | +6 (7) |
| E2B / 4d25de1e / overnight_screen_baseline_e2b_r1_screening_45s_en | +70 (1) | -4 (7) |
| E2B / 4d25de1e / overnight_screen_baseline_e2b_r2_screening_45s_en | -47 (1) | +4 (7) |
| E4B / 54f77975 / overnight_screen_baseline_e4b_r0_screening_45s_en | -1,413 (1) | +60 (7) |
| E4B / 54f77975 / overnight_screen_baseline_e4b_r1_screening_45s_en | +787 (1) | +160 (7) |
| E4B / 54f77975 / overnight_screen_baseline_e4b_r2_screening_45s_en | +56 (1) | +202 (7) |

## Actual changed-output examples

First changed matched segment in deterministic experiment/model/repeat order, capped for readability. These are unapproved predictions, not references or semantic-error judgments. Full texts and all changes remain in the linked analysis.

No changed matched-output examples are present in this analysis snapshot; this does not establish accuracy.

Actual within-session preview rewrites below are a different kind of change: normal provisional revisions, not changed matched finals or proved errors.

- overnight_screen_allocation_1024_e2b_r0_screening_45s_en, utterance 1:
  First preview source: Yeah.
  Last preview source: And then also I noticed too that um that um when it comes to these two
  First preview translation: Sí.
  Last preview translation: Y también me di cuenta de que um um um cuando se trata de estos dos

- overnight_screen_allocation_1024_e4b_r0_screening_45s_en, utterance 2:
  First preview source: that um when it comes to these two criminals
  Last preview source: that um when it comes to these two criminals, their their life started out uh at least up until this moment. There
  First preview translation: que um cuando se trata de estos dos criminales
  Last preview translation: que cuando se trata de estos dos criminales, su vida comenzó al menos hasta este momento.

## Provenance and limits

- Endpoint classification contract v1: `/Users/willem/Code/vibes/SRTranslate/tools/overnight_analysis.py` (SHA-256 `5bf246c2111590d74ecaa6c672236a5904bb5f6bbea85bc77a1841168d1e1f25`); raw labels/measurements are unchanged.
- Analysis SHA-256: `0ce884b43839ce4e9a01dbb0dba7e3211906338f049439f572c2acfb24dc0442`; frozen manifest SHA-256: `39490b15768ed0eec048e0701a0792c2ac4a2faaa794f7ecb2aa73d567bac168`.
- Source-code cohorts: f0999f802c52dd82a851e891ee2da78b4470f9ceb347a93948f4bef77901eccd.
- Missing sessions: 0; input failures: 0. Invalid sessions are excluded from aggregates and remain visible in full analysis.
- Locked source/audio/model identity is distinct from browser visibility. Synthetic ES, natural EN, different endpoints and different runtime/model cohorts remain separate.
- Source-window coverage is not acoustic speech recall; hymn/silence gaps require listening or annotations. No WER or reference-quality score is inferred.
- This renderer does not run inference, alter selection, update defaults or approve outputs.
