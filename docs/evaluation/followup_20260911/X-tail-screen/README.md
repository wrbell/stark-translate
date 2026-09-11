# X — declared p95-tail screen `tail_screen_20260911`

**Outcome: both arms REJECTED on the pre-declared gates.** `draft_g3` (E2B OptiQ as mlx-lm draft for E4B finals,
γ=3) fails G1–G6 on both clips; `serial_finals` (no overlap of a final's STT with the previous final's
translation) fails G1 on both clips and G2–G5 on clip A. Neither result changes a production default, and no
combination arm follows. Eligible Gemma-routed silence finals: 173–174 per arm (45 on clip A, 128–129 on clip B);
per the tool's per-clip rule (≥ 50 on each clip) this remains a **screen without p95 claim**.

This is a server-timed replay screen on two 360 s church clips in the promoted Torch 2.13 environment; it supports
no visible-ACK, human-quality or production-default claim. Declaration: [`protocol.json`](protocol.json)
(SHA256 `1406cc78379c8775…`, written 15:45Z before any run; one amendment at 16:10Z, before any candidate result was read —
see below). Report tool: `tools/tail_screen_report.py` (PR #209); raw per-run replay summaries under [`runs/`](runs/),
tool outputs [`report.md`](report.md) / [`report.json`](report.json) (declared rules) and
[`report_amended.md`](report_amended.md) (pageout sub-rule removed, see amendment).

## Setup

- Source `main` @ `6f160bb` (harness PR #209 merged), interpreter `venv/bin/python` (Torch 2.13.0 / TorchAudio 2.11.0,
  promoted earlier today), production defaults (`--profile standard`, Parakeet EN, E4B OptiQ finals, Marian CT2 previews,
  0.5 s silence, 0.6 s cadence, final-aware partials on), `--no-tts`, `STARK_EXPERIMENT_TRACE=true` on every run.
- Clips (360 s, cut with `tools/replay_bench.py --prepare`): **A** = `Gospel_Message_(12_14_25)` @ 1290 s
  (SHA256 `d55d4db2edc85acd…`, Silero speech coverage 0.895); **B** = `Gospel_Message_(2_8_26)` @ 1170 s
  (`fc2eccd7a35459ba…`, coverage 0.760 — the best 360 s window in that sermon is 0.777, so the declared floor
  was 0.75). No music-like gaps; no music hold occurred.
- Arms: `ctl` (no experiment env); `draft_g3` = `STARK_EXPERIMENT_DRAFT_MODEL_ID=mlx-community/gemma-4-e2b-it-OptiQ-4bit`,
  `STARK_EXPERIMENT_DRAFT_TOKENS=3`; `serial_finals` = `STARK_EXPERIMENT_SERIAL_FINALS=true`. Order per clip
  `ctl, draft, serial | serial, draft, ctl | ctl, serial, draft`; 18 runs, all rc 0 (`runs.jsonl`, `screen.log`).
- Amendment (16:10Z): the draft arm's abort/G6 sub-rule "vm_stat pageouts grew during the run" was replaced by
  "growth > max(5× the clip's median control-run growth, 20 000 pages)" because the counter is machine-wide and the
  first plain control run already grew by 1 877 pages. The in-progress first draft run was discarded and the screen
  resumed with the same order. Peak-Metal budget (15.5 GiB) unchanged. Under the declared literal rule G6 fails for
  the draft arm (`report.md`); under the amended rule it passes on memory (`report_amended.md`: peak Metal 12.4–12.8 GiB,
  pageout growth 1.5–3.9 k vs control 1.0–1.9 k). The outcome is REJECTED either way because of G1–G5.

## Results (pooled over 3 repeats per clip; nearest-rank; ms)

| clip | arm | Gemma / Marian silence n | Gemma silence p50 / p95 | Marian silence p50 / p95 | smart+hard cuts p50 / p95 | `stt_call` p50 / p95 | `stt_dispatch` p95 | Gemma `translation_call` p50 / p95 | `decode_ms_a` p95 | tok/s p50 | draft accept |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A | ctl | 45 / 18 | 1970.5 / 2424.6 | 804.0 / 1666.9 | 2595.5 / 5602.1 | 401.8 / 511.4 | 7.3 | 1024.9 / 1446.0 | 1121.9 | 31.4 | — |
| A | draft_g3 | 45 / 18 | 2761.7 / 6641.3 | 869.4 / 4973.8 | 4437.9 / 7741.0 | 343.7 / 2487.0 | 3265.7 | 952.9 / 2524.1 | 931.5 | 39.3 | 0.72 |
| A | serial_finals | 45 / 17 | 2025.0 / 2812.0 | 804.7 / 2694.3 | 2790.4 / 6506.5 | 417.4 / 517.7 | 975.9 | 1023.0 / 1441.2 | 1132.8 | 31.5 | — |
| B | ctl | 129 / 75 | 1618.7 / 1914.5 | 868.8 / 984.0 | 3233.2 / 6107.7 | 367.5 / 476.5 | 3.6 | 753.5 / 976.3 | 695.7 | 35.2 | — |
| B | draft_g3 | 128 / 75 | 2249.0 / 7134.5 | 1036.9 / 7469.6 | 5682.7 / 9849.2 | 433.1 / 2982.2 | 3036.1 | 697.1 / 1125.9 | 602.4 | 48.1 | 0.759 |
| B | serial_finals | 129 / 75 | 1638.2 / 2234.0 | 882.0 / 964.9 | 3296.6 / 6105.4 | 373.6 / 465.8 | 3.7 | 771.3 / 1000.6 | 717.1 | 34.2 | — |

Gate table per clip (all-route silence p50/p95 and the tool's G1–G7 verdicts) is in `report.md`; summary:
clip A ctl silence p50/p95 1 619.5 / 2 379.6 → draft 2 242.0 / 6 544.5, serial 1 703.0 / 2 694.3;
clip B ctl 1 478.2 / 1 876.4 → draft 1 817.6 / 7 134.5, serial 1 504.7 / 1 918.3.

Per-run health and memory (`chunk_count`, previews, peak Metal / RSS GiB, pageout growth, `final_stt_waited_for_translation`):

| run | finals | previews | Metal GiB | RSS GiB | pageouts Δ | waited |
|---|---:|---:|---:|---:|---:|---:|
| ts0911_A_ctl_r0 | 61 | 517 | 8.52 | 4.06 | 1877 | 0 |
| ts0911_A_draft_g3_r0 | 59 | 362 | 12.74 | 3.83 | 2650 | 0 |
| ts0911_A_serial_finals_r0 | 61 | 514 | 8.61 | 5.2 | 2201 | 4 |
| ts0911_A_serial_finals_r1 | 61 | 502 | 8.71 | 4.0 | 6061 | 3 |
| ts0911_A_draft_g3_r1 | 60 | 332 | 12.5 | 3.76 | 3570 | 0 |
| ts0911_A_ctl_r1 | 61 | 518 | 8.49 | 4.8 | 1525 | 0 |
| ts0911_A_ctl_r2 | 61 | 518 | 8.58 | 4.14 | 1241 | 0 |
| ts0911_A_serial_finals_r2 | 60 | 517 | 8.54 | 4.0 | 382 | 3 |
| ts0911_A_draft_g3_r2 | 61 | 384 | 12.44 | 3.78 | 7590 | 0 |
| ts0911_B_ctl_r0 | 80 | 447 | 8.48 | 5.22 | 1730 | 0 |
| ts0911_B_draft_g3_r0 | 80 | 377 | 12.64 | 4.07 | 2050 | 0 |
| ts0911_B_serial_finals_r0 | 80 | 449 | 8.64 | 5.4 | 936 | 1 |
| ts0911_B_serial_finals_r1 | 80 | 447 | 8.56 | 4.16 | 1920 | 1 |
| ts0911_B_draft_g3_r1 | 75 | 271 | 12.81 | 4.1 | 4252 | 0 |
| ts0911_B_ctl_r1 | 80 | 447 | 8.41 | 5.49 | 1074 | 0 |
| ts0911_B_ctl_r2 | 80 | 450 | 8.51 | 4.33 | 987 | 0 |
| ts0911_B_serial_finals_r2 | 80 | 446 | 8.54 | 3.35 | 2698 | 1 |
| ts0911_B_draft_g3_r2 | 80 | 295 | 12.47 | 3.92 | 3937 | 0 |

Text identity (the tool's G4 aligns finals by row order, so a single extra or missing chunk shifts every later row —
its 0.19 share on some pairs is an alignment artifact; the sequence-aligned view below is the honest number):

| candidate run | finals cand / ctl | English matched (share) | Spanish identical among matched (share) |
|---|---|---|---|
| ts0911_A_draft_g3_r0 | 59 / 61 | 59 (0.967) | 53/59 (0.898) |
| ts0911_A_serial_finals_r0 | 61 / 61 | 61 (1.0) | 61/61 (1.0) |
| ts0911_A_serial_finals_r1 | 61 / 61 | 61 (1.0) | 61/61 (1.0) |
| ts0911_A_draft_g3_r1 | 60 / 61 | 60 (0.984) | 54/60 (0.9) |
| ts0911_A_serial_finals_r2 | 60 / 61 | 60 (0.984) | 60/60 (1.0) |
| ts0911_A_draft_g3_r2 | 61 / 61 | 61 (1.0) | 55/61 (0.902) |
| ts0911_B_draft_g3_r0 | 80 / 80 | 80 (1.0) | 78/80 (0.975) |
| ts0911_B_serial_finals_r0 | 80 / 80 | 80 (1.0) | 80/80 (1.0) |
| ts0911_B_serial_finals_r1 | 80 / 80 | 80 (1.0) | 80/80 (1.0) |
| ts0911_B_draft_g3_r1 | 75 / 80 | 75 (0.938) | 73/75 (0.973) |
| ts0911_B_serial_finals_r2 | 80 / 80 | 80 (1.0) | 80/80 (1.0) |
| ts0911_B_draft_g3_r2 | 80 / 80 | 80 (1.0) | 78/80 (0.975) |

## Reading

- **`draft_g3` — REJECTED, with mechanism.** As a decoder the draft does what the text bench promised: Gemma
  `translation_call` p50 fell 7 % (clip A 1 025 → 953 ms; B 754 → 697), `decode_ms_a` p95 fell (1 122 → 932; 696 → 602),
  throughput rose from 31–35 to 39–48 tok/s at 0.72–0.76 acceptance. But with E4B + E2B + Parakeet resident
  (12.4–12.8 GiB Metal, ~16.5 GB with RSS on an 18 GB machine) the **STT side collapsed**: `stt_call` p95 511 → 2 487 ms
  (A) and 477 → 2 982 ms (B), and the final STT **dispatch wait** p95 went from 4–7 ms to 3.0–3.3 s — finals queued
  behind busy pool workers. Preview counts dropped 27–39 %, every Gemma-routed p95 doubled or tripled, and the
  Spanish text of ~10 % of matched finals differed from control on clip A (2.5 % on B; English identical) — the live
  draft path (non-streaming `translate_mlx`) is not byte-identical to the streaming production path even under greedy
  decoding. The hypothesis "the draft shortens the translation tail" is true in isolation and false in the live
  pipeline on this memory budget; it is closed unless a much smaller draft (or a text-only budget) is available.
- **`serial_finals` — REJECTED.** The overlap it removes is rare on these clips (1–4 waits per 60–80 finals) and
  removing it only adds wait: the `stt_dispatch` p95 on clip A became 976 ms and the Gemma silence p95 rose 16 %
  (2 425 → 2 812) with no improvement on clip B (1 915 → 2 234). The deliberate STT/translation overlap stays.
- **Control tails on the promoted environment are translation-dominated**: in the worst decile of silence finals,
  `translation_call` > 800 ms in 95–100 % of finals and `stt_call` > 800 ms in only 14 %; Gemma-routed finals generate
  19 (A) / 14 (B) tokens at the median and 35 / 22 at p95, at 31–35 tok/s. The joint STT+translation tail seen in the
  overnight endurance cohort did not reproduce here. Marian-routed silence finals are already sub-second at the median
  (804–869 ms) but their p95 (985–1 667 ms) is the `translation_lock_wait` behind a Gemma final (510 ms p95 on A).
- The `physical_stt` trace now carries `translation_active` / `concurrent_partial` / `concurrent_final` on every run
  (`STARK_EXPERIMENT_TRACE=true`), so the next hypothesis can be chosen from measured overlap rather than inferred.

## What this does and does not establish

Rejected arms are not re-run as confirmations and do not enter combinations. Nothing here certifies human quality,
visible-browser delivery or a production default. `caption-delivery-goal` stays open: the sub-second median is a
Gemma-route problem (its median is ~1.6–2.0 s here) and neither a draft on this memory budget nor serializing finals
moves it.
