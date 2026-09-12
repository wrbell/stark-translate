# P1 — runtime overhead fixes: micro-bench and paired identity screen → **PASS** (PR #217)

**What was screened.** Four output-identical changes to the Mac inference path, implemented by Codex from a Claude
spec in `codex/series4-p1-runtime` (`e349589` + profiler fix `77b4998` + wired-limit flip `53804d4`, on main `2d01840`):

1. keep-warm scheduled after the final's translation completes instead of at submit time (`STARK_WARMUP_AFTER_FINAL=0` restores);
2. first `translation_stream` callback after the first token instead of the third (`STARK_STREAM_FIRST_TOKEN_BATCH_SIZE`, live default 1);
3. the hash-pinned Parakeet joint scalar decode installed at engine load, with fallback to the stock decode on any pin
   mismatch and restore on unload (`STARK_PARAKEET_JOINT_EVAL=0` disables) — the transform that was output-exact 18/18
   in the [September 10 receipt](../../mac_followup_20260910/profiling/joint-development-r1.json) and shelved on the arm gate;
4. Metal wired limit set once at load — **made opt-in during this screen** (`STARK_MLX_WIRED_LIMIT=1`), see below.

Policy (approved 2026-09-11): exact-output engineering changes merge on a paired identity screen, not on the 15 %/150 ms
experiment gate. Production defaults are unchanged (`settings.py`, `models.lock.json`, profiles, `pyproject.toml`).

## Micro-bench ([`microbench.md`](microbench.md), receipts in [`receipts/`](receipts/))

Same `venv` interpreter, control checkout `5227a73` vs the branch: seven `benchmark_mlx_accel` runs produced byte-identical
text for all 21 sentences; the branch sits at −0.3 % of the warm controls and the wired limit is neutral on or off
(mlx-lm already sets the same limit inside every generation call). Parakeet joint decode: 18/18 FLEURS pairs exact,
−17.6 ms (−9.6 %) p50 per call on the promoted runtime. The Parakeet readback profiler needed a fix to keep
instrumenting the stock decode once the engine installs the joint method (`77b4998`).

## Screen 1 (aborted) → wired limit rejected ([`screen1/`](screen1/))

Protocol `p1_identity_screen_series4_20260912` (`screen1/protocol.json`, declared before run 1): control = worktree at
`5227a73`, candidate = the branch with all four changes on, 3 alternating pairs per 360 s clip. After clip A (3/3
pairs byte-identical, Gemma-routed silence first-token p50 −9.9 %) `tools/tail_screen_report.py` G6 failed on
**peak process RSS 7.8–8.0 GB vs 4.1 GB control** (Metal peak unchanged at 9.0 GB): wiring the recommended working set
at load keeps every buffer resident between calls. With no speed effect in isolation, the change was made opt-in
(`53804d4`) and the screen stopped after one clip-B control (`screen1/report_interim.md`, `screen1/runs.jsonl`).

## Screen r2 → PASS ([`screen/`](screen/))

Protocol `p1_identity_screen_series4_20260912_r2` (`screen/protocol2.json`, declared before its run 1; tags
`p1s0912b_*`): same design, candidate = `53804d4`. 12/12 runs rc 0, 2026-09-12T00:50Z → 02:06Z, one process at a time,
`venv/bin/python` for both arms, trace on, HF offline. Report: [`screen/report.md`](screen/report.md)
(`tools/tail_screen_report.py`, 63 input files hashed), per-run metrics [`screen/metrics.json`](screen/metrics.json)
(`screen_metrics.py`), identity guard [`screen/identity.json`](screen/identity.json) (`identity_guard.py`).

| gate (declared) | clip A | clip B | result |
|---|---|---|---|
| G_identity: byte-identical finals, sequence-aligned, all pairs | 61/61 EN and ES in 3/3 pairs; 0 unmatched | 80/80 in 3/3 pairs; 0 unmatched | **PASS** (tool G4 identical share 1.0) |
| G_tail: silence p95 not worse than max(5 %, 100 ms) | 3365 vs 3234 ms (+131 ms, +4.1 %; allowance 162 ms); Gemma-silence 3393 vs 3273 (+120 ms, +3.7 %; allowance 164 ms) | 2527 vs 2658 (−4.9 %); Gemma-silence 2696 vs 2875 (−6.2 %) | **PASS** (tool G2 pass, cuts p95 −5 % / +0.5 %) |
| G_preview: previews within 2 pp | 486 vs 450 per run (+8 %) | 435 vs 429 (+1.5 %) | **PASS** (tool G5) |
| G_memory: Metal within 1 GiB; RSS within the tool's relative rule | Metal 8.56–8.63 vs 8.56 GiB; RSS 1.9–3.4 GiB (limit 3.95) | RSS and Metal within limits | **PASS** (tool G6) |
| tool G7 (returncode, lifecycle, chunk count) | 61 finals every run | 80 finals every run | PASS |
| tool G1 (15 % / 300 ms Gemma-silence p95 median gate) | FAIL | FAIL | not a criterion (identity screen); reported |

Reported, not gated (pooled per clip, Gemma-routed silence finals, n = 45 / 129):

| clip | first-token p50 / p95 (ms) ctl → cand | payload-ready p50 / p95 ctl → cand | final STT p50 ctl → cand | generation-lock wait p95 | all finals ready p50 / p95 |
|---|---|---|---|---|---|
| A | 1417 / 2824 → 1264 / 2573 (−10.8 % / −8.9 %) | 2246 / 3273 → 2046 / 3393 (−8.9 % / +3.7 %) | 526 → 405 ms (−23 %) | 96 → 100 ms | 2968 / 6946 → 2583 / 6008 (−13 % / −13.5 %) |
| B | 1223 / 2268 → 1131 / 2024 (−7.5 % / −10.7 %) | 1741 / 2875 → 1699 / 2696 (−2.4 % / −6.2 %) | 426 → 370 ms (−13 %) | 90 → 94 ms | 1650 / 4706 → 1590 / 4372 (−3.6 % / −7.1 %) |

The gain is where the changes act: the final STT call (joint decode) and the first streamed token. The
generation-lock wait p95 did not move, so the keep-warm reordering is not visible at this sample size; it is kept as
an output-identical simplification, not a measured win. The first-token change is not visible in a headless replay
(no display client); its browser-side effect is what P6 will measure.

**Decision.** PASS on every declared gate on both clips → auto-merge enabled on #217 (2026-09-12T02:12Z).
Recorded in `docs/latency_next_experiments.md` as an engineering change, not an experiment arm. This screen is a
machine-timed replay of two sermons with 12 runs; it supports no p95 claim (`p95_claim_eligible: false`), no
human-quality or visible-display statement, and it does not restate the caption-delivery goal.

Screen-1 evidence stays as the record of the wired-limit rejection: the run tags `p1s0912_*` and `p1s0912b_*` are
distinct in both checkouts' `metrics/` directories.
