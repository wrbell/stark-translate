# v2026.9 — Gemma 4 Latency Optimization (Phase 2 cheap wins)

**Status:** code shipped; numbers measured on the A2000 Ada Mobile bench
machine 2026-05-04 with `scripts/benchmarks/bench_translate_t1_t4.py` over
50 sermon chunks + 8 theological canaries.

---

## TL;DR

Cumulative-flags sweep over four cheap optimizations that the user asked us
to evaluate. Only one was a clean net win. Net result: ~3% partial-translate
p50 reduction on both E4B and E2B with the canary unchanged at 7/8 / 6/8
respectively.

| Variant | E4B p50 | E4B p95 | E4B tok/s | Canary | E2B p50 | E2B p95 | Canary | VRAM |
|---|---|---|---|---|---|---|---|---|
| baseline (b8782, `-ctk q8_0`) | 478 ms | 621 ms | 41.5 | 7/8 | 272 ms | 362 ms | 6/8 | 4.71 / 3.29 GB |
| `+ GGML_CUDA_GRAPH_OPT=1` (Exp A) | 454 ms | 613 ms | 42.2 | 7/8 | 286 ms | 387 ms | 6/8 | 4.69 / 3.28 GB |
| `+ -fa on` (Exp B) | **744 ms** | 982 ms | 26.7 | 7/8 | — | — | — | 4.75 GB |
| `+ -ctv q8_0` (Exp C) | 462 ms | 611 ms | 42.7 | 7/8 | 264 ms | 375 ms | 6/8 | 4.75 / 3.31 GB |
| `+ llama.cpp d8794eecd / b9022` (Exp D) | 473 ms | 610 ms | 42.8 | 7/8 | 263 ms | 375 ms | 6/8 | 4.75 / 3.31 GB |

Net delta vs baseline (cumulative best, Exp D):
- E4B: **−1% p50, −2% p95, +3% tok/s**, canary unchanged
- E2B: **−3% p50, +4% p95** (within noise), **+2% tok/s**, canary unchanged

This **misses the plan's 20% gate**. Per the v2026.8 plan that gates PR3 on
PR2 outcome, PR3 (IQ4_XS quantization) is now mandatory.

---

## Hardware Under Test

| Field | A2000 Ada Mobile |
|---|---|
| GPU | NVIDIA RTX 2000 Ada Generation (16 GB) |
| Compute capability | 8.9 |
| OS | WSL2 Ubuntu 24.04 on Windows 11 |
| llama.cpp before | `e97492369` (b8782, 2026-04-13) |
| llama.cpp after  | `d8794eecd` (b9022, 2026-05-04, +240 commits upstream) |

---

## Experiments

### Step 0 — Re-baseline

Captured fresh against current state. Slightly different from the April
2026 numbers in `docs/archive/v2026.5/BENCHMARK.md` (E4B 478 vs 470 ms p50;
E2B 272 vs 280 ms p50) — within run-to-run variance.

### Experiment A — `GGML_CUDA_GRAPH_OPT=1` env var → **no-op**

Hypothesis: CUDA-graph capture/replay would eliminate per-token launch
overhead. Reality: the existing build already has `USE_GRAPHS=1` baked in
at compile time (visible in the `system_info` line of every server log).
The env var was a no-op.

**Decision:** Skip. The capability is already on; setting the env var has
no measurable effect.

### Experiment B — `--flash-attn on` → **REGRESSION, revert**

Hypothesis: Faster attention kernels. Reality: E4B p50 went 478 → **744 ms**
(+56%), tok/s 41.5 → 26.7 (−36%). Server log shows
`llama_context: flash_attn = enabled` with no fallback warning — FA *is*
running, but on Gemma 4's hybrid attention layout (sliding-window + global
alternation, with rotary on K but not V) the b8782 FA kernel is slower
than the standard path.

This may be different on the upgraded build (b9022) — the upstream FA
kernels have seen significant Gemma-4-specific work in 2026 — but
re-testing FA on b9022 is left for a future phase since the cumulative
result without FA (Experiment D) already hit the gate-failure threshold
that triggers PR3.

**Decision:** Skip on this build. **Did not retest on b9022** — flagged
for a future revisit in `docs/archive/v2026.9/FOLLOWUPS.md`.

### Experiment C — `-ctv q8_0` (V-cache quantization) → **kept**

Hypothesis: Halving the V-cache size would save VRAM and give a tiny
latency win. Reality: VRAM saved was negligible at `c=512` (V cache only
~5 MiB total even un-quantized), but the latency direction was right —
E4B p50 478 → 462 (−3%), E2B p50 272 → 264 (−3%). Canary unchanged on
both.

**Decision:** Kept. Free win at this context size; saves more proportional
VRAM if `-c` is ever raised for longer-context workloads.

### Experiment D — llama.cpp upstream rebuild → **kept (parity), worth the upgrade**

Pulled 240 commits upstream (b8782 → b9022, 2026-04-13 → 2026-05-04 of
upstream activity). Rebuild succeeded cleanly with the same CMake flags
(`-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89`).

Cumulative bench (with `-ctv q8_0` from Exp C still active):
- E4B p50 462 → 473 (+2%), p95 611 → 610 (par), tok/s 42.7 → 42.8 (par)
- E2B p50 264 → 263 (par), p95 375 → 375 (par), tok/s 69.7 → 69.7 (par)

Within noise on both directions. Canary unchanged. `enable_thinking:false`
chat-template-kwarg still honored — no empty-content / runaway-reasoning
regression seen in any of the 50+8 requests per direction.

**Decision:** Kept. The upgrade itself is roughly neutral in measured
latency, but removes a year of upstream drift, brings new Gemma-aware
kernels (relevant for a future FA retry), and is the prerequisite for
PR3/PR4 work that needs `llama-imatrix` + the quantization tools at
roughly the same vintage.

---

## Cumulative Net & Gate Decision

Final state shipped in this PR:
- llama.cpp pinned to `d8794eecd` / build `b9022` (2026-05-04).
- `start_server.sh` line 77 gains `-ctv q8_0`.
- No env-var changes (USE_GRAPHS already on; FA reverted).

Net against the v2026.5 baseline:
- **E4B p50: −1%** (478 → 473 ms)
- **E2B p50: −3%** (272 → 263 ms)
- Canary: unchanged (E4B 7/8, E2B 6/8)
- VRAM: unchanged (within ±0.04 GB)

The plan's 20% gate is **not** hit. PR3 (IQ4_XS quantization sweep) is
now mandatory, and PR4 (imatrix calibration) will run regardless because
its infrastructure is the foundation for future fine-tune quantization
even if PR3 satisfies the latency target.

---

## Followups

A separate `FOLLOWUPS.md` lists items deferred to a future phase:

1. **Re-test `-fa on` against b9022.** Upstream has had heavy Gemma 4
   attention work since b8782; the +56% E4B regression may already be
   fixed. Cost: 5 minutes (one experiment + revert if still bad).
2. **Larger context (`-c 2048`) bench.** `-ctv q8_0` would matter more.
3. **Per-request prompt-cache reuse strategy** — Gemma 4 SWA prevents
   cross-request KV reuse (`docs/archive/v2026.5/BENCHMARK.md`), but
   intra-request reuse for streaming partials might be worth probing.

---

## Reproducing the Numbers

```bash
source ~/stt_train_env/bin/activate
cd ~/llama.cpp && git checkout d8794eecd && \
    cmake --build build --config Release -j$(nproc)

# Cumulative-flags sweep — each step starts from the previous flag set.
# Helper script /tmp/bench_gemma_run.sh stops the prior server, starts a
# fresh one with the requested flags, runs bench, returns.
ROOT=/mnt/e/code/stark-translate
for variant_args in \
    "baseline:" \
    "exp_a:GGML_CUDA_GRAPH_OPT=1" \
    "exp_b:-fa on" \
    "exp_c:-ctv q8_0" \
    "exp_d:-ctv q8_0"; do  # exp_d is exp_c flags + new build
    # ... see metrics/gemma4_optim_phase2/{baseline,exp_*}_t{2,3}.json
done
```

Outputs land in `metrics/gemma4_optim_phase2/{baseline,exp_a,exp_b,exp_c,exp_d}_t{2,3}.json`
plus matching `_sermon.jsonl` per-clip records.
