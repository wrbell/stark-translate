# v2026.10 — IQ4_XS Quantization Sweep (Phase 2 escalation)

**Status:** code shipped (CONFIGS extension, GGUF artifacts produced locally);
**default model unchanged**. Numbers measured on the A2000 Ada Mobile bench
machine 2026-05-04 with the v2026.9 cumulative flag set (`-ctv q8_0`,
llama.cpp `d8794eecd`/`b9022`).

---

## TL;DR

PR2 missed the 20% latency-improvement gate, so per the v2026.8 plan PR3
(IQ4_XS quantization) was mandatory. We re-quantized E2B and E4B from the
HF bf16 base via `llama-quantize ... IQ4_XS`, hosted alongside Q4_K_M (do
not replace), and benched against PR2's cumulative-best flag set.

| Variant | p50 | p95 | tok/s | Canary | VRAM | Δ p50 vs Q4_K_M |
|---|---|---|---|---|---|---|
| **E4B Q4_K_M (PR2 default)** | 478 ms | 621 ms | 41.5 | 7/8 | 4.71 GB | — |
| E4B IQ4_XS | 434 ms | 559 ms | 45.1 | **5/8** ❌ | 4.55 GB | **−9%** |
| **E2B Q4_K_M (PR2 fallback)** | 272 ms | 362 ms | 68.4 | 6/8 | 3.29 GB | — |
| E2B IQ4_XS | 261 ms | 338 ms | 74.4 | 6/8 ✓ | 3.23 GB | **−4%** |

**Verdict:**
- **E4B IQ4_XS does NOT ship as default.** Latency is real (−9% p50), but
  the canary regressed from 7/8 to **5/8** — three theological terms lost
  vs Q4_K_M. This trips the plan's quality gate (`canary E4B ≥ 7/8 with
  no items lost vs Q4_K_M baseline`). Q4_K_M remains production default.
- **E2B IQ4_XS is acceptable as the low-VRAM alternative.** −4% p50 with
  no canary loss, both versions stuck at 6/8 (E2B never had Jacobo/Santiago
  resolution).

The E4B canary regression is exactly the failure mode PR4's imatrix
calibration is designed to address — calibration on a sermon corpus with
canary sentences over-weighted. PR4 follows directly.

---

## Hardware Under Test

Same as v2026.9 (A2000 Ada Mobile, 16 GB, b9022 llama.cpp). Server flags
inherited from PR2: `-ngl 999 -c 512 -ctk q8_0 -ctv q8_0`.

---

## E4B Canary Failures (the regression)

| # | Source EN | Expected ES term | Q4_K_M output | IQ4_XS output | IQ4_XS hit? |
|---|---|---|---|---|---|
| 1 | atonement | expiación | "...la expiación..." | "...la expiación..." | ✓ |
| 2 | covenant | **pacto** | "...un pacto con..." | "...una **alianza** con..." | ✗ |
| 3 | grace | gracia | "...solo por gracia..." | "...solo por gracia..." | ✓ |
| 4 | righteousness | justicia | "...la justicia..." | "...la justicia..." | ✓ |
| 5 | James (epistle) | Santiago | "...libro de Santiago..." | "...libro de Santiago..." | ✓ |
| 6 | James (apostle) | **Jacobo** | "Jacobo y Juan..." | "**James** and John..." | ✗ |
| 7 | sanctification | santificación | "...la santificación..." | "...la santificación..." | ✓ |
| 8 | breaking of bread | **partimiento del pan** | "...la **ruptura** del pan..." | "...la **fracción** del pan..." | ✗ |

Item 6 is particularly concerning — IQ4_XS leaves "James and John" in
English, untranslated. The Jacobo/Santiago disambiguation that motivated
choosing E4B in the first place is the canonical example of what the
church demo cares about; losing it is a P1 quality regression.

Item 2 (`pacto` → `alianza`) is a Catholic-vs-Protestant translation
shift, not strictly wrong but loses the Brethren-tradition vocabulary.

Item 8 (`partimiento` → `fracción`) was already regressed under Q4_K_M
(`ruptura`), so this is sideways drift, not a new regression.

---

## E2B Canary (no regression)

E2B Q4_K_M was already 6/8 (it never had Jacobo/Santiago resolution; that's
why E4B exists as the production default). E2B IQ4_XS preserves the same
6/8 score on the same 6 items — no quality movement.

---

## Files

- `models/gemma-4-e2b-it-iq4xs.gguf` (3.08 GB, sha256 `dd94faae…1041befd`)
- `models/gemma-4-e4b-it-iq4xs.gguf` (4.74 GB, sha256 `656fa4ee…b141a3b6`)
- Both built locally via `llama-quantize bf16.gguf out.gguf IQ4_XS` with
  no imatrix. **Not yet hosted on HF**; reproducible via the recipe in
  the "Reproducing the Numbers" section below.

`scripts/benchmarks/bench_translate_t1_t4.py` gains two new CONFIGS:

- `t2-iq4xs` — E2B IQ4_XS, no draft
- `t3-iq4xs` — E4B IQ4_XS, no draft

These are useful for PR4's imatrix bench (we'll re-quantize the same
artifacts with `--imatrix` and compare against this baseline).

---

## Reproducing the Numbers

```bash
# 1. Convert HF bf16 -> GGUF bf16 (one-time, ~3 min E2B / ~5 min E4B)
source ~/stt_train_env/bin/activate
python ~/llama.cpp/convert_hf_to_gguf.py \
    /mnt/d/Data/stt-data/cache/hub/models--google--gemma-4-e2b-it/snapshots/<sha>/ \
    --outtype bf16 --outfile /tmp/gemma-4-e2b-it-bf16.gguf
python ~/llama.cpp/convert_hf_to_gguf.py \
    /mnt/d/Data/stt-data/cache/hub/models--google--gemma-4-e4b-it/snapshots/<sha>/ \
    --outtype bf16 --outfile /tmp/gemma-4-e4b-it-bf16.gguf

# 2. Quantize bf16 -> IQ4_XS (CPU; ~2 min E2B / ~3 min E4B)
~/llama.cpp/build/bin/llama-quantize \
    /tmp/gemma-4-e2b-it-bf16.gguf models/gemma-4-e2b-it-iq4xs.gguf IQ4_XS
~/llama.cpp/build/bin/llama-quantize \
    /tmp/gemma-4-e4b-it-bf16.gguf models/gemma-4-e4b-it-iq4xs.gguf IQ4_XS

# 3. Bench
ROOT=/mnt/e/code/stark-translate
for cfg in t3-iq4xs t2-iq4xs; do
    /tmp/bench_gemma_run.sh exp_iq4xs $cfg \
        $ROOT/models/gemma-4-${cfg/t?-iq4xs/}-it-iq4xs.gguf -ctv q8_0
done
```

Result files: `metrics/gemma4_optim_phase2/exp_iq4xs_t{2,3}-iq4xs.json` +
matching `_sermon.jsonl` per-clip records.

---

## Decision

PR3 ships:
1. The IQ4_XS GGUF artifacts (locally produced; not in `models.lock.json`
   yet — the user can decide whether to host on HF).
2. The CONFIGS extension in `bench_translate_t1_t4.py` so PR4's imatrix
   work can reuse the same harness.
3. This writeup documenting the canary regression that motivates PR4.

PR3 does **NOT** swap the production default. `start_server.sh` continues
to point at `gemma-4-e4b-it-q4km.gguf`. No changes to operator-facing
behavior.

PR4 (imatrix calibration) follows next, with the explicit goal of
recovering the E4B canary at IQ4_XS-level latency.
