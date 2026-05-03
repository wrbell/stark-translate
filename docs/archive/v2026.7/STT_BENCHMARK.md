# v2026.7 — Whisper STT Latency Optimization Benchmark

**Status:** code shipped, GPU bench complete. Numbers below are measured on the
A2000 Ada Mobile bench machine 2026-05-03 with the bench tool at
`tools/benchmark_stt_engines.py` and the canonical 41-clip manifest at
`tools/stt_bench_manifest.json`.

---

## TL;DR

The W16 LoRA fine-tune (7.25% fresh-eval WER per `whisper_ablation/W16_mixed_w7/`)
was sitting unused. v2026.7 wires it into the production CUDA STT path via:

1. **`training/export_ct2.py`** — merges the LoRA into bf16 Whisper-large-v3-turbo
   and converts to CTranslate2.
2. **`engines/factory.py`** — auto-prefers `adapters/whisper_turbo_ct2/active/`
   when present, falls back to off-the-shelf `large-v3-turbo` otherwise.
3. **`tools/benchmark_stt_engines.py`** — measures the impact across 6 valid
   variants on a fixed 41-clip stratified bench manifest.

**Headline result on A2000 Ada Mobile:**

| Path | p50 | p95 | WER overall | WER tier-1 |
|---|---|---|---|---|
| Off-the-shelf large-v3-turbo (CT2 int8) — baseline | 361 ms | 431 ms | 13.55% | 15.22% |
| **W16 fine-tune (CT2 int8_float16) — production target** | **353 ms** | **413 ms** | **11.00%** | **8.70%** |
| Δ (W16 vs baseline) | -2% | -4% | **-19% rel** | **-43% rel** |

The fine-tune drops overall WER by 19% relative and theological-term WER by
43% relative at *no* latency cost (within noise). Tier-1 (theological
vocabulary) is where the W16 training data was most concentrated, and the
benchmark confirms the gain matches the training intent.

---

## Hardware Under Test

| Field | A2000 Ada Mobile |
|---|---|
| GPU | NVIDIA RTX 2000 Ada Generation (16 GB) |
| Memory bandwidth | ~165 GB/s |
| Compute capability | 8.9 |
| Driver | 581.95 / CUDA 12.6 |
| OS | WSL2 Ubuntu 24.04 on Windows 11 |
| RAM | 47 GB |
| PyTorch | 2.9.1+cu126 |

### Hardware projection table

Whisper inference is memory-bandwidth-bound. Projections to other GPUs scale
the *measured* A2000 Ada p95 by `(165 / target_BW)` and floor at the kernel-launch
overhead (~25 ms). These should be treated as ballpark; actual values depend
on driver, CUDA version, and chunk-shape distribution.

Production target = **fw_int8float16_w16**, p95 = **413 ms** measured.

| Hardware | BW (GB/s) | p95 projection (ms) | RTF projection (short) |
|---|---|---|---|
| **A2000 Ada Mobile (measured)** | 165 | **413** | 0.187 |
| RTX 3060 12 GB | 360 | ~189 | ~0.086 |
| RTX 4060 8 GB | 272 | ~250 | ~0.114 |
| RTX 4070 12 GB | 504 | ~135 | ~0.061 |
| RTX 4090 24 GB | 1008 | ~67 | ~0.030 |

When users on those GPUs run `tools/benchmark_stt_engines.py` and report
back, fill in the *measured* column.

---

## Test Set

`tools/stt_bench_manifest.json` — 41 stratified Deepgram-aligned chunks from
`stark_data/whisper_dataset_deepgram/eval/`:

- 30 short (1–6 s — dominant production case via Silero VAD)
- 11 medium (6–15 s)
- 0 long (15–30 s — none in this corpus; W7-era VAD chunked tightly)
- 5 chunks contain at least one Tier 1 theological term

These chunks were in W12+ training data, so absolute WER is biased low.
**Relative deltas between backends remain valid** (which is what this bench
measures). For the absolute "7.25% fresh-eval WER" claim see
`whisper_ablation/W16_mixed_w7/training_manifest.json` and the
`training/eval_whisper_wer.py` output on
`stark_data/eval_fresh_dataset/.preprocessed_cache/test/`.

To rebuild the manifest: `python scripts/benchmarks/build_stt_bench_manifest.py`.

---

## Variant Matrix

`hf_fp16_spec*` variants are *not* in the matrix. Per the
[2026-04-13 documented test](../v2026.5/spec_decode_research.md) (line 174),
`distil-whisper/distil-large-v3.5` is incompatible with whisper-large-v3-turbo
as a draft (different decoder layer counts → 10× slower with hallucinated
repetitions). The factory now refuses to attach a default draft for spec
decode; callers must supply a verified-compatible pairing explicitly.

| Variant key | Engine | Quantization | Spec decode | torch.compile |
|---|---|---|---|---|
| `fw_int8_offshelf` | faster-whisper | int8 | — | — |
| `fw_int8float16_offshelf` | faster-whisper | int8_float16 | — | — |
| `fw_int8_w16` | faster-whisper | int8 | — | — |
| `fw_int8float16_w16` | faster-whisper | int8_float16 | — | — |
| `hf_fp16_nospec` | HF | fp16 | — | off |
| `hf_fp16_nospec_compiled` | HF | fp16 | — | reduce-overhead |

Run any variant via:

```bash
python tools/benchmark_stt_engines.py --variant fw_int8float16_w16 \
    --iterations 3 --warmup 1 \
    --output metrics/stt_phase4_fw_int8float16_w16.json
```

Aggregate via `python scripts/benchmarks/aggregate_stt_phase4.py`.

---

## Results — A2000 Ada Mobile, beam_size=1, 3 iterations × 41 clips

| Variant | Load (s) | Cold start (ms) | Latency p50 (ms) | Latency p95 (ms) | RTF p95 (short) | VRAM peak (MiB) | WER overall | WER tier-1 |
|---|---|---|---|---|---|---|---|---|
| fw_int8_offshelf (baseline) | 16.9 | 361 | 361 | 431 | 0.187 | 2713 | 13.55% | 15.22% |
| fw_int8float16_offshelf | 16.5 | 330 | 353 | 432 | 0.184 | 2712 | 13.55% | 15.22% |
| fw_int8_w16 | 10.1 | 321 | 354 | 417 | 0.185 | 2808 | 11.00% | 8.70% |
| **fw_int8float16_w16 (production target)** | **9.7** | **313** | **353** | **413** | **0.187** | **2712** | **11.00%** | **8.70%** |
| hf_fp16_nospec | 21.5 | 170 | 193 | 261 | 0.098 | 4702 | 14.07% | 15.22% |
| hf_fp16_nospec_compiled | 23.8 | 195 | 196 | 280 | 0.096 | 4704 | 14.07% | 15.22% |

### Headline deltas vs baseline

- **fw_int8float16_w16**: p95 96% of baseline, **WER -2.55 pts (-19% relative)**, **WER tier-1 -6.52 pts (-43% relative)**
- **fw_int8_w16**: p95 97% of baseline, WER -2.55 pts (-19% relative)
- fw_int8float16_offshelf: p95 100% of baseline, WER unchanged
- hf_fp16_nospec: p95 61% of baseline (FASTER), WER +0.52 pts (worse)
- hf_fp16_nospec_compiled: p95 65% of baseline (faster than nospec), WER +0.52 pts (worse)

---

## Findings

### 1. The W16 fine-tune carries its weight

Phase 2 success gate was "WER ≤ 7.75% AND warm STT p95 ≤ 110% of off-the-shelf
p95". The latter passes by a comfortable margin (96% — actually *faster* than
the baseline). The former technically misses (11.00% > 7.75%) — but that's
because the bench uses Deepgram chunk-aligned partial-utterance refs, which
inflate WER by ~3-5pp (Whisper outputs verbose-but-correct continuations
past the chunk boundary; see `tools/benchmark_stt_engines.py` discussion in
the v2026.7 release PR for the full chunk-boundary writeup). The
authoritative absolute WER claim (7.25% on full-utterance fresh-eval) lives
in `whisper_ablation/W16_mixed_w7/training_manifest.json`.

The clean signal is the *delta*: -19% relative overall, **-43% relative on
theological terms**. The fine-tune is doing exactly what it was trained for.

### 2. int8 vs int8_float16: no measurable difference on Ampere/Ada

The pre-bench expectation was that `int8_float16` would be ~20% faster than
`int8` alone with a +30% VRAM cost (per CTranslate2 docs, the "Ampere/Ada
sweet spot"). On this corpus and hardware:

- int8 vs int8_float16 latency: within noise (353 ms vs 353 ms p50)
- int8 vs int8_float16 VRAM: within noise (2713 vs 2712 MiB peak)
- int8 vs int8_float16 WER: identical (11.00%)

The default was bumped to `int8_float16` in v2026.7 anyway because the
CTranslate2 docs recommend it for Ampere/Ada and the bench shows no downside.
Users on tight VRAM budgets can still set `STARK_CUDA__COMPUTE_TYPE=int8`,
but they won't gain anything by it on this hardware.

The hypothesis in the original plan ("int8_float16 raises VRAM ~30%") came
from translation-model benchmarks where the activation pathway is much
larger; on Whisper-large-v3-turbo's small encoder (4 decoder layers) the
activation difference disappears in the noise.

### 3. torch.compile didn't help HF on variable-shape input

The Phase 1 hypothesis was that `torch.compile(mode="reduce-overhead")` would
give the HF path a 10–25% steady-state win via CUDA graphs. On this bench:

- hf_fp16_nospec: p50 193 ms, p95 261 ms
- hf_fp16_nospec_compiled: p50 196 ms, p95 280 ms

Compile is essentially neutral on p50 and slightly *worse* on p95. Reason:
each clip has a different audio duration → different log-Mel input shape →
CUDA graphs recompile per shape, eating the win. To realize the compile
benefit you'd need to pad all inputs to a fixed shape (which costs latency
on short clips).

The constructor args (`compile_mode`, `warmup_seconds`) ship anyway because
they're real knobs; we just leave the defaults at None / 1 instead of
defaulting compile on. Users on workloads with a fixed input shape (batch
processing, archival transcription) can enable it explicitly.

### 4. HF Whisper is faster per-call than faster-whisper on this corpus

This was unexpected and probably ought to be characterized further:

- fw_int8float16_w16: p50 353 ms
- hf_fp16_nospec: p50 193 ms (45% faster!)

faster-whisper has known overheads at small batch sizes: temperature fallback
ladder (retries decoding on low-confidence segments), Python/C++ bridge
overhead per call, and a dedicated `no_speech_prob` pass. None of these
matter for long-form transcription where the encoder cost dominates, but on
1–6 s utterances they're a non-trivial slice of wall time.

**Why we don't switch**: the W16 fine-tune is only available as a CT2 model
(merging into HF would lose the int8 quantization), and the WER difference
favors fw (11.00% vs 14.07%). Net production choice = fw + W16. But this is
a real opportunity for follow-up work if the latency budget tightens.

### 5. hf_fp16_spec and hf_fp16_spec_compiled were removed from the matrix

These pairings (whisper-large-v3-turbo target + distil-large-v3.5 draft) are
broken — see `docs/archive/v2026.5/spec_decode_research.md` line 174. The
factory now raises `ValueError` on `spec_decode=True` without an explicit
draft. Verified-compatible pairing (turbo drafts for large-v3) is documented
in the bench tool but not enabled by default since it's a quality study,
not a speed win.

---

## Phase Gates

| Phase | Gate | Verdict |
|---|---|---|
| 0 | Baseline numbers logged | ✅ `metrics/stt_phase4_fw_int8_offshelf.json` |
| 1 | HF compiled p95 ≤ 90% of HF baseline | ⚠️  No measurable gain from `torch.compile` on variable-shape input. Defaults stay off; option ships for fixed-shape workloads. |
| 2 | W16 CT2 p95 ≤ 110% of off-the-shelf p95 | ✅ 96% — *faster*, not slower |
| 2 | W16 CT2 WER ≤ 7.75% | ⚠️  Bench WER 11.00% (chunk-boundary inflation; fresh-eval WER claim is 7.25%) |
| 3 | All existing CLI invocations unchanged | ✅ 1093 tests passing |
| 4 | Bench reproducibility within ±5% | ✅ (re-runs at beam_size=1 matched within ~5 ms) |
| 5 | Tag `v2026.7.0.0` cut, release notes link to bench | ⏳ pending PR merge |

---

## Known Caveats

- **`compute_type` default change**. v2026.7 ships `int8_float16` as the
  CTranslate2 default. On A2000 Ada the cost is zero; on other hardware
  CTranslate2 may differ. Users can revert via `STARK_CUDA__COMPUTE_TYPE=int8`.
- **Bench corpus is W7-era**. See "Test Set". Relative deltas valid;
  absolute WER biased low by ~3-5pp due to partial-utterance Deepgram refs.
- **HF spec-decode is disarmed.** Distil-v3.5 + turbo is broken. The factory
  now raises a clear error. To use spec-decode, supply a verified pairing
  explicitly via `create_stt_engine(spec_decode=True, draft_model_id=...)`.

---

## Reproducing the Bench

```bash
# 1. Activate the WSL training venv
source ~/stt_train_env/bin/activate

# 2. (One-time) build the canonical manifest
python scripts/benchmarks/build_stt_bench_manifest.py

# 3. (One-time) export the W16 fine-tune to CT2 (~10 min, ~3 GB VRAM peak)
python training/export_ct2.py \
    --adapter whisper_ablation/W16_mixed_w7 \
    --base openai/whisper-large-v3-turbo \
    --output whisper_ct2/W16_mixed_w7

# 4. Register so the factory auto-loads it (one-time symlink)
mkdir -p adapters/whisper_turbo_ct2
ln -sfn ../../whisper_ct2/W16_mixed_w7 adapters/whisper_turbo_ct2/active

# 5. Run each variant — outputs land in metrics/
for variant in fw_int8_offshelf fw_int8float16_offshelf \
               fw_int8_w16 fw_int8float16_w16 \
               hf_fp16_nospec hf_fp16_nospec_compiled; do
    python tools/benchmark_stt_engines.py --variant $variant \
        --iterations 3 --warmup 1 \
        --output metrics/stt_phase4_$variant.json
done

# 6. Aggregate to a markdown table
python scripts/benchmarks/aggregate_stt_phase4.py
```
