# IMMEDIATE_TODO.md — Live Demo Session Notes (March 1, 2026)

Observations from today's live church service run (`dry_run_ab.py`, 4B default mode).

---

## Speaker 1 Issues

- ~~**Mic inflections**~~ — DONE. VAD threshold raised 0.30→0.42, min buffer 0.5→0.7s, pre-STT RMS energy gate added.
- ~~**Erroneous "thank you"s (TOP NEED)**~~ — DONE. 4-tier hallucination suppression in `dry_run_ab.py` (Tier 1: garbage phrases, Tier 2: confidence+duration, Tier 3: low-conf short, Tier 4: dedup). "Thank you" caught by Tier 1 exact match + Tier 2 confidence gate.
- ~~**"That we / Daniel"**~~ — DONE. Phrase corrections `"damn you" → "Daniel"` and `"danic" → "Daniel"` added.
- **Restart between speakers** — Test whether restarting the pipeline between Speaker 1 and Speaker 2 fixes the inflection issue. Accumulated state/context may be degrading STT quality over the session.
- ~~**Erroneous text filter**~~ — DONE. 4-tier suppression system implemented in `dry_run_ab.py:307-373`. Catches phantom phrases, low-confidence hallucinations, short-duration garbage, and consecutive duplicates.

## Speaker 2 Issues

- ~~**"Sagat Ruvan" hallucination**~~ — DONE. Caught by Tier 3 (low-confidence short utterance filter, conf=0.06 well below threshold).

---

## ~~NOW: Display QoL~~ — DONE

Three changes to the web UI:

1. ~~**Eliminate pause between bold finals and italic partials**~~ — DONE. Backend now allows partials for new utterances during final translation (`_final_pending_utterance_id` tracking). Frontend only clears partials matching the finalized chunk_id.
2. ~~**Flip layout: top (English) / bottom (Spanish)**~~ — DONE. Already stacked EN top / ES bottom in `audience_display.html`.
3. ~~**Keep italics for partials**~~ — DONE. Partials display in italic gray, finals in normal weight.

---

## ~~NOW: Validation Pipeline~~ — DONE

A 5-step post-sermon validation run using existing models on both the locally recorded WAV and YouTube audio from the same date (e.g., today's teaching message):

1. ~~**Run MarianMT & 12B on our recorded transcription**~~ — DONE. Multi-speaker validation pipeline implemented in `tools/validate_session.py`.
2. ~~**Run Whisper on YouTube downloaded audio**~~ — DONE. YouTube audio download + Whisper STT integrated.
3. ~~**Run MarianMT & 12B on Whisper-transcribed YouTube audio**~~ — DONE. Both translation models run on YT-derived transcription.
4. ~~**Compare live results to both downloaded results**~~ — DONE. Cross-comparison with WER/quality metrics.
5. ~~**Prepare findings**~~ — DONE. Validation reports generated in `metrics/validation_*.md`.

---

## LATER: C++ / GPU Optimization Proposal

**Philosophy:** Dev in Python, prod optimized in C++

### Stark-Translate: Layered Optimization Implementation Plan

- **Project:** stark-translate real-time multilingual translation pipeline
- **Target:** Sub-1s translation latency for live 10-second audio chunks
- **Hardware path:** RTX 2070 (current) → RTX 3060 12GB → RTX 3090 24GB

### Layer Overview

| Layer | Cost | Expected Result | Prerequisite |
|-------|------|-----------------|-------------|
| 1. Baseline on RTX 2070 | $0 | ~3–4.5s end-to-end | Existing hardware |
| 2. Runtime optimization | $0 | ~1.2–1.5s translation | Layer 1 working |
| 3. RTX 3060 GPU swap | ~$200 | VRAM headroom, same speed | Layer 1 or 2 |
| 4. RTX 3090 GPU swap | ~$600 | Sub-1s with vanilla tooling | Replaces Layer 2+3 |

### Layer 1: Baseline Dual-Model Inference on RTX 2070

**Goal:** Both models loaded simultaneously at INT8, ~3–4.5s end-to-end, stable over 100+ chunks.

**Steps:**
1. Environment setup — CUDA toolkit verification, install faster-whisper/bitsandbytes/transformers/accelerate/torch
2. Load Whisper Large-V3-Turbo at INT8 (~1.3 GB VRAM)
3. Load TranslateGemma 4B at INT8 (~6.2–6.5 GB total)
4. Memory guardrails — `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, disable gradients, empty cache
5. Build inference pipeline
6. 100-chunk stress test

**Pass criteria:** Total < 5s average, VRAM peak < 7.5 GB, no memory leaks over 100 chunks

**Expected baselines:**

| Metric | Value |
|--------|-------|
| ASR | 1.0–1.5s |
| Translation | 2.0–3.5s |
| Total | 3.0–4.5s |
| VRAM peak | 6.2–7.0 GB |

### Layer 2: Runtime Optimization (TRT-LLM or llama.cpp)

**Goal:** Replace HuggingFace `generate()` with compiled inference engine, sub-1.5s translation.

**Two paths:**
- **llama.cpp (recommended first)** — Easier setup, GGUF Q8_0 conversion, ~25–40 tok/s on 2070
  - `CMAKE_ARGS="-DGGML_CUDA=on"`, convert to GGUF Q8_0, `n_ctx=256`, `n_gpu_layers=-1`
- **TRT-LLM** — Higher complexity, ~30–45 tok/s

**Optional:** Reduce chunk size from 10s to 5s for sub-1s translation (tradeoff: may split sentences awkwardly)

**Pass criteria:** Translation < 1.5s (10s chunks) or < 0.9s (5s chunks)

### Layer 3: RTX 3060 12GB GPU Swap

**Goal:** Eliminate VRAM pressure, 5 GB headroom.

- **Cost:** ~$200 used (eBay), ~$40 net after selling 2070
- **Requirements:** 550W PSU, 8-pin PCIe power
- **Optional:** Try Whisper at FP16 instead of INT8 (~7.5 GB total, still fits with 4.5 GB to spare)

**Pass criteria:** Same latency, VRAM peak < 8 GB, 4+ GB headroom

### Layer 4: RTX 3090 24GB GPU Swap

**Goal:** Sub-1s with vanilla HuggingFace, no TRT-LLM/llama.cpp needed.

- **Cost:** ~$550–700 used, ~$400–540 net
- **Requirements:** 750W PSU minimum, 2x 8-pin PCIe, 3-slot physical clearance
- **Option A:** FP16 everything (~10.6 GB, max quality)
- **Option B (recommended):** INT8 everything (~6.5 GB, max speed ~45–55 tok/s). 3090's 936 GB/s bandwidth gives ~2x speed over 2070's 448 GB/s.

**Pass criteria:** Translation < 1.2s with vanilla HuggingFace at INT8

### GPU Comparison Card

| Spec | RTX 2070 | RTX 3060 | RTX 3090 |
|------|----------|----------|----------|
| VRAM | 8 GB | 12 GB | 24 GB |
| Memory bandwidth | 448 GB/s | 360 GB/s | 936 GB/s |
| INT8 tensor cores | Turing (basic) | Ampere (mature) | Ampere (mature) |
| Used price (Feb 2026) | ~$160 | ~$200 | ~$550–700 |
| PSU requirement | 550W | 550W | 750W |
| Both models fit (INT8)? | Barely (1.4 GB free) | Yes (5 GB free) | Yes (17 GB free) |
| Both models fit (FP16)? | No | No | Yes (13 GB free) |
| HF `generate()` tok/s | ~15–18 | ~15–20 | ~45–55 |
| llama.cpp tok/s | ~25–40 | ~25–35 | ~60–80 |
| Translation time (HF, 50 tok) | ~3s | ~2.5–3s | ~0.9–1.2s |

### VRAM Budget Cheat Sheet

| Component | FP16 | INT8 | Q4 |
|-----------|------|------|-----|
| Whisper Large-V3-Turbo | 1.6 GB | 0.9 GB | — |
| TranslateGemma 4B | 8.0 GB | 5.0 GB | 3.4 GB |
| CUDA context (fixed) | 0.4 GB | 0.4 GB | 0.4 GB |
| KV cache (per 60-token chunk) | 0.025 GB | 0.025 GB | 0.025 GB |
| PyTorch allocator overhead | 0.2 GB | 0.2 GB | 0.2 GB |
| **Total** | **10.2 GB** | **6.5 GB** | **5.0 GB** |

### Recommended Upgrade Path

```
Current: RTX 2070 + Layer 1 → ~3-4.5s (works today, $0)
     |
     +-- Need sub-1.5s? → Add Layer 2 (TRT-LLM/llama.cpp, $0, ~1 weekend)
     |
     +-- Want VRAM peace of mind? → Layer 3: swap to RTX 3060 (~$40 net)
     |
     +-- Want sub-1s with zero complexity? → Layer 4: swap to RTX 3090 (~$400-540 net)
```
