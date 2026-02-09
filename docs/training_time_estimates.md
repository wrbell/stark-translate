# Training Time Estimates — A2000 Ada 16GB (Windows/WSL2)

> **Hardware:** NVIDIA A2000 Ada -- 4,352 CUDA cores, ~16 TFLOPS FP16/BF16, 16GB GDDR6, 288 GB/s bandwidth
> **System:** 64GB RAM, WSL2, CUDA
> **Relative performance:** ~2x a T4, ~60% of an RTX 3090 (per CLAUDE.md)
> **Date:** 2026-02-08

---

## Table of Contents

1. [GPU Performance Baseline](#1-gpu-performance-baseline)
2. [Data Preprocessing Time Estimates](#2-data-preprocessing-time-estimates)
3. [Whisper LoRA Training Time](#3-whisper-lora-training-time)
4. [TranslateGemma 4B QLoRA Training Time](#4-translategemma-4b-qlora-training-time)
5. [TranslateGemma 12B QLoRA Training Time](#5-translategemma-12b-qlora-training-time)
6. [MarianMT Full Fine-Tune Training Time](#6-marianmt-full-fine-tune-training-time)
7. [Evaluation Time](#7-evaluation-time)
8. [End-to-End Timeline Scaling Table](#8-end-to-end-timeline-scaling-table)
9. [Overnight Training Schedule](#9-overnight-training-schedule)
10. [Per-Cycle Time After First Cycle](#10-per-cycle-time-after-first-cycle)

---

## 1. GPU Performance Baseline

### A2000 Ada Positioning

The A2000 Ada's stated specs place it in the mid-range professional GPU tier. For training throughput estimation, the key comparison points are:

| GPU | CUDA Cores | FP16 TFLOPS | Mem BW (GB/s) | VRAM |
|-----|-----------|-------------|---------------|------|
| T4 | 2,560 | 8.1 (FP16) / 65 (INT8) | 320 | 16 GB |
| **A2000 Ada** | **4,352** | **~16** | **288** | **16 GB** |
| RTX 3060 12GB | 3,584 | 12.7 | 360 | 12 GB |
| RTX 3070 | 5,888 | 20.3 | 448 | 8 GB |
| RTX 3090 | 10,496 | 35.6 | 936 | 24 GB |

The A2000 Ada sits between the RTX 3060 and RTX 3070 in raw compute, but has lower memory bandwidth than either. For training workloads (which are more compute-bound than inference), we can estimate:

- **~1.3x the throughput of a T4** for training (despite 2x raw TFLOPS, memory bandwidth limits throughput)
- **~55-65% of an RTX 3090** as stated in project docs
- **~0.85-0.95x an RTX 3060 12GB** for memory-bandwidth-bound workloads (lower bandwidth: 288 vs 360 GB/s)
- **~1.1-1.25x an RTX 3060 12GB** for compute-bound workloads (more CUDA cores)

### Key Throughput Assumptions

These are estimated from published benchmarks on comparable GPUs, scaled by the A2000 Ada's position:

| Workload | Estimated Throughput | Derivation |
|----------|---------------------|------------|
| Whisper LoRA training step (batch=4, 30s audio, bf16) | ~1.5-2.2 s/step | Scaled from T4 and RTX 3060 benchmarks |
| QLoRA 4B forward+backward (batch=1, seq=512, 4-bit) | ~2.5-3.5 s/step | Scaled from RTX 4060 benchmarks (~1.4 s/step at 8GB) |
| QLoRA 12B forward+backward (batch=1, seq=512, 4-bit) | ~5.5-7.5 s/step | ~2.2x the 4B time due to larger model |
| MarianMT full fine-tune step (batch=32, seq=128) | ~0.15-0.25 s/step | 298MB model, trivially small |
| Whisper large-v3 inference (30s chunk, fp16) | ~1.5-2.5s per chunk | ~10-15x real-time |
| Demucs htdemucs (GPU) | ~0.8-1.2x real-time | GPU-accelerated source separation |

---

## 2. Data Preprocessing Time Estimates

### 2.1 yt-dlp Download

Download time is network-bound, not GPU-bound.

| Network Speed | Time per Hour of Audio | 30 hrs | 50 hrs | 250 hrs |
|---------------|----------------------|--------|--------|---------|
| 50 Mbps | ~3.5 min | 1.7 hrs | 2.9 hrs | 14.5 hrs |
| 100 Mbps | ~1.8 min | 0.9 hrs | 1.5 hrs | 7.3 hrs |
| 200 Mbps | ~0.9 min | 0.5 hrs | 0.8 hrs | 3.6 hrs |

**Assumptions:** YouTube audio at ~128 kbps = ~57.6 MB/hr. Actual download includes overhead (metadata, throttling, subtitle fetches). Real-world throughput is typically 50-70% of rated speed due to YouTube throttling.

**Best estimate at 100 Mbps:** ~2 min per video (including subtitle fetch). 30 videos = ~1 hr. 275 videos = ~9 hrs.

### 2.2 Audio Preprocessing (10-Step Pipeline)

The pipeline runs per-file, sequentially through 10 steps. Bottleneck analysis:

| Step | Operation | Speed | GPU? | Notes |
|------|-----------|-------|------|-------|
| 1 | yt-dlp download | Network-bound | No | See above |
| 2 | ffmpeg format conversion (16kHz mono) | ~50-100x real-time | No | CPU, trivially fast |
| 3 | Initial quality gate (SNR, clipping) | ~200x real-time | No | NumPy, trivially fast |
| 4 | inaSpeechSegmenter classification | ~3-5x real-time | CPU | TF-based, moderate speed |
| 5 | **Demucs source separation** | **~0.8-1.2x real-time** | **Yes** | **Primary GPU bottleneck** |
| 6 | ffmpeg bandpass filter | ~50-100x real-time | No | Trivially fast |
| 7 | noisereduce spectral gating | ~5-10x real-time | No | CPU, FFT-based |
| 8 | pyloudnorm normalization | ~200x real-time | No | Trivially fast |
| 9 | Silero VAD chunking | ~100x real-time | CPU | Very fast |
| 10 | Final quality gate | ~200x real-time | No | NumPy, trivially fast |

**Demucs is the bottleneck.** On GPU with htdemucs:
- Published benchmarks show ~1.5x real-time on CPU, with GPU providing ~5x speedup (MPS) to ~20x speedup (CUDA)
- On the A2000 Ada, expect ~3-6x real-time for htdemucs with CUDA
- However, not all files need demucs -- only those with significant music segments (typically 30-50% of church recordings have music)

**Conditional demucs:** The `preprocess_audio.py` script skips demucs when music < 10 seconds. Assuming ~40% of files have significant music:

| Data Size | Total Audio | Files Needing Demucs | Demucs Time | Other Steps | Total Pipeline |
|-----------|------------|---------------------|-------------|-------------|----------------|
| 10 hrs | 10 hrs | ~4 hrs audio | 0.7-1.3 hrs | 0.4 hrs | **1.1-1.7 hrs** |
| 20 hrs | 20 hrs | ~8 hrs audio | 1.3-2.7 hrs | 0.8 hrs | **2.1-3.5 hrs** |
| 30 hrs | 30 hrs | ~12 hrs audio | 2.0-4.0 hrs | 1.2 hrs | **3.2-5.2 hrs** |
| 50 hrs | 50 hrs | ~20 hrs audio | 3.3-6.7 hrs | 2.0 hrs | **5.3-8.7 hrs** |
| 250 hrs | 250 hrs | ~100 hrs audio | 17-33 hrs | 10 hrs | **27-43 hrs** |

### 2.3 VAD Segmentation

Silero VAD runs at ~100x real-time on CPU. This is already included in step 9 of the pipeline above.

Standalone timing: 30 hrs audio -> ~18 min. 50 hrs -> ~30 min. Negligible.

### 2.4 Whisper large-v3 Pseudo-Labeling

This is the **second major GPU bottleneck** after demucs.

**Key parameters from `transcribe_church.py`:**
- Model: `distil-whisper/distil-large-v3` (default) or `openai/whisper-large-v3` (max quality)
- Backend: transformers pipeline or faster-whisper
- Batch size: 8 (default)
- fp16 on CUDA

**Inference speed estimation:**

Whisper large-v3 on comparable GPUs:
- RTX 3060: ~15-25x real-time (faster-whisper, fp16, batch processing)
- T4: ~8-12x real-time (faster-whisper, fp16)
- A2000 Ada (estimated): ~10-18x real-time

Distil-Whisper large-v3 is ~5.8x faster than Whisper large-v3 for inference, so:
- Distil-Whisper on A2000 Ada: ~50-100x real-time (very fast, not the bottleneck)
- Whisper large-v3 on A2000 Ada: ~10-18x real-time (the recommended model for pseudo-labeling)

**Using Whisper large-v3 (recommended for labels):**

After VAD chunking, audio is split into 1-30s segments. Average segment ~15s. The number of segments depends on speech density:
- Church sermons: ~70-80% speech -> ~42-48 min of speech per hour of raw audio
- At ~15s average segment -> ~170-190 segments per hour of raw audio

With faster-whisper at ~12x real-time on A2000 Ada:

| Data Size | Speech Audio | Segments (~15s avg) | Inference Time | Including Overhead |
|-----------|-------------|--------------------|-----------------|--------------------|
| 10 hrs | ~7 hrs speech | ~1,680 | 0.6 hrs | **~0.8 hrs** |
| 20 hrs | ~14 hrs speech | ~3,360 | 1.2 hrs | **~1.5 hrs** |
| 30 hrs | ~21 hrs speech | ~5,040 | 1.8 hrs | **~2.2 hrs** |
| 50 hrs | ~35 hrs speech | ~8,400 | 2.9 hrs | **~3.5 hrs** |
| 250 hrs | ~175 hrs speech | ~42,000 | 14.6 hrs | **~18 hrs** |

**Overhead** accounts for model loading (~30s), file I/O, and JSON serialization (~20% of inference time).

**Using distil-whisper (faster but lower quality labels):**

Divide the above times by ~4-5x. The 30-hour dataset would take ~0.5 hrs instead of ~2.2 hrs.

### 2.5 Human Correction

Not GPU-bound. Estimated at 5-8x real-time for review + correction of the bottom 20% by confidence.

| Data Size | Segments to Correct (20%) | Human Time (6x real-time) |
|-----------|--------------------------|--------------------------|
| 10 hrs | ~336 segs (~1.4 hrs audio) | **~8.4 hrs** |
| 20 hrs | ~672 segs (~2.8 hrs audio) | **~16.8 hrs** |
| 30 hrs | ~1,008 segs (~4.2 hrs audio) | **~25.2 hrs** |
| 50 hrs | ~1,680 segs (~7.0 hrs audio) | **~42 hrs** |

### 2.6 Total Preprocessing Summary

| Data Size | Download (100Mbps) | Preprocess | Pseudo-label (large-v3) | Human Correction | **Total** |
|-----------|-------------------|------------|------------------------|-----------------|-----------|
| 10 hrs | 0.3 hrs | 1.1-1.7 hrs | 0.8 hrs | 8.4 hrs | **10.6-11.2 hrs** |
| 20 hrs | 0.7 hrs | 2.1-3.5 hrs | 1.5 hrs | 16.8 hrs | **21.1-22.5 hrs** |
| 30 hrs | 1.0 hrs | 3.2-5.2 hrs | 2.2 hrs | 25.2 hrs | **31.6-33.6 hrs** |
| 50 hrs | 1.7 hrs | 5.3-8.7 hrs | 3.5 hrs | 42 hrs | **52.5-55.9 hrs** |
| 250 hrs | 9 hrs | 27-43 hrs | 18 hrs | 210 hrs | **264-280 hrs** |

**Human correction dominates.** Everything else is automated and can run overnight.

---

## 3. Whisper LoRA Training Time

### 3.1 Configuration (from `train_whisper.py`)

```
Model:           distil-whisper/distil-large-v3
Base model:      8-bit quantized (BitsAndBytesConfig load_in_8bit=True)
LoRA:            r=32, alpha=64, target q_proj+v_proj, dropout=0.05
Batch size:      4 per device
Grad accumulation: 4 steps (effective batch size = 16)
Optimizer:       adamw_bnb_8bit
Precision:       bf16
Gradient checkpointing: enabled
Learning rate:   1e-4
Warmup:          500 steps
Replay ratio:    0.3 (70% church + 30% general domain)
```

### 3.2 Dataset Size Calculations

**Segments per hour of audio:**
- After VAD chunking, average segment duration ~15s
- ~70-80% speech ratio -> ~170-190 segments per hour of raw audio
- With replay ratio 0.3: total segments = church_segments / 0.7 (adds ~43% more samples)

| Raw Audio | Church Segments | + Replay (30%) | Total Training Samples |
|-----------|----------------|----------------|----------------------|
| 10 hrs | ~1,800 | ~770 | **~2,570** |
| 20 hrs | ~3,600 | ~1,540 | **~5,140** |
| 30 hrs | ~5,400 | ~2,310 | **~7,710** |
| 50 hrs | ~9,000 | ~3,860 | **~12,860** |

### 3.3 Steps Per Epoch

Steps per epoch = total_samples / effective_batch_size

Effective batch size = batch_size * grad_accum = 4 * 4 = 16

| Raw Audio | Total Samples | Steps/Epoch |
|-----------|--------------|-------------|
| 10 hrs | ~2,570 | ~161 |
| 20 hrs | ~5,140 | ~321 |
| 30 hrs | ~7,710 | ~482 |
| 50 hrs | ~12,860 | ~804 |

### 3.4 Time Per Step

Whisper LoRA training on a mid-range GPU with:
- 8-bit base model + bf16 LoRA layers
- Batch size 4, gradient checkpointing
- Seq2Seq with audio features (80-channel mel spectrogram, ~3000 frames for 30s)

**Published references:**
- Whisper-large-v2 LoRA on T4 (8-bit, batch=4): ~3-4 s/step
- Whisper-large-v2 LoRA on RTX 3090 (8-bit, batch=4): ~1.0-1.5 s/step
- Distil-Whisper is ~40% fewer decoder layers -> ~25-35% faster per step

**A2000 Ada estimate (distil-whisper, 8-bit, batch=4, bf16):**
- ~1.5-2.2 s/step
- **Best estimate: ~1.8 s/step**

### 3.5 VRAM Usage

| Component | VRAM |
|-----------|------|
| Distil-Whisper large-v3 (8-bit) | ~0.8 GB |
| LoRA parameters (r=32, q_proj+v_proj) | ~0.1 GB |
| Optimizer states (8-bit AdamW) | ~0.2 GB |
| Activations (batch=4, grad checkpointing) | ~4-6 GB |
| Audio feature buffers | ~1-2 GB |
| PyTorch overhead + CUDA context | ~1-2 GB |
| **Total estimated** | **~8-11 GB** |

Fits comfortably within 16GB with ~5-8 GB headroom.

### 3.6 Total Training Time

At ~1.8 s/step:

| Raw Audio | Steps/Epoch | 3 Epochs | 5 Epochs |
|-----------|-------------|----------|----------|
| 10 hrs | 161 | 483 steps = **~14.5 min** | 805 steps = **~24 min** |
| 20 hrs | 321 | 963 steps = **~29 min** | 1,605 steps = **~48 min** |
| 30 hrs | 482 | 1,446 steps = **~43 min** | 2,410 steps = **~1.2 hrs** |
| 50 hrs | 804 | 2,412 steps = **~1.2 hrs** | 4,020 steps = **~2.0 hrs** |

**Note:** These times are significantly lower than the CLAUDE.md initial estimates (5-8 hrs for 20 hrs audio). The difference is because:

1. **Distil-Whisper** is used (not full Whisper-large), which has fewer decoder layers
2. **8-bit quantization** reduces base model memory and compute
3. The initial CLAUDE.md estimates were conservative upper bounds

**Adjusted estimates including warmup, eval, checkpointing, and I/O overhead (~40% overhead):**

| Raw Audio | 3 Epochs (adjusted) | 5 Epochs (adjusted) |
|-----------|---------------------|---------------------|
| 10 hrs | **~20 min** | **~34 min** |
| 20 hrs | **~41 min** | **~1.1 hrs** |
| 30 hrs | **~1.0 hr** | **~1.7 hrs** |
| 50 hrs | **~1.7 hrs** | **~2.8 hrs** |

**With extended target modules** (q_proj, v_proj, k_proj, out_proj, fc1, fc2): multiply by ~2.5-3x. The 30-hour dataset at 3 epochs would take ~2.5-3.0 hrs.

### 3.7 Sensitivity Analysis

| Scenario | Impact on Time |
|----------|---------------|
| Full Whisper-large-v3 instead of distil | ~1.4-1.6x slower |
| Expand target modules to all 6 layers | ~2.5-3x slower |
| Increase batch size to 8 (may OOM) | ~0.7x faster per step, fewer steps |
| Use full fp16 instead of 8-bit base | ~0.9x faster per step, more VRAM |
| Curriculum learning (3-phase) | Same total time, different ordering |

---

## 4. TranslateGemma 4B QLoRA Training Time

### 4.1 Configuration (from `train_gemma.py`)

```
Model:           google/translategemma-4b-it
Quantization:    4-bit NF4, double quantization, bf16 compute
LoRA:            r=16, alpha=16, target all-linear, dropout=0.05
Batch size:      2 per device
Grad accumulation: 4 steps (effective batch size = 8)
Optimizer:       paged_adamw_32bit
Precision:       bf16
Gradient checkpointing: enabled
Packing:         enabled (max_seq_length=512)
Learning rate:   2e-4
```

### 4.2 Dataset and Packing Analysis

**Input data:**
- Bible parallel corpus: ~155,000 verse pairs
- Theological glossary: 229 terms, 458 training pairs
- Total: ~155,458 examples

**Token count estimation:**
- Average Bible verse: ~25-35 English words + ~30-40 Spanish words
- After TranslateGemma chat template formatting: ~80-120 tokens per example
- Average: ~100 tokens per formatted example (including template overhead)

**Packing efficiency:**
- max_seq_length = 512 tokens
- At ~100 tokens per example, ~4-5 verses pack per sequence
- Effective packed sequences: ~155,458 / 4.5 = ~34,500 packed sequences

### 4.3 Steps Per Epoch

Steps per epoch = packed_sequences / effective_batch_size = 34,500 / 8 = ~4,313

### 4.4 Time Per Step

QLoRA 4B model training on mid-range GPUs:
- RTX 4060 (8GB, 4-bit QLoRA, batch=1): ~1.4 s/step (published benchmark)
- The A2000 Ada has 2x the VRAM (16 vs 8 GB) and ~1.3x the compute of an RTX 4060
- With batch=2 instead of 1: ~1.5-1.8x the per-step time of batch=1
- With `all-linear` targets: more LoRA parameters than typical q_proj+v_proj

**A2000 Ada estimate (4B, 4-bit, batch=2, all-linear, bf16):**
- ~2.5-3.5 s/step
- **Best estimate: ~3.0 s/step**

### 4.5 VRAM Usage

| Component | VRAM |
|-----------|------|
| TranslateGemma 4B (4-bit NF4) | ~2.6 GB |
| LoRA parameters (r=16, all-linear) | ~0.3 GB |
| Optimizer states (paged AdamW 32-bit) | ~0.6 GB |
| Activations (batch=2, grad checkpointing, seq=512) | ~3-5 GB |
| Gradient buffers | ~1-2 GB |
| PyTorch overhead + CUDA context | ~1-2 GB |
| **Total estimated** | **~9-13 GB** |

Fits within 16GB with ~3-7 GB headroom. Comfortable.

### 4.6 Total Training Time

At ~3.0 s/step, 4,313 steps/epoch:

| Epochs | Total Steps | Time (raw) | + Overhead (30%) | **Total** |
|--------|------------|------------|-------------------|-----------|
| 1 | 4,313 | 3.6 hrs | 4.7 hrs | **~4.7 hrs** |
| 3 | 12,939 | 10.8 hrs | 14.0 hrs | **~14 hrs** |
| 5 | 21,565 | 18.0 hrs | 23.4 hrs | **~23 hrs** |

**Overhead** includes warmup, logging, checkpointing (every 500 steps), and data loading.

**This aligns well with CLAUDE.md's estimate of 8-12 hrs for 3 epochs.** The difference from the lower end is that:
- `all-linear` targeting adds more trainable parameters than typical configs
- `modules_to_save=["lm_head", "embed_tokens"]` in the config adds additional trainable parameters
- Packing efficiency may vary (less efficient packing = more steps)

**Revised range: 10-14 hrs for 3 epochs** (accounting for packing variability and the `modules_to_save` overhead).

### 4.7 Glossary Impact

The 458 glossary pairs are <0.3% of the dataset. They add negligible training time but are repeated naturally through the shuffled dataset. If upsampled 10x for emphasis (~4,580 pairs), add ~1% to training time.

---

## 5. TranslateGemma 12B QLoRA Training Time

### 5.1 Configuration (from `train_gemma.py`)

```
Model:           google/translategemma-12b-it
Quantization:    4-bit NF4, double quantization, bf16 compute
LoRA:            r=16, alpha=16, target all-linear, dropout=0.05
Batch size:      1 per device
Grad accumulation: 8 steps (effective batch size = 8)
Optimizer:       paged_adamw_32bit
Packing:         enabled (max_seq_length=512)
```

### 5.2 Will It Fit?

| Component | VRAM |
|-----------|------|
| TranslateGemma 12B (4-bit NF4) | ~6.5 GB |
| LoRA parameters (r=16, all-linear, 12B) | ~0.5 GB |
| Optimizer states (paged AdamW 32-bit) | ~1.0 GB |
| Activations (batch=1, grad checkpointing, seq=512) | ~3-4 GB |
| Gradient buffers | ~1.5-2.5 GB |
| PyTorch overhead + CUDA context | ~1-2 GB |
| **Total estimated** | **~14-16.5 GB** |

**This is extremely tight on 16GB VRAM.** Risk assessment:

| Scenario | Likelihood | Mitigation |
|----------|-----------|------------|
| Fits with <1 GB headroom | 50% | Monitor with `nvidia-smi` |
| OOM during backward pass | 30% | Reduce `max_seq_length` to 384 or 256 |
| OOM during optimizer step | 20% | Use `optim="paged_adamw_8bit"` instead of 32bit |

**Recommendations if OOM occurs:**
1. Reduce `max_seq_length` from 512 to 384 (saves ~1 GB activation memory)
2. Switch to 8-bit paged optimizer (saves ~0.5 GB)
3. Remove `modules_to_save` (saves ~0.3 GB)
4. As last resort: reduce LoRA rank from 16 to 8

### 5.3 Steps Per Epoch

Same dataset, same packing -> ~34,500 packed sequences.
Steps per epoch = 34,500 / 8 = ~4,313 (same as 4B, since effective batch size is identical).

### 5.4 Time Per Step

The 12B model is ~3x the parameters of the 4B model, but with 4-bit quantization the forward pass scales roughly linearly with parameter count. The backward pass through LoRA layers also scales.

**Estimate:** ~2.0-2.5x the 4B time per step due to:
- Larger attention matrices (more heads, larger hidden dim)
- More linear layers to compute LoRA gradients through
- Higher memory pressure causing more paging

**A2000 Ada estimate (12B, 4-bit, batch=1, all-linear, bf16):**
- ~5.5-7.5 s/step
- **Best estimate: ~6.5 s/step**

### 5.5 Total Training Time

At ~6.5 s/step, 4,313 steps/epoch:

| Epochs | Total Steps | Time (raw) | + Overhead (30%) | **Total** |
|--------|------------|------------|-------------------|-----------|
| 1 | 4,313 | 7.8 hrs | 10.1 hrs | **~10 hrs** |
| 3 | 12,939 | 23.4 hrs | 30.4 hrs | **~30 hrs** |
| 5 | 21,565 | 38.9 hrs | 50.6 hrs | **~51 hrs** |

**This exceeds CLAUDE.md's estimate of 18-27 hrs for 3 epochs.** The higher estimate here reflects:
- `all-linear` + `modules_to_save` is more expensive than typical QLoRA
- Memory pressure at 14-16 GB causes VRAM paging overhead
- batch=1 with grad_accum=8 has less parallelism efficiency

**Revised realistic range: 22-30 hrs for 3 epochs**, acknowledging that:
- If memory fits cleanly (no paging), closer to 22 hrs
- If memory is tight and paging occurs, closer to 30+ hrs
- Strongly recommend training the 4B model first, only moving to 12B if quality justifies the 2x+ time investment

---

## 6. MarianMT Full Fine-Tune Training Time

### 6.1 Configuration (from `train_marian.py`)

```
Model:           Helsinki-NLP/opus-mt-en-es (298 MB, ~74M parameters)
Quantization:    None (full precision bf16)
Fine-tune:       Full (all parameters trainable)
Batch size:      32 per device
Grad accumulation: 1
Max length:      128 tokens
Optimizer:       AdamW (implicit default)
Learning rate:   5e-5
Eval batch size: 64
```

### 6.2 Dataset and Steps

**Input:** ~155,458 examples (Bible + glossary)
**Tokenization:** MarianMT tokenizer, max_length=128. Bible verses average ~25-35 words -> ~40-60 tokens. Well within 128.

Steps per epoch = 155,458 / 32 = ~4,858

### 6.3 Time Per Step

MarianMT is a tiny model (74M parameters, 298MB). At batch=32 with 128-token sequences on the A2000 Ada:

- The entire model + optimizer + activations fits in ~2-3 GB VRAM
- Forward + backward pass is extremely fast
- **Estimated: ~0.15-0.25 s/step**
- **Best estimate: ~0.2 s/step**

### 6.4 VRAM Usage

| Component | VRAM |
|-----------|------|
| MarianMT model (bf16) | ~0.6 GB |
| Optimizer states (AdamW) | ~1.2 GB |
| Activations (batch=32, seq=128) | ~0.8-1.2 GB |
| Gradient buffers | ~0.6 GB |
| PyTorch overhead | ~0.5 GB |
| **Total estimated** | **~3.7-4.1 GB** |

Trivially fits. Could increase batch size to 64 or 128 for even faster training.

### 6.5 Total Training Time

At ~0.2 s/step, 4,858 steps/epoch:

| Epochs | Total Steps | Time (raw) | + Eval/Checkpointing (20%) | **Total** |
|--------|------------|------------|---------------------------|-----------|
| 1 | 4,858 | 16.2 min | 19.4 min | **~20 min** |
| 3 | 14,574 | 48.6 min | 58.3 min | **~1.0 hr** |
| 5 | 24,290 | 81.0 min | 97.2 min | **~1.6 hrs** |
| 10 | 48,580 | 162 min | 194.4 min | **~3.2 hrs** |

**Eval overhead:** With `eval_steps=1000` and `predict_with_generate=True`, evaluation every 1000 steps generates translations for 1000 examples. This adds ~2-3 min per eval. At 5 epochs (~25 evals), that's ~50-75 min of eval time.

**Adjusted totals with eval:**

| Epochs | Training | Eval (~25 evals at 5 ep) | **Total** |
|--------|----------|-------------------------|-----------|
| 3 | ~50 min | ~30 min | **~1.3 hrs** |
| 5 | ~81 min | ~50 min | **~2.2 hrs** |
| 10 | ~162 min | ~100 min | **~4.4 hrs** |

MarianMT is extremely fast to iterate on. This makes it an excellent "first pass" model for rapid experimentation before committing to the multi-day TranslateGemma runs.

---

## 7. Evaluation Time

### 7.1 Automatic Metrics

| Metric | Tool | Compute | Time for 3.1K Verses | Notes |
|--------|------|---------|----------------------|-------|
| SacreBLEU | `sacrebleu` | CPU | **<1 min** | N-gram matching, trivially fast |
| chrF++ | `sacrebleu` | CPU | **<1 min** | Character-level, trivially fast |
| COMET | `unbabel-comet` | GPU | **~15-30 min** | Neural metric, needs GPU inference |
| COMETKiwi (QE) | `wmt22-cometkiwi-da` | GPU | **~10-20 min** | Reference-free, slightly lighter |

### 7.2 Whisper WER Evaluation

Evaluate Whisper on held-out sermon segments (~500 segments):

| Step | Time | Notes |
|------|------|-------|
| Inference on 500 segments (~2 hrs audio) | ~10-15 min | Distil-Whisper at ~50x real-time |
| WER computation with `jiwer` | <1 min | CPU, trivially fast |
| **Total** | **~12-16 min** | |

### 7.3 Full Evaluation Suite

| Component | Time |
|-----------|------|
| Translation metrics (BLEU + chrF++ + COMET) on 3.1K verses | ~20-35 min |
| Whisper WER on held-out audio | ~12-16 min |
| Theological term spot-check (~500 terms) | ~5 min (automated) |
| Per-genre stratified analysis | <5 min (post-processing) |
| **Total evaluation** | **~45-60 min** |

---

## 8. End-to-End Timeline Scaling Table

All times in hours. Bible corpus is fixed at ~155K pairs (does not scale with audio hours).

### GPU-Only Tasks (Automated)

| Data Size | Download | Preprocess | Pseudo-label | Whisper LoRA (3 ep) | Whisper LoRA (5 ep) | Gemma 4B QLoRA (3 ep) | Gemma 12B QLoRA (3 ep) | MarianMT (5 ep) | Eval | **Total (all models)** |
|-----------|----------|------------|-------------|--------------------|--------------------|----------------------|----------------------|----------------|------|----------------------|
| 10 hrs | 0.3 | 1.4 | 0.8 | 0.3 | 0.6 | 12 | 26 | 2.2 | 1.0 | **44 hrs** |
| 20 hrs | 0.7 | 2.8 | 1.5 | 0.7 | 1.1 | 12 | 26 | 2.2 | 1.0 | **47 hrs** |
| 30 hrs | 1.0 | 4.2 | 2.2 | 1.0 | 1.7 | 12 | 26 | 2.2 | 1.0 | **50 hrs** |
| 50 hrs | 1.7 | 7.0 | 3.5 | 1.7 | 2.8 | 12 | 26 | 2.2 | 1.0 | **55 hrs** |

**Key insight:** TranslateGemma training dominates total GPU time regardless of audio dataset size, because it uses the fixed 155K Bible corpus. Whisper training and audio preprocessing scale with data but are comparatively fast.

### Practical "Skip 12B" Timeline

If you train only 4B + MarianMT (and skip the expensive 12B):

| Data Size | Download | Preprocess | Pseudo-label | Whisper LoRA (3 ep) | Gemma 4B QLoRA (3 ep) | MarianMT (5 ep) | Eval | **Total** |
|-----------|----------|------------|-------------|--------------------|-----------------------|----------------|------|-----------|
| 10 hrs | 0.3 | 1.4 | 0.8 | 0.3 | 12 | 2.2 | 1.0 | **18 hrs** |
| 20 hrs | 0.7 | 2.8 | 1.5 | 0.7 | 12 | 2.2 | 1.0 | **21 hrs** |
| 30 hrs | 1.0 | 4.2 | 2.2 | 1.0 | 12 | 2.2 | 1.0 | **24 hrs** |
| 50 hrs | 1.7 | 7.0 | 3.5 | 1.7 | 12 | 2.2 | 1.0 | **29 hrs** |

### Including Human Correction Time

Add human correction time from Section 2.5:

| Data Size | GPU Automated (skip 12B) | Human Correction | **Grand Total** |
|-----------|-------------------------|-----------------|----------------|
| 10 hrs | 18 hrs | 8 hrs | **26 hrs** |
| 20 hrs | 21 hrs | 17 hrs | **38 hrs** |
| 30 hrs | 24 hrs | 25 hrs | **49 hrs** |
| 50 hrs | 29 hrs | 42 hrs | **71 hrs** |

---

## 9. Overnight Training Schedule

### 9.1 What Fits in a 10-Hour Overnight Window

| Job | Duration | Fits Overnight? |
|-----|----------|----------------|
| Audio preprocessing (30 hrs raw) | 4-5 hrs | Yes |
| Pseudo-labeling (30 hrs, Whisper large-v3) | 2.2 hrs | Yes |
| Preprocessing + pseudo-labeling together | 6-7 hrs | Yes (sequential, same GPU) |
| Whisper LoRA (30 hrs audio, 5 epochs) | 1.7 hrs | Yes (trivially) |
| MarianMT (5 epochs) | 2.2 hrs | Yes (trivially) |
| TranslateGemma 4B QLoRA (3 epochs) | 12 hrs | Barely -- may extend to morning |
| TranslateGemma 12B QLoRA (3 epochs) | 26 hrs | No -- needs 2.5-3 nights |
| Whisper LoRA + MarianMT + Eval | 5 hrs | Yes (all three in one night) |

### 9.2 Parallelization Opportunities

**CPU and GPU can overlap:**
- While GPU runs demucs source separation, CPU handles ffmpeg conversion, noisereduce, VAD chunking for other files
- While GPU runs training, CPU handles data loading/preprocessing for next batch

**Sequential GPU dependencies:**
- Preprocessing must finish before pseudo-labeling can start
- Pseudo-labeling must finish before Whisper LoRA training (same data)
- Bible corpus preparation (CPU-only) is independent of audio pipeline
- TranslateGemma and Whisper training are independent of each other (different data sources)

### 9.3 Proposed Multi-Day Schedule (30 hrs audio, 4B + MarianMT + 12B)

```
Day 1 (Evening): Start preprocessing + pseudo-labeling overnight
  [GPU] Preprocess 30 hrs audio (4-5 hrs)
  [GPU] Pseudo-label with Whisper large-v3 (2.2 hrs)
  [CPU] Download Bible corpus, run prepare_bible_corpus.py
  -> Total: ~7 hrs, finishes by morning

Day 2 (Morning): Review pseudo-labels, start human correction
  [CPU] Sample 100 segments for quality assessment
  [CPU] Begin human correction of bottom 20% (~25 hrs total, spread over days)

Day 2 (Evening): Start MarianMT + Whisper LoRA overnight
  [GPU] MarianMT full fine-tune, 5 epochs (2.2 hrs)
  [GPU] Whisper LoRA training, 3 epochs (1.0 hr)
  [GPU] Evaluation suite (1.0 hr)
  -> Total: ~4.2 hrs, finishes well before morning

Day 3 (Evening): Start TranslateGemma 4B QLoRA overnight
  [GPU] TranslateGemma 4B QLoRA, 3 epochs (12 hrs)
  -> Finishes by morning or mid-morning

Day 4 (Morning): Evaluate 4B results
  [GPU] Evaluation suite (1 hr)
  [CPU] Compare 4B vs MarianMT vs baseline
  [CPU] Decision: is 12B worth training?

Day 4-6 (If 12B is justified): TranslateGemma 12B QLoRA
  [GPU] Start 12B training (26 hrs = ~2.5 overnight sessions)
  Night 1: Steps 1-5,500 (~10 hrs)
  Night 2: Steps 5,500-11,000 (~10 hrs) -- use --resume
  Night 3: Steps 11,000-12,939 (~6 hrs) -- finishes early
  -> Total: 3 nights

Day 6-7: Final evaluation and adapter transfer
  [GPU] Full evaluation suite
  [CPU] Transfer adapters to Mac
  [Mac] Test inference with fine-tuned models
```

### 9.4 Aggressive Schedule (Skip 12B, Minimize Human Time)

```
Day 1 Evening:  Preprocess + pseudo-label (7 hrs overnight)
Day 2 Evening:  Whisper LoRA + MarianMT + eval (4 hrs overnight)
Day 3 Evening:  TranslateGemma 4B QLoRA (12 hrs overnight)
Day 4 Morning:  Evaluate, transfer adapters to Mac, test

Total: 4 days (3 overnight runs) + human correction in parallel
```

---

## 10. Per-Cycle Time After First Cycle

### 10.1 What Changes in Subsequent Cycles

The active learning feedback loop (infer -> flag -> correct -> retrain) gets faster each cycle:

| Component | Cycle 1 | Cycle 2 | Cycle 3 | Cycle 4-5 |
|-----------|---------|---------|---------|-----------|
| **Re-pseudo-labeling** | N/A (first pass) | 2.2 hrs | 2.0 hrs | 1.8 hrs |
| **Human correction** | 25 hrs (20% of all segs) | 10-15 hrs (flagged only) | 5-8 hrs | 3-5 hrs |
| **Whisper LoRA** (incremental, 2 epochs) | 1.0 hr (3 ep) | 0.7 hr | 0.7 hr | 0.7 hr |
| **TranslateGemma 4B** (incremental, 1-2 epochs) | 12 hrs (3 ep) | 5-8 hrs (1-2 ep) | 4-5 hrs | 3-4 hrs |
| **MarianMT** (incremental, 2 epochs) | 2.2 hrs (5 ep) | 0.9 hr (2 ep) | 0.9 hr | 0.9 hr |
| **Evaluation** | 1.0 hr | 1.0 hr | 1.0 hr | 1.0 hr |
| **Total GPU time** | ~16 hrs | ~10-13 hrs | ~8-10 hrs | ~7-9 hrs |
| **Total with human time** | ~41 hrs | ~20-28 hrs | ~13-18 hrs | ~10-14 hrs |

### 10.2 Why It Gets Faster

1. **Human correction shrinks:** The model improves, so fewer segments are flagged. By cycle 3-4, only 5-10% of segments need review instead of 20%.
2. **Incremental training:** Don't retrain from scratch. Resume from last checkpoint with 1-2 additional epochs on the updated dataset.
3. **Re-pseudo-labeling is optional:** If only corrections change (not new audio), you can skip re-labeling and just update the training data.
4. **Evaluation stabilizes:** Once you know which metrics matter, eval becomes more targeted.

### 10.3 Full 5-Cycle Timeline (30 hrs audio, 4B + MarianMT)

| Cycle | GPU Time | Human Time | Cumulative GPU | Cumulative Human |
|-------|----------|------------|----------------|-----------------|
| 1 | 24 hrs | 25 hrs | 24 hrs | 25 hrs |
| 2 | 10 hrs | 12 hrs | 34 hrs | 37 hrs |
| 3 | 8 hrs | 7 hrs | 42 hrs | 44 hrs |
| 4 | 7 hrs | 4 hrs | 49 hrs | 48 hrs |
| 5 | 7 hrs | 3 hrs | 56 hrs | 51 hrs |
| **Total** | **56 GPU-hrs** | **51 human-hrs** | | |

**Wall-clock time:** ~3-4 weeks with overnight GPU runs and part-time human correction (~2-3 hrs/day).

### 10.4 Expected Quality Improvements Per Cycle

| Cycle | Expected WER Improvement | Expected BLEU Improvement |
|-------|------------------------|--------------------------|
| 1 | 20-40% relative reduction | +3-5 BLEU points |
| 2 | 10-20% relative reduction | +2-3 BLEU points |
| 3 | 5-10% relative reduction | +1-2 BLEU points |
| 4 | 2-5% relative reduction | +0.5-1 BLEU point |
| 5 | Diminishing returns | Diminishing returns |

---

## Appendix A: Comparison with CLAUDE.md Initial Estimates

| Task | CLAUDE.md Estimate | Revised Estimate | Difference | Reason |
|------|-------------------|-----------------|------------|--------|
| Audio preprocessing (50 hrs) | 4-6 hrs | 5.3-8.7 hrs | Higher | More conservative demucs speed estimate |
| VAD segmentation (50 hrs) | ~30 min | ~30 min | Same | Confirmed |
| Pseudo-labeling (50 hrs) | 3-5 hrs | 3.5 hrs | Same range | Confirmed |
| Human correction | 15-25 hrs | 25-42 hrs | Higher | Scaled to actual data sizes |
| Whisper LoRA (20 hrs, 3 ep) | 5-8 hrs | ~0.7 hr | **Much lower** | 8-bit base + distil model = very fast |
| Whisper LoRA (50 hrs, 3 ep) | 11-15 hrs | ~1.7 hrs | **Much lower** | Same reason |
| TranslateGemma 4B (3 ep) | 8-12 hrs | 10-14 hrs | Slightly higher | all-linear + modules_to_save overhead |
| TranslateGemma 12B (3 ep) | 18-27 hrs | 22-30 hrs | Higher | Memory pressure, conservative estimate |
| Evaluation | 30-60 min | 45-60 min | Same range | Confirmed |

**The biggest revision is Whisper LoRA training time**, which is dramatically faster than initially estimated. This is because:
1. Distil-Whisper has ~50% fewer decoder layers than full Whisper-large
2. 8-bit quantization of the base model reduces memory and compute
3. LoRA on just q_proj + v_proj is a very small set of trainable parameters
4. Audio segments average ~15s (not 30s), reducing per-step compute

---

## Appendix B: Risk Factors and Mitigations

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| TranslateGemma 12B OOM | 30% | Training fails | Reduce seq_length, use 8-bit optimizer |
| Demucs slower than estimated | 20% | Preprocessing takes 2x longer | Run overnight, skip demucs for initial tests |
| Whisper pseudo-labels too noisy | 15% | Poor training labels | Use faster-whisper with confidence filtering |
| Bible corpus packing less efficient | 25% | 20-30% more training steps | Adjust estimated time upward |
| CUDA/bitsandbytes version conflicts | 20% | Hours of debugging | Pin versions in requirements.txt |
| WSL2 GPU passthrough issues | 10% | Cannot use GPU at all | Dual-boot Ubuntu as fallback |
| Disk I/O bottleneck in WSL2 | 15% | 10-20% slower training | Store data on Windows partition, not WSL filesystem |

---

## Appendix C: Quick Reference Commands

```bash
# Preprocessing (overnight job)
nohup python preprocess_audio.py --input stark_data/raw --output stark_data/cleaned --resume > preprocess.log 2>&1 &

# Pseudo-labeling (overnight job, after preprocessing)
nohup python transcribe_church.py --backend faster-whisper --resume > transcribe.log 2>&1 &

# Whisper LoRA (quick job, ~1 hr)
nohup python train_whisper.py --epochs 3 --dataset stark_data/cleaned > whisper_train.log 2>&1 &

# MarianMT (quick job, ~2 hrs)
nohup python train_marian.py --epochs 5 > marian_train.log 2>&1 &

# TranslateGemma 4B (overnight job, ~12 hrs)
nohup python train_gemma.py A --epochs 3 > gemma4b_train.log 2>&1 &

# TranslateGemma 12B (multi-night job, ~26 hrs, use tmux)
tmux new-session -s gemma12b
python train_gemma.py B --epochs 3
# If interrupted: python train_gemma.py B --epochs 3 --resume

# Monitor GPU usage
watch -n 1 nvidia-smi
```

---

## Appendix D: Sources and Benchmarks Referenced

- [Whisper GPU performance benchmarks (GitHub Discussion #918)](https://github.com/openai/whisper/discussions/918)
- [Whisper Large v3 benchmark on SaladCloud](https://blog.salad.com/whisper-large-v3/)
- [QLoRA fine-tuning efficiency on consumer GPUs (arXiv:2509.12229)](https://arxiv.org/html/2509.12229v1)
- [faster-whisper: CTranslate2-based Whisper](https://github.com/SYSTRAN/faster-whisper)
- [Demucs processing speed and GPU acceleration](https://github.com/CarlGao4/Demucs-Gui/blob/main/usage.md)
- [NVIDIA RTX 2000 Ada datasheet](https://resources.nvidia.com/en-us-briefcase-for-datasheets/proviz-rtx-2000-ada)
- [RTX A2000 vs RTX 3060 benchmarks (BIZON)](https://bizon-tech.com/gpu-benchmarks/NVIDIA-RTX-3060-vs-NVIDIA-RTX-A2000/590vs631)
- [Fine-tuning Distil-Whisper with LoRA (arXiv:2503.22692)](https://arxiv.org/pdf/2503.22692)
- [Whisper PEFT/INT8 training on T4 (GitHub Discussion #988)](https://github.com/openai/whisper/discussions/988)
- [Fine-tune Whisper with LoRA on SageMaker (AWS Blog)](https://aws.amazon.com/blogs/machine-learning/fine-tune-whisper-models-on-amazon-sagemaker-with-lora/)
- [OpenAI Whisper benchmarked on 18 GPUs (Tom's Hardware)](https://www.tomshardware.com/news/whisper-audio-transcription-gpus-benchmarked)
