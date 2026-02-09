# RTX 2070 Inference Endpoint Feasibility Study

**Project:** SRTranslate -- Live Bilingual Speech-to-Text for Church Deployments
**Date:** February 2026
**Purpose:** Evaluate the NVIDIA GeForce RTX 2070 as a low-cost inference endpoint for deploying the SRTranslate pipeline at church sites beyond the primary MacBook development machine.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [RTX 2070 Hardware Profile](#2-rtx-2070-hardware-profile)
3. [Pipeline Component Analysis](#3-pipeline-component-analysis)
   - 3.1 [Silero VAD](#31-silero-vad)
   - 3.2 [Distil-Whisper large-v3 (STT)](#32-distil-whisper-large-v3-stt)
   - 3.3 [TranslateGemma 4B (Translation -- Approach A)](#33-translategemma-4b-translation--approach-a)
   - 3.4 [TranslateGemma 12B (Translation -- Approach B)](#34-translategemma-12b-translation--approach-b)
   - 3.5 [MarianMT (Translation -- Fallback)](#35-marianmt-translation--fallback)
4. [VRAM Budget Analysis](#4-vram-budget-analysis)
5. [Framework Selection for NVIDIA Inference](#5-framework-selection-for-nvidia-inference)
6. [Expected Latency Estimates](#6-expected-latency-estimates)
7. [M3 Pro MLX vs. RTX 2070 CUDA Comparison](#7-m3-pro-mlx-vs-rtx-2070-cuda-comparison)
8. [Recommended Deployment Configurations](#8-recommended-deployment-configurations)
9. [System Build and Cost Estimate](#9-system-build-and-cost-estimate)
10. [Risks and Mitigations](#10-risks-and-mitigations)
11. [Conclusion and Recommendation](#11-conclusion-and-recommendation)
12. [Sources](#12-sources)

---

## 1. Executive Summary

**Verdict: FEASIBLE with constraints.** The RTX 2070 (8 GB VRAM) can run the full SRTranslate inference pipeline -- VAD + Distil-Whisper STT + TranslateGemma 4B translation -- within its 8 GB VRAM envelope when using optimized frameworks and quantization. The 12B translation model does NOT fit alongside Whisper in 8 GB. A complete church deployment endpoint (used RTX 2070 + host PC) can be assembled for approximately $350--500, making it a cost-effective alternative to per-site MacBook deployments.

**Key findings:**

| Question | Answer |
|----------|--------|
| Full pipeline (VAD + STT + 4B) in 8 GB? | **Yes** -- ~5.8--6.5 GB peak with int8 Whisper + 4-bit Gemma 4B |
| 4B + 12B simultaneously? | **No** -- 12B alone requires ~7 GB in 4-bit, leaving no room for Whisper |
| Expected end-to-end latency? | **~300--600 ms** (VAD + STT + translation) for a 5-second audio chunk |
| Cost per endpoint? | **~$350--500** (used GPU + budget host PC) |
| MLX replacement framework? | **faster-whisper (CTranslate2)** for STT + **exllamav2** or **llama.cpp CUDA** for translation |

---

## 2. RTX 2070 Hardware Profile

| Specification | RTX 2070 | RTX 2070 Super | Notes |
|---------------|----------|----------------|-------|
| Architecture | Turing (TU106) | Turing (TU104) | Compute capability 7.5 |
| CUDA Cores | 2,304 | 2,560 | +11% on Super |
| Tensor Cores | 288 (Gen 2) | 320 (Gen 2) | FP16, INT8, INT4 support |
| VRAM | 8 GB GDDR6 | 8 GB GDDR6 | Same capacity |
| Memory Bandwidth | 448 GB/s | 448 GB/s | Identical |
| FP32 TFLOPS | 7.5 | 9.1 | |
| FP16 TFLOPS | ~15 | ~18.2 | Turing has 2x FP16 on dedicated cores |
| INT8 TOPS | ~30 | ~36.4 | Via Tensor Cores |
| TDP | 175 W | 215 W | Super draws 23% more power |
| PCIe | 3.0 x16 | 3.0 x16 | ~16 GB/s host-device bandwidth |
| Idle Power | ~10--12 W | ~9--11 W | Measured, varies by vendor |

**Key architectural notes:**

- **Turing has dedicated FP16 hardware** that runs at 2x the FP32 rate -- unlike Ampere/Ada which share FP16 on FP32 cores. This is a genuine advantage for FP16 inference workloads like Whisper.
- **INT8 Tensor Cores** are natively supported, enabling CTranslate2 int8 inference without emulation.
- **No BF16 or FP8 support.** BF16 operations fall back to FP32. This means bitsandbytes NF4 with BF16 compute dtype will be slower than on Ampere+ GPUs; use FP16 compute dtype instead.
- **448 GB/s memory bandwidth** is the primary bottleneck for LLM token generation (memory-bound). For context, the RTX 4090 offers 1,008 GB/s -- roughly 2.25x the bandwidth. LLM token generation speed scales approximately linearly with memory bandwidth, so the RTX 2070 will generate tokens at roughly 40--45% the speed of an RTX 4090 all else being equal.

---

## 3. Pipeline Component Analysis

### 3.1 Silero VAD

| Metric | Value |
|--------|-------|
| Model Size | ~2 MB (v5) |
| VRAM Usage | 0 MB (CPU-only) |
| Inference Speed | < 1 ms per 30 ms audio chunk on a single CPU thread |
| Framework | PyTorch JIT or ONNX |

**Assessment: Trivial.** Silero VAD is designed to run on CPU. It processes audio chunks in sub-millisecond time and consumes no GPU memory. The v5 release is 3x faster than v4 for TorchScript and runs at well over 100x real-time. No adaptation needed for RTX 2070 deployment -- it simply runs on the host CPU.

### 3.2 Distil-Whisper large-v3 (STT)

Distil-Whisper large-v3 is a 756M parameter model (vs. 1.55B for full Whisper large-v3), achieving 6.3x faster inference than the full model with < 1% WER degradation.

| Configuration | Framework | VRAM Usage | Notes |
|---------------|-----------|------------|-------|
| FP16 | faster-whisper (CTranslate2) | ~1.5 GB | Baseline, good accuracy |
| INT8_FLOAT16 | faster-whisper (CTranslate2) | ~0.9--1.0 GB | Best balance for 8 GB cards |
| INT8 | faster-whisper (CTranslate2) | ~0.8 GB | Slightly lower accuracy |
| FP16 (full large-v3) | faster-whisper | ~2.9--3.0 GB | For reference only |
| INT8 (full large-v3) | faster-whisper | ~1.5 GB | Viable if distil not available |

**Benchmark reference data** (faster-whisper large-v2 on RTX 3070 Ti 8 GB, CUDA 12.4):

| Configuration | Transcription Time (13 min audio) | VRAM |
|---------------|-------------------------------------|------|
| FP16, batch 1 | 63 s | 4,525 MB |
| FP16, batch 8 | 17 s | 6,090 MB |
| INT8, batch 1 | 59 s | 2,926 MB |
| INT8, batch 8 | 16 s | 4,500 MB |

The distil variant is roughly half the parameter count of large-v2, so VRAM usage will be proportionally lower. For live streaming with batch size 1, INT8 Distil-Whisper large-v3 should require approximately **0.8--1.0 GB VRAM** on an RTX 2070.

**Real-time factor estimate:** On the RTX 2080 Ti (11 GB, same Turing architecture), Whisper large-v2 FP16 processes 16 minutes of audio (971 seconds) in 255 seconds -- roughly 3.8x real-time. The distil variant is 6.3x faster, yielding an estimated **~24x real-time** on the 2080 Ti. The RTX 2070 has fewer CUDA cores (~74% of the 2080 Ti), so estimate **~18x real-time** -- meaning a 5-second audio chunk processes in roughly **280 ms**. With INT8 and CTranslate2 optimizations, this could drop to **150--250 ms**.

**TensorRT optimization:** NVIDIA's [whisper_trt](https://github.com/NVIDIA-AI-IOT/whisper_trt) project shows ~3x speedup over PyTorch on Jetson Orin. On the RTX 2070, TensorRT could bring processing time down further, but the conversion pipeline is more complex. Faster-whisper (CTranslate2) provides the best effort-to-performance ratio for initial deployment.

### 3.3 TranslateGemma 4B (Translation -- Approach A)

TranslateGemma 4B has approximately 4 billion parameters. In 4-bit quantization:

| Quantization Method | Model Size (Disk) | VRAM (Model Only) | VRAM (+ KV Cache) | Framework |
|--------------------|--------------------|--------------------|--------------------|-----------|
| bitsandbytes NF4 | ~2.5 GB | ~2.6 GB | ~3.0--3.5 GB | transformers + bnb |
| GPTQ 4-bit | ~2.3 GB | ~2.4 GB | ~2.8--3.3 GB | exllamav2, AutoGPTQ |
| AWQ 4-bit | ~2.3 GB | ~2.4 GB | ~2.8--3.3 GB | vLLM, AutoAWQ |
| GGUF Q4_K_M | ~2.5 GB | ~2.6 GB | ~3.0--3.5 GB | llama.cpp |
| EXL2 4.0 bpw | ~2.0 GB | ~2.2 GB | ~3.0--3.5 GB | exllamav2 |

**KV cache note:** Bible verses and sermon sentences are short (typically < 100 tokens input, < 150 tokens output). The KV cache overhead for TranslateGemma at these sequence lengths is modest -- roughly 0.4--0.8 GB depending on context size and quantization. This is well within the VRAM budget.

**Expected performance on RTX 2070:**

Based on available benchmarks for comparable 4B-class models on 8 GB Turing GPUs:

| Metric | Estimate | Basis |
|--------|----------|-------|
| Prompt processing | ~500--1,000 tokens/s | Extrapolated from 7B benchmarks at ~65 tok/s generation, prompt is compute-bound |
| Token generation | ~50--80 tokens/s | RTX 2070 Super achieves ~65 tok/s on 7B models; 4B should be faster |
| Translation latency (30-token output) | ~400--600 ms | 30 tokens / 60 tok/s = 500 ms |

**Framework comparison for 4B inference on RTX 2070:**

| Framework | Speed | VRAM Efficiency | Ease of Use | Turing Support |
|-----------|-------|-----------------|-------------|----------------|
| **exllamav2** (EXL2) | Fastest (up to 85% faster than bnb) | Good | Moderate | Yes |
| **llama.cpp** (GGUF) | Fast | Best | Easy (Ollama) | Yes |
| **bitsandbytes** (NF4) | Slowest | Good | Easiest | Yes (CUDA 11.8+) |
| **vLLM** (AWQ/GPTQ) | Very fast (batched) | Moderate | Moderate | Yes (SM 7.5+) |
| **TensorRT-LLM** | Fastest (optimized) | Good | Complex setup | Yes (SM 7.5+) |

**Recommendation:** Use **exllamav2** with EXL2 quantization for maximum single-request speed, or **llama.cpp** with GGUF Q4_K_M for simplicity and broad compatibility. Both natively support the RTX 2070's compute capability 7.5.

### 3.4 TranslateGemma 12B (Translation -- Approach B)

TranslateGemma 12B has approximately 12 billion parameters.

| Quantization | Model Size (Disk) | VRAM (Model Only) | VRAM (+ KV Cache) |
|-------------|--------------------|--------------------|---------------------|
| 4-bit (NF4/GPTQ/AWQ) | ~6.6 GB | ~7.0 GB | ~7.5--8.0 GB |
| 3-bit (EXL2 3.0 bpw) | ~4.5 GB | ~5.0 GB | ~5.5--6.5 GB |
| 2-bit (extreme) | ~3.5 GB | ~4.0 GB | ~4.5--5.5 GB |

**Assessment: NOT FEASIBLE for co-resident operation.**

At 4-bit quantization, the 12B model alone consumes ~7.0--7.5 GB of the 8 GB VRAM budget. Whisper needs at minimum ~0.8 GB (INT8 distil). Together: **~7.8--8.3 GB** -- exceeding the 8 GB physical limit before accounting for CUDA context overhead (~300--500 MB), PyTorch allocator fragmentation, and the operating system's GPU memory reservation.

**Can the 12B model run at all on the RTX 2070?**

In isolation (without Whisper loaded), yes -- but only with aggressive 4-bit quantization and short context lengths. It would need to be loaded/unloaded between Whisper runs, which is impractical for a real-time pipeline.

**3-bit quantization as a Hail Mary:** At 3.0 bits-per-weight (EXL2), the 12B model drops to ~5.0 GB, potentially leaving room for INT8 Whisper (~0.8 GB) at ~5.8 GB total. However:
- 3-bit quantization degrades translation quality significantly (perplexity increases 15--25%)
- Translation accuracy is the entire point of the 12B model over 4B
- This defeats the purpose of using the larger model

**Verdict: Use 4B on the RTX 2070. Reserve 12B for machines with 12+ GB VRAM (RTX 3060 12 GB, RTX 3080, RTX 4060 Ti 16 GB, or the M3 Pro MacBook).**

### 3.5 MarianMT (Translation -- Fallback)

Helsinki-NLP/opus-mt-en-es is a compact encoder-decoder translation model.

| Configuration | Model Size | VRAM | Inference Speed |
|---------------|------------|------|-----------------|
| FP32 (PyTorch) | ~298 MB | ~400 MB | Fast |
| FP16 (PyTorch) | ~150 MB | ~250 MB | Faster |
| INT8 (CTranslate2) | ~76 MB | ~150--200 MB | ~7,500 tokens/s |
| FP16 (CTranslate2) | ~150 MB | ~250 MB | ~5,000 tokens/s |

**Assessment: Trivially fits on any GPU.** CTranslate2 natively supports MarianMT/OPUS-MT models and provides both INT8 and FP16 inference. At < 200 MB VRAM, MarianMT leaves maximum headroom for Whisper. Translation latency for a typical sentence (20--40 tokens) would be **< 10 ms** on the RTX 2070.

The tradeoff is quality: MarianMT has a lower translation quality ceiling than TranslateGemma, particularly for theological vocabulary and context-dependent disambiguation. However, it makes an excellent fallback for resource-constrained deployments or as a secondary translation signal.

---

## 4. VRAM Budget Analysis

### Configuration A: Distil-Whisper INT8 + TranslateGemma 4B (4-bit)

| Component | VRAM | Notes |
|-----------|------|-------|
| CUDA context + driver | ~300--500 MB | Allocated on first CUDA call |
| Silero VAD | 0 MB | CPU only |
| Distil-Whisper large-v3 (INT8) | ~800--1,000 MB | via faster-whisper/CTranslate2 |
| TranslateGemma 4B (EXL2 4.0bpw) | ~2,200--2,500 MB | via exllamav2 |
| TranslateGemma 4B KV cache | ~400--800 MB | Short sequences, < 256 tokens |
| Whisper audio buffer/features | ~100--200 MB | 30-second Mel spectrogram + decoder state |
| **Total** | **~3,800--5,000 MB** | **47--63% of 8 GB** |

**Headroom: 3.0--4.2 GB remaining.** This is comfortable. There is room for:
- Batch processing (processing multiple audio chunks)
- Larger KV cache if needed
- Operating system GPU memory pressure
- PyTorch memory fragmentation

### Configuration B: Distil-Whisper INT8 + TranslateGemma 12B (4-bit)

| Component | VRAM | Notes |
|-----------|------|-------|
| CUDA context + driver | ~300--500 MB | |
| Distil-Whisper large-v3 (INT8) | ~800--1,000 MB | |
| TranslateGemma 12B (GPTQ 4-bit) | ~7,000--7,500 MB | |
| KV cache (12B) | ~500--1,000 MB | |
| **Total** | **~8,600--10,000 MB** | **107--125% of 8 GB** |

**Verdict: Does NOT fit.** Even with the most aggressive quantization, the 12B model plus Whisper exceeds the 8 GB envelope by at least 600 MB. Partial CPU offloading would destroy real-time latency.

### Configuration C: Distil-Whisper INT8 + MarianMT INT8

| Component | VRAM | Notes |
|-----------|------|-------|
| CUDA context + driver | ~300--500 MB | |
| Distil-Whisper large-v3 (INT8) | ~800--1,000 MB | |
| MarianMT en-es (INT8 CT2) | ~150--200 MB | |
| Working memory | ~100--200 MB | |
| **Total** | **~1,350--1,900 MB** | **17--24% of 8 GB** |

**Headroom: 6+ GB remaining.** This is the most conservative configuration. MarianMT adds negligible overhead. Ideal for resource-constrained machines or when translation quality requirements are modest.

### Configuration D (Optimal): Sequential Model Loading

An alternative strategy loads Whisper and TranslateGemma sequentially rather than simultaneously:

1. Load Distil-Whisper, process audio chunk, extract English text
2. Unload Whisper, load TranslateGemma 4B
3. Translate English to Spanish
4. Unload TranslateGemma, reload Whisper for next chunk

This approach allows even the 12B model to run (in isolation) but introduces model swap latency of ~1--3 seconds per direction. For a live real-time pipeline, this is **not acceptable** -- the audience display would stall during model swaps. Sequential loading is only viable for batch/post-service processing.

---

## 5. Framework Selection for NVIDIA Inference

MLX is Apple Silicon-only. The RTX 2070 deployment needs CUDA-native alternatives.

### STT Framework: faster-whisper (CTranslate2)

**Recommendation: faster-whisper** is the clear choice for Whisper inference on NVIDIA GPUs.

| Feature | faster-whisper | whisper.cpp | TensorRT (whisper_trt) |
|---------|---------------|-------------|------------------------|
| Speed vs. OpenAI Whisper | 4--6x faster | 2--3x faster | ~6--10x faster |
| VRAM efficiency | Excellent (INT8) | Good | Excellent |
| Ease of setup | `pip install` | CMake build | Complex engine build |
| Distil-Whisper support | Native | Partial | Manual conversion |
| Word-level timestamps | Yes | Yes | Limited |
| INT8 quantization | Yes, native | Yes | Yes |
| Turing (SM 7.5) support | Yes | Yes | Yes |
| Streaming/chunked mode | Yes | Yes | Yes |
| Python API | Yes | Via bindings | Via triton/custom |

faster-whisper wraps CTranslate2, which has mature CUDA support and handles INT8 quantization transparently. The `int8_float16` compute type is ideal for Turing: INT8 weights with FP16 accumulation, leveraging the RTX 2070's native INT8 Tensor Core support.

### Translation Framework Decision Matrix

| Framework | Best For | VRAM Overhead | Speed (single request) | Turing Support | Setup Complexity |
|-----------|---------|---------------|----------------------|----------------|------------------|
| **exllamav2** | Maximum tok/s, EXL2 quant | Low | Fastest (with Flash Attn) | Yes | Moderate |
| **llama.cpp (GGUF)** | Simplicity, Ollama wrapper | Lowest | Fast | Yes | Easy |
| **bitsandbytes (NF4)** | Quick prototyping | Moderate | Slowest | Yes | Easiest |
| **vLLM (AWQ/GPTQ)** | Multi-request batching | Higher | Fast (batched) | Yes (SM 7.5+) | Moderate |
| **TensorRT-LLM** | Maximum throughput | Low | Fastest (compiled) | Yes (SM 7.5+) | High |

**Primary recommendation: exllamav2** with EXL2 4.0 bpw quantization.

Rationale:
- ExLlamaV2 generates ~85% more tokens/second than bitsandbytes and ~47% more than llama.cpp in comparable benchmarks on the same hardware
- Flash Attention support gives significant speedups for prompt processing
- EXL2 format offers fine-grained bits-per-weight control (can tune between 3.0 and 5.0 bpw)
- Native CUDA kernel optimization for consumer GPUs
- Lower VRAM overhead than vLLM for single-user scenarios

**Fallback recommendation: llama.cpp** via Ollama for simpler deployment.

Rationale:
- Single-binary deployment via Ollama
- GGUF Q4_K_M offers excellent quality-to-size ratio
- No Python dependency management needed
- REST API for easy integration
- Better community support and model availability

**Note on TensorRT-LLM:** While TensorRT-LLM supports SM 7.5+ and can deliver the best absolute performance, it requires model-specific engine compilation and has a complex setup process. The performance gain over exllamav2 may not justify the engineering effort for a church deployment scenario. However, it remains an option for future optimization.

**Note on bitsandbytes:** While bitsandbytes is the easiest to set up (just add `load_in_4bit=True` to the transformers model loading call), it is the slowest option for inference. On Turing GPUs specifically, bitsandbytes NF4 with BF16 compute falls back to FP32 (since Turing lacks native BF16). Use `bnb_4bit_compute_dtype=torch.float16` to avoid this penalty.

---

## 6. Expected Latency Estimates

Latency estimates for a typical pipeline run processing a 5-second audio chunk containing a single sentence (~10--15 English words, translating to ~15--20 Spanish words).

### Per-Component Latency

| Component | Framework | Estimated Latency | Confidence |
|-----------|-----------|-------------------|------------|
| **Silero VAD** | PyTorch CPU | < 5 ms | High (measured) |
| **Audio preprocessing** | NumPy/SciPy | ~5--10 ms | High |
| **Distil-Whisper (INT8)** | faster-whisper | ~150--300 ms | Medium (extrapolated) |
| **TranslateGemma 4B (4-bit)** | exllamav2 | ~300--600 ms | Medium (extrapolated) |
| **TranslateGemma 4B (4-bit)** | llama.cpp | ~400--750 ms | Medium (extrapolated) |
| **MarianMT (INT8)** | CTranslate2 | ~5--15 ms | High (measured at ~7.5K tok/s) |

### End-to-End Pipeline Latency

| Configuration | Total Latency | Meets < 500ms Target? |
|---------------|---------------|----------------------|
| VAD + Whisper + MarianMT | ~160--330 ms | **Yes** |
| VAD + Whisper + Gemma 4B (exllamav2) | ~460--910 ms | **Partially** (median likely ~600 ms) |
| VAD + Whisper + Gemma 4B (llama.cpp) | ~560--1,060 ms | **No** (median likely ~750 ms) |

### Latency Breakdown Methodology

**Whisper estimate basis:** The RTX 2080 Ti processes Whisper large-v2 FP16 in 255 seconds for 971 seconds of audio (~3.8x real-time). Distil-Whisper is ~6.3x faster. Adjusting for RTX 2070 having ~74% of the 2080 Ti's CUDA cores: `5s / (3.8 * 6.3 * 0.74) = ~280 ms` for FP16. INT8 provides an additional ~10--20% speedup on Turing Tensor Cores, yielding ~230 ms. Rounding conservatively: 150--300 ms.

**TranslateGemma 4B estimate basis:** RTX 2070 Super benchmarks show ~65 tok/s for Mistral 7B (larger model). A 4B model should decode faster; estimate ~60--80 tok/s. For a 20-token Spanish output: `20 / 70 = ~285 ms` decode time, plus ~100--200 ms prompt processing. Total: ~400--500 ms typical, with variance to 600 ms.

**Note:** These are estimates extrapolated from available benchmark data on similar hardware. Actual performance should be validated with on-device benchmarks before committing to deployment.

---

## 7. M3 Pro MLX vs. RTX 2070 CUDA Comparison

| Dimension | M3 Pro (18 GB, MLX) | RTX 2070 (8 GB, CUDA) |
|-----------|---------------------|------------------------|
| **Unified Memory** | 18 GB shared CPU/GPU | 8 GB dedicated VRAM + system RAM |
| **Memory Bandwidth** | 150 GB/s | 448 GB/s |
| **FP16 TFLOPS** | ~7 (18-core GPU) | ~15 |
| **INT8 TOPS** | N/A (MLX uses FP16/4-bit) | ~30 |
| **Power (inference load)** | ~15--25 W (full SoC) | ~120--175 W (GPU only) |
| **Noise** | Fanless / near-silent | Active cooling required |
| **Form Factor** | Laptop | Desktop PCIe card |

### Model Fit Comparison

| Model | M3 Pro (18 GB) | RTX 2070 (8 GB) |
|-------|----------------|-----------------|
| Distil-Whisper large-v3 | 4-bit MLX (~1.2 GB) | INT8 CT2 (~0.8--1.0 GB) |
| TranslateGemma 4B | 4-bit MLX (~2.5 GB) | 4-bit EXL2 (~2.2--2.5 GB) |
| TranslateGemma 12B | 4-bit MLX (~7.0 GB) | **Does not fit alongside Whisper** |
| Both 4B + 12B simultaneously | Yes (~9 GB peak) | **No** |
| Full pipeline peak VRAM | ~4--5 GB | ~4--5 GB (4B config) |

### Measured vs. Estimated Latency

| Component | M3 Pro MLX (measured) | RTX 2070 CUDA (estimated) |
|-----------|-----------------------|---------------------------|
| Distil-Whisper (5s audio) | ~70--210 ms partials | ~150--300 ms |
| TranslateGemma 4B | ~650 ms | ~400--600 ms |
| TranslateGemma 12B | ~1,400 ms | N/A (does not fit) |
| MarianMT | ~10--30 ms | ~5--15 ms |
| End-to-end (VAD+STT+4B) | ~800--900 ms | ~500--900 ms |
| End-to-end (VAD+STT+MarianMT) | ~100--250 ms | ~160--330 ms |

**Key observations:**

1. **Memory bandwidth advantage:** The RTX 2070's 448 GB/s bandwidth is 3x the M3 Pro's 150 GB/s. Since LLM token generation is memory-bandwidth-bound, the RTX 2070 should generate tokens ~2--3x faster for the same model at the same quantization level. This is why the 4B translation estimate is actually faster on the RTX 2070 than the measured M3 Pro time.

2. **Whisper is compute-bound, not memory-bound.** The encoder's convolution and attention layers are compute-intensive. Here the M3 Pro's lower raw TFLOPS hurts less because MLX's Metal backend is well-optimized, while CTranslate2's CUDA INT8 kernels leverage the RTX 2070's Tensor Cores effectively. The result is roughly comparable STT latency.

3. **The M3 Pro's advantage is capacity.** 18 GB unified memory means both the 4B and 12B translation models can coexist with Whisper. The RTX 2070 cannot replicate the Approach B (12B) configuration.

4. **Power and noise.** The MacBook runs silently at ~20 W during inference. The RTX 2070 system will draw ~120--175 W under load with active fan cooling. For a church sound booth, noise management matters.

---

## 8. Recommended Deployment Configurations

### Tier 1: Best Quality (RTX 2070 Budget)

**Configuration:** Distil-Whisper large-v3 (INT8) + TranslateGemma 4B (EXL2 4.0 bpw)

| Parameter | Value |
|-----------|-------|
| STT framework | faster-whisper, `compute_type="int8_float16"` |
| Translation framework | exllamav2, EXL2 4.0 bpw quantization |
| VRAM usage | ~4.5--5.5 GB peak |
| End-to-end latency | ~500--900 ms |
| Translation quality | Good (BLEU ~38--42 baseline, higher post-fine-tune) |

**Pros:** Best translation quality available within 8 GB. TranslateGemma 4B handles theological vocabulary and context-dependent disambiguation better than MarianMT.

**Cons:** Translation latency occasionally exceeds 500 ms. Requires EXL2 model conversion (one-time cost). More complex dependency management (Python + exllamav2 + faster-whisper).

### Tier 2: Lowest Latency (RTX 2070 Budget)

**Configuration:** Distil-Whisper large-v3 (INT8) + MarianMT en-es (INT8)

| Parameter | Value |
|-----------|-------|
| STT framework | faster-whisper, `compute_type="int8_float16"` |
| Translation framework | CTranslate2 (same engine as faster-whisper) |
| VRAM usage | ~1.5--2.0 GB peak |
| End-to-end latency | ~160--330 ms |
| Translation quality | Adequate (BLEU ~32--36 baseline) |

**Pros:** Sub-500 ms guaranteed. Minimal VRAM usage leaves room for other services. Both STT and translation use CTranslate2, simplifying the dependency stack. Single `pip install ctranslate2 faster-whisper` covers both.

**Cons:** Lower translation quality ceiling. Worse theological term accuracy. No fine-grained control over translation style.

### Tier 3: Simplified Deployment

**Configuration:** Distil-Whisper large-v3 (GGUF/llama.cpp) + TranslateGemma 4B (GGUF/Ollama)

| Parameter | Value |
|-----------|-------|
| STT framework | faster-whisper (still best for Whisper) |
| Translation framework | Ollama (wraps llama.cpp) |
| VRAM usage | ~4.0--5.0 GB peak |
| End-to-end latency | ~600--1,100 ms |
| Translation quality | Good |

**Pros:** Ollama provides a simple REST API and automatic GGUF model management. Easy to update models without redeploying code. Lower engineering complexity for volunteers maintaining church systems.

**Cons:** ~20% slower than exllamav2 for token generation. Ollama adds a small HTTP overhead (~5--10 ms per request). Less fine-grained control over quantization parameters.

### Recommendation

**For initial deployment: Tier 2 (MarianMT)** -- Start with the simplest, fastest configuration. MarianMT's translation quality is adequate for live display and can be fine-tuned on the biblical parallel corpus. Once the deployment is stable, upgrade to Tier 1 (TranslateGemma 4B) for better theological accuracy.

**For production: Tier 1 (TranslateGemma 4B via exllamav2)** -- The quality improvement of TranslateGemma 4B over MarianMT is significant for theological content, and the latency is acceptable for live captioning (most audiences tolerate up to ~1 second delay for translated subtitles).

---

## 9. System Build and Cost Estimate

### Used RTX 2070 Pricing (February 2026)

| Source | RTX 2070 | RTX 2070 Super | Notes |
|--------|----------|----------------|-------|
| eBay (average) | ~$140--160 | ~$165--200 | Varies by condition and seller |
| Best Value GPU tracker | ~$146 | ~$163 | 30-day rolling average |
| Price trend | Declining | Stable | Down ~37% from Nov 2022 |

The RTX 2070 Super offers ~11% more CUDA cores for ~$20--40 more. Worth the premium if available.

### Complete Endpoint Build

| Component | Budget Option | Mid-Range Option | Notes |
|-----------|--------------|-------------------|-------|
| **GPU** | Used RTX 2070 ($140) | Used RTX 2070 Super ($175) | eBay / local marketplace |
| **CPU** | Used i5-8400 / Ryzen 5 2600 ($30--40) | Used i5-10400 / Ryzen 5 3600 ($50--70) | Any 6-core with PCIe 3.0 |
| **Motherboard** | Used mATX B360/B450 ($30--50) | Used ATX B460/B550 ($40--60) | PCIe 3.0 x16 slot required |
| **RAM** | 16 GB DDR4 used ($15--25) | 32 GB DDR4 used ($30--40) | 16 GB sufficient for inference |
| **Storage** | 256 GB SATA SSD ($15--20) | 500 GB NVMe SSD ($25--35) | Models fit in < 10 GB |
| **PSU** | 500 W 80+ Bronze used ($20--30) | 550 W 80+ Gold ($35--50) | RTX 2070 TDP = 175 W |
| **Case** | Basic mATX ($20--30) | Quiet case w/ dampening ($40--60) | Sound booth consideration |
| **OS** | Ubuntu 22.04 LTS (free) | Ubuntu 22.04 LTS (free) | CUDA 12.x support |
| **Total** | **~$270--335** | **~$395--490** | |

### Recurring Costs

| Item | Monthly Cost |
|------|-------------|
| Electricity (175 W * 4 hrs/week * 4.3 weeks) | ~$0.50--1.00 |
| Internet (already available at church) | $0 |
| Software updates | $0 (all open-source) |
| **Total recurring** | **< $1/month** |

### Cost Comparison

| Deployment Option | Upfront Cost | Capable of 12B? | Portability | Noise |
|-------------------|-------------|------------------|-------------|-------|
| RTX 2070 endpoint (budget) | ~$300 | No | Low (desktop) | Moderate |
| RTX 2070 endpoint (mid-range) | ~$450 | No | Low (desktop) | Low--Moderate |
| Used M1 MacBook Air 16 GB | ~$500--600 | Tight (MLX) | High | Silent |
| Used M2 MacBook Air 16 GB | ~$700--800 | Yes (MLX) | High | Silent |
| M3 Pro MacBook 18 GB (current dev) | ~$1,800 (already owned) | Yes (MLX) | High | Silent |
| RTX 3060 12 GB endpoint | ~$350--500 | Maybe (tight) | Low | Moderate |

---

## 10. Risks and Mitigations

### Risk 1: VRAM Fragmentation Under Load

**Risk:** PyTorch/CUDA memory allocators can fragment VRAM over time, causing OOM errors even when total allocation is within budget.

**Mitigation:**
- Use exllamav2 or llama.cpp, which manage their own memory pools and avoid PyTorch's allocator
- If using PyTorch (bitsandbytes), set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- Implement a watchdog that restarts the inference process if VRAM usage exceeds 7.0 GB
- Pre-allocate model memory at startup, avoid dynamic allocation during inference

### Risk 2: Turing-Specific Compatibility Issues

**Risk:** Some frameworks optimize primarily for Ampere/Ada (SM 8.0+) and may have suboptimal or broken codepaths for Turing (SM 7.5).

**Mitigation:**
- faster-whisper/CTranslate2: Mature Turing support, widely tested
- exllamav2: Actively tested on Turing GPUs (RTX 2060/2070/2080 series)
- Avoid frameworks that require BF16 or FP8 (not available on Turing)
- Pin CUDA toolkit to 12.x (full Turing support confirmed)
- Test thoroughly before deployment; maintain a known-good software configuration

### Risk 3: Thermal Throttling in Enclosed Spaces

**Risk:** Church sound booths are often small, enclosed spaces. The RTX 2070 draws 175 W under load and requires adequate airflow to avoid thermal throttling.

**Mitigation:**
- Ensure the PC case has at least two intake fans and one exhaust
- Monitor GPU temperature via `nvidia-smi` and alert if > 80 C
- Consider an aftermarket GPU cooler or a model with a large heatsink (e.g., EVGA XC Ultra)
- Set a temperature target in software (`nvidia-smi -pl 150` to cap power at 150 W for cooler operation at ~5% performance cost)

### Risk 4: TranslateGemma Chat Template Compatibility

**Risk:** TranslateGemma requires a specific chat template with `source_lang_code` and `target_lang_code` fields. exllamav2 and llama.cpp may not natively support this template.

**Mitigation:**
- Implement a thin wrapper that formats the prompt according to TranslateGemma's expected template before passing to the inference engine
- The template is simple text formatting, not a framework limitation
- Test with sample translations during integration to verify correct prompt formatting
- MarianMT fallback does not have this issue (standard seq2seq input)

### Risk 5: Model Availability in Non-MLX Formats

**Risk:** The current pipeline uses MLX-format models (`mlx-community/translategemma-*-4bit`). These are Apple-only. GPTQ, AWQ, or EXL2 quantized versions of TranslateGemma may need to be created from the base model.

**Mitigation:**
- Download the base TranslateGemma model from Google (`google/translategemma-4b-it`)
- Quantize to EXL2 using exllamav2's conversion tool: `python convert.py -i <model_dir> -o <output_dir> -cf <output_dir> -b 4.0`
- Quantize to GGUF using llama.cpp's `convert_hf_to_gguf.py` + `llama-quantize`
- This is a one-time cost of ~30--60 minutes per model
- Pre-quantized GGUF models for Gemma variants are increasingly available on HuggingFace

### Risk 6: Driver and Framework Version Drift

**Risk:** NVIDIA driver updates, CUDA toolkit changes, or Python package updates could break the inference stack.

**Mitigation:**
- Pin all versions in `requirements.txt` with exact pins
- Use Docker containers for reproducible deployment (e.g., `nvidia/cuda:12.4.1-runtime-ubuntu22.04`)
- Document the exact working configuration (driver version, CUDA version, Python packages)
- Maintain a tested VM/container image that can be deployed to new hardware

---

## 11. Conclusion and Recommendation

### The RTX 2070 is a viable church inference endpoint with the following caveats:

**What works:**
- The full Approach A pipeline (VAD + Distil-Whisper + TranslateGemma 4B) fits in 8 GB VRAM with ~3 GB headroom when using optimized quantization (INT8 Whisper + 4-bit Gemma)
- End-to-end latency of ~500--900 ms is acceptable for live subtitle display
- Used hardware cost of ~$300--450 per endpoint is affordable for church budgets
- All required frameworks (faster-whisper, exllamav2, llama.cpp) have confirmed support for Turing (SM 7.5)
- MarianMT provides an ultra-low-latency fallback at < 330 ms end-to-end

**What does not work:**
- TranslateGemma 12B (Approach B) does not fit alongside Whisper in 8 GB VRAM
- The M3 Pro's advantage of running both 4B and 12B simultaneously is not replicable
- Power consumption and fan noise require consideration for church sound booths
- Setup complexity is higher than the MacBook (no Ollama-in-a-box for the full pipeline yet)

### Recommended rollout plan:

1. **Phase 1 (Proof of concept):** Build one RTX 2070 endpoint with the Tier 2 (MarianMT) configuration. Validate latency, stability, and audio quality in the actual church environment. Cost: ~$300.

2. **Phase 2 (Quality upgrade):** Convert TranslateGemma 4B to EXL2 format, deploy Tier 1 configuration. A/B test against MarianMT to quantify translation quality improvement for theological content. Cost: $0 (software only).

3. **Phase 3 (Scale):** If the RTX 2070 endpoint proves reliable, build additional units for other church sites. Consider the RTX 3060 12 GB ($180--220 used) as an alternative with more VRAM headroom.

4. **Phase 4 (Optimization):** Explore TensorRT compilation for both Whisper and TranslateGemma to squeeze additional latency savings. This is engineering-intensive but could bring Tier 1 latency below 500 ms consistently.

### Alternative GPUs to consider:

If the RTX 2070 proves too constrained, these alternatives offer better value-to-VRAM ratios:

| GPU | VRAM | Used Price | Fits 12B? | Notes |
|-----|------|-----------|-----------|-------|
| RTX 3060 12 GB | 12 GB | ~$180--220 | Tight (4-bit) | Best VRAM-per-dollar; Ampere |
| RTX 3060 Ti 8 GB | 8 GB | ~$180--200 | No | Faster than 2070 but same VRAM |
| RTX 3080 10 GB | 10 GB | ~$280--350 | Tight | Much faster, slightly more VRAM |
| RTX 4060 8 GB | 8 GB | ~$250--280 | No | Ada Lovelace, most power efficient |
| Intel Arc A770 16 GB | 16 GB | ~$180--220 | Yes | 16 GB VRAM, but weaker software ecosystem |

The **RTX 3060 12 GB** is the strongest alternative: 50% more VRAM, Ampere architecture (BF16 support), and comparable used pricing. It could potentially fit the 12B model alongside Whisper, though it would be tight (~11 GB of 12 GB).

---

## 12. Sources

- [faster-whisper GitHub repository (SYSTRAN/faster-whisper)](https://github.com/SYSTRAN/faster-whisper) -- CTranslate2 Whisper benchmarks, VRAM data
- [Whisper GPU performance benchmarks (openai/whisper Discussion #918)](https://github.com/openai/whisper/discussions/918) -- Multi-GPU comparison data, Turing FP16 behavior
- [OpenAI Whisper GPU Transcription Benchmarks (Tom's Hardware)](https://www.tomshardware.com/news/whisper-audio-transcription-gpus-benchmarked) -- 18-GPU WPM comparison
- [RTX 2070 for LLMs (TechReviewer)](https://www.techreviewer.com/tech-specs/nvidia-rtx-2070-gpu-for-llms/) -- LLM inference benchmarks, VRAM guidance
- [GPTQ vs AWQ vs EXL2 vs llama.cpp comparison (oobabooga)](https://oobabooga.github.io/blog/posts/gptq-awq-exl2-llamacpp/) -- Quantization benchmark data on RTX 3090
- [RTX 2070 Price Tracker (bestvaluegpu.com)](https://bestvaluegpu.com/history/new-and-used-rtx-2070-price-history-and-specs/) -- Historical pricing data
- [RTX 2070 Super Price Tracker (bestvaluegpu.com)](https://bestvaluegpu.com/history/new-and-used-rtx-2070-super-price-history-and-specs/) -- Super variant pricing
- [TensorRT-LLM GitHub (NVIDIA)](https://github.com/NVIDIA/TensorRT-LLM) -- SM 7.5+ support confirmation
- [TensorRT-LLM consumer GPU benchmarks (Menlo Research)](https://menlo.ai/blog/benchmarking-nvidia-tensorrt-llm) -- RTX 3090/4090 benchmark data
- [whisper_trt (NVIDIA-AI-IOT)](https://github.com/NVIDIA-AI-IOT/whisper_trt) -- TensorRT Whisper optimization
- [ExLlamaV2: Local LLM Inference on Consumer GPUs (Medium)](https://medium.com/@shouke.wei/exllamav2-revolutionizing-local-llm-inference-on-consumer-gpus-e14213f610bf) -- ExLlamaV2 overview
- [Distil-Whisper large-v3 (HuggingFace)](https://huggingface.co/distil-whisper/distil-large-v3) -- Model specifications
- [Silero VAD v5 (GitHub)](https://github.com/snakers4/silero-vad) -- VAD model specs and benchmarks
- [OPUS-MT CTranslate2 guide](https://opennmt.net/CTranslate2/guides/opus_mt.html) -- MarianMT INT8 inference
- [LLMs on 8GB VRAM benchmark guide (yWian)](https://www.ywian.com/blog/llms-on-8gb-vram-a-benchmark-guide) -- 8 GB VRAM deployment strategies
- [Gemma 3 QAT models (Google Developers Blog)](https://developers.googleblog.com/en/gemma-3-quantized-aware-trained-state-of-the-art-ai-to-consumer-gpus/) -- Quantization-aware training for consumer GPUs
- [llama.cpp CUDA performance (GitHub Discussion #15013)](https://github.com/ggml-org/llama.cpp/discussions/15013) -- CUDA backend benchmarks
- [TensorRT-LLM GeForce support (GitHub Issue #146)](https://github.com/NVIDIA/TensorRT-LLM/issues/146) -- Consumer GPU compatibility
