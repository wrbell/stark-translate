# NVIDIA Translation Optimization — C++ Inference Runtime

> Implementation plan for replacing HuggingFace `transformers.generate()` with compiled
> inference runtimes on NVIDIA GPUs. Target: **sub-1s translation** for TranslateGemma 4B.

**Status:** Planning
**Scope:** CUDA translation path only. STT already fast (CTranslate2 via faster-whisper). Mac MLX path unchanged.
**Date:** 2026-03-01

---

## Problem Statement

The current CUDA translation path (`CUDAGemmaEngine` in `engines/cuda_engine.py`) uses
HuggingFace `transformers.generate()` + bitsandbytes NF4 quantization:

- **~15-18 tok/s** on RTX 2070 (8 GB VRAM)
- **~2-3.5s per translation** for a typical 40-60 token output
- Bottleneck: Python-level autoregressive loop, no KV-cache optimization, no fused kernels

Goal: **sub-1s translation** via a compiled C++ inference runtime, without sacrificing
translation quality on theological content.

---

## 1. Runtime Comparison

| Runtime | Expected tok/s (RTX 2070) | Pros | Cons | Integration Effort |
|---------|--------------------------|------|------|-------------------|
| **llama.cpp** (GGUF) | 25-40 | Simplest deployment, mature CUDA backend, `llama-cpp-python` bindings, active community | ~20-30% slower than exllamav2 at batch=1 | 3-5 days |
| **exllamav2** (EXL2) | 40-65 | Fastest single-request perf on consumer GPUs, excellent quantization quality | Smaller community, Turing Flash Attention limited, Python-only API | 5-7 days |
| **vLLM** | 35-55 | Best throughput for batched serving, PagedAttention | Overkill for single-request, higher VRAM overhead, server-mode only | N/A (wrong use case) |
| **TensorRT-LLM** | 45-70 | Maximum absolute perf, INT4-AWQ support | Complex build system, frequent breaking changes, NVIDIA-only | 1-2 weeks |

### Recommendation

**Phase 1: llama.cpp** — lowest risk, widest hardware support, simplest build. Gets us to
~25-40 tok/s (~500-800ms per translation) with 3-5 days of work.

**Phase 2 (optional): exllamav2** — if llama.cpp isn't fast enough. Gets ~40-65 tok/s
(~300-500ms) but with a narrower compatibility window.

vLLM and TensorRT-LLM are not recommended. vLLM's PagedAttention optimizes batched
throughput (not our single-request use case). TensorRT-LLM has the highest absolute
performance but a painful build process and frequent API breaks.

---

## 2. Model Conversion Pipeline

### 2.1 Merge LoRA → FP16 HuggingFace

Reuse existing logic from `tools/convert_models_to_both.py`:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("google/translategemma-4b-it", torch_dtype=torch.float16)
model = PeftModel.from_pretrained(base, "fine_tuned_gemma_mi_A/")
merged = model.merge_and_unload()
merged.save_pretrained("exports/gemma-4b-merged-fp16/")
tokenizer = AutoTokenizer.from_pretrained("google/translategemma-4b-it")
tokenizer.save_pretrained("exports/gemma-4b-merged-fp16/")
```

### 2.2 Convert FP16 → GGUF

```bash
# Clone llama.cpp (if not already)
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

# Convert HF model to GGUF F16
python convert_hf_to_gguf.py ../exports/gemma-4b-merged-fp16/ \
    --outfile ../exports/gemma-4b-f16.gguf \
    --outtype f16

# Quantize to Q4_K_M (best quality/speed tradeoff)
./llama-quantize ../exports/gemma-4b-f16.gguf \
    ../exports/gemma-4b-q4km.gguf Q4_K_M
```

### 2.3 Convert FP16 → EXL2 (optional Phase 2)

```bash
python exllamav2/convert.py \
    -i exports/gemma-4b-merged-fp16/ \
    -o exports/gemma-4b-exl2-4.0bpw/ \
    -b 4.0 \
    -cf exports/gemma-4b-exl2-4.0bpw/
```

### 2.4 New Functions in `tools/convert_models_to_both.py`

Add `export_gemma_gguf()` and `export_gemma_exl2()`:

```python
def export_gemma_gguf(
    merged_dir: str | Path,
    output_path: str | Path,
    quant_type: str = "Q4_K_M",
    llama_cpp_dir: str | Path = "llama.cpp",
) -> Path:
    """Convert merged FP16 HF model → GGUF quantized."""
    ...

def export_gemma_exl2(
    merged_dir: str | Path,
    output_dir: str | Path,
    bits_per_weight: float = 4.0,
    exllamav2_dir: str | Path = "exllamav2",
) -> Path:
    """Convert merged FP16 HF model → EXL2 quantized."""
    ...
```

---

## 3. New Engine Class — `LlamaCppGemmaEngine`

### 3.1 Location

New file: `engines/llamacpp_engine.py` (keeps `cuda_engine.py` focused on HF/bitsandbytes).

### 3.2 Interface

Implements `TranslationEngine` ABC from `engines/base.py`:

```python
class LlamaCppGemmaEngine(TranslationEngine):
    """TranslateGemma inference via llama.cpp GGUF backend."""

    def __init__(
        self,
        model_path: str | Path,
        n_gpu_layers: int = -1,  # offload all to GPU
        n_ctx: int = 256,
        source_lang: str = "en",
        target_lang: str = "es",
    ):
        from llama_cpp import Llama
        self._llm = Llama(
            model_path=str(model_path),
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            verbose=False,
        )
        self._source_lang = source_lang
        self._target_lang = target_lang

    def translate(self, text: str) -> TranslationResult:
        prompt = self._format_prompt(text)
        t0 = time.perf_counter()
        output = self._llm(
            prompt,
            max_tokens=128,
            stop=["<end_of_turn>"],
            temperature=0.0,
        )
        elapsed = time.perf_counter() - t0
        translated = output["choices"][0]["text"].strip()
        return TranslationResult(
            text=translated,
            latency_ms=elapsed * 1000,
            model_name="llamacpp-gemma-4b",
        )

    def _format_prompt(self, text: str) -> str:
        """Reproduce TranslateGemma's Jinja2 chat template as raw string.

        Must match `tokenizer.apply_chat_template()` output exactly.
        Verified by unit test in tests/test_llamacpp_engine.py.
        """
        return (
            "<start_of_turn>user\n"
            f"Translate the following text from {self._source_lang} "
            f"to {self._target_lang}.\n"
            f"{text}<end_of_turn>\n"
            "<start_of_turn>model\n"
        )
```

### 3.3 Critical: Prompt Format Verification

TranslateGemma uses a specific chat template that includes `source_lang_code` and
`target_lang_code`. The `_format_prompt()` method must reproduce this exactly.

**Unit test** (in `tests/test_llamacpp_engine.py`):

```python
def test_prompt_matches_chat_template():
    """Verify manual prompt matches tokenizer.apply_chat_template() output."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("google/translategemma-4b-it")

    messages = [{"role": "user", "content": {
        "source_lang_code": "en",
        "target_lang_code": "es",
        "text": "The Lord is my shepherd."
    }}]
    expected = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    actual = engine._format_prompt("The Lord is my shepherd.")
    assert actual == expected
```

### 3.4 EOS Token Handling

llama.cpp handles EOS differently from HF `generate()`. The `stop=["<end_of_turn>"]`
parameter is critical — without it, the model generates padding tokens indefinitely
(same issue we hit with MLX, see MEMORY.md).

---

## 4. Settings & Factory Integration

### 4.1 `settings.py` — New Fields in `TranslationSettings`

```python
class TranslationSettings(BaseSettings):
    # ... existing fields ...
    cuda_translation_runtime: str = "bitsandbytes"  # bitsandbytes | llamacpp | exllamav2
    cuda_gguf_model_path: str | None = None         # path to .gguf file
    cuda_exl2_model_path: str | None = None         # path to exl2 directory
```

Environment variables: `STARK_TRANSLATE__CUDA_TRANSLATION_RUNTIME`, etc.

### 4.2 `engines/factory.py` — Dispatch

```python
def create_translation_engine(backend: str, ...) -> TranslationEngine:
    if backend == "cuda":
        runtime = settings.translate.cuda_translation_runtime
        if runtime == "llamacpp":
            from engines.llamacpp_engine import LlamaCppGemmaEngine
            return LlamaCppGemmaEngine(model_path=settings.translate.cuda_gguf_model_path)
        elif runtime == "exllamav2":
            from engines.exllamav2_engine import ExllamaV2GemmaEngine
            return ExllamaV2GemmaEngine(model_path=settings.translate.cuda_exl2_model_path)
        else:  # bitsandbytes (default)
            return CUDAGemmaEngine(...)
```

### 4.3 `dry_run_ab.py` — Runtime Branching

In `load_cuda_translation_models()`, branch on the runtime setting. No changes to the
pipeline loop — the engine ABC ensures compatible interfaces.

### 4.4 Rollback

Single env var change to revert:
```bash
export STARK_TRANSLATE__CUDA_TRANSLATION_RUNTIME=bitsandbytes
```

Or remove the env var entirely (defaults to `bitsandbytes`).

---

## 5. Hardware Configuration Matrix

### 5.1 Expected Performance (llama.cpp, Q4_K_M)

| GPU | VRAM | Bandwidth | 4B tok/s | 4B Translation (50 tok) | 12B Fits? |
|-----|------|-----------|----------|------------------------|-----------|
| RTX 2070 (8 GB) | 8 GB | 448 GB/s | 25-40 | 500-800ms | No |
| RTX 3060 (12 GB) | 12 GB | 360 GB/s | 25-35 | 600-800ms | Yes (tight, ~10.5 GB) |
| RTX 3090 (24 GB) | 24 GB | 936 GB/s | 50-80 | 250-400ms | Yes (comfortable) |
| A2000 Ada (16 GB) | 16 GB | 288 GB/s | 20-30 | 700-1000ms | Yes (tight, ~10.5 GB) |

### 5.2 VRAM Budget (llama.cpp)

| Component | Q4_K_M | Q5_K_M | Q8_0 |
|-----------|--------|--------|------|
| TranslateGemma 4B | ~2.4 GB | ~2.8 GB | ~4.2 GB |
| TranslateGemma 12B | ~6.9 GB | ~8.0 GB | ~12.0 GB |
| Whisper (faster-whisper INT8) | ~0.9 GB | ~0.9 GB | ~0.9 GB |
| KV cache (256 ctx) | ~0.1 GB | ~0.1 GB | ~0.1 GB |
| CUDA context | ~0.4 GB | ~0.4 GB | ~0.4 GB |
| **Total (4B only)** | **~3.8 GB** | **~4.2 GB** | **~5.6 GB** |
| **Total (4B + 12B)** | **~10.7 GB** | **~12.2 GB** | **~17.6 GB** |

### 5.3 Quantization Quality vs Speed

| Quant | Relative Quality | Relative Speed | Recommendation |
|-------|-----------------|----------------|----------------|
| Q4_K_M | ~98% of FP16 | Fastest | Default — best speed/quality tradeoff |
| Q5_K_M | ~99% of FP16 | ~10% slower | Fallback if Q4_K_M degrades theological terms |
| Q8_0 | ~99.5% of FP16 | ~30% slower | Only if VRAM allows and quality is critical |

---

## 6. Testing Strategy

### 6.1 Quality Equivalence

Run 100 test sentences (50 general + 50 theological) through both the old HF path and
the new llama.cpp path. Compare:

| Metric | Acceptable Delta |
|--------|-----------------|
| SacreBLEU | < 2 points difference |
| chrF++ | < 2 points difference |
| Theological term accuracy | < 5% degradation |

If Q4_K_M fails the theological term check, upgrade to Q5_K_M.

### 6.2 Prompt Format Verification

Unit test comparing manual prompt string to `tokenizer.apply_chat_template()` output.
This is the highest-risk item — a mismatch means degraded quality with no error signal.

### 6.3 Latency Benchmarks

Extend `tools/benchmark_latency.py` with `--translation-runtime` flag:

```bash
python tools/benchmark_latency.py --translation-runtime llamacpp --runs 50
python tools/benchmark_latency.py --translation-runtime bitsandbytes --runs 50
```

Compare p50, p95, p99 latency distributions.

### 6.4 EOS Token Verification

Verify that `stop=["<end_of_turn>"]` works correctly:
- Model generates translation text
- Stops at `<end_of_turn>` token (id=106)
- Does NOT generate pad tokens or continue past the translation

### 6.5 Stress Test

100-chunk continuous run (simulating a full sermon):
- No VRAM growth (check with `nvidia-smi` at start and end)
- No latency degradation over time
- No crashes or segfaults

---

## 7. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Chat template reproduction mismatch | Medium | High (silent quality loss) | Unit test comparing to `apply_chat_template()` |
| llama.cpp doesn't support Gemma 2 architecture | Low | High (blocks entire plan) | Verify first: `llama.cpp --print-supported-models`. Gemma 2 supported since ggerganov/llama.cpp#8271 |
| Q4_K_M degrades theological term accuracy | Medium | Medium | Fallback to Q5_K_M; test with 50 theological sentences first |
| `llama-cpp-python` CUDA build fragility | Medium | Medium | Pin version, document build flags, provide pre-built wheel instructions |
| KV cache memory leak over long sessions | Low | Medium | Monitor VRAM in stress test; set `n_ctx=256` (short translations) |
| exllamav2 Turing Flash Attention gaps | Low (Phase 2 only) | Low | Only affects optional Phase 2; llama.cpp as fallback |

---

## 8. Files to Modify (When Implementing)

| File | Change | Lines (est.) |
|------|--------|-------------|
| `engines/llamacpp_engine.py` | **New**: `LlamaCppGemmaEngine` class | ~80 |
| `engines/factory.py` | Add `engine_type="llamacpp"` dispatch | ~15 |
| `settings.py` | Add `cuda_translation_runtime`, `cuda_gguf_model_path` fields | ~10 |
| `tools/convert_models_to_both.py` | Add `export_gemma_gguf()`, `export_gemma_exl2()` | ~60 |
| `dry_run_ab.py` | Branch `load_cuda_translation_models()` on runtime | ~20 |
| `requirements-nvidia.txt` | Add `llama-cpp-python` (commented, with build notes) | ~5 |
| `tests/test_llamacpp_engine.py` | **New**: prompt format, constructor, mock translate | ~80 |

**Total estimated new/changed code: ~270 lines**

---

## 9. Implementation Phases

| Phase | Scope | Effort | Deliverable |
|-------|-------|--------|-------------|
| **1. Convert & Verify** | GGUF conversion, prompt format test | 1 day | `.gguf` file + passing prompt test |
| **2. Engine Class** | `LlamaCppGemmaEngine` + factory wiring | 1-2 days | Translation working end-to-end |
| **3. Quality Gate** | 100-sentence quality comparison | 0.5 day | SacreBLEU/chrF++ report |
| **4. Stress Test** | 100-chunk latency + VRAM stability | 0.5 day | Benchmark report |
| **5. (Optional) exllamav2** | EXL2 conversion + engine class | 2-3 days | Faster alternative if needed |

**Total: 3-5 days** for llama.cpp path, +2-3 days for optional exllamav2.

---

## 10. Migration Path from Immediate TODO

The "LATER: C++ / GPU Optimization Proposal" section in [`immediate_todo.md`](./immediate_todo.md)
contains the original Layer 1-4 hardware upgrade path. This document supersedes the Layer 2
(Runtime Optimization) section with concrete implementation details. Layers 1, 3, and 4
(hardware swaps) remain valid and complementary.
