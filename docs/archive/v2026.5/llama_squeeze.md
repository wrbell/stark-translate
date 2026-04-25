# PLAN.md — Inference Engine Benchmark: llama.cpp vs HuggingFace on 3060

> **Goal:** Benchmark llama.cpp (GGUF + n-gram spec decode) against the existing `CUDAGemmaStreamingEngine` (HF + prompt cache) for translation latency. Benchmark two Whisper configs for STT. Pick winners, wire into the existing engine system.
>
> **Machine:** RTX 3060 12GB desktop (CUDA, drop-in from RTX 2070)
>
> **Critical constraint:** The codebase has a mature engine abstraction (`engines/base.py`, `engines/factory.py`) and fine-tuned LoRA adapters from 15+ training runs. Any new engine must integrate with the factory pattern and support adapter loading. This is not a greenfield rewrite — it's adding a new backend option.
>
> **Time:** One afternoon for benchmarks, one follow-up session for integration.

---

## What already exists (don't rebuild these)

The codebase has working inference engines that handle the full pipeline today:

| Component | Class | File | Status |
|-----------|-------|------|--------|
| STT (CUDA) | `FasterWhisperEngine` | `engines/cuda_engine.py` | Working. Fallback retry, active learning logs, confidence flagging |
| STT (MLX) | `MLXWhisperEngine` | `engines/mlx_engine.py` | Working. Same fallback/logging features |
| Translation (CUDA) | `CUDAGemmaStreamingEngine` | `engines/cuda_engine.py` | Working. Streaming, prompt cache (~50-80ms savings), HF speculative decode via `assistant_model=` |
| Translation (MLX) | `MLXGemmaEngine` | `engines/mlx_engine.py` | Working. Prompt cache, EOS fix |
| Translation (CPU) | `MarianEngine` | `engines/mlx_engine.py` | Working. Fast partials (~80ms) |
| VRAM detection | `detect_vram_tier()` | `engines/cuda_engine.py` | Auto-selects `full_ab` / `4b_only` / `marian` |
| Factory | `create_stt_engine()`, `create_translation_engine()` | `engines/factory.py` | Dispatches by backend with lazy imports |

**The existing `CUDAGemmaStreamingEngine` already does:**
- Pre-computed KV cache for the fixed TranslateGemma chat template prefix (SPLIT_HERE marker technique)
- Deep-clone of KV cache per request via `_clone_past_key_values()` (GPU tensor `.clone()`, fast)
- HF-native speculative decoding: `assistant_model=` parameter (4B drafts for 12B)
- Token-by-token streaming via `TextIteratorStreamer` with configurable batching
- EOS fix (`<end_of_turn>` id=106 added to `_eos_token_ids`)
- Dynamic max-tokens cap (Spanish ~15-25% longer than English)

**The question is whether llama.cpp with n-gram speculation beats this on the 3060.**

---

## What the benchmark tests

### PART 1: STT (2 configs)

| Config | Description | VRAM | Quality |
|--------|-------------|------|---------|
| **S1** | faster-whisper distil-v3.5 INT8 on GPU | ~0.8 GB | Distil quality (good enough for fine-tuned adapter) |
| **S2** | HF spec decode: distil-v3.5 drafts for Whisper Large V3 | ~4.2 GB | Large V3 quality (best available) |

S1 is what `FasterWhisperEngine` already does (just swap model ID to `distil-large-v3.5`). S2 is new — uses HF `assistant_model` API.

### PART 2: Translation (4 configs)

| Config | Description | VRAM | Notes |
|--------|-------------|------|-------|
| **T1** | `CUDAGemmaStreamingEngine` with prompt cache (existing code) | ~3.0 GB (NF4) | **Baseline — this is what you're already running** |
| **T2** | llama.cpp TG 4B baseline (no spec decode) | ~2.5 GB (Q4_K_M) | Tests llama.cpp alone vs HF |
| **T3** | llama.cpp TG 4B + `--spec-type ngram-mod` | ~2.5 GB (Q4_K_M) | The n-gram spec decode hypothesis |
| **T4** | llama.cpp TG 4B + Gemma 4 E2B draft | ~3.8 GB (Q4_K_M) | Tokenizer compatibility test — passes or fails at startup |

T1 is the critical baseline. If llama.cpp doesn't beat the existing HF engine with prompt caching, there's no reason to add complexity.

---

## TurboQuant re-evaluation: still skip

TurboQuant (ICLR 2026, Google) compresses KV cache 72-78% via mixed-precision quantization. Re-evaluated against the codebase:

**Why it still doesn't help here:**
1. TranslateGemma translations use ~512 max context tokens. KV cache at this length is ~50-100 MB for the 4B model. Saving 75% of that is ~37-75 MB — noise on a 12 GB card.
2. `CUDAGemmaStreamingEngine._clone_past_key_values()` already does fast GPU tensor clones. The prompt cache clone is not the bottleneck (code warns at >20ms, implying it's normally well under that).
3. TurboQuant is not in upstream llama.cpp (forks only). Untested on TranslateGemma's head dimensions.
4. llama.cpp already has built-in KV cache quantization via `-ctk q8_0` / `-ctk q4_0` flags, achieving similar memory savings with zero extra work. Include this as a free add-on in the T3 benchmark.

**What to do instead:** Add `-ctk q8_0` to all llama.cpp configs (free, no quality loss, already proven in the TranslateGemma llama.cpp issue #19231). Skip TurboQuant entirely.

---

## Task 1: Install llama.cpp + Python deps

```bash
# llama.cpp with CUDA
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(nproc)

# Python deps (most already installed from the existing project)
pip install aiohttp  # HTTP client for llama-server
```

**Verify:**
```bash
./build/bin/llama-server --version
python -c "from engines.cuda_engine import CUDAGemmaStreamingEngine; print('HF engine OK')"
```

---

## Task 2: STT Config S1 — faster-whisper distil-v3.5 INT8

This is a one-line model swap in `FasterWhisperEngine`. No new code needed.

```python
# bench_stt.py
from faster_whisper import WhisperModel
import time

model = WhisperModel("distil-large-v3.5", device="cuda", compute_type="int8")

test_audio = "test_sermon_clip.wav"  # 10-30s WAV at 16kHz mono

for i in range(3):
    t0 = time.perf_counter()
    segments, info = model.transcribe(test_audio, language="en", beam_size=1)
    text = " ".join(s.text.strip() for s in segments)
    ms = (time.perf_counter() - t0) * 1000
    print(f"Run {i+1}: {ms:.0f} ms — {text[:80]}...")
```

**Record:** avg latency → `stt_s1_ms`, VRAM (`nvidia-smi`), output text.

---

## Task 3: STT Config S2 — HF spec decode (distil-v3.5 drafts for Large V3)

```python
# bench_stt_spec.py
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
import torch, time, torchaudio

device = "cuda"
dtype = torch.float16

target = AutoModelForSpeechSeq2Seq.from_pretrained(
    "openai/whisper-large-v3", torch_dtype=dtype, low_cpu_mem_usage=True
).to(device)

draft = AutoModelForSpeechSeq2Seq.from_pretrained(
    "distil-whisper/distil-large-v3.5", torch_dtype=dtype, low_cpu_mem_usage=True
).to(device)

processor = AutoProcessor.from_pretrained("openai/whisper-large-v3")

audio, sr = torchaudio.load("test_sermon_clip.wav")
if sr != 16000:
    audio = torchaudio.transforms.Resample(sr, 16000)(audio)
input_features = processor(
    audio.squeeze().numpy(), sampling_rate=16000, return_tensors="pt"
).input_features.to(device, dtype=dtype)

for i in range(3):
    t0 = time.perf_counter()
    with torch.no_grad():
        output = target.generate(
            input_features, assistant_model=draft,
            max_new_tokens=128, language="en",
        )
    text = processor.batch_decode(output, skip_special_tokens=True)[0]
    ms = (time.perf_counter() - t0) * 1000
    print(f"Run {i+1}: {ms:.0f} ms — {text[:80]}...")
```

**Record:** avg latency → `stt_s2_ms`, VRAM, output text.

---

## Task 4: Pick STT winner

```
Config S1 (faster-whisper distil-v3.5 INT8):  _____ ms, ~0.8 GB VRAM
Config S2 (HF spec decode, Large V3 quality):  _____ ms, ~4.2 GB VRAM

Quality difference noticeable on sermon content? [yes/no]
Winner: Config S___
```

**Decision logic:** S2 gives Large V3 quality at ~2× standalone V3 speed, but uses ~3.4 GB more VRAM. On the 3060 (12 GB), S2 leaves ~7.8 GB for translation — still comfortable. Pick S2 unless latency is significantly worse than S1 for the quality you need. The fine-tuned Whisper LoRA adapters (W12 etc.) are trained on Large V3 Turbo, so S1 with the fine-tuned adapter may already match S2's quality — test both with and without adapter if available.

---

## Task 5: Translation Config T1 — existing HF engine (BASELINE)

This runs the code you already have. Load `CUDAGemmaStreamingEngine` and benchmark it.

```python
# bench_translate_hf.py
import time, torch
from engines.cuda_engine import CUDAGemmaStreamingEngine

engine = CUDAGemmaStreamingEngine(
    model_id="google/translategemma-4b-it",
    use_prompt_cache=True,
)
engine.load()

phrases = [
    "Good morning everyone, welcome to our service.",
    "Let us open our Bibles to Romans chapter 8 verse 28.",
    "Justification by faith is the cornerstone of the Gospel.",
    "The apostle Paul wrote to the Philippians from prison.",
    "Go therefore and make disciples of all nations, baptizing them in the name of the Father and of the Son and of the Holy Spirit.",
]

# Warmup
engine.translate("Hello world.")

for phrase in phrases:
    result = engine.translate(phrase)
    print(f"{result.latency_ms:.0f} ms | {result.tokens_per_second:.1f} tok/s | {result.text}")

engine.unload()
```

**Record:** avg latency → `trans_t1_ms`, avg tok/s → `trans_t1_tps`.

---

## Task 6: Translation Config T2 — llama.cpp baseline (no spec decode)

```bash
./build/bin/llama-server \
  --hf-repo mradermacher/translategemma-4b-it-GGUF \
  --hf-file translategemma-4b-it.Q4_K_M.gguf \
  --port 8080 -ngl 99 --jinja --flash-attn -c 2048 \
  -ctk q8_0 \
  --chat-template-kwargs '{"source_lang_code": "en", "target_lang_code": "es"}'
```

Run the same 5 phrases via curl. **Record:** avg tok/s → `trans_t2_tps`. Stop server.

---

## Task 7: Translation Config T3 — llama.cpp + n-gram spec decode

```bash
./build/bin/llama-server \
  --hf-repo mradermacher/translategemma-4b-it-GGUF \
  --hf-file translategemma-4b-it.Q4_K_M.gguf \
  --port 8080 -ngl 99 --jinja --flash-attn -c 2048 \
  -ctk q8_0 \
  --spec-type ngram-mod \
  --spec-ngram-size-n 12 \
  --draft-min 8 \
  --draft-max 32 \
  --chat-template-kwargs '{"source_lang_code": "en", "target_lang_code": "es"}'
```

Run same 5 phrases. **Record:** avg tok/s → `trans_t3_tps`, acceptance rate. Stop server.

---

## Task 8: Translation Config T4 — llama.cpp + Gemma 4 E2B draft

Tokenizer compatibility test. If llama-server starts, the tokenizers match.

```bash
./build/bin/llama-server \
  --hf-repo mradermacher/translategemma-4b-it-GGUF \
  --hf-file translategemma-4b-it.Q4_K_M.gguf \
  -md --hf-repo bartowski/google_gemma-4-E2B-it-GGUF \
  -md --hf-file google_gemma-4-E2B-it-Q4_K_M.gguf \
  --draft 16 --draft-min 5 \
  --port 8080 -ngl 99 -ngld 99 --jinja --flash-attn -c 2048 \
  -ctk q8_0 \
  --chat-template-kwargs '{"source_lang_code": "en", "target_lang_code": "es"}'
```

**Check exact flag names:** `./build/bin/llama-server --help | grep -i draft`

**If it starts:** Run same 5 phrases. Record tok/s and acceptance rate.
**If tokenizer error:** Log `T4: TOKENIZER_FAIL`. Move on — T3 is the fallback.

---

## Task 9: Pick translation winner

```
Config T1 (HF + prompt cache):   _____ tok/s (BASELINE — what you already have)
Config T2 (llama.cpp baseline):  _____ tok/s
Config T3 (llama.cpp + n-gram):  _____ tok/s, ___% acceptance
Config T4 (llama.cpp + E2B):     _____ tok/s, ___% acceptance (or TOKENIZER_FAIL)

Winner: Config T___
```

**Decision logic:**
- If T1 (existing HF engine) wins or ties: **don't add llama.cpp**. The complexity isn't worth it. Focus on other optimizations (adapter quality, streaming UX).
- If T3 or T4 beats T1 by >30%: **add llama.cpp as a new engine type** following the pattern in `engines/CLAUDE.md` → "Adding a New Engine."
- If T2 beats T1 but T3 doesn't beat T2: n-gram spec decode doesn't help for this content length. Use llama.cpp baseline if it's faster, but the win is from GGUF quants, not speculation.

---

## PART 3: Integration (only if llama.cpp wins)

### Task 10: Add `LlamaCppGemmaEngine` to the engine system

Follow the 4-step pattern from `engines/CLAUDE.md`:

**New file:** `engines/llamacpp_engine.py`

```python
"""llama.cpp translation engine via HTTP API to llama-server."""

import aiohttp, time, logging
from engines.base import TranslationEngine, TranslationResult

logger = logging.getLogger(__name__)

class LlamaCppGemmaEngine(TranslationEngine):
    """Translation via llama-server HTTP API.

    Requires llama-server running externally. Does NOT manage
    the server process — start it via start_server.sh.

    Constructor args:
        url:  llama-server base URL (default: http://localhost:8080)
    """
    def __init__(self, url: str = "http://localhost:8080"):
        self._url = f"{url}/v1/chat/completions"
        self._loaded = False

    def load(self) -> None:
        """Verify llama-server is reachable."""
        import requests
        try:
            resp = requests.get(f"{self._url.rsplit('/', 2)[0]}/health", timeout=5)
            resp.raise_for_status()
            self._loaded = True
            logger.info("LlamaCppGemmaEngine connected to %s", self._url)
        except Exception as e:
            raise RuntimeError(f"llama-server not reachable at {self._url}: {e}")

    def translate(self, text: str, *, source_lang: str = "en",
                  target_lang: str = "es") -> TranslationResult:
        if not self._loaded:
            raise RuntimeError("Engine not loaded — call load() first")

        import requests
        input_words = len(text.split())
        max_tok = max(32, int(input_words * 1.8))

        t0 = time.perf_counter()
        resp = requests.post(self._url, json={
            "messages": [{"role": "user", "content": text}],
            "max_tokens": max_tok,
            "temperature": 0.1,
        }, timeout=30)
        data = resp.json()
        es_text = data["choices"][0]["message"]["content"].strip()
        latency_ms = (time.perf_counter() - t0) * 1000

        # Estimate tok/s from response metadata if available
        usage = data.get("usage", {})
        out_tokens = usage.get("completion_tokens", len(es_text.split()))
        tps = out_tokens / (latency_ms / 1000) if latency_ms > 0 else 0

        return TranslationResult(text=es_text, latency_ms=latency_ms,
                                 tokens_per_second=tps)

    def unload(self) -> None:
        self._loaded = False
        logger.info("LlamaCppGemmaEngine disconnected")

    @property
    def model_id(self) -> str:
        return f"llamacpp:{self._url}"

    @property
    def backend(self) -> str:
        return "llamacpp"
```

**Register in `engines/factory.py`:**

```python
# In create_translation_engine(), add:
if engine_type == "llamacpp":
    from engines.llamacpp_engine import LlamaCppGemmaEngine
    return LlamaCppGemmaEngine(url=kwargs.get("url", "http://localhost:8080"))
```

**Add to `tests/conftest.py`:** Append `"llama_cpp"` to `_MOCK_MODULES`.

### Task 11: Write `start_server.sh`

```bash
#!/bin/bash
# Start llama-server with winning translation config.
# Uncomment the spec decode block matching your benchmark winner.

LLAMA_BIN="${LLAMA_BIN:-./llama.cpp/build/bin/llama-server}"

$LLAMA_BIN \
  --hf-repo mradermacher/translategemma-4b-it-GGUF \
  --hf-file translategemma-4b-it.Q4_K_M.gguf \
  #
  # Config T3 (n-gram spec decode — winner if T3 > T1):
  # --spec-type ngram-mod --spec-ngram-size-n 12 \
  # --draft-min 8 --draft-max 32 \
  #
  # Config T4 (E2B draft — winner if T4 > T3 and tokenizer passed):
  # -md bartowski/google_gemma-4-E2B-it-Q4_K_M.gguf \
  # --draft 16 --draft-min 5 -ngld 99 \
  #
  --port 8080 -ngl 99 --jinja --flash-attn -c 2048 \
  -ctk q8_0 \
  --chat-template-kwargs '{"source_lang_code": "en", "target_lang_code": "es"}'
```

### Task 12: Adapter loading for llama.cpp

The codebase has fine-tuned QLoRA adapters from S6 (TranslateGemma) and W12+ (Whisper). For llama.cpp, adapters must be merged into the base model before GGUF export:

```python
# export_merged_gguf.py — run once after training
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Merge LoRA into base weights
base = AutoModelForCausalLM.from_pretrained("google/translategemma-4b-it")
model = PeftModel.from_pretrained(base, "hybrid_runs/S6_balanced")
merged = model.merge_and_unload()

# Save merged model (HF format)
merged.save_pretrained("models/translategemma-4b-s6-merged")
tokenizer = AutoTokenizer.from_pretrained("hybrid_runs/S6_balanced")
tokenizer.save_pretrained("models/translategemma-4b-s6-merged")

# Then convert to GGUF externally:
# python llama.cpp/convert_hf_to_gguf.py models/translategemma-4b-s6-merged \
#   --outfile models/translategemma-4b-s6.Q4_K_M.gguf --outtype q4_k_m
```

**This is a prerequisite for using llama.cpp with fine-tuned models.** The HF engine loads adapters at runtime via PEFT — no merge step needed. If you're iterating on training frequently, the HF path has less friction. If you've locked in an adapter version, the GGUF export is a one-time cost.

### Task 13: Update `stt.py` for the STT winner

**New file:** `stt.py` — wraps both STT configs behind a flag.

```python
import time

STT_MODE = "fast"  # "fast" = faster-whisper, "spec" = HF speculative decode

if STT_MODE == "fast":
    from faster_whisper import WhisperModel

    class WhisperSTT:
        def __init__(self):
            self.model = WhisperModel("distil-large-v3.5", device="cuda",
                                       compute_type="int8")
            # Warmup
            import numpy as np
            silence = np.zeros(16000, dtype=np.float32)
            list(self.model.transcribe(silence, language="en")[0])

        def transcribe(self, audio_input, language="en"):
            t0 = time.perf_counter()
            segments, _ = self.model.transcribe(audio_input, language=language,
                                                 beam_size=1)
            text = " ".join(s.text.strip() for s in segments)
            return text, (time.perf_counter() - t0) * 1000

elif STT_MODE == "spec":
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
    import torch

    class WhisperSTT:
        def __init__(self):
            device, dtype = "cuda", torch.float16
            self.processor = AutoProcessor.from_pretrained("openai/whisper-large-v3")
            self.target = AutoModelForSpeechSeq2Seq.from_pretrained(
                "openai/whisper-large-v3", torch_dtype=dtype,
                low_cpu_mem_usage=True).to(device)
            self.draft = AutoModelForSpeechSeq2Seq.from_pretrained(
                "distil-whisper/distil-large-v3.5", torch_dtype=dtype,
                low_cpu_mem_usage=True).to(device)
            self.device, self.dtype = device, dtype

        def transcribe(self, audio_array, language="en"):
            inputs = self.processor(audio_array, sampling_rate=16000,
                                     return_tensors="pt"
                                     ).input_features.to(self.device, dtype=self.dtype)
            t0 = time.perf_counter()
            with torch.no_grad():
                output = self.target.generate(inputs, assistant_model=self.draft,
                                               max_new_tokens=128, language=language)
            text = self.processor.batch_decode(output, skip_special_tokens=True)[0]
            return text, (time.perf_counter() - t0) * 1000
```

### Task 14: Theological spot-check

**New file:** `spot_check.py` — same 8 canary sentences used in `evaluate_sermon.py`.

```python
import requests, sys

SPOT_CHECK = [
    ("Christ's atonement covers all sins.", "expiación"),
    ("The new covenant was sealed in blood.", "pacto"),
    ("Justified by grace through faith.", "gracia"),
    ("The righteousness of God is revealed.", "justicia"),
    ("The epistle of James teaches about works.", "Santiago"),
    ("James and John left their nets.", "Jacobo"),
    ("He preached about sanctification.", "santificación"),
    ("The propitiation for our sins.", "propiciación"),
]

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8080"

correct = 0
for en, expected in SPOT_CHECK:
    resp = requests.post(f"{URL}/v1/chat/completions", json={
        "messages": [{"role": "user", "content": en}],
        "max_tokens": 128, "temperature": 0.1,
    })
    es = resp.json()["choices"][0]["message"]["content"]
    found = expected.lower() in es.lower()
    correct += found
    print(f"{'✓' if found else '✗'} {en[:50]}...")
    print(f"  → {es.strip()}")
    if not found:
        print(f"  expected: '{expected}' MISSING")
    print()

print(f"Result: {correct}/{len(SPOT_CHECK)}")
```

Run against both the HF engine (via a quick wrapper) and llama-server to compare theological term accuracy across backends.

---

## Order of operations

| # | Task | Time | Output |
|---|------|------|--------|
| 1 | Install llama.cpp | 10 min | Binary ready |
| | **── STT Benchmark ──** | | |
| 2 | STT S1 (faster-whisper distil-v3.5) | 5 min | `stt_s1_ms` |
| 3 | STT S2 (HF spec decode) | 10 min | `stt_s2_ms` |
| 4 | Pick STT winner | 2 min | Decision logged |
| | **── Translation Benchmark ──** | | |
| 5 | T1: existing HF engine (**baseline**) | 5 min | `trans_t1_tps` |
| 6 | T2: llama.cpp baseline | 5 min | `trans_t2_tps` |
| 7 | T3: llama.cpp + n-gram | 5 min | `trans_t3_tps` + acceptance |
| 8 | T4: llama.cpp + E2B draft | 5 min | `trans_t4_tps` (or TOKENIZER_FAIL) |
| 9 | Pick translation winner | 5 min | Decision: add engine or not |
| | **── Integration (only if llama.cpp wins) ──** | | |
| 10 | Write `LlamaCppGemmaEngine` | 15 min | New engine file |
| 11 | Write `start_server.sh` | 5 min | Server startup |
| 12 | Export merged GGUF (if adapter needed) | 20 min | GGUF file |
| 13 | Write `stt.py` | 10 min | STT module |
| 14 | Spot-check both backends | 5 min | Theological accuracy |
| 15 | Live test with mic | 15 min | End-to-end validation |

**If llama.cpp loses:** Skip tasks 10-12, use winning STT config with existing `CUDAGemmaStreamingEngine`. Total time: ~1 hour.

**If llama.cpp wins:** Full integration. Total time: ~2.5 hours.

---

## VRAM budget (worst case: S2 + T4)

| Component | VRAM |
|-----------|------|
| Whisper Large V3 (target) | ~3.0 GB |
| distil-v3.5 (draft, shared encoder) | ~1.2 GB |
| TG 4B Q4_K_M (target, llama-server) | ~2.5 GB |
| Gemma 4 E2B Q4_K_M (draft) | ~1.3 GB |
| CUDA overhead | ~0.5 GB |
| **Total** | **~8.5 GB** |
| **Headroom on 12 GB** | **~3.5 GB** |

Even the heaviest configuration fits comfortably.

---

## Files

| File | Action | Purpose |
|------|--------|---------|
| `bench_stt.py` | **New** | STT S1 benchmark |
| `bench_stt_spec.py` | **New** | STT S2 benchmark |
| `bench_translate_hf.py` | **New** | Translation T1 benchmark (existing engine) |
| `start_server.sh` | **New** (if llama.cpp wins) | llama-server startup |
| `engines/llamacpp_engine.py` | **New** (if llama.cpp wins) | llama.cpp translation engine |
| `engines/factory.py` | **Modified** (if llama.cpp wins) | Register new engine type |
| `stt.py` | **New** | Whisper module (both modes) |
| `spot_check.py` | **New** | Theological term check |
| `export_merged_gguf.py` | **New** (if llama.cpp wins + adapter) | Merge LoRA → GGUF |

---

## What NOT to do

- **Don't replace the HF engines.** They work. If llama.cpp wins, *add* it as an option.
- **Don't skip the T1 baseline.** The whole point is to know whether llama.cpp actually improves over what you have.
- **Don't merge adapters unless llama.cpp wins the benchmark.** The HF path loads adapters at runtime via PEFT — zero friction for training iteration.
- **No TurboQuant.** KV cache is not the bottleneck at 512-token context. llama.cpp's `-ctk q8_0` already covers this for free.
- **No Gemma 4 E4B today.** Post-sermon summaries and verse extraction are batch tasks — separate session.
- **No fine-tuning.** WSL desktop, separate session. W12+ and S6 adapters are already trained.
- **No containerization yet.** Get the benchmark numbers first. Containerize the winner.

---

## Architecture after this session

```
┌─────────────┐     ┌──────────────────────────────────────────────┐
│   Browser    │◄────│  Python pipeline (WebSocket :8765)           │
│  ab_display  │     │                                              │
│   .html      │     │  Mic → Silero VAD                            │
└─────────────┘     │       │                                       │
                    │  STT Engine (via engines/factory.py)          │
                    │  [S1: FasterWhisperEngine, ~0.8 GB]           │
                    │  [S2: HF spec decode, ~4.2 GB]                │
                    │       │ english text                           │
                    │       ▼                                        │
                    │  Translation Engine (via engines/factory.py)  │
                    │  [T1: CUDAGemmaStreamingEngine — existing]     │
                    │  [T2-T4: LlamaCppGemmaEngine — if it wins]    │
                    │       │ spanish text                           │
                    │       ▼                                        │
                    │  WebSocket → Browser                          │
                    └──────────────────────────────────────────────┘

If llama.cpp wins:
  llama-server :8080 (separate process, GPU, CUDA):
    Target: TranslateGemma 4B Q4_K_M (~2.5 GB)
    + n-gram spec OR E2B draft
    + KV cache quant (-ctk q8_0)

If HF wins:
  No external process. CUDAGemmaStreamingEngine runs in-process
  with prompt cache + optional assistant_model for 12B spec decode.
```

---

## Benchmark results template

Copy into `BENCHMARK.md` after running:

```markdown
# Benchmark Results — [DATE]

## Hardware
GPU: RTX 3060 12GB
CUDA: [version]
llama.cpp: [commit hash]

## STT (10s sermon clip, 3 runs avg)
| Config | Model | Latency | VRAM | Notes |
|--------|-------|---------|------|-------|
| S1 | faster-whisper distil-v3.5 INT8 | ___ms | ~0.8 GB | |
| S2 | HF spec (distil-v3.5 → Large V3) | ___ms | ~4.2 GB | |
| **Winner** | **S___** | | | |

## Translation (5 phrases, avg tok/s)
| Config | Setup | tok/s | Accept% | VRAM | Notes |
|--------|-------|-------|---------|------|-------|
| T1 | HF CUDAGemmaStreamingEngine + cache | ___ | N/A | ~3.0 GB | **BASELINE** |
| T2 | llama.cpp TG 4B baseline | ___ | N/A | ~2.5 GB | |
| T3 | llama.cpp TG 4B + n-gram | ___ | ___% | ~2.5 GB | |
| T4 | llama.cpp TG 4B + E2B draft | ___ | ___% | ~3.8 GB | or TOKENIZER_FAIL |
| **Winner** | **T___** | | | | |

## Decision
- [ ] llama.cpp beats T1 by >30% → add LlamaCppGemmaEngine
- [ ] llama.cpp ties or loses → keep CUDAGemmaStreamingEngine, skip integration

## Theological spot-check (8 terms)
| Term | HF engine | llama.cpp | Expected |
|------|-----------|-----------|----------|
| atonement | ✓/✗ | ✓/✗ | expiación |
| covenant | ✓/✗ | ✓/✗ | pacto |
| ... | | | |
| **Score** | __/8 | __/8 | |

## Combined pipeline (winners)
STT S___ + Translation T___ = ___ms + ___ms = ___ms E2E
Total VRAM: ___ GB / 12 GB
```

---

## Future sessions (separate plans)

- **12B offline evaluation:** Batch translate logged transcripts with TG 12B, compare against 4B outputs. If llama.cpp is the winner, use TG 12B GGUF via llama-server with TG 4B as draft (`-md`).
- **Adapter iteration:** Next Whisper W15+ or TranslateGemma training cycle on WSL. If using llama.cpp, add `export_merged_gguf.py` to the adapter deployment pipeline in `docs/deploy.md`.
- **Desktop GPU upgrade:** If 4070 Ti Super, re-run T1 baseline — the HF engine with 672 GB/s bandwidth may be fast enough to skip llama.cpp entirely.
- **Gemma 4 E4B:** Post-sermon summaries + verse extraction. Batch task, separate session.
- **Containerization:** Two containers if llama.cpp wins (server + pipeline). One container if HF wins (everything in-process). Either way, simpler after the benchmark settles the question.
