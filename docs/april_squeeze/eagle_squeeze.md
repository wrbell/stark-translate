# EAGLE-3 Draft Head Training Plan — TranslateGemma 4B on A2000 Ada

> **Goal:** Train a custom EAGLE-3 speculative decoding head for TranslateGemma 4B,
> using existing sermon + Bible data, to accelerate EN→ES inference on NVIDIA hardware.
>
> **Machine:** Windows Desktop, WSL2/Ubuntu, NVIDIA A2000 Ada (16GB VRAM), 64GB RAM
>
> **Target model:** `google/translategemma-4b-it` (Gemma 3 architecture, ~3GB in NF4)
>
> **Expected outcome:** 1.5–2.5× latency reduction on single-stream translation,
> lossless output (mathematically identical to vanilla decoding)
>
> **Time estimate:** ~15–25 hours total (8–12 engineering, 4–8 GPU, rest verification)

---

## Gate 0: Benchmark Existing Speculative Decoding First

**Do NOT start EAGLE-3 work until this benchmark is complete.**

Your `CUDAGemmaStreamingEngine` already supports HuggingFace's native speculative
decoding via `assistant_model=`. This path is fully wired but untested on your hardware.
Test it before investing in EAGLE-3.

```python
# In your existing codebase — no new code needed:
from engines.cuda_engine import CUDAGemmaStreamingEngine

engine = CUDAGemmaStreamingEngine(
    model_id="google/translategemma-12b-it",
    assistant_model_id="google/translategemma-4b-it",  # 4B drafts for 12B
    use_prompt_cache=True,
)
engine.load()

# Benchmark with your existing canary sentences from benchmark_gemma4.py
import time
test = [
    "And you know, when we think about the atonement — what Christ did for us on that cross...",
    "God made a covenant with Abraham, and brothers, He keeps His promises.",
    "It's only by grace, friends. We can't earn it — it's grace through faith.",
    # ... full CANARY_SENTENCES list from benchmark_gemma4.py
]
for phrase in test:
    t0 = time.perf_counter()
    result = engine.translate(phrase)
    ms = (time.perf_counter() - t0) * 1000
    print(f"  {ms:.0f}ms | {phrase[:50]}... → {result.text[:50]}...")
```

Also test Gemma 4 E2B as assistant for TranslateGemma — your `benchmark_gemma4.py`
already loads E2B, so the tokenizer compatibility question can be answered empirically:

```python
engine_e2b = CUDAGemmaStreamingEngine(
    model_id="google/translategemma-4b-it",
    assistant_model_id="google/gemma-4-e2b-it",  # cross-family draft
)
```

### Decision Matrix

| Existing HF spec decoding result | Action |
|----------------------------------|--------|
| 12B+4B assistant < 1.5s/sentence | **Stop here.** Ship HF spec decoding. EAGLE not worth the SGLang migration. |
| 12B+4B assistant 1.5–2.5s | EAGLE-3 worth exploring if you need < 1s. Proceed to Phase 1. |
| 12B+4B assistant > 2.5s or errors | HF spec decoding broken/slow on A2000. EAGLE-3 on SGLang is the only spec-decode path. Proceed. |
| 4B standalone already < 500ms | **Stop here.** No speculative decoding needed. 4B alone meets target. |

Your observed baseline: "TranslateGemma translation (4B or 12B) ~2–3s/input" (training/CLAUDE.md).

---

## TurboQuant Assessment: Not Applicable

TurboQuant (ICLR 2026) compresses the KV cache — the memory that stores previously
computed attention states during long-sequence generation. It achieves quality neutrality
at 3.5 bits per channel and up to 8× attention logit speedup on H100 GPUs.

**TurboQuant does not help your use case.** Here's why:

Your translation prompts are ~30–40 tokens (chat template prefix + source sentence).
Your translation outputs are ~40–80 tokens (a Bible verse or sermon sentence in Spanish).
Total sequence length: ~70–120 tokens.

TurboQuant's benefits scale with sequence length. At 128K tokens, the KV cache dominates
memory and attention computation. At 120 tokens, the KV cache is negligible — a few MB
at most. Your bottleneck is the per-token autoregressive forward pass through 34 decoder
layers, not the attention lookup over a tiny KV cache.

The 8× attention speedup reported on H100s applies to the attention kernel specifically,
which at short sequences is a small fraction of total inference time. On an A2000 Ada with
much less memory bandwidth, the relative gains would be even smaller.

TurboQuant also requires custom Triton attention kernels to realize its gains. These
kernels exist for H100 (with JAX) and have community implementations for RTX 4090/5090,
but not for the A2000 Ada's Ampere architecture. The community `turboquant-pytorch`
repo notes that "hybrid decode dequantizes all history" and the fused kernels aren't
used in the hybrid path yet — meaning the theoretical speedup isn't realized in practice
even on supported hardware.

**Where TurboQuant would matter for you:** If you later scale to paragraph-level or
document-level translation (feeding full sermon transcripts as context), the KV cache
grows substantially. At that point, TurboQuant becomes relevant. For sentence-level
real-time translation, it's not.

---

## Prerequisites

Before starting this plan, the following must already be complete:

- [ ] **Gate 0 benchmark complete** — HF speculative decoding results recorded
- [ ] WSL2 + CUDA 12.x installed, `nvidia-smi` shows A2000 Ada
- [ ] Python 3.12 venv with PyTorch CUDA, Transformers, bitsandbytes
- [ ] TranslateGemma 4B downloaded and verified (`google/translategemma-4b-it`)
- [ ] Expanded sermon chunks available (`ablation/sermon_whisper_chunks_expanded.json`)
- [ ] Bible verse pairs exported (`bible_data/aligned/verse_pairs_train.jsonl`)
- [ ] Deepgram transcripts available (`stark_data/deepgram_transcripts/*.deepgram.json`)

---

## Architecture Context

### Why This Is Novel (and What Makes It Hard)

No public EAGLE-3 head exists for any Gemma 3-based model. SpecForge and the SafeAILab
EAGLE repo ship configs for LLaMA, Qwen, and (via Thoughtworks' fork) Gemma 4. TranslateGemma
inherits Gemma 3's architecture: a 5:1 local/global attention pattern (5 sliding-window layers
with window=1024 followed by 1 global attention layer). This hybrid attention is structurally
similar to — but distinct from — Gemma 4's hybrid attention that Thoughtworks already solved.

The key engineering task is writing a SpecForge-compatible config and potentially patching the
hidden state extraction to handle Gemma 3's specific layer layout. Thoughtworks' Gemma 4 work
is the closest reference and required fixing three bugs in SGLang's KV cache handling. Expect
similar (but not identical) issues for Gemma 3.

### EAGLE-3 Head Mechanics

The head is a tiny module (~200–300MB) that:
1. Reads TranslateGemma's hidden states from three internal layers (early, mid, late)
2. Fuses them via a learned FC layer (3 × hidden_dim → hidden_dim)
3. Runs 1–2 lightweight transformer decoder layers to predict the next feature vector
4. Uses TranslateGemma's frozen LM head to convert features → token IDs
5. TranslateGemma verifies the proposed tokens in a single batched forward pass

Output is mathematically identical to vanilla decoding — the head only affects speed.

### Why Translation Should Excel

Translation output is low-entropy compared to open-ended chat. The target language is known,
sentence structure is predictable, and the vocabulary is constrained (especially for biblical text).

Red Hat's finding that EAGLE-3 "performs poorly on translation" used a head trained on ShareGPT
chat data — the distribution mismatch was the problem, not translation itself. A head trained
on actual EN→ES translation hidden states should perform strongly.

### Integration Architecture

EAGLE-3 requires SGLang as a serving backend. This does NOT replace your existing
`CUDAGemmaStreamingEngine` — it runs alongside it as a new engine class.

Following the 4-step pattern from `engines/CLAUDE.md`:

1. **Implement the ABC:** `SGLangEagleEngine(TranslationEngine)` in `engines/sglang_engine.py`
2. **Required methods:** `load()` starts SGLang server subprocess, `translate()` hits
   `http://localhost:30000/v1/chat/completions`, `unload()` kills the subprocess
3. **Register in factory.py:** `engine_type="eagle"` branch in `create_translation_engine()`
4. **Mock in tests:** Add `"sglang"` to `_MOCK_MODULES` in `tests/conftest.py`

Your existing `CUDAGemmaStreamingEngine` stays untouched. The factory routes to the
right engine based on config. Prompt caching, streaming, EOS fix — all of that keeps
working for the non-EAGLE path.

Thread model note: SGLang runs as a separate process, so `SGLangEagleEngine` is just
an HTTP client. No thread pool needed — unlike `CUDAGemmaStreamingEngine` which uses
`ThreadPoolExecutor(max_workers=2)` for STT/Translation overlap. If both engines coexist,
the factory handles the difference transparently.

---

## Phase 0: Framework Setup (~2–3 hours)

### Install SpecForge + SGLang

```bash
cd ~/stt_project
source stt_train_env/bin/activate

# Clone SpecForge (LMSYS official)
git clone https://github.com/sgl-project/SpecForge.git
cd SpecForge
pip install -e .

# Clone Thoughtworks' SGLang fork for Gemma architecture reference
git clone https://github.com/tails-mpt/sglang.git tw-sglang

# SGLang itself
pip install "sglang[all]"
```

### Gemma 3 Architecture Config

SpecForge needs a JSON config describing the draft head architecture. No Gemma 3 config
exists — you must create one from TranslateGemma 4B's dimensions.

TranslateGemma 4B (Gemma 3) key parameters:
- Hidden size: 2560
- Num attention heads: 10
- Num KV heads: 2 (GQA)
- Intermediate size: 10240
- Num layers: 34 (28 local sliding-window + 6 global, in 5:1 pattern)
- Vocab size: 262144
- RoPE theta: 1,000,000 (global layers), 10,000 (local layers)

```json
{
    "model_type": "gemma3",
    "hidden_size": 2560,
    "num_attention_heads": 10,
    "num_key_value_heads": 2,
    "intermediate_size": 10240,
    "num_hidden_layers": 1,
    "vocab_size": 262144,
    "eagle3_num_layers": 1,
    "eagle3_low_layer": 5,
    "eagle3_mid_layer": 17,
    "eagle3_high_layer": 33,
    "eagle3_layer_types": {
        "5": "local_sliding_window_1024",
        "17": "local_sliding_window_1024",
        "33": "global_full_attention"
    },
    "fc_hidden_size": 7680,
    "rope_theta_local": 10000.0,
    "rope_theta_global": 1000000.0,
    "head_dim": 256,
    "max_position_embeddings": 8192,
    "rms_norm_eps": 1e-06
}
```

Layer selection rationale:
- Layer 5 (~15% depth): local attention layer — captures surface-level token patterns
- Layer 17 (~50% depth): local attention layer — syntactic/semantic structure
- Layer 33 (~97% depth): **global** attention layer — near-output translation decisions

**Critical:** The `eagle3_layer_types` field is non-standard. SpecForge may not use it,
but you need this information when patching the hidden state extraction. Local and global
layers have different hidden state distributions because local layers only attend within
a 1024-token window. If acceptance rates are low, try swapping layer 17 for layer 16
(the global layer in the 3rd block: layers 0–5 local + layer 5 global, 6–11 local +
layer 11 global, 12–17 local + **layer 17 is local, layer 16 is the global**).

Correction on layer indexing: with 5:1 pattern in 34 layers, global layers are at
indices 5, 11, 17, 23, 29, 33. So layer 17 IS a global layer. Verify by inspecting:

```python
from transformers import AutoConfig
config = AutoConfig.from_pretrained("google/translategemma-4b-it")
# Check config.text_config.sliding_window_pattern or equivalent
```

### Verify TranslateGemma Hidden States

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "google/translategemma-4b-it",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    output_hidden_states=True,
)
tokenizer = AutoTokenizer.from_pretrained("google/translategemma-4b-it")

# --- EOS fix (CRITICAL — without this, generation produces 256 pad tokens) ---
eot_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
tokenizer._eos_token_ids = {tokenizer.eos_token_id, eot_id}

messages = [{"role": "user", "content": [
    {"type": "text", "source_lang_code": "en",
     "target_lang_code": "es",
     "text": "For God so loved the world."}
]}]
input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model(**inputs)

print(f"Num hidden state layers: {len(outputs.hidden_states)}")
for layer_idx in [5, 17, 33]:
    hs = outputs.hidden_states[layer_idx]
    print(f"  Layer {layer_idx}: shape={hs.shape}, dtype={hs.dtype}, "
          f"mean={hs.float().mean():.4f}, std={hs.float().std():.4f}")
```

If this fails or hidden states have unexpected shapes, TranslateGemma may need a custom
modeling file. Check `transformers.models.gemma3.modeling_gemma3` source.

---

## Phase 1: Training Data Preparation (~1–2 hours)

### What EAGLE-3 Training Data Is

EAGLE-3 does NOT train on input→output translation pairs. It trains on the target model's
own hidden states while generating translations. The training loop captures what
TranslateGemma's internal layers look like at each generation step, then teaches the
draft head to predict those patterns.

### Load From Your Existing Data

Your data lives in two places, not the directory structure from the old plan:

```python
"""
prepare_eagle_data.py — Convert existing data to EAGLE training prompts.
Only needs the English source side. TranslateGemma generates translations
during hidden state collection (Phase 2).
"""
import json
import random

def load_sermon_chunks(path="ablation/sermon_whisper_chunks_expanded.json",
                       min_chars=20):
    """Load from the expanded whisper chunks JSON (actual format)."""
    with open(path) as f:
        chunks = json.load(f)
    # Filter by length, deduplicate
    seen = set()
    sources = []
    for chunk in chunks:
        text = chunk.get("en", "").strip()
        if len(text) >= min_chars and text not in seen:
            seen.add(text)
            sources.append(text)
    return sources

def load_bible_sources(path="bible_data/aligned/verse_pairs_train.jsonl"):
    """Extract English source sentences from Bible parallel corpus."""
    sources = []
    with open(path) as f:
        for line in f:
            pair = json.loads(line)
            sources.append(pair["en"])
    return sources

def build_eagle_dataset(output_path="eagle_data/translation_prompts.jsonl",
                        max_samples=50000):
    """
    Build a balanced dataset mixing Bible + sermon sources.
    Target: 30K–50K samples.

    Mix: 60% Bible (domain vocabulary), 40% sermon (natural speech patterns)
    """
    bible = load_bible_sources()
    sermons = load_sermon_chunks()

    print(f"Bible sources: {len(bible)}")
    print(f"Sermon sources: {len(sermons)}")

    bible_n = min(int(max_samples * 0.6), len(bible))
    sermon_n = min(int(max_samples * 0.4), len(sermons))

    bible_sample = random.sample(bible, bible_n)
    sermon_sample = random.sample(sermons, sermon_n)

    all_sources = bible_sample + sermon_sample
    random.shuffle(all_sources)

    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        for src in all_sources:
            f.write(json.dumps({"en": src}, ensure_ascii=False) + "\n")

    print(f"Wrote {len(all_sources)} prompts to {output_path}")

if __name__ == "__main__":
    build_eagle_dataset()
```

### Disk Space for Offline Training

Per sample: 3 layers × 2560 dims × 2 bytes (bf16) × ~40 output tokens ≈ 620 KB.
50K samples ≈ 31 GB. Store on native Linux filesystem (`~/`), NOT `/mnt/c/`.

---

## Phase 2: Hidden State Generation (~3–5 hours GPU)

Load TranslateGemma 4B in bf16, generate translations, capture hidden states.

**CRITICAL:** Two fixes from your codebase must be applied here:

1. **EOS fix** — without adding `<end_of_turn>` (id=106) to `_eos_token_ids`,
   the model generates 256 pad tokens per sample, wasting ~5s and capturing
   garbage hidden states for the pad region.

2. **Precision matching** — generate in bf16 (not fp16, which causes inf/nan on
   MPS per your `engines/CLAUDE.md`, and likely has similar issues on CUDA for
   logit-sensitive operations). If you serve via SGLang in bf16, train in bf16.
   Thoughtworks measured 32% hidden state divergence from precision mismatch.

```python
"""
generate_hidden_states.py — Offline hidden state extraction.
VRAM: ~8–10 GB (model in bf16 + generation buffer)
Time: ~3–5 hours for 50K samples on A2000 Ada
"""
import torch
import json
import os
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

LAYERS_TO_CAPTURE = [5, 17, 33]
OUTPUT_DIR = "eagle_data/hidden_states"
MAX_NEW_TOKENS = 128

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading TranslateGemma 4B in bf16...")
    model = AutoModelForCausalLM.from_pretrained(
        "google/translategemma-4b-it",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        output_hidden_states=True,
    )
    tokenizer = AutoTokenizer.from_pretrained("google/translategemma-4b-it")
    model.eval()

    # --- EOS fix (from engines/cuda_engine.py) ---
    eot_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
    default_eos = tokenizer.eos_token_id
    tokenizer._eos_token_ids = {default_eos, eot_id}
    print(f"EOS fix applied: added <end_of_turn> (id={eot_id})")

    # Load prompts
    prompts = []
    with open("eagle_data/translation_prompts.jsonl") as f:
        for line in f:
            prompts.append(json.loads(line)["en"])

    print(f"Generating hidden states for {len(prompts)} prompts...")

    for idx, en_text in enumerate(tqdm(prompts)):
        messages = [{"role": "user", "content": [
            {"type": "text", "source_lang_code": "en",
             "target_lang_code": "es", "text": en_text}
        ]}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Generate with hidden state capture
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )

        # Extract hidden states from generation steps
        # outputs.hidden_states is a tuple of (num_gen_steps) tuples of (num_layers) tensors
        gen_hidden = {layer: [] for layer in LAYERS_TO_CAPTURE}
        for step_hidden in outputs.hidden_states[1:]:  # skip prompt prefill
            for layer in LAYERS_TO_CAPTURE:
                hs = step_hidden[layer][:, -1, :].cpu()  # last position only
                gen_hidden[layer].append(hs)

        # Stack and save
        token_ids = outputs.sequences[0][inputs.input_ids.shape[1]:].cpu().tolist()

        sample_dir = os.path.join(OUTPUT_DIR, f"sample_{idx:06d}")
        os.makedirs(sample_dir, exist_ok=True)

        json.dump({"token_ids": token_ids, "source": en_text},
                  open(os.path.join(sample_dir, "meta.json"), "w"))

        for layer, states in gen_hidden.items():
            stacked = torch.cat(states, dim=0).to(torch.bfloat16)
            np.save(os.path.join(sample_dir, f"layer_{layer}.npy"),
                    stacked.numpy())

        if (idx + 1) % 1000 == 0:
            print(f"  [{idx+1}/{len(prompts)}] Saved to {sample_dir}")

if __name__ == "__main__":
    main()
```

---

## Phase 3: Train the EAGLE-3 Head (~2–4 hours GPU)

The draft head is small (~50–80M params, ~200–300MB):
- FC fusion: 7680 → 2560 (three-layer concatenation)
- 1 transformer decoder layer (same dims as TranslateGemma's layers)
- Output: predicted hidden state → frozen LM head → token logits

VRAM: ~4–6 GB (draft head only in offline mode; target model NOT loaded).

### If SpecForge Supports Gemma 3

```bash
torchrun --standalone --nproc_per_node 1 \
    SpecForge/scripts/train_eagle3_offline.py \
    --target-model-path google/translategemma-4b-it \
    --draft-model-config eagle_data/gemma3-4b-eagle3.json \
    --hidden-states-dir eagle_data/hidden_states/ \
    --output-dir outputs/translategemma-4b-eagle3 \
    --num-epochs 1 \
    --batch-size 4 \
    --learning-rate 1e-4 \
    --max-length 1024
```

### If SpecForge Needs Gemma 3 Patches

Most likely scenario. The adaptation path:

1. Start from Thoughtworks' Gemma 4 config as reference
2. Copy and modify `modeling_gemma3.py` from HuggingFace Transformers to expose
   hidden states at layers 5, 17, 33 (annotate with `# [MODIFIED]`)
3. Handle the 5:1 attention pattern: local layers use sliding-window (window=1024),
   global layers use full attention. The draft head reads hidden states (not running
   attention), so this mostly affects KV cache at *inference* time, not training.
4. Write the config JSON from Phase 0

Alternative: SafeAILab EAGLE repo with DeepSpeed:
```bash
cd EAGLE/eagle/traineagle3
deepspeed main.py --deepspeed_config ds_config.json
```

DeepSpeed config for single A2000 Ada:
```json
{
    "train_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "bf16": {"enabled": true},
    "zero_optimization": {"stage": 0},
    "gradient_clipping": 1.0
}
```

---

## Phase 4: Validation (~2–3 hours)

### Acceptance Rate Measurement

The critical metric. Directly determines speedup.

| α (acceptance rate) | Avg accepted/cycle | Est. speedup | Verdict |
|---------------------|-------------------|--------------|---------|
| < 50%               | < 2.0             | < 1.3×       | Retrain with different layers or more data |
| 50–70%              | 2.0–3.0           | 1.3–1.7×     | Acceptable; tune layers for improvement |
| 70–85%              | 3.0–4.5           | 1.7–2.2×     | Good — expected for domain-matched translation |
| 85%+                | 4.5+              | 2.2×+        | Excellent |

### Use Your Existing Eval Infrastructure

Do NOT write a standalone benchmark script. Your codebase has three-tier evaluation:

**Tier 1 (verse holdout):** Use `evaluate_translation.py` with the EAGLE-served model.
Compare BLEU/chrF++/COMET against your existing TranslateGemma results. Outputs must
be identical (EAGLE is lossless) — any difference indicates a bug.

**Tier 2 (sermon chunks):** Use `evaluate_sermon.py` with `--adapter` pointing to the
EAGLE-served model via the new `SGLangEagleEngine`. The dual-ceiling methodology
(12B + DeepL) and kill-switch verdicts apply unchanged.

**Tier 3 (theological canary):** The same 8 sentences from `benchmark_gemma4.py` and
`evaluate_sermon.py`:
```python
CANARY_SENTENCES = [
    ("And you know, when we think about the atonement — ...", "expiación"),
    ("God made a covenant with Abraham, ...", "pacto"),
    # ... same 8 sentences used across your eval suite
]
```

The metric that matters for EAGLE is **latency**, not translation quality (which must be
identical). Record:
- End-to-end latency per sentence (ms)
- Tokens per second
- Acceptance rate α (from SGLang Prometheus metrics)
- Average tokens accepted per verification cycle

### Tuning Knobs (If Acceptance Is Low)

In priority order:

1. **Layer selection:** Try different low/mid/high extraction points. Swap mid layer
   between local (17) and adjacent global (17 if global, or 11). 9 combinations ×
   30min each = half a day.

2. **More training data:** Scale from 30K to 50K if α < 60%.

3. **Training epochs:** Try 2 epochs if α < 60% with 1 epoch.

4. **Precision mismatch check:** Compare hidden states from your generation script vs
   SGLang's output. If > 5% divergence, regenerate using SGLang's data generation.

---

## Phase 5: Integration as New Engine (~3–4 hours)

### `engines/sglang_engine.py`

```python
"""SGLang EAGLE-3 backend for accelerated TranslateGemma inference."""

import logging
import subprocess
import time
from typing import Any

import requests

from engines.base import TranslationEngine, TranslationResult

logger = logging.getLogger(__name__)


class SGLangEagleEngine(TranslationEngine):
    """Translation engine wrapping SGLang with EAGLE-3 speculative decoding.

    Runs SGLang as a subprocess server. Translation calls hit the OpenAI-
    compatible /v1/chat/completions endpoint.

    Constructor args:
        model_id:       TranslateGemma model repo.
        eagle_head_path: Path to trained EAGLE-3 draft head.
        port:           SGLang server port (default: 30000).
        dtype:          Model precision (default: "bfloat16").
    """

    def __init__(
        self,
        model_id: str = "google/translategemma-4b-it",
        eagle_head_path: str = "outputs/translategemma-4b-eagle3/epoch_0",
        port: int = 30000,
        dtype: str = "bfloat16",
    ):
        self._model_id_str = model_id
        self._eagle_head_path = eagle_head_path
        self._port = port
        self._dtype = dtype
        self._process = None
        self._loaded = False

    def load(self) -> None:
        """Start SGLang server subprocess with EAGLE-3 config."""
        cmd = [
            "python", "-m", "sglang.launch_server",
            "--model-path", self._model_id_str,
            "--speculative-algorithm", "EAGLE3",
            "--speculative-draft-model-path", self._eagle_head_path,
            "--speculative-num-steps", "3",
            "--speculative-eagle-topk", "4",
            "--speculative-num-draft-tokens", "8",
            "--dtype", self._dtype,
            "--mem-fraction-static", "0.8",
            "--cuda-graph-max-bs", "1",
            "--context-length", "4096",
            "--host", "127.0.0.1",
            "--port", str(self._port),
        ]

        logger.info("Starting SGLang EAGLE-3 server: %s", " ".join(cmd))
        self._process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )

        # Wait for server to be ready (poll /health endpoint)
        url = f"http://127.0.0.1:{self._port}/health"
        for attempt in range(120):  # 2 min timeout
            try:
                resp = requests.get(url, timeout=1)
                if resp.status_code == 200:
                    logger.info("SGLang server ready on port %d", self._port)
                    self._loaded = True
                    return
            except requests.ConnectionError:
                pass
            time.sleep(1)

        raise RuntimeError("SGLang server failed to start within 2 minutes")

    def translate(
        self,
        text: str,
        *,
        source_lang: str = "en",
        target_lang: str = "es",
    ) -> TranslationResult:
        """Translate via SGLang's OpenAI-compatible API."""
        if not self._loaded:
            raise RuntimeError("Engine not loaded -- call load() first")

        input_words = len(text.split())
        max_tok = max(32, int(input_words * 1.8))

        messages = [{"role": "user", "content": [
            {"type": "text", "source_lang_code": source_lang,
             "target_lang_code": target_lang, "text": text}
        ]}]

        t0 = time.perf_counter()
        resp = requests.post(
            f"http://127.0.0.1:{self._port}/v1/chat/completions",
            json={
                "model": self._model_id_str,
                "messages": messages,
                "max_tokens": max_tok,
                "temperature": 0,
            },
            timeout=30,
        )
        latency_ms = (time.perf_counter() - t0) * 1000

        data = resp.json()
        translation = data["choices"][0]["message"]["content"]
        clean = translation.split("<end_of_turn>")[0].strip()

        # Estimate tokens/s from usage stats if available
        usage = data.get("usage", {})
        out_tokens = usage.get("completion_tokens", len(clean.split()))
        tps = out_tokens / (latency_ms / 1000) if latency_ms > 0 else 0.0

        return TranslationResult(
            text=clean,
            latency_ms=latency_ms,
            tokens_per_second=tps,
        )

    def unload(self) -> None:
        """Kill the SGLang server subprocess."""
        if self._process:
            self._process.terminate()
            self._process.wait(timeout=10)
            self._process = None
        self._loaded = False
        logger.info("SGLangEagleEngine unloaded")

    @property
    def model_id(self) -> str:
        return f"{self._model_id_str}+eagle3"

    @property
    def backend(self) -> str:
        return "sglang"
```

### Register in `factory.py`

Add to `create_translation_engine()`:

```python
elif engine_type == "eagle":
    from engines.sglang_engine import SGLangEagleEngine
    return SGLangEagleEngine(
        model_id=model_id or "google/translategemma-4b-it",
        **kwargs,
    )
```

### Add to `tests/conftest.py`

```python
_MOCK_MODULES = [..., "sglang"]
```

---

## Phase 6: Benchmarking (~1–2 hours)

### Side-by-Side Comparison

Run the same test set through all three paths:

1. **Vanilla:** `CUDAGemmaEngine` (no spec decoding)
2. **HF Speculative:** `CUDAGemmaStreamingEngine` with `assistant_model`
3. **EAGLE-3:** `SGLangEagleEngine`

Use the same CANARY_SENTENCES + verse holdout + sermon chunks from your existing evals.

### Metrics to Record

| Metric | Source | Target |
|--------|--------|--------|
| End-to-end latency (ms) | Timer around `translate()` | < 1000ms |
| Tokens per second | `TranslationResult.tokens_per_second` | > 60 tok/s |
| Acceptance rate (α) | SGLang Prometheus `/metrics` | > 70% |
| Avg tokens accepted/cycle | SGLang metrics | > 3.0 |
| Translation quality | Must be IDENTICAL to vanilla | Lossless check |
| Theological term accuracy | Same 8 canary sentences | Identical to vanilla |
| VRAM usage | `nvidia-smi` during serving | < 14 GB |

### Lossless Verification

EAGLE-3 is mathematically lossless. Run 500 verse pairs through both vanilla and EAGLE,
diff the outputs. Any difference is a bug:

```python
mismatches = 0
for en, es_ref in verse_pairs:
    vanilla = vanilla_engine.translate(en).text
    eagle = eagle_engine.translate(en).text
    if vanilla != eagle:
        mismatches += 1
        print(f"MISMATCH: {en[:50]}...")
        print(f"  Vanilla: {vanilla[:80]}")
        print(f"  EAGLE:   {eagle[:80]}")
assert mismatches == 0, f"{mismatches} lossless violations!"
```

---

## VRAM Budget (Observed A2000 Ada Numbers)

From `training/CLAUDE.md` actual measurements, NOT estimates:

| Component | VRAM (NF4) | Source |
|-----------|-----------|--------|
| TranslateGemma 4B | ~6–8 GB | "TranslateGemma 4B load (4-bit QLoRA) ~6-8 GB" |
| TranslateGemma 12B | ~7 GB | "TranslateGemma 12B load (4-bit) ~7 GB" |
| EAGLE-3 head | ~0.3 GB | ~50-80M params in bf16 |
| SGLang overhead (CUDA graphs, KV cache) | ~2–4 GB | Estimated — measure during Phase 5 |
| **Total (4B + EAGLE + SGLang)** | **~9–12 GB** | **Fits in 16 GB** |

Note: SGLang's memory profile differs from HuggingFace's. CUDA graph compilation
and KV cache pools may use more than expected. Monitor with `nvidia-smi` during
initial server startup.

---

## Risk Register

### High Risk

| Risk | Mitigation |
|------|------------|
| **SpecForge doesn't support Gemma 3** | Use SafeAILab EAGLE repo with custom `modeling_gemma3_kv.py`; reference Thoughtworks' Gemma 4 patches. Budget 4–6 extra hours. |
| **Hidden state divergence (train vs serve)** | Generate states using SGLang's own offline pipeline if possible. If using HF, verify states match SGLang output on 10 samples before full generation. |
| **SGLang crashes on Gemma 3's 5:1 attention** | Gemma 4's hybrid attention required KV cache fixes. Gemma 3's pattern is different. Check Thoughtworks' fork for reference patches. |
| **SGLang doesn't handle TranslateGemma's chat template** | TranslateGemma uses `source_lang_code`/`target_lang_code` fields. May need a custom chat template handler in SGLang config. |

### Medium Risk

| Risk | Mitigation |
|------|------------|
| **Low acceptance rate (< 50%)** | Retune layer selection; increase data; check precision. |
| **SGLang server startup VRAM exceeds 16GB** | Reduce `mem-fraction-static` to 0.7; reduce `context-length` to 2048 (translation needs ~120 tokens max). |
| **HTTP overhead negates EAGLE gains** | Localhost HTTP adds ~1-2ms. If EAGLE saves 500ms+ per sentence, this is negligible. If gains are < 100ms, HTTP overhead matters. |

### Low Risk

| Risk | Mitigation |
|------|------------|
| **Disk space for hidden states** | 50K samples ≈ 31 GB. If tight, use online training mode (both model + head in VRAM — fits with 4B in 4-bit + head). |
| **Training too slow** | 1 epoch of 50K samples: 2–4 hours. If > 8 hours, reduce to 30K. |

---

## Timeline

| Phase | Task | GPU Hours | Human Hours |
|-------|------|-----------|-------------|
| Gate 0 | Benchmark existing HF spec decoding | 0.5 | 1 |
| 0 | Framework setup + Gemma 3 config | 0 | 2–3 |
| 1 | Data preparation | 0 | 1–2 |
| 2 | Hidden state generation (50K samples) | 3–5 | 0.5 |
| 3 | Train EAGLE-3 head | 2–4 | 0.5 |
| 4 | Validation + tuning | 1–2 | 2–3 |
| 5 | Engine integration (`sglang_engine.py`) | 0 | 3–4 |
| 6 | Benchmarking (3-way comparison) | 0.5 | 1–2 |
| **Total** | | **~7–12** | **~11–16** |

Run Phases 2 and 3 overnight with `tmux`. The A2000 Ada handles both comfortably.

---

## Sequencing Within the Broader Project

This plan sits AFTER the core pipeline works and fine-tuning is complete.

```
1. ✅ Base A/B test running (no fine-tuning)
2. ✅ Whisper LoRA (W1-W15 ablation complete)
3. ✅ TranslateGemma QLoRA (S1-S9 complete, S6 winner)
4. ✅ Fine-tuned pipeline meets quality targets
5. → Gate 0: Benchmark HF speculative decoding (this plan)
6. → EAGLE-3 training (this plan, Phases 0-6)
7.   Deploy via factory.py engine_type="eagle"
```

If vanilla TranslateGemma 4B in NF4 on A2000 Ada already meets your < 500ms target
for sermon sentences, neither HF speculative decoding nor EAGLE-3 is needed.
If you're at 2–3s and need sub-1s, Gate 0 determines whether the simpler HF path
suffices or EAGLE-3 is required.
