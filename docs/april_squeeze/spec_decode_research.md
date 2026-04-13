# Research: Speculative Decoding for Gemma 4 ↔ TG 4B and Whisper v3.5 → v3

## Context

Researching speculative decoding feasibility for two scenarios:
1. **Translation:** Gemma 4 E2B as draft for TranslateGemma 4B (or vice versa)
2. **STT:** Distil-Whisper v3.5 as draft for Whisper Large-V3 (or V3-Turbo)

## Key Finding: Universal Assisted Generation (UAG)

**HuggingFace transformers (v4.46+, we have v5.5.3) supports cross-tokenizer speculative decoding.** This is called [Universal Assisted Generation](https://huggingface.co/blog/universal_assisted_generation). It works by:

1. Draft model generates candidate tokens with its own tokenizer
2. Candidates are decoded to text, then re-encoded with the target tokenizer
3. Target model verifies using its own token IDs
4. Accepted tokens are converted back to draft tokenizer format for next iteration

This means **Gemma 4 E2B can draft for TranslateGemma 4B even if tokenizers differ** — just pass `tokenizer=`, `assistant_tokenizer=`, and `assistant_model=` to `generate()`.

## Scenario 1: Gemma 4 E2B ↔ TranslateGemma 4B

### Tokenizer Compatibility

- TranslateGemma is built on **Gemma 3** architecture (262K vocab)
- Gemma 4 E2B uses **Gemma 4** tokenizer (262K vocab, but with new control tokens)
- Both are Gemma family — likely share 99%+ of vocabulary
- Even if there are differences, **UAG handles this automatically** in transformers 5.x

### Prompt Format Difference

- TranslateGemma: custom template with `source_lang_code`/`target_lang_code` fields
- Gemma 4 E2B: generic instruct template

**This doesn't matter for UAG** — only the target model's prompt format is used. The draft model generates freely and the target verifies against its own tokenization.

### Which Direction?

Two options based on benchmark results:

**Option A: E2B drafts for TG 4B** (small drafts for specialized)
- E2B is faster (2.24s vs 2.46s) and lighter (but 6.3 GB vs 3.0 GB)
- Doesn't make sense — draft should be smaller/faster than target, but E2B is already faster AND better quality

**Option B: TG 4B drafts for E2B** (specialized drafts for general)
- TG 4B at 3.0 GB is smaller than E2B at 6.3 GB
- But TG 4B's lower quality (BLEU 39.3 vs 60.5) means low acceptance rate
- High rejection rate = no speedup

**Option C: Just use E2B standalone** — it's already faster and better than TG 4B. Spec decode between them adds complexity for questionable gain.

**Option D: E2B drafts for E4B** — E4B has best quality (7/8 canary, BLEU 62.2) but is too slow at NF4 (8.5s). E2B drafting could speed it up significantly. Same tokenizer family, higher acceptance rate.

### VRAM Requirements Per Scenario (Translation Only)

| Scenario | Draft VRAM | Target VRAM | Combined | On 16 GB A2000 | On 12 GB RTX 3060 | On 18 GB M3 Pro |
|----------|-----------|-------------|----------|----------------|-------------------|-----------------|
| **TG 4B standalone** | — | 3.0 GB | 3.0 GB | 13 GB free | 9 GB free | 15 GB free |
| **E2B standalone** | — | 6.3 GB | 6.3 GB | 9.7 GB free | 5.7 GB free | 11.7 GB free |
| **E4B standalone** | — | 15.0 GB | 15.0 GB | 1.4 GB free | OOM | OOM |
| **A: E2B drafts TG 4B** | 6.3 GB | 3.0 GB | 9.3 GB | 6.7 GB free | 2.7 GB free | 8.7 GB free |
| **B: TG 4B drafts E2B** | 3.0 GB | 6.3 GB | 9.3 GB | 6.7 GB free | 2.7 GB free | 8.7 GB free |
| **C: E2B standalone** | — | 6.3 GB | 6.3 GB | 9.7 GB free | 5.7 GB free | 11.7 GB free |
| **D: E2B drafts E4B** | 6.3 GB | 15.0 GB | 21.3 GB | **OOM** | **OOM** | **OOM** |
| **D (GGUF): E2B Q4 drafts E4B Q4** | ~1.5 GB | ~4.5 GB | ~6.0 GB | 10 GB free | 6 GB free | 12 GB free |

### VRAM Requirements: Full Pipeline (Translation + STT)

Adding Whisper Turbo STT (~1.5 GB MLX, ~0.9 GB CUDA faster-whisper) to each translation scenario:

| Scenario | STT | Translation (draft+target) | Overhead | Total | Fits 12 GB? | Fits 16 GB? | Fits 18 GB? |
|----------|-----|---------------------------|----------|-------|-------------|-------------|-------------|
| **TG 4B + Whisper** | 0.9 GB | 3.0 GB | 0.5 GB | 4.4 GB | Yes (7.6 free) | Yes | Yes |
| **E2B + Whisper** | 0.9 GB | 6.3 GB | 0.5 GB | 7.7 GB | Yes (4.3 free) | Yes | Yes |
| **E2B drafts TG 4B + Whisper** | 0.9 GB | 9.3 GB | 0.5 GB | 10.7 GB | Tight (1.3 free) | Yes (5.3 free) | Yes |
| **E4B + Whisper (NF4)** | 0.9 GB | 15.0 GB | 0.5 GB | 16.4 GB | OOM | Barely | OOM |
| **E2B drafts E4B + Whisper (GGUF)** | 0.9 GB | ~6.0 GB | 0.3 GB | ~7.2 GB | Yes (4.8 free) | Yes | Yes |
| **E2B drafts E4B + Whisper spec decode** | 0.9+0.1 GB | ~6.0 GB | 0.3 GB | ~7.3 GB | Yes (4.7 free) | Yes | Yes |

### Option D Deep Dive: E2B Drafts for E4B

E4B has the best translation quality (7/8 canary, BLEU 62.2, chrF++ 76.8) but is unusable at NF4 on 16 GB (15 GB VRAM, 8.5s/verse). The opportunity:

**Why E2B → E4B is the best spec decode scenario:**
- Same Gemma 4 tokenizer exactly — no UAG overhead, traditional spec decode
- E2B quality is close to E4B (BLEU 60.5 vs 62.2) → high acceptance rate expected (70-85%)
- At 70% acceptance with 5 draft tokens: ~3x speedup on E4B → 8.5s → ~2.8s
- E4B's 7/8 canary accuracy is the best of any model tested

**The blocker is VRAM:** E2B (6.3 GB) + E4B (15.0 GB) = 21.3 GB at NF4. Doesn't fit anywhere.

**The fix is GGUF (Pillar 1):** At Q4_K_M quantization via llama.cpp:
- E2B Q4: ~1.5 GB (vs 6.3 GB NF4 — PLE tables compress much better in GGUF)
- E4B Q4: ~4.5 GB (vs 15.0 GB NF4)
- Combined: ~6.0 GB — fits on every target device with room for Whisper

**This makes Option D the strongest case for the llama.cpp path in Pillar 1.** GGUF unlocks E4B (best quality) + E2B (fast draft) in 6 GB total, vs 21 GB at NF4. The PLE architecture that inflates NF4 VRAM compresses well under GGUF because the embedding lookup tables are mostly sparse.

### Recommendation

**Skip E2B ↔ TG 4B spec decode.** E2B already wins on both speed and quality — adding spec decode is solving a problem that doesn't exist.

**Prioritize E2B → E4B via GGUF (Pillar 1).** This is the highest-value spec decode scenario: best quality model (E4B) made practical via GGUF compression + E2B draft for speed. Requires llama.cpp integration first.

## Scenario 2: Whisper Distil-v3.5 → Large-V3 (or Turbo)

### Known Working Configuration

HuggingFace has an [official blog post](https://huggingface.co/blog/whisper-speculative-decoding) documenting this exact setup:

- **Draft:** `distil-whisper/distil-large-v3.5` (2 decoder layers, ~756M params)
- **Target:** `openai/whisper-large-v3` (32 decoder layers, ~1.5B params)
- **Result:** 2x faster, mathematically identical output
- **Why it works:** Distil-Whisper keeps the encoder frozen and shares the same tokenizer. Only 2 extra decoder layers need loading.

### Our Setup: Draft for Whisper Turbo

We use `whisper-large-v3-turbo` (4 decoder layers, ~809M params) as our primary STT model. Two draft options:

**Option A: Distil-v3.5 drafts for Turbo**
- Distil-v3.5 has 2 decoder layers, Turbo has 4
- Same tokenizer (Whisper unified BPE)
- Distil is faster than Turbo (fewer layers) → valid draft
- **Should work** — same tokenizer, smaller model drafts for larger

**Option B: Turbo drafts for Large-V3**
- Turbo has 4 decoder layers, Large-V3 has 32
- Same tokenizer
- Turbo is 8x faster → excellent draft
- **Known to work** — Turbo was designed for this (it IS a distilled v3)

### Implementation

Already documented in `docs/turbo_inference.md` (S2 config):

```python
# Load both models, share encoder
target = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3")
draft = WhisperForConditionalGeneration.from_pretrained("distil-whisper/distil-large-v3.5")

# Spec decode — encoder runs once, decoder layers verified
output = target.generate(
    input_features,
    assistant_model=draft,
    max_new_tokens=128,
    language="en",
)
```

**Key advantage:** Encoder only runs once (shared between target and draft). The speedup comes from fewer decoder verification passes.

### VRAM Impact (STT Only)

| Config | Encoder | Target Decoder | Draft Decoder | Total STT | Notes |
|--------|---------|----------------|---------------|-----------|-------|
| Turbo only | 1.2 GB | 0.3 GB (4 layers) | — | **1.5 GB** | Current setup |
| Turbo + distil-v3.5 draft | 1.2 GB (shared) | 0.3 GB | 0.1 GB (2 layers) | **1.6 GB** | +100 MB only |
| Large-V3 + Turbo draft | 1.2 GB (shared) | 1.5 GB (32 layers) | 0.3 GB (4 layers) | **3.0 GB** | Only if we use V3 |

The encoder is shared between target and draft (identical weights, loaded once). Draft decoder adds only its extra layers. This is why Whisper spec decode is so cheap — 100 MB for 2x speedup.

### VRAM Impact (STT + Translation Combined)

| STT Config | Translation Config | STT VRAM | Translation VRAM | Overhead | **Total** | Fits 12 GB? |
|-----------|-------------------|----------|-----------------|----------|-----------|-------------|
| Turbo (current) | E2B standalone | 1.5 GB | 6.3 GB | 0.5 GB | **8.3 GB** | Yes (3.7 free) |
| Turbo + distil draft | E2B standalone | 1.6 GB | 6.3 GB | 0.5 GB | **8.4 GB** | Yes (3.6 free) |
| Turbo + distil draft | E2B drafts E4B (GGUF) | 1.6 GB | 6.0 GB | 0.3 GB | **7.9 GB** | Yes (4.1 free) |
| Turbo + distil draft | TG 4B standalone | 1.6 GB | 3.0 GB | 0.5 GB | **5.1 GB** | Yes (6.9 free) |

**Best combined config:** Whisper Turbo + distil-v3.5 draft (1.6 GB) + E2B drafts E4B GGUF (6.0 GB) = **7.9 GB total** — fits on 12 GB RTX 3060 with 4.1 GB headroom. Gets best STT speed (2x) AND best translation quality (E4B 7/8 canary).

### Recommendation

**Do Whisper spec decode (Distil-v3.5 → Turbo).** Low risk, proven pattern, minimal VRAM overhead (~100 MB for 2 extra decoder layers), 1.5-2x STT speedup. Implement in `MLXWhisperEngine` and `FasterWhisperEngine`.

## Summary

| Scenario | Feasible? | Worth Doing? | Risk | Priority |
|----------|-----------|-------------|------|----------|
| E2B ↔ TG 4B | Yes (UAG) | **No** — E2B already wins | Low | Skip |
| E2B → E4B (via GGUF) | Yes (same tokenizer) | **Maybe** — if E4B fits at lower VRAM | Medium | Defer to Pillar 1 |
| Distil-v3.5 → Turbo (STT) | **Yes** (proven) | **Yes** — 1.5-2x STT speedup, trivial VRAM | **Low** | **High** |
| Turbo → Large-V3 (STT fallback) | Yes (proven) | Maybe — only if we use Large-V3 | Low | Low |

## What This Addresses in 2026.5

| 2026.5 Section | What This Research Resolves |
|----------------|---------------------------|
| **Pillar 1, Phase 1D** (wire spec decode into pipeline) | Whisper distil-v3.5 → Turbo is the STT spec decode path. MLX + CUDA. |
| **Pillar 1, Phase 1A-1B** (llama.cpp) | E2B → E4B GGUF spec decode is the strongest argument for llama.cpp — unlocks E4B at 6 GB combined |
| **Pillar 3, Phase 3C** (Gemma 4 engine integration) | E2B replaces TG 4B as default; E2B↔TG 4B spec decode is NOT needed |
| **Cross-pillar: spec decode rejection logging** | All spec decode paths feed rejection data to the logger |

## Implementation Sequence

### Step 1: Whisper STT Spec Decode (Low Risk, High Value)

**Branch:** `feat/whisper-spec-decode`

**Implementation:**
1. Modify `MLXWhisperEngine.load()` in `engines/mlx_engine.py`:
   - Add `draft_model_id` parameter (default: `distil-whisper/distil-large-v3.5`)
   - Load draft model alongside primary, share encoder weights
2. Modify `MLXWhisperEngine.transcribe()`:
   - Pass `assistant_model=self._draft_model` to `generate()`
3. Modify `FasterWhisperEngine` in `engines/cuda_engine.py`:
   - Same pattern for CUDA path (faster-whisper may need different API — check CTranslate2 spec decode support)
4. Add `stt_draft_model` setting to `STTSettings` in `settings.py`
5. Wire `--stt-draft` flag in `dry_run_ab.py`

**Testing:**
- `tests/test_whisper_spec_decode.py`:
  - Mock both target + draft models, verify `assistant_model=` passed to generate
  - Test that encoder is shared (same object reference)
  - Test fallback when draft model fails to load (graceful degradation)
- Integration: run 10 audio chunks with and without draft, verify identical output + faster wall clock
- Benchmark: `tools/benchmark_latency.py` with `--stt-draft` flag, measure speedup on 50 chunks

**Quality gate:** Output must be byte-identical to non-spec-decode (speculative decoding is lossless). Latency must improve by >= 1.3x.

**PR review checklist:**
- [ ] Identical STT output verified (diff test on 50 chunks)
- [ ] VRAM increase <= 150 MB (draft decoder only)
- [ ] No regression on existing STT tests
- [ ] `_MOCK_MODULES` updated if new dependencies added
- [ ] Settings documented (`STARK_STT_DRAFT_MODEL`)

### Latency Implications of Model Changes

**Current pipeline latency (TG 4B, no spec decode):**
```
Partial: Whisper Turbo (~500ms) + MarianMT (~250ms) = ~750ms
Final:   Whisper Turbo (~500ms) + TG 4B (~550ms MLX / ~2.46s CUDA) = ~1.1s MLX / ~3.0s CUDA
```

**With E2B replacing TG 4B:**
```
Final:   Whisper Turbo (~500ms) + E2B (~2.24s CUDA) = ~2.7s CUDA (9% faster than TG 4B)
```
E2B is faster than TG 4B on CUDA (2.24s vs 2.46s). MLX latency TBD — E2B MLX weights may not exist yet (need mlx-community conversion). At similar param count, expect comparable or slightly slower MLX speed due to PLE overhead.

**With Whisper spec decode (Distil-v3.5 → Turbo):**
```
Partial: Whisper Turbo+draft (~250-330ms) + MarianMT (~250ms) = ~500-580ms (25-33% faster)
Final:   Whisper Turbo+draft (~250-330ms) + E2B (~2.24s CUDA) = ~2.5-2.6s CUDA
```
STT drops from ~500ms to ~250-330ms (1.5-2x). Translation unchanged. End-to-end partial improves from 750ms to ~530ms.

**With E2B → E4B GGUF spec decode (Pillar 1):**
```
Final:   Whisper Turbo+draft (~300ms) + E4B+E2B draft (~2.5-3.0s GGUF) = ~2.8-3.3s
```
E4B at GGUF Q4_K_M should run at ~25-40 tok/s (vs ~2 tok/s at NF4). With E2B drafting at 70%+ acceptance: effective ~2.5-3.0s. Quality jumps to 7/8 canary, BLEU 62.2. Comparable latency to current TG 4B but much better quality.

**Best-case combined pipeline (all spec decode):**
```
Partial: Whisper+draft (~300ms) + MarianMT (~250ms) = ~550ms
Final:   Whisper+draft (~300ms) + E4B+E2B GGUF draft (~2.8s) = ~3.1s CUDA
         vs current: Whisper (~500ms) + TG 4B (~2.46s) = ~3.0s CUDA
```
Similar total latency, but dramatically better translation quality (BLEU 62.2 vs 39.3, canary 7/8 vs 5/8).

### Step 2: E2B as Default Translation Model (Medium Risk)

**Branch:** `feat/gemma4-e2b-default`

**Implementation:**
1. Add Gemma 4 instruct prompt path to `MLXGemmaEngine` and `CUDAGemmaStreamingEngine`
   - New prompt builder: plain text instruction (no `source_lang_code` template)
   - Output cleaning: strip preamble text before translation
2. Update `TranslationSettings` defaults:
   - `mlx_model_4b` → E2B MLX repo (when available) or keep TG 4B as fallback
   - `cuda_model_4b` → `google/gemma-4-e2b-it`
   - Add `model_family: str = "gemma4"` setting
3. Update `factory.py` to route based on `model_family`
4. Keep TG 4B as `--model-family translategemma` option

**Testing:**
- `tests/test_gemma4_prompt.py`: verify instruct prompt formatting, preamble stripping
- Integration: translate 8 canary sentences, verify 6/8 theological accuracy
- Benchmark: compare E2B vs TG 4B on 100 sermon chunks (latency + quality)

**Quality gate:** E2B canary >= 6/8, BLEU >= 55 (vs TG 4B's 39.3), no hallucination regression.

**PR review checklist:**
- [ ] Canary accuracy >= 6/8
- [ ] BLEU/chrF++ numbers in PR description
- [ ] TG 4B still works as `--model-family translategemma`
- [ ] Settings documented (`STARK_TRANSLATE_MODEL_FAMILY`)

### Step 3: llama.cpp Install + GGUF Export + E2B→E4B Spec Decode (Higher Risk)

**Branch:** `feat/llamacpp-engine`

**Prerequisites — llama.cpp install (WSL2, A2000 Ada CUDA):**
```bash
# Clone and build with CUDA support
cd /home/wbell
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build --config Release -j$(nproc)

# Verify
./build/bin/llama-server --help | head -5
./build/bin/llama-quantize --help | head -5

# Install Python conversion dependencies
pip install -r requirements/requirements-convert_hf_to_gguf.txt
```

Ada architecture is compute capability 8.9 — use `-DCMAKE_CUDA_ARCHITECTURES=89`. Build takes ~5 min.

**GGUF Export:**
```bash
# Convert E2B to GGUF
python llama.cpp/convert_hf_to_gguf.py \
  --outfile models/gemma-4-e2b-it-f16.gguf \
  --outtype f16 \
  google/gemma-4-e2b-it

# Quantize to Q4_K_M
./llama.cpp/build/bin/llama-quantize \
  models/gemma-4-e2b-it-f16.gguf \
  models/gemma-4-e2b-it-q4km.gguf Q4_K_M

# Same for E4B
python llama.cpp/convert_hf_to_gguf.py \
  --outfile models/gemma-4-e4b-it-f16.gguf \
  --outtype f16 \
  google/gemma-4-e4b-it

./llama.cpp/build/bin/llama-quantize \
  models/gemma-4-e4b-it-f16.gguf \
  models/gemma-4-e4b-it-q4km.gguf Q4_K_M
```

**Implementation:**
1. Create `start_server.sh` — launches llama-server with E4B target + E2B draft:
   ```bash
   ./llama.cpp/build/bin/llama-server \
     -m models/gemma-4-e4b-it-q4km.gguf \
     -md models/gemma-4-e2b-it-q4km.gguf \
     --draft 16 --draft-min 5 \
     --host 127.0.0.1 --port 8090 \
     -ngl 999 -c 512
   ```
2. Create `engines/llamacpp_engine.py` implementing `TranslationEngine` (HTTP client to llama-server)
3. Register in `factory.py` as `engine_type="llamacpp"`
4. Benchmark T1-T4 configs from `docs/april_squeeze/llama_squeeze.md`

**Testing:**
- `tests/test_llamacpp_engine.py`: mock HTTP calls, test prompt formatting
- Integration: 8 canary sentences through llama-server, verify accuracy
- Benchmark: T1 (HF baseline) vs T2 (GGUF) vs T3 (n-gram spec) vs T4 (E2B draft)
- VRAM: verify E2B+E4B combined <= 6.5 GB at Q4_K_M

**Quality gate:** Gate 1A from 2026.5 — T3/T4 must beat T1 by >30% tok/s.

**PR review checklist:**
- [ ] GGUF export reproducible (document exact commands)
- [ ] Canary sentences 6/8+ through llama-server
- [ ] VRAM budget table in PR description
- [ ] `start_server.sh` included with health check

### Merge Order & Dependencies

```
Step 1 (whisper spec decode) ──────────────────────┐
                                                     ├── Can merge independently
Step 2 (E2B default) ─────────────────────────────┘

Step 3 (llama.cpp + E2B→E4B spec decode) ── Depends on Step 2 (E2B prompt format)
```

Steps 1 and 2 are independent and can be developed in parallel. Step 3 depends on Step 2's prompt format work.

### Timeline

| Step | Effort | Depends On | When |
|------|--------|-----------|------|
| Step 1: Whisper spec decode | 1 day code + test | Nothing | Week 1 |
| Step 2: E2B default model | 1-2 days code + test | Nothing | Week 1 |
| Step 3: llama.cpp + E4B spec decode | 2-3 days | Step 2, llama.cpp install | Week 2 |

## Sources

- [Universal Assisted Generation (HuggingFace blog)](https://huggingface.co/blog/universal_assisted_generation)
- [Speculative Decoding for 2x Faster Whisper](https://huggingface.co/blog/whisper-speculative-decoding)
- [HuggingFace Assisted Decoding docs](https://huggingface.co/docs/transformers/assisted_decoding)
- [distil-whisper/distil-large-v3.5](https://huggingface.co/distil-whisper/distil-large-v3.5)
- [Gemma 4 Prompt Formatting](https://ai.google.dev/gemma/docs/core/prompt-formatting-gemma4)
- [llama.cpp Gemma 4 tokenizer fix PR](https://github.com/ggml-org/llama.cpp/pull/21343)
- [grimjim on spec decode tokenizer requirements](https://huggingface.co/posts/grimjim/820999393776814)
