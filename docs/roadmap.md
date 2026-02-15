# Roadmap — Stark Road Bilingual Speech-to-Text

> Living document tracking the full project trajectory from Mac prototype
> through Windows training to RTX 2070 edge deployment.
>
> **Last updated:** 2026-02-14

---

## Current State

```
Mac (M3 Pro 18GB, MLX)           Windows (A2000 Ada 16GB, CUDA)
  Live inference prototype          Training pipeline built
  Two-pass: MarianMT + Gemma 4B    Bible corpus: 269K verse pairs
  P7 latency: ~150ms partials      5 test sermons downloaded
              ~911ms finals         Seattle run designed (Feb 12-17)
  Pipeline overlap (6C) done
  P5 caption comparison fixed

Accent-Diverse STT (code complete, data collection pending)
  download_sermons.py --accent       Accent-tagged downloads → stark_data/raw/{accent}/
  preprocess_audio.py                Accent propagation through preprocessing
  prepare_whisper_dataset.py (NEW)   Temperature-balanced accent sampling (T=0.5)
  train_whisper.py                   AccentBalancedTrainer + per-accent WER eval
                                     Forced language="en" (Scottish→Welsh fix)

Multilingual (planned, not started)
  Hindi & Chinese todo list          docs/multi_lingual.md
  Multilingual research proposal     docs/multilingual_tuning_proposal.md
  Accent tuning plan                 docs/accent_tuning_plan.md
```

---

## Phase 1: Training & Validation (Current → Week 4)

### 1A. Seattle Training Run (Feb 12-17, unattended)

See [`seattle_training_run.md`](seattle_training_run.md) for full details.

| Stage | Model | Data | GPU Time |
|-------|-------|------|----------|
| 1 | Audio preprocessing | 30 sermons (~30 hrs) | ~5 hrs |
| 2 | Whisper large-v3 pseudo-labeling | Cleaned audio → text | ~2.5 hrs |
| 3 | Baseline evaluation | Holdout set | ~0.3 hrs |
| 4 | **Whisper LoRA** | Church audio + pseudo-labels + 30% Common Voice | ~1 hr |
| 5 | **MarianMT full fine-tune** | 269K Bible verse pairs (EN→ES) + glossary | ~2.2 hrs |
| 6 | **TranslateGemma 4B QLoRA** | Same 269K pairs + glossary | ~12 hrs |
| 7 | Post-training evaluation | Holdout set | ~1 hr |
| 8 | **BONUS: TranslateGemma 12B QLoRA** | Same 269K pairs (may OOM on 16GB) | ~26 hrs |

**Go/No-Go gate:** WER improvement >10% relative, BLEU improvement >+2 points.

### 1B. Post-Seattle: LoRA → MLX Conversion + Mac Testing

- Convert PyTorch LoRA adapters to MLX format (~30 min per model)
- Run A/B comparison with fine-tuned models on Mac
- Record baseline vs fine-tuned metrics

### 1C. Active Learning Cycles (Weeks 3-4)

3-5 cycles of: infer → flag low-confidence → human correct → retrain.
Each cycle: ~1 overnight GPU run + 3-5 days human correction.

### 1D. Accent-Diverse STT Tuning (Weeks 2-5) — CODE COMPLETE

See [`accent_tuning_plan.md`](accent_tuning_plan.md) for full 4-week execution timeline.

**Problem:** Base Whisper has 22-34% WER on Scottish English vs <10% on Standard American. Visiting speakers at Stark Road have Scottish, Canadian, and British accents.

**Implementation (done):**
- `download_sermons.py --accent {label}` — tags downloads with accent metadata, organizes into `stark_data/raw/{accent}/` subdirectories
- `preprocess_audio.py` — reads accent from companion JSON, writes chunks to `stark_data/cleaned/chunks/{accent}/`
- `prepare_whisper_dataset.py` **(new)** — temperature-based accent balancing (T=0.5), confidence filtering, stratified train/eval split, outputs HuggingFace audiofolder with accent column
- `train_whisper.py` — `AccentBalancedTrainer` with `WeightedRandomSampler`, per-accent WER eval (`wer_scottish`, `wer_canadian`, etc.), `wer_accent_gap` fairness metric, forced `language="en"` to prevent Scottish→Welsh misclassification

**Data collection targets:**

| Accent | Hours | Priority | Rationale |
|--------|-------|----------|-----------|
| Midwest | 20-50 hrs | High | Primary domain (Stark Road) |
| Scottish | 3-5 hrs | **Critical** | 22-34% WER gap |
| British | 2-3 hrs | Medium | 10-15% WER |
| Canadian | 1-2 hrs | Low | Already handled well |
| West Coast | 0-1 hrs | Lowest | Well-covered by base model |

**Quality target:** Accent WER gap < 5% absolute (max accent WER - min accent WER).

**Execution:**
1. Week 1: Download accent-tagged audio from YouTube playlists (user provides URLs)
2. Week 2: Preprocess, build balanced dataset, train Round 1
3. Week 3: Active learning on worst accent (Scottish), train Rounds 2-3
4. Week 4: Integration test on Mac, verify no Midwest regression

---

## Phase 2: Accuracy Deepening (Weeks 4-8)

### 2A. Expanded Data Collection During Mac Inference

**Auto-pseudo-labeling from live sessions:**
- During live church services, capture high-confidence EN→ES pairs from the pipeline
- Filter by QE score (CometKiwi > 0.85) and STT confidence (avg_logprob > -0.2)
- Append to `sermon_pairs.jsonl` — real-domain translation training data
- Target: 10-20 hours/week of paired data from actual church use

**Diagnostic enrichment:**
- Log hardware snapshots (VRAM, latency per layer) for 2070 profiling
- Track error patterns: untranslated terms, theological misses, hallucinations
- Use to identify which model components need targeted fine-tuning

**Active learning queue:**
- Route low-QE chunks (< 0.8) to operator review interface
- SQLite queue with export to A2000 for batch retraining
- Target: 1-5K curated samples/month focused on edge cases

### 2B. Iterative Fine-Tuning with Sermon Pairs

- Fold Mac-collected sermon pairs into Bible corpus for translation training
- Weight samples by diagnostic scores (higher loss on low-QE chunks)
- Run multi-epoch QLoRA with mixed precision on A2000
- Expected: +3-7% BLEU on theological text

### 2C. Hybrid STT-Translation Tuning

- Fine-tune Whisper on sermon audio, then use its outputs as inputs for Gemma QLoRA
- End-to-end STT→translation tuning reduces pipeline errors
- Example: Whisper mishears "propitiation" → translation fails. Joint training catches this.
- Target: <5% WER on church speech

### 2D. Ensemble Distillation (12B → 4B)

- Train 12B on diverse data subsets (bible vs sermons)
- Distill 12B knowledge into 4B on A2000
- Matches 12B baseline (+2-4% over single model) in smaller footprint
- Critical for 2070 deployment where 12B won't fit comfortably

### 2E. Multilingual Expansion: Hindi & Chinese — PLANNED

See [`multi_lingual.md`](multi_lingual.md) for actionable checklist and [`multilingual_tuning_proposal.md`](multilingual_tuning_proposal.md) for full research.

**Key insight:** TranslateGemma natively supports Hindi and Chinese — no new model needed. Fine-tuning is domain adaptation (theological register), not teaching new languages. Same models, different LoRA adapters per language.

**Recommended order:**
1. Accent-tune Whisper first (Phase 1D) — better STT benefits all downstream translations
2. Then fine-tune TranslateGemma for Hindi/Chinese

**Summary:**

| Phase | Duration | What |
|-------|----------|------|
| Zero-shot baseline | 1 day | Test TranslateGemma with `target_lang_code="hi"` / `"zh-Hans"` |
| Data preparation | 3-5 days | ~155K EN-HI + ~310K EN-ZH biblical verse pairs, theological glossaries |
| Hindi QLoRA | 1 night | r=32, 768 max_seq_length (high token fertility), separate adapter |
| Chinese QLoRA | 1 night | r=32, 512 max_seq_length, separate adapter |
| Evaluation | 1 day | chrF++, COMET, theological term accuracy |
| Pipeline integration | 1-2 days | Adapter switching, partial models, WebSocket multi-language |
| Display updates | 1 day | Devanagari/CJK fonts, mobile language selector |

**Key decisions:**
- Hindi partial display: English partial + Hindi final only (SOV word order garbles partials)
- Chinese: 神 (Shen) for God, not 上帝 (Shangdi) — matches CUV majority edition
- Separate LoRA adapters per language (no joint training — risk of catastrophic forgetting)

**Total:** ~8-10 days, ~13-20 GPU hours, ~28-44 human hours

---

## Phase 3: RTX 2070 Edge Deployment (Weeks 8-12)

### Why RTX 2070?

The Mac is a development machine. For permanent church installation:
- RTX 2070: 8GB VRAM, 2304 CUDA cores, Turing architecture (CUDA 10.1+)
- Always-on, no laptop needed at services
- Lower cost than Mac hardware for a dedicated appliance

### 3A. Hardware Constraints

| Resource | RTX 2070 | Mac M3 Pro | Impact |
|----------|----------|------------|--------|
| VRAM | 8 GB GDDR6 | 18 GB unified | 4B fits, 12B needs aggressive quantization |
| Compute | 2304 CUDA cores | 18 GPU cores (Metal) | ~1.5-2x faster than M3 for transformer inference |
| Memory BW | 448 GB/s | 150 GB/s | 3x bandwidth advantage |
| Framework | PyTorch/CUDA, TensorRT | MLX only | Need framework migration |

### 3B. Framework Migration (MLX → PyTorch CUDA / TensorRT)

**Step 1: Model export**
- Convert MLX models to HuggingFace format via `mlx_lm.convert`
- Load in PyTorch with CUDA backend
- Verify output equivalence on test set

**Step 2: TensorRT optimization**
- Build TensorRT engines for int8/FP16 inference
- Calibrate with Mac-collected sermon data (1K+ samples)
- Expected: 4B finals ~500-800ms (vs 911ms on Mac)

**Step 3: `torch.compile` baseline**
- Before full TensorRT, test `torch.compile` for 10-20% speedup
- Lower effort, good interim solution

**Script needed:** `export_to_cuda.py` — converts LoRA adapters + base models to CUDA-optimized format

### 3C. Quantization & Compression

**Calibration-aware quantization:**
- Use A2000 to test 3-4 bit quantization (bitsandbytes) on fine-tuned models
- Calibrate on sermon diagnostics collected from Mac sessions
- Expected: 20-30% faster than 4-bit on 2070 (~400ms partials)

**Quantization-Aware Training (QAT):**
- Retrain with quantization in the loop on A2000
- Minimal accuracy drop vs post-training quantization
- Critical for 3-bit models where PTQ degrades badly

**ONNX export for TensorRT:**
- Export quantized models to ONNX → TensorRT engine builder
- int8 engines are 2-3x faster than FP16 on Turing GPUs

### 3D. Pipeline Adaptations for Edge

**Compute split:**
- STT (Whisper): GPU — benefits most from CUDA acceleration
- VAD (Silero): CPU — tiny model, CPU is fine
- Translation: GPU — TranslateGemma 4B int8, TensorRT engine
- MarianMT: CPU or GPU — small enough for either

**VRAM budget (8 GB):**

| Component | FP16 | INT8 | INT4 |
|-----------|------|------|------|
| Distil-Whisper | ~1.5 GB | ~750 MB | ~375 MB |
| TranslateGemma 4B | ~8 GB | ~4 GB | ~2.5 GB |
| MarianMT | ~600 MB | ~300 MB | ~150 MB |
| **Total** | **~10 GB** (OOM) | **~5 GB** | **~3 GB** |

**Recommendation:** INT8 for Whisper + TranslateGemma 4B = ~4.75 GB, leaves 3.25 GB headroom for KV cache and overhead. Comfortable fit.

**Runtime profiling:**
- Auto-select model variants based on available VRAM
- Fallback: MarianMT-only mode if TranslateGemma can't fit
- Log profiling data for continuous optimization

### 3E. 12B on RTX 2070

12B at FP16 = ~24 GB — way too large. Options:
- **4-bit quantized:** ~6 GB — tight fit with Whisper (~1.5 GB) = 7.5 GB total. Possible but no headroom.
- **3-bit quantized:** ~4.5 GB — fits with Whisper = 6 GB. 2 GB headroom. Viable if accuracy holds.
- **Distilled 4B (from Phase 2D):** ~2.5 GB INT8 — best option. 12B quality in 4B package.

**Decision:** Don't target raw 12B on 2070. Instead, distill 12B → 4B during Phase 2.

---

## Phase 4: Production Polish (Weeks 12+)

### 4A. Permanent Installation

- Dedicated Windows PC with RTX 2070 at church
- Auto-start pipeline on boot
- WebSocket → projection system (OBS/NDI)
- Operator tablet for controls

### 4B. Continuous Improvement Loop

```
Church service → Live inference → Log diagnostics
    ↓
Sync logs to A2000 (weekly)
    ↓
Retrain on new data (monthly)
    ↓
Deploy updated models to 2070
    ↓
Repeat
```

### 4C. Future Features

- Post-sermon 5-sentence summary (features/diarize.py + features/summarize_sermon.py)
- Verse reference extraction (features/extract_verses.py)
- Multi-language support (TranslateGemma supports 36 languages)
- Live dashboard for operators (Streamlit or web UI)

---

## Target Metrics by Phase

### STT & Spanish Translation

| Metric | Current (Base) | Phase 1 Target | Phase 2 Target | Phase 3 (2070) |
|--------|---------------|----------------|----------------|----------------|
| Whisper WER (church, Midwest) | ~12-18% | <10% | <5% | <5% |
| Whisper WER (Scottish) | ~22-34% | <15% | <10% | <10% |
| Accent WER gap (max-min) | ~15-24% | <10% | <5% | <5% |
| SacreBLEU (biblical, ES) | ~35-42 | >44 | >48 | >46 (post-quant) |
| Theological terms (ES) | ~40-60% | >65% | >85% | >80% (post-quant) |
| Partial latency | 153ms (Mac) | 153ms | 153ms | <100ms (TensorRT) |
| Final latency | 911ms (Mac) | ~600ms (6C) | ~500ms (adaptive) | <500ms (TensorRT) |
| End-to-end perceived | ~900ms | ~600ms | ~400ms | <400ms |

### Multilingual (Phase 2E)

| Metric | Hindi Target | Chinese Target |
|--------|-------------|----------------|
| chrF++ improvement over zero-shot | > +2 points | > +2 points |
| COMET improvement over zero-shot | > +0.02 | > +0.02 |
| Theological term accuracy | > 80% | > 80% |
| Hindi honorific accuracy (तू for God) | > 90% | N/A |
| Chinese 神/上帝 consistency | N/A | > 95% |
| Final latency (4B) | ~700-800ms | ~600-700ms |

---

## Key Decisions Pending

| Decision | When | Options |
|----------|------|---------|
| Train 12B? | Post-Seattle (check if 4B is sufficient) | Train if 4B BLEU < 44 |
| RTX 2070 timing | After Phase 2 accuracy targets met | Buy/configure hardware |
| TensorRT vs torch.compile | Phase 3 start | TensorRT = more work, faster; torch.compile = quick win |
| 12B deployment strategy | Phase 2D results | Distill to 4B vs aggressive quantization |
| Scottish accent playlist sources | Phase 1D Week 1 | User provides YouTube playlist URLs |
| Chinese 神 vs 上帝 | Phase 2E data prep | 神 recommended (CUV majority edition) |
| Hindi partial model | Phase 2E integration | IndicTrans2-Dist (quality) vs opus-mt-en-hi (simplicity) |
| Hindi partial display | Phase 2E integration | English partial + Hindi final (recommended) vs delayed partial |
| Train 12B for Hindi/Chinese? | Phase 2E eval | Only if 4B theological accuracy < 70% |

---

## Reference Documents

| Doc | Contents |
|-----|----------|
| [`training_plan.md`](training_plan.md) | Full training schedule, channel inventory, go/no-go gates |
| [`seattle_training_run.md`](seattle_training_run.md) | 6-day unattended run design with data flow |
| [`training_time_estimates.md`](training_time_estimates.md) | A2000 Ada GPU time estimates per model |
| [`accent_tuning_plan.md`](accent_tuning_plan.md) | 4-week accent-diverse STT tuning plan (code complete) |
| [`multi_lingual.md`](multi_lingual.md) | Hindi & Chinese actionable todo list |
| [`multilingual_tuning_proposal.md`](multilingual_tuning_proposal.md) | Full Hindi/Chinese research: corpora, glossaries, QLoRA, evaluation |
| [`rtx2070_feasibility.md`](rtx2070_feasibility.md) | RTX 2070 hardware analysis |
| [`fast_stt_options.md`](fast_stt_options.md) | Lightning-whisper-mlx feasibility (not viable) |
| [`projection_integration.md`](projection_integration.md) | OBS/NDI/ProPresenter integration |
