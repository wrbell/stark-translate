# Roadmap — Stark Road Bilingual Speech-to-Text

> Living document tracking the full project trajectory from Mac prototype
> through Windows training to RTX 2070 edge deployment.
>
> **Last updated:** 2026-02-17

---

## Current State

```
Mac (M3 Pro 18GB, MLX)           Windows (A2000 Ada 16GB, CUDA)
  Live inference prototype          Training pipeline built
  Whisper Large-V3-Turbo (STT)     Bible corpus: 269K verse pairs
  engines/ package (MLX + CUDA)    5 test sermons downloaded
  settings.py (pydantic-settings)  train_whisper.py → Turbo base
  Backend: --backend auto|mlx|cuda Piper TTS training scripts ready
  Fallback: lazy-load on low conf
  P7 latency: ~150ms partials
              ~911ms finals
  Pipeline overlap (6C) done
  P5 caption comparison fixed

Production Endpoints (implemented):
  1. Mac (M-series, 8-18 GB) — MLX, --backend=mlx
  2. RTX 2070 Windows (8 GB VRAM) — CUDA/faster-whisper, --backend=cuda --no-ab
  3. Dev M3 Pro 18 GB — full A/B (both 4B + 12B)

New in feature/turbo-dual-prod-v2:
  engines/              STT + Translation engine ABCs, MLX + CUDA impl, factory
  settings.py           Unified config (STARK_ env prefix, .env support)
  requirements-nvidia.txt  CUDA inference deps
  tools/convert_models_to_both.py  Dual-endpoint model export
  training/train_piper.py          Piper voice fine-tuning
  training/prepare_piper_dataset.py  LJSpeech format conversion
  training/export_piper_onnx.py    ONNX export + validation
  training/evaluate_piper.py       TTS quality assessment

Accent-Diverse STT (code complete, data collection pending)
  download_sermons.py --accent       Accent-tagged downloads
  preprocess_audio.py                Accent propagation
  prepare_whisper_dataset.py         Temperature-balanced sampling
  train_whisper.py                   AccentBalancedTrainer + per-accent WER

Multilingual (planned, not started)
  Hindi & Chinese todo list          docs/multi_lingual.md
  Multilingual research proposal     docs/multilingual_tuning_proposal.md
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

## Phase 1T: Whisper Large-V3-Turbo Migration (~10-12 working days)

> **Model:** OpenAI whisper-large-v3-turbo — decoder layers pruned 32→4, ~809M params, MIT license.
> **Gains:** 5-8x faster than large-v3 on Apple Silicon (MLX), ~2-3x vs Distil-large-v3 on Turing GPUs (faster-whisper INT8).
> **Trade-off:** ~0.2-1% WER regression on clean English, up to ~1-2% on long-form/noisy/church-domain (recoverable with LoRA re-tune).

### 1T-A. Mac Turbo Partials (Days 1-2 — lowest risk, highest immediate value)

> **Status: DONE** — Wholesale Turbo swap (both partials and finals) in `dry_run_ab.py`.

Replace Distil-large-v3 with `mlx-community/whisper-large-v3-turbo-4bit` in partial path of `dry_run_ab.py`. Expected: partials drop from ~300-380ms → **<150-200ms**. Memory drops from ~1.5-2 GB → ~1.1-1.6 GB (4-bit), freeing headroom for Gemma.

Rollback: 1-line config revert. Test on 5-10 sermon clips before deploying to live services.

### 1T-B. Engines Package & Factory (Days 2-3)

> **Status: DONE** — `engines/` package: base.py, mlx_engine.py, cuda_engine.py, factory.py, active_learning.py.

Create `engines/` package with abstract STT interface so Mac (MLX) and Windows (faster-whisper) share common API. Factory pattern loads Turbo by default, falls back to Distil via config.

### 1T-C. Turbo Fine-Tune + Dual-Endpoint Export (Days 4-6, A2000)

> **Status: PARTIAL** — `train_whisper.py` updated, `tools/convert_models_to_both.py` created, fallback logic in engines. Remaining: actual fine-tuning run on A2000.

- LoRA re-tune on church audio with Turbo base (~4-8 hrs on A2000, faster than large-v3)
- Target: <1% WER delta vs current Distil on church sermons
- Export Mac (4-bit MLX) + Windows (CT2 INT8) variants via `tools/convert_models_to_both.py`
- Fallback logic: If avg_logprob < -1.2 or compression_ratio >2.4, retry with old Distil (lazy loaded)

### 1T-D. RTX 2070 Integration + Benchmarks (Days 7-8)

> **Status: PENDING** — Blocked on fine-tuning and hardware access.

- Convert Gemma LoRAs to GGUF Q4_K_M for llama.cpp
- Full pipeline test: Turbo STT + Gemma translation → target 1.3-2.0s finals
- VRAM budget: Turbo INT8 ~1.5-2.5 GB + Gemma Q4 ~3.2 GB (tight but viable on 8 GB)

### 1T-E. Unified Config + Deployment (Days 9-11)

> **Status: PARTIAL** — `settings.py` created. `run_live.py` deferred.

- `settings.py` with pydantic-settings: Turbo flags, model paths, endpoint selection
- `run_live.py` unified entrypoint defaulting to Turbo on both endpoints
- Loguru logging tagged with "turbo" + endpoint + chunk_duration

### 1T-F. Testing, Regression & Release (Days 11-12)

> **Status: PENDING** — Needs models running to benchmark.

- Benchmark: Turbo vs old STT on 50 clips (focus long sermons, chunked vs sequential)
- Theological term audit: glossary 229 terms, verify initial_prompt conditioning with 4-layer decoder
- Docs: Turbo migration notes in `fast_stt_options.md` + `system_architecture.md`
- Tag release

### Turbo Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| WER regression (0.2-2% on long-form) | LoRA re-tune on church audio + active-learning priority for low-confidence segments |
| Chunking artifacts in long sermons | Test both chunked (speed) and sequential (accuracy) modes |
| 2070 VRAM tight | Auto layer offload in llama.cpp; monitor nvidia-smi |
| Hallucination on noisy pulpit audio | Existing compression_ratio >2.4 filter + theological prompt injection |
| Conversion failures | Validate exports with sample audio; backup originals |

---

## Phase 1G: Inference Dual-Target — Mac + RTX 2070 (~6-10 hours)

> **Two production endpoints** from a single codebase: mid-tier MacBook M-series (M2 base / M3 non-Pro,
> 8-16 GB unified memory, MLX) and RTX 2070 Windows (8 GB VRAM, CUDA/faster-whisper).
> Backend auto-detected via `--backend=auto`. Same displays, same WebSocket protocol.

### 1G-A. Backend Selection & CUDA Fallback (1-2 hours)

> **Status: DONE** — Backend selection, import guards, CUDA fallback paths, --no-ab, --low-vram implemented.

Add `--backend` flag (auto/mlx/cuda) to `dry_run_ab.py` with import guards for MLX availability. CUDA fallback paths:

| Component | MLX Path (Mac) | CUDA Path (2070/Windows) |
|-----------|---------------|-------------------------|
| STT | mlx-whisper | faster-whisper (CUDA) or transformers pipeline |
| Translation (finals) | mlx-lm Gemma 4B/12B | bitsandbytes 4-bit Gemma 4B or MarianMT |
| Translation (partials) | MarianMT (PyTorch) | MarianMT (PyTorch) — unchanged |

Add `--no-ab` flag to skip 12B on low-VRAM devices. Add `--low-vram` for Marian-only mode.

### 1G-B. CUDA Model Loading & Compatibility (2-4 hours)

> **Status: DONE** — bitsandbytes 4-bit Gemma with EOS fix, --dry-run-text flag.

- Gemma 4B via `AutoModelForCausalLM.from_pretrained(..., load_in_4bit=True, device_map="cuda")`
- Generate wrapper to mimic `mlx_lm.generate` signature
- Dry-run test without mic: `--backend=cuda --no-ab --dry-run-text "Test sentence"`

### 1G-C. Cross-Platform Validation (1-2 hours)

> **Status: PENDING** — Needs cross-platform testing.

- Verify HTTP/WebSocket servers work unchanged on Windows
- Test display HTML files on Windows + LAN phone access
- Validate mid-tier MacBook (M2/M3 non-Pro) as second prod endpoint

### 1G-D. Packaging & Launch Scripts (1-2 hours)

> **Status: PARTIAL** — requirements-nvidia.txt created. Launch scripts deferred.

```
run_mac.sh:       python dry_run_ab.py --backend=mlx "$@"        # Full A/B
run_nvidia.bat:   python dry_run_ab.py --backend=cuda --no-ab %*  # 4B/Marian mode
```

Create `requirements-nvidia.txt` (torch cu121 + transformers + bitsandbytes + faster-whisper).
README section documenting hardware reqs, setup, limitations per endpoint.

### Production Endpoint Tiers

| Endpoint | Hardware | Backend | Capabilities | Role |
|----------|----------|---------|-------------|------|
| Mac (primary dev) | M3 Pro 18 GB | MLX | Full A/B, 4B + 12B | Development + demo |
| Mid-tier MacBook | M2/M3 8-16 GB | MLX | 4B only (--no-ab) | Second prod endpoint |
| RTX 2070 | 8 GB VRAM | CUDA | 4B/Marian (--no-ab) | Church permanent install |

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

## Phase 3: Dual-Endpoint Edge Deployment (Weeks 8-12)

### Why Two Production Endpoints?

The M3 Pro is a development machine. For production:
- **RTX 2070 (primary church install):** 8GB VRAM, 2304 CUDA cores, always-on, no laptop needed
- **Mid-tier MacBook (portable/second venue):** M2/M3 non-Pro, 8-16 GB, MLX inference, portable for events
- Both run `--no-ab` (4B only); dev M3 Pro retains full A/B for testing
- Phase 1G establishes the shared codebase; Phase 3 optimizes for production deployment

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

## Phase 3T: Piper TTS Multi-Language Audio Output (~2-3 months)

> **Goal:** Fine-tuned, multi-language text-to-speech via Piper (rhasspy/piper, ONNX runtime, MIT license).
> Simultaneous audio output to separate hardware channels via AudioFetch. Attendees select language on phone.
>
> **Priority:** English → Spanish → Hindi → Chinese. RTX 2070 for inference, A2000 Ada for training.

### 3T-A. Data & Dataset Preparation (1-2 weeks)

> **Status: DONE** — `prepare_piper_dataset.py` created.

- Extend `download_sermons.py` for TTS-quality audio collection (10-20 hrs per language)
- Update `preprocess_audio.py` for Piper: 16-22 kHz normalize, silence trimming
- Create `training/prepare_piper_dataset.py`: WAVs + transcripts → LJSpeech format
- Reuse existing Whisper pseudo-labeling + theological glossary for text accuracy

### 3T-B. Voice Fine-Tuning (2-4 weeks per language)

> **Status: DONE (scripts)** — `train_piper.py`, `export_piper_onnx.py`, `evaluate_piper.py` created. Actual training pending data + A2000.

Fine-tune from Piper base checkpoints on A2000 Ada:

| Language | Base Checkpoint | Priority |
|----------|----------------|----------|
| English | `en_US-lessac-high` | First (existing data) |
| Spanish | `es_ES-carlfm-high` | Second (most immediate need) |
| Hindi | `hi_IN-kusal-medium` | Third |
| Chinese | `zh_CN-huayan-medium` | Fourth |

Training: batch 16-32, epochs 2000-4000, tensorboard monitoring + sample generation every 50 epochs.
Export: `.ckpt` → `.onnx` + `.json` via onnx-simplifier.
Evaluation: WER round-trip (TTS → STT → compare), prosody checks, theological term pronunciation tests.

### 3T-C. Multi-Channel Integration (1-2 weeks)

> **Status: PENDING**

- Pre-load PiperVoice dict per language at pipeline startup
- Concurrent synthesis via ThreadPoolExecutor on final translations
- Route to specific audio devices via sounddevice (device_map dict)
- Flags: `--tts-enabled`, `--tts-voice custom|stock`, `--multi-channel`
- Target: <200ms added latency for TTS synthesis

### 3T-D. Hardware Routing & AudioFetch (1 week)

> **Status: PENDING**

```
RTX 2070 Pipeline Output:
  Channel 1: Original mic → Main PA (unchanged)
  Channel 2: TTS Spanish → AudioFetch input 2
  Channel 3: TTS Hindi → AudioFetch input 3
  Channel 4: TTS Chinese → AudioFetch input 4

Attendee phone → AudioFetch app → Select language channel
```

Hardware: USB audio adapters or multi-channel interface for separate outputs.
Fallback: Virtual cables (Voicemeeter/VB-Cable) for testing without hardware.

### 3T-E. Testing & Rollout

> **Status: PENDING**

- End-to-end live test: full pipeline + multi-lang audio sync in church
- Docs: `piper_tts_integration.md`, `multi_channel_routing.md`
- Diagnostics: TTS latency/confidence per utterance in JSONL
- Iterate with congregation feedback

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
| Partial latency | 153ms (Mac) | **<150-200ms (Turbo)** | <150ms | <100ms (TensorRT) |
| Final latency | 911ms (Mac) | ~600ms (6C) | ~500ms (adaptive) | <500ms (TensorRT) |
| End-to-end perceived | ~900ms | ~400ms (Turbo) | ~300ms | <400ms |

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
| Turbo chunked vs sequential mode | Phase 1T benchmarks | Chunked = faster, sequential = more accurate for long sermons |
| Scottish accent playlist sources | Phase 1D Week 1 | User provides YouTube playlist URLs |
| Chinese 神 vs 上帝 | Phase 2E data prep | 神 recommended (CUV majority edition) |
| Hindi partial model | Phase 2E integration | IndicTrans2-Dist (quality) vs opus-mt-en-hi (simplicity) |
| Hindi partial display | Phase 2E integration | English partial + Hindi final (recommended) vs delayed partial |
| Train 12B for Hindi/Chinese? | Phase 2E eval | Only if 4B theological accuracy < 70% |
| TTS voice selection per language | Phase 3T-B start | Fine-tune from Piper base vs train from scratch |
| AudioFetch vs software routing | Phase 3T-D | Hardware (AudioFetch Express) vs software (Voicemeeter) |
| TTS on Mac vs 2070 only | Phase 3T-C | 2070 recommended (always-on church install) |
| Mid-tier MacBook model (M2 vs M3) | Phase 1G hardware purchase | M2 Air 16GB or M3 base 16GB |
| CUDA Gemma loading method | **RESOLVED** | bitsandbytes 4-bit (implemented in engines/cuda_engine.py) |
| Separate venvs vs unified | **RESOLVED** | Separate (requirements-nvidia.txt created) |

---

## Reference Documents

| Doc | Contents |
|-----|----------|
| [`implementation_plans.md`](implementation_plans.md) | Practical execution guide: scripts, VRAM budgets, code changes, quality gates |
| [`training_plan.md`](training_plan.md) | Full training schedule, channel inventory, go/no-go gates |
| [`seattle_training_run.md`](seattle_training_run.md) | 6-day unattended run design with data flow |
| [`training_time_estimates.md`](training_time_estimates.md) | A2000 Ada GPU time estimates per model |
| [`accent_tuning_plan.md`](accent_tuning_plan.md) | 4-week accent-diverse STT tuning plan (code complete) |
| [`multi_lingual.md`](multi_lingual.md) | Hindi & Chinese actionable todo list |
| [`multilingual_tuning_proposal.md`](multilingual_tuning_proposal.md) | Full Hindi/Chinese research: corpora, glossaries, QLoRA, evaluation |
| [`rtx2070_feasibility.md`](rtx2070_feasibility.md) | RTX 2070 hardware analysis |
| [`fast_stt_options.md`](fast_stt_options.md) | Lightning-whisper-mlx feasibility (not viable) |
| [`projection_integration.md`](projection_integration.md) | OBS/NDI/ProPresenter integration |
| `piper_tts_integration.md` | Piper TTS setup, voice fine-tuning, ONNX export (planned) |
| `multi_channel_routing.md` | AudioFetch multi-channel hardware routing (planned) |
