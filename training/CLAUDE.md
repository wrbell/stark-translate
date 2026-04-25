# training/ — Fine-Tuning & Data Preparation (Windows/WSL)

All training runs on the **Windows Desktop** (WSL2, A2000 Ada 16GB, 64GB RAM). Adapters transfer to Mac for inference. See `CLAUDE-windows.md` for environment setup.

## Overview

Domain adaptation for two models:
1. **Whisper LoRA** — church sermon STT (English acoustic domain)
2. **TranslateGemma QLoRA** — biblical EN↔ES translation

## Audio Preprocessing (10-Step Pipeline)

Raw YouTube church audio → clean training data. Order matters.

1. **Download** — `yt-dlp` from Stark Road Gospel Hall YouTube
2. **Format** — 16kHz mono WAV via ffmpeg
3. **Initial quality gate** — SNR (reject < 10 dB), clipping (reject > 1% at ±0.99)
4. **Classify segments** — `inaSpeechSegmenter` tags speech/music/noise
5. **Source separation** — `demucs` (`htdemucs`, `--two-stems vocals`)
6. **Denoise** — ffmpeg bandpass (80Hz–8kHz) + `noisereduce` (non-stationary, `prop_decrease=0.6–0.8`)
7. **Normalize** — `pyloudnorm` to -16 LUFS, true peak -1 dBTP
8. **VAD chunking** — `silero-vad`, 1–30s segments, 100ms padding
9. **Speaker diarization** (optional) — `pyannote-audio` for primary speaker
10. **Final quality gate** — SNR > 15 dB, duration 1–30s, silence ratio < 50%

**Critical insight:** Don't over-clean. Whisper was trained on noisy audio. Match training noise to expected demo conditions.

## Data Quality Assessment

Before fine-tuning, establish a baseline on 50–100 stratified segments:

| Baseline WER | Strategy |
|-------------|----------|
| < 10% | Use directly with confidence-based filtering |
| 10–20% | Filter worst segments by `avg_logprob` |
| 20–30% | Weakly supervised pretraining, then fine-tune on clean subset |
| > 30% | Re-transcribe with Whisper large-v3 instead of YouTube captions |

**Recommendation:** Re-transcribe all audio with Whisper large-v3 (~7% WER) rather than YouTube auto-captions (~15% WER).

## Biblical Parallel Text Corpus (~155K Verse Pairs)

**Primary datasets:**
- `bible-nlp/biblenlp-corpus` (HuggingFace, 833 languages, CC-BY-4.0)
- `Helsinki-NLP/bible_para` (CC0-1.0)
- `scrollmapper/bible_databases` (GitHub, SQL/JSON/CSV, numeric verse IDs)

| Pair | Register | License |
|------|----------|---------|
| KJV ↔ RVR1909 | Formal-to-formal, archaic | Public domain |
| ASV ↔ RVR1909 | Formal-to-formal, slightly modern | Public domain |
| WEB ↔ Español Sencillo | Modern-to-modern | PD / CC BY-SA 4.0 |
| BBE ↔ RVR1909 | Simplified EN to formal ES | Public domain |
| YLT ↔ RVR1909 | Hyper-literal EN to formal ES | Public domain |

**COPYRIGHT WARNING:** Do NOT use ESV, NASB, NIV, NLT, NVI, LBLA, RVR1960, or DHH. Fair use caps at ~500 verses — not bulk ML training. Stick to pre-1923 or public-domain translations.

**Supplementary data:** Tiered theological glossary (50 boost + 229 master terms), bilingual catechism excerpts, bilingual sermon transcripts, hybrid synthetic translations (60% 12B + 40% DeepL glossary-enforced).

## Sermon Audio via Pseudo-Labeling

1. Collect 20–50 hours of church audio (soundboard preferred)
2. Segment into 5–30s chunks via Silero VAD
3. Run base Distil-Whisper for initial transcriptions
4. Filter by confidence (compression ratio > 2.4 or avg log-prob < -1.0 → discard)
5. Human-correct bottom 20% (prioritize theological terms, biblical names)
6. Format as HuggingFace Dataset (`audio` + `sentence` columns)

**Data volume thresholds:** 5–10h = vocabulary improvement, 20–50h = strong adaptation (sweet spot), 50–100h+ = production-grade.

## Deepgram Oracle Transcription

Deepgram Nova-3 serves as ground-truth label source for Whisper fine-tuning (replacing pseudo-labels from Distil-Whisper).

- **Script:** `training/transcribe_with_deepgram.py` — async, resume support, 300s timeout for large files (40–160 MB)
- **Boosted terms:** 50 Tier 1 theological keyterms from `bible_data/glossary/tier1_boost.json` passed via Deepgram `keyterm` parameter
- **Output:** `.deepgram.json` per sermon with word-level timestamps + confidence scores
- **Cost:** ~$0.0043/min, ~$9 for 35 hours of audio
- **Env var:** `STARK_DEEPGRAM__API_KEY`

## Tiered Glossary System

Two-tier glossary replaces the flat 229-term list:

| Tier | Count | Token Budget | Purpose |
|------|-------|-------------|---------|
| Tier 1 (Boost) | 50 terms | <420 tokens | Deepgram `keyterm` parameter for STT boosting |
| Tier 2 (Master) | 229 terms | — | Normalization, active learning, translation glossary enforcement |

- **Script:** `tools/glossary.py` — `load_tier()`, `validate_boost()`, `build_and_save_tiers()`
- **Build:** `python build_glossary.py --build-tiers   # in training/`
- **Files:** `bible_data/glossary/tier1_boost.json`, `bible_data/glossary/tier2_master.json`

## Data Organization

Training data is split by a fixed cutoff date for reproducible evaluation.

- **Cutoff:** 2026-03-14 (train on historical sermons, eval on future)
- **Sort script:** `tools/sort_sermons.py --output-dir stt-data --catalog stark_data/playlist_catalog.json`
- **Directory structure:** `stt-data/{type}/{year}/` — types: `gospel`, `ministry`, `conference`, `throwback`
- **Manifest:** `stt-data/manifest.json`
- **Catalog:** 333 total sermons (35 local + 298 from playlist catalog)

## Whisper LoRA Configuration

Target `q_proj` + `v_proj` (minimum); expand to `k_proj`, `out_proj`, `fc1`, `fc2` for maximum adaptation.

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Rank (r) | 32 | Most validated across Whisper LoRA studies |
| Alpha | 64 | 2× rank, standard scaling |
| Target modules | `q_proj`, `v_proj` | Minimum viable |
| Dropout | 0.05 | Light regularization |
| Learning rate | 1e-4 | Standard LoRA |
| Batch size | 4 (effective 16 via grad accum) | Conservative for 16GB VRAM |
| Precision | BF16 | Ada native |
| Gradient checkpointing | Enabled | Essential memory savings |
| Max steps | 4,000 (or 3–5 epochs) | First cycle |
| VRAM usage | ~8–10 GB | Comfortable on A2000 Ada |

Mix 70–80% domain data with 20–30% general English (LibriSpeech/Common Voice) for catastrophic forgetting safety.

### Deepgram-Whisper Alignment

- **Script:** `training/align_deepgram_chunks.py` — aligns faster-whisper chunk boundaries with Deepgram word timestamps
- **Output:** HuggingFace audiofolder dataset for Whisper fine-tuning
- **Dataset prep:** `prepare_whisper_dataset.py --gt-source deepgram` for direct Deepgram transcript loading (bypasses pseudo-labeling)
- **Sharded Arrow writes:** `--preprocess-cache` mode streams rows in batches of 1,000 to Arrow shards on disk (`_shards_train/`, `_shards_test/`). Peak RAM ~960 MB of mel data per shard vs 190 GB unbatched. Crash-resume: skips completed shards on re-run.
- **Memory cap:** 12 GB hard virtual memory limit via `resource.setrlimit(RLIMIT_AS)` — prevents OOM-kill on 75K+ chunk datasets
- **Crash recovery:** `training/recover_shards.py` rebuilds DatasetDict from completed shards, streaming one shard at a time (never more than one in memory)

### Whisper LoRA Ablation Design (W0–W9) & Scaling (W12–W15)

Full test matrix defined in `docs/archive/training/whisper_tuning_test_matrix.md`.

| Run | Purpose |
|-----|---------|
| W0 | Baseline (no fine-tuning) |
| W1–W6 | Ablation: learning rate, target modules, replay ratio, data size |
| W7–W9 | Scale winner: epochs, rank |

- **Script:** `training/run_whisper_ablation.sh`
- **Eval metrics:** overall WER, theological term WER, accent fairness gap, general English regression

### W12 Data Scaling Run

W7 config (lr=1e-4, r=32, q_proj+v_proj, replay=0.3, 1 epoch) on the full Deepgram-aligned dataset:

- **Training data:** 198K chunks from 328 sermons (~290 GB Arrow cache at `/mnt/d/Data/stt-data/whisper_dataset_sttdata/.preprocessed_cache/`)
- **Fresh eval set:** 4 post-cutoff sermons (2,706 examples, test split) — Gospel+Teaching from 3/22/26 and 3/29/26
- **Baseline WER on fresh eval:** **21.41%** (normalized)
- **DO NOT TRAIN ON:** `4Es8SrciqV0`, `vRT5RswIHu8`, `FOVTvZednUQ`, `yOzWGOTvTaA`

### W15 Hard Example Mining & Curriculum Learning

Curriculum learning pipeline for targeted Whisper adaptation:

1. **Mine** — `training/mine_hard_examples.py`: batched fp16 inference over chunk pool, per-chunk WER against Deepgram ground truth, Tier 1 theological term detection, resume support, JSONL output
   - Key flags: `--adapter`, `--chunks-json`, `--deepgram-dir`, `--audio-dir`, `--output`, `--batch-size`, `--resume`
2. **Filter** — `training/build_hard_subset.py`: WER-bounded selection (default 0.15–0.80), stratified per-source caps, optional `--include-tier1` to always keep theological chunks
   - Key flags: `--wer-min`, `--wer-max`, `--target-size`, `--max-per-source`, `--include-tier1`
3. **Quality rank** — `training/filter_chunks_by_confidence.py`: top-N selection by `logprob`, `confidence`, or `combined` metric, with min/max duration filtering
4. **Train** — `training/train_whisper.py --init-from <adapter>`: load pre-trained adapter weights with fresh optimizer state (new learning trajectory, no momentum carry-over)
5. **Repeat** — Re-mine on the updated adapter, filter harder examples, train again. 2–4 cycles typical for convergence.

## TranslateGemma QLoRA Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Rank (r) | 16 | Validated for domain-specific translation |
| Alpha | 16 | 1× rank for QLoRA |
| Target modules | `all-linear` | All linear layers |
| Quantization | NF4, double quant, BF16 compute | Standard QLoRA |
| Batch size | 1 (effective 4 via grad accum) | Safe for 16GB |
| Learning rate | 2e-4 | Standard QLoRA rate |
| Epochs | 3 | ~155K verse pairs |
| Packing | Enabled | Multiple verses per sequence |
| Optimizer | `paged_adamw_32bit` | Memory-efficient |
| Max seq length | 512 | Verses rarely exceed 200 tokens |
| VRAM usage | ~10–12 GB | Comfortable on A2000 Ada |

**TranslateGemma chat template** requires `source_lang_code` and `target_lang_code` fields exactly.

**MarianMT** (`Helsinki-NLP/opus-mt-en-es`, ~298MB, ~80ms) supports full fine-tuning without LoRA — lower quality ceiling but faster iteration.

### TranslateGemma S1–S9 Results Summary

Three-phase sweep to find optimal QLoRA configuration:

**Phase 1 — Config sweep (S1–S3):** Learning rate, steps, NEFTune noise. **S1 won** (lr=1e-5, 50 steps).

**Phase 2 — Ratio sweep (S4–S6):** Verse/sermon mix ratio. **S6 won** (balanced 1:1 verse/sermon, COMET proximity to 12B baseline = -0.0002).

**Phase 3 — Scale-up (S7–S9):** Data scaling from 1,800 to 10,000 sermon pairs.

| Run | Data | Key Result |
|-----|------|------------|
| S6 | 1,800 verse + 1,800 sermon (balanced) | **Winner** — COMET prox 12B = -0.0002 |
| S7 | 5,000 60/40 hybrid | Control for S8 |
| S8 | 5,000 100% DeepL | Tests if 12B adds value |
| S9 | 10,000 100% DeepL | Diminishing returns test |

**Key finding:** Verse pairs are NOT harmful — balanced ratio + more data is the formula.

**Hybrid data composition:** 60% TranslateGemma 12B translations + 40% DeepL glossary-enforced translations.

**Data provenance:** All runs tracked in `hybrid_runs/data_provenance.md` with per-sermon chunk breakdowns. `generate_hybrid_synthetic.py` saves `_provenance.json` sidecars and supports `--train-only` to prevent eval data leakage.

### Gemma 4 Benchmark

`training/benchmark_gemma4.py` compares next-gen models against current TranslateGemma:

| Shortname | Model | Prompt Type |
|-----------|-------|-------------|
| `tg4b` | `google/translategemma-4b-it` | translategemma |
| `tg12b` | `google/translategemma-12b-it` | translategemma |
| `e2b` | `google/gemma-4-e2b-it` | gemma4_instruct |
| `e4b` | `google/gemma-4-e4b-it` | gemma4_instruct |

Three evaluation tiers: Tier 1 (Bible verse holdout, BLEU/chrF++/COMET), Tier 2 (Deepgram sermon chunks, COMET-QE + hallucination ratio), Tier 3 (8 theological canary sentences, term accuracy).

> **VRAM caveat (v2026.5):** the result file `metrics/gemma4_benchmark/comparison.json` reports VRAM via `torch.cuda.max_memory_allocated()`, which undercounts on Gemma 4 by ~2× (misses bnb scratch buffers + bf16 PLE embeddings). Treat those numbers as PyTorch-allocator lower bounds, not actual card usage. For accurate per-model VRAM see `docs/archive/v2026.5/BENCHMARK.md` Phase 1A (continuous nvidia-smi sampling).

### Phase 1A — llama.cpp vs HF (v2026.5, 2026-04-25)

`bench_translate_t1_t4.py` extends the Gemma 4 benchmark with three **GGUF/llama.cpp** configs (T2: E2B Q4_K_M, T3: E4B Q4_K_M, T4: E4B + E2B speculative) plus the four HF NF4 configs above. **Result: GGUF wins by 5–9× speedup AND 4× VRAM reduction.** T3 (E4B Q4_K_M) is the new production default for CUDA. T4 spec decode is a single-GPU loss — bookkeeping overhead eats the speedup at α=0.65 acceptance. Full matrix + per-canary disambiguation table in `docs/archive/v2026.5/BENCHMARK.md`.

## Theological Vocabulary Challenges

| English | Spanish Options | Context Rule |
|---------|----------------|--------------|
| Atonement | *expiación* vs *propiciación* | Removal of sin vs. appeasing wrath |
| Covenant | *pacto* (Protestant) vs *alianza* (Catholic) | Match audience |
| Righteousness | *justicia* | Also "justice" — theological context needed |
| James (apostle) | *Jacobo* | Person context (Mark 3:17) |
| James (epistle) | *Santiago* | Book/letter context |
| Sanctification | *santificación* | Consistent |
| Grace | *gracia* | Also "humor/charm" — theological context needed |

**Mitigation:** Tiered glossary system (50 Tier 1 boost terms + 229 Tier 2 master terms) for soft constraint training, Deepgram keyterm boosting, dictionary augmentation, and spot-checking. See **Tiered Glossary System** section above.

## Evaluation Strategy

**Automatic metrics:** SacreBLEU + chrF++ + COMET (use all three).

**Holdout:** ~3,100 verses (10%) stratified by genre — Pentateuch, History, Poetry, Prophecy, Gospels, Epistles, Apocalyptic. Multiple Spanish references (RVR1909 + Español Sencillo).

**Human evaluation:** Adequacy, fluency, theological precision (most critical, not captured by auto metrics).

**Expected improvement targets:**
- SacreBLEU: +3–8 points | chrF++: +2–5 | COMET: +0.02–0.05
- Theological term accuracy: 80%+ (vs. ~40–60% base)
- Whisper WER: 10–30% relative reduction

## Anti-Forgetting Safeguards

- LoRA preserves base weights (disable adapter = instant baseline recovery)
- Replay buffer: 70% general-domain + 30% church-domain
- Learning rate 40× lower than pretraining (~6.25e-5 for Whisper-large)
- Elastic Weight Consolidation if forgetting persists
- **Curriculum learning:** Clean → medium → hard samples. Yields ~5–7% additional WER reduction.

## Compute Timeline

Total estimate: **~48–73 GPU-hrs**, **~33–53 human-hrs** over ~5 weeks.

| Task | Wall Clock | VRAM |
|------|-----------|------|
| Bible corpus download + alignment | ~15 min | CPU |
| Sermon audio download (50 hrs) | ~1–2 hrs | Network |
| Audio preprocessing (10-step) | ~4–6 hrs | GPU (demucs bottleneck) |
| Pseudo-labeling (Whisper large-v3) | ~3–5 hrs | ~8 GB |
| Human correction (bottom 20%) | ~15–25 hrs | Manual |
| Distil-Whisper LoRA (20h audio) | ~5–8 hrs | ~8–10 GB |
| Distil-Whisper LoRA (50h audio) | ~11–15 hrs | ~8–10 GB |
| TranslateGemma 4B QLoRA | ~8–12 hrs | ~10–12 GB |
| TranslateGemma 12B QLoRA | ~18–27 hrs | ~14–15 GB |
| Deepgram Oracle transcription (35h) | ~70 min | API (no GPU) |
| Whisper ablation W1–W6 (6 runs) | ~6–10 hrs | ~8–10 GB |
| Whisper scale W7–W9 (3 runs) | ~4–8 hrs | ~8–10 GB |
| TranslateGemma S7 scaled (5K chunks) | ~4–6 hrs | ~10–12 GB |
| Evaluation (BLEU/WER/COMET) | ~30–60 min | ~6 GB |

**Cycle timing:** Cycle 1 ~40–62 hrs (includes data prep). Cycles 2–5 ~17–30 hrs each.

**Key bottlenecks:** Human correction dominates cycle 1. TranslateGemma 12B tight on VRAM (~14–15 GB peak). Demucs is the data-prep GPU bottleneck — run overnight.

## Observed Benchmarks (A2000 Ada 16GB, WSL2)

Actual wall-clock times from ablation runs (2026-03-20/21).

| Task | Time | VRAM | Notes |
|------|------|------|-------|
| faster-whisper large-v3 (word timestamps, fp16) | ~8x real-time | ~5 GB | 146 min audio = 19 min (614s + 513s) |
| TranslateGemma 4B load (4-bit QLoRA) | ~3-5 min | ~6-8 GB | |
| TranslateGemma 12B load (4-bit) | ~5-8 min | ~7 GB | Fits comfortably on 16GB |
| TranslateGemma translation (4B or 12B) | ~2-3s/input | — | Same speed for both |
| COMET scoring (wmt22-comet-da) | ~3-5 min / 200 pairs | ~2-3 GB | |
| train_gemma.py (per step) | ~4s | ~10-12 GB | |
| evaluate_translation.py (500 verses) | ~47 min | ~6-8 GB | ~5.4s/verse |
| evaluate_sermon.py (28 inputs, 3 models) | ~20-30 min | ~8 GB peak | Sequential model loading |
| A1 training (50 steps) | ~5 min | ~10 GB | |
| B4 training (1114 steps) | ~63 min | ~10 GB | |
| Deepgram Nova-3 API (per sermon) | ~2 min / 60-min sermon | API | ~$0.0043/min, 300s timeout for 40–160 MB files |
| faster-whisper large-v3 batch (33 files) | ~191 min total | ~5 GB peak | Batch transcription pipeline |

## Related Work

- eBible Corpus (2023): NLLB-600M outperformed SMT and OpenNMT on Bible translation
- "From Priest to Doctor" (COLING 2025): Bible-trained models carry strong domain transfer
- BibleNLP community (`biblenlp.github.io`): 833-language ecosystem
- No published Whisper fine-tuning for church/religious speech — genuine research gap
- Domain-adapted Whisper: aviation WER 70%→28% (LoRA), industrial jargon ~1–2% WER after 5 epochs

---

## Adapter Export & Transfer

After training, LoRA adapters must be transferred to inference machines:

1. **Location**: Adapters saved to `fine_tuned_*/` dirs on WSL (safetensors format)
2. **Transfer to Mac**: Copy adapter dir to `adapters/{model}/active/` (see `docs/deploy.md` for automated rsync pipeline)
3. **Validate on Mac**: Run health check — 5 canonical sentences, check expected substrings + latency < 5s per sentence
4. **Version naming**: `cycle{N}_{YYYYMMDD}_{sha256[:8]}` (SHA-256 over `adapter_model.safetensors`)
5. **Manifest**: `adapters/manifest.json` tracks all versions with training metrics (final loss, eval WER, epochs, data size)
6. **Rollback**: Two-slot system — `active/` + `previous/`. Swap back if health check fails post-activation.

## Adding a New Language Corpus

When adding Hindi or Chinese translation training (see `docs/archive/research/multi_lingual.md`):

1. **Find aligned verse pairs** — `bible-nlp/biblenlp-corpus` has 833 languages (CC-BY-4.0). Hindi IRV: `hin2017` (~31K verses). Chinese CUV-S: `cmn-cu89s` (~31K verses, public domain).
2. **Prepare with `prepare_bible_corpus.py`** — input: two translation dirs, output: JSONL pairs with `source_lang_code`, `target_lang_code`, `source_text`, `target_text`, `verse_id`.
3. **Build theological glossary** (100–150 terms minimum) using `training/build_glossary.py` as template. Include honorifics (Hindi तू for divine address) and denomination-specific terms (Chinese 圣灵 not 圣神).
4. **QLoRA config**: Same as Spanish but **r=32** (new language direction needs higher rank). `max_seq_length=768` for Hindi (2.5–3.5x token fertility), 512 for Chinese.
5. **Evaluation**: chrF++ (primary — handles morphology and no-space scripts), COMET, theological term accuracy. Use `--tokenize zh` for Chinese SacreBLEU.
6. **Copyright**: Same rules apply — only pre-1923 or explicitly public domain translations. No ESV, NASB, NIV, NLT, NVI, LBLA, RVR1960, DHH.

## Go/No-Go Gates

Quick reference for training convergence criteria (from `docs/archive/research/accent_tuning_plan.md` and evaluation strategy above):

- **WER improvement**: > 10% relative (minimum), > 20% relative (target)
- **BLEU improvement**: > +2 points (minimum), > +4 points (target)
- **Theological term accuracy**: > 65% (minimum), > 80% (target)
- **Accent fairness gap**: < 10% absolute (minimum), < 5% absolute (target)
- **Stop condition**: Improvement < 2% relative for 2 consecutive cycles on worst-performing metric
