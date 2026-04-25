# Training Plan — March 2026

> **Date:** 2026-03-18
> **Hardware:** NVIDIA RTX 2000 Ada 16GB, 64GB RAM, WSL2
> **Inference:** MacBook M3 Pro, 18GB unified, MLX
> **Data:** 39 sermons → 13,562 clean chunks (30.4 hrs), 242K Bible verse pairs, 458 glossary pairs

---

## Status as of March 18

| Phase | Status | Output |
|-------|--------|--------|
| E — Audio preprocessing | Done | 13,562 chunks (3.3 GB) |
| C — Bible corpus alignment | Done | 242K train + 27K test verse pairs, 458 glossary pairs |
| G — Whisper pseudo-labeling | Running overnight | ~13,562 transcript JSONs with confidence scores |
| I — TranslateGemma 4B QLoRA | Chained after G | Adapter at `/mnt/d/Data/stt-data/models/adapters/gemma/4b_cycle1/` |

---

## Remaining Runs

### 1. Quality Assessment (Phase F)

**What:** Sample 100 pseudo-labeled transcripts, manually transcribe them, compute baseline WER. The WER result determines filtering strategy for Whisper LoRA.

**How:**
```bash
source ~/stt_train_env/bin/activate
cd /mnt/e/code/stark-translate

# Sample 100 segments
python training/assess_quality.py sample \
  --input /mnt/d/Data/stt-data/stark_data/transcripts \
  --n 100

# Manually fill "manual_text" field in assessment_sample.jsonl
# (listen to each audio clip, type the correct transcription)

# Compute WER
python training/assess_quality.py evaluate \
  --input assessment_sample.jsonl
```

**Duration:** ~5 min compute + 2-4 hrs manual transcription (the bottleneck)

**Decision matrix from output:**
| Baseline WER | Action |
|-------------|--------|
| < 10% | Use transcripts directly with confidence filtering |
| 10-20% | Filter worst segments by `avg_logprob` before training |
| 20-30% | Weakly supervised pretraining, then fine-tune on clean subset |
| > 30% | Re-transcribe with `whisper-large-v3` (non-turbo) |

---

### 2. Evaluate TranslateGemma 4B Adapter

**What:** Transfer the 4B QLoRA adapter to Mac, run A/B comparison against base model. Check BLEU/chrF++ on holdout set and spot-check theological terms.

**How:**
```bash
# On WSL — copy adapter to Mac (adjust path for your setup)
rsync -av /mnt/d/Data/stt-data/models/adapters/gemma/4b_cycle1/ \
  mac:~/code/stark-translate/adapters/gemma/4b_cycle1/

# On Mac — run A/B comparison
python tools/validate_roundtrip.py \
  --adapter adapters/gemma/4b_cycle1 \
  --test-data bible_data/aligned/verse_pairs_test.jsonl

# Spot-check theological terms
python tools/validate_roundtrip.py \
  --adapter adapters/gemma/4b_cycle1 \
  --test-data bible_data/glossary/glossary_pairs.jsonl
```

**Duration:** ~30 min (transfer + evaluation)

**Go/no-go:** BLEU improvement > +2 points and theological term accuracy > 65%. If not met, scale up `--max-pairs` (see run 5).

---

### 3. Whisper LoRA Fine-Tuning

**What:** Fine-tune Distil-Whisper on church audio using Phase G pseudo-labels. This is the highest-impact remaining training run — adapts the STT model to church acoustics, speaker patterns, and theological vocabulary.

**Prerequisites:**
- Phase F complete (determines filtering strategy)
- Phase G transcripts available

**How:**
```bash
source ~/stt_train_env/bin/activate
cd /mnt/e/code/stark-translate

# If Phase F says WER > 10%, filter first:
# python training/filter_transcripts.py \
#   --input /mnt/d/Data/stt-data/stark_data/transcripts \
#   --min-logprob -0.7 \
#   --output /mnt/d/Data/stt-data/stark_data/transcripts_filtered

nohup python training/train_whisper.py \
  --input /mnt/d/Data/stt-data/stark_data/transcripts/ \
  --output /mnt/d/Data/stt-data/models/adapters/whisper/cycle1 \
  --model distil-whisper/distil-large-v3.5 \
  --lora-r 32 \
  --lora-alpha 64 \
  --epochs 3 \
  --general-mix 0.3 \
  > /mnt/d/Data/stt-data/whisper_lora.log 2>&1 &
```

**Duration:** ~5-8 hrs GPU (~8-10 GB VRAM)

**Config:** r=32, alpha=64, target `q_proj`+`v_proj`, 70% church + 30% LibriSpeech anti-forgetting mix

**Expected improvement:** 10-30% relative WER reduction on church audio

---

### 4. TranslateGemma 12B QLoRA

**What:** Train the larger 12B model for higher-quality final translations. At inference, the 4B adapter serves as speculative draft and the 12B produces finals.

**Prerequisites:**
- Run 2 shows 4B adapter is working (BLEU > +2)
- Not concurrent with Whisper LoRA (combined VRAM exceeds 16 GB)

**How:**
```bash
source ~/stt_train_env/bin/activate
cd /mnt/e/code/stark-translate

nohup python training/train_gemma.py B \
  --bible-data bible_data/aligned/verse_pairs_train.jsonl \
  --glossary-data bible_data/glossary/glossary_pairs.jsonl \
  --max-pairs 20000 \
  --output /mnt/d/Data/stt-data/models/adapters/gemma/12b_cycle1 \
  > /mnt/d/Data/stt-data/gemma_12b.log 2>&1 &
```

**Duration:** ~18-27 hrs GPU (~14-15 GB VRAM, tight on A2000)

**Config:** batch=1, grad_accum=8, effective batch=8. Same data as 4B run.

**Risk:** VRAM is tight at ~14-15 GB on a 16 GB card. If OOM:
- Reduce `--max-pairs 10000`
- Or reduce `--max-seq-length 256`
- Or add `--lora-r 8` (lower rank)

---

### 5. Scale Up (if plateauing)

**What:** If 4B BLEU improvement < +2 points at 20K pairs, scale up the training data.

**How:**
```bash
# Option A: More Bible pairs
python training/train_gemma.py A \
  --max-pairs 50000 \
  --output /mnt/d/Data/stt-data/models/adapters/gemma/4b_50k

# Option B: Glossary-augmented verse pairs (soft constraints)
python build_glossary.py \
  --output bible_data/glossary/ \
  --augment bible_data/aligned/verse_pairs_train.jsonl \
  --augment-output bible_data/aligned/verse_pairs_train_augmented.jsonl

python training/train_gemma.py A \
  --bible-data bible_data/aligned/verse_pairs_train_augmented.jsonl \
  --max-pairs 20000 \
  --output /mnt/d/Data/stt-data/models/adapters/gemma/4b_augmented
```

**Duration:** Same as original runs (~8-12 hrs for 4B, ~18-27 hrs for 12B)

**Warning:** Beyond 50K pairs, returns diminish and risk of register overfitting increases (model starts producing archaic Spanish for modern input).

---

### 6. Feedback Loop (Cycles 2-5)

**What:** Deploy adapters to Mac → run live → flag low-confidence translations → human-correct → add to training data → retrain. This is the active learning loop that drives continuous improvement.

**How (each cycle):**
```bash
# 1. On Mac: run live, collect flagged segments
python main.py --lang en --tts --adapter adapters/gemma/4b_cycle1
# Low-confidence outputs saved to flagged_translations.jsonl

# 2. Human review: correct flagged translations
# Edit flagged_translations.jsonl, fill "corrected_es" field

# 3. On WSL: retrain with sermon data included
python training/train_gemma.py A \
  --bible-data bible_data/aligned/verse_pairs_train.jsonl \
  --glossary-data bible_data/glossary/glossary_pairs.jsonl \
  --sermon-data /mnt/d/Data/stt-data/stark_data/live_sessions/sermon_pairs.jsonl \
  --max-pairs 20000 \
  --output /mnt/d/Data/stt-data/models/adapters/gemma/4b_cycle2

# 4. Evaluate and compare against previous cycle
```

**Duration per cycle:** ~17-30 hrs GPU + ~5-10 hrs human correction

**Expected:** 3-5 cycles total. First cycle gives 20-40% relative improvement; cycles 2-5 give diminishing but meaningful gains. Stop when improvement < 2% relative for 2 consecutive cycles.

---

## Schedule Overview

| # | Run | Depends on | GPU hrs | Human hrs | VRAM |
|---|-----|-----------|---------|-----------|------|
| 1 | Quality assessment (Phase F) | G output | 0 | 2-4 | — |
| 2 | Evaluate Gemma 4B on Mac | I output | 0.5 | 0.5 | Mac |
| 3 | Whisper LoRA | Phase F | 5-8 | 0.5 | ~8-10 GB |
| 4 | TranslateGemma 12B QLoRA | Run 2 pass | 18-27 | 0.5 | ~14-15 GB |
| 5 | Scale up (if needed) | Run 2 fail | 8-27 | 0.5 | ~10-15 GB |
| 6 | Feedback cycle 2 | Runs 2-4 | 8-12 | 5-10 | ~10-12 GB |
| 7 | Feedback cycle 3 | Cycle 2 | 8-12 | 5-10 | ~10-12 GB |

**Critical path:** Phase F (manual transcription) → Whisper LoRA → Mac eval → feedback loops

**Overnight-friendly runs:** Whisper LoRA (run 3), Gemma 12B (run 4), and each feedback cycle can all be kicked off as overnight `nohup` jobs.

---

## Go/No-Go Gates

| Metric | Minimum | Target |
|--------|---------|--------|
| WER relative improvement | > 10% | > 20% |
| BLEU improvement | > +2 pts | > +4 pts |
| Theological term accuracy | > 65% | > 80% |
| Accent fairness gap | < 10% abs | < 5% abs |
| **Stop condition** | Improvement < 2% relative for 2 consecutive cycles |

---

## Storage Budget

| Asset | Size | Location |
|-------|------|----------|
| Clean chunks | 3.3 GB | `/mnt/d/Data/stt-data/stark_data/cleaned/chunks/` |
| Transcripts (Phase G) | ~200 MB | `/mnt/d/Data/stt-data/stark_data/transcripts/` |
| Bible corpus | ~150 MB | `bible_data/aligned/` |
| Gemma 4B adapter | ~200 MB | `/mnt/d/Data/stt-data/models/adapters/gemma/4b_cycle1/` |
| Gemma 12B adapter | ~400 MB | `/mnt/d/Data/stt-data/models/adapters/gemma/12b_cycle1/` |
| Whisper adapter | ~100 MB | `/mnt/d/Data/stt-data/models/adapters/whisper/cycle1/` |
| Model cache (HF) | ~15-20 GB | `~/.cache/huggingface/` |
| **Total (excl. cache)** | **~4.5 GB** | |
