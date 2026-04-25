# Accent-Diverse STT Tuning Plan

> **Status:** Implementation started (download/preprocessing/training code written)
> **Last updated:** 2026-02-14
> **Goal:** Reduce accent WER gap to <5% absolute between best and worst accent
> **Hardware:** A2000 Ada 16GB (training), M3 Pro 18GB (inference)
> **Code:** `download_sermons.py --accent`, `preprocess_audio.py`, `prepare_whisper_dataset.py`, `train_whisper.py`

---

## Problem Statement

Base Whisper has significant accent bias:

| Accent | Estimated WER | Gap vs Standard American |
|--------|--------------|--------------------------|
| Standard American (Midwest) | 7-10% | Baseline |
| Canadian English | 8-12% | +1-2% |
| West Coast American | 7-10% | ~0% |
| British RP | 10-15% | +3-5% |
| Scottish English | 22-34% | **+15-24%** |

Scottish English is the worst-performing — WER can be 3x higher than standard American. This matters because Stark Road Gospel Hall has visiting speakers with diverse accents including Scottish preachers.

---

## Architecture (Already Implemented)

The accent-diverse training pipeline adds four capabilities to the existing codebase:

### 1. Accent-Tagged Downloads (`download_sermons.py --accent`)

```bash
python download_sermons.py --accent scottish "https://youtube.com/playlist?list=PLxxx"
python download_sermons.py --accent canadian "https://youtube.com/playlist?list=PLyyy"
python download_sermons.py --accent midwest   # default Stark Road playlist
python download_sermons.py --accent westcoast "https://youtube.com/playlist?list=PLzzz"
python download_sermons.py --accent british "https://youtube.com/playlist?list=PLwww"
```

- Downloads organized into `stark_data/raw/{accent}/`
- Accent label stored in per-video JSON metadata and JSONL log
- Choices: `midwest`, `scottish`, `canadian`, `westcoast`, `british`, `general`

### 2. Accent-Propagating Preprocessing (`preprocess_audio.py`)

- Reads accent from companion `.json` metadata
- Writes chunks to `stark_data/cleaned/chunks/{accent}/`
- Accent stored in `preprocessing_log.json`
- Searches accent subdirectories recursively for input WAVs

### 3. Accent-Balanced Dataset Builder (`prepare_whisper_dataset.py`)

- Temperature-based accent balancing (T=0.5 default: square-root smoothing)
- Confidence filtering (compression_ratio > 2.4, avg_logprob < -1.0, no_speech_prob > 0.6)
- Stratified train/eval split by accent
- Outputs HuggingFace audiofolder with `metadata.csv` (file_name, transcription, accent)

### 4. Accent-Balanced Training (`train_whisper.py`)

- `AccentBalancedTrainer` with `WeightedRandomSampler` — minority accents sampled more frequently
- Per-accent WER evaluation: `wer_scottish`, `wer_canadian`, `wer_midwest`, `wer_westcoast`, `wer_overall`
- `wer_accent_gap` fairness metric (max - min accent WER)
- `--accent-balance` / `--no-accent-balance` flag
- Forces `language="en"` to prevent Scottish→Welsh misclassification

---

## Data Collection Plan

### Target Hours by Accent

| Accent | Target Hours | Min Speakers | Priority | Source Strategy |
|--------|-------------|-------------|----------|----------------|
| **Midwest** | 20-50 hrs | 10+ | High (primary domain) | Stark Road Gospel Hall YouTube |
| **Scottish** | 3-5 hrs | 5+ | **Critical** (22-34% WER gap) | Scottish gospel preachers on YouTube |
| **Canadian** | 1-2 hrs | 3+ | Low (already handled well) | Canadian gospel halls on YouTube |
| **British** | 2-3 hrs | 5+ | Medium (10-15% WER) | British assembly preachers on YouTube |
| **West Coast** | 0-1 hrs | — | Lowest (well-covered by base) | Optional; included via replay data |

### Source Playlists

The user will provide YouTube playlist URLs for each accent. Example workflow:

```bash
# Phase 1: Midwest (primary domain — Stark Road)
python download_sermons.py --accent midwest -n 30

# Phase 2: Scottish (highest priority accent gap)
python download_sermons.py --accent scottish "PLAYLIST_URL_HERE" -n 20

# Phase 3: British
python download_sermons.py --accent british "PLAYLIST_URL_HERE" -n 15

# Phase 4: Canadian
python download_sermons.py --accent canadian "PLAYLIST_URL_HERE" -n 10
```

### Speaker Diversity Guidelines

- **Minimum 5 distinct speakers per accent** — prevents overfitting to one voice
- **Mixed ages and genders** where possible (though gospel preaching skews male/older)
- **Varied recording quality** — some room reverb, some soundboard, to match real conditions
- **Content should be similar** — theological/biblical speech, not casual conversation

---

## Execution Timeline

### Week 1: Midwest Baseline + Scottish Data Collection

| Day | Task | GPU Hrs | Human Hrs |
|-----|------|---------|-----------|
| Mon | Download 30 Midwest sermons (`--accent midwest`) | — | 0.5 |
| Mon-Tue (overnight) | Preprocess + pseudo-label Midwest audio | 7 | — |
| Tue | Verify preprocessing, spot-check 10 transcripts | — | 1 |
| Tue | **Identify Scottish preacher playlists on YouTube** | — | 2-3 |
| Wed | Download Scottish audio (`--accent scottish`) | — | 0.5 |
| Wed-Thu (overnight) | Preprocess + pseudo-label Scottish audio | 2-3 | — |
| Thu | Verify Scottish transcripts — check for misrecognition patterns | — | 1 |
| Thu | Download British audio (`--accent british`) | — | 0.5 |
| Thu-Fri (overnight) | Preprocess + pseudo-label British audio | 1-2 | — |
| Fri | Download Canadian audio (`--accent canadian`) | — | 0.5 |
| Fri-Sat (overnight) | Preprocess + pseudo-label Canadian audio | 0.5-1 | — |

**Week 1 output:** ~25-55 hrs of accent-tagged, preprocessed, pseudo-labeled audio across 4 accents.

### Week 2: Dataset Preparation + Training Round 1

| Day | Task | GPU Hrs | Human Hrs |
|-----|------|---------|-----------|
| Mon | Human correction: bottom 10% of Scottish transcripts (highest error rate) | — | 4-6 |
| Mon | Run `prepare_whisper_dataset.py` — generate accent-balanced dataset | — | 0.5 |
| Mon | Verify: check `metadata.csv` accent distribution, spot-check balance | — | 0.5 |
| Mon-Tue (overnight) | **Whisper LoRA Round 1** with accent-balanced sampler | 1-2 | — |
| Tue | Evaluate: check per-accent WER, identify worst-performing accent | — | 1 |
| Tue | **Baseline comparison:** per-accent WER before vs after fine-tuning | — | 1 |
| Tue-Wed | Human correction: Scottish segments with WER > 30% | — | 3-5 |
| Wed-Thu (overnight) | **Whisper LoRA Round 2** with corrected data | 1-2 | — |
| Thu | Evaluate Round 2, compare accent WER gap | — | 1 |
| Fri | Transfer adapters to Mac, test live inference with Scottish test audio | — | 1-2 |

### Week 3: Active Learning + Gap Closing

| Day | Task | GPU Hrs | Human Hrs |
|-----|------|---------|-----------|
| Mon | Re-pseudo-label all accent data with Round 2 model | 2-3 | — |
| Mon-Tue | Flag bottom 5-15% by confidence per accent, route to correction | — | 3-5 |
| Tue-Wed (overnight) | **Whisper LoRA Round 3** | 1-2 | — |
| Wed | Evaluate: per-accent WER + accent gap metric | — | 1 |
| Thu | If Scottish WER > 15%: collect 2 more hrs Scottish audio, retrain | 0.5-1 | 2-3 |
| Fri | Final evaluation, transfer best model to Mac | — | 1-2 |

### Week 4: Integration + Demo

| Day | Task | GPU Hrs | Human Hrs |
|-----|------|---------|-----------|
| Mon | Integrate accent-tuned Whisper into live pipeline on Mac | — | 2-3 |
| Tue | Test with Scottish test recordings in church environment | — | 2 |
| Wed | Record final per-accent WER metrics for documentation | — | 1 |
| Wed | Update `docs/roadmap.md` with accent tuning results | — | 0.5 |

---

## Quality Gates

### Per-Round Gates

| Gate | Metric | Minimum | Target |
|------|--------|---------|--------|
| Round 1 | Overall WER improvement | > 5% relative | > 15% relative |
| Round 1 | Scottish WER improvement | > 10% relative | > 25% relative |
| Round 1 | Accent WER gap | < 20% absolute | < 10% absolute |
| Round 2 | Scottish WER | < 20% absolute | < 15% absolute |
| Round 2 | Accent WER gap | < 15% absolute | < 8% absolute |
| Round 3 | Scottish WER | < 15% absolute | < 10% absolute |
| **Final** | **Accent WER gap** | **< 10% absolute** | **< 5% absolute** |
| **Final** | **Midwest WER** | **< 8% absolute** | **< 5% absolute** |
| **Final** | **No regression on Midwest** | WER same or better | — |

### Convergence Criteria

Stop accent tuning when ANY of these hold:

- Accent WER gap < 5% absolute for two consecutive rounds
- All accents below 10% absolute WER
- WER improvement < 2% relative for two consecutive rounds on worst accent
- Scottish WER is within 5% of Midwest WER

### Anti-Regression Checks

- **Always evaluate Midwest WER** — this is the primary domain, must not regress
- If Midwest WER increases by > 2% absolute: increase replay ratio to 50%+, reduce accent balancing temperature
- Keep accent balancing temperature at T=0.5 (square-root smoothing) — T→0 (uniform) risks over-representing tiny accent pools

---

## Accent Balancing Details

### Temperature-Based Sampling (prepare_whisper_dataset.py)

Weight per accent: `count^(1/T)` where T is the temperature parameter.

| Temperature | Effect | When to Use |
|-------------|--------|-------------|
| T=1.0 | Proportional (no change) | When accents are already balanced |
| T=0.5 | Square-root smoothing | **Default — good balance** |
| T=0.3 | Aggressive rebalancing | When minority accent has <5% of data |
| T→0 | Uniform (all accents equal) | Risky — can degrade majority accent |

**Example at T=0.5:**

| Accent | Raw Count | Raw % | Balanced Count | Balanced % |
|--------|-----------|-------|----------------|------------|
| Midwest | 5000 | 80% | 3162 | 57% |
| Scottish | 500 | 8% | 1000 | 18% |
| British | 400 | 6.4% | 894 | 16% |
| Canadian | 200 | 3.2% | 632 | 11% |

### WeightedRandomSampler (train_whisper.py)

During training, each batch is sampled with weights inversely proportional to accent frequency:

```
weight[sample] = total / (num_accents * count[accent_of_sample])
```

This ensures every batch has roughly equal accent representation, even if the dataset is 80% Midwest.

---

## Metrics & Logging

### Per-Accent WER (logged during training eval)

```
wer_overall: 0.087
wer_midwest: 0.065
wer_scottish: 0.142
wer_british: 0.098
wer_canadian: 0.078
wer_accent_gap: 0.077   # max - min = 0.142 - 0.065
```

### Tracking Dashboard

After each round, record in `metrics/accent_wer.csv`:

```csv
round,model,wer_overall,wer_midwest,wer_scottish,wer_british,wer_canadian,wer_accent_gap,timestamp
0,base,0.12,0.08,0.28,0.13,0.09,0.20,2026-02-20T10:00:00
1,lora_r1,0.09,0.07,0.18,0.10,0.08,0.11,2026-02-22T10:00:00
2,lora_r2,0.08,0.06,0.14,0.09,0.07,0.08,2026-02-24T10:00:00
3,lora_r3,0.07,0.06,0.11,0.08,0.07,0.05,2026-02-26T10:00:00
```

---

## Relationship to Other Plans

### Integration with Multilingual Pipeline

Accent tuning improves the **STT (Whisper) layer** — the input to all translation pipelines. Better STT directly benefits:

- English-Spanish translations (fewer transcription errors → fewer translation errors)
- Hindi translations (theological terms correctly transcribed → correctly translated)
- Chinese translations (same benefit)

**Recommended order:**
1. Accent-tune Whisper first (this plan)
2. Then fine-tune TranslateGemma for Hindi/Chinese (see `multi_lingual.md`)
3. The improved Whisper produces better input text for all languages

### Integration with Training Plan

This plan slots into the existing `training_plan.md` as a parallel workstream during Phase 1:

```
Existing Plan:                     Accent Plan:
  Week 1: Gospel download            Week 1: Multi-accent download
  Night 1: Preprocess                Night 1-4: Preprocess all accents
  Week 2: Human correction           Week 2: Accent-balanced training
  Night 2: Whisper LoRA              Night 2-3: Accent-aware LoRA
  Night 3: TranslateGemma            (independent)
```

The accent-tuned Whisper LoRA adapters **replace** the standard LoRA adapters from `training_plan.md`. The TranslateGemma training is independent and can run on the same schedule.

### Integration with Roadmap

Accent tuning adds to Phase 1 (Training & Validation) in `roadmap.md`:

- **Phase 1A expanded:** Seattle training run includes accent-diverse data
- **Phase 2C impact:** Hybrid STT-Translation tuning benefits from accent-robust Whisper
- **Phase 3 (RTX 2070):** Accent robustness transfers — LoRA adapters work on any backend

---

## Compute Summary

| Phase | GPU Hours | Human Hours | Calendar |
|-------|-----------|-------------|----------|
| Data collection (all accents) | 11-17 | 4-7 | Week 1 |
| Human correction (Scottish focus) | — | 10-16 | Weeks 1-2 |
| Training rounds 1-3 | 3-6 | 3-5 | Week 2-3 |
| Evaluation + transfer | 1-2 | 3-5 | Week 3 |
| Integration + demo testing | — | 4-6 | Week 4 |
| **Total** | **~15-25** | **~24-39** | **~4 weeks** |

---

## Risk Mitigation

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Scottish playlists too short (<2 hrs) | Medium | Insufficient data for accent adaptation | Supplement with non-church Scottish audio (podcasts, lectures) |
| Scottish WER doesn't improve | Low | Gap remains | Try data augmentation (pitch shift, speed perturbation) |
| Midwest WER regresses | Medium | Primary domain degraded | Increase replay ratio, reduce accent balance temperature |
| Accent-balanced training takes longer | Low | ~20% more training time | Schedule extra overnight run |
| `language="en"` prevents valid Welsh content | Very low | N/A for this use case | Accept — we only care about English |
| Over-fitting to few Scottish speakers | Medium | Poor generalization | Ensure 5+ distinct speakers, use dropout |
