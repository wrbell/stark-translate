# Training Plan -- Stark Road Bilingual Speech-to-Text

> **Date:** 2026-02-08
> **Training hardware:** NVIDIA A2000 Ada 16GB, 64GB RAM, WSL2
> **Inference hardware:** MacBook M3 Pro, 18GB unified memory, MLX
> **Source channel:** [Stark Road Gospel Hall on YouTube](https://www.youtube.com/@StarkRoadGospelHall)
> **Time estimates derived from:** [`docs/training_time_estimates.md`](training_time_estimates.md)

---

## Table of Contents

1. [Channel Inventory](#1-channel-inventory)
2. [Phase 1: Gospel Meetings (Primary Dataset)](#2-phase-1-gospel-meetings-primary-dataset)
3. [Phase 2: Ministry Meetings (Domain Expansion)](#3-phase-2-ministry-meetings-domain-expansion)
4. [Phase 3: Other Sources](#4-phase-3-other-sources)
5. [Phase 4: Active Learning Cycles](#5-phase-4-active-learning-cycles)
6. [Timeline Summary](#6-timeline-summary)
7. [Hardware Schedule](#7-hardware-schedule)
8. [Storage Requirements](#8-storage-requirements)
9. [Quality Milestones and Go/No-Go Gates](#9-quality-milestones-and-gono-go-gates)
10. [Step-by-Step Execution Checklist](#10-step-by-step-execution-checklist)

---

## 1. Channel Inventory

The Stark Road Gospel Hall YouTube channel contains **52 playlists** with **zero overlap** between the two primary playlists (Gospel and Ministry). The channel's `/videos` tab shows only 102 public videos, but playlists contain many more (likely unlisted videos accessible via playlist links).

### 1.1 Primary Playlists

| Playlist | Videos | Total Hours | Avg per Video | Content Type |
|----------|--------|-------------|---------------|-------------|
| **Gospel** | 275 | 249.8 hrs | ~55 min | Evangelistic preaching |
| **Ministry** | 325 | 339.0 hrs | ~63 min | Teaching and exposition |

**Combined:** 600 unique videos, ~589 hours of audio. No overlap between these two playlists.

### 1.2 Conference Playlists

| Playlist | Videos | Total Hours | Avg per Video |
|----------|--------|-------------|---------------|
| Annual Conferences (2022-2023) | 13 | 27.7 hrs | ~128 min |
| Conference 2025 | 8 | 15.1 hrs | ~113 min |
| Conference 2024 | 8 | 15.5 hrs | ~116 min |
| Conference 2020 | 2 | 3.2 hrs | ~95 min |

**Conference total:** 31 videos, ~61.5 hours. Longer sessions, multiple speakers, higher vocabulary diversity.

### 1.3 Bible Reading / Book Study Playlists

| Playlist | Videos | Total Hours | Avg per Video |
|----------|--------|-------------|---------------|
| Revelation | 29 | 49.0 hrs | ~101 min |
| Acts: The Gospel in Motion | 32 | 43.8 hrs | ~82 min |
| John | 35 | 36.1 hrs | ~62 min |
| Throwback Thursdays | 74 | 71.9 hrs | ~58 min |
| Mark: A Study of the Gospel of Mark | 23 | 21.1 hrs | ~55 min |
| 1 Timothy: A Healthy Household of God | 6 | 15.0 hrs | ~150 min |
| Colossians Bible Study | 10 | 10.6 hrs | ~64 min |
| Bi-monthly Bible Readings (all periods) | 14 | 24.5 hrs | ~105 min |
| 1 Corinthians Bible Reading | 3 | 3.0 hrs | ~59 min |
| 1 Peter | 5 | 7.2 hrs | ~87 min |
| 2 Peter | 3 | 3.9 hrs | ~77 min |
| 1-3 John | 3 | 5.1 hrs | ~102 min |
| Jude | 2 | 3.5 hrs | ~104 min |
| Genesis, the Foundation of all Doctrine | 11 | 12.2 hrs | ~67 min |
| James: Faith in Action | 6 | 4.0 hrs | ~40 min |
| Bible Series: 1 and 2 Thessalonians | 9 | 5.8 hrs | ~39 min |
| Galatians Bible Study | 5 | 3.3 hrs | ~40 min |
| Philemon | 1 | 0.5 hrs | ~31 min |
| Titus: Godly Living | 3 | 1.6 hrs | ~32 min |
| 2 Timothy: Faithful to the end | 3 | 1.9 hrs | ~38 min |

### 1.4 Gospel Series (Sub-playlists)

| Playlist | Videos | Total Hours | Avg per Video |
|----------|--------|-------------|---------------|
| Fall Gospel Series 2025 | 16 | 18.9 hrs | ~71 min |
| Fall Gospel Series: Are You Right with God? | 17 | 20.7 hrs | ~73 min |
| Jesus Saves! Gospel Series | 16 | 19.4 hrs | ~73 min |
| Peace? It's Possible! Gospel Series | 15 | 16.3 hrs | ~65 min |
| Gospel Series 2021 | 20 | 17.8 hrs | ~53 min |
| Gospel Series 2020 | 24 | 14.2 hrs | ~36 min |
| Easter Gospel Series with Gene Higgins | 9 | 8.7 hrs | ~58 min |
| God Saved Me | 6 | 6.9 hrs | ~69 min |
| What's Wrong With This World? | 14 | 7.9 hrs | ~34 min |

**Note:** Many of these sub-series are likely subsets of the main "Gospel" playlist (275 videos). They provide useful category tags but do not represent additional unique content beyond the Gospel playlist total.

### 1.5 Other / Special Playlists

| Playlist | Videos | Total Hours | Notes |
|----------|--------|-------------|-------|
| Gilliland Ministry 2024 | 4 | 4.8 hrs | Guest speaker series |
| Coulson Ministry 2024 | 4 | 4.6 hrs | Guest speaker series |
| Bible Teaching with Mark Sweetnam | 3 | 3.5 hrs | Guest speaker |
| A Biblical Perspective on Relationships | 4 | 6.0 hrs | Topical series |
| Christ in the Offerings | 6 | 6.9 hrs | Typology/doctrinal |
| Summer 2021: Prophetic Subjects | 4 | 4.0 hrs | Eschatology |
| Summer 2021: The NT Assembly | 2 | 2.4 hrs | Ecclesiology |
| Summer Ministry 2020: Sermon on the Mount | 7 | 6.1 hrs | Expository |
| Sunday School | 5 | 6.6 hrs | Teaching, different style |
| Questions & Answers Series | 6 | 9.0 hrs | Interactive Q&A |
| Christmas Programs | 2 | 1.6 hrs | Special programs |
| VBS 2021 - Amazing Rescues | 6 | 1.6 hrs | Children's content |
| VBS 2020 - Under the Sea | 5 | 2.4 hrs | Children's content |
| God's Great Remedies (Children) | 6 | 1.9 hrs | Children's lessons |

### 1.6 Unique Content Estimate

Most sub-series playlists are subsets of the two main playlists. The primary unique content pools are:

| Source | Unique Videos (est.) | Unique Hours (est.) |
|--------|---------------------|---------------------|
| Gospel (main playlist) | ~275 | ~250 hrs |
| Ministry (main playlist) | ~325 | ~339 hrs |
| Conferences (combined) | ~31 | ~62 hrs |
| Children's / VBS content | ~17 | ~6 hrs |
| **Estimated unique total** | **~648** | **~657 hrs** |

**For training purposes, we exclude children's content** (different register, multiple children speaking, songs) and focus on the ~651 hours of adult sermon and teaching content.

---

## 2. Phase 1: Gospel Meetings (Primary Dataset)

The Gospel playlist is the natural starting point: evangelistic preaching with consistent vocabulary, typically one primary speaker per session, and strong theological term density.

### 2.1 Round 1 -- Quick Start (30 Sermons, ~30 Hours)

**Goal:** Establish a working fine-tuned pipeline as fast as possible. Validate the entire workflow before committing to larger data collection.

#### Step 1: Download (Automated)

```bash
python download_sermons.py -n 30
```

| Task | Time | Notes |
|------|------|-------|
| Download 30 videos at 100 Mbps | ~1.0 hr | Audio only, 16kHz mono WAV |
| Disk usage | ~1.7 GB | ~57 MB/hr raw audio |

#### Step 2: Audio Preprocessing (Automated, GPU)

```bash
nohup python training/preprocess_audio.py --input stark_data/raw --output stark_data/cleaned --resume > preprocess.log 2>&1 &
```

| Task | Time | Notes |
|------|------|-------|
| 10-step pipeline (30 hrs raw) | 3.2--5.2 hrs | Demucs is the bottleneck (~40% of files need it) |
| Output: cleaned, VAD-segmented chunks | ~5,040 segments | ~15s avg, 1--30s range |

#### Step 3: Pseudo-Labeling (Automated, GPU)

```bash
nohup python training/transcribe_church.py --backend faster-whisper --resume > transcribe.log 2>&1 &
```

| Task | Time | Notes |
|------|------|-------|
| Whisper large-v3 on 30 hrs (~21 hrs speech) | ~2.2 hrs | ~12x real-time on A2000 Ada |
| Output: JSON transcripts per segment | ~5,040 transcripts | With confidence scores |

#### Step 4: Human Correction (Manual)

| Task | Time | Notes |
|------|------|-------|
| Correct bottom 20% by confidence | ~25 hrs | ~1,008 segments, 6x real-time |
| Can be spread over 8--10 days | ~2.5--3 hrs/day | Use Label Studio for efficiency |

**Strategy to reduce human time:** Start with the bottom 10% only (~12.5 hrs). If the baseline WER from Whisper large-v3 is already < 10%, the labels may be good enough for a first training round without full correction.

#### Step 5: Bible Corpus Preparation (Automated, CPU -- runs in parallel)

```bash
python training/prepare_bible_corpus.py
python build_glossary.py
```

| Task | Time | Notes |
|------|------|-------|
| Download Bible corpus from HuggingFace/GitHub | ~5 min | ~500 MB total |
| Align verse pairs (JOIN on verse ID) | ~10 min | ~155K pairs output |
| Build theological glossary | ~5 min | ~229 terms, 458 training pairs |
| **Total** | **~20 min** | Fully automated, CPU-only |

This is **independent of audio data** and can run anytime.

#### Step 6: Model Training (Automated, GPU)

All three translation models use the same Bible corpus. Whisper uses the sermon audio. These are independent and can be scheduled across different nights.

**Whisper LoRA (Quick)**

```bash
nohup python training/train_whisper.py --epochs 3 --dataset stark_data/cleaned > whisper_train.log 2>&1 &
```

| Parameter | Value |
|-----------|-------|
| Data | ~7,710 samples (5,400 church + 2,310 replay) |
| Steps/epoch | ~482 |
| Time/step | ~1.8s |
| **Total (3 epochs, with overhead)** | **~1.0 hr** |
| VRAM | ~8--11 GB |

**MarianMT Full Fine-Tune (Quick)**

```bash
nohup python training/train_marian.py --epochs 5 > marian_train.log 2>&1 &
```

| Parameter | Value |
|-----------|-------|
| Data | ~155,458 examples (Bible + glossary) |
| Steps/epoch | ~4,858 |
| Time/step | ~0.2s |
| **Total (5 epochs, with eval)** | **~2.2 hrs** |
| VRAM | ~3.7--4.1 GB |

**TranslateGemma 4B QLoRA (Overnight)**

```bash
nohup python training/train_gemma.py A --epochs 3 > gemma4b_train.log 2>&1 &
```

| Parameter | Value |
|-----------|-------|
| Data | ~155,458 examples, packed to ~34,500 sequences |
| Steps/epoch | ~4,313 |
| Time/step | ~3.0s |
| **Total (3 epochs, with overhead)** | **~10--14 hrs** |
| VRAM | ~9--13 GB |

#### Step 7: Evaluation (Automated, GPU)

```bash
python training/evaluate_translation.py --models all
```

| Task | Time |
|------|------|
| Translation metrics (BLEU + chrF++ + COMET) on 3.1K holdout verses | ~20--35 min |
| Whisper WER on held-out sermon segments (~500 segments) | ~12--16 min |
| Theological term spot-check (~229 terms) | ~5 min |
| **Total** | **~45--60 min** |

#### Round 1 Summary

| Task | GPU Hours | Human Hours | Calendar Days |
|------|-----------|-------------|---------------|
| Download | 1.0 | 0 | Day 1 |
| Preprocessing + pseudo-labeling | 5.4--7.4 | 0 | Night 1 |
| Bible corpus prep | 0.3 | 0 | Day 1 (parallel) |
| Human correction (bottom 20%) | 0 | 12.5--25 | Days 2--10 |
| Whisper LoRA + MarianMT + eval | 4.2 | 0 | Night 2 |
| TranslateGemma 4B QLoRA | 10--14 | 0 | Night 3 |
| Final evaluation + adapter transfer | 1.0 | 1.0 | Day 4 |
| **Round 1 Total** | **~22--28 GPU-hrs** | **~14--26 human-hrs** | **~10--14 days** |

**Critical path:** Human correction is the calendar bottleneck. GPU work finishes in 3 overnight sessions. If human correction is deferred (using raw Whisper labels), GPU-only work completes in **4 days**.

### 2.2 Round 2 -- Expansion (Add 50 Sermons, Total ~80 Hours)

**Trigger:** Round 1 evaluation shows WER improvement and BLEU improvement justify more data.

#### Incremental Work

| Task | Time | Notes |
|------|------|-------|
| Download 50 additional videos | ~1.7 hrs | Cumulative: 80 sermons |
| Preprocess new 50 hrs | 5.3--8.7 hrs | Overnight |
| Pseudo-label new 50 hrs | ~3.5 hrs | Can chain after preprocessing |
| Human correction (new segments, bottom 20%) | ~42 hrs | Spread over 2--3 weeks |
| Re-pseudo-label old 30 hrs with improved Whisper | ~2.2 hrs | Optional but recommended |

#### Retraining

| Model | Time | Notes |
|-------|------|-------|
| Whisper LoRA (80 hrs, 3 epochs) | ~1.7 hrs | Starts from Round 1 checkpoint |
| MarianMT (5 epochs, same Bible corpus) | ~2.2 hrs | Restart from scratch (fast enough) |
| TranslateGemma 4B (1--2 more epochs) | ~5--8 hrs | Continue from Round 1 checkpoint |
| Evaluation | ~1.0 hr | |
| **Round 2 GPU Total** | **~17--25 GPU-hrs** | |
| **Round 2 Human Total** | **~42 human-hrs** | Dominates calendar time |

**Calendar time:** ~2--3 weeks (dominated by human correction at ~3 hrs/day).

### 2.3 Round 3 -- Full Gospel Corpus (All 275 Sermons, ~250 Hours)

**Trigger:** Only if Round 2 shows the WER/BLEU improvement curve has not yet plateaued.

| Task | Time | Notes |
|------|------|-------|
| Download remaining 195 videos | ~6.5 hrs | Overnight |
| Preprocess 170 additional hrs | 18--28 hrs | 2--3 overnight runs |
| Pseudo-label 170 additional hrs | ~12 hrs | Overnight |
| Human correction (new segments, ~20%) | ~140 hrs | Major commitment |
| Whisper LoRA retrain (250 hrs, 3 epochs) | ~3--4 hrs | From checkpoint |
| TranslateGemma 4B (1 more epoch) | ~5 hrs | From checkpoint |
| **Round 3 GPU Total** | **~45--54 GPU-hrs** | ~5 overnight runs |
| **Round 3 Human Total** | **~140 human-hrs** | ~6--8 weeks at 3 hrs/day |

**Recommendation:** Round 3 is only worthwhile if quality metrics from Round 2 show continued strong improvement. By Round 2 (80 hrs audio), diminishing returns are likely for Whisper. The additional data primarily helps with speaker diversity and rare vocabulary coverage.

**Shortcut:** Use improved Whisper from Round 2 to pseudo-label the full corpus, skip human correction entirely, and use confidence-filtered labels. This gets 80% of the benefit at 10% of the human cost.

---

## 3. Phase 2: Ministry Meetings (Domain Expansion)

The Ministry playlist (325 videos, 339 hours) contains a different speaking style: more teaching/expository vs. the Gospel playlist's evangelistic preaching.

### 3.1 Why Ministry Data Matters

| Dimension | Gospel Playlist | Ministry Playlist |
|-----------|----------------|-------------------|
| Speaking style | Evangelistic, persuasive | Expository, didactic |
| Vocabulary | Soteriological (salvation, atonement, grace) | Broader theological (ecclesiology, eschatology, typology) |
| Speaker count | Fewer speakers, often guests | More regular speakers |
| Audience interaction | Minimal | Q&A, discussion |
| Average length | ~55 min | ~63 min |
| Scripture density | Moderate (key verses) | High (verse-by-verse) |

### 3.2 Training Approach

**Start from Phase 1 checkpoints** -- do not train from scratch. The Ministry data builds on what the Gospel data already taught.

#### Round 1: Sample 30 Ministry Meetings (~30 hrs)

| Task | GPU Hours | Human Hours |
|------|-----------|-------------|
| Download + preprocess + pseudo-label | ~7 hrs | 0 |
| Human correction (bottom 10% -- lighter touch) | 0 | ~12.5 hrs |
| Whisper LoRA (continue from Phase 1, 2 epochs) | ~0.7 hrs | 0 |
| Evaluation | ~1.0 hr | 0 |
| **Total** | **~9 GPU-hrs** | **~12.5 human-hrs** |

**Key question:** Does the Ministry data improve or degrade Gospel-domain WER? Evaluate on both domains separately. If Ministry data helps (expected -- more speaker diversity), proceed with expansion.

#### Round 2: Full Ministry Expansion (100+ hours)

Only if Round 1 shows clear benefit. Approach mirrors Phase 1 Round 2.

| Task | GPU Hours | Human Hours |
|------|-----------|-------------|
| Download 70 more videos (~70 hrs) | ~2.5 hrs | 0 |
| Preprocess + pseudo-label | ~12 hrs | 0 |
| Human correction (bottom 10%) | 0 | ~25 hrs |
| Whisper LoRA retrain (combined 160 hrs, 2 epochs) | ~1.4 hrs | 0 |
| Evaluation | ~1.0 hr | 0 |
| **Total** | **~17 GPU-hrs** | **~25 human-hrs** |

### 3.3 Translation Model Impact

The Ministry playlist does **not** change the translation training data -- TranslateGemma and MarianMT train on the fixed 155K Bible corpus regardless of audio source. However, Ministry meetings contain more diverse source text patterns, so improved Whisper transcription indirectly feeds better text to the translation models.

---

## 4. Phase 3: Other Sources (Conferences, Bible Readings, Specials)

### 4.1 Conferences (~62 Hours, 31 Videos)

**Value:** High. Multiple speakers per session, longer format, diverse theological topics, higher vocabulary breadth. Conference audio often has different acoustics (larger rooms, different PA systems).

| Task | GPU Hours | Human Hours |
|------|-----------|-------------|
| Download all 31 conference videos | ~1.0 hr | 0 |
| Preprocess + pseudo-label | ~10 hrs | 0 |
| Human correction (bottom 10%) | 0 | ~19 hrs |
| Whisper LoRA continue (2 epochs) | ~1.0 hr | 0 |
| **Total** | **~12 GPU-hrs** | **~19 human-hrs** |

**Schedule:** After Phase 2 Round 1 if quality is still improving.

### 4.2 Bible Reading Series (~300+ Hours across multiple playlists)

The verse-by-verse reading playlists (Revelation, Acts, John, Mark, etc.) total over 300 hours across ~30 playlists. These are highly valuable because:

- Dense Scripture quotation (tests theological term accuracy)
- Slower, more deliberate speech (different acoustic profile)
- Long-form sustained attention (tests consistency over time)

**Approach:** Sample 20--30 hours across different book studies. Do not download all 300 hours.

| Task | GPU Hours | Human Hours |
|------|-----------|-------------|
| Download ~30 sampled videos | ~1.0 hr | 0.5 (selecting samples) |
| Preprocess + pseudo-label | ~7 hrs | 0 |
| Human correction (bottom 10%) | 0 | ~12.5 hrs |
| Whisper LoRA continue (2 epochs) | ~0.7 hrs | 0 |
| **Total** | **~9 GPU-hrs** | **~13 human-hrs** |

### 4.3 Throwback Thursdays (74 Videos, 72 Hours)

Historical recordings -- likely lower audio quality (older recordings, different equipment). **Defer** unless Phase 1--2 models struggle with older recording styles.

### 4.4 Children's Content (17 Videos, ~6 Hours)

**Exclude.** Different register, multiple young speakers, songs, and sound effects. Not representative of the target use case (adult sermon transcription).

### 4.5 Diminishing Returns Analysis

Based on published Whisper fine-tuning literature and the project's domain-adapted Whisper studies cited in CLAUDE.md:

| Total Training Hours | Expected WER (Relative to Baseline) | Marginal Improvement |
|---------------------|--------------------------------------|---------------------|
| 0 (base model) | 100% (baseline) | -- |
| 30 hrs (Phase 1 R1) | 60--80% (20--40% reduction) | High |
| 80 hrs (Phase 1 R2) | 50--70% (30--50% reduction) | Moderate |
| 160 hrs (+ Phase 2) | 45--60% (40--55% reduction) | Low-moderate |
| 250+ hrs (+ Phase 3) | 40--55% (45--60% reduction) | Low |
| 400+ hrs (full corpus) | 38--52% (48--62% reduction) | Diminishing |

**Recommendation:** Stop adding new audio data when the WER improvement from the last data increment is < 5% relative. This is likely to occur around 80--160 hours of training data.

---

## 5. Phase 4: Active Learning Cycles

Each phase above implicitly includes the active learning feedback loop: **infer -> flag -> correct -> retrain**. This section describes the explicit cycle structure.

### 5.1 Cycle Structure

Each active learning cycle follows this pattern:

```
1. Run inference on held-out audio with current model
2. Score segments by confidence (avg_logprob, no_speech_prob, compression_ratio)
3. Flag bottom 5--15% for human review
4. Human corrects flagged segments in Label Studio
5. Merge corrections into training dataset
6. Retrain Whisper LoRA (2 epochs from last checkpoint)
7. Evaluate on held-out test set
8. Compare metrics to previous cycle -- stop if converged
```

### 5.2 Per-Cycle Time (After Initial Training)

From `training_time_estimates.md` Section 10:

| Component | Cycle 2 | Cycle 3 | Cycle 4 | Cycle 5 |
|-----------|---------|---------|---------|---------|
| Re-pseudo-labeling | 2.2 hrs | 2.0 hrs | 1.8 hrs | 1.8 hrs |
| Human correction (flagged only) | 10--15 hrs | 5--8 hrs | 3--5 hrs | 2--3 hrs |
| Whisper LoRA (incremental, 2 ep) | 0.7 hrs | 0.7 hrs | 0.7 hrs | 0.7 hrs |
| TranslateGemma 4B (1--2 ep) | 5--8 hrs | 4--5 hrs | 3--4 hrs | 3--4 hrs |
| MarianMT (incremental, 2 ep) | 0.9 hrs | 0.9 hrs | 0.9 hrs | 0.9 hrs |
| Evaluation | 1.0 hr | 1.0 hr | 1.0 hr | 1.0 hr |
| **GPU total** | ~10--13 hrs | ~8--10 hrs | ~7--9 hrs | ~7--9 hrs |
| **Human total** | ~10--15 hrs | ~5--8 hrs | ~3--5 hrs | ~2--3 hrs |

### 5.3 Expected Quality Improvements

| Cycle | WER Improvement (Relative) | BLEU Improvement | Notes |
|-------|---------------------------|------------------|-------|
| 1 (initial training) | 20--40% reduction | +3--5 points | Biggest single jump |
| 2 | 10--20% reduction | +2--3 points | Strong continued gains |
| 3 | 5--10% reduction | +1--2 points | Moderate gains |
| 4 | 2--5% reduction | +0.5--1 point | Approaching plateau |
| 5 | < 2% reduction | < 0.5 point | Diminishing returns -- stop |

### 5.4 Convergence Criteria

Stop the active learning loop when ANY of these hold:

- WER improvement < 2% relative for two consecutive cycles
- BLEU improvement < 0.5 points for two consecutive cycles
- Human correction yield < 5% of flagged segments actually had errors
- Total WER on held-out set < 5% absolute (production quality)

---

## 6. Timeline Summary

### 6.1 Master Schedule

```
WEEK 1: Setup + Phase 1 Round 1 Data Collection
  Mon:    Download 30 Gospel sermons (~1 hr)
          Start Bible corpus preparation (~20 min, CPU)
  Mon-Tue (overnight): Audio preprocessing + pseudo-labeling (7 hrs GPU)
  Tue-Fri: Begin human correction (~3 hrs/day = 12 hrs)
  Fri (overnight): Whisper LoRA + MarianMT + eval (4.2 hrs GPU)

WEEK 2: Phase 1 Round 1 Training + Continued Correction
  Sat-Sun (overnight): TranslateGemma 4B QLoRA (12 hrs GPU)
  Mon:    Evaluate all models, transfer adapters to Mac
          Test inference with fine-tuned models on Mac
  Mon-Fri: Continue human correction if needed (~3 hrs/day)
  Fri:    Complete Round 1 evaluation

  ** GO/NO-GO GATE: Review WER and BLEU improvements **
  ** If WER improved > 15% relative AND BLEU > +2 points: proceed to Round 2 **

WEEK 3-4: Phase 1 Round 2 (if approved)
  Week 3 Mon: Download 50 more Gospel sermons
  Week 3 Mon-Tue (overnight): Preprocess + pseudo-label (9-12 hrs GPU)
  Week 3-4: Human correction (~3 hrs/day, ~14 days)
  Week 4 Fri (overnight): Retrain Whisper + MarianMT (4 hrs GPU)
  Week 4 Sat (overnight): Continue TranslateGemma 4B (5-8 hrs GPU)

  ** GO/NO-GO GATE: WER still improving? **

WEEK 5: Active Learning Cycle 2
  Mon-Tue (overnight): Re-pseudo-label with improved model (2.2 hrs GPU)
  Tue-Fri: Human correction on flagged segments (~10-15 hrs)
  Fri (overnight): Incremental retrain all models (10 hrs GPU)
  Sat: Evaluate

WEEK 6: Phase 2 Start (Ministry Data)
  Mon: Download 30 Ministry sermons
  Mon-Tue (overnight): Preprocess + pseudo-label (7 hrs GPU)
  Tue-Fri: Human correction (~3 hrs/day)
  Fri-Sat (overnight): Retrain Whisper from Phase 1 checkpoint (1 hr GPU)
  Sat: Evaluate on both Gospel AND Ministry test sets

  ** GO/NO-GO GATE: Ministry data helps Gospel WER? **

WEEK 7-8: Active Learning Cycles 3-4
  Continue the infer -> flag -> correct -> retrain loop
  Each cycle: ~1 overnight GPU run + 3-5 days human correction

WEEK 9: Final Evaluation + Demo Preparation
  Transfer final adapters to Mac
  Run full A/B comparison with fine-tuned models
  Tune Streamlit dashboard for demo
  Test in church environment

WEEK 10+: Phase 3 (if needed) + Production Polish
  Conference data, Bible readings, etc.
  Only if quality metrics justify continued investment
```

### 6.2 Cumulative GPU and Human Hours

| Milestone | GPU Hours (Cumulative) | Human Hours (Cumulative) | Calendar |
|-----------|----------------------|--------------------------|----------|
| Phase 1, Round 1 complete | 22--28 hrs | 14--26 hrs | Week 2 |
| Phase 1, Round 2 complete | 39--53 hrs | 56--68 hrs | Week 4 |
| Active Learning Cycle 2 | 49--66 hrs | 66--83 hrs | Week 5 |
| Phase 2, Round 1 complete | 58--75 hrs | 78--96 hrs | Week 6 |
| Active Learning Cycle 3 | 66--85 hrs | 83--104 hrs | Week 7 |
| Active Learning Cycle 4 | 73--94 hrs | 86--109 hrs | Week 8 |
| Demo-ready | 75--96 hrs | 87--110 hrs | Week 9 |
| Phase 3 (optional) | 87--108 hrs | 106--129 hrs | Week 10+ |

### 6.3 Accelerated Schedule (Minimum Viable Training)

If time is tight, here is the fastest path to a fine-tuned demo:

```
Day 1:   Download 30 sermons + Bible corpus prep
Night 1: Preprocess + pseudo-label (7 hrs)
Day 2:   Quick human review (4 hrs -- top-priority corrections only)
Night 2: Whisper LoRA + MarianMT + eval (4.2 hrs)
Night 3: TranslateGemma 4B QLoRA (12 hrs)
Day 4:   Evaluate, transfer to Mac, test

Total: 4 days, ~23 GPU-hrs, ~5 human-hrs
```

This gives a functional fine-tuned system with minimal human correction. Quality will be lower than the full plan but still substantially better than the base models.

---

## 7. Hardware Schedule

### 7.1 A2000 Ada GPU Utilization

The GPU cannot run two training jobs simultaneously -- it must be scheduled sequentially. However, CPU tasks can overlap with GPU tasks.

#### Overnight Run Templates

**Night Type A: Data Pipeline (7 hrs)**
```bash
# Chain preprocessing and pseudo-labeling
nohup bash -c '
  python training/preprocess_audio.py --input stark_data/raw --output stark_data/cleaned --resume &&
  python training/transcribe_church.py --backend faster-whisper --resume
' > pipeline.log 2>&1 &
```

**Night Type B: Quick Training (4 hrs)**
```bash
# Chain Whisper, MarianMT, and evaluation
nohup bash -c '
  python training/train_whisper.py --epochs 3 --dataset stark_data/cleaned &&
  python training/train_marian.py --epochs 5 &&
  python training/evaluate_translation.py --models all
' > train_quick.log 2>&1 &
```

**Night Type C: TranslateGemma 4B (12 hrs)**
```bash
nohup python training/train_gemma.py A --epochs 3 > gemma4b_train.log 2>&1 &
```

**Night Type D: TranslateGemma 12B (26 hrs -- spans 3 nights)**
```bash
# Use tmux for multi-night training
tmux new-session -s gemma12b
python training/train_gemma.py B --epochs 3
# If interrupted: python training/train_gemma.py B --epochs 3 --resume
```

#### GPU Schedule for Phase 1 Round 1

| Night | Job Type | Duration | What Runs |
|-------|----------|----------|-----------|
| Night 1 | Type A | ~7 hrs | Preprocess 30 hrs + pseudo-label |
| Night 2 | Type B | ~4 hrs | Whisper LoRA + MarianMT + eval |
| Night 3 | Type C | ~12 hrs | TranslateGemma 4B QLoRA |
| Day 4 AM | Eval | ~1 hr | Final evaluation, then idle |

### 7.2 CPU Parallelization

While the GPU runs training overnight, the CPU can handle:

- Bible corpus preparation (download, alignment, glossary building)
- Human correction work in Label Studio (runs in browser)
- Data analysis and metric visualization
- Adapter file transfer to Mac (scp/rsync)

### 7.3 Mac Testing Schedule

After each training round, adapters transfer to the Mac for inference testing:

| Step | Time | Command |
|------|------|---------|
| Copy LoRA adapters (WSL -> Mac) | ~1 min | `scp -r fine_tuned_*/ mac:~/Code/vibes/SRTranslate/` |
| Test Whisper with adapter | ~5 min | Run on sample audio, check WER |
| Test TranslateGemma with adapter | ~5 min | Run on sample text, check BLEU |
| Run full A/B dry run | ~15 min | `python dry_run_ab.py` with fine-tuned models |
| **Total per test cycle** | **~30 min** | |

### 7.4 TranslateGemma 12B Decision

The 12B model is expensive to train (~26 hrs, spanning 3 nights) and tight on VRAM (14--16.5 GB on the 16 GB A2000 Ada with 30% OOM risk). The decision to train 12B should be based on Round 1 results:

| 4B Quality vs Base | Decision |
|-------------------|----------|
| BLEU +5 or more, theological terms > 80% | Skip 12B -- 4B is good enough |
| BLEU +2--4, theological terms 60--80% | Train 12B as potential upgrade |
| BLEU < +2, theological terms < 60% | Investigate -- may be a data issue, not model size |

If 12B training is attempted:

| Night | Steps | Duration |
|-------|-------|----------|
| Night 1 | Steps 1--5,500 | ~10 hrs |
| Night 2 | Steps 5,500--11,000 (--resume) | ~10 hrs |
| Night 3 | Steps 11,000--12,939 (--resume) | ~6 hrs |

---

## 8. Storage Requirements

### 8.1 Per-Phase Disk Usage

| Data Type | Per Hour of Raw Audio | 30 hrs | 80 hrs | 250 hrs |
|-----------|----------------------|--------|--------|---------|
| Raw WAV (16kHz mono) | ~57 MB | 1.7 GB | 4.6 GB | 14.3 GB |
| Cleaned + segmented WAV | ~45 MB | 1.4 GB | 3.6 GB | 11.3 GB |
| JSON transcripts | ~0.5 MB | 15 MB | 40 MB | 125 MB |
| **Audio subtotal** | ~102 MB | **3.1 GB** | **8.2 GB** | **25.7 GB** |

| Data Type | Size |
|-----------|------|
| Bible corpus (155K aligned pairs) | ~50 MB |
| Theological glossary | ~1 MB |
| Fine-tuned Whisper LoRA adapter | ~50--100 MB |
| Fine-tuned TranslateGemma 4B LoRA adapter | ~200--400 MB |
| Fine-tuned TranslateGemma 12B LoRA adapter | ~400--800 MB |
| Fine-tuned MarianMT (full model copy) | ~600 MB |
| Training checkpoints (kept temporarily) | ~2--5 GB |
| Metrics and logs | ~50 MB |

### 8.2 Total Disk Budget

| Phase | Cumulative Storage |
|-------|--------------------|
| Phase 1, Round 1 (30 hrs) | ~6 GB |
| Phase 1, Round 2 (80 hrs) | ~14 GB |
| Phase 1, Round 3 (250 hrs) | ~35 GB |
| + Phase 2 (100 hrs Ministry) | ~50 GB |
| + Phase 3 (60 hrs Conference) | ~60 GB |
| **Maximum (all data + all models)** | **~65 GB** |

WSL2 with 64GB RAM has ample disk space. Keep training checkpoints on the Windows partition (`/mnt/c/...`) for faster I/O.

---

## 9. Quality Milestones and Go/No-Go Gates

### 9.1 Baseline Targets (Before Fine-Tuning)

Establish these on base (non-fine-tuned) models before any training:

| Metric | Base Distil-Whisper | Target |
|--------|-------------------|--------|
| WER on church audio | ~12--18% (estimated) | Measure exactly |
| WER on clean speech (LibriSpeech test-clean) | ~3--4% | Sanity check |

| Metric | Base TranslateGemma 4B | Base MarianMT | Target |
|--------|----------------------|---------------|--------|
| SacreBLEU on Bible holdout | ~35--42 (estimated) | ~30--38 (estimated) | Measure exactly |
| chrF++ on Bible holdout | ~55--65 (estimated) | ~50--60 (estimated) | Measure exactly |
| Theological term accuracy | ~40--60% (estimated) | ~30--50% (estimated) | Measure exactly |

### 9.2 Post-Training Quality Gates

| Gate | Metric | Minimum to Proceed | Target |
|------|--------|-------------------|--------|
| Phase 1 R1 | WER improvement | > 10% relative | > 20% relative |
| Phase 1 R1 | BLEU improvement | > +2 points | > +4 points |
| Phase 1 R1 | Theological terms | > 65% | > 80% |
| Phase 1 R2 | WER improvement (cumulative) | > 25% relative | > 35% relative |
| Phase 2 | No regression on Gospel WER | Gospel WER stays same or improves | -- |
| Phase 2 | Ministry WER | > 10% relative improvement | > 20% relative |
| Demo-ready | Absolute WER | < 8% | < 5% |
| Demo-ready | SacreBLEU | > 42 | > 48 |
| Demo-ready | Theological terms | > 75% | > 85% |

### 9.3 When to Stop

| Scenario | Action |
|----------|--------|
| Phase 1 R1 meets all targets | Skip R2/R3, proceed to active learning cycles |
| Phase 1 R1 meets minimum but not targets | Proceed to R2 |
| Phase 1 R1 fails minimum gates | Debug -- check data quality, preprocessing, training config |
| Phase 1 R2 meets targets | Skip R3, proceed to Phase 2 |
| Phase 2 regresses Gospel performance | Revert, increase replay ratio, retrain |
| 3 consecutive cycles < 2% improvement | Stop training, focus on deployment |

---

## 10. Step-by-Step Execution Checklist

### Pre-Flight (Do Once)

- [ ] Verify Windows/WSL2 environment per `CLAUDE-windows.md`
- [ ] Verify Mac environment per `CLAUDE-macbook.md`
- [ ] Run `nvidia-smi` on WSL2 -- confirm A2000 Ada visible with 16 GB
- [ ] Test `python training/train_whisper.py --help` -- confirm all dependencies installed
- [ ] Test `python training/train_gemma.py --help` -- confirm bitsandbytes + QLoRA working
- [ ] Test `python training/train_marian.py --help` -- confirm MarianMT fine-tune working
- [ ] Establish SSH/SCP path between Windows and Mac for adapter transfer
- [ ] Run base model inference on a sample sermon -- establish baseline WER
- [ ] Run base translation on sample text -- establish baseline BLEU

### Phase 1, Round 1

- [ ] **Day 1:** `python download_sermons.py -n 30` (Mac or Windows)
- [ ] **Day 1:** `python training/prepare_bible_corpus.py` (CPU, either machine)
- [ ] **Day 1:** `python build_glossary.py` (CPU, either machine)
- [ ] **Night 1:** Start preprocessing + pseudo-labeling (GPU, ~7 hrs)
- [ ] **Day 2:** Verify preprocessing output: check `stark_data/cleaned/` segment count
- [ ] **Day 2:** Verify transcripts: spot-check 10 random segments for quality
- [ ] **Day 2:** Compute baseline WER on 50 sampled segments (manual transcription)
- [ ] **Days 2--10:** Human correction in Label Studio (~2.5--3 hrs/day)
- [ ] **Night 2:** Start Whisper LoRA + MarianMT training + eval (GPU, ~4.2 hrs)
- [ ] **Day 3:** Check training logs for loss convergence, no NaN/Inf
- [ ] **Night 3:** Start TranslateGemma 4B QLoRA (GPU, ~12 hrs)
- [ ] **Day 4:** Check TranslateGemma training logs
- [ ] **Day 4:** Run full evaluation suite
- [ ] **Day 4:** Transfer adapters to Mac: `scp -r fine_tuned_*/ mac:~/Code/vibes/SRTranslate/`
- [ ] **Day 4:** Test on Mac: `python dry_run_ab.py` with fine-tuned adapters
- [ ] **Day 4:** Record baseline vs. fine-tuned metrics in `metrics/ab_metrics.csv`
- [ ] **Day 4:** **GO/NO-GO DECISION** -- review results, decide on Round 2

### Phase 1, Round 2 (If Approved)

- [ ] `python download_sermons.py -n 80` (downloads 50 new, skips existing 30)
- [ ] Overnight: preprocess + pseudo-label new audio
- [ ] Human correction on new segments (~2--3 weeks)
- [ ] Retrain all models from Phase 1 checkpoints
- [ ] Evaluate on both old and new test segments
- [ ] Transfer to Mac and test

### Subsequent Phases

Follow the same pattern: download -> preprocess -> pseudo-label -> correct -> train -> evaluate -> transfer -> test.

---

## Appendix A: Playlist Reference Table

Complete inventory of all 52 playlists on the Stark Road Gospel Hall YouTube channel, sorted by total hours.

| # | Playlist | Videos | Hours | Avg Min | Category |
|---|----------|--------|-------|---------|----------|
| 1 | Ministry | 325 | 339.0 | 63 | Teaching |
| 2 | Gospel | 275 | 249.8 | 55 | Evangelistic |
| 3 | Throwback Thursdays | 74 | 71.9 | 58 | Historical |
| 4 | Revelation | 29 | 49.0 | 101 | Book study |
| 5 | Acts: The Gospel in Motion | 32 | 43.8 | 82 | Book study |
| 6 | John | 35 | 36.1 | 62 | Book study |
| 7 | Annual Conferences (2022--2023) | 13 | 27.7 | 128 | Conference |
| 8 | Mark: Gospel of Mark | 23 | 21.1 | 55 | Book study |
| 9 | Fall Gospel: Are You Right with God? | 17 | 20.7 | 73 | Gospel series |
| 10 | Jesus Saves! Gospel Series | 16 | 19.4 | 73 | Gospel series |
| 11 | Fall Gospel Series 2025 | 16 | 18.9 | 71 | Gospel series |
| 12 | Gospel Series 2021 | 20 | 17.8 | 53 | Gospel series |
| 13 | Peace? It's Possible! | 15 | 16.3 | 65 | Gospel series |
| 14 | Conference 2024 | 8 | 15.5 | 116 | Conference |
| 15 | Conference 2025 | 8 | 15.1 | 113 | Conference |
| 16 | 1 Timothy | 6 | 15.0 | 150 | Book study |
| 17 | Gospel Series 2020 | 24 | 14.2 | 36 | Gospel series |
| 18 | Genesis | 11 | 12.2 | 67 | Book study |
| 19 | Colossians Bible Study | 10 | 10.6 | 64 | Book study |
| 20 | Bi-monthly Bible Readings (2024-25) | 6 | 10.5 | 105 | Bible reading |
| 21 | Bi-monthly Bible Readings (pre-2024) | 6 | 10.6 | 105 | Bible reading |
| 22 | Questions & Answers Series | 6 | 9.0 | 90 | Interactive |
| 23 | Easter Gospel (Gene Higgins) | 9 | 8.7 | 58 | Gospel series |
| 24 | What's Wrong With This World? | 14 | 7.9 | 34 | Gospel series |
| 25 | 1 Peter | 5 | 7.2 | 87 | Book study |
| 26 | God Saved Me | 6 | 6.9 | 69 | Gospel series |
| 27 | Christ in the Offerings | 6 | 6.9 | 69 | Book study |
| 28 | Sunday School | 5 | 6.6 | 79 | Teaching |
| 29 | Summer Ministry 2020: Sermon on Mount | 7 | 6.1 | 52 | Book study |
| 30 | A Biblical Perspective on Relationships | 4 | 6.0 | 90 | Topical |
| 31 | Bible Series: 1-2 Thessalonians | 9 | 5.8 | 39 | Book study |
| 32 | 1-3 John | 3 | 5.1 | 102 | Book study |
| 33 | Gilliland Ministry 2024 | 4 | 4.8 | 72 | Guest speaker |
| 34 | Coulson Ministry 2024 | 4 | 4.6 | 69 | Guest speaker |
| 35 | James: Faith in Action | 6 | 4.0 | 40 | Book study |
| 36 | Summer 2021: Prophetic Subjects | 4 | 4.0 | 61 | Topical |
| 37 | 2 Peter | 3 | 3.9 | 77 | Book study |
| 38 | Bible Teaching with Mark Sweetnam | 3 | 3.5 | 70 | Guest speaker |
| 39 | Jude | 2 | 3.5 | 104 | Book study |
| 40 | Bi-Monthly Bible Readings 25-26 | 2 | 3.4 | 102 | Bible reading |
| 41 | Galatians Bible Study | 5 | 3.3 | 40 | Book study |
| 42 | Conference 2020 | 2 | 3.2 | 95 | Conference |
| 43 | 1 Corinthians Bible Reading | 3 | 3.0 | 59 | Bible reading |
| 44 | Summer 2021: NT Assembly | 2 | 2.4 | 73 | Topical |
| 45 | VBS 2020: Under the Sea | 5 | 2.4 | 29 | Children |
| 46 | 2 Timothy: Faithful to the End | 3 | 1.9 | 38 | Book study |
| 47 | God's Great Remedies (Children) | 6 | 1.9 | 19 | Children |
| 48 | Titus: Godly Living | 3 | 1.6 | 32 | Book study |
| 49 | Christmas Programs | 2 | 1.6 | 49 | Special |
| 50 | VBS 2021: Amazing Rescues | 6 | 1.6 | 16 | Children |
| 51 | Philemon | 1 | 0.5 | 31 | Book study |

---

## Appendix B: Risk Mitigation

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Human correction takes longer than estimated | 40% | Delays by 1--2 weeks | Start with bottom 10% only; defer rest to later cycles |
| TranslateGemma 12B OOM on A2000 Ada | 30% | Cannot train 12B | Use 4B only (still strong baseline); reduce seq_length |
| Whisper pseudo-labels too noisy for training | 15% | Poor initial WER improvement | Use faster-whisper with strict confidence filtering |
| Bible corpus packing less efficient | 25% | 20--30% more TranslateGemma training time | Adjust schedule by 1 night |
| Fine-tuned Whisper regresses on non-church audio | 10% | Model less general | Increase replay ratio from 30% to 50% |
| LoRA adapters incompatible with MLX on Mac | 15% | Cannot use fine-tuned models on Mac | Test adapter loading early (Day 1 of pre-flight) |
| Audio quality too variable across playlists | 20% | Some data hurts more than helps | Stratify by audio quality in preprocessing; reject noisy files |
| Diminishing returns hit earlier than expected | 30% | Wasted data collection effort | Evaluate after every round; stop early if plateau |

---

## Appendix C: Translation Model Comparison Strategy

Three translation models are trained in parallel on the same Bible corpus. Evaluation determines which to deploy:

| Model | Training Time | Inference Speed (Mac) | Quality Ceiling | Use Case |
|-------|--------------|----------------------|----------------|----------|
| MarianMT (298MB) | ~2.2 hrs | ~30ms/sentence | Lower (statistical MT) | Partials, fast fallback |
| TranslateGemma 4B (4-bit) | ~12 hrs | ~350ms/sentence | High | Final translations |
| TranslateGemma 12B (4-bit) | ~26 hrs | ~800ms/sentence | Highest | Quality-critical passages |

**Decision framework after Round 1:**

1. If MarianMT fine-tuned BLEU > TranslateGemma 4B base BLEU: MarianMT fine-tuning is sufficient for MVP
2. If TranslateGemma 4B fine-tuned BLEU > MarianMT fine-tuned by > 5 points: Use 4B for finals
3. If TranslateGemma 12B fine-tuned BLEU > 4B fine-tuned by > 3 points: Use 12B if latency acceptable
4. Otherwise: Stay with current two-pass architecture (MarianMT partials + TranslateGemma 4B finals)
