# Whisper v3-turbo LoRA — Test Matrix

## Context

Whisper large-v3-turbo is our STT backbone. Base model works well on general English but struggles with theological vocabulary (propitiation, sanctification, etc.) and speaker-specific patterns at Stark Road. We're fine-tuning with LoRA using Deepgram Nova-3 oracle transcriptions as ground truth labels.

**Key lesson from TranslateGemma ablation**: Learning rate was the dominant variable. At lr=1e-5 (Gemma), the model was safe for full epochs. At lr=1e-5 (the standard Gemma lr), catastrophic forgetting hit by step 150. Whisper LoRA literature recommends lr=1e-4, but we should verify this on our domain.

**Training data**: ~24,595 chunks from 34 sermons (before 3/14/26 cutoff), with Deepgram-aligned ground truth labels. Eval: 1+ sermons from after cutoff.

---

## Phase W0 — Baseline (1 run, ~30 min)

Evaluate base Whisper v3-turbo on our eval set to establish WER floor.

```bash
python training/assess_quality.py cross-check \
    --input stark_data/deepgram_transcripts \
    --model openai/whisper-large-v3-turbo \
    --n 200
```

**Expected baseline WER**: ~8-15% on church sermon audio (higher on theological terms).

---

## Phase W1 — Ablation: Find Dominant Variable (6 runs, ~9-12 hrs)

Each run uses full dataset (~24K chunks), 1 epoch. Change ONE variable per run.

| ID | Variable | Value | Other defaults | Rationale |
|----|----------|-------|----------------|-----------|
| W1 | lr=1e-4, 1 epoch | Baseline LoRA config | r=32, qv, replay=0.3 | CLAUDE.md recommended config |
| W2 | lr=5e-5, 1 epoch | Half lr | r=32, qv, replay=0.3 | Is 1e-4 too aggressive? |
| W3 | lr=1e-5, 1 epoch | Ultra-conservative lr | r=32, qv, replay=0.3 | TranslateGemma's winning lr zone |
| W4 | target=qkvo, 1 epoch | Expanded targets | lr=1e-4, r=32, replay=0.3 | More adaptation capacity |
| W5 | replay=0, 1 epoch | No anti-forgetting | lr=1e-4, r=32, qv | Is replay necessary or overhead? |
| W6 | data=5K chunks | Reduced data | lr=1e-4, r=32, qv, replay=0.3 | Diminishing returns check |

**Controls held constant:**
- model: `openai/whisper-large-v3-turbo`
- batch_size: 4, grad_accum: 4 (effective 16)
- precision: BF16 LoRA on 8-bit quantized base
- gradient_checkpointing: enabled
- lora_dropout: 0.05
- alpha: 2 × rank (64 for r=32)
- seed: 42
- VRAM: ~8-10 GB

### Commands

```bash
DS="stark_data/whisper_dataset_deepgram"
DS_5K="stark_data/whisper_dataset_deepgram_5k"  # subset

# W1: Baseline LoRA
python training/train_whisper.py --dataset $DS --lr 1e-4 --epochs 1 \
    --target-modules q_proj v_proj --lora-r 32 --lora-alpha 64 \
    --replay-ratio 0.3 -o whisper_ablation/W1_baseline

# W2: Half lr
python training/train_whisper.py --dataset $DS --lr 5e-5 --epochs 1 \
    --target-modules q_proj v_proj --lora-r 32 --lora-alpha 64 \
    --replay-ratio 0.3 -o whisper_ablation/W2_lr5e5

# W3: Conservative lr
python training/train_whisper.py --dataset $DS --lr 1e-5 --epochs 1 \
    --target-modules q_proj v_proj --lora-r 32 --lora-alpha 64 \
    --replay-ratio 0.3 -o whisper_ablation/W3_lr1e5

# W4: Expanded targets (q, k, v, out)
python training/train_whisper.py --dataset $DS --lr 1e-4 --epochs 1 \
    --target-modules q_proj k_proj v_proj out_proj --lora-r 32 --lora-alpha 64 \
    --replay-ratio 0.3 -o whisper_ablation/W4_qkvo

# W5: No replay buffer
python training/train_whisper.py --dataset $DS --lr 1e-4 --epochs 1 \
    --target-modules q_proj v_proj --lora-r 32 --lora-alpha 64 \
    --replay-ratio 0 -o whisper_ablation/W5_noreplay

# W6: Small data (5K chunks)
python training/train_whisper.py --dataset $DS_5K --lr 1e-4 --epochs 1 \
    --target-modules q_proj v_proj --lora-r 32 --lora-alpha 64 \
    --replay-ratio 0.3 -o whisper_ablation/W6_5k
```

### Decision Logic

| Result | Interpretation | Next step |
|--------|---------------|-----------|
| W1 best | Default config works | Scale to 3 epochs (Phase W2) |
| W2 or W3 best (lr-limited) | Same dynamic as TranslateGemma — lr dominates | Use winning lr, test epoch scaling |
| W4 best (expanded targets) | Acoustic model benefits from wider adaptation | Combine with lr winner |
| W5 best (no replay) | Domain close enough to general English | Drop replay, save training time |
| W6 ≈ W1 (5K matches full) | Data saturation — diminishing returns | Focus on quality over quantity |
| All worse than baseline | Ground truth issue or config problem | Inspect Deepgram alignment quality |

---

## Phase W2 — Scale Winner (3 runs, ~9-15 hrs)

Take the Phase W1 winner and test epoch scaling + rank.

| ID | Variable | Value | Rationale |
|----|----------|-------|-----------|
| W7 | Winner + 3 epochs | Multi-pass | Standard training duration |
| W8 | Winner + 5 epochs | Extended | Overfitting check |
| W9 | Winner + r=64, alpha=128 | Higher rank | More capacity if W1 was limited |

### Commands

```bash
WINNER_LR="TBD"         # fill from W1 results
WINNER_TARGETS="TBD"    # fill from W1 results
WINNER_REPLAY="TBD"     # fill from W1 results
WINNER_RANK=32           # default unless W9

# W7: 3 epochs
python training/train_whisper.py --dataset $DS --lr $WINNER_LR --epochs 3 \
    --target-modules $WINNER_TARGETS --lora-r $WINNER_RANK \
    --replay-ratio $WINNER_REPLAY -o whisper_ablation/W7_3epochs

# W8: 5 epochs
python training/train_whisper.py --dataset $DS --lr $WINNER_LR --epochs 5 \
    --target-modules $WINNER_TARGETS --lora-r $WINNER_RANK \
    --replay-ratio $WINNER_REPLAY -o whisper_ablation/W8_5epochs

# W9: rank=64
python training/train_whisper.py --dataset $DS --lr $WINNER_LR --epochs 3 \
    --target-modules $WINNER_TARGETS --lora-r 64 --lora-alpha 128 \
    --replay-ratio $WINNER_REPLAY -o whisper_ablation/W9_rank64
```

### Decision Logic

| Result | Interpretation | Next step |
|--------|---------------|-----------|
| W7 > W8 (3 > 5 epochs) | 5 epochs overfits | Ship W7 |
| W8 > W7 (5 > 3 epochs) | Still learning | Try 7 epochs or add data |
| W9 > W7 (r=64 helps) | Rank 32 was limiting | Ship W9 |
| All plateau | Adaptation saturated | Ship best, start active learning cycle 2 |

---

## Evaluation Metrics — Go/No-Go Gates

| Metric | Floor | Minimum | Target | Kill | How measured |
|--------|-------|---------|--------|------|-------------|
| Overall WER | < base | > 10% rel. reduction | > 20% rel. reduction | > 5% abs. regression | jiwer.wer() on eval set |
| Theological term WER | — | > 30% rel. reduction | > 50% rel. reduction | — | WER on Tier 1 term chunks |
| Accent fairness gap | < 15% abs. | < 10% abs. | < 5% abs. | > 20% abs. | max - min per-accent WER |
| General English WER | no regression | < 2% abs. increase | no increase | > 5% abs. increase | LibriSpeech test-clean |
| Latency | — | < 2x base | < 1.2x base | > 3x base | Wall clock per chunk |

### Theological Term Evaluation

Filter eval chunks to those containing any of the 50 Tier 1 boost terms (from `bible_data/glossary/tier1_boost.json`). Compute WER separately on this subset. **This is the primary signal** — general WER improvement is nice but theological accuracy is the mission.

Terms of particular interest (historically failing):
- propitiation, sanctification, justification
- substitutionary atonement, breaking of bread
- Pharisee, Sadducee, presbyter

### Kill Switch

If general English WER regresses > 5% absolute, **KILL** the adapter immediately. LoRA preserves base weights so disabling the adapter is instant revert — zero risk.

---

## Evaluation Commands

```bash
# Built-in per-epoch WER (runs automatically during training)

# Standalone adapter evaluation
python training/assess_quality.py evaluate \
    --adapter whisper_ablation/W1_baseline \
    --eval-set stark_data/whisper_dataset_deepgram/eval

# 3-way comparison: Whisper base vs adapter vs Deepgram oracle
python training/assess_quality.py cross-check \
    --input stark_data/deepgram_transcripts \
    --adapter whisper_ablation/W1_baseline \
    --n 200

# General English regression check
python training/assess_quality.py evaluate \
    --adapter whisper_ablation/W1_baseline \
    --eval-set librispeech_test_clean

# Theological term WER (filter to Tier 1 chunks)
python training/assess_quality.py evaluate \
    --adapter whisper_ablation/W1_baseline \
    --eval-set stark_data/whisper_dataset_deepgram/eval \
    --filter-terms bible_data/glossary/tier1_boost.json
```

---

## Phase W1/W2 Results (2026-03-25)

All WER normalized (lowercase, cleaned). Eval: 251 chunks from 1 held-out sermon (3/15/26).

| Run | Config | WER(norm) | WER(raw) | CER(norm) | vs Base |
|-----|--------|-----------|----------|-----------|---------|
| **Base** | No fine-tuning | **20.78%** | 24.18% | 22.40% | — |
| **W7** | **3 epochs, r=32, qv, replay=0.3** | **5.63%** | 8.01% | 3.88% | **-72.9%** |
| W1 | 1 epoch, r=32, qv, replay=0.3 | 5.83% | 8.46% | 4.66% | -71.9% |
| W5 | 1 epoch, no replay | 6.44% | 9.12% | 4.66% | -69.0% |
| W4 | 1 epoch, qkvo targets | 6.69% | 9.68% | 4.87% | -67.8% |
| W2 | 1 epoch, lr=5e-5 | 7.55% | 10.24% | 5.56% | -63.7% |
| W6 | 1 epoch, 5K data only | 7.55% | 10.14% | 5.72% | -63.7% |
| W3 | 1 epoch, lr=1e-5 | 8.01% | 10.59% | 6.08% | -61.5% |

### Key Findings

1. **lr=1e-4 is optimal** — lower lr (W2, W3) consistently worse
2. **q_proj+v_proj sufficient** — expanding to qkvo (W4) didn't help
3. **Replay helps marginally** — W5 (no replay) close to W1 (6.44 vs 5.83)
4. **More data helps** — W6 (5K) worse than W1 (17K)
5. **More epochs helps slightly** — W7 (3 epochs) edges W1 (5.63 vs 5.83)
6. **73% WER reduction** — 20.78% → 5.63% on held-out sermon audio

---

## Phase W3 — Scaling Experiments (W8-W12)

### W8 Result (2026-03-27)

| Run | Config | WER(norm) | Loss | vs W7 |
|-----|--------|-----------|------|-------|
| **W7** | 3 epochs, r=32 | **5.63%** | 0.12 | — |
| **W8** | **5 epochs**, r=32 | **7.96%** | 0.10 | **+41% worse** |

**W8 overfits.** Lower training loss (0.10) but higher eval WER (7.96%). 5 epochs memorizes the 17K training chunks without improving generalization. **3 epochs is the sweet spot for this dataset size.**

### W9 (running)

| Run | Config | Steps | ETA | What it answers |
|-----|--------|-------|-----|-----------------|
| **W9** | W7 + **r=64, alpha=128** | 3,555 | ~Sat Mar 28 AM | Is LoRA capacity the bottleneck? 13M params (2× W7). |

### Key Insight: Data is the Bottleneck

The overfitting curve proves the model has extracted everything from 17K chunks:
- 1 epoch (W1): 5.83% — most learning happens here
- 3 epochs (W7): 5.63% — marginal gain, diminishing returns
- 5 epochs (W8): 7.96% — overfitting, worse than 1 epoch

**More epochs won't help. More data will.** The scaling trend (5K→17K: 7.55%→5.63%) shows data isn't saturated — we just ran out of transcribed sermons.

---

## Phase W4 — Data Scaling & Augmentation

### Tier 1: More Training Data (highest impact, ~$93)

332 downloaded YouTube sermons in `stt-data/` are untranscribed. Transcribing with Deepgram + aligning gives ~100K-200K chunks — 6-12x current training data.

| Step | Tool | Time | Cost | GPU? |
|------|------|------|------|------|
| Deepgram transcribe 297 new sermons | `transcribe_with_deepgram.py` | ~6 hrs | ~$93 | No (API) |
| Align chunks | `align_deepgram_chunks.py` | ~5 hrs | Free | Minimal |
| Whisper transcribe (for chunk boundaries) | `transcribe_sermons.sh` | ~12 hrs | Free | ~5 GB |
| Train W12 (W7 config, 1 epoch, ~100K chunks) | `train_whisper.py` | ~45 hrs | Free | ~9.5 GB |

**Expected outcome**: WER drops to 2-3% based on scaling trend. With 6x more speakers and topics, overfitting risk is minimal even at 3 epochs.

**Data quality**: Deepgram Nova-3 with 50 theological keyterms — same oracle quality as current training data.

### Tier 2: Data Augmentation (free, proven)

Add to training pipeline without new data. Implement in the data collator:

**SpecAugment** — mask random frequency bands + time spans in mel spectrograms:
- Standard ASR augmentation (Park et al., 2019)
- Typically 5-10% relative WER reduction
- Zero cost, applied on-the-fly during training

**Speed perturbation** — randomly adjust playback 0.9x-1.1x:
- Creates synthetic speaker variation from same audio
- Well-studied, ~5% relative improvement
- Applied during preprocessing

Combined with current 17K data: could push W7's 5.63% to ~4.5-5.0% without any new recordings.

### Tier 3: Curriculum Learning / Hard Example Mining

After W7 training:
1. Run W7 inference on ALL training chunks
2. Find chunks where W7 still gets >20% WER (the hard examples)
3. Extract the hard subset (~2-3K chunks expected)
4. Fine-tune W7 for 1 additional epoch on hard examples only
5. This focuses adaptation on specific weaknesses (theological terms, unusual accents)

**Expected**: 0.5-1% absolute WER reduction on the hardest segments.

### Tier 4: Evaluation Robustness

Before claiming further improvements, strengthen eval:

| Improvement | Why | How |
|-------------|-----|-----|
| More eval data | 251 chunks from 1 sermon is statistically weak | Add 3/22/26 ministry sermon (already Deepgram-transcribed) |
| Theological term WER | Domain-specific accuracy matters most | Filter eval to Tier 1 term chunks, report separately |
| Cross-speaker eval | Current eval is 1 speaker | Hold out sermons from different speakers |

### Tier 5: Model Architecture

| Option | Tradeoff | When |
|--------|----------|------|
| Whisper large-v3 (32 decoder layers) | Higher ceiling but 8x slower inference | Only if turbo plateaus below target |
| Decoder-only targets (W10) | Tests vocabulary vs acoustic adaptation | After W9 results |
| No replay + more church data (W11) | Replace LibriSpeech with domain data | After data scaling |

### Decision Tree

```
W9 result?
├── W9 >> W7 → r=64 helps, use r=64 for all future runs
├── W9 ≈ W7 → r=32 is sufficient, ship W7 config
│
Then: scale with data
├── Deepgram transcribe 332 sermons ($93 API)
├── Align + train W12 (W7 config, 100K+ chunks, 1 epoch)
├── Expected: 2-3% WER
│
Optional: augmentation
├── Add SpecAugment + speed perturbation to collator
├── Retrain with augmentation → W13
├── Expected: additional 5-10% relative improvement
```

---

## Estimated Runtime

| Step | Time | VRAM |
|------|------|------|
| W1-W6: Ablation (6 runs) | ~30 hrs actual | ~9.5 GB |
| W7: 3 epochs | ~25 hrs | ~9.5 GB |
| W8: 5 epochs | ~25 hrs | ~9.5 GB |
| W9: r=64, 3 epochs | ~17 hrs | ~10 GB |
| WER eval per run | ~2 min | ~3 GB (fp16) |
| **Total (W1-W9)** | **~97 hrs** | |

---

## Data Requirements

| Item | Path | Status |
|------|------|--------|
| Deepgram oracle transcripts (35 sermons) | `stark_data/deepgram_transcripts/*.deepgram.json` | **Done** |
| Aligned chunks dataset (17K train, 251 eval) | `stark_data/whisper_dataset_deepgram/.preprocessed_cache/` | **Done** |
| 5K subset (for W6) | `stark_data/whisper_dataset_deepgram_5k/.preprocessed_cache/` | **Done** |
| Tier 1 boost terms | `bible_data/glossary/tier1_boost.json` | **Done** (50 terms) |
| LibriSpeech replay | Auto-downloaded by train_whisper.py | Available via HuggingFace |
| Merged W1 model (for live test) | `models/whisper-turbo-w1-merged/` | **Done** (1.5 GB) |
| 332 YouTube sermons (for W12) | `stt-data/` sorted by type/year | **Downloaded**, not yet transcribed |
