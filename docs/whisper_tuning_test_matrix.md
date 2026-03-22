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

## Estimated Runtime

| Step | Time | VRAM |
|------|------|------|
| W0: Baseline WER | ~30 min | ~5 GB |
| W1-W6: Ablation (6 × ~1.5 hrs) | ~9 hrs | ~8-10 GB |
| Analysis + pick winner | ~15 min | — |
| W7-W9: Scale (3 × ~3-5 hrs) | ~12 hrs | ~8-10 GB |
| Final evaluation | ~30 min | ~5 GB |
| **Total** | **~22 hrs** | |

---

## Data Requirements

| Item | Path | Status |
|------|------|--------|
| Deepgram oracle transcripts | `stark_data/deepgram_transcripts/*.deepgram.json` | Phase 1A (running) |
| Aligned chunks dataset | `stark_data/whisper_dataset_deepgram/` | Needs `align_deepgram_chunks.py` |
| 5K subset (for W6) | `stark_data/whisper_dataset_deepgram_5k/` | Subsample of above |
| Tier 1 boost terms | `bible_data/glossary/tier1_boost.json` | Done (50 terms) |
| LibriSpeech test-clean | Auto-downloaded by train_whisper.py | Available via HuggingFace |
