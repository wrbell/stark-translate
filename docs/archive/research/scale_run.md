# TranslateGemma QLoRA — Scale-Up: Synthetic Sermon Data Strategy

## Context

Phase 1 ablation (6 runs at 8K pairs) and Phase 2 B-series (5 runs) are complete.
**Plain verse pairs are a weak training signal** for an already-instruction-tuned model
like TranslateGemma (see `docs/more_data.md`). They cause overfitting to Bible phrasing
(BLEU up) without improving semantic quality (COMET flat/down).

The new path: **12B-distilled sermon data** — translate real sermon content with 12B,
filter with COMET, and use as training data for 4B. This is knowledge distillation
that directly teaches 4B to produce 12B-like outputs → higher speculative draft
acceptance rate → lower latency.

> **STATUS (2026-03-21): Planned (after Phase 2.5).** Phase 2.5 sermon smoke test must
> show COMET proximity gain > 0.005 on at least one adapter (A1 or B4) before proceeding.

### Baselines (complete)

| Run | COMET | BLEU | chrF++ | Theo terms | Pairs | Steps |
|-----|-------|------|--------|------------|-------|-------|
| Base model (no FT) | **0.7516** | 19.7 | — | — | — | — |
| A1 (COMET-optimal) | **0.752** | 20.4 | 45.2 | 5/8 | 8K | 50 |
| A4 (lr=1e-6, full) | 0.740 | 20.7 | 44.9 | 5/8 | 8K | ~1114 |
| B4 (BLEU-optimal) | 0.742 | **21.2** | **45.3** | **5/8** | 8K | ~1114 |

**No fine-tuned model meaningfully improves COMET over base (0.7516).** Fine-tuning
learns Bible n-gram patterns that boost BLEU but degrade semantic quality.

### Available Data

| Dataset | Count | Notes |
|---------|-------|-------|
| Bible verse pairs (train) | 242,091 | Modern-register prioritized by subsampler |
| Glossary pairs | 507 | 229 terms × sentence + bare pairs |
| Sermon WAVs | ~35 | `stark_data/raw/midwest/` (~35 hrs audio) |
| Test holdout | 27,130 | Stratified by genre |

---

## Metrics & Goals (COMET-Primary)

All runs evaluated on the same 500-verse holdout sample + Phase 2.5 sermon smoke test.

| Metric | Floor (no-go below) | Minimum | Target |
|--------|---------------------|---------|--------|
| COMET | > 0.740 | > 0.752 (beat A1) | > 0.770 |
| Glossary regressions | < 10 lost | < 5 lost | 0 lost |
| chrF++ | > 40.0 | > 45.0 | > 48.0 |
| BLEU | > 17.7 (sanity) | — | — |

---

## Pre-Gate: Phase 2.5

Phase 2.5 runs both A1 and B4 through the sermon smoke test using cached 12B translations.
COMET proximity to 12B on clean sermon chunks is the primary signal.

| Outcome | Meaning | Action |
|---------|---------|--------|
| COMET proximity gain > 0.005 | Adapter helps on sermons | Lock winner → proceed to synthetic data runs |
| COMET gain ±0.005 | Noise range | 12B cache ready → go straight to synthetic data |
| COMET proximity < -0.01 | Adapter hurts sermons | KILL adapter, deploy base, still try synthetic data |

---

## Synthetic Data Generation Pipeline

### Step 1: Transcribe sermon audio

```bash
# Transcribe 10-15 sermons with Whisper large-v3
for wav in stark_data/raw/midwest/*.wav; do
    python -c "
import whisper, json, os
model = whisper.load_model('large-v3')
result = model.transcribe('$wav', language='en', word_timestamps=True)
name = os.path.splitext(os.path.basename('$wav'))[0]
chunks = [{'en': s['text'].strip(), 'start': s['start'], 'end': s['end'],
           'source': name} for s in result['segments'] if len(s['text'].split()) > 4]
with open(f'ablation/sermon_chunks_{name}.json', 'w') as f:
    json.dump(chunks, f, indent=2)
print(f'{name}: {len(chunks)} chunks')
"
done
```

### Step 2: Translate with 12B (distillation source)

```bash
# Translate all chunks with TranslateGemma 12B (~2.1s/chunk)
python training/generate_12b_cache.py \
    --chunks ablation/sermon_chunks_*.json \
    --output ablation/sermon_12b_translations.json
```

12B 4-bit (~7GB) fits on A2000 Ada 16GB. Cost: ~2.1s/chunk × 1000 chunks ≈ 35 min.

### Step 3: COMET filter

```bash
# Score each EN→ES pair with COMET, keep > 0.75
python training/filter_by_comet.py \
    --input ablation/sermon_12b_translations.json \
    --threshold 0.75 \
    --output ablation/sermon_distilled_pairs.jsonl
```

Expected yield: ~500-1000 high-quality sermon EN→ES pairs from ~10 hrs of audio.
These are in the right register (spoken, informal, theological-in-context) AND
represent 12B's translation preferences (the distillation target).

---

## Scale-Up Phase — Synthetic Data Test Matrix (3 runs)

All runs use the Phase 2.5 winning config. The variable is data mix ratio.
Glossary oversampling at 2x.

| ID | Data Mix | Verse Pairs | Sermon Pairs | Glossary | Rationale |
|----|----------|-------------|--------------|----------|-----------|
| S1 | verse 65% + sermon 30% + glossary 5% | ~8K | ~500 | 2x | Conservative mix, test if sermon data improves COMET |
| S2 | verse 50% + sermon 45% + glossary 5% | ~4K | ~1K | 2x | Higher sermon fraction |
| S3 | sermon only + glossary | 0 | ~1K | 2x | Does removing verse pairs help? |

Each run uses `--sermon-data` flag in train_gemma.py (already implemented).

### Commands

```bash
# --- Config: Phase 2.5 determines winner ---
# IF A1 wins (COMET-optimal):
# WINNING_LR="1e-5"; WINNING_STEPS="50"; WINNING_EXTRAS=""
# IF B4 wins (BLEU-optimal):
WINNING_LR="1e-6"
WINNING_STEPS="1114"
WINNING_EXTRAS="--neftune 5"

# --- S1: verse 65% + sermon 30% + glossary 5% ---
python training/train_gemma.py A --max-pairs 8000 --glossary-oversample 2 \
    --lr $WINNING_LR --max-steps $WINNING_STEPS $WINNING_EXTRAS \
    --sermon-data ablation/sermon_distilled_pairs.jsonl \
    -o scale_runs/S1_verse65_sermon30

python training/evaluate_translation.py --adapter scale_runs/S1_verse65_sermon30 \
    --max-samples 500 --output-file scale_runs/S1_metrics.json \
    --compare-base

# --- S2: verse 50% + sermon 45% + glossary 5% ---
python training/train_gemma.py A --max-pairs 4000 --glossary-oversample 2 \
    --lr $WINNING_LR --max-steps $WINNING_STEPS $WINNING_EXTRAS \
    --sermon-data ablation/sermon_distilled_pairs.jsonl \
    -o scale_runs/S2_verse50_sermon45

python training/evaluate_translation.py --adapter scale_runs/S2_verse50_sermon45 \
    --max-samples 500 --output-file scale_runs/S2_metrics.json \
    --compare-base

# --- S3: sermon only + glossary ---
python training/train_gemma.py A --max-pairs 0 --glossary-oversample 2 \
    --lr $WINNING_LR --max-steps $WINNING_STEPS $WINNING_EXTRAS \
    --sermon-data ablation/sermon_distilled_pairs.jsonl \
    -o scale_runs/S3_sermon_only

python training/evaluate_translation.py --adapter scale_runs/S3_sermon_only \
    --max-samples 500 --output-file scale_runs/S3_metrics.json \
    --compare-base

# --- Re-run sermon smoke test on best ---
python training/evaluate_sermon.py \
    --chunks ablation/sermon_test_chunks.json \
    --adapter scale_runs/${BEST} \
    --ceiling-cache ablation/sermon_12b_translations.json \
    --output scale_runs/${BEST}_sermon_eval.json
```

---

## Time Estimates

Based on observed ~4s/step training, ~5.4s/verse eval.

| Task | Time |
|------|------|
| Whisper transcription (~10 sermons) | ~2-3 hrs |
| 12B translation (~1000 chunks) | ~35 min |
| COMET filtering | ~15 min |
| S1 training + eval | ~2-3 hrs |
| S2 training + eval | ~2-3 hrs |
| S3 training + eval | ~1-2 hrs |
| Sermon smoke test (winner) | ~20 min |
| **Total** | **~8-12 hrs** |

---

## Decision Logic

### S1 → S2 gate

| S1 Result vs Ablation Best | Interpretation | Next |
|-----------------------------|---------------|------|
| COMET > 0.752 | Sermon data helps | Run S2 and S3 to find optimal mix |
| COMET 0.740-0.752 | Marginal | Run S2 (higher sermon fraction might help) |
| COMET < 0.740 | Sermon data at this mix hurts | Skip S2, try S3 (sermon-only, no verse dilution) |

### Final acceptance gate

| Metric | Threshold | Action if below |
|--------|-----------|-----------------|
| COMET | > 0.752 | No improvement over A1 — investigate data quality |
| COMET | > 0.740 | Semantic quality degraded — reject |
| Glossary regressions | < 5 | Too many terms lost — increase glossary oversample |
| BLEU | > 17.7 | Sanity check — model broken if below |

---

## Decision Tree Summary

```
Phase 2.5: Sermon smoke test (A1 + B4, ~40 min GPU)
  │
  ├── Lock winner config
  │
  ├── Generate 12B translation cache (already done in Phase 2.5 Step 0)
  │
  ├── COMET-filter cached translations → distilled pairs
  │     Filter: keep pairs with COMET > 0.75
  │     Output: ~500-1000 sermon EN→ES pairs (12B quality, sermon register)
  │
  ├── S1: verse 65% + sermon 30% + glossary 5%
  │     ├── COMET > 0.752 → proceed to S2/S3
  │     └── COMET < 0.740 → skip S2, try S3
  │
  ├── S2: verse 50% + sermon 45% + glossary 5%
  ├── S3: sermon only + glossary (no verse pairs)
  │
  ├── Winner: sermon smoke test + full glossary eval
  │     ├── COMET > 0.752, regressions < 5 → export adapter
  │     └── Still no gain → deploy base model, consider CPT (more_data.md #1)
  │
  └── Phase 4 (future): CPT → Light SFT
        Expected +0.08-0.15 COMET from continued pretraining
```

---

## Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Sermon audio quality varies | Medium | Noisy transcriptions → bad training pairs | COMET filter removes low-quality pairs |
| 12B translations inconsistent | Low | Distillation teaches bad habits | COMET > 0.75 threshold gates quality |
| Sermon-only training loses Bible knowledge | Medium | COMET on Bible holdout drops | S1 tests conservative mix; glossary preserved |
| Winning config doesn't transfer to new data | Medium | Need to re-tune hyperparameters | S1 tests this directly |
| GPU time for 12B translation | Low | ~35 min is manageable | Already cached from Phase 2.5 Step 0 |
