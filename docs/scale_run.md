# TranslateGemma QLoRA — Scale-Up Test Matrix

## Context

Phase 1 ablation (6 runs at 8K pairs) identifies the dominant hyperparameters.
Phase 2 B-series (2-3 runs) combines winners. These **scale-up runs** test whether
more training data improves quality once the config is locked.

TranslateGemma already translates EN→ES well. Fine-tuning is domain adaptation
(theological vocabulary, biblical register), not re-learning translation. Research
(ALMA-R, arXiv:2409.03454) shows diminishing returns beyond 10K-30K pairs for
LLM translation FT — ALMA-R matched GPT-4 with only 22K pairs.

### Baselines

| Run | BLEU | chrF++ | COMET | Theo terms | Pairs | Steps |
|-----|------|--------|-------|------------|-------|-------|
| Base model (no FT) | 19.7 | — | — | — | — | — |
| Nopack D0 (forgetting) | 9.7 | 29.5 | 0.559 | 2/8 (25%) | 8K | 1114 |
| A1 (50 steps) | 20.4 | 45.2 | 0.752 | TBD | 8K | 50 |
| Best ablation (TBD) | TBD | TBD | TBD | TBD | 8K | TBD |
| Best B-series (TBD) | TBD | TBD | TBD | TBD | 8K | TBD |

### Available Data

| Dataset | Count | Notes |
|---------|-------|-------|
| Bible verse pairs (train) | 242,091 | Modern-register prioritized by subsampler |
| Glossary pairs | 458 | 229 terms × 2 phrasing variants |
| Test holdout | 27,130 | Stratified by genre (eval samples from this) |

---

## Metrics & Goals

All runs evaluated on the same 500-verse holdout sample for direct comparison to ablation.
Final winner gets full 27K-verse holdout eval + theological spot-check.

| Metric | Floor (no-go below) | Minimum | Target | Stretch |
|--------|---------------------|---------|--------|---------|
| BLEU | 17.7 (-10% of base) | > ablation best | ablation best + 2 | ablation best + 4 |
| chrF++ | 40.0 | > ablation best | > 48.0 | > 52.0 |
| COMET | 0.720 | > ablation best | > 0.770 | > 0.800 |
| Theo terms | 3/8 (37%) | > 5/8 (62%) | > 6/8 (75%) | > 7/8 (87%) |

**Regression test:** Any run scoring below the floor on ANY metric is rejected — it means
more data caused forgetting and the config needs adjustment before scaling further.

---

## Stepping Math

Training step count depends on data size. Effective batch size = 2 × 4 grad_accum = **8**.

| Pairs | Glossary (2x) | Total | After 5% eval split | Steps/epoch |
|-------|---------------|-------|---------------------|-------------|
| 8K | 687 (3x*) | 8,687 | ~8,253 | ~1,032 |
| 20K | 916 (2x) | 20,916 | ~19,870 | **~2,484** |
| 50K | 916 (2x) | 50,916 | ~48,370 | **~6,046** |

*Ablation used 3x (old default). Scale-up uses 2x (new default via `--glossary-oversample`).

Observed training speed: **~4s/step** (nopack, A2000 Ada).

---

## Scale-Up Phase — Test Matrix (4 runs, conditional)

All runs use the **winning config from ablation/B-series** (lr, rank, replay, dropout, neftune — TBD).
The only variable is data size and training duration. Glossary oversampling reduced from 3x → 2x.

**Controls held constant:** packing=False, bf16, cosine scheduler, warmup=0.1, max_grad_norm=0.5, seed=42, all winning ablation hyperparams.

| ID | Variable | Value | Rationale |
|----|----------|-------|-----------|
| S1a | pairs=20K, steps=N | Same steps as ablation winner | More data at same compute — tests if data diversity alone helps |
| S1b | pairs=20K, steps=~2.5N | Proportional to epoch fraction | Same epoch fraction at 20K — tests if more training on more data compounds |
| S2a | pairs=50K, steps=best(S1) | Same steps as S1 winner | More data, same compute — tests diminishing returns |
| S2b | pairs=50K, steps=~6N | Proportional to epoch fraction | Full epoch at 50K — max data + max training |

S2a and S2b are **conditional** — only run if S1 shows BLEU improvement > +2 over ablation best.

### Commands

```bash
# --- S1a: 20K pairs, same steps as ablation winner ---
python training/train_gemma.py A --max-pairs 20000 --glossary-oversample 2 \
    --max-steps ${BEST_STEPS} ${WINNING_ARGS} \
    -o scale_runs/S1a_20k_same_steps

python training/evaluate_translation.py --adapter scale_runs/S1a_20k_same_steps \
    --max-samples 500 --output-file scale_runs/S1a_metrics.json

# --- S1b: 20K pairs, proportional steps ---
python training/train_gemma.py A --max-pairs 20000 --glossary-oversample 2 \
    --max-steps ${SCALED_STEPS_20K} ${WINNING_ARGS} \
    -o scale_runs/S1b_20k_scaled

python training/evaluate_translation.py --adapter scale_runs/S1b_20k_scaled \
    --max-samples 500 --output-file scale_runs/S1b_metrics.json

# --- S2a: 50K pairs, same steps as S1 winner (conditional) ---
python training/train_gemma.py A --max-pairs 50000 --glossary-oversample 2 \
    --max-steps ${S1_BEST_STEPS} ${WINNING_ARGS} \
    -o scale_runs/S2a_50k_same_steps

python training/evaluate_translation.py --adapter scale_runs/S2a_50k_same_steps \
    --max-samples 500 --output-file scale_runs/S2a_metrics.json

# --- S2b: 50K pairs, proportional steps (conditional) ---
python training/train_gemma.py A --max-pairs 50000 --glossary-oversample 2 \
    --max-steps ${SCALED_STEPS_50K} ${WINNING_ARGS} \
    -o scale_runs/S2b_50k_scaled

python training/evaluate_translation.py --adapter scale_runs/S2b_50k_scaled \
    --max-samples 500 --output-file scale_runs/S2b_metrics.json

# --- Final: full holdout eval on winner ---
python training/evaluate_translation.py --adapter scale_runs/${WINNER} \
    --output-file scale_runs/${WINNER}_full_metrics.json
# (no --max-samples → runs all 27,130 test verses)
```

### Replay buffer at 50K

If ablation A6 showed replay helps, increase replay ratio at 50K to counterbalance
archaic pair dilution:

| Data scale | Replay ratio | Replay pairs | Rationale |
|------------|-------------|--------------|-----------|
| 8K | 20% | ~1,600 | Ablation default |
| 20K | 20% | ~4,000 | Same ratio, more absolute pairs |
| 50K | **30%** | ~15,000 | Extra anchoring — archaic pairs now dominate |

---

## Time Estimates

Based on observed ~4s/step training, ~5.4s/verse eval (500 verses ≈ 47 min).

### If ablation winner is ~50 steps (A1 scenario)

| Run | Train steps | Train time | Eval time | Total |
|-----|-------------|-----------|-----------|-------|
| S1a (20K, 50 steps) | 50 | ~3 min | 47 min | ~50 min |
| S1b (20K, ~120 steps) | 120 | ~8 min | 47 min | ~55 min |
| S2a (50K, 50 steps) | 50 | ~3 min | 47 min | ~50 min |
| S2b (50K, ~290 steps) | 290 | ~19 min | 47 min | ~66 min |
| Full holdout (winner) | — | — | ~41 hrs | ~41 hrs |
| **S1 only** | | | | **~1.7 hrs** |
| **All + full holdout** | | | | **~45 hrs** |

### If ablation winner is ~150 steps (A2 scenario)

| Run | Train steps | Train time | Eval time | Total |
|-----|-------------|-----------|-----------|-------|
| S1a (20K, 150 steps) | 150 | ~10 min | 47 min | ~57 min |
| S1b (20K, ~360 steps) | 360 | ~24 min | 47 min | ~71 min |
| S2a (50K, 150 steps) | 150 | ~10 min | 47 min | ~57 min |
| S2b (50K, ~880 steps) | 880 | ~59 min | 47 min | ~106 min |
| Full holdout (winner) | — | — | ~41 hrs | ~41 hrs |
| **S1 only** | | | | **~2.1 hrs** |
| **All + full holdout** | | | | **~46 hrs** |

### If ablation winner is full-epoch (~1114 steps, A3/A4/A5/A6 scenario)

| Run | Train steps | Train time | Eval time | Total |
|-----|-------------|-----------|-----------|-------|
| S1a (20K, 1114 steps) | 1114 | ~74 min | 47 min | ~121 min |
| S1b (20K, ~2484 steps) | 2484 | ~166 min | 47 min | ~213 min |
| S2a (50K, 1114 steps) | 1114 | ~74 min | 47 min | ~121 min |
| S2b (50K, ~6046 steps) | 6046 | ~403 min | 47 min | ~450 min |
| Full holdout (winner) | — | — | ~41 hrs | ~41 hrs |
| **S1 only** | | | | **~5.6 hrs** |
| **All + full holdout** | | | | **~56 hrs** |

**Note:** Full holdout eval (27,130 verses at 5.4s/verse) is ~41 hours. Consider reducing
to 3,100 verses (10% stratified sample) for ~4.7 hrs, or keeping 500 verses if genre
breakdown at 500 is sufficient. The 500-verse eval has been stable across ablation runs.

---

## Code Changes Required

| Change | File | Effort |
|--------|------|--------|
| ~~Add `--glossary-oversample` CLI flag~~ | `training/train_gemma.py` | **DONE** — default 2, CLI-configurable |
| ~~Create `training/run_scale.sh`~~ | `training/run_scale.sh` | **DONE** — template with TBD placeholders |
| Fill in `${WINNING_ARGS}` placeholders | `training/run_scale.sh` | After B-series results are in |

---

## Decision Logic

### S1a → S1b gate

| S1a Result vs Ablation Best | Interpretation | Next |
|-----------------------------|---------------|------|
| BLEU > ablation + 2 | Data helps at same compute | Run S1b to test if more steps also help |
| BLEU within ±2 | Data diversity alone doesn't help | Run S1b anyway — maybe needs more steps |
| BLEU < ablation - 2 | More data hurts (register dilution?) | Check archaic/modern ratio, investigate |

### S1 → S2 gate

| Best S1 vs Ablation Best | Interpretation | Next |
|--------------------------|---------------|------|
| BLEU > ablation + 2 | Data scaling works | Run S2a/S2b at 50K |
| BLEU within ±2 | Saturated at 8K-20K | Skip S2, lock best config |
| Theo terms improved even if BLEU flat | Vocab coverage growing | Run S2a (more terms to learn) |

### S2 → Final gate

| Best S2 vs Best S1 | Interpretation | Next |
|---------------------|---------------|------|
| BLEU > S1 + 2 | Still scaling (unlikely) | Consider 100K, but probably stop |
| BLEU within ±2 | Saturated | Lock S1 config (simpler, faster) |
| BLEU < S1 | Archaic dilution confirmed | Revert to S1, increase replay if retrying |

### Final acceptance gate

| Metric | Threshold | Action if below |
|--------|-----------|-----------------|
| BLEU | > 17.7 | Adapter is worse than base — reject, investigate |
| BLEU | > 19.7 | No improvement over base — skip adapter deployment |
| COMET | > 0.720 | Semantic quality degraded — reject |
| Theo terms | > 5/8 | Theological vocabulary not learned — add glossary data or increase oversample |

---

## Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Archaic register dilution at 50K | Medium | Model produces formal/archaic Spanish | Prioritize modern translations in subsampler (already implemented), increase replay |
| Full holdout eval too slow (~41 hrs) | High | Blocks next iteration | Use 3,100 stratified sample (~4.7 hrs) or stay at 500 verses |
| Glossary underweight at 20K+ (2.2%) | Low | Theological terms regress | Monitor spot-check, increase oversample back to 3x if needed |
| Winning config doesn't transfer to larger data | Medium | Ablation hyperparams overfit to 8K regime | S1a tests this directly — if it fails, re-tune at 20K |
| GPU contention with Whisper pipeline | Low | OOM or slowdown | Run scale-up on dedicated GPU time (overnight) |

---

## Decision Tree Summary

```
Ablation + B-series complete → lock winning config
  │
  ├── Add --glossary-oversample flag (reduce 3x → 2x)
  ├── Create run_scale.sh
  │
  ├── S1a: 20K, same steps
  │     ├── BLEU > ablation + 2 → data helps
  │     └── BLEU flat → still run S1b (different test)
  │
  ├── S1b: 20K, proportional steps
  │     ├── S1b > S1a → more training on more data compounds
  │     └── S1b ≤ S1a → diminishing returns from more steps
  │
  ├── Gate: best S1 vs ablation
  │     ├── Improved > +2 → proceed to S2
  │     └── Flat/worse → STOP, lock ablation config
  │
  ├── S2a: 50K, same steps as S1 winner (conditional)
  ├── S2b: 50K, proportional steps (conditional)
  │     ├── S2 > S1 + 2 → consider 100K (unlikely to be needed)
  │     └── S2 ≤ S1 → lock S1 config
  │
  └── Winner: full holdout eval (3,100 verses) + theological spot-check
        ├── Passes acceptance gate → export adapter, transfer to Mac
        └── Fails → back to Phase 2 B-series or investigate data quality
```
