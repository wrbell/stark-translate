# Phase 3: Hybrid 12B + DeepL Pro Synthetic Sermon Data Pipeline

## Rationale

Phase 2.5 established that pure Bible verse pair fine-tuning is a weak training signal for sermon-domain translation:

- **A1** (COMET-optimal): COMET proximity to 12B = -0.0038 (WARN) — mild regression
- **B4** (BLEU-optimal): COMET proximity to 12B = -0.0244 (KILL) — clearly harmful

The new approach: **hybrid distillation** from real sermon audio using two commercial-grade translation sources:
- **70% TranslateGemma-12B** — our quality ceiling model, produces contextual theological translations
- **30% DeepL Pro** — commercial API with strict glossary enforcement for the 3 failing theological terms (Santiago, Jacobo, propiciación)

DeepL adds fluency diversity and glossary enforcement that the 12B model alone cannot guarantee.

## Architecture

```
sermon_whisper_chunks.json (1880 chunks)
         │
         ├── filter (≥20 chars, deduplicate)
         ├── cap (1200 chunks, seeded)
         │
         ├── 70% ──── TranslateGemma-12B ──┐
         │            (cache-first)          │
         │                                   ├── merge ── hybrid_sermon_pairs.jsonl
         │                                   │
         └── 30% ──── DeepL Pro API ────────┘
                      (glossary-enforced)

hybrid_sermon_pairs.jsonl ──── train_gemma.py --sermon-data ──── S1/S2/S3 adapters
                                                                       │
                                                            evaluate_sermon.py
                                                        (dual ceiling: 12B + DeepL)
```

## Privacy

- Whisper transcription runs locally (no audio leaves the machine)
- DeepL sees sermon text only — this is public YouTube content, no PII
- The DeepL glossary is uploaded temporarily and deleted after translation

## Cost

| Component | Cost |
|-----------|------|
| DeepL Pro API (~360 chunks, ~25K chars) | ~$0.72 |
| 12B GPU (~500 uncached, ~62 min on A2000 Ada) | Electricity only |
| Total | < $1.00 |

## DeepL Glossary

207 bare terms (1-2 words) from `bible_data/glossary/glossary_pairs.jsonl`, filtered by `len(en.split()) <= 2`.

Key enforced mappings:
- "James" → "Santiago" (default — most common usage is the epistle)
- "propitiation" → "propiciación"
- "covenant" → "pacto"
- "atonement" → "expiación"

Note: DeepL glossary doesn't support context-based disambiguation. "James" as apostle (→ "Jacobo") vs. epistle (→ "Santiago") is the model's job post-training.

## Data Budget

| Category | Count |
|----------|-------|
| Raw whisper chunks | 1,880 |
| After filter (≥20 chars, deduped) | ~1,862 |
| Capped for training | 1,200 (default) |
| Via 12B (70%) | ~840 |
| Via DeepL (30%) | ~360 |
| 12B cache hits (estimated) | ~340 |
| 12B uncached (need inference) | ~500 |

## Training Configs (S1/S2/S3 Sweep)

All use the same data mix: verse pairs (8000, --max-pairs) + hybrid sermon pairs + glossary (2x oversample).

| Run | LR | Steps | Extras | Rationale |
|-----|-----|-------|--------|-----------|
| S1 | 1e-5 | 50 | — | Conservative (A1 config). Tests minimal updates on new data. |
| S2 | 3e-6 | 100 | — | Middle ground. Tests moderate lr with sermon-domain data. |
| S3 | 1e-6 | full epoch | neftune=5 | Aggressive (B4 config). Tests if the problem was data quality, not training intensity. |

## Evaluation: Dual Ceiling

Each adapter is evaluated against two quality ceilings simultaneously:

1. **12B ceiling** — TranslateGemma-12B base translations (cached from Phase 2.5)
2. **DeepL ceiling** — DeepL Pro with glossary enforcement (computed during eval)

Metrics computed for each ceiling:
- COMET proximity gain (primary)
- chrF++ proximity gain (secondary)

Plus existing gates: hallucination ratio, theological term accuracy, archaic register markers.

## Quality Targets

| Metric | Floor | Minimum | Target |
|--------|-------|---------|--------|
| COMET (Bible holdout) | > 0.740 | > 0.752 (beat base) | > 0.780 |
| COMET proximity to 12B | > -0.01 | > 0.0 | > +0.005 |
| COMET proximity to DeepL | > -0.01 | > 0.0 | > +0.005 |
| Glossary regressions | < 10 lost | < 5 lost | < 3 lost |
| BLEU | > 17.7 (sanity) | — | — |

**Verdict rule**: Adapter must PASS or WARN on **both** 12B and DeepL proximity. KILL on either = reject.

## Commands

### Prerequisites

```bash
source /home/wbell/stt_train_env/bin/activate
pip install deepl
export DEEPL_KEY="your-key-here"
```

### Generate hybrid data only (~65 min)

```bash
python training/generate_hybrid_synthetic.py \
    --deepl-key "$DEEPL_KEY" \
    --max-chunks 1200
```

### Quick test (300 chunks, ~15 min)

```bash
python training/generate_hybrid_synthetic.py \
    --deepl-key "$DEEPL_KEY" \
    --max-chunks 300
```

### Full pipeline (~4.5 hrs)

```bash
export DEEPL_KEY="your-key-here"
nohup bash training/run_hybrid_scale.sh &
tail -f hybrid_runs/hybrid_log.txt
```

### Individual evaluation with dual ceiling

```bash
python training/evaluate_sermon.py \
    --chunks ablation/sermon_test_chunks_v2.json \
    --adapter hybrid_runs/S1_lr1e5_50steps \
    --ceiling-cache ablation/sermon_12b_translations_v2.json \
    --deepl-key "$DEEPL_KEY" \
    --output hybrid_runs/S1_sermon_eval.json
```

### DeepL reference comparison on holdout verses

```bash
python training/evaluate_translation.py \
    --adapter hybrid_runs/S1_lr1e5_50steps \
    --glossary-only --compare-base \
    --deepl-key "$DEEPL_KEY"
```

## Estimated Runtime

### Phase A: S1/S2/S3 (training config sweep)

| Step | Time |
|------|------|
| Data generation (12B + DeepL, 1200 chunks) | ~65 min |
| S1 training (50 steps) | ~5 min |
| S2 training (100 steps) | ~8 min |
| S3 training (~1114 steps) | ~75 min |
| 3 sermon evals (dual ceiling) | ~90 min |
| 3 glossary evals | ~15 min |
| **Phase A total** | **~4.5 hrs** |

### Phase B: S4/S5/S6 (sermon-dominant ratio sweep)

| Step | Time |
|------|------|
| Data generation (12B + DeepL, 1800 chunks, 60/40) | ~75 min |
| S4 training (sermon-only) | ~8 min |
| S5 training (sermon-heavy) | ~8 min |
| S6 training (balanced) | ~8 min |
| 3 sermon evals (dual ceiling) | ~90 min |
| 3 glossary evals | ~15 min |
| **Phase B total** | **~3.5 hrs** |

### Phase C: Expanded sermon pool (parallel with eval)

| Step | Time | VRAM |
|------|------|------|
| Transcribe 18 sermons (faster-whisper large-v3) | ~2.5 hrs | ~5 GB |
| Regenerate hybrid data from expanded pool | ~2-3 hrs | ~7 GB |
| Re-run winning config with expanded data | ~1.5 hrs | ~10 GB |

---

## Phase B: Sermon-Dominant Ratio Sweep (S4/S5/S6)

### Rationale

S1/S2/S3 test training config (lr/steps/neftune) with a fixed data mix: 8000 verse pairs + ~1200 hybrid sermon pairs. Sermon data is only ~11.5% of training. Given that Phase 2.5 showed verse pairs alone produce WARN/KILL on sermon eval, drowning 1200 sermon pairs in 8000 verse pairs likely isn't the fix.

S4/S5/S6 use the **winning training config** from S1/S2/S3, but vary the **verse/sermon ratio** to be sermon-dominant.

### Changes from S1/S2/S3

- **DeepL ratio increased to 40%** (from 30%): Stronger glossary enforcement for sermon-dominant training. 720 glossary-enforced examples vs 360. Cost: ~$1.50 total.
- **All 1800 available chunks used** (up from 1200): Maximize sermon signal.
- **Glossary reduced to 1x oversampling** (from 2x): 40% of sermon pairs already have DeepL glossary enforcement baked in, so less oversampling needed.

### Data Mix

| Run | Verse pairs | Glossary | Sermon pairs | Total | Sermon % | What it tests |
|-----|------------|----------|-------------|-------|----------|---------------|
| S4 | 0 | 507 (1x) | 1800 | ~2,307 | **78%** | Pure sermon + glossary. Are verse pairs harmful? |
| S5 | 500 | 507 (1x) | 1800 | ~2,807 | **64%** | Light verse anchor. Minimal verse pairs. |
| S6 | 1800 | 507 (1x) | 1800 | ~4,107 | **44%** | Balanced. Equal verse/sermon. |

### Commands

```bash
# Edit winning config in script first, then run:
export DEEPL_KEY="your-key-here"
nohup bash training/run_hybrid_ratio.sh &
tail -f hybrid_runs/ratio_sweep_log.txt
```

### Decision Matrix

| Outcome | Meaning | Next step |
|---------|---------|-----------|
| S4 wins (0 verse) | Verse pairs actively harmful | Drop verse pairs, scale up sermon data (Phase C) |
| S5 wins (500 verse) | Light verse anchor helps | Keep small verse buffer, scale up sermon data |
| S6 wins (1800 verse) | Balance matters | Maintain ratio, scale up both |
| All three similar | Ratio doesn't matter | Scale up sermon data, use simplest config |
| All three KILL | 1800 sermon pairs insufficient | Need Phase C (18,000+ chunks) before re-running |

---

## Phase C: Expanded Sermon Pool

### Data Inventory

**Current**: 1,880 raw chunks from 2 conference sermons (`conference_sat_gospel` + `conference_sun_gospel`).

**Available**: 18 additional gospel messages in `stark_data/raw/midwest/` (~3.6 GB, each ~60-70 min). At ~900 chunks per sermon → ~16,000+ new chunks.

### Transcription

```bash
# ~2.5 hrs, ~5 GB VRAM (can overlap with Phase B eval)
nohup bash training/transcribe_sermons.sh &
tail -f ablation/transcribe_sermons_log.txt
```

Output: `ablation/sermon_whisper_chunks_expanded.json` (merged ~18,000+ chunks).

### Regenerate with expanded pool

```bash
python training/generate_hybrid_synthetic.py \
    --deepl-key "$DEEPL_KEY" \
    --whisper-chunks ablation/sermon_whisper_chunks_expanded.json \
    --max-chunks 5000 --ratio-deepl 0.40 \
    --output bible_data/synthetic/hybrid_sermon_pairs_expanded.jsonl
```
