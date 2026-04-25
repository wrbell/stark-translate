# TranslateGemma QLoRA Tuning — Test Matrix

## Context

TranslateGemma 4B base translates EN→ES at BLEU 19.7. Two fine-tuning attempts both caused catastrophic forgetting:

| Run | Config | Steps | BLEU | Delta |
|-----|--------|-------|------|-------|
| Base (no adapter) | — | — | 19.7 | — |
| Packed (cycle 1) | lr=1e-5, 8K pairs, packing=True | 312 | 9.0 | -54% |
| Nopack (cycle 1b) | lr=1e-5, 8K pairs, packing=False | 1114 | **1.8 (2 verses!)** | eval broken |

**Critical discovery:** The nopack eval only ran on 2 verses because it used the stub file at `bible_data/holdout/verse_pairs_test.jsonl` (2 lines) instead of the real data at `bible_data/aligned/verse_pairs_test.jsonl` (27,130 lines). The nopack adapter has **never been properly evaluated**.

---

## Phase 0 — Diagnostic (1 run, ~30 min)

Re-evaluate the existing nopack adapter on the full 500-verse holdout to determine if it's actually working. Zero-cost (adapter already exists) and could save the entire ablation study.

| ID | What | Command |
|----|------|---------|
| D0 | Re-eval nopack on 500 verses | `python training/evaluate_translation.py --adapter fine_tuned_gemma_mi_A_nopack --test bible_data/aligned/verse_pairs_test.jsonl --max-samples 500` |

**Decision:**
- **BLEU > 17.7** (within 10% of base): The adapter works. Skip to Phase 3 optimization.
- **BLEU < 17.7**: Proceed to Phase 1 ablation.

---

## Phase 1 — Ablation: Find the Dominant Variable (6 runs, ~4-5 hrs)

Each run uses 8K pairs, 1 epoch, packing=False. Change ONE variable per run. ~35-45 min each.

| ID | Variable | Value | Other defaults | Rationale |
|----|----------|-------|----------------|-----------|
| A1 | max_steps | 50 | lr=1e-5, r=16 | Test if fewer steps prevents forgetting |
| A2 | max_steps | 150 | lr=1e-5, r=16 | Find the step-count cliff |
| A3 | lr | 5e-6 | 1114 steps (full epoch), r=16 | Half the current lr |
| A4 | lr | 1e-6 | 1114 steps (full epoch), r=16 | Ultra-conservative lr |
| A5 | rank | 4 | lr=1e-5, alpha=8, 1114 steps | Less adapter capacity = less forgetting |
| A6 | replay | 20% general pairs | lr=1e-5, r=16, 1114 steps | Mix WMT/OPUS EN→ES to anchor general ability |

**Controls held constant:** packing=False, glossary=3x (A-series used old default), bf16, cosine scheduler, warmup=0.1, max_grad_norm=0.5, dropout=0.05, seed=42.

### Commands

```bash
# A1: 50 steps
python training/train_gemma.py A --max-pairs 8000 --max-steps 50 -o ablation/A1_steps50

# A2: 150 steps
python training/train_gemma.py A --max-pairs 8000 --max-steps 150 -o ablation/A2_steps150

# A3: lr=5e-6
python training/train_gemma.py A --max-pairs 8000 --lr 5e-6 -o ablation/A3_lr5e6

# A4: lr=1e-6
python training/train_gemma.py A --max-pairs 8000 --lr 1e-6 -o ablation/A4_lr1e6

# A5: rank=4, alpha=8
python training/train_gemma.py A --max-pairs 8000 --lora-r 4 --lora-alpha 8 -o ablation/A5_rank4

# A6: 20% replay
python training/train_gemma.py A --max-pairs 8000 --replay-ratio 0.2 -o ablation/A6_replay20

# Evaluate all (500 verses each)
for d in ablation/A*/; do
  name=$(basename "$d")
  python training/evaluate_translation.py --adapter "$d" --max-samples 500 \
    --output-file "ablation/${name}_metrics.json"
done
```

### Replay data source (for A6)

Uses `Helsinki-NLP/opus-100` EN→ES split from HuggingFace (~1M pairs). The `--replay-ratio 0.2` flag loads pairs automatically via streaming and mixes them into training data. This mirrors Whisper's 30% LibriSpeech approach.

### Decision logic

| Result | Interpretation | Next step |
|--------|---------------|-----------|
| A1 or A2 works (step-limited) | Model needs very few gradient updates | Short training + more targeted data |
| A3 or A4 works (lr-limited) | Model tolerates full-epoch training at lower lr | Scale up data safely |
| A5 works (rank-limited) | Adapter has too much capacity | Reduce rank for gentle adaptation |
| A6 works (replay helps) | Catastrophic forgetting is the core issue | Always include replay data |
| Nothing works | Chat template or data format may be subtly wrong | Manual inspection needed |

---

## Phase 1 — Ablation Results (2026-03-20)

| Run | BLEU | chrF++ | COMET | Steps | LR | Rank | Notes |
|-----|------|--------|-------|-------|----|------|-------|
| Base (no FT) | 19.7 | — | **0.7516** | — | — | 16 | Reference |
| **A1 (50 steps)** | **20.4** | **45.2** | **0.752** | 50 | 1e-5 | 16 | Best COMET, 5 min train |
| A2 (150 steps) | 12.3 | 28.9 | 0.560 | 150 | 1e-5 | 16 | Forgetting cliff confirmed |
| A3 (lr=5e-6, full) | 8.4 | 27.0 | 0.533 | ~1114 | 5e-6 | 16 | Halving lr not enough |
| **A4 (lr=1e-6, full)** | **20.7** | **44.9** | **0.740** | ~1114 | 1e-6 | 16 | Best BLEU, survives full epoch |
| A5 (rank=4, full) | 10.0 | 29.1 | 0.551 | ~1114 | 1e-5 | 4 | Rank reduction doesn't prevent forgetting |
| A6 (replay 20%) | 8.5 | 28.0 | 0.541 | ~1114 | 1e-5 | 16 | Replay at lr=1e-5 → catastrophic forgetting (4/8 theo) |

### Key findings

1. **Learning rate is the dominant variable.** At lr=1e-5, forgetting hits by step 150 regardless
   of rank (A2, A5). At lr=1e-6, a full 1114-step epoch is safe (A4).
2. **The forgetting cliff at lr=1e-5 is between steps 50–150.** A1 (50 steps) is fine; A2 (150 steps)
   has already lost 7.4 BLEU.
3. **A1 and A4 are nearly tied.** A1 got 97% of A4's BLEU in 5 min vs 68 min. A1 actually wins
   on COMET (0.752 vs 0.740). Open question: did A4's 22x more updates teach anything useful,
   or did it just barely avoid breaking?
4. **Rank reduction doesn't help.** A5 (rank=4) still forgot at lr=1e-5. Adapter capacity isn't
   the bottleneck — learning rate is.
5. **Replay at lr=1e-5 doesn't help.** A6 (replay 20%) still forgot — BLEU 8.5, worse than base.
   Replay can't compensate for an lr that's too high.

---

## Phase 2 — Combine Winners (5 runs, ~3-5 hrs)

**Goal:** Find the sweet spot between A1 (fast, minimal updates) and A4 (slow, full epoch).
Test whether lr=1e-6 unlocks quality gains at moderate step counts, and whether
regularization (neftune, dropout=0, replay) can push further.

All runs use 8K pairs, packing=False, rank=16, alpha=32 unless noted.

| ID | Variable | Value | Other defaults | Rationale |
|----|----------|-------|----------------|-----------|
| B1 | lr=1e-6, 50 steps | Conservative lr + minimal steps | r=16 | Does safe lr help even at low steps? Isolates lr effect vs A1. |
| B2 | lr=1e-6, 150 steps | Conservative lr + moderate steps | r=16 | Can we push past A1's quality with safe lr? A2 forgot here at lr=1e-5. |
| B3 | lr=1e-6, full + dropout=0 | Full epoch, no LoRA dropout | r=16 | Kikuyu study: dropout=0 better for short/conservative runs. |
| B4 | lr=1e-6, full + neftune=5 | Full epoch + noise regularization | r=16 | Kikuyu study: neftune=5 helped. Does it compound with safe lr? |
| B5 | lr=1e-6, full + replay=20% | Full epoch + replay buffer | r=16 | If A6 shows replay helps, combine with safe lr. If A6 flops, skip. |

**Controls held constant:** packing=False, glossary=2x (new default; A-series used 3x), bf16, cosine scheduler, warmup=0.1,
max_grad_norm=0.5, seed=42, max_pairs=8000.

**Note:** Glossary oversampling reduced from 3x→2x vs A-series. Minor variable (5.1% vs 10.3%
of training data at 8K) compared to lr changes being tested.

### Commands

```bash
# B1: lr=1e-6, 50 steps
python training/train_gemma.py A --max-pairs 8000 --lr 1e-6 --max-steps 50 \
  -o ablation/B1_lr1e6_50

# B2: lr=1e-6, 150 steps
python training/train_gemma.py A --max-pairs 8000 --lr 1e-6 --max-steps 150 \
  -o ablation/B2_lr1e6_150

# B3: lr=1e-6, full epoch, dropout=0
python training/train_gemma.py A --max-pairs 8000 --lr 1e-6 --lora-dropout 0 \
  -o ablation/B3_lr1e6_nodrop

# B4: lr=1e-6, full epoch, neftune=5
python training/train_gemma.py A --max-pairs 8000 --lr 1e-6 --neftune 5 \
  -o ablation/B4_lr1e6_neftune

# B5: lr=1e-6, full epoch, replay=20% (conditional on A6 results)
python training/train_gemma.py A --max-pairs 8000 --lr 1e-6 --replay-ratio 0.2 \
  -o ablation/B5_lr1e6_replay

# Evaluate all (500 verses each)
for d in ablation/B*/; do
  name=$(basename "$d")
  python training/evaluate_translation.py --adapter "$d" --max-samples 500 \
    --output-file "ablation/${name}_metrics.json"
done
```

### B-Series Results (2026-03-21)

| Run | BLEU | chrF++ | COMET | Steps | LR | Extras | Theo | Train |
|-----|------|--------|-------|-------|----|--------|------|-------|
| B1 (lr=1e-6, 50 steps) | 19.6 | 44.5 | 0.750 | 50 | 1e-6 | — | 5/8 | 4 min |
| B2 (lr=1e-6, 150 steps) | 20.0 | 44.7 | 0.751 | 150 | 1e-6 | — | 5/8 | 10 min |
| B3 (lr=1e-6, dropout=0) | 20.4 | 45.2 | 0.742 | ~1114 | 1e-6 | dropout=0 | 5/8 | 60 min |
| **B4 (lr=1e-6, neftune=5)** | **21.2** | **45.3** | 0.742 | ~1114 | 1e-6 | neftune=5 | **5/8** | **63 min** |
| B5 (lr=1e-6, replay 20%) | 20.5 | 44.5 | 0.735 | ~1114 | 1e-6 | replay=20% | 5/8 | 77 min |

### B-Series Key Findings

1. **B4 (neftune=5) is the overall BLEU winner at 21.2** — the only run to clearly beat A4 (20.7).
   NEFTune noise regularization provides a consistent +0.5 BLEU over the dropout=0 baseline (B3).
2. **COMET peaked at A1 (0.752), not B4 (0.742).** The neural quality metric suggests A1's
   translations are slightly more semantically accurate. B4 trained 12x longer and may have
   subtly overfit to Bible phrasing patterns that boost BLEU but not semantic quality.
3. **Replay didn't help at safe lr.** B5 (+0.5 over base) was marginal and worse than B4.
   Combined with A6's catastrophic forgetting at lr=1e-5, replay is not worth the complexity.
4. **Theological terms hit a hard wall at 5/8 (62%).** Every working run scores exactly 5/8.
   The same 3 terms fail every time — these are **data problems**, not hyperparameter problems:
   - **James (epistle) → Santiago:** Bible data has 0 Santiago, 155 Jacobo. 6 glossary pairs overwhelmed.
   - **James (apostle) → Jacobo:** Model doesn't disambiguate — needs contextual signal it never learned.
   - **Propitiation → propiciación:** Model prefers the synonym "expiación" (more common in training data). Only 2 glossary pairs.
5. **The improvement ceiling is low.** Best BLEU 21.2 = only +1.5 over base (19.7). Across all
   11 runs, the working range is 19.6–21.2 — a spread of 1.6 BLEU, within the noise floor.

### Dual Assessment: BLEU vs COMET

When sorted by COMET (semantic quality), the picture is different from BLEU:

| Run | COMET | BLEU | Steps | Config |
|-----|-------|------|-------|--------|
| Base (no FT) | **0.7516** | 19.7 | — | — |
| A1 (50 steps) | **0.752** | 20.4 | 50 | lr=1e-5 |
| B2 (150 steps) | 0.751 | 20.0 | 150 | lr=1e-6 |
| B1 (50 steps) | 0.750 | 19.6 | 50 | lr=1e-6 |
| B4 (neftune=5) | 0.742 | **21.2** | ~1114 | lr=1e-6, neftune=5 |
| A4 (lr=1e-6) | 0.740 | 20.7 | ~1114 | lr=1e-6 |

**No fine-tuned model meaningfully improves COMET over base (0.7516).** A1 is +0.0006 — within
noise. B4 (the BLEU "winner") is actually a 1% COMET regression. More training steps = worse
COMET. Fine-tuning learns Bible n-gram patterns that boost BLEU but degrade semantic quality.

- **BLEU winner: B4** (21.2) — but COMET 0.742, below base
- **COMET winner: A1** (0.752) — but within noise of base (0.7516)
- **Both go to Phase 2.5 sermon smoke test to settle it**

### Metric hierarchy

COMET > COMET proximity to 12B > glossary regressions > chrF++ > BLEU (reported only)

---

## Phase 2.5 — Sermon Smoke Test (~1.5 hrs)

**Goal:** Verify that the winning adapter improves translation on real sermon content, not
just Bible verse pairs. Scale-up (Phase 3) costs 3-20+ GPU hours — this phase is the gate
that prevents wasting that time on an adapter that only helps on verse pairs but degrades
the actual production use case.

### Why this matters

Bible verses and sermon speech are fundamentally different:

| Property | Bible verses (training/eval) | Sermon speech (production) |
|----------|------------------------------|----------------------------|
| Register | Formal, literary | Spoken, informal, with fillers |
| Length | 1 sentence, self-contained | Fragments to multi-sentence |
| Input quality | Clean text | STT output with errors |
| Theological terms | In canonical phrasing | In casual explanation context |
| Examples | "The righteousness of God is revealed." | "Daniel was a man prayer." (STT error) |

An adapter that scores well on verse BLEU but degrades sermon translation is worse than
useless — it would actively harm the live demo.

### Three-model evaluation

Translate every test input with **three models**:

1. **4B base** (no adapter) — current production quality
2. **4B + adapter** (A1 and B4 tested separately) — what we're testing
3. **12B base** (no adapter) — quality ceiling / pseudo-reference

12B serves as a quality anchor. In production, 12B replaces 4B output on silence detection,
so 12B's translation is the "correct" answer for our use case. If fine-tuned 4B moves toward
12B quality, the speculative drafts (shown while speaking) improve — that's the actual goal.

### Test data: 3 tiers

**Tier 1 — Real sermon chunks (20 chunks)**

Source: `stark_data/corrections/review_queue_20260301.tsv` (2 live sessions, 2026-03-01).
Each has raw STT English, STT confidence (0.05–0.91), and existing base Gemma/Marian outputs.

Stratify by category for per-category analysis:

| Category | Chunks | What it tests |
|----------|--------|---------------|
| Theological (conf > 0.7) | ~5 | Term accuracy in sermon context |
| Clean general (conf > 0.7) | ~5 | Baseline regression on normal speech |
| STT noise (conf < 0.5) | ~5 | Graceful degradation vs hallucination |
| Short fragments | ~5 | Hallucination on ambiguous input |

**Tier 2 — Sermon-style theological sentences (8 sentences, new)**

The existing spot-check in `evaluate_translation.py` uses formal sentences:
- "Christ's atonement covers all sins." → expects "expiación"

But sermons sound like this:
- "And you know, when we think about the atonement — what Christ did for us on the cross..."

Same 8 theological terms, sermon register. Tests whether term accuracy transfers from
formal training data to casual spoken context.

| Formal (existing spot-check) | Sermon-style (new) | Expected term |
|-----------------------------|--------------------|---------------|
| "Christ's atonement covers all sins." | "And you know, when we think about the atonement — what Christ did for us on that cross..." | expiación |
| "The covenant between God and Abraham." | "God made a covenant with Abraham, and brothers, He keeps His promises." | pacto |
| "We are saved by grace through faith." | "It's only by grace, friends. We can't earn it — it's grace through faith." | gracia |
| "The righteousness of God is revealed." | "And Paul writes about the righteousness of God here in Romans, and what does he mean by that?" | justicia |
| "James wrote about faith and works." | "Now, if you turn to the book of James — James has a lot to say about faith and works." | Santiago |
| "James and John were fishermen." | "You think about James and John, just regular fishermen on the Sea of Galilee." | Jacobo |
| "He preached about sanctification." | "The speaker tonight was talking about sanctification — what it means to be set apart." | santificación |
| "The propitiation for our sins." | "First John chapter 2 — He is the propitiation for our sins, and not for ours only." | propiciación |

**Tier 3 — Full sermon segment (1 segment, ~2-3 minutes)**

Pick one high-quality WAV from `stark_data/raw/midwest/` (conference sermon, soundboard audio).
Transcribe with Whisper, translate full segment with all 3 models. Tests register drift over
sustained context — individual chunks might look fine, but does the adapter shift register
across a full sermon passage?

### Step 0: Generate + cache 12B translations (~1-2 hrs GPU, do once)

Generate 12B translations for ALL sermon chunks up front. This cache serves both
Phase 2.5 evaluation (as pseudo-references) and Phase 3a distillation data.

1. Transcribe sermon WAVs from `stark_data/raw/midwest/` with Whisper large-v3
2. Translate ALL chunks with 12B (both ~20 test chunks AND ~500-1000 full-corpus chunks)
3. Cache to `ablation/sermon_12b_translations.json`

This is the most expensive step but feeds both Phase 2.5 eval and Phase 3a data generation.

### Automated metrics (run first, ~5 min GPU)

These run before human review. If any automated kill switch triggers, stop immediately.

**1. 12B proximity gain (primary quantitative signal)**

Use 12B translations as pseudo-references. Compute COMET between each 4B variant and 12B:

```
D_base    = COMET(4B_base_output, 12B_output)       # how far 4B base is from 12B
D_adapter = COMET(4B_adapter_output, 12B_output)     # how far 4B adapter is from 12B
proximity_gain = D_adapter - D_base                   # positive = adapter moved toward 12B
```

Computed on clean chunks only (STT confidence > 0.7). Noisy chunks excluded — 12B may also
struggle with garbled input, making it a poor reference.

| Proximity gain | Interpretation |
|----------------|---------------|
| > +0.005 COMET | Adapter clearly moved 4B toward 12B quality |
| 0 to +0.005 | Marginal — proceed with caution, check human review |
| < 0 | **Kill switch: adapter moved AWAY from 12B.** Do not scale up. |

**2. Hallucination ratio (safety check)**

For each chunk: `ratio = len(output_words) / len(input_words)`

Compare adapter ratio distribution vs base ratio distribution on short/low-confidence chunks.
A model trained on verse pairs may "fill in" missing context when given fragmentary input.

| Result | Interpretation |
|--------|---------------|
| Adapter ratio ≈ base ratio | No hallucination increase |
| Adapter ratio > base ratio × 1.5 on low-conf chunks | **Kill switch: adapter hallucinates on noisy input.** |

**3. Sermon-style theological term accuracy (Tier 2)**

Run the 8 sermon-style sentences through all 3 models. Expected term found = pass.

| Result | Interpretation |
|--------|---------------|
| Adapter ≥ base on sermon-style terms | Theological knowledge transfers to casual register |
| Adapter < base on sermon-style but ≥ base on formal | Overfitting to formal phrasing — terms only recognized in verse context |
| 12B > both 4B variants significantly | 12B has theological knowledge 4B can't learn — distillation path (Phase 4 of Unsloth plan) |

**4. Register markers (automated scan)**

Count archaic Spanish markers in all outputs across the 20 chunks + 8 sermon sentences:
- Archaic forms: vosotros, habéis, sois, he aquí, mas (instead of pero), empero
- Formal liturgical: bienaventurado, menester, ved aquí

```
archaic_count_base    = count markers in 4B base outputs
archaic_count_adapter = count markers in 4B adapter outputs
archaic_count_12B     = count markers in 12B outputs
```

| Result | Interpretation |
|--------|---------------|
| adapter ≈ base | No register drift |
| adapter > base × 2 | **Warning: adapter shifting toward archaic register.** Check human review. |
| adapter > 12B × 2 | **Kill switch: adapter is more archaic than 12B.** Biblical text has contaminated the register. |

### Human review (targeted, ~30 min)

Only after automated metrics pass. Focus on cases where automated metrics can't judge:

1. **Review Tier 1 comparison table** — scan the 3 columns (4B base / 4B adapter / 12B) for
   the ~10 theological and low-confidence chunks. Mark each adapter output as:
   - **Better** than base (closer to 12B, better term, more natural)
   - **Same** as base
   - **Worse** than base (further from 12B, wrong term, hallucination, archaic register)

2. **Review Tier 2 sermon-style spot-check** — for any term the adapter missed that it got
   right in formal context, note it. This indicates overfitting to verse phrasing.

3. **Review Tier 3 full segment** — read through the full 2-3 minute translation. Does it
   read naturally as spoken Spanish? Any jarring register shifts mid-passage?

### Commands

```bash
# --- Tier 1: Export sermon chunks ---
python -c "
import csv, json
with open('stark_data/corrections/review_queue_20260301.tsv') as f:
    reader = csv.DictReader(f, delimiter='\t')
    chunks = [{'en': row['recorded_english'], 'confidence': float(row['stt_confidence']),
               'chunk_id': row['chunk_id'], 'existing_gemma': row['spanish_gemma'],
               'existing_marian': row['spanish_marian']}
              for row in reader if row['recorded_english'].strip()]
with open('ablation/sermon_test_chunks.json', 'w') as f:
    json.dump(chunks, f, indent=2, ensure_ascii=False)
print(f'Exported {len(chunks)} chunks')
"

# Step 0: Generate 12B translation cache (do once)
python training/generate_12b_cache.py \
    --chunks ablation/sermon_test_chunks.json \
    --output ablation/sermon_12b_translations.json

# Run sermon eval for A1 (COMET-optimal)
python training/evaluate_sermon.py \
    --chunks ablation/sermon_test_chunks.json \
    --adapter ablation/A1_steps50 \
    --ceiling-cache ablation/sermon_12b_translations.json \
    --output ablation/sermon_eval_A1.json

# Run sermon eval for B4 (BLEU-optimal)
python training/evaluate_sermon.py \
    --chunks ablation/sermon_test_chunks.json \
    --adapter ablation/B4_lr1e6_neftune \
    --ceiling-cache ablation/sermon_12b_translations.json \
    --output ablation/sermon_eval_B4.json

# --- Tier 3: Full sermon segment (optional but recommended) ---
# Pick a conference sermon with good audio
python -c "
import whisper
model = whisper.load_model('large-v3')
result = model.transcribe('stark_data/raw/midwest/4_Conference_2025_-_Saturday_Gospel_10_18_25_gsbiiVJ4_Bs.wav',
                          language='en', word_timestamps=True)
# Extract 2-3 minute segment (e.g., minutes 5-8)
segments = [s for s in result['segments'] if 300 <= s['start'] <= 480]
text = ' '.join(s['text'].strip() for s in segments)
with open('ablation/sermon_segment.txt', 'w') as f:
    f.write(text)
print(f'Extracted {len(segments)} segments, {len(text.split())} words')
"
```

**Note:** `training/evaluate_sermon.py` is implemented. It:
- Loads 4B base, 4B+adapter, and 12B sequentially (not simultaneously — VRAM)
- Translates all inputs with each model
- Computes proximity gain, hallucination ratio, register markers, sermon-style term accuracy
- Outputs a single JSON with all metrics + a comparison table for human review
- Includes automated kill switch verdicts (PASS/WARN/KILL)

### Go/No-Go gates

**Automated kill switches (any one = stop, do not proceed to scale-up):**

| Gate | Threshold | What it catches |
|------|-----------|-----------------|
| COMET proximity gain | < -0.01 | KILL: adapter degrades sermon quality |
| COMET proximity gain | < 0.005 | WARN: marginal improvement |
| Hallucination ratio | > 1.5x base on low-conf chunks | KILL: adapter fabricates content |
| Archaic register | > 2x 12B on all outputs | KILL: biblical training contaminated register |

### Decision tree

```
├── OUTCOME A: COMET proximity gain > 0.005
│     ├── Lock winner (A1 or B4)
│     └── Proceed to Phase 3 with synthetic sermon data
│
├── OUTCOME B: Both in noise range (COMET gain ±0.005)
│     ├── Verse pair FT doesn't help on sermons
│     ├── 12B translations already cached → go straight to Phase 3a
│     └── Try synthetic sermon data before giving up
│
└── OUTCOME C: COMET proximity < -0.01
      ├── KILL: Bible verse FT actively harms sermon quality
      └── Deploy base model, try Phase 3 with distilled data
```

**Human review (for borderline cases):**

| Result | Decision |
|--------|----------|
| ≥ 70% of chunks rated "Better" or "Same" | Proceed to Phase 3 scale-up |
| 50-70% "Better/Same" | Borderline — check if failures are concentrated in one category |
| < 50% "Better/Same" | Do not proceed. Investigate failure mode before scaling. |

### Time estimate

| Task | Time | Type |
|------|------|------|
| Generate 12B cache (Step 0) | ~1-2 hrs | GPU (one-time) |
| Translate 20 chunks × 3 models × 2 adapters | ~12 min | GPU (sequential model loading) |
| Translate 8 sermon-style sentences × 3 models | ~2 min | GPU |
| Transcribe + translate sermon segment × 3 | ~10 min | GPU |
| Automated metrics computation | ~1 min | CPU |
| Human review (targeted, ~18 chunks + segment) | ~30 min | Human |
| **Total** | **~40 min GPU + ~30 min human** | (excl. Step 0 cache) |

If any automated kill switch fires, human review is skipped — saving 30 minutes.
Total cost is trivial compared to 3-20+ hours of scale-up.

---

## Phase 3 — Synthetic Sermon Data + COMET-Primary Gates

**Status: Planned (after Phase 2.5).** Plain verse pairs are a weak signal — see
`docs/more_data.md`. The new path is 12B-distilled sermon data.

**Pre-gate:** Phase 2.5 must show COMET proximity gain > 0.005 on at least one adapter
before proceeding. If Bible verse FT can't improve sermon COMET at all, synthetic data
is the only path worth trying.

### Phase 3a: Synthetic sermon data via 12B distillation (~2-3 hrs GPU)

1. Transcribe 10-15 sermon WAVs from `stark_data/raw/midwest/` with Whisper large-v3
2. Segment into natural chunks (sentence/paragraph level)
3. Translate each chunk with TranslateGemma 12B (the quality ceiling)
4. Score each pair with COMET (filter threshold > 0.75)
5. Output as JSONL compatible with `train_gemma.py --sermon-data`

**Why 12B, not 4B:** This is knowledge distillation. Training 4B on "what 12B would say"
directly teaches 4B to produce 12B-like outputs → higher speculative draft acceptance
rate → lower latency for the congregation.

12B 4-bit (~7GB) fits on A2000 Ada 16GB. Cost: ~2.1s/chunk × 1000 chunks ≈ 35 min.
Expected yield: ~500-1000 high-quality sermon EN→ES pairs.

### Phase 3b: Retrain with distilled + verse + glossary mix

| ID | Config | Data mix | Rationale |
|----|--------|----------|-----------|
| S1 | Winner config, verse + sermon (65/35) | ~8K verse + ~500 sermon | Test if sermon data improves COMET |
| S2 | Winner config, verse + sermon (50/50) | ~4K verse + ~1K sermon | Higher sermon fraction |
| S3 | Winner config, sermon only | ~1K sermon pairs | Does removing verse pairs help? |

Each run uses `--sermon-data` flag in train_gemma.py (already implemented).

### Phase 3c: Evaluate with COMET-primary gates

| Metric | Floor | Minimum | Target |
|--------|-------|---------|--------|
| COMET | > 0.740 | > 0.752 | > 0.770 |
| Glossary regressions | < 10 lost | < 5 lost | 0 lost |
| chrF++ | > 40.0 | > 45.0 | > 48.0 |
| BLEU | > 17.7 (sanity) | — | — |

Re-run Phase 2.5 sermon smoke test on the new adapter.

### Phase 4 (future, if Phase 3 works)

CPT → Light SFT (method #1 from `more_data.md`). Raw monolingual sermon text + Bible text
for continued pretraining, then light SFT. Expected +0.08-0.15 COMET.

### Go/No-Go gates (COMET-primary)

| Metric | Floor (reject) | Minimum | Target |
|--------|---------------|---------|--------|
| COMET | > 0.740 | > 0.752 (beat A1) | > 0.770 |
| Glossary regressions | < 10 lost | < 5 lost | 0 lost |
| chrF++ | > 40.0 | > 45.0 | > 48.0 |
| BLEU | > 17.7 (sanity) | — | — |

---

## Timeline (updated with actuals)

| Phase | Runs | Wall Clock | Human Time | Status |
|-------|------|-----------|------------|--------|
| 0: Diagnostic | 1 | 47 min | 5 min | **Done** (BLEU 9.7) |
| 1: Ablation | 6 | ~9.5 hrs | 30 min | **Done** (A1-A6 complete) |
| 2: Combine | 5 | ~5.5 hrs | 20 min | **Done** (B1-B5 complete) |
| 2.5: Sermon smoke test (A1 + B4) | 2 | ~40 min GPU | ~45 min human | **Next** |
| 3: Synthetic sermon data + retrain | 3 | ~3-5 hrs GPU | ~30 min | Planned (after 2.5) |
| **Total (completed)** | **12** | **~15.5 hrs** | **~55 min** |

Phase 2.5 is the next gate — tests both A1 (COMET-optimal) and B4 (BLEU-optimal) on real
sermon content with COMET-primary evaluation.

---

## Key Insights from Community Research

| Source | Finding | Application |
|--------|---------|-------------|
| Kikuyu case study | dropout=0 beat dropout=0.1 for short runs | Test in Phase 2 |
| Kikuyu case study | neftune=5 helped, neftune=7 hurt | Test in Phase 2 |
| Kikuyu case study | `embed_tokens` in LoRA targets disrupts vocabulary | Already excluded |
| Kikuyu case study | Over-regularization worse than under-regularization | Guides Phase 2 choices |
| TranslateGemma paper | RL stage optimizes translation quality metrics | Aggressive FT undoes RL-learned preferences |
| Unsloth/trl warning | Packing without flash attention = cross-contamination | Already using packing=False |
| Google QLoRA guide | lr=2e-4 designed for new tasks, not domain adaptation | Our 1e-5 may still be too high |
| ALMA-R paper | 22K pairs matched GPT-4 for translation FT | 8K-20K is reasonable range |

---

## Risks

1. **Eval data path** — Fixed: default now points to `bible_data/aligned/verse_pairs_test.jsonl` (27,130 lines).
2. **No replay buffer** — Fixed: `--replay-ratio` flag loads from `Helsinki-NLP/opus-100` automatically.
3. **No `--max-steps` flag** — Fixed: `--max-steps N` stops training after N gradient steps.
4. **GPU contention** — Whisper pipeline uses GPU. Gemma ablation must wait or interleave.
