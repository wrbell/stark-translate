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
| Base (no FT) | 19.7 | — | — | — | — | 16 | Reference |
| **A1 (50 steps)** | **20.4** | **45.2** | **0.752** | 50 | 1e-5 | 16 | Best COMET, 5 min train |
| A2 (150 steps) | 12.3 | 28.9 | 0.560 | 150 | 1e-5 | 16 | Forgetting cliff confirmed |
| A3 (lr=5e-6, full) | 8.4 | 27.0 | 0.533 | ~1114 | 5e-6 | 16 | Halving lr not enough |
| **A4 (lr=1e-6, full)** | **20.7** | **44.9** | **0.740** | ~1114 | 1e-6 | 16 | Best BLEU, survives full epoch |
| A5 (rank=4, full) | 10.0 | 29.1 | 0.551 | ~1114 | 1e-5 | 4 | Rank reduction doesn't prevent forgetting |
| A6 (replay 20%) | TBD | TBD | TBD | ~1114 | 1e-5 | 16 | Pending — does replay anchor general ability? |

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

### Decision logic

| Result | Interpretation | Next step |
|--------|---------------|-----------|
| B2 > A1 and A4 | lr=1e-6 + 150 steps is the sweet spot | Lock config, proceed to scale-up |
| B1 ≈ A1, B2 ≈ A4 | lr=1e-6 doesn't help at low steps, marginal at moderate | A1 may be optimal; test B3/B4 for regularization gains |
| B3 or B4 > A4 | Regularization pushes quality beyond safe-lr baseline | Combine winning regularization with B2's step count |
| B5 > A4 (and A6 worked) | Replay + safe lr compounds | Include replay in final config |
| Nothing beats A1 significantly | Minimal updates genuinely optimal for this model | Lock A1 config, scale-up runs test more diverse data in 50 steps |

### Time estimates

| Run | Train time | Eval time | Total |
|-----|-----------|-----------|-------|
| B1 (50 steps) | ~3 min | ~47 min | ~50 min |
| B2 (150 steps) | ~10 min | ~47 min | ~57 min |
| B3 (full, ~1114 steps) | ~74 min | ~47 min | ~121 min |
| B4 (full, ~1114 steps) | ~74 min | ~47 min | ~121 min |
| B5 (full, ~1114 steps) | ~74 min | ~47 min | ~121 min |
| **Total** | | | **~7.8 hrs** |

B1 + B2 finish in ~1.8 hrs and give early signal. B3-B5 can be skipped if B2 clearly wins.

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
2. **4B + adapter** (Phase 2 winner) — what we're testing
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

### Automated metrics (run first, ~5 min GPU)

These run before human review. If any automated kill switch triggers, stop immediately.

**1. 12B proximity gain (primary quantitative signal)**

Use 12B translations as pseudo-references. Compute chrF++ between each 4B variant and 12B:

```
D_base    = chrF++(4B_base_output, 12B_output)     # how far 4B base is from 12B
D_adapter = chrF++(4B_adapter_output, 12B_output)   # how far 4B adapter is from 12B
proximity_gain = D_adapter - D_base                  # positive = adapter moved toward 12B
```

Computed on clean chunks only (STT confidence > 0.7). Noisy chunks excluded — 12B may also
struggle with garbled input, making it a poor reference.

| Proximity gain | Interpretation |
|----------------|---------------|
| > +2 chrF++ points | Adapter clearly moved 4B toward 12B quality |
| 0 to +2 | Marginal — proceed with caution, check human review |
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

# --- Translate with all 3 models ---
# Script: training/evaluate_sermon.py (to be written — loads each model,
# translates all chunks + sermon-style sentences, outputs comparison JSON
# with automated metrics)

python training/evaluate_sermon.py \
    --chunks ablation/sermon_test_chunks.json \
    --adapter ablation/${WINNER} \
    --base-model google/translategemma-4b-it \
    --ceiling-model google/translategemma-12b-it \
    --segment ablation/sermon_segment.txt \
    --output ablation/sermon_eval_results.json

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
| Proximity gain | < 0 on clean chunks | Adapter moved away from 12B quality |
| Hallucination ratio | > 1.5x base on low-conf chunks | Adapter fabricates content on noisy input |
| Archaic register | > 2x 12B on all outputs | Biblical verse training contaminated spoken register |

**Quantitative gates (all must pass):**

| Gate | Threshold | What it measures |
|------|-----------|-----------------|
| Proximity gain | > 0 on clean chunks | Adapter moved toward 12B quality |
| Sermon-style theo terms | ≥ formal accuracy - 1 | Theological terms transfer to casual register |
| Hallucination ratio | ≤ base × 1.2 | No meaningful hallucination increase |

**Human review (for borderline cases):**

| Result | Decision |
|--------|----------|
| ≥ 70% of chunks rated "Better" or "Same" | Proceed to Phase 3 scale-up |
| 50-70% "Better/Same" | Borderline — check if failures are concentrated in one category |
| < 50% "Better/Same" | Do not proceed. Investigate failure mode before scaling. |

### Time estimate

| Task | Time | Type |
|------|------|------|
| Translate 20 chunks × 3 models | ~6 min | GPU (sequential model loading) |
| Translate 8 sermon-style sentences × 3 models | ~2 min | GPU |
| Transcribe + translate sermon segment × 3 | ~10 min | GPU |
| Automated metrics computation | ~1 min | CPU |
| Human review (targeted, ~18 chunks + segment) | ~30 min | Human |
| **Total** | **~50 min** | ~20 min GPU + ~30 min human |

If any automated kill switch fires, human review is skipped — saving 30 minutes.
Total cost is trivial compared to 3-20+ hours of scale-up.

---

## Phase 3 — Validation & Scale (2-4 runs, ~3-20 hrs)

Once Phase 2 locks the winning config and Phase 2.5 confirms it works on sermon content,
scale up data. See `docs/scale_run.md` for the full scale-up test matrix with time
estimates and decision gates.

| ID | Config | Rationale |
|----|--------|-----------|
| S1a | Best Phase 2 config at 20K pairs, same steps | Test if more data diversity helps at same compute |
| S1b | Best Phase 2 config at 20K pairs, proportional steps | Test if more training on more data compounds |
| S2a | 50K pairs, same steps (conditional) | Only if S1 BLEU > Phase 2 best + 2 |
| S2b | 50K pairs, proportional steps (conditional) | Only if S2a shows continued scaling |

### Go/No-Go gates

| Metric | Floor (reject) | Minimum | Target |
|--------|---------------|---------|--------|
| BLEU | > 17.7 (-10% base) | > 20.7 (beat A4) | > 22.7 (+3 over base) |
| chrF++ | > 40.0 | > 45.0 | > 48.0 |
| COMET | > 0.720 | > 0.740 (match A4) | > 0.770 |
| Theological terms | > 3/8 (37%) | > 5/8 (62%) | > 7/8 (87%) |

---

## Timeline (updated with actuals)

| Phase | Runs | Wall Clock | Human Time | Status |
|-------|------|-----------|------------|--------|
| 0: Diagnostic | 1 | 47 min | 5 min | Done (BLEU 9.7) |
| 1: Ablation | 6 | ~9.5 hrs | 30 min | Running (5/6 done) |
| 2: Combine | 5 | ~7.8 hrs | 20 min | Planned |
| 2.5: Sermon smoke test | 1 | ~5 min GPU | ~45 min human | Planned |
| 3: Scale-up | 2-4 | ~3-20 hrs | 10 min | Planned (see scale_run.md) |
| **Total** | **15-17** | **~21-38 hrs** | **~2 hrs** |

Can be spread across 3-4 sessions. Phase 2 B1+B2 give early signal in ~1.8 hrs.
Phase 2.5 is mostly human review time — GPU cost is trivial (20 chunks).

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
