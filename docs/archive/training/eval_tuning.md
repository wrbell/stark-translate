# Next Steps: Post-Training Evaluation, Whisper LoRA, and Feedback Loop

## Context

TranslateGemma 4B QLoRA training (Phase I) completed successfully:
- **Adapter:** `fine_tuned_gemma_mi_A/` (147 MB safetensors)
- **Results:** train_loss=1.085, eval_loss=0.950, token_accuracy=80%, 5h41m runtime
- **No overfitting** (train/eval loss close, early stopping didn't trigger)

The project has completed Phases 0–1.7 + Phase C (Bible corpus, 242K+27K pairs) + Phase E (audio preprocessing, 13,562 clean chunks) + Phase G (pseudo-labeling, 13,563 transcripts) + Phase I (Gemma 4B). The next steps follow the training plan in `docs/training_plan_march.md`.

---

## Step 1: Evaluate Gemma 4B Adapter (Run 2 from training_plan_march.md)

**Goal:** Determine if the adapter meets go/no-go gates before proceeding.

**Script:** `training/evaluate_translation.py` (323 lines)
**Test data:** `bible_data/holdout/verse_pairs_test.jsonl` (27,130 verses, stratified by genre)

**Run on WSL** (model + adapter already local):
```bash
source ~/stt_train_env/bin/activate

# Quick spot-check first (8 theological terms, ~2 min)
python training/evaluate_translation.py --spot-check-only

# Then full corpus eval (27K verses — will take hours on A2000)
# Start with a small sample to validate the pipeline works:
python training/evaluate_translation.py --max-samples 500 \
  --output-file metrics/gemma_4b_cycle1_eval.json

# If that works, full eval overnight:
nohup python training/evaluate_translation.py \
  --output-file metrics/gemma_4b_cycle1_full.json \
  > gemma_eval.log 2>&1 &
```

**Also run base model for comparison** (A/B):
```bash
python training/evaluate_translation.py --adapter "" --max-samples 500 \
  --output-file metrics/gemma_4b_base_eval.json
```

**Go/no-go gates:**
| Metric | Minimum | Target |
|--------|---------|--------|
| BLEU improvement vs base | > +2 pts | > +4 pts |
| Theological term accuracy | > 65% (5/8) | > 80% (7/8) |
| chrF++ improvement | > +2 pts | > +5 pts |

**If gates fail:** Scale up with `--max-pairs 50000` and retrain.

---

## Step 2: Quality Assessment (Phase F)

**Goal:** Sample 100 pseudo-labeled transcripts, manually transcribe them, compute baseline WER. Determines filtering strategy for Whisper LoRA.

**Can run in parallel with Step 1** (no GPU needed for sampling, human transcription is the bottleneck).

**Script:** `training/assess_quality.py`
**Data:** 13,563 transcript JSONs at `/mnt/d/Data/stt-data/stark_data/transcripts/`

```bash
# Sample 100 segments stratified by confidence
python training/assess_quality.py smart-sample \
  --input /mnt/d/Data/stt-data/stark_data/transcripts \
  --n 100

# Manual: listen to each clip, type correct transcription in "manual_text" field
# This is ~2-4 hrs of human work (the bottleneck)

# Compute baseline WER
python training/assess_quality.py evaluate \
  --input assessment_smart_sample.jsonl
```

**Decision matrix:**
| Baseline WER | Action for Whisper LoRA |
|-------------|------------------------|
| < 10% | Use transcripts directly with confidence filtering |
| 10–20% | Filter worst segments by `avg_logprob` before training |
| 20–30% | Weakly supervised pretraining first |
| > 30% | Re-transcribe with `whisper-large-v3` (non-turbo) |

---

## Step 3: Whisper LoRA Fine-Tuning (Phase H)

**Goal:** Adapt Distil-Whisper to church acoustics, speaker patterns, and theological vocabulary.

**Prerequisites:** Step 2 complete (determines filtering strategy)

**Script:** `training/train_whisper.py`
**Data:** 13,562 clean audio chunks + pseudo-labels

```bash
# If Phase F says WER > 10%, filter first:
# python training/filter_transcripts.py \
#   --input /mnt/d/Data/stt-data/stark_data/transcripts \
#   --min-logprob -0.7 \
#   --output /mnt/d/Data/stt-data/stark_data/transcripts_filtered

nohup python training/train_whisper.py \
  --dataset /mnt/d/Data/stt-data/stark_data/transcripts/ \
  --output fine_tuned_whisper_cycle1 \
  --model distil-whisper/distil-large-v3.5 \
  --lora-r 32 \
  --lora-alpha 64 \
  --epochs 3 \
  --replay-ratio 0.3 \
  > whisper_lora.log 2>&1 &
```

**Config:** r=32, alpha=64, target q_proj+v_proj, 70% church + 30% LibriSpeech anti-forgetting
**Duration:** ~5-8 hrs GPU, ~8-10 GB VRAM
**Expected:** 10-30% relative WER reduction on church audio

---

## Step 4: TranslateGemma 12B QLoRA (Phase I continued)

**Goal:** Train the larger 12B model for higher-quality final translations. At inference, 4B serves as speculative draft and 12B produces finals.

**Prerequisites:** Step 1 passes go/no-go (BLEU > +2). Cannot run concurrent with Step 3 (combined VRAM > 16 GB).

```bash
nohup python training/train_gemma.py B \
  --max-pairs 20000 \
  > gemma_12b.log 2>&1 &
```

**Duration:** ~18-27 hrs GPU, ~14-15 GB VRAM (tight on A2000)
**Risk:** May OOM. If so: reduce `--max-pairs 10000` or `--lora-r 8`

---

## Step 5: Transfer Adapters to Mac + Live A/B

**Goal:** Deploy working adapters to Mac for inference testing.

**Prerequisites:** Steps 1, 3, and/or 4 complete

```bash
# From WSL -> Mac
rsync -av fine_tuned_gemma_mi_A/ mac:~/code/stark-translate/adapters/gemma/4b_cycle1/
rsync -av fine_tuned_whisper_cycle1/ mac:~/code/stark-translate/adapters/whisper/cycle1/

# On Mac — run live A/B with adapter
python dry_run_ab.py --lang en --adapter adapters/gemma/4b_cycle1
```

**Deployment pipeline:** Full 6-phase deploy described in `docs/deploy.md` (not yet implemented as scripts — manual rsync for now).

---

## Execution Order

```
Step 1 (eval Gemma 4B)  ─────────────┐
                                      ├── Step 4 (Gemma 12B, if Step 1 passes)
Step 2 (quality assessment) ──┐       │
                              ├───── Step 3 (Whisper LoRA)
                              │       │
                              │       └── Step 5 (transfer + live A/B)
         [human transcription]│
```

Steps 1 and 2 can run **in parallel** (Step 1 uses GPU, Step 2 is mostly human work).
Steps 3 and 4 must run **sequentially** (both need the GPU).

**Total estimated time:**
- GPU: ~30-40 hrs (eval ~2h + Whisper ~8h + Gemma 12B ~25h)
- Human: ~3-5 hrs (quality assessment transcription)

---

## Verification

After each step, check against go/no-go gates in `docs/training_plan_march.md`:
- **Gemma eval:** BLEU > +2, theological accuracy > 65%
- **Whisper eval:** WER relative improvement > 10%
- **Stop condition:** Improvement < 2% relative for 2 consecutive cycles

Monitor file: `training_monitor.log` (10-min interval VRAM/progress logging)
