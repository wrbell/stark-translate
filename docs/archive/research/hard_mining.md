# Hard Example Mining for Whisper LoRA (W15)

## Goal

Push WER lower on the failure cases that survive W7's training, without re-training the easy majority. Curriculum learning from a strong base, focused on the audio + text patterns that the current best adapter still gets wrong.

**Target:** 0.5–1.5% absolute WER reduction overall, with >2% absolute reduction on theological-term chunks specifically.

**Anti-goal:** Catastrophic forgetting of W7's general improvements. We want to *extend* W7, not replace it.

---

## Why hard mining now

- **W7 is the current winner** (5.63% on midwest eval, 7.26% on fresh eval).
- **W12 (1 epoch on 198K)** lost to W7 by ~2% on the fresh eval. More data + single pass made things worse, not better.
- **W13 (3 epochs on top 17K stt-data)** is testing data quality. Result will inform W14/W15 priorities.
- **Hard mining is independent of data scaling.** Even if W14 (combined 50K) wins, hard mining can stack on top.

The intuition: most chunks are *already correct*. Training the model on chunks it already gets right wastes gradient steps. Training on chunks it gets *wrong* concentrates learning where it matters.

---

## Definition of "hard"

A chunk is **hard** for adapter X if X's transcription has a normalized WER ≥ 15% against the Deepgram ground truth.

A chunk is **theologically critical** if its Deepgram text contains any Tier 1 boost term (50 phrases from `bible_data/glossary/tier1_boost.json`).

A chunk is **selected for W15** if:
- `WER ≥ 15%` (the model is making real errors), AND
- (`WER < 80%` OR `theologically_critical`) — exclude pure noise chunks unless they contain target vocabulary.

The 80% upper bound prevents the model from training on garbage chunks (silence, cross-talk, mic dropout) where Deepgram and Whisper both hallucinate.

### Why WER, not loss

- **Loss is per-token, WER is per-word.** WER reflects what users perceive; loss can be low for fluent-but-wrong predictions.
- **Loss is dominated by easy tokens.** A 50-token chunk where W7 nails 49 has a low average loss even if the one wrong token was a critical theological term.
- **WER is what we report.** Mining on WER gives the most direct signal for the metric we care about.

---

## Mining algorithm

```
For each adapter A in {W7, W14_winner}:
  Load A on top of base Whisper turbo
  For each source sermon S:
    Load full WAV (memory-mapped via soundfile)
    For each chunk C in chunks_for(S):
      audio_segment = WAV[C.start*sr : C.end*sr]
      mel = WhisperProcessor(audio_segment)
      with torch.inference_mode():
        prediction = A.generate(mel, language='en', task='transcribe', ...)
      reference = deepgram_text(C)
      record = {
        source: S,
        chunk_idx: C.idx,
        start: C.start, end: C.end,
        prediction: prediction,
        reference: reference,
        wer: jiwer.wer(normalize(reference), normalize(prediction)),
        has_tier1: any(term in reference.lower() for term in tier1_terms),
        adapter: A.name,
      }
      append record to JSONL
```

### Streaming output

Write one JSONL line per chunk, flushed every 100 chunks. If the script crashes mid-mine, resume by skipping records already written.

```
mining/
  w7_mined.jsonl     ← 215K records, ~80 MB
  w14_mined.jsonl    ← 215K records, ~80 MB (only if W14 wins)
```

### Batched inference

Single-chunk inference is ~0.4 s/chunk. Batching to 8 chunks/call gives ~3× speedup:
- Group chunks by similar duration to minimize padding waste
- Pad to longest chunk in batch (up to 30s ≈ 3000 mel frames)
- Use `model.generate(input_features=batch, ...)`
- Decode + compute per-chunk WER on CPU after each batch

**Estimated mining time at 0.13 s/chunk effective:** 215K × 0.13 = ~7.8 hrs.

---

## Pool composition

The mining pool combines all available aligned chunks across the 363 sermons:

| Source | Aligned chunks | Notes |
|--------|---------------|-------|
| midwest (33 sermons) | ~22K | high-quality soundboard |
| stt-data non-midwest (~283 sermons) | ~193K | YouTube, variable quality |
| **Total mining pool** | **~215K** | |

*Important*: do not mine on the **fresh eval set** sermons (4 post-cutoff sermons, 2,706 examples). Those are off-limits.

---

## Selection thresholds

After mining produces `w7_mined.jsonl`, build the hard subset:

```python
hard = [r for r in records if r.wer >= 0.15 and r.wer < 0.80]
critical = [r for r in records if r.has_tier1]
hard_critical = list({r.id: r for r in hard + critical}.values())  # union, dedup

# Stratify by source: cap each sermon's contribution at 5% of subset
balanced = stratified_cap(hard_critical, max_per_source=N//20)

# Sort by WER descending (hardest first), keep top N
selected = sorted(balanced, key=-WER)[:N]
```

**Target subset size**: 8,000–12,000 chunks
- Small enough for fast training (~3 hrs at 1 epoch)
- Large enough for diverse coverage
- Each sermon contributes ≤5% (max ~500 chunks per sermon) to prevent overfitting to one bad-audio sermon

### Expected distribution

Based on W7's 7.26% overall WER on the fresh eval, extrapolating to the training pool:
- ~5% of 215K = **~10,800 chunks with WER ≥ 15%** (the "hard" set)
- ~12% of 215K = **~25,800 chunks with Tier 1 terms** (the "critical" set)
- Union (after dedup) ≈ **~30K candidates**
- After stratified cap and top-N selection: **~10K final**

If the hard set is much smaller (e.g., <3K), the model is already saturated and W15 has limited upside. If much larger (e.g., >30K), W7 is undertrained and we should re-run W7 with more epochs first.

---

## Training strategy

Three options, ordered by expected risk-adjusted value:

### Option A: Continue W7 (recommended)
Initialize W15 from W7's adapter weights, train 1 additional epoch on the hard subset.

```bash
python training/train_whisper.py \
    --dataset stark_data/whisper_dataset_w15_hard \
    --lr 5e-5 --epochs 1 \
    --target-modules q_proj v_proj \
    --lora-r 32 --lora-alpha 64 \
    --replay-ratio 0.5 \
    --init-from whisper_ablation/W7_3epochs \
    -o whisper_ablation/W15_hard_w7
```

**Why lr=5e-5 (half of W7's 1e-4)**: smaller dataset, smaller learning rate to prevent overshooting. Standard recipe for fine-tuning a fine-tuned model.

**Why replay=0.5 (vs 0.3 in W7)**: higher replay ratio to anchor against forgetting. The hard subset is biased toward failure cases — without strong replay, the model could become *good at hard cases but worse on easy ones*.

**Implementation note**: `train_whisper.py` does not currently support `--init-from`. We need to add a flag that loads adapter weights from another directory while starting fresh optimizer state. Two-line patch:

```python
# In fine_tune_whisper() after get_peft_model():
if init_from is not None:
    from peft import PeftModel
    state = PeftModel.from_pretrained(base_model, init_from).state_dict()
    model.load_state_dict(state, strict=False)
```

### Option B: Cold start on hard subset
Train a fresh adapter from base Whisper on hard examples only.

**Why this is bad**: the model never sees the easy patterns that make a transcription *fluent*. It will be a specialist on hard cases but worse on the 95% of normal speech.

**Use only as a baseline**: if Option A reaches 4% WER and Option B reaches 8%, we know the fine-tuning is working as expected (continuation > cold start).

### Option C: Mix hard + W7 random sample
Build a dataset that's 70% hard examples + 30% random easy examples from W7's training set, train from W7 init.

**Tradeoff vs Option A**: more diverse but more compute and more risk of overfitting to the hard examples since they're the majority. The 0.5 replay ratio in Option A already provides this anchoring through LibriSpeech.

**Skip unless Option A regresses general WER**: this is the fallback if we see catastrophic forgetting in the W15 eval.

---

## Implementation: scripts to write

### 1. `training/mine_hard_examples.py` (new)

```
Args:
  --adapter PATH         (e.g., whisper_ablation/W7_3epochs)
  --chunks-json PATH     (e.g., ablation/sermon_whisper_chunks_w14_combined.json)
  --deepgram-dir PATH    (e.g., /mnt/d/Data/stt-data/deepgram_transcripts)
  --audio-dir PATH       (e.g., stt-data)
  --output-jsonl PATH    (e.g., whisper_ablation/w7_mined.jsonl)
  --tier1-glossary PATH  (default: bible_data/glossary/tier1_boost.json)
  --batch-size INT       (default: 8)
  --resume               (skip records already in output JSONL)

Behavior:
  - Load adapter on top of base Whisper turbo (fp16, eval mode)
  - For each chunk: extract audio segment, run inference, compute WER
  - Write JSONL line per chunk
  - Resume support: if output exists, skip processed (source, chunk_idx) pairs
  - Memory cap: process one source at a time, never hold full sermon WAV
  - Log progress every 1000 chunks with running WER and time
```

### 2. `training/build_hard_subset.py` (new)

```
Args:
  --mined-jsonl PATH         (input from mine_hard_examples.py)
  --chunks-json PATH         (whisper chunks JSON to build subset from)
  --output-chunks-json PATH  (filtered subset chunks JSON)
  --wer-min FLOAT            (default: 0.15)
  --wer-max FLOAT            (default: 0.80)
  --target-size INT          (default: 10000)
  --max-per-source INT       (default: target_size // 20)
  --include-tier1            (always include chunks with Tier 1 terms)

Behavior:
  - Load mined records
  - Filter by WER bounds
  - Compute Tier 1 indicator
  - Stratify by source (cap per-source contribution)
  - Sort by WER descending
  - Take top N
  - Write filtered chunks JSON for align_deepgram_chunks.py to consume
```

### 3. `training/train_whisper.py` (modify)

Add `--init-from PATH` flag that loads adapter state from another directory before training starts. ~10 lines.

### 4. End-to-end pipeline

```bash
# Step 1: Mine (one-shot, ~8 hrs)
python training/mine_hard_examples.py \
    --adapter whisper_ablation/W7_3epochs \
    --chunks-json ablation/sermon_whisper_chunks_w14_combined.json \
    --deepgram-dir /mnt/d/Data/stt-data/deepgram_transcripts \
    --audio-dir stt-data \
    --output-jsonl whisper_ablation/w7_mined.jsonl

# Step 2: Build hard subset (~5 min)
python training/build_hard_subset.py \
    --mined-jsonl whisper_ablation/w7_mined.jsonl \
    --chunks-json ablation/sermon_whisper_chunks_w14_combined.json \
    --output-chunks-json ablation/sermon_whisper_chunks_w15_hard.json \
    --target-size 10000

# Step 3: Align hard subset → preprocessed cache (~30 min)
python training/align_deepgram_chunks.py \
    --whisper-chunks ablation/sermon_whisper_chunks_w15_hard.json \
    --deepgram-dir /mnt/d/Data/stt-data/deepgram_transcripts \
    --audio-dir stt-data \
    --output /mnt/d/Data/stt-data/whisper_dataset_w15_hard \
    --preprocess-cache

# Step 4: Train W15 from W7 init (~3 hrs)
python training/train_whisper.py \
    --dataset /mnt/d/Data/stt-data/whisper_dataset_w15_hard \
    --lr 5e-5 --epochs 1 \
    --target-modules q_proj v_proj \
    --lora-r 32 --lora-alpha 64 \
    --replay-ratio 0.5 \
    --init-from whisper_ablation/W7_3epochs \
    -o whisper_ablation/W15_hard_w7

# Step 5: Eval (~17 min)
python training/eval_whisper_wer.py \
    --adapter whisper_ablation/W15_hard_w7 \
    --eval-set stark_data/eval_fresh_dataset
```

---

## Runtime estimates

| Step | Time | VRAM | Disk |
|------|------|------|------|
| Mine W7 on 215K pool (batched, fp16) | **~8 hrs** | ~5 GB | ~80 MB JSONL |
| Build hard subset (sort + filter) | ~5 min | CPU | ~5 MB JSON |
| Align hard subset (~10K chunks → cache) | ~30 min | CPU | ~10 GB cache |
| Train W15 (1 epoch, ~10K, replay) | **~3 hrs** | ~9.5 GB | ~2 GB adapter |
| Eval W15 on fresh eval set | ~17 min | ~3 GB | — |
| **Total W15 cycle** | **~12 hrs** | | |

If both W7 and W14 are mined (one after the other for a comparison):
- **Total: ~24 hrs** for two W15 variants

---

## Memory & OOM considerations

### Mining script
- One source WAV at a time (memory-mapped via `soundfile.read(start=, stop=)`)
- Per-chunk audio segment: ~1 MB max (30s × 16kHz × 2 bytes)
- Batch of 8 mel spectrograms: ~8 MB
- Predictions buffer: tiny (text strings)
- **Peak RAM: ~2 GB**
- **VRAM: ~5 GB** (Whisper turbo fp16 + LoRA + batch)

### Subset build script
- Loads JSONL records (~80 MB) into memory
- Sorts and filters
- **Peak RAM: ~500 MB**

### Cache build & training
- Identical to W13/W14 patterns. Sharded streaming generator caps RAM at ~5 GB.
- Hard subset is ~10K chunks = ~10 shards = trivial.

**No OOM risks beyond what's already understood from W12/W13/W14 runs.**

---

## Risks & mitigations

### Risk 1: Mining identifies "hard" examples that are actually bad data
**Symptom**: many "hard" chunks contain Deepgram hallucinations on silent or noisy audio.
**Mitigation**:
- Upper WER bound (`wer < 0.80`) excludes garbage
- Manual spot-check of top 50 hardest chunks before training
- Inspect `prediction` vs `reference` columns in mined JSONL
- If >20% of "hard" examples are garbage, raise the upper bound or add an SNR filter

### Risk 2: Catastrophic forgetting on easy examples
**Symptom**: W15 fresh eval WER is *worse* than W7, even though W15 trained on W7's failures.
**Mitigation**:
- Replay ratio 0.5 (vs 0.3 in W7)
- Single epoch only
- Lower learning rate (5e-5 vs 1e-4)
- Eval W15 on the OLD midwest eval set too — if midwest WER regresses by >1% absolute, kill W15 and try Option C (mixed easy + hard)

### Risk 3: Overfitting to a few bad-audio sermons
**Symptom**: 80% of "hard" chunks come from 5 sermons with terrible mic quality.
**Mitigation**: stratified per-source cap (`max_per_source = target_size // 20 = 500`).

### Risk 4: Init-from-adapter doesn't work
**Symptom**: training starts with high loss as if from scratch.
**Mitigation**:
- Verify adapter weights are loaded by checking the first batch's loss before training (~0.1 if loaded correctly, ~1.0+ if not)
- Fallback: copy W7 checkpoint dir to W15 output, use `--resume` (this re-uses optimizer state too — slightly less clean but works)

### Risk 5: Mining is too slow
**Symptom**: 8 hrs estimate stretches to 20+ hrs.
**Mitigation**:
- Reduce pool size: mine only on top 100K chunks by Whisper logprob (skip the obvious junk)
- Increase batch size to 16 (if VRAM allows after eval-mode loading)
- Use chunked Whisper inference mode (faster but lower precision — acceptable for mining)

### Risk 6: Hard subset is too small or too theological-skewed
**Symptom**: only 2K chunks meet criteria, or 8K of 10K contain Tier 1 terms.
**Mitigation**:
- Lower WER threshold to 0.10 if too few hard chunks
- Cap Tier 1 contribution at 50% of selected to maintain general WER

---

## Decision logic

### When to run W15

Run W15 **after** W14 completes (regardless of W14 result).

| W14 vs W7 | W15 priority | Reason |
|-----------|-------------|--------|
| W14 < W7 (W14 wins) | HIGH — mine on W14 | W14 is the new best, mine its failures |
| W14 ≈ W7 (tie) | HIGH — mine on W7 | W7 is simpler, equal quality |
| W14 > W7 (W14 worse) | HIGH — mine on W7 | W7 is still the best |

### Success criteria for W15

| Metric | Floor | Target | Kill |
|--------|-------|--------|------|
| W15 fresh WER vs W7 fresh WER | ≤ W7 (no regression) | ≥ 0.5% absolute reduction | > 0.5% absolute regression |
| W15 fresh WER on Tier 1 chunks | ≤ W7 | ≥ 2% absolute reduction | > 2% absolute regression |
| W15 old midwest WER vs W7 old WER | within +0.5% (no big regression) | matches W7 | > 1% absolute regression |
| Wall time | < 12 hrs | < 8 hrs | > 24 hrs |

If W15 ships, the new pipeline is: **W7 → W15** (or **W14 → W15** if W14 won).

---

## Future variants (post-W15)

Once W15 is established as a recipe, the same machinery enables:

### W16: Iterative hard mining
Mine W15's failures, build W16 hard subset, train W16 from W15 init. Diminishing returns expected, but worth a single iteration.

### W17: Multi-adapter ensemble
Mine W7 *and* W15 separately. Find chunks where they disagree (high "uncertainty"). Use those as the W17 hard subset — train on cases where two models can't agree.

### W18: Cross-adapter distillation
Use W14 (large data) as the *teacher*: its predictions on the full pool become pseudo-labels for hard chunks. Train W7 to match W14's predictions on chunks where W7 is wrong but W14 is right. This propagates W14's strength back into W7's tighter recipe.

---

## Summary

| Aspect | Decision |
|--------|---------|
| **Mining metric** | Per-chunk WER vs Deepgram, computed via batched fp16 inference |
| **Selection threshold** | WER ∈ [0.15, 0.80], OR Tier 1 term present |
| **Subset size** | 10K, stratified per source (≤500/sermon) |
| **Init strategy** | Continue from W7 adapter, fresh optimizer state |
| **Hyperparameters** | lr=5e-5, epochs=1, replay=0.5, r=32, qv |
| **Eval** | Fresh eval set + old midwest eval (regression check) |
| **Total time** | ~12 hrs (8 mine + 30 align + 3 train + 17 min eval) |
| **Required new code** | `mine_hard_examples.py`, `build_hard_subset.py`, `--init-from` flag in `train_whisper.py` |
| **Risk profile** | LOW — small dataset, short training, well-understood failure modes |
