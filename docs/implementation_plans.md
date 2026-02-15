# Implementation Plans — Practical Execution Guide

> **Complements:** [`roadmap.md`](roadmap.md) (what) + this document (how)
>
> **Last updated:** 2026-02-14

This document provides concrete implementation details for each phase in the roadmap. It covers the practical "how" — scripts to run, VRAM budgets, code changes needed, quality gates, and expected timelines.

---

## Phase 1: Training & Validation

> Already well-documented in existing docs. Summary of key execution steps here.

### 1A. Seattle Training Run

See [`seattle_training_run.md`](seattle_training_run.md) for the full 6-day unattended run design.

**Pre-flight checklist (Windows/WSL):**
```bash
# Verify CUDA
python -c "import torch; print(torch.cuda.get_device_name(0))"

# Verify bitsandbytes
python -c "import bitsandbytes as bnb; print(bnb.__version__)"

# Verify data
wc -l bible_data/aligned/verse_pairs_train.jsonl  # expect ~266K lines
ls stark_data/raw/*.wav | wc -l                    # expect 30+ files
```

**Critical training script fixes needed before running** (identified by audit):

1. **`train_whisper.py` — Missing `compute_metrics` function:**
   The trainer has no WER evaluation during training. Add:
   ```python
   import evaluate
   metric = evaluate.load("wer")

   def compute_metrics(pred):
       pred_ids = pred.predictions
       label_ids = pred.label_ids
       label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
       pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
       label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
       wer = 100 * metric.compute(predictions=pred_str, references=label_str)
       return {"wer": wer}
   ```

2. **`train_whisper.py` — Missing model config overrides:**
   ```python
   model.config.forced_decoder_ids = None
   model.config.suppress_tokens = []
   model.config.use_cache = False
   model.generation_config.language = "english"
   model.generation_config.task = "transcribe"
   ```

3. **`train_whisper.py` — Wrong data collator:**
   Replace `DataCollatorForSeq2Seq` with a custom `DataCollatorSpeechSeq2SeqWithPadding` that handles audio feature padding and -100 label masking correctly. See the [HuggingFace Whisper fine-tuning guide](https://huggingface.co/blog/fine-tune-whisper) for the reference implementation.

4. **`train_whisper.py` — Add weight_decay and cosine schedule:**
   ```python
   weight_decay=0.01,
   lr_scheduler_type="cosine",
   metric_for_best_model="wer",
   greater_is_better=False,
   ```

5. **`train_gemma.py` — Fix fallback chat template format:**
   Replace the `<start_of_turn>` Gemma 2-style format with the community-validated TranslateGemma format:
   ```python
   json_payload = json.dumps([{
       "type": "text",
       "source_lang_code": "en",
       "target_lang_code": "es",
       "text": example["en"]
   }], ensure_ascii=False)
   formatted = f"user\n{json_payload}\nmodel\n{example['es']}"
   ```

6. **`train_gemma.py` — Consider `Gemma3ForCausalLM`:**
   `AutoModelForCausalLM` may hit a `vocab_size` config issue with TranslateGemma's Gemma 3 architecture. If it fails, use:
   ```python
   from transformers import Gemma3ForCausalLM
   model = Gemma3ForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, device_map="auto")
   ```

7. **`train_gemma.py` — Copy `preprocessor_config.json` after training:**
   Known TRL bug (#3110) — manually copy from base model to output directory.

8. **Missing audiofolder conversion script:**
   `transcribe_church.py` outputs per-file JSONs, but `train_whisper.py` expects HuggingFace `audiofolder` format. The new `prepare_whisper_dataset.py` fills this gap.

### 1B. LoRA to MLX Conversion

**Steps:**
```bash
# 1. Merge LoRA into base model (on Windows)
python -c "
from peft import PeftModel
from transformers import AutoModelForCausalLM
base = AutoModelForCausalLM.from_pretrained('google/translategemma-4b-it')
model = PeftModel.from_pretrained(base, 'fine_tuned_gemma_mi_A/')
merged = model.merge_and_unload()
merged.save_pretrained('merged_gemma_4b_mi/')
"

# 2. Transfer merged model to Mac
scp -r merged_gemma_4b_mi/ mac:~/Code/vibes/SRTranslate/

# 3. Convert to MLX 4-bit on Mac
python -m mlx_lm.convert \
  --hf-path merged_gemma_4b_mi/ \
  -q --q-bits 4 \
  --upload-repo wrbell/translategemma-4b-mi-4bit  # optional
```

**Verification:** Run 10 test sentences through both the merged PyTorch model and the MLX-converted model, compare outputs for equivalence.

**Time:** ~30 min per model (dominated by quantization).

### 1C. Active Learning Cycles

**Cycle structure (each ~3-7 days):**

```
Day 1:   Run live inference → collect diagnostics JSONL + audio chunks
Day 2-4: Human review of flagged segments (bottom 5-15% by confidence)
Day 5:   Transfer corrections to Windows, retrain overnight
Day 6:   Convert → deploy updated model
Day 7:   Evaluate improvement, identify next cycle's focus
```

**Flagging criteria (from `dry_run_ab.py` diagnostics):**

| Signal | Flag | Auto-reject |
|--------|------|-------------|
| `avg_logprob` | < -0.5 | < -1.0 |
| `no_speech_prob` | > 0.3 | > 0.6 |
| `compression_ratio` | > 2.0 | > 2.4 (hallucination) |
| `qe_score` (CometKiwi) | < 0.7 | < 0.5 |
| `marian_similarity` | < 0.6 | < 0.4 |

**Expected convergence:** 3-5 cycles. First cycle typically yields 20-40% relative WER reduction. Diminishing returns after cycle 3.

### 1D. Accent-Diverse STT Tuning

See [`accent_tuning_plan.md`](accent_tuning_plan.md) — fully documented with code complete.

---

## Phase 2: Accuracy Deepening

### 2A. Auto-Collection from Live Inference

**Goal:** Automatically collect high-quality EN→ES translation pairs from live church services for continuous training data enrichment.

**Architecture:**

```
Live Pipeline (dry_run_ab.py)
    │
    ├─ STT confidence (avg_logprob, compression_ratio)
    ├─ Translation QE (CometKiwi score)
    ├─ MarianMT agreement (cosine similarity)
    │
    ▼
Composite Quality Score
    │
    ├─ >= 0.85 → AUTO-ACCEPT → sermon_pairs.jsonl
    ├─ 0.70-0.85 → MODERATE → needs spot-check
    ├─ 0.50-0.70 → REVIEW → human_review_queue.jsonl
    └─ < 0.50 → REJECT → logged but not used
```

**Composite quality scoring formula:**
```python
def composite_score(stt_confidence, translation_qe, marian_agreement):
    """
    Weighted composite quality score for auto-collection.

    Weights reflect: translation QE is most predictive of pair quality,
    STT confidence catches acoustic failures, Marian agreement is a
    cheap cross-check signal.
    """
    score = (
        0.45 * translation_qe +      # CometKiwi (0-1)
        0.35 * stt_confidence +       # normalized avg_logprob (0-1)
        0.20 * marian_agreement       # cosine similarity (0-1)
    )
    return score
```

**STT confidence normalization:**
```python
def normalize_stt_confidence(avg_logprob):
    """Map avg_logprob (-inf, 0] to [0, 1] with -0.5 as midpoint."""
    # Logistic sigmoid centered at -0.3
    return 1.0 / (1.0 + math.exp(-10 * (avg_logprob + 0.3)))
```

**Exclusion filters (applied before scoring):**
```python
EXCLUDE_IF = {
    "compression_ratio": lambda x: x > 2.4,        # hallucination
    "no_speech_prob": lambda x: x > 0.6,            # silence
    "text_length": lambda x: len(x.split()) < 3,    # too short
    "untranslated": lambda x: detect_untranslated(x), # copy-through
    "length_ratio": lambda x: not (0.8 < x < 2.5),  # length anomaly
}
```

**Storage format (`sermon_pairs.jsonl`):**
```json
{
  "en": "For God so loved the world that he gave his only begotten Son",
  "es": "Porque de tal manera amo Dios al mundo, que ha dado a su Hijo unigenito",
  "source": "live_session",
  "session_id": "2026-02-15_10-30",
  "quality_score": 0.91,
  "stt_confidence": 0.88,
  "translation_qe": 0.93,
  "marian_agreement": 0.85,
  "timestamp": "2026-02-15T10:32:15Z"
}
```

**Expected yield:** ~150-200 auto-accepted pairs per hour of live church use (at the >= 0.85 threshold). A typical 90-minute service yields ~225-300 pairs. After 10 services: ~2,500-3,000 curated real-domain pairs.

**Calibration:** Run CometKiwi on 200-500 manually-verified pairs from the first few sessions. If auto-accept rate is too high (> 80%), tighten threshold to 0.88. If too low (< 30%), loosen to 0.82. Target: 40-60% auto-accept rate.

**Implementation changes to `dry_run_ab.py`:**
1. Add `composite_score()` calculation after each final translation
2. Write high-quality pairs to `stark_data/live_sessions/sermon_pairs.jsonl`
3. Write moderate/review pairs to separate files for later triage
4. Add a session summary at end: total pairs collected, quality distribution, new theological terms

**Distribution shift mitigation:**
Live sermon data will be heavily biased toward the primary speaker's topics and vocabulary. Implement three-tier collection:
- **Tier 1:** High-confidence auto-accepts (backbone of training data)
- **Tier 2:** Actively seek diversity — weight novel vocabulary higher, downweight repeated phrases
- **Tier 3:** Periodically add general-domain pairs (5-10% from OPUS/Tatoeba) to prevent overfitting to a single speaker

### 2B. Iterative Fine-Tuning with Sermon Pairs

**Goal:** Fold auto-collected sermon pairs into Bible corpus for translation retraining.

**Training data mix:**
```
bible_data/aligned/verse_pairs_train.jsonl   ~266K pairs (formal biblical register)
stark_data/live_sessions/sermon_pairs.jsonl   Variable (spoken sermon register)
bible_data/glossary/glossary_pairs.jsonl      ~458 pairs (10-20x oversampled)
```

**Recommended mixing strategy:**
- **First retraining (after ~2K sermon pairs):** 70% Bible, 25% sermon, 5% glossary
- **Subsequent retrainings:** Shift toward 50% Bible, 40% sermon, 10% glossary as sermon corpus grows
- **Loss weighting:** Apply 2x loss weight on sermon pairs and glossary terms (newer, higher-value data)

**Practical steps:**
```bash
# 1. Merge all training data
python -c "
import json
pairs = []
# Bible corpus
with open('bible_data/aligned/verse_pairs_train.jsonl') as f:
    for line in f: pairs.append(json.loads(line))
# Sermon pairs
with open('stark_data/live_sessions/sermon_pairs.jsonl') as f:
    for line in f: pairs.append(json.loads(line))
# Glossary (oversampled)
with open('bible_data/glossary/glossary_pairs.jsonl') as f:
    glossary = [json.loads(line) for line in f]
pairs.extend(glossary * 15)  # 15x oversample
# Shuffle and write
import random; random.shuffle(pairs)
with open('training_data/mixed_train.jsonl', 'w') as f:
    for p in pairs: f.write(json.dumps(p, ensure_ascii=False) + '\n')
"

# 2. Retrain (overnight on A2000)
python training/train_gemma.py A \
  --data training_data/mixed_train.jsonl \
  --epochs 2 \
  --lr 1e-4  # Lower LR for incremental training
```

**Verse grouping for discourse coherence:**
Research shows individual verses (15-30 tokens) don't teach cross-sentence coherence. Group 3-5 consecutive verses:
```python
def group_verses(pairs, group_size=4):
    """Group consecutive verses into passages for discourse-level training."""
    grouped = []
    # Sort by verse_id to get consecutive verses
    sorted_pairs = sorted(pairs, key=lambda p: p.get("verse_id", ""))
    for i in range(0, len(sorted_pairs) - group_size + 1, group_size):
        chunk = sorted_pairs[i:i+group_size]
        en_passage = " ".join(p["en"] for p in chunk)
        es_passage = " ".join(p["es"] for p in chunk)
        grouped.append({"en": en_passage, "es": es_passage, "type": "passage"})
    return grouped
```

Mix 50% verse-level + 50% passage-level pairs in training data.

**Expected improvement:** +3-7% BLEU on theological text per retraining cycle. Diminishing returns after 3 cycles.

**Data sizing recommendation:**
Research shows TranslateGemma already has strong translation capability — it needs domain adaptation, not re-learning translation. Start with 10K-30K high-quality pairs (not the full 155K). Scale up only if improvement plateaus. 155K pairs for 3 epochs risks memorizing biblical phrasing rather than learning generalizable theological patterns.

**Archaic register risk:**
KJV (1611), ASV (1901), YLT (1898), BBE (1949) paired with RVR1909 are predominantly archaic/formal texts. Risk: model learns to produce overly formal Spanish for modern spoken input.
- **Mitigation:** Prioritize WEB x Espanol Sencillo (most modern pair) at 60-70% of Bible data
- **Mitigation:** Mix in 5K-10K general-domain modern EN-ES pairs from OPUS/Tatoeba
- **Mitigation:** Monitor COMET on both biblical AND modern test text — flag any regression on modern register

### 2C. Hybrid STT-Translation Tuning

**Goal:** Make the translation model robust to ASR errors, and make the STT model more accurate on theological terms.

Three complementary approaches, executed sequentially:

#### Approach 1: Dictionary-Based Post-Correction (~0 training cost)

**Concept:** A lightweight correction layer between Whisper output and TranslateGemma input. Catches common ASR misrecognitions of theological terms.

```python
THEOLOGICAL_CORRECTIONS = {
    # phonetic confusion pairs: wrong → correct
    "proposition": "propitiation",
    "atone meant": "atonement",
    "pre destination": "predestination",
    "expy asian": "expiation",
    "sanctions": "sanctification",
    "just a fiction": "justification",
    "inter session": "intercession",
    "trans substantiation": "transubstantiation",
    "evangelical ism": "evangelicalism",
    "sotter iology": "soteriology",
    "escatology": "eschatology",
    "pneuma tology": "pneumatology",
    "christology": "Christology",
    "saint of vacation": "sanctification",
    # Biblical names
    "nebula can answer": "Nebuchadnezzar",
    "malachite": "Malachi",
    "habba cook": "Habakkuk",
    "zero babel": "Zerubbabel",
}

def correct_theological_terms(text):
    """Context-aware dictionary correction. <1ms, zero training cost."""
    corrected = text
    for wrong, right in THEOLOGICAL_CORRECTIONS.items():
        # Case-insensitive replacement
        corrected = re.sub(re.escape(wrong), right, corrected, flags=re.IGNORECASE)
    return corrected
```

**Integration:** Add to `dry_run_ab.py` between STT output and translation input. Log corrections to diagnostics for dictionary expansion.

**Effectiveness:** Catches ~30-50% of theological term errors. Population grows over time from live session diagnostics (homophone flags).

#### Approach 2: Weighted Whisper Fine-Tuning on Theological Terms (~2-4 GPU hours)

**Concept:** Apply higher cross-entropy loss weight to theological vocabulary tokens during Whisper LoRA training.

```python
class TheologicalWeightedTrainer(Seq2SeqTrainer):
    def __init__(self, theological_token_ids, term_weight=3.0, **kwargs):
        super().__init__(**kwargs)
        self.theological_ids = set(theological_token_ids)
        self.term_weight = term_weight

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        # Standard cross-entropy
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        flat_logits = logits.view(-1, logits.size(-1))
        flat_labels = labels.view(-1)
        per_token_loss = loss_fct(flat_logits, flat_labels)

        # Apply higher weight to theological tokens
        weights = torch.ones_like(flat_labels, dtype=torch.float)
        for tid in self.theological_ids:
            weights[flat_labels == tid] = self.term_weight

        # Mask padding (-100)
        mask = flat_labels != -100
        weighted_loss = (per_token_loss * weights * mask).sum() / mask.sum()

        return (weighted_loss, outputs) if return_outputs else weighted_loss
```

**Theological token extraction:**
```python
# Extract token IDs for all 229 glossary terms
from transformers import WhisperTokenizer
tokenizer = WhisperTokenizer.from_pretrained("distil-whisper/distil-large-v3")
theological_ids = set()
for term in THEOLOGICAL_GLOSSARY.keys():
    ids = tokenizer.encode(term, add_special_tokens=False)
    theological_ids.update(ids)
# ~500-800 unique token IDs
```

**Expected improvement:** 15-25% relative WER reduction on theological terms specifically. Combined with dictionary correction: ~60-70% error reduction on theological vocabulary.

#### Approach 3: Pipeline-Aware TranslateGemma Training (~8-12 GPU hours)

**Concept:** Fine-tune TranslateGemma on Whisper's actual noisy outputs rather than clean English text. This teaches the translation model to handle ASR errors gracefully.

**Steps:**
```bash
# 1. Run Whisper on all training audio
python training/transcribe_church.py \
  --input stark_data/cleaned/chunks \
  --output stark_data/whisper_outputs \
  --backend faster-whisper

# 2. Create noisy-source training pairs
python -c "
import json, glob
pairs = []
for transcript in glob.glob('stark_data/whisper_outputs/*.json'):
    with open(transcript) as f:
        data = json.load(f)
    # Pair Whisper output (noisy EN) with reference translation (clean ES)
    # Need to match with corresponding Spanish reference...
    whisper_en = data['text']
    # Get clean reference from original Bible corpus alignment
    pairs.append({'en': whisper_en, 'es': reference_es, 'source': 'pipeline_aware'})
"

# 3. Mix pipeline-aware pairs with clean pairs
#    70% clean Bible pairs + 30% pipeline-aware pairs
```

**Synthetic ASR error augmentation** (alternative to running Whisper on everything):
```python
def inject_asr_errors(text, error_rate=0.05):
    """Inject phonetically-plausible ASR errors into clean text."""
    words = text.split()
    result = []
    for word in words:
        if random.random() < error_rate and word.lower() in ASR_CONFUSION_MAP:
            result.append(random.choice(ASR_CONFUSION_MAP[word.lower()]))
        else:
            result.append(word)
    return " ".join(result)

ASR_CONFUSION_MAP = {
    "propitiation": ["proposition", "propagation", "propitiation"],
    "atonement": ["atone meant", "a tone ment", "atonement"],
    "righteousness": ["righteous nest", "write just ness"],
    "sanctification": ["sanctions", "saint of vacation"],
    # ... extend from live session homophone logs
}
```

**Combined effectiveness:** All three approaches are complementary:
- Dictionary correction: immediate, ~30-50% of theological errors
- Weighted Whisper: better STT on theological terms, ~15-25% WER reduction
- Pipeline-aware Gemma: translation handles remaining ASR errors gracefully
- **Combined estimate: ~60-85% reduction in end-to-end theological translation errors**

### 2D. Ensemble Distillation (12B to 4B)

**Goal:** Transfer the 12B model's superior translation quality into the 4B model's smaller footprint. Critical for RTX 2070 deployment where 12B won't fit.

**Why distillation?** The 12B model consistently produces better translations than 4B (especially for complex theological text), but requires ~7 GB RAM and ~800ms per translation. Distillation aims to close this quality gap while keeping the 4B's speed and memory profile.

#### Recommended Approach: Sequence-Level Knowledge Distillation (Seq-KD)

**This is the most practical approach for 16GB VRAM.** Models are loaded sequentially, never simultaneously.

The idea is simple: use the fine-tuned 12B model to generate "teacher translations" for all training data, then train the 4B student on those teacher translations instead of (or in addition to) the original reference translations.

**Why Seq-KD over alternatives:**
- **Word-level KD (GKD):** Requires loading both 12B and 4B simultaneously (~14-18 GB in 4-bit). Too tight for A2000 Ada 16GB. Also requires TRL's `GKDTrainer` which adds complexity.
- **Offline logit distillation:** Would require saving full logit distributions for 155K+ sequences — hundreds of GB of disk space. Impractical.
- **Seq-KD:** Only requires the teacher's text output (a few MB). Teacher and student run sequentially. Well-validated in NMT literature.

#### Step-by-Step Implementation

**Step 1: Fine-tune the 12B teacher model (~18-27 GPU hours)**

This is the Seattle training run Stage 8. If it completes without OOM:
```bash
python training/train_gemma.py B \
  --data bible_data/aligned/verse_pairs_train.jsonl \
  --epochs 3
```

If 12B OOMs on the A2000 Ada (14-15 GB peak with `modules_to_save`):
- Remove `modules_to_save=["lm_head", "embed_tokens"]` from `train_gemma.py`
- Reduce effective batch size: `--batch-size 1 --grad-accum 8`
- If still OOMs: reduce `max_seq_length` to 384 (Bible verses rarely exceed 200 tokens)

**Step 2: Generate teacher translations (~4-6 GPU hours)**

New script: `training/generate_teacher_translations.py`

```python
"""
Generate teacher translations from the fine-tuned 12B model.
These translations are used as training targets for 4B distillation.
"""
import json
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def generate_teacher_translations(
    base_model="google/translategemma-12b-it",
    adapter_path="fine_tuned_gemma_mi_B/",
    input_file="bible_data/aligned/verse_pairs_train.jsonl",
    output_file="training_data/teacher_translations.jsonl",
    batch_size=8,
    max_new_tokens=256,
):
    # Load fine-tuned 12B model
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        load_in_4bit=True,  # 4-bit for inference efficiency
        device_map="auto",
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    # Read training data
    with open(input_file) as f:
        pairs = [json.loads(line) for line in f]

    # Generate translations
    with open(output_file, "w") as out:
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i+batch_size]
            sources = [p["en"] for p in batch]

            # Format prompts using TranslateGemma template
            prompts = []
            for src in sources:
                payload = json.dumps([{
                    "type": "text",
                    "source_lang_code": "en",
                    "target_lang_code": "es",
                    "text": src
                }], ensure_ascii=False)
                prompts.append(f"user\n{payload}\nmodel\n")

            # Generate
            inputs = tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # greedy for consistency
                num_beams=4,      # beam search for quality
            )

            # Decode and write
            for j, (pair, output) in enumerate(zip(batch, outputs)):
                teacher_translation = tokenizer.decode(
                    output[inputs.input_ids.shape[1]:],
                    skip_special_tokens=True
                ).strip()

                out.write(json.dumps({
                    "en": pair["en"],
                    "es_reference": pair["es"],        # original human reference
                    "es_teacher": teacher_translation,  # 12B teacher output
                }, ensure_ascii=False) + "\n")

            if (i // batch_size) % 100 == 0:
                print(f"  [{i}/{len(pairs)}] generated")

if __name__ == "__main__":
    generate_teacher_translations()
```

**VRAM:** 12B in 4-bit inference = ~7 GB. Comfortable on A2000 Ada.
**Time:** ~155K pairs at batch 8 with beam search = ~4-6 hours.

**Step 3: Train 4B student on mixed data (~8-12 GPU hours)**

Train the 4B model on a mixture of:
- **50% teacher translations** (12B outputs — smooth, consistent style)
- **50% human reference translations** (original Bible corpus — ground truth accuracy)

```bash
# Prepare mixed training data
python -c "
import json, random
mixed = []

# Teacher translations
with open('training_data/teacher_translations.jsonl') as f:
    for line in f:
        d = json.loads(line)
        mixed.append({'en': d['en'], 'es': d['es_teacher'], 'source': 'teacher'})

# Human references (subsample to match teacher count)
with open('bible_data/aligned/verse_pairs_train.jsonl') as f:
    refs = [json.loads(line) for line in f]
random.shuffle(refs)
for r in refs[:len(mixed)]:
    mixed.append({'en': r['en'], 'es': r['es'], 'source': 'reference'})

random.shuffle(mixed)
with open('training_data/distillation_train.jsonl', 'w') as f:
    for m in mixed:
        f.write(json.dumps(m, ensure_ascii=False) + '\n')
print(f'Total: {len(mixed)} pairs ({len(mixed)//2} teacher + {len(mixed)//2} reference)')
"

# Train 4B on mixed data
python training/train_gemma.py A \
  --data training_data/distillation_train.jsonl \
  --epochs 3
```

**Step 4: Evaluate distilled 4B (~1 hour)**

```bash
# Compare: base 4B vs fine-tuned 4B vs distilled 4B vs fine-tuned 12B
python training/evaluate_translation.py \
  --adapter fine_tuned_gemma_mi_A/   # base fine-tuned 4B
python training/evaluate_translation.py \
  --adapter distilled_gemma_mi_A/    # distilled 4B
python training/evaluate_translation.py \
  --adapter fine_tuned_gemma_mi_B/   # fine-tuned 12B (teacher)
```

**Expected results:**

| Model | BLEU | chrF++ | Theological Accuracy |
|-------|------|--------|---------------------|
| Base 4B (no fine-tuning) | ~35-42 | ~55-62 | ~40-60% |
| Fine-tuned 4B (Bible QLoRA) | ~44-48 | ~62-68 | ~65-80% |
| **Distilled 4B (Seq-KD)** | **~46-50** | **~64-70** | **~70-85%** |
| Fine-tuned 12B (teacher) | ~48-52 | ~66-72 | ~75-90% |

The distilled 4B should close 50-70% of the gap between fine-tuned 4B and fine-tuned 12B.

#### Complete Timeline

| Step | GPU Time | VRAM Peak | Prerequisites |
|------|----------|-----------|---------------|
| 1. Fine-tune 12B teacher | 18-27 hrs | ~14-15 GB | Bible corpus ready |
| 2. Generate teacher translations | 4-6 hrs | ~7 GB | Step 1 complete |
| 3. Train 4B student on mixed data | 8-12 hrs | ~10-12 GB | Step 2 complete |
| 4. Evaluate all variants | ~1 hr | ~7 GB | Steps 1-3 complete |
| **Total** | **~31-46 hrs** | | **3-5 overnight runs** |

#### Advanced: Iterative Distillation (Optional)

If the first round of distillation shows promising improvement:

1. **Round 2:** Add sermon pairs to the teacher translation set. Re-generate teacher translations with the 12B on sermon data. Retrain 4B student.
2. **Round 3:** Use the distilled 4B as a draft model for speculative decoding with the 12B teacher. This produces even higher-quality translations (verification ensures 12B-quality output, drafting ensures 4B-speed).

### 2E. Multilingual Expansion

See [`multi_lingual.md`](multi_lingual.md) and [`multilingual_tuning_proposal.md`](multilingual_tuning_proposal.md) — fully documented.

---

## Phase 3: RTX 2070 Edge Deployment

See [`rtx2070_feasibility.md`](rtx2070_feasibility.md) for detailed hardware analysis.

### 3A. Framework Migration: MLX to PyTorch CUDA

**The core challenge:** The Mac pipeline uses MLX for all inference. The RTX 2070 needs PyTorch + CUDA (or TensorRT). This is primarily a framework swap, not a model change.

**Step 1: Export models from MLX to HuggingFace format**

```bash
# On Mac: convert MLX models back to HuggingFace format
python -m mlx_lm.convert \
  --mlx-path mlx-community/translategemma-4b-it-4bit \
  --hf-path exported_gemma_4b/ \
  --dequantize  # export as float16, then re-quantize on CUDA
```

Alternatively, use the merged LoRA models from Phase 1B (already in HuggingFace format).

**Step 2: Validate on CUDA**

```python
# On Windows/RTX 2070
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained(
    "merged_gemma_4b_mi/",
    torch_dtype=torch.float16,
    device_map="cuda",
)
tokenizer = AutoTokenizer.from_pretrained("merged_gemma_4b_mi/")

# Test 10 sentences, compare output to MLX version
```

**Step 3: Quantize for RTX 2070 (8 GB VRAM)**

Use either bitsandbytes (4-bit) or GPTQ for static quantization:

```python
# Option A: bitsandbytes dynamic 4-bit (simplest)
from transformers import BitsAndBytesConfig
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
)
model = AutoModelForCausalLM.from_pretrained(
    "merged_gemma_4b_mi/",
    quantization_config=bnb_config,
    device_map="cuda",
)

# Option B: GPTQ static quantization (faster inference)
# Requires calibration data
from auto_gptq import AutoGPTQForCausalLM
model = AutoGPTQForCausalLM.from_pretrained(
    "merged_gemma_4b_mi/",
    quantize_config={"bits": 4, "group_size": 128, "desc_act": True},
)
model.quantize(calibration_dataset)
model.save_quantized("quantized_gemma_4b_mi/")
```

**Step 4: Whisper migration**

For Whisper on CUDA, use `faster-whisper` (CTranslate2 backend) for optimal inference speed:

```bash
pip install faster-whisper
```

```python
from faster_whisper import WhisperModel
model = WhisperModel("distil-large-v3", device="cuda", compute_type="int8_float16")
# ~750 MB VRAM in int8
```

**Note:** faster-whisper's CTranslate2 backend is CUDA-only (no libomp conflict on Linux/Windows).

### 3B. VRAM Budget (RTX 2070, 8 GB)

| Component | Quantization | VRAM | Notes |
|-----------|-------------|------|-------|
| Distil-Whisper | INT8 (faster-whisper) | ~750 MB | CTranslate2 backend |
| TranslateGemma 4B | INT4 (bitsandbytes/GPTQ) | ~2.5 GB | Fine-tuned + distilled |
| MarianMT | FP16 | ~600 MB | Small enough for FP16 |
| Silero VAD | FP32 (CPU) | ~2 MB | Runs on CPU, no VRAM |
| KV cache + overhead | — | ~1-2 GB | Depends on sequence length |
| **Total** | | **~5-6 GB** | **2-3 GB headroom** |

### 3C. Pipeline Adaptation Script

New file: `deploy_rtx2070.py` — mirrors `dry_run_ab.py` but uses:
- `faster-whisper` instead of `mlx-whisper`
- `transformers` with CUDA instead of `mlx-lm`
- `torch.compile()` for 10-20% speedup
- Same WebSocket/HTTP serving layer (framework-agnostic)

**Key differences from Mac pipeline:**
1. No `ThreadPoolExecutor` serialization needed — CUDA handles concurrent operations
2. No `mx.set_cache_limit` — use `torch.cuda.empty_cache()` for memory management
3. VAD still runs on CPU (Silero + PyTorch CPU)
4. No numba/libomp conflict on Linux

### 3D. TensorRT Optimization (Optional, Weeks 10-12)

TensorRT provides 2-3x speedup over PyTorch for transformer inference on NVIDIA GPUs.

```bash
# Export to ONNX
python -c "
import torch
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained('merged_gemma_4b_mi/', torch_dtype=torch.float16)
dummy = torch.randint(0, 32000, (1, 128)).cuda()
torch.onnx.export(model, dummy, 'gemma_4b.onnx', opset_version=17)
"

# Build TensorRT engine
trtexec --onnx=gemma_4b.onnx \
  --fp16 \
  --saveEngine=gemma_4b.engine \
  --workspace=4096
```

**Caveat:** TensorRT export for decoder-only models with KV cache is complex. Consider using `TensorRT-LLM` instead of raw TensorRT for LLM inference:

```bash
pip install tensorrt-llm
python -m tensorrt_llm.commands.build \
  --model_dir merged_gemma_4b_mi/ \
  --dtype float16 \
  --use_gemm_plugin float16 \
  --max_batch_size 1 \
  --max_input_len 512 \
  --max_output_len 256
```

**Expected latency on RTX 2070:**

| Component | PyTorch FP16 | PyTorch INT4 | TensorRT FP16 | TensorRT INT8 |
|-----------|-------------|-------------|---------------|---------------|
| Whisper STT | ~200ms | ~120ms | ~80ms | ~60ms |
| Gemma 4B translate | ~600ms | ~350ms | ~250ms | ~180ms |
| MarianMT partial | ~40ms | — | ~20ms | — |
| **Total final** | **~800ms** | **~470ms** | **~330ms** | **~240ms** |

The RTX 2070's 448 GB/s memory bandwidth (3x Mac's 150 GB/s) is the key advantage for transformer inference, which is memory-bandwidth-bound.

---

## Phase 4: Production Polish

### 4A. Auto-Start Pipeline

```bash
# systemd service (Linux/WSL)
# /etc/systemd/system/stark-translate.service
[Unit]
Description=Stark Road Bilingual Translation
After=network.target

[Service]
Type=simple
User=stark
WorkingDirectory=/home/stark/SRTranslate
ExecStart=/home/stark/SRTranslate/stt_env/bin/python deploy_rtx2070.py
Restart=always
RestartSec=10
Environment=CUDA_VISIBLE_DEVICES=0

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable stark-translate
sudo systemctl start stark-translate
```

### 4B. Continuous Improvement Loop

**Weekly cycle:**
```
Sunday:  Church service → live inference → diagnostics JSONL + audio chunks
Monday:  Auto-sync diagnostics to A2000 via rsync/scp
         Auto-triage: high-confidence pairs → training queue
         Low-confidence pairs → operator review interface
Monthly: Retrain on accumulated data (overnight on A2000)
         Transfer updated models to RTX 2070
         A/B test: old vs new model on held-out sermon segments
```

**Monitoring dashboard (Streamlit):**
- Per-session WER trend (should decrease over time)
- Translation QE distribution (should shift rightward)
- Theological term accuracy on spot-check sentences
- System health: VRAM usage, latency percentiles, error rate

### 4C. Projection Integration

See [`projection_integration.md`](projection_integration.md) for OBS/NDI/ProPresenter setup.

**Quickest path (< 1 hour):** OBS Browser Source pointing at `http://localhost:8080/displays/obs_overlay.html`. Transparent background, composited over camera feed or presentation slides.

---

## Summary: Execution Order and Dependencies

```
Phase 1A ──→ Phase 1B ──→ Phase 1C ──→ Phase 2B
  │                                       ↑
  │           Phase 1D (parallel) ────────┘
  │
  └──→ Phase 2A (starts with first live session)
         │
         └──→ Phase 2C (after 1-2 active learning cycles)
                │
                └──→ Phase 2D (after 12B fine-tuning complete)
                       │
                       └──→ Phase 3A-3D (after distillation validated)
                              │
                              └──→ Phase 4A-4C (after 2070 pipeline working)

Phase 2E (multilingual) can start anytime after Phase 1A completes.
```

**Total estimated timeline:** 12-16 weeks from Phase 1A start to Phase 4A deployment.

**Total estimated compute:** ~100-150 GPU hours on A2000 Ada across all phases.
