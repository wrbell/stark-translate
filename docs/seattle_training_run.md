# Seattle Training Run — Feb 12-17, 2026 (6 Days Unattended)

> **Goal:** Run the full training pipeline end-to-end while away, to validate
> that every stage works and produces measurable quality improvements.
> **Hardware:** Windows Desktop — A2000 Ada 16GB, 64GB RAM, WSL2
> **Duration:** Thursday evening → Tuesday morning = ~5 nights, ~120 GPU-hours available

---

## What Trains On What (Data Flow)

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA SOURCES                                  │
│                                                                      │
│  YouTube Sermons (audio)          Bible Parallel Corpus (text)       │
│  ┌──────────────────────┐         ┌──────────────────────────┐      │
│  │ 30 Gospel sermons    │         │ scrollmapper/bible_databases│    │
│  │ ~30 hrs audio        │         │ 5 EN × 2 ES = 269K pairs  │    │
│  │ Downloaded via yt-dlp │         │ Already built on Mac!      │    │
│  └──────────┬───────────┘         └───────────┬──────────────┘      │
│             │                                  │                     │
│             ▼                                  │                     │
│  ┌───────────────────────────────┐               │                     │
│  │ training/preprocess_audio.py │               │                     │
│  │ 10-step pipeline              │               │                     │
│  │ → VAD segments                │               │                     │
│  └───────────────┬───────────────┘               │                     │
│             │                                  │                     │
│             ▼                                  │                     │
│  ┌───────────────────────────────┐               │                     │
│  │ training/transcribe_church.py│               │                     │
│  │ Whisper large-v3              │               │                     │
│  │ → pseudo-labels               │               │                     │
│  │ (text for each chunk)         │               │                     │
│  └───────────────┬───────────────┘               │                     │
│             │                                  │                     │
└─────────────┼──────────────────────────────────┼─────────────────────┘
              │                                  │
              ▼                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      MODEL TRAINING                                  │
│                                                                      │
│  ┌────────────────────────────────────┐                              │
│  │ training/train_whisper.py (LoRA)            │                     │
│  │                                    │                              │
│  │ INPUT:  Church audio segments      │                              │
│  │         + their pseudo-labels      │                              │
│  │         + 30% Common Voice replay  │                              │
│  │                                    │                              │
│  │ LEARNS: Church acoustics (reverb,  │                              │
│  │         PA system, room noise) +   │                              │
│  │         theological vocabulary     │                              │
│  │         (decoder learns word       │                              │
│  │         patterns like "atonement", │                              │
│  │         "propitiation", etc.)      │                              │
│  │                                    │                              │
│  │ OUTPUT: fine_tuned_whisper_mi/     │                              │
│  │         (LoRA adapter, ~50-100MB)  │                              │
│  │ TIME:   ~1 hr (3 epochs, 30 hrs)  │                              │
│  └────────────────────────────────────┘                              │
│                                                                      │
│  ┌────────────────────────────────────┐                              │
│  │ training/train_marian.py (full fine-tune)   │                     │
│  │                                    │                              │
│  │ INPUT:  bible_data/aligned/        │◄── 269K EN→ES verse pairs    │
│  │         verse_pairs_train.jsonl    │    + glossary pairs           │
│  │         + glossary_pairs.jsonl     │                              │
│  │                                    │                              │
│  │ LEARNS: Biblical EN→ES translation │                              │
│  │         patterns, theological      │                              │
│  │         vocabulary, formal register│                              │
│  │                                    │                              │
│  │ OUTPUT: fine_tuned_marian_mi/      │                              │
│  │         (full model copy, ~600MB)  │                              │
│  │ TIME:   ~2.2 hrs (5 epochs)       │                              │
│  └────────────────────────────────────┘                              │
│                                                                      │
│  ┌────────────────────────────────────┐                              │
│  │ training/train_gemma.py A (QLoRA, 4B)      │                     │
│  │                                    │                              │
│  │ INPUT:  bible_data/aligned/        │◄── Same 269K pairs           │
│  │         verse_pairs_train.jsonl    │    + glossary pairs           │
│  │         + glossary_pairs.jsonl     │                              │
│  │                                    │                              │
│  │ LEARNS: Same as MarianMT but with │                              │
│  │         much higher quality ceiling│                              │
│  │         — contextual disambiguation│                              │
│  │         (covenant→pacto vs alianza)│                              │
│  │         better handling of complex │                              │
│  │         theological sentences      │                              │
│  │                                    │                              │
│  │ OUTPUT: fine_tuned_gemma_mi_A/     │                              │
│  │         (LoRA adapter, ~200-400MB) │                              │
│  │ TIME:   ~12 hrs (3 epochs)        │                              │
│  └────────────────────────────────────┘                              │
│                                                                      │
│  ┌────────────────────────────────────┐                              │
│  │ training/evaluate_translation.py           │                     │
│  │                                    │                              │
│  │ INPUT:  bible_data/holdout/        │◄── 27K held-out pairs        │
│  │         verse_pairs_test.jsonl     │                              │
│  │         + all trained models       │                              │
│  │                                    │                              │
│  │ MEASURES: SacreBLEU, chrF++, COMET │                              │
│  │           per model, per genre     │                              │
│  │                                    │                              │
│  │ OUTPUT: metrics/eval_results.json  │                              │
│  │ TIME:   ~1 hr                     │                              │
│  └────────────────────────────────────┘                              │
└─────────────────────────────────────────────────────────────────────┘
```

### Summary: What Data Goes Where

| Model | Training Data | What It Learns | Output |
|-------|--------------|----------------|--------|
| **Whisper LoRA** | Church audio WAVs + pseudo-labels from Whisper large-v3 | Church acoustics (reverb, PA), speaker patterns, theological vocab in decoder | `fine_tuned_whisper_mi/` |
| **MarianMT** | 269K Bible verse pairs (EN→ES) + 458 glossary pairs | Biblical EN→ES translation, theological terms, formal register | `fine_tuned_marian_mi/` |
| **TranslateGemma 4B** | Same 269K Bible verse pairs + glossary pairs | Same as MarianMT but higher quality — contextual disambiguation, complex sentences | `fine_tuned_gemma_mi_A/` |

**Key insight:** Whisper and the translation models train on **completely different data**:
- Whisper: **audio** (church sermons) → **text** (English transcription)
- MarianMT/TranslateGemma: **text** (English Bible) → **text** (Spanish Bible)

The **bible_data/** corpus is used ONLY for translation training, not for Whisper.
The **sermon audio** is used ONLY for Whisper training, not for translation.

They complement each other in the final pipeline:
```
Mic audio → [fine-tuned Whisper] → English text → [fine-tuned TranslateGemma/MarianMT] → Spanish text
```

---

## Pre-Flight Checklist (Before Leaving Thursday)

### Wednesday Night / Thursday Morning — Setup WSL

```bash
# 1. Verify CUDA
nvidia-smi  # Should show A2000 Ada, 16GB

# 2. Clone repo into WSL
git clone git@github.com:wrbell/stark-translate.git ~/SRTranslate
cd ~/SRTranslate

# 3. Create virtual environment
python3.12 -m venv stt_env
source stt_env/bin/activate

# 4. Install training dependencies
pip install -r requirements-windows.txt

# 5. Verify key packages
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -c "import bitsandbytes; print('bitsandbytes OK')"
python -c "from peft import LoraConfig; print('peft OK')"
python -c "from trl import SFTTrainer; print('trl OK')"
python -c "import faster_whisper; print('faster-whisper OK')"

# 6. Quick smoke test — load TranslateGemma 4B (downloads ~2.6GB on first run)
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True,
                          bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=torch.bfloat16)
model = AutoModelForCausalLM.from_pretrained('google/translategemma-4b-it',
                                              quantization_config=bnb, device_map='auto')
print(f'Model loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB')
del model; torch.cuda.empty_cache()
print('Smoke test passed!')
"

# 7. Bible corpus is already in the repo
ls bible_data/aligned/verse_pairs_train.jsonl  # Should exist (242K pairs)
ls bible_data/aligned/verse_pairs_test.jsonl   # Should exist (27K pairs)
ls bible_data/glossary/glossary_pairs.jsonl    # Should exist (458 pairs)
```

### Thursday Afternoon — Download Sermons

```bash
# Download 30 most recent Gospel sermons (~30 hrs, ~1.7 GB)
python download_sermons.py -n 30

# Verify downloads
ls stark_data/raw/*.wav | wc -l  # Should be 30
du -sh stark_data/raw/           # Should be ~1.7 GB
```

---

## The Run Script

Create this as `seattle_run.sh` and launch in tmux before leaving:

```bash
#!/usr/bin/env bash
# seattle_run.sh — Full unattended training pipeline
# Expected total runtime: ~25-30 hours (fits easily in 6 days)
# Run in tmux: tmux new -s training && bash seattle_run.sh

set -euo pipefail
cd ~/SRTranslate
source stt_env/bin/activate

LOG_DIR="logs/seattle_run_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "========================================" | tee "$LOG_DIR/progress.log"
echo "Seattle Training Run — $(date)"          | tee -a "$LOG_DIR/progress.log"
echo "========================================" | tee -a "$LOG_DIR/progress.log"

# ── Stage 1: Audio Preprocessing (~5 hrs) ─────────────────────────
echo "[$(date +%H:%M)] Stage 1: Audio preprocessing..." | tee -a "$LOG_DIR/progress.log"
python training/preprocess_audio.py \
    --input stark_data/raw \
    --output stark_data/cleaned \
    --resume \
    2>&1 | tee "$LOG_DIR/01_preprocess.log"
echo "[$(date +%H:%M)] Stage 1 complete." | tee -a "$LOG_DIR/progress.log"

# ── Stage 2: Pseudo-labeling with Whisper large-v3 (~2.5 hrs) ────
echo "[$(date +%H:%M)] Stage 2: Pseudo-labeling..." | tee -a "$LOG_DIR/progress.log"
python training/transcribe_church.py \
    --backend faster-whisper \
    --resume \
    2>&1 | tee "$LOG_DIR/02_transcribe.log"
echo "[$(date +%H:%M)] Stage 2 complete." | tee -a "$LOG_DIR/progress.log"

# ── Stage 3: Baseline evaluation (before fine-tuning) (~20 min) ──
echo "[$(date +%H:%M)] Stage 3: Baseline evaluation..." | tee -a "$LOG_DIR/progress.log"
python training/evaluate_translation.py --models all \
    2>&1 | tee "$LOG_DIR/03_baseline_eval.log" || true
echo "[$(date +%H:%M)] Stage 3 complete." | tee -a "$LOG_DIR/progress.log"

# ── Stage 4: Whisper LoRA fine-tune (~1 hr) ──────────────────────
echo "[$(date +%H:%M)] Stage 4: Whisper LoRA training..." | tee -a "$LOG_DIR/progress.log"
python training/train_whisper.py \
    --dataset stark_data/cleaned \
    --output fine_tuned_whisper_mi \
    --epochs 3 \
    --replay-ratio 0.3 \
    2>&1 | tee "$LOG_DIR/04_whisper_lora.log"
echo "[$(date +%H:%M)] Stage 4 complete." | tee -a "$LOG_DIR/progress.log"

# ── Stage 5: MarianMT full fine-tune (~2.2 hrs) ─────────────────
echo "[$(date +%H:%M)] Stage 5: MarianMT training..." | tee -a "$LOG_DIR/progress.log"
python training/train_marian.py \
    --bible-data bible_data/aligned/verse_pairs_train.jsonl \
    --epochs 5 \
    2>&1 | tee "$LOG_DIR/05_marian.log"
echo "[$(date +%H:%M)] Stage 5 complete." | tee -a "$LOG_DIR/progress.log"

# ── Stage 6: TranslateGemma 4B QLoRA (~12 hrs) ──────────────────
echo "[$(date +%H:%M)] Stage 6: TranslateGemma 4B QLoRA..." | tee -a "$LOG_DIR/progress.log"
python training/train_gemma.py A \
    --bible-data bible_data/aligned/verse_pairs_train.jsonl \
    --epochs 3 \
    2>&1 | tee "$LOG_DIR/06_gemma4b.log"
echo "[$(date +%H:%M)] Stage 6 complete." | tee -a "$LOG_DIR/progress.log"

# ── Stage 7: Post-training evaluation (~1 hr) ───────────────────
echo "[$(date +%H:%M)] Stage 7: Post-training evaluation..." | tee -a "$LOG_DIR/progress.log"
python training/evaluate_translation.py --models all \
    2>&1 | tee "$LOG_DIR/07_post_eval.log" || true
echo "[$(date +%H:%M)] Stage 7 complete." | tee -a "$LOG_DIR/progress.log"

# ── Stage 8 (BONUS): TranslateGemma 12B if time allows (~26 hrs) ─
echo "[$(date +%H:%M)] Stage 8: TranslateGemma 12B QLoRA..." | tee -a "$LOG_DIR/progress.log"
python training/train_gemma.py B \
    --bible-data bible_data/aligned/verse_pairs_train.jsonl \
    --epochs 3 \
    2>&1 | tee "$LOG_DIR/08_gemma12b.log" || {
    echo "[$(date +%H:%M)] Stage 8 failed (likely OOM). See log." | tee -a "$LOG_DIR/progress.log"
}

echo "========================================" | tee -a "$LOG_DIR/progress.log"
echo "ALL STAGES COMPLETE — $(date)"           | tee -a "$LOG_DIR/progress.log"
echo "========================================" | tee -a "$LOG_DIR/progress.log"

# Summary of what was produced
echo ""
echo "Outputs:"
echo "  fine_tuned_whisper_mi/   — Whisper LoRA adapter"
echo "  fine_tuned_marian_mi/    — MarianMT full model"
echo "  fine_tuned_gemma_mi_A/   — TranslateGemma 4B LoRA"
echo "  fine_tuned_gemma_mi_B/   — TranslateGemma 12B LoRA (if completed)"
echo "  $LOG_DIR/               — All training logs"
echo ""
echo "Next: scp adapters to Mac and test with dry_run_ab.py"
```

---

## Timeline: What Happens While You're Gone

```
Thursday Evening (before leaving):
  18:00  Launch seattle_run.sh in tmux
  18:00  Stage 1: Audio preprocessing starts (~5 hrs)
  23:00  Stage 2: Pseudo-labeling starts (~2.5 hrs)

Friday Early Morning:
  01:30  Stage 3: Baseline eval (~20 min)
  02:00  Stage 4: Whisper LoRA (~1 hr)
  03:00  Stage 5: MarianMT (~2.2 hrs)
  05:15  Stage 6: TranslateGemma 4B QLoRA starts (~12 hrs)

Friday Evening:
  17:00  Stage 7: Post-training eval (~1 hr)
  18:00  Stage 8: TranslateGemma 12B QLoRA starts (~26 hrs)

Saturday → Sunday:
  12B training continues...
  (If it OOMs, the script catches the error and reports it)

Sunday Evening:
  ~20:00  Everything complete. Desktop idle.

Monday-Tuesday:
  Desktop idle, all results waiting.
```

**Total GPU time:** ~50 hrs (including 12B attempt)
**Calendar time:** ~50 hrs = finishes Saturday night / Sunday morning
**Remaining idle time:** ~3 days of safety margin

---

## What You Get Back (Tuesday)

### Artifacts to Transfer to Mac

```bash
# From Windows, after returning:
scp -r fine_tuned_whisper_mi/ mac:~/Code/vibes/SRTranslate/
scp -r fine_tuned_marian_mi/ mac:~/Code/vibes/SRTranslate/
scp -r fine_tuned_gemma_mi_A/ mac:~/Code/vibes/SRTranslate/
scp -r fine_tuned_gemma_mi_B/ mac:~/Code/vibes/SRTranslate/  # if 12B completed
scp -r logs/seattle_run_*/ mac:~/Code/vibes/SRTranslate/logs/
```

### What to Check First

1. **`logs/seattle_run_*/progress.log`** — Did every stage complete?
2. **`logs/seattle_run_*/03_baseline_eval.log`** — Baseline metrics (before)
3. **`logs/seattle_run_*/07_post_eval.log`** — Post-training metrics (after)
4. **Compare:** Did BLEU improve? Did theological term accuracy improve?
5. **`logs/seattle_run_*/08_gemma12b.log`** — Did 12B complete or OOM?

### Test on Mac

```bash
# Load fine-tuned Whisper (need to convert LoRA to MLX format first)
# Load fine-tuned TranslateGemma (need to merge LoRA + requantize for MLX)
# Run A/B comparison:
python dry_run_ab.py  # with adapter_path= arguments
```

**Note:** LoRA → MLX conversion is a separate step needed after returning.
The adapters are in HuggingFace/PyTorch format; MLX needs its own format.
This is a ~30 min task per model on the Mac.

---

## Risk Mitigations for Unattended Run

| Risk | Mitigation |
|------|------------|
| OOM on TranslateGemma 12B | `|| true` catches error, script continues, 4B results still valid |
| Power outage | `--resume` flags on most scripts; tmux session survives reconnect |
| Disk full | 30 hrs audio = ~3 GB; Bible corpus = ~50 MB; models = ~2 GB. Total <10 GB |
| WSL crashes | Set Windows to auto-login + auto-start WSL (optional) |
| Network drops during download | Download sermons before leaving; all training is offline |
| Training diverges (NaN loss) | Check logs on return; can re-run individual stages |

---

## What This Run Does NOT Include

- **Human correction** — Raw Whisper pseudo-labels only (no Label Studio review)
- **Active learning cycles** — This is Round 1 only; cycles 2-5 need human input
- **Mac inference testing** — LoRA → MLX conversion happens after returning
- **12B guarantee** — May OOM; 4B is the primary target

This is a **proof-of-concept run** to validate the full pipeline end-to-end.
Quality will improve significantly in subsequent rounds with human correction.
