# Live Validation Plan — March 25, 2026

## Goal

A/B test W1 Whisper adapter vs base model on tonight's live ministry sermon at Stark Road Gospel Hall. This is the first real-world validation of the fine-tuned Whisper LoRA on unseen live audio.

## What We're Testing

| Component | Run A (baseline) | Run B (W1 adapter) |
|-----------|-----------------|-------------------|
| **Whisper STT** | Base v3-turbo | **W1 merged** (8.5% WER vs 24% base on eval set) |
| **TranslateGemma** | Base 4B | Base 4B (adapters showed no meaningful improvement) |
| **Expected improvement** | — | Theological terms, biblical names, domain-specific vocabulary |

## What to Listen For

The 65% WER reduction was measured on held-out sermon audio. In live testing, pay attention to:

### Theological terms (where base Whisper struggles most)
- **propitiation** → base may output "proposition" or "propitation"
- **sanctification** → base may miss or garble
- **justification** → may confuse with "just a vacation"
- **atonement** → usually correct but check
- **reconciliation** → long word, often truncated

### Biblical names
- **James/Santiago** → context-dependent (apostle vs epistle)
- **Pharisees/Sadducees** → uncommon words, base Whisper often garbles
- **Nicodemus, Barnabas, Elijah** → proper nouns base may not know

### Domain patterns
- **"Let us pray"** → common phrase, should be clean
- **"Breaking of bread"** → specific theological phrase
- **"The Word of the Lord"** → stock phrase
- **Verse references** (e.g., "John 3:16") → check number accuracy

## Setup

### 1. Transfer merged model to Mac

```bash
# On WSL (source):
rsync -av --progress models/whisper-turbo-w1-merged/ willem@macbook:~/Code/vibes/SRTranslate/models/whisper-turbo-w1-merged/
```

~1.5 GB transfer over LAN (~30 seconds on gigabit).

### 2. Configure inference on Mac

For MLX inference, convert the merged model to MLX format:

```bash
# On Mac:
pip install mlx-whisper
python -c "
from mlx_whisper import convert
convert.convert('models/whisper-turbo-w1-merged', 'models/whisper-turbo-w1-mlx')
"
```

Or use the PyTorch model directly with faster-whisper/transformers (slower but no conversion needed).

### 3. A/B test configuration

**Run A (baseline):**
```python
# settings.py — no changes needed
# Uses: mlx-community/whisper-large-v3-turbo (base)
```

**Run B (W1 adapter):**
```python
# settings.py:
# STARK_STT__WHISPER_MODEL=models/whisper-turbo-w1-mlx
# Or pass via env var
```

### 4. Run the live demo

```bash
# Run A: baseline
python dry_run_ab.py --lang en

# Run B: W1 adapter (after switching model path)
STARK_STT__WHISPER_MODEL=models/whisper-turbo-w1-mlx python dry_run_ab.py --lang en
```

## What to Record

For each run, capture:
1. **JSONL diagnostics** — auto-saved by `dry_run_ab.py`
2. **Screenshots** of live translation display — especially on theological passages
3. **Qualitative notes** — which terms were garbled vs correct
4. **Latency observations** — any noticeable speed difference?

## Evaluation Criteria

| Metric | Baseline expectation | W1 target | Method |
|--------|---------------------|-----------|--------|
| Theological term accuracy | ~40-60% | **>80%** | Manual spot-check on 10 key terms |
| Overall readability | Decent | Better | Qualitative |
| Latency | <1s partial, <2s final | Same or faster | Subjective |
| Hallucinations | Occasional | Fewer | Count instances |

## Post-Test Analysis

After the sermon:

```bash
# Download tonight's sermon audio
python download_sermons.py -s "YOUTUBE_URL" --output-dir stark_data/raw/eval_fresh

# Transcribe with Deepgram (ground truth)
python training/transcribe_with_deepgram.py --input stark_data/raw/eval_fresh \
    --output stark_data/eval_fresh_transcripts

# Compare base vs W1 WER on this specific sermon
python training/eval_whisper_wer.py --eval-set stark_data/eval_fresh \
    --adapter models/whisper-turbo-w1-merged --output eval_tonight_w1.json
python training/eval_whisper_wer.py --eval-set stark_data/eval_fresh \
    --output eval_tonight_base.json
```

## Merged Model Location

| File | Path | Size |
|------|------|------|
| Merged W1 (PyTorch fp16) | `models/whisper-turbo-w1-merged/` | 1.5 GB |
| W1 LoRA adapter only | `whisper_ablation/W1_baseline/` | 26 MB |
| Base model | `openai/whisper-large-v3-turbo` (HuggingFace) | 1.5 GB |

## Training Provenance

W1 was trained on:
- **16,956 sermon chunks** from 33 pre-cutoff sermons (Jun 2025 – Mar 2026)
- **Ground truth**: Deepgram Nova-3 with 50 theological keyterms
- **Config**: lr=1e-4, LoRA r=32, alpha=64, q_proj+v_proj, 1 epoch, 30% LibriSpeech replay
- **Result**: 24.18% → 8.46% WER on held-out sermon (65% relative reduction)

Tonight's sermon is completely unseen — different speakers, different date, ministry (not gospel). This tests generalization.
