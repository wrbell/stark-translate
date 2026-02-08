# stark-translate

Live bilingual (English/Spanish) speech-to-text for church outreach at Stark Road Gospel Hall, Farmington Hills, MI.

Real-time mic input, on-device transcription + translation, displayed in a 4-quadrant browser UI. Designed to outperform YouTube's built-in captions with domain-adapted models fine-tuned on church sermon audio and biblical text.

## How It Works

```
Mic → Silero VAD → Distil-Whisper STT → TranslateGemma EN→ES → Browser Display
```

Two translation models run in parallel for A/B comparison:

| Approach | Model | Latency | Quality |
|----------|-------|---------|---------|
| **A** (Lightweight) | TranslateGemma 4B | ~70-210ms | Good |
| **B** (High-Accuracy) | TranslateGemma 12B | ~140-410ms | Better |

## Quick Start (Mac)

```bash
# Install system deps
brew install ffmpeg portaudio

# Create env + install packages
python3.11 -m venv stt_env
source stt_env/bin/activate
pip install -r requirements-mac.txt

# Login to HuggingFace (required for TranslateGemma)
huggingface-cli login

# Download & verify all models
python setup_models.py

# Run the A/B test
python dry_run_ab.py

# Open the display (separate terminal)
open ab_display.html
```

## Architecture

Two machines, two roles:

| Machine | Role | Key Scripts |
|---------|------|-------------|
| **MacBook** (M3 Pro, 18GB) | Inference, live demos, A/B testing | `dry_run_ab.py`, `ab_display.html`, `setup_models.py` |
| **Windows Desktop** (WSL2, A2000 Ada) | Audio preprocessing, fine-tuning | `preprocess_audio.py`, `train_whisper.py`, `train_gemma.py` |

### Mac Inference Pipeline

- `dry_run_ab.py` — Mic → VAD → STT → parallel translation → WebSocket broadcast + CSV logging
- `ab_display.html` — Self-contained 4-quadrant browser display (dark theme, auto-scroll, live stats)
- `setup_models.py` — One-command model download + verification for either machine

### Windows Training Pipeline

- `preprocess_audio.py` — 10-step audio cleaning (convert, quality gate, demucs, denoise, normalize, VAD chunk)
- `transcribe_church.py` — Whisper large-v3 pseudo-labeling for training data
- `prepare_bible_corpus.py` — Download & align ~155K Bible verse pairs (public domain)
- `build_glossary.py` — EN→ES theological glossary (~65 terms, expandable to 500)
- `assess_quality.py` — Sample segments for manual review, compute baseline WER
- `train_whisper.py` — Whisper LoRA fine-tuning (r=32, encoder+decoder)
- `train_gemma.py` — TranslateGemma QLoRA fine-tuning (r=16, 4-bit NF4)
- `train_marian.py` — MarianMT full fine-tune fallback (298MB, fast iteration)
- `evaluate_translation.py` — SacreBLEU/chrF++/COMET scoring + theological term spot-check

## Fine-Tuning Data

| Source | Size | License |
|--------|------|---------|
| Church sermon audio (Stark Road YT) | 20-50 hrs | Fair use |
| Bible verse pairs (KJV/ASV/WEB/BBE/YLT ↔ RVR1909) | ~155K pairs | Public domain |
| Theological glossary | ~65-500 terms | Original |

## Project Docs

- [`CLAUDE.md`](./CLAUDE.md) — Full project overview, architecture, research basis
- [`CLAUDE-macbook.md`](./CLAUDE-macbook.md) — Mac inference environment guide
- [`CLAUDE-windows.md`](./CLAUDE-windows.md) — Windows/WSL training environment guide

## License

Private project. All Bible translation data uses public domain or CC-licensed sources only.
