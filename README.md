# stark-translate

Live bilingual (English/Spanish) speech-to-text for church outreach at Stark Road Gospel Hall, Farmington Hills, MI.

Real-time mic input, fully on-device transcription and translation, displayed in browser. Uses a two-pass pipeline for fast partials and high-quality finals, with A/B model comparison for translation quality analysis.

## Architecture

```
                              Two-Pass Pipeline
                              ================

  Mic (48kHz) ──> Resample 16kHz ──> Silero VAD ──┐
                                                    │
            ┌───────────────────────────────────────┘
            │
            ├─ PARTIAL (on 1s of new speech, while speaker is talking)
            │    mlx-whisper STT (~500ms)
            │    MarianMT EN→ES (~80ms)                    ← italic in UI
            │    Total: ~580ms
            │
            └─ FINAL (on silence gap or 8s max utterance)
                 mlx-whisper STT (~500ms, word timestamps)
                 TranslateGemma 4B EN→ES (~650ms)          ← replaces partial
                 TranslateGemma 12B EN→ES (~1.4s, --ab)    ← side-by-side
                 Total: ~1.3s (4B) / ~1.9s (A/B)
                                     │
                                     ▼
                          WebSocket (0.0.0.0:8765)
                                     │
              ┌──────────┬───────────┼───────────┐
              ▼          ▼           ▼           ▼
          Audience    A/B/C       Mobile      CSV +
          Display    Compare     Display     Diagnostics
         (projector) (operator)  (QR code)    (JSONL)
```

All inference runs natively on Apple Silicon via MLX (4-bit quantized). No cloud APIs, no internet required at runtime.

## Display Modes

| Display | File | Purpose |
|---------|------|---------|
| **Audience** | `audience_display.html` | Projector-friendly side-by-side EN/ES with fading context, fullscreen toggle, QR code overlay for phones |
| **A/B/C Comparison** | `ab_display.html` | Operator view showing Gemma 4B / MarianMT / 12B side-by-side with latency stats |
| **Mobile** | `mobile_display.html` | Responsive phone/tablet view with model toggle and Spanish-only mode, accessible via LAN |
| **Church** | `church_display.html` | Simplified church-oriented layout |

Phones connect by scanning the QR code on the audience display or navigating to `http://<LAN-IP>:8080/mobile_display.html`.

## Quick Start

```bash
# Prerequisites
brew install ffmpeg portaudio

# Create env
python3.11 -m venv stt_env
source stt_env/bin/activate
pip install -r requirements-mac.txt

# HuggingFace login (required for TranslateGemma)
huggingface-cli login

# Download all models
python setup_models.py

# Run — 4B only (default, ~4.3 GB RAM)
python dry_run_ab.py

# Run — A/B mode with both 4B and 12B (~11.3 GB RAM)
python dry_run_ab.py --ab

# Open displays in browser
open audience_display.html
open ab_display.html
```

### Key Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--ab` | off | Load both 4B and 12B for A/B comparison |
| `--http-port` | 8080 | HTTP server port (serves display pages to phones over LAN) |
| `--ws-port` | 8765 | WebSocket server port |
| `--vad-threshold` | 0.3 | VAD speech detection sensitivity (0-1) |
| `--gain` | auto | Mic gain multiplier (auto-calibrates by default) |
| `--device` | auto | Audio input device index |
| `--chunk-duration` | 2.0 | Seconds of speech to accumulate |

## Models

| Component | Model | Framework | Size | Latency |
|-----------|-------|-----------|------|---------|
| VAD | Silero VAD | PyTorch | ~2 MB | <1ms |
| STT | Distil-Whisper large-v3 | mlx-whisper | ~1.5 GB | ~500ms |
| Translate (partials) | MarianMT opus-mt-en-es | CTranslate2 int8 | ~76 MB | ~50ms |
| Translate (partials) | MarianMT opus-mt-en-es | PyTorch fp32 | ~300 MB | ~80ms |
| Translate A (finals) | TranslateGemma 4B 4-bit | mlx-lm | ~2.5 GB | ~650ms |
| Translate B (finals) | TranslateGemma 12B 4-bit | mlx-lm | ~7 GB | ~1.4s |

Both CTranslate2 and PyTorch MarianMT backends run in parallel for latency comparison logging.

## Features

- **Two-pass STT pipeline** -- fast italic partials (MarianMT, ~580ms) replaced by high-quality finals (TranslateGemma, ~1.3s) on silence detection
- **A/B translation comparison** -- 4B and 12B TranslateGemma run in parallel via `run_in_executor`, logged to CSV
- **Theological Whisper prompt** -- biases STT toward church vocabulary (atonement, propitiation, mediator, etc.) to reduce homophone errors
- **Previous-text context** -- last transcription fed to Whisper for cross-chunk accuracy
- **Profanity filter** -- allows biblical terms (e.g., "wrath") that generic filters block
- **Speculative decoding support** -- 4B model drafts tokens for 12B verification (`--num-draft-tokens`)
- **Confidence scoring** -- segment-level `avg_logprob` mapped to green/yellow/red indicators
- **Translation QE** -- length ratio + untranslated content detection per chunk
- **Hallucination detection** -- flags segments with `compression_ratio > 2.4`
- **Word-level timestamps** -- per-word confidence logged for fine-tuning prioritization
- **Automated diagnostics** -- homophone flags, bad sentence splits, Marian/Gemma divergence tracking
- **Per-chunk audio saving** -- WAV files saved to `stark_data/live_sessions/` for Whisper fine-tuning
- **Structured review queue** -- JSONL diagnostics with priority scoring for active learning
- **Hardware profiling** -- per-session CPU/RAM/GPU snapshots for portability planning
- **LAN serving** -- HTTP server + WebSocket on `0.0.0.0` so phones connect over local network
- **229-term theological glossary** -- covers 66 books, 31 proper names, theological concepts, liturgical terms

## Hardware Requirements

### Mac (Inference)

- Apple Silicon (M1/M2/M3/M4)
- **8 GB+** unified memory for 4B-only mode (~4.3 GB used)
- **18 GB+** unified memory for A/B mode (~11.3 GB used)
- Python 3.11, macOS with Metal support

### Windows (Training)

- NVIDIA GPU with 16 GB+ VRAM (tested on A2000 Ada)
- 64 GB+ system RAM recommended
- WSL2 with CUDA toolkit
- Used for audio preprocessing (demucs), pseudo-labeling (Whisper large-v3), and LoRA/QLoRA fine-tuning

## Training

Fine-tuning runs on the Windows desktop and adapters transfer to the Mac for inference:

- **Whisper LoRA** (r=32) on 20-50 hours of church sermon audio
- **TranslateGemma QLoRA** (r=16, 4-bit NF4) on ~155K biblical verse pairs (public domain)
- **MarianMT full fine-tune** as a lightweight fallback (298 MB)

Training data: church audio via yt-dlp + Bible parallel corpus (KJV/ASV/WEB/BBE/YLT paired with RVR1909). See [`CLAUDE.md`](./CLAUDE.md) for the full architecture, training strategy, and compute timeline.

## Project Structure

```
├── dry_run_ab.py              # Main pipeline: mic → VAD → STT → translate → WebSocket
├── setup_models.py            # One-command model download + verification
├── build_glossary.py          # EN→ES theological glossary (229 terms)
├── download_sermons.py        # yt-dlp sermon downloader
│
├── audience_display.html      # Projector display (EN/ES side-by-side)
├── ab_display.html            # A/B/C operator comparison display
├── mobile_display.html        # Phone/tablet responsive display
├── church_display.html        # Simplified church layout
│
├── preprocess_audio.py        # 10-step audio cleaning (Windows)
├── transcribe_church.py       # Whisper large-v3 pseudo-labeling (Windows)
├── prepare_bible_corpus.py    # Bible verse pair alignment (Windows)
├── train_whisper.py           # Whisper LoRA fine-tuning (Windows)
├── train_gemma.py             # TranslateGemma QLoRA fine-tuning (Windows)
├── train_marian.py            # MarianMT full fine-tune (Windows)
├── evaluate_translation.py    # SacreBLEU/chrF++/COMET scoring
├── assess_quality.py          # Baseline WER assessment
├── translation_qe.py          # Reference-free translation QE
│
├── ct2_opus_mt_en_es/         # CTranslate2 int8 MarianMT model
├── stark_data/                # Church audio + transcripts + corrections
├── bible_data/                # Biblical parallel text corpus
├── metrics/                   # CSV logs, diagnostics JSONL, hardware profiles
│
├── CLAUDE.md                  # Full architecture, training strategy, research basis
├── CLAUDE-macbook.md          # Mac environment setup guide
├── CLAUDE-windows.md          # Windows/WSL training environment guide
└── todo.md                    # Task tracking
```

## Docs

- [`CLAUDE.md`](./CLAUDE.md) -- Full project overview, 6-layer architecture, fine-tuning strategy, compute timeline
- [`CLAUDE-macbook.md`](./CLAUDE-macbook.md) -- Mac inference environment setup
- [`CLAUDE-windows.md`](./CLAUDE-windows.md) -- Windows/WSL training environment setup

## License

Private project. All Bible translation training data uses public domain or CC-licensed sources only.
