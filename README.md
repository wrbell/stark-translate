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
            │    mlx-whisper STT (~300ms)
            │    MarianMT EN→ES CT2 int8 (~50ms)           ← italic in UI
            │    Total: ~350ms
            │
            └─ FINAL (on silence gap or 8s max utterance)
            │    mlx-whisper STT (~300ms, word timestamps)
            │    TranslateGemma 4B EN→ES (~350ms)          ← replaces partial
            │    TranslateGemma 12B EN→ES (~800ms, --ab)   ← side-by-side
            │    Total: ~650ms (4B) / ~1.1s (A/B)
            │
            │    Pipeline overlap (6C): translation runs on utterance N
            │    while STT runs on utterance N+1, hiding translation latency.
                                     │
                                     ▼
                          WebSocket (0.0.0.0:8765)
                           HTTP (0.0.0.0:8080)
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
| **Audience** | `displays/audience_display.html` | Projector-friendly side-by-side EN/ES with fading context, fullscreen toggle, QR code overlay for phones |
| **A/B/C Comparison** | `displays/ab_display.html` | Operator view showing Gemma 4B / MarianMT / 12B side-by-side with latency stats |
| **Mobile** | `displays/mobile_display.html` | Responsive phone/tablet view with model toggle and Spanish-only mode, accessible via LAN |
| **Church** | `displays/church_display.html` | Simplified church-oriented layout |
| **OBS Overlay** | `displays/obs_overlay.html` | Transparent overlay for OBS Studio / streaming integration |

Phones connect by scanning the QR code on the audience display or navigating to `http://<LAN-IP>:8080/displays/mobile_display.html`.

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
open displays/audience_display.html
open displays/ab_display.html
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
| STT | Distil-Whisper large-v3 | mlx-whisper | ~1.5 GB | ~300ms |
| Translate (partials) | MarianMT opus-mt-en-es | CTranslate2 int8 | ~76 MB | ~50ms |
| Translate A (finals) | TranslateGemma 4B 4-bit | mlx-lm | ~2.5 GB | ~350ms |
| Translate B (finals) | TranslateGemma 12B 4-bit | mlx-lm | ~7 GB | ~800ms |

Pipeline overlap (P7-6C) hides translation latency by running translation on utterance N while STT processes utterance N+1.

## Features

- **Two-pass STT pipeline** -- fast italic partials (MarianMT, ~350ms) replaced by high-quality finals (TranslateGemma, ~650ms) on silence detection
- **Pipeline overlap** -- translation runs concurrently with next utterance's STT, hiding translation latency
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
├── dry_run_ab.py              # Main pipeline: mic → VAD → STT → translate → WebSocket + HTTP
├── setup_models.py            # One-command model download + verification
├── build_glossary.py          # EN→ES theological glossary (229 terms)
├── download_sermons.py        # yt-dlp sermon downloader
│
├── displays/
│   ├── audience_display.html  # Projector display (EN/ES side-by-side, QR overlay)
│   ├── ab_display.html        # A/B/C operator comparison display
│   ├── mobile_display.html    # Phone/tablet responsive display
│   ├── church_display.html    # Simplified church layout
│   └── obs_overlay.html       # Transparent overlay for OBS Studio
│
├── training/                  # Windows/WSL training scripts (CUDA)
│   ├── preprocess_audio.py    # 10-step audio cleaning pipeline
│   ├── transcribe_church.py   # Whisper large-v3 pseudo-labeling
│   ├── prepare_bible_corpus.py # Bible verse pair alignment
│   ├── train_whisper.py       # Whisper LoRA fine-tuning
│   ├── train_gemma.py         # TranslateGemma QLoRA fine-tuning
│   ├── train_marian.py        # MarianMT full fine-tune
│   ├── evaluate_translation.py # SacreBLEU/chrF++/COMET scoring
│   └── assess_quality.py      # Baseline WER assessment
│
├── tools/                     # Mac benchmarking & monitoring
│   ├── live_caption_monitor.py # YouTube caption comparison (post/live/trend)
│   ├── translation_qe.py      # Reference-free translation QE
│   ├── benchmark_latency.py   # End-to-end latency profiling
│   ├── stt_benchmark.py       # STT-only benchmarking
│   └── test_adaptive_model.py # Adaptive model selection testing
│
├── features/                  # Standalone future features
│   ├── diarize.py             # Speaker diarization (pyannote-audio)
│   ├── extract_verses.py      # Bible verse reference extraction
│   └── summarize_sermon.py    # Post-sermon 5-sentence summary
│
├── docs/
│   ├── seattle_training_run.md # 6-day unattended training plan (Feb 12-17)
│   ├── training_plan.md       # Full training schedule + go/no-go gates
│   ├── training_time_estimates.md # A2000 Ada GPU time estimates
│   ├── roadmap.md             # Mac → Windows → RTX 2070 deployment roadmap
│   ├── rtx2070_feasibility.md # RTX 2070 hardware analysis
│   ├── projection_integration.md # OBS/NDI/ProPresenter integration
│   ├── fast_stt_options.md    # Lightning-whisper-mlx feasibility study
│   └── previous_actions.md    # Completed work log
│
├── ct2_opus_mt_en_es/         # CTranslate2 int8 MarianMT model
├── stark_data/                # Church audio + transcripts + corrections
├── bible_data/                # Biblical parallel text corpus (269K pairs)
└── metrics/                   # CSV logs, diagnostics JSONL, hardware profiles
```

## Docs

| Doc | Contents |
|-----|----------|
| [`CLAUDE.md`](./CLAUDE.md) | Full project overview, 6-layer architecture, fine-tuning strategy, compute timeline |
| [`CLAUDE-macbook.md`](./CLAUDE-macbook.md) | Mac inference environment setup |
| [`CLAUDE-windows.md`](./CLAUDE-windows.md) | Windows/WSL training environment setup |
| [`docs/seattle_training_run.md`](./docs/seattle_training_run.md) | 6-day unattended training run design (Feb 12-17) |
| [`docs/roadmap.md`](./docs/roadmap.md) | Full project roadmap: Mac → training → RTX 2070 deployment |
| [`docs/training_plan.md`](./docs/training_plan.md) | Training schedule, data sources, go/no-go gates |
| [`todo.md`](./todo.md) | Phased task list aligned to Seattle training schedule |

## License

Private project. All Bible translation training data uses public domain or CC-licensed sources only.
