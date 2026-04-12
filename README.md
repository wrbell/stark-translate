# stark-translate

[![Lint](https://github.com/wrbell/stark-translate/actions/workflows/lint.yml/badge.svg)](https://github.com/wrbell/stark-translate/actions/workflows/lint.yml)
[![Test](https://github.com/wrbell/stark-translate/actions/workflows/test.yml/badge.svg)](https://github.com/wrbell/stark-translate/actions/workflows/test.yml)
[![Security](https://github.com/wrbell/stark-translate/actions/workflows/security.yml/badge.svg)](https://github.com/wrbell/stark-translate/actions/workflows/security.yml)
[![codecov](https://codecov.io/gh/wrbell/stark-translate/graph/badge.svg)](https://codecov.io/gh/wrbell/stark-translate)

Live bidirectional (English/Spanish) speech-to-text for church outreach at Stark Road Gospel Hall, Farmington Hills, MI.

Real-time mic input, fully on-device transcription and translation, displayed in browser. Supports both English and Spanish speakers via `--lang` flag. Uses a two-pass pipeline for fast partials and high-quality finals, with A/B model comparison for translation quality analysis.

## Architecture

```
                              Two-Pass Pipeline
                              ================

  Mic (48kHz) ──> Resample 16kHz (<1ms) ──> Silero VAD (<1ms) ──┐
                                                                  │
            ┌─────────────────────────────────────────────────────┘
            │
            ├─ PARTIAL (every 0.6s of new speech, while speaker is talking)
            │    Whisper Large-V3-Turbo STT (~500ms)
            │    MarianMT EN↔ES PyTorch (~250ms)             ← italic in UI
            │    Total: ~750ms
            │
            └─ FINAL (on 0.5s silence gap or 8s max utterance)
                 Whisper Large-V3-Turbo STT (~500ms)
                 TranslateGemma 4B EN↔ES (~550ms)            ← replaces partial
                 ├─ Piper TTS (~40ms/word EN, --tts)         ← audio output
                 TranslateGemma 12B EN↔ES (~2.1s, --ab)      ← side-by-side
                 Total: ~1.1s (4B) / ~2.6s (A/B sequential)

                 Pipeline overlap: translation runs on utterance N
                 while STT runs on utterance N+1, hiding translation latency.
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

Inference runs on Apple Silicon (MLX) or NVIDIA GPUs (CUDA). No cloud APIs, no internet required at runtime.

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

# Run — Spanish speaker mode (ES→EN translation)
python dry_run_ab.py --lang es

# Open displays in browser
open displays/audience_display.html
open displays/ab_display.html
```

### NVIDIA/CUDA Setup

```bash
# NVIDIA/CUDA setup (RTX 2070 or similar)
pip install -r requirements-nvidia.txt
python dry_run_ab.py --backend=cuda --no-ab
```

### Key Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--lang` | en | Source language: `en` (English→Spanish) or `es` (Spanish→English) |
| `--ab` | off | Load both 4B and 12B for A/B comparison |
| `--backend` | auto | Inference backend: auto, mlx, cuda, cpu |
| `--no-ab` | off | Skip 12B model (for low-VRAM devices) |
| `--low-vram` | off | Marian-only translation mode |
| `--dry-run-text` | off | Test with text input instead of mic |
| `--http-port` | 8080 | HTTP server port (serves display pages to phones over LAN) |
| `--ws-port` | 8765 | WebSocket server port |
| `--vad-threshold` | 0.3 | VAD speech detection sensitivity (0-1) |
| `--gain` | auto | Mic gain multiplier (auto-calibrates by default) |
| `--device` | auto | Audio input device index |
| `--chunk-duration` | 2.0 | Seconds of speech to accumulate |
| `--tts` | off | Enable Piper TTS audio synthesis for translated text |
| `--tts-output` | ws | TTS output mode: ws (WebSocket), wav (file), both |
| `--log-level` | WARNING | Logging level: DEBUG, INFO, WARNING, ERROR |

## Testing

```bash
# Run full test suite (806 tests, no GPU required)
pytest tests/ -v

# With coverage report
pytest tests/ -v --cov=engines --cov=tools --cov=features --cov-report=term-missing

# Lint
ruff check . && ruff format --check .
mypy engines/ settings.py
```

Tests run on CI (Ubuntu, Python 3.11 + 3.12) without GPU or model downloads. Heavy ML dependencies are mocked.

## Models

| Component | Model | Framework | Size | Latency (typical) |
|-----------|-------|-----------|------|--------------------|
| VAD | Silero VAD | PyTorch | ~2 MB | <1ms |
| STT | Whisper Large-V3-Turbo | mlx-whisper / faster-whisper | ~1.5 GB | ~500ms |
| Translate (partials) | MarianMT opus-mt-en-es / es-en | PyTorch (CPU) | ~298 MB | ~250ms |
| Translate A (finals) | TranslateGemma 4B 4-bit | mlx-lm | ~2.5 GB | ~550ms |
| Translate B (finals) | TranslateGemma 12B 4-bit | mlx-lm | ~7 GB | ~2.1s |
| TTS (EN) | Piper en_US-lessac-high | ONNX Runtime | ~63 MB | ~40ms/word |
| TTS (ES) | Piper es_MX-claude-high | ONNX Runtime | ~63 MB | ~8ms/word |

CUDA variants: bitsandbytes 4-bit for TranslateGemma, faster-whisper INT8 for STT.

Pipeline overlap (P7-6C) hides translation latency by running translation on utterance N while STT processes utterance N+1.

## Features

- **Bidirectional language support** -- `--lang en` (English→Spanish, default) or `--lang es` (Spanish→English) with automatic model selection
- **Two-pass STT pipeline** -- fast italic partials (MarianMT, ~750ms) replaced by high-quality finals (TranslateGemma, ~1.1s) on silence detection
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
- **Piper TTS audio synthesis** -- `--tts` flag enables text-to-speech for translated text (EN + ES voices, ONNX, thread-safe)
- **End-to-end roundtrip quality testing** -- `tools/roundtrip_test.py` measures STT WER and roundtrip translation accuracy
- **Post-session validation pipeline** -- YouTube WER comparison with text-anchor alignment (`tools/validate_session.py`)
- **Deepgram Nova-3 oracle transcription** -- ground-truth labeling with 50 theological keyterm boosting
- **Tiered glossary system** -- Tier 1 (50 boost terms for Deepgram) + Tier 2 (229 master terms for training)
- **Data integrity pipeline** -- SHA-256 lockfile, stratified eval sets, adapter health checks
- **TranslateGemma fine-tuning** -- S1-S9 ablation sweep (S6 winner: balanced ratio, COMET prox 12B = -0.0002)
- **Whisper LoRA fine-tuning** -- W12 data scaling (198K chunks, 21.41% baseline WER) + W15 hard example mining with curriculum learning
- **Hard example mining** -- mine → filter by WER bounds → stratified caps → curriculum train with `--init-from`
- **Dual-target inference** -- runs on Apple Silicon (MLX) or NVIDIA GPUs (CUDA) from a single codebase
- **Unified configuration** -- pydantic-settings with STARK_ env prefix, .env file support
- **STT fallback** -- automatic retry with fallback model on low-confidence or hallucinated segments

## Hardware Requirements

### Mac (Inference)

- Apple Silicon (M1/M2/M3/M4)
- **8 GB+** unified memory for 4B-only mode (~4.3 GB used)
- **18 GB+** unified memory for A/B mode (~11.3 GB used)
- Python 3.11, macOS with Metal support

### NVIDIA (Inference -- RTX 2070 or similar)

- NVIDIA GPU with 6 GB+ VRAM
- CUDA 12.x toolkit
- Python 3.11, Windows or Linux
- `pip install -r requirements-nvidia.txt`

### Windows (Training)

- NVIDIA GPU with 16 GB+ VRAM (tested on A2000 Ada)
- 64 GB+ system RAM recommended
- WSL2 with CUDA toolkit
- Used for audio preprocessing (demucs), pseudo-labeling (Whisper large-v3), and LoRA/QLoRA fine-tuning

## Training

Fine-tuning runs on the Windows desktop and adapters transfer to the Mac for inference:

- **Whisper LoRA** (r=32) on 20-50 hours of church sermon audio, with accent-balanced sampling across Midwest/Scottish/British/Canadian accents
- **TranslateGemma QLoRA** (r=16, 4-bit NF4) on ~155K biblical verse pairs (public domain) for Spanish; Hindi and Chinese adapters planned (~155K-310K pairs each)
- **MarianMT full fine-tune** as a lightweight fallback (298 MB)

Training data: church audio via yt-dlp + Bible parallel corpus (KJV/ASV/WEB/BBE/YLT paired with RVR1909). Accent-diverse audio tagged via `--accent` flag and balanced with temperature-based sampling. See [`CLAUDE.md`](./CLAUDE.md) for the full architecture, training strategy, and compute timeline.

## Project Structure

```
├── dry_run_ab.py              # Main pipeline: mic → VAD → STT → translate → WebSocket + HTTP
├── settings.py                # Unified pydantic-settings config (STARK_ env prefix, .env support)
├── setup_models.py            # One-command model download + verification
├── build_glossary.py          # EN→ES theological glossary (229 terms)
├── download_sermons.py        # yt-dlp sermon downloader
├── requirements-mac.txt       # Mac/MLX pip dependencies
├── requirements-nvidia.txt    # NVIDIA/CUDA inference dependencies
│
├── engines/                   # STT + translation + TTS engine abstraction (MLX + CUDA)
│   ├── base.py                # ABCs: STTEngine, TranslationEngine, TTSEngine, result dataclasses
│   ├── mlx_engine.py          # MLXWhisperEngine, MLXGemmaEngine, MarianEngine, PiperTTSEngine
│   ├── cuda_engine.py         # FasterWhisperEngine, CUDAGemmaEngine
│   ├── factory.py             # create_stt_engine(), create_translation_engine(), create_tts_engine()
│   └── active_learning.py     # Fallback event JSONL logger
│
├── displays/
│   ├── audience_display.html  # Projector display (EN/ES side-by-side, QR overlay)
│   ├── ab_display.html        # A/B/C operator comparison display
│   ├── mobile_display.html    # Phone/tablet responsive display
│   ├── church_display.html    # Simplified church layout
│   └── obs_overlay.html       # Transparent overlay for OBS Studio
│
├── training/                  # Windows/WSL training scripts (CUDA)
│   ├── preprocess_audio.py    # 10-step audio cleaning pipeline (accent-aware)
│   ├── transcribe_church.py   # Whisper large-v3 pseudo-labeling
│   ├── transcribe_with_deepgram.py # Deepgram Nova-3 oracle transcription
│   ├── align_deepgram_chunks.py # Align Deepgram words to Whisper chunk boundaries
│   ├── prepare_bible_corpus.py # Bible verse pair alignment
│   ├── prepare_whisper_dataset.py # Accent-balanced audiofolder builder
│   ├── prepare_piper_dataset.py # LJSpeech format conversion for Piper TTS
│   ├── train_whisper.py       # Whisper LoRA fine-tuning (curriculum learning, --init-from)
│   ├── train_gemma.py         # TranslateGemma QLoRA fine-tuning
│   ├── train_marian.py        # MarianMT full fine-tune
│   ├── train_piper.py         # Piper TTS voice fine-tuning
│   ├── mine_hard_examples.py  # Hard example mining (batched fp16, per-chunk WER, Tier 1)
│   ├── build_hard_subset.py   # WER-bounded filtering with stratified caps
│   ├── filter_chunks_by_confidence.py # Quality-ranked chunk selection
│   ├── recover_shards.py      # Rebuild DatasetDict from Arrow shards after OOM
│   ├── benchmark_gemma4.py    # TranslateGemma vs Gemma 4 comparison benchmark
│   ├── export_piper_onnx.py   # Piper TTS model export to ONNX
│   ├── evaluate_translation.py # SacreBLEU/chrF++/COMET scoring
│   ├── evaluate_piper.py      # Piper TTS quality assessment
│   └── assess_quality.py      # Baseline WER assessment
│
├── tools/                     # Mac benchmarking & monitoring
│   ├── live_caption_monitor.py # YouTube caption comparison (post/live/trend)
│   ├── translation_qe.py      # Reference-free translation QE
│   ├── benchmark_latency.py   # End-to-end latency profiling
│   ├── stt_benchmark.py       # STT-only benchmarking
│   ├── roundtrip_test.py      # End-to-end STT + translation roundtrip quality test
│   ├── validate_session.py    # Post-session validation vs YouTube captions
│   ├── prepare_finetune_data.py # Fine-tuning data export from live sessions
│   ├── download_roundtrip_texts.py # Download test texts for roundtrip testing
│   ├── convert_models_to_both.py # Model format conversion (MLX ↔ CUDA)
│   └── test_adaptive_model.py # Adaptive model selection testing
│
├── features/                  # Standalone future features
│   ├── diarize.py             # Speaker diarization (pyannote-audio)
│   ├── extract_verses.py      # Bible verse reference extraction
│   └── summarize_sermon.py    # Post-sermon 5-sentence summary
│
├── docs/
│   ├── immediate_todo.md      # Live demo session notes + session issues
│   ├── todo.md                # Phased task list
│   ├── release_plan.md        # Release planning
│   ├── optimized.md           # NVIDIA C++ inference optimization plan
│   ├── deploy.md              # Automated adapter deployment system plan
│   ├── training_plan.md       # Full training schedule + go/no-go gates
│   ├── roadmap.md             # Mac → Windows → RTX 2070 deployment roadmap
│   ├── accent_tuning_plan.md  # 4-week accent-diverse STT tuning plan
│   ├── macos_libomp_fix.md    # libomp conflict diagnosis + fix
│   └── ...                    # 18 docs total
│
├── stark_data/                # Church audio + transcripts + corrections
├── bible_data/                # Biblical parallel text corpus (155K pairs)
└── metrics/                   # CSV logs, diagnostics JSONL, hardware profiles
```

## Docs

| Doc | Contents |
|-----|----------|
| [`CLAUDE.md`](./CLAUDE.md) | Project overview, 6-layer architecture summary, CI/CD, phase checklist |
| [`CLAUDE-macbook.md`](./CLAUDE-macbook.md) | Mac inference environment setup |
| [`CLAUDE-windows.md`](./CLAUDE-windows.md) | Windows/WSL training environment setup |
| [`docs/optimized.md`](./docs/optimized.md) | NVIDIA C++ inference optimization plan (llama.cpp / exllamav2) |
| [`docs/deploy.md`](./docs/deploy.md) | Automated adapter deployment, health checks, hot-reload, rollback |
| [`docs/immediate_todo.md`](./docs/immediate_todo.md) | Live demo session notes and immediate action items |
| [`docs/roadmap.md`](./docs/roadmap.md) | Full project roadmap: Mac → training → RTX 2070 deployment |
| [`docs/training_plan.md`](./docs/training_plan.md) | Training schedule, data sources, go/no-go gates |
| [`docs/accent_tuning_plan.md`](./docs/accent_tuning_plan.md) | 4-week accent-diverse STT tuning plan (code complete) |
| [`docs/multi_lingual.md`](./docs/multi_lingual.md) | Hindi & Chinese actionable todo list |
| [`docs/macos_libomp_fix.md`](./docs/macos_libomp_fix.md) | macOS libomp conflict diagnosis and fix |
| [`docs/todo.md`](./docs/todo.md) | Phased task list |
| [`engines/CLAUDE.md`](./engines/CLAUDE.md) | Engine layer: MLX thread safety, model IDs, confidence thresholds, critical fixes |
| [`training/CLAUDE.md`](./training/CLAUDE.md) | Fine-tuning: audio preprocessing, Bible corpus, LoRA/QLoRA configs, compute timeline |
| [`tools/CLAUDE.md`](./tools/CLAUDE.md) | Monitoring: YouTube comparison, text-anchor alignment, translation QE tiers |
| [`displays/CLAUDE.md`](./displays/CLAUDE.md) | Browser displays: WebSocket protocol, HTTP serving, display modes |
| [`features/CLAUDE.md`](./features/CLAUDE.md) | Post-processing: diarization, sermon summary, verse extraction |

## Development Status

**What's done:**
- Bidirectional language support: `--lang en` (EN→ES) and `--lang es` (ES→EN) with automatic model selection
- Wholesale swap to Whisper Large-V3-Turbo (both partials and finals)
- `engines/` package: MLX + CUDA engine implementations with factory auto-detection
- CUDA streaming inference runtime: `CUDAGemmaStreamingEngine` with prompt cache, speculative decoding, VRAM tier auto-detection
- `settings.py`: pydantic-settings unified config (`STARK_` env prefix, `.env` support, `CUDASettings`)
- Backend selection (`--backend auto|mlx|cuda`) with CUDA fallback paths
- STT fallback logic (lazy-load fallback model on low confidence / hallucination)
- Piper TTS engine integration (`--tts` flag, WebSocket + WAV output, EN + ES voices)
- Validation pipeline (`tools/validate_session.py`) with text-based anchor alignment (19.6% WER on live session)
- Roundtrip quality test: STT WER 3.8-8.8%, roundtrip WER ~56%, ~134ms/word (`tools/roundtrip_test.py`)
- TranslateGemma S1-S9 ablation: S6 winner (balanced 1:1, COMET prox 12B = -0.0002)
- Whisper W12 data scaling: 198K chunks from 328 sermons, 21.41% baseline WER
- W15 hard example mining: curriculum learning pipeline (mine → filter → train with `--init-from`)
- Deepgram Nova-3 oracle: 35 sermons transcribed with 50 theological keyterms
- Alignment hardening: sharded Arrow writes (1000 rows/shard), 12GB memory cap, crash recovery
- Data integrity: SHA-256 lockfile, stratified eval sets, adapter health checks, training manifests
- Fine-tuning data prep tools (review queue + dataset export)
- CI/CD pipeline: 7 GitHub Actions workflows (lint, test, security, release, label, commitlint, stale) + Codecov
- 806 tests with coverage threshold (≥18%), pre-commit hooks, CalVer versioning
- Dependabot for automated dependency updates
- Structured logging (`--log-level`, session log files, VAD event logging)
- Rolling session stats (5-min averages broadcast to operator displays)

**What's next:**
- Whisper curriculum learning: continue W15 mine → filter → train cycles to drive WER below 10%
- Gemma 4 benchmark: evaluate E2B/E4B vs TranslateGemma on theological translation
- Adapter evaluation & transfer: deploy best adapters to Mac, re-run A/B comparison
- Active learning feedback loop: flag → correct → retrain (3-5 cycles)
- Hindi & Chinese translation adapters
- See [`docs/roadmap.md`](./docs/roadmap.md) for full project roadmap

## License

Private project. All Bible translation training data uses public domain or CC-licensed sources only.
