# stark-translate

[![Lint](https://github.com/wrbell/stark-translate/actions/workflows/lint.yml/badge.svg)](https://github.com/wrbell/stark-translate/actions/workflows/lint.yml)
[![Test](https://github.com/wrbell/stark-translate/actions/workflows/test.yml/badge.svg)](https://github.com/wrbell/stark-translate/actions/workflows/test.yml)
[![Security](https://github.com/wrbell/stark-translate/actions/workflows/security.yml/badge.svg)](https://github.com/wrbell/stark-translate/actions/workflows/security.yml)
[![codecov](https://codecov.io/gh/wrbell/stark-translate/graph/badge.svg)](https://codecov.io/gh/wrbell/stark-translate)

Fully on-device, live bilingual speech-to-text for church outreach at Stark Road Gospel Hall (Farmington Hills, MI). English/Spanish, real-time mic input, browser display. No cloud APIs, no internet required at runtime.

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

Runs on Apple Silicon (MLX) or NVIDIA GPUs (CUDA) via `--backend auto|mlx|cuda`.

## Quick Start

```bash
# Mac
brew install ffmpeg portaudio
python3.11 -m venv stt_env && source stt_env/bin/activate
pip install -r requirements-mac.txt
huggingface-cli login          # Required for TranslateGemma
python setup_models.py         # Download all models
python dry_run_ab.py           # 4B only (~4.3 GB), or --ab for A/B (~11.3 GB)

# NVIDIA
pip install -r requirements-nvidia.txt
python dry_run_ab.py --backend=cuda --no-ab
```

Key flags: `--lang es` (Spanish speaker mode), `--tts` (audio output), `--ab` (A/B comparison), `--dry-run-text "test"` (no mic), `--vad-threshold 0.3`, `--log-level DEBUG`.

## Models

| Component | Model | Size | Latency |
|-----------|-------|------|---------|
| STT | Whisper Large-V3-Turbo | ~1.5 GB | ~500ms |
| Translation (partials) | MarianMT opus-mt-en-es / es-en | ~298 MB | ~250ms |
| Translation A (finals) | TranslateGemma 4B 4-bit | ~2.5 GB | ~550ms |
| Translation B (finals) | TranslateGemma 12B 4-bit | ~7 GB | ~2.1s |
| TTS | Piper EN/ES (ONNX) | ~63 MB | ~40ms/word |
| VAD | Silero VAD | ~2 MB | <1ms |

CUDA variants: bitsandbytes NF4 for TranslateGemma, faster-whisper INT8 for STT. Pipeline overlap hides translation latency by running translation(N) concurrent with STT(N+1).

## Displays

Five browser-based displays served over LAN. Phones connect via QR code on the audience display.

| Display | Purpose |
|---------|---------|
| `audience_display.html` | Projector: EN/ES side-by-side, fading context, fullscreen, QR overlay |
| `ab_display.html` | Operator: 4B / MarianMT / 12B comparison with latency stats |
| `mobile_display.html` | Phone/tablet: responsive, model toggle, Spanish-only mode |
| `church_display.html` | Simplified church layout |
| `obs_overlay.html` | Transparent overlay for OBS Studio streaming |

## Training

Fine-tuning runs on Windows/WSL (A2000 Ada 16GB). Adapters transfer to Mac for inference.

**Translation (TranslateGemma QLoRA):** S1-S9 ablation sweep complete. S6 winner: balanced 1:1 verse/sermon ratio, COMET proximity to 12B base = -0.0002. Trained on ~155K biblical verse pairs (public domain KJV/ASV/WEB/BBE/YLT paired with RVR1909) + DeepL-augmented sermon pairs.

**STT (Whisper LoRA):** W12 data scaling run on 198K Deepgram-aligned chunks from 328 sermons. Baseline WER on fresh eval: 21.41%. W15 hard example mining pipeline for curriculum learning: mine chunks by WER, filter by difficulty bounds, train with `--init-from` for adapter weight initialization.

**Data pipeline:** Deepgram Nova-3 oracle transcription (50 theological keyterms), tiered glossary (50 boost + 229 master terms), sharded Arrow alignment (12GB memory cap), SHA-256 data lockfile, stratified eval sets.

## Hardware

| Target | RAM/VRAM | Config |
|--------|----------|--------|
| Mac (M1-M4) 8 GB+ | ~4.3 GB | 4B-only (`--no-ab`) |
| Mac (M1-M4) 18 GB+ | ~11.3 GB | Full A/B (`--ab`) |
| NVIDIA 6 GB+ | ~4.7 GB | 4B + Whisper (`--backend cuda`) |
| NVIDIA 15 GB+ | ~12 GB | Full A/B on CUDA |
| Training (A2000 Ada 16GB) | ~8-12 GB | LoRA/QLoRA fine-tuning |

## Testing & CI

```bash
pytest tests/ -v                    # 806 tests, no GPU required
ruff check . && ruff format --check .
mypy engines/ settings.py
```

Seven CI workflows: lint, test (3.11 + 3.12), security (pip-audit), release, label, commitlint, stale. CalVer versioning (`YYYY.M.W.PATCH`), Codecov coverage, Dependabot.

## Project Structure

```
dry_run_ab.py                  Main pipeline: mic → VAD → STT → translate → display
settings.py                    Unified config (pydantic-settings, STARK_ prefix)
setup_models.py                One-command model download

engines/                       STT + translation + TTS engine layer
  base.py                      ABCs and result dataclasses
  mlx_engine.py                Apple Silicon (MLX) implementations
  cuda_engine.py               NVIDIA CUDA implementations (streaming, prompt cache)
  factory.py                   Auto-detect backend and create engines

displays/                      5 browser display modes (static HTML/CSS/JS)

training/                      Windows/WSL training scripts
  train_whisper.py             Whisper LoRA (curriculum learning, --init-from)
  train_gemma.py               TranslateGemma QLoRA
  align_deepgram_chunks.py     Deepgram-Whisper alignment (sharded Arrow)
  mine_hard_examples.py        Hard example mining (batched fp16, Tier 1 detection)
  build_hard_subset.py         WER-bounded filtering with stratified caps
  benchmark_gemma4.py          TranslateGemma vs Gemma 4 comparison

tools/                         Monitoring & validation
  live_caption_monitor.py      YouTube caption comparison
  translation_qe.py            3-tier translation quality estimation
  validate_session.py          Post-session validation pipeline
  manage_adapters.py           Adapter lifecycle (register, activate, rollback)
  health_check.py              5-canary-sentence adapter verification
  glossary.py                  Tiered glossary (50 boost + 229 master terms)

features/                      Post-processing (not yet integrated with live pipeline)
  diarize.py                   Speaker diarization
  extract_verses.py            Bible verse reference extraction
  summarize_sermon.py          Post-sermon summary
```

## Documentation

| Doc | Contents |
|-----|----------|
| [`CLAUDE.md`](./CLAUDE.md) | Project overview, architecture, CI/CD, phase checklist |
| [`CLAUDE-macbook.md`](./CLAUDE-macbook.md) | Mac inference environment |
| [`CLAUDE-windows.md`](./CLAUDE-windows.md) | Windows/WSL training environment |
| [`engines/CLAUDE.md`](./engines/CLAUDE.md) | Engine layer: MLX thread safety, CUDA streaming, VRAM tiers |
| [`training/CLAUDE.md`](./training/CLAUDE.md) | Fine-tuning: data pipeline, LoRA/QLoRA configs, ablation results |
| [`tools/CLAUDE.md`](./tools/CLAUDE.md) | Monitoring: YouTube comparison, translation QE, adapter deployment |
| [`displays/CLAUDE.md`](./displays/CLAUDE.md) | Display modes, WebSocket protocol |
| [`features/CLAUDE.md`](./features/CLAUDE.md) | Diarization, sermon summary, verse extraction |
| [`docs/roadmap.md`](./docs/roadmap.md) | Full project roadmap and metrics |

## Status

**Done:** Bidirectional EN/ES inference (MLX + CUDA), two-pass pipeline with overlap, 5 display modes, Piper TTS, TranslateGemma S1-S9 ablation (S6 winner), Whisper W12 data scaling (198K chunks), W15 hard mining pipeline, CUDA streaming runtime with prompt cache + speculative decoding, Deepgram oracle (35 sermons), data integrity pipeline, 806 tests, 7 CI workflows.

**Next:** Continue Whisper curriculum learning (target WER <10%), evaluate Gemma 4 models, deploy adapters to Mac for live A/B, active learning feedback loop (3-5 cycles), Hindi & Chinese adapters.

## License

Private project. All Bible translation training data uses public domain or CC-licensed sources only.
