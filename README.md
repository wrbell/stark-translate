# stark-translate

[![Lint](https://github.com/wrbell/stark-translate/actions/workflows/lint.yml/badge.svg)](https://github.com/wrbell/stark-translate/actions/workflows/lint.yml)
[![Test](https://github.com/wrbell/stark-translate/actions/workflows/test.yml/badge.svg)](https://github.com/wrbell/stark-translate/actions/workflows/test.yml)
[![Security](https://github.com/wrbell/stark-translate/actions/workflows/security.yml/badge.svg)](https://github.com/wrbell/stark-translate/actions/workflows/security.yml)
[![codecov](https://codecov.io/gh/wrbell/stark-translate/graph/badge.svg)](https://codecov.io/gh/wrbell/stark-translate)

Fully on-device, live bilingual speech-to-text for church outreach at Stark Road Gospel Hall (Farmington Hills, MI). English/Spanish, real-time mic input, browser display. No cloud APIs, no internet required at runtime.

> **Release lines:** **main** is v2026.13. The v2026.14 candidate on
> `codex/mac-reliability-roadmap` adds reliability, schema 2 timing, setup, and
> Review/export — see [`docs/current_architecture.md`](docs/current_architecture.md)
> and [`docs/mac_implementation_status.md`](docs/mac_implementation_status.md).
> Remaining work: [`docs/backlog.json`](docs/backlog.json).

## Architecture

```
                              Two-Pass Pipeline
                              ================

  Mic (48kHz) ──> Resample 16kHz (<1ms) ──> Silero VAD (<1ms) ──┐
                                                                  │
            ┌─────────────────────────────────────────────────────┘
            │
            ├─ PARTIAL (every 0.6s of new speech, while speaker is talking)
            │    Mac: Parakeet EN / Whisper ES · Marian CT2 CPU partial
            │    CUDA: W16 Whisper CT2 + Marian CT2            ← italic in UI
            │
            └─ FINAL (on 0.5s silence gap or 8s max utterance)
                 Same STT · Gemma 4 E4B (MLX OptiQ Mac / llama.cpp CUDA)
                 ← replaces partial
                 ├─ Piper TTS (~40ms/word EN, --tts)         ← audio output
                 Gemma 4 E2B Q4_K_M via llama.cpp (~280ms)   ← low-VRAM fallback
                 Total: ~820ms (E4B) / ~630ms (E2B)

    STT measurements: A2000 Ada Mobile (165 GB/s BW), beam_size=1, 41-clip
    bench on 1-15s utterances. WER 11.00% overall / 8.70% on theological
    terms (W16 fine-tune drops WER 19% / 43% relative vs off-the-shelf).
    Full bench: docs/archive/v2026.7/STT_BENCHMARK.md.
    Faster GPUs scale ~linearly with memory bandwidth (RTX 3060 12GB ≈ ~190ms p95).

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

Run these commands from a checkout or extracted Mac source ZIP. Package publication
is pending; a locally built wheel or source install provides the current code.
See [Mac installation and readiness](docs/packaging/macos.md) for full setup and
optional evaluation/diarization dependencies.

```bash
# Mac (Apple Silicon)
brew install ffmpeg portaudio
python3.11 -m venv venv
venv/bin/python -m pip install '.[mlx]'
venv/bin/stark-translate setup --backend mlx    # Mac defaults, both languages
venv/bin/stark-translate doctor --backend mlx --lang en
VENV="$PWD/venv" ./run_operator.sh              # Open /operator/ on port 9000

# NVIDIA (Linux)
python3.11 -m venv venv
venv/bin/python -m pip install '.[cuda]'
venv/bin/stark-translate setup --backend cuda
venv/bin/stark-translate operator

# Optional MarianMT CT2 conversion on Mac.
# Skips automatically if adapters/marian_ct2/<dir>/active/model.bin already exists.
venv/bin/python scripts/convert_marian_ct2.py \
    --model-id Helsinki-NLP/opus-mt-en-es \
    --output adapters/marian_ct2/en-es/active --quantization int8
venv/bin/python scripts/convert_marian_ct2.py \
    --model-id Helsinki-NLP/opus-mt-es-en \
    --output adapters/marian_ct2/es-en/active --quantization int8
```

The legacy `requirements-mac.txt` / `requirements-nvidia.txt` install path is
still supported for the WSL training environment but deprecated for inference;
see `CLAUDE-windows.md` for training-side instructions.

Key flags: `--lang es` (Spanish speaker mode), `--tts` (audio output), `--ab` (A/B comparison), `--dry-run-text "test"` (no mic), `--vad-threshold 0.3`, `--log-level DEBUG`.

## Models

| Component | Model | Size | Latency (CUDA) |
|-----------|-------|------|----------------|
| **STT default** (v2026.7) | **Whisper Large-V3-Turbo + W16 LoRA → CTranslate2 int8_float16** | **~1.5 GB** | **see `docs/archive/v2026.7/STT_BENCHMARK.md`** |
| STT (off-the-shelf fallback) | faster-whisper large-v3-turbo (CT2) | ~1.5 GB | ~500ms (pre-W16) |
| STT (alt path) | HF Whisper + distil-large-v3.5 spec decode | ~3 GB | varies |
| **Translation default (partials, v2026.8)** | **MarianMT opus-mt-{en-es,es-en} → CTranslate2 int8_float16** | **~80 MB / dir** | **~57 ms p50 / ~116 ms p95 — see `docs/archive/v2026.8/MARIAN_BENCHMARK.md`** |
| Translation (partials, HF fallback) | MarianMT opus-mt-{en-es,es-en} HF transformers | ~298 MB | ~167ms CUDA / ~360ms CPU |
| **Translation default** (CUDA, finals) | **Gemma 4 E4B Q4_K_M (llama.cpp)** | **5.0 GB GGUF / 4.9 GB VRAM** | **~470ms** |
| Translation low-VRAM (CUDA, finals) | Gemma 4 E2B Q4_K_M (llama.cpp) | 3.2 GB GGUF / 3.5 GB VRAM | ~280ms |
| Translation (Mac MLX, default) | **Gemma 4 E4B OptiQ-4bit** | OptiQ | ~2–3s medium |
| Translation (Mac MLX, opt-out) | TranslateGemma 4B / 12B 4-bit | ~2.5 GB / ~7 GB | ~550ms / ~2.1s |
| TTS | Piper EN/ES (ONNX) | ~63 MB | ~40ms/word |
| VAD | Silero VAD | ~2 MB | <1ms |

CUDA path now uses **llama.cpp via `engines/llamacpp_engine.py`** (v2026.5+) for ~5–9× speedup and ~4× VRAM reduction vs HF NF4. Caller starts `llama-server` (see `start_server.sh`). **Mac MLX default is Gemma 4 OptiQ E4B** (`--model-family translategemma` to opt out). Pipeline overlap hides translation latency by running translation(N) concurrent with STT(N+1). See [`docs/mlx_cuda_parity.md`](./docs/mlx_cuda_parity.md) and [`docs/archive/v2026.5/BENCHMARK.md`](./docs/archive/v2026.5/BENCHMARK.md).

## Displays

Five browser-based displays served over LAN. Phones connect via QR code on the audience display.

| Display | Purpose |
|---------|---------|
| `audience_display.html` | Projector: EN/ES side-by-side, fading context, fullscreen, QR overlay |
| `ab_display.html` | Operator: A (default) / MarianMT / B comparison with latency stats |
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
| Mac (M1-M4) 8 GB+ | ~5–6 GB | Gemma 4 OptiQ E4B (or `--gemma4-size e2b`) |
| Mac (M1-M4) 18 GB+ | ~11.3 GB | TG A/B (`--model-family translategemma --ab`) or E4B+MTS |
| **NVIDIA 6 GB+** (RTX 3060/4060) | **~5 GB** | **Whisper + Gemma 4 E2B Q4_K_M (llama.cpp)** |
| NVIDIA 8 GB+ | ~6 GB | Whisper + Gemma 4 E4B Q4_K_M (llama.cpp) |
| Training (A2000 Ada 16GB) | ~8-12 GB | LoRA/QLoRA fine-tuning |

> **CUDA HF NF4 not recommended:** Gemma 4 E2B/E4B HF NF4 occupy 14–15 GB on GPU due to bf16 Per-Layer Embeddings. Use llama.cpp Q4_K_M instead (4× less VRAM). See [`docs/archive/v2026.5/BENCHMARK.md`](./docs/archive/v2026.5/BENCHMARK.md).

## Testing & CI

```bash
pytest tests/ -v
ruff check . && ruff format --check .
mypy engines/ settings.py
python tools/render_backlog.py validate
pytest tests/test_documentation.py -v
```

Seven CI workflows: lint, test (3.11 + 3.12, coverage gate in `test.yml`), security
(pip-audit), release, label, commitlint, stale. CalVer in `pyproject.toml`.
Latest validated CPU suite counts: [`docs/mac_implementation_status.md`](docs/mac_implementation_status.md).

## Project Structure

```
dry_run_ab.py                  Main pipeline: mic → VAD → STT → translate → display
settings.py                    Unified config (pydantic-settings, STARK_ prefix)
(deleted in v2026.7 cleanup; replaced by `stark-translate setup`)

engines/                       STT + translation + TTS engine layer
  base.py                      ABCs and result dataclasses
  mlx_engine.py                Apple Silicon (MLX) implementations
  cuda_engine.py               NVIDIA CUDA HF implementations (streaming, prompt cache)
  llamacpp_engine.py           NVIDIA CUDA via llama.cpp HTTP (recommended for production, v2026.5+)
  factory.py                   Auto-detect backend and create engines
start_server.sh                Launch llama-server with default Gemma 4 GGUFs

displays/                      5 browser display modes (static HTML/CSS/JS)

training/                      Windows/WSL training scripts
  train_whisper.py             Whisper LoRA (curriculum learning, --init-from)
  train_gemma.py               TranslateGemma QLoRA
  align_deepgram_chunks.py     Deepgram-Whisper alignment (sharded Arrow)
  mine_hard_examples.py        Hard example mining (batched fp16, Tier 1 detection)
  build_hard_subset.py         WER-bounded filtering with stratified caps
  benchmark_gemma4.py          TranslateGemma vs Gemma 4 comparison (HF only)
scripts/benchmarks/bench_translate_t1_t4.py       Phase 1A benchmark: HF vs llama.cpp, all 4 models, VRAM sampler

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
| [`CLAUDE.md`](./CLAUDE.md) / [`AGENTS.md`](./AGENTS.md) | Project overview, phase checklist (human + agent guides) |
| [`docs/current_architecture.md`](./docs/current_architecture.md) | Current inference/operator contracts (v2026.14 candidate) |
| [`docs/backlog.json`](./docs/backlog.json) | Machine-readable remaining tasks (render: `tools/render_backlog.py`) |
| [`docs/mac_implementation_status.md`](./docs/mac_implementation_status.md) | Local validation evidence and open gates |
| [`CLAUDE-macbook.md`](./CLAUDE-macbook.md) | Mac inference environment |
| [`CLAUDE-windows.md`](./CLAUDE-windows.md) | Windows/WSL training environment |
| [`engines/`](./engines/CLAUDE.md) | Engine layer — see paired `AGENTS.md` in each subdirectory |
| [`training/`](./training/CLAUDE.md) | Fine-tuning and data pipeline |
| [`tools/`](./tools/CLAUDE.md) | Monitoring, QE, adapter deployment |
| [`displays/`](./displays/CLAUDE.md) | Display modes and WebSocket protocol |
| [`features/`](./features/CLAUDE.md) | Diarization, summary, verse extraction |
| [`docs/operator_runbook.md`](./docs/operator_runbook.md) | Day-of-event workflow for non-technical operators |
| [`docs/roadmap.md`](./docs/roadmap.md) | Long-range roadmap and archived metrics |

## Status

**Shipped on main (v2026.13):** bidirectional EN/ES inference, operator control plane,
Mac latency fixes (#180–191), Parakeet EN STT, Marian CT2 Mac path, replay harness,
TTS routing, live diarization code behind `--diarize`. See [`docs/archive/`](docs/archive/)
for version-specific benchmarks — do not treat legacy `e2e_latency_ms` as speech-end-to-display.

**v2026.14 candidate (local branch):** operator reliability, schema 2 timing, reproducible
setup, Review/export, frozen screening — validated in
[`docs/mac_implementation_status.md`](docs/mac_implementation_status.md). Publication and
main merge pending root integration.

**Open gates:** natural Spanish references, bilingual review, two-speaker diarization gate,
physical second output, Sunday dry-run (#134), WSL training cycle, lite CPU and RTX 2070
validation — tracked in [`docs/backlog.json`](docs/backlog.json).

## License

Private project. All Bible translation training data uses public domain or CC-licensed sources only.
