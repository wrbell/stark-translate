# CLAUDE.md — Live Bilingual Speech-to-Text

A fully on-device, live bidirectional speech-to-text system (English/Spanish) for church outreach at Stark Road Gospel Hall (Farmington Hills, MI). Supports `--lang en` (EN→ES) and `--lang es` (ES→EN). Includes Piper TTS (`--tts`). All Python, MLX on Apple Silicon for inference, CUDA on NVIDIA for training.

**Two-pass pipeline** for fast partials and high-quality finals:
- **Partials (while speaking):** mlx-whisper STT + MarianMT PyTorch (~750ms) — displayed in italics
- **Finals (on silence):**
  - **Mac (MLX):** mlx-whisper STT + TranslateGemma 4B/12B 4-bit (~1.1s / ~2.6s) — replaces partial
  - **CUDA (v2026.5+):** faster-whisper + Gemma 4 E4B Q4_K_M via llama.cpp (~1.0s) — production default
- **CUDA model options:** Gemma 4 E2B Q4_K_M (3.5 GB VRAM, ~280ms) for low-VRAM, E4B Q4_K_M (4.9 GB, ~470ms, 7/8 canary) for best quality

---

## Environment Split

| Machine | Role | Doc |
|---------|------|-----|
| **MacBook** (M3 Pro, 18GB, 18-core GPU, Metal 4) | Inference, live demos, monitoring, UI | [`CLAUDE-macbook.md`](./CLAUDE-macbook.md) |
| **Windows Desktop** (WSL2, A2000 Ada 16GB, 64GB RAM) | Audio preprocessing, fine-tuning, training | [`CLAUDE-windows.md`](./CLAUDE-windows.md) |

Model transfer: WSL → copy LoRA adapters to Mac project root.

---

## Architecture: Six Quality Layers

1. **Audio Preprocessing** (WSL) — 10-step pipeline: download → format → quality gate → classify → separate → denoise → normalize → VAD chunk → diarize → final gate. See [`training/CLAUDE.md`](./training/CLAUDE.md).
2. **Data Quality Assessment** (WSL) — Baseline WER sampling, strategy by WER range. See [`training/CLAUDE.md`](./training/CLAUDE.md).
3. **Confidence-Based Flagging** (Mac) — `avg_logprob`, `no_speech_prob`, `compression_ratio` thresholds for review/reject. See [`engines/CLAUDE.md`](./engines/CLAUDE.md).
4. **YouTube Caption Comparison** (Mac) — Local vs YouTube windowed WER, text-anchor alignment for large offsets. See [`tools/CLAUDE.md`](./tools/CLAUDE.md).
5. **Translation QE** (Mac) — 3-tier quality estimation: CometKiwi/LaBSE (always-on), back-translation (triggered), BLASER/BETO (batch). See [`tools/CLAUDE.md`](./tools/CLAUDE.md).
6. **Active Learning Feedback Loop** (both) — infer → flag → correct → retrain. 3–5 cycles typical. First cycle yields 20–40% relative WER reduction.

---

## Recent Additions (2026-04-25)

- **v2026.6 — Phase 9 operator control plane** (PRs #60–#66, tags `v2026.6.0.0` and `v2026.6.1.0`). FastAPI app at `uvicorn operator_app.main:app --port 9000` + vanilla-JS SPA at `/operator/`. Replaces the developer-grade run_church.sh + 5-tab workflow with a single browser UI a non-technical volunteer can drive. Pre-flight gates the Start button; mid-session controls (pause/resume/lang_flip/vad/fallback); live observability sparklines (VRAM/CPU/latency/confidence) over `/ws/control`; audio device enumeration with USB hotplug toast; live verse highlights + post-session summary trigger; systemd unit + launchd plist + bootstrap.sh for first-time install. Operator runbook at [`docs/operator_runbook.md`](./docs/operator_runbook.md).
- **Phase 1D — llama.cpp wired into the live pipeline** (PR #59). `dry_run_ab.py --backend cuda` auto-prefers `LlamaCppEngine` when a llama-server is reachable. `--engine {auto,llamacpp,hf}` and `--llamacpp-url` for explicit control. Repurposed `--ab` mode loads E4B (8090) + E2B (8091) under llama.cpp.
- **v2026.5 release — Phase 1A complete** (`docs/archive/v2026.5/BENCHMARK.md`): Gate 1A passes by 8.9×. llama.cpp-served Gemma 4 GGUF replaces HF NF4 as production CUDA path. T3 (E4B Q4_K_M) is default at 41 tok/s + 7/8 canary in 4.9 GB VRAM. T2 (E2B Q4_K_M) low-VRAM fallback at 66 tok/s in 3.5 GB. Spec decode (T4) deferred — single-GPU loss on this hardware.
- **Production bug fix** — `engines/llamacpp_engine.py` now passes `chat_template_kwargs: {"enable_thinking": false}` for Gemma 4. Without it the model emits chain-of-thought into `reasoning_content` and leaves `content` empty until max_tokens. 14× latency improvement.
- **VRAM accounting corrected** — prior `metrics/gemma4_benchmark/comparison.json` undercounted by 2× because `torch.cuda.max_memory_allocated()` misses bnb scratch + bf16 PLE. New benchmark uses continuous nvidia-smi sampling (`scripts.benchmarks.bench_translate_t1_t4.VramSampler`). HF E2B NF4 actually peaks at 14 GB, not 6 GB.
- **W16 corrective Whisper run** — fresh-eval WER 7.25% (better than W7's 7.61%). W15 hard-only curriculum failure rooted in `--init-from` silently failing to load weights (now fixed).

## Earlier Additions (2026-04-11)

- **Hard Example Mining (W15)** — Curriculum learning pipeline: `mine_hard_examples.py` (batched fp16, per-chunk WER, Tier 1 detection), `build_hard_subset.py` (WER-bounded filtering, stratified caps), `--init-from` in `train_whisper.py` for adapter weight initialization
- **CUDA Streaming Runtime** — `CUDAGemmaStreamingEngine` with TextIteratorStreamer, prompt cache (~50-80ms savings), speculative decoding (4B drafts 12B), VRAM tier auto-detection
- **Alignment Hardening** — Sharded Arrow writes (1000 rows/shard, 5 GB RAM vs 190 GB), 12GB memory cap via `resource.setrlimit`, crash-resume via `recover_shards.py`
- **W12 Data Scaling** — 198K Deepgram-aligned chunks from 328 sermons; baseline WER on fresh eval: 21.41%
- **Deepgram Nova-3 Oracle** — Ground-truth transcription with 50 theological keyterms (`training/transcribe_with_deepgram.py`)
- **Tiered Glossary** — Tier 1 (50 boost terms for Deepgram) + Tier 2 (229 master for training) in `tools/glossary.py`
- **Data Organization** — Sermons sorted into `stt-data/{type}/{year}/` with 2026-03-14 training cutoff (`tools/sort_sermons.py`)
- **Data Integrity** — SHA-256 lockfile (`tools/lock_data.py`), stratified eval sets (`tools/build_eval_sets.py`), training manifests
- **Adapter Management** — Health checks (`tools/health_check.py`), version tracking (`tools/manage_adapters.py`)
- **TranslateGemma Results** — S1-S9 sweep: S6 won (balanced 1:1 verse/sermon, COMET prox 12B = -0.0002)
- **Gemma 4 Benchmark** — 4-model comparison tool (`training/benchmark_gemma4.py`): TranslateGemma 4B/12B vs Gemma 4 E2B/E4B across Bible, sermon, and canary tiers (note: VRAM numbers in result file are PyTorch-only; see Phase 1A for corrected values)

---

## Subdirectory Guides

| Directory | CLAUDE.md Contents |
|-----------|--------------------|
| [`engines/`](./engines/CLAUDE.md) | Engine ABCs, MLX thread safety, model IDs, memory budget, confidence thresholds, critical fixes |
| [`training/`](./training/CLAUDE.md) | Audio preprocessing, data assessment, Bible corpus, LoRA/QLoRA configs, theological vocab, compute timeline |
| [`tools/`](./tools/CLAUDE.md) | YouTube comparison, text-anchor alignment, translation QE tiers, validation pipeline |
| [`displays/`](./displays/CLAUDE.md) | Display modes, WebSocket protocol, HTTP serving, auto-reconnect |
| [`features/`](./features/CLAUDE.md) | Diarization, post-sermon summary, verse extraction |

---

## Extension Patterns

Adding new capabilities follows documented patterns in each subdirectory:

- **New engine backend** (llama.cpp, exllamav2): See `engines/CLAUDE.md` § "Adding a New Engine"
- **New language** (Hindi, Chinese): See `engines/CLAUDE.md` § "Adding a New Language" + `training/CLAUDE.md` § "Adding a New Language Corpus"
- **New display**: See `displays/CLAUDE.md` § "Multi-Language Protocol Extension"
- **Adapter deployment**: See `tools/CLAUDE.md` § "Adapter Deployment Pipeline"
- **Active learning cycle**: See `tools/CLAUDE.md` § "Active Learning Feedback Loop"

---

## CI/CD Pipeline

Seven GitHub Actions workflows:

| Workflow | Trigger | What it does |
|----------|---------|-------------|
| **Lint** (`lint.yml`) | push / PR | Ruff check + format, mypy, bandit, vulture (advisory), HTML tidy |
| **Test** (`test.yml`) | push / PR | pytest (855 tests, Python 3.11 + 3.12), coverage ≥18%, Codecov, PR comment |
| **Release** (`release.yml`) | `v*` tag | Creates GitHub Release |
| **Security** (`security.yml`) | push / PR / weekly | pip-audit on both requirements files |
| **Label** (`label.yml`) | PR | Auto-labels by changed paths |
| **Commitlint** (`commitlint.yml`) | PR | Conventional commit format |
| **Stale** (`stale.yml`) | scheduled | Auto-closes stale issues/PRs |

### Running Locally

```bash
# Lint
ruff check . && ruff format --check .
mypy engines/ settings.py
bandit -r engines/ features/ tools/ settings.py -s B101,B603,B607 --severity-level medium

# Tests
pytest tests/ -v --cov=engines --cov=tools --cov=features --cov-report=term-missing

# Pre-commit (runs ruff + format on staged files)
pre-commit run --all-files
```

### Version Numbering

CalVer: `YYYY.M.W.PATCH` (e.g., `2026.5.0.0`). Single source of truth in `pyproject.toml`.

---

## Next Steps (Ordered)

- [x] **Phase 0 — Setup:** Configure both environments per machine-specific docs
- [x] **Phase 1 — Baseline:** Run base A/B test (no fine-tuning) to establish latency and WER baselines
- [x] **Phase 1.5 — CI/CD:** GitHub Actions, pre-commit, CalVer versioning, 806 tests
- [x] **Phase 1.6 — Spanish STT:** Bidirectional language support, ES→EN translation
- [x] **Phase 1.7 — TTS & Roundtrip:** Piper TTS integration, roundtrip quality test, validation pipeline
- [x] **Phase 2 — Data collection:** Download and sample 10–20 hours of Stark Road audio via `yt-dlp` — 333 sermons cataloged, 160+ downloaded, Deepgram oracle complete (35/35)
- [x] **Phase 3 — Quality assessment:** Manually transcribe 50–100 sample segments, compute baseline WER — 500 stratified verse holdout + 422 sermon eval chunks built
- [ ] **Phase 4 — Preprocessing:** Run the 10-step audio cleaning pipeline on all collected data
- [x] **Phase 5 — Re-transcribe:** Generate clean labels with Whisper large-v3 (not YouTube auto-captions) — Deepgram Nova-3 with 50 theological keyterms (35 sermons)
- [x] **Phase 6 — Fine-tune (round 1):** TranslateGemma S1-S9 complete (S6 winner: balanced ratio), Whisper W12 data scaling (198K chunks, 21.41% baseline WER), W15 hard mining bug fixed, **W16 corrective run = 7.25% fresh-eval WER**, **v2026.5 Phase 1A: Gemma 4 E4B Q4_K_M wins production default (5–9× speedup, 4× less VRAM)**
- [ ] **Phase 7 — Evaluate:** Transfer adapters to Mac, re-run A/B with fine-tuned models + live YT comparison
- [ ] **Phase 8 — Feedback loop:** Route flagged segments to correction → retrain (repeat 2–4 more cycles)
- [x] **Phase 9 — Demo:** **Shipped as v2026.6 operator control plane** (FastAPI + vanilla JS, not Streamlit as originally sketched). Pre-flight, mid-session controls, live observability, audio hotplug, verse highlights, summary trigger, systemd/launchd, bootstrap.sh. See [`docs/operator_runbook.md`](./docs/operator_runbook.md).
- [ ] **Phase 10 — Integrate & polish:** Deferred 9.4.1 (multi-channel TTS routing) and 9.6.1 (live diarization on rolling buffer). Continuous improvement loop (live → log → retrain monthly). macOS Shortcuts for voice-command triggers if useful. systemd auto-start at the church PC is already in place via Phase 9.5.
