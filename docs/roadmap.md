# Roadmap — Stark Road Bilingual Speech-to-Text

> Living document tracking the full project trajectory from Mac prototype
> through Windows training to production deployment.
>
> **Last updated:** 2026-04-11

---

## Current State

```
Mac (M3 Pro 18GB, MLX)                Windows (A2000 Ada 16GB, CUDA/WSL2)
  Live inference prototype (stable)      Training pipeline hardened
  Whisper Large-V3-Turbo (STT)           198K aligned chunks (328 sermons)
  engines/ package (MLX + CUDA)          TranslateGemma S1-S9 sweep → S6 winner
  settings.py (pydantic-settings)        Whisper W12 data scaling (198K chunks)
  Backend: --backend auto|mlx|cuda       W15 hard example mining + curriculum
  Piper TTS (EN + ES, --tts)             Deepgram Nova-3 oracle (35 sermons)
  Pipeline overlap (STT N+1 ∥ TT N)     Tiered glossary (50 boost + 229 master)
  5 display modes (WebSocket)            12GB memory-capped alignment pipeline
  806 tests, 7 CI workflows              benchmark_gemma4.py (Gemma 4 comparison)

Production Endpoints (implemented):
  1. Mac M-series (8-18 GB) — MLX, --backend=mlx
  2. NVIDIA GPU (6-16 GB VRAM) — CUDA, --backend=cuda
  3. Dev M3 Pro 18 GB — full A/B (4B + 12B)
```

---

## Completed Work

### Phase 1: Infrastructure & Inference (Done)

- **Engines package** — ABCs (`STTEngine`, `TranslationEngine`, `TTSEngine`), MLX + CUDA implementations, factory auto-detection
- **Whisper Large-V3-Turbo swap** — Both partials and finals, <150ms partials on Mac
- **CUDA streaming runtime** — `CUDAGemmaStreamingEngine` with TextIteratorStreamer, prompt cache (~50-80ms savings), speculative decoding (4B drafts 12B), VRAM tier detection (15GB/5.5GB thresholds)
- **Dual-target inference** — `--backend auto|mlx|cuda`, `--no-ab`, `--low-vram` flags
- **Unified config** — `settings.py` with pydantic-settings, `STARK_` env prefix, `CUDASettings` for CUDA-specific knobs
- **Piper TTS** — EN + ES voices, ONNX runtime, WebSocket + WAV output, `--tts` flag
- **Bidirectional language support** — `--lang en` (EN→ES) and `--lang es` (ES→EN)
- **Pipeline overlap** — Translation on utterance N while STT runs on N+1
- **5 display modes** — Audience, A/B, Mobile, Church, OBS overlay
- **CI/CD** — 7 GitHub Actions (lint, test, security, release, label, commitlint, stale), 806 tests, Codecov, pre-commit, CalVer

### Phase 2: Data Collection (Done)

- **333 sermons cataloged**, 160+ downloaded, organized into `stt-data/{type}/{year}/`
- **Deepgram Nova-3 oracle** — 35 sermons transcribed with 50 theological keyterms ($9 total)
- **Tiered glossary** — Tier 1 (50 boost terms for Deepgram), Tier 2 (229 master terms for training)
- **Data integrity** — SHA-256 lockfile, 2026-03-14 training cutoff, adapter health checks
- **Evaluation sets** — 500 stratified verse holdout + 422 sermon eval chunks + fresh eval set (4 post-cutoff sermons, 2,706 examples)

### Phase 3: TranslateGemma Fine-Tuning (Done)

- **S1-S9 ablation sweep** — Learning rate, steps, NEFTune, verse/sermon ratio, data scale
- **S6 winner** — Balanced 1:1 verse/sermon ratio, COMET proximity to 12B base = -0.0002 (effectively tied)
- **Quantization benchmarks** — 4B at 4-bit: 3.0 GB VRAM, 12B at 4-bit: 7.3 GB, 12B at 8-bit: OOM on 16 GB
- **DeepL synthetic data** — 10K+ glossary-enforced sermon pairs for training augmentation

### Phase 4: Whisper LoRA Training (In Progress)

- **W0-W9 ablation designed** — Test matrix covering learning rate, target modules, replay ratio, data scale
- **W12 data scaling** — 198K Deepgram-aligned chunks from 328 sermons, 290 GB Arrow cache
  - Baseline WER on fresh eval: **21.41%** (normalized)
  - Config: lr=1e-4, r=32, q_proj+v_proj, replay=0.3, 1 epoch
- **W15 hard example mining** — Curriculum learning pipeline:
  - `mine_hard_examples.py` — Batched fp16 inference, per-chunk WER, Tier 1 detection, resume support
  - `build_hard_subset.py` — WER-bounded filtering (0.15-0.80), stratified per-source caps
  - `filter_chunks_by_confidence.py` — Top-N selection by logprob/confidence
  - `recover_shards.py` — Rebuild DatasetDict from shards after OOM
  - `--init-from` in `train_whisper.py` — Load pre-trained adapter weights with fresh optimizer
- **Alignment hardening** — Sharded Arrow writes (1000 rows/shard), 12GB memory cap, streaming to disk (5 GB RAM vs 190 GB before)

---

## Active Work

### Whisper Curriculum Learning (Current)

Continue W15 hard mining pipeline: mine → filter → build subset → train with `--init-from` → re-mine on new adapter. Target: reduce WER from 21.41% baseline toward <10% on church audio.

### Gemma 4 Evaluation

`benchmark_gemma4.py` compares TranslateGemma 4B/12B vs Gemma 4 E2B/E4B across 3 tiers:
- Tier 1: Bible verse holdout (BLEU/chrF++/COMET)
- Tier 2: Deepgram sermon chunks (COMET-QE + hallucination ratio)
- Tier 3: 8 theological canary sentences (term accuracy)

---

## Upcoming Phases

### Phase 5: Adapter Evaluation & Transfer (Next)

- Transfer best Whisper + TranslateGemma adapters to Mac
- Re-run A/B comparison with fine-tuned vs base models
- Live YouTube caption comparison with fine-tuned STT
- Smoke test: 5 canary sentences + theological term audit

### Phase 6: Active Learning Feedback Loop

- Route low-confidence segments to operator review
- Human correction workflow → merge corrections into training data
- Retrain on corrected data (repeat 2-4 cycles)
- Target: 20-40% relative WER reduction per cycle

### Phase 7: Live Demo Deployment

- Streamlit dashboard for Farmington Hills coffee shop outreach event
- macOS Shortcuts for voice-command triggers
- OBS/NDI integration for projection system
- Operator tablet controls

### Phase 8: Multilingual Expansion (Hindi & Chinese)

TranslateGemma natively supports Hindi and Chinese — fine-tuning is domain adaptation only.

| Phase | Duration | What |
|-------|----------|------|
| Zero-shot baseline | 1 day | Test with `target_lang_code="hi"` / `"zh-Hans"` |
| Data preparation | 3-5 days | ~155K EN-HI + ~310K EN-ZH biblical verse pairs |
| Hindi QLoRA | 1 night | r=32, 768 max_seq_length, separate adapter |
| Chinese QLoRA | 1 night | r=32, 512 max_seq_length, separate adapter |
| Evaluation + integration | 2-3 days | chrF++/COMET, adapter switching, display updates |

Key decisions: Hindi → English partial + Hindi final (SOV word order garbles partials); Chinese → 神 (Shen) for God (CUV majority edition).

### Phase 9: Piper TTS Multi-Language

- Fine-tune Piper voices per language from base checkpoints
- Multi-channel audio routing (AudioFetch or virtual cables)
- Scripts ready: `prepare_piper_dataset.py`, `train_piper.py`, `export_piper_onnx.py`, `evaluate_piper.py`

### Phase 10: Production Polish

- Dedicated hardware at church (auto-start on boot)
- Continuous improvement loop: live inference → log diagnostics → retrain monthly
- Post-sermon summary, verse extraction, diarization integration

---

## Observed Metrics

### STT Performance

| Metric | Base (no fine-tune) | W12 (198K chunks) | Target |
|--------|--------------------|--------------------|--------|
| WER (church, fresh eval) | 21.41% | In progress | <10% |
| WER (Scottish accent) | ~22-34% | — | <10% |
| Accent WER gap | ~15-24% | — | <5% |

### Translation Performance

| Metric | TranslateGemma 4B (base) | S6 (fine-tuned 4B) | 12B (base) |
|--------|--------------------------|---------------------|------------|
| COMET | Baseline | -0.0002 vs 12B | Baseline |
| Config | — | 1:1 verse/sermon | — |

### CUDA Inference (A2000 Ada 16GB)

| Component | Latency | VRAM |
|-----------|---------|------|
| faster-whisper large-v3 (fp16) | 5-10x real-time | ~5 GB |
| TranslateGemma 4B (4-bit) | 2-3s/input | 3.0 GB |
| TranslateGemma 12B (4-bit) | 2-3s/input | 7.3 GB |
| Prompt cache savings | 50-80ms/call | — |

### Mac Inference (M3 Pro 18GB, MLX)

| Component | Latency |
|-----------|---------|
| Partial (STT + MarianMT) | ~750ms |
| Final 4B (STT + TranslateGemma) | ~1.1s |
| Final 12B (STT + TranslateGemma) | ~2.6s |
| Piper TTS | ~40ms/word EN, ~8ms/word ES |

---

## Key Decisions (Resolved)

| Decision | Resolution |
|----------|------------|
| TranslateGemma data ratio | S6: balanced 1:1 verse/sermon |
| CUDA Gemma loading | bitsandbytes 4-bit (NF4) |
| 12B on 16GB GPU | 4-bit fits at 7.3 GB; 8-bit OOM |
| Separate venvs | Yes — `requirements-mac.txt` + `requirements-nvidia.txt` |
| STT model | Whisper Large-V3-Turbo (both Mac + CUDA) |
| Pipeline threading | MLX: 1 worker (Metal not thread-safe); CUDA: 2 workers |
| Training data alignment | Deepgram word timestamps + faster-whisper chunk boundaries |
| OOM mitigation | Sharded Arrow writes (1000 rows), 12GB memory cap |

## Key Decisions (Pending)

| Decision | When | Options |
|----------|------|---------|
| Gemma 4 vs TranslateGemma | After benchmark_gemma4.py results | E2B/E4B may outperform TG on theological text |
| W15 curriculum iterations | After first mining cycle | 2-4 cycles typical for convergence |
| Scottish accent data sources | Before accent tuning | User provides YouTube playlist URLs |
| Hindi/Chinese timing | After Whisper fine-tuning stabilizes | Hindi first (higher demand) |
| TTS voice fine-tuning | Phase 9 | Fine-tune from Piper base vs train from scratch |
| Production hardware | Phase 10 | Dedicated PC at church vs portable Mac |

---

## Reference Documents

| Doc | Contents |
|-----|----------|
| [`training_plan.md`](training_plan.md) | Full training schedule, channel inventory, go/no-go gates |
| [`accent_tuning_plan.md`](accent_tuning_plan.md) | 4-week accent-diverse STT tuning plan (code complete) |
| [`hard_mining.md`](hard_mining.md) | W15 hard example mining design |
| [`multi_lingual.md`](multi_lingual.md) | Hindi & Chinese actionable todo list |
| [`multilingual_tuning_proposal.md`](multilingual_tuning_proposal.md) | Full Hindi/Chinese research: corpora, glossaries, evaluation |
| [`rtx2070_feasibility.md`](rtx2070_feasibility.md) | RTX 2070 hardware analysis |
| [`fast_stt_options.md`](fast_stt_options.md) | Lightning-whisper-mlx feasibility (not viable) |
| [`projection_integration.md`](projection_integration.md) | OBS/NDI/ProPresenter integration |
| [`turbo_inference.md`](turbo_inference.md) | Turbo model inference details |
| [`data_pipeline_status.md`](data_pipeline_status.md) | Data pipeline current state |
