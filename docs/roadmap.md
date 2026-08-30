# Roadmap — Stark Road Bilingual Speech-to-Text

> Living document tracking the full project trajectory from Mac prototype
> through Windows training to production deployment.
>
> **Last updated:** 2026-08-30

---

## Current State

```
Mac (M3 Pro 18GB, MLX)                Windows (A2000 Ada 16GB, CUDA/WSL2)
  Live inference prototype (stable)      Training pipeline hardened
  Whisper Large-V3-Turbo (STT)           198K aligned chunks (328 sermons)
  engines/ package (MLX + CUDA)          TranslateGemma S1-S9 sweep → S6 winner
  settings.py (pydantic-settings)        Whisper W12 data scaling (198K chunks)
  Backend: --backend auto|mlx|cuda       W15 hard mining + W16 = 7.25% fresh-eval WER
  Piper TTS (EN + ES, --tts)             Deepgram Nova-3 oracle (35 sermons)
  Pipeline overlap (STT N+1 ∥ TT N)     Tiered glossary (50 boost + 229 master)
  Display modes + operator UI            llama.cpp E4B Q4_K_M = production CUDA default
  v2026.6/v2026.7 shipped                W17 curriculum scripted (DoRA + hard-mix)

Production Endpoints (implemented):
  1. Mac M-series (8-18 GB) — MLX, --backend=mlx
  2. NVIDIA GPU (6-16 GB VRAM) — CUDA via llama.cpp (--engine auto|llamacpp|hf)
  3. Operator control plane — FastAPI + vanilla JS at http://host:9000/operator/

See `docs/operator_runbook.md` for the day-of-event workflow and `bootstrap.sh`
for first-time church PC setup.

**Next WSL execution:** `docs/wsl_pipeline_refresh.md` (Phase 4 → E4B SFT → W17 → Mac → AL).
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

### Pipeline refresh (WSL execution)

**Primary runbook:** [`docs/wsl_pipeline_refresh.md`](./wsl_pipeline_refresh.md)

Ordered stages on the A2000 Ada box:

1. Phase 4 full audio preprocess (`run_phase4_preprocess.sh`)
2. Gemma 4 E4B domain SFT → GGUF (`run_gemma4_e4b_domain_sft.sh`, 8-canary sanity)
3. W17 Whisper DoRA + hard-mix → CT2 + `benchmark_stt_engines.py` gate (must ≤ W16)
4. Optional Parakeet EN-only bench — adopt only if it beats W17; bilingual default stays Whisper
5. Mac transfer / Phase 7 A/B + live YT compare
6. Phase 8 active learning (`merge_corrections.py` → retrain → `deploy_adapters.py`)

**Status notes:** W16 shipped in production CT2 path (7.25% fresh-eval WER). W17 is scripted in-repo, not yet trained. Scripts and garbage-filter hardening landed with the 2026-08 pipeline refresh (PR #162).

### Gemma 4 Evaluation (reference)

`benchmark_gemma4.py` / Phase 1A llama.cpp matrix remain the eval harnesses. Production CUDA finals stay **E4B Q4_K_M**; next accuracy lever is domain SFT of that model (stage 2 above), not a larger base on 16 GB.

---

## Upcoming Phases

### Phase 5: Adapter Evaluation & Transfer (Next)

- Transfer best Whisper + Gemma adapters to Mac — see refresh runbook §5
- Re-run A/B comparison with fine-tuned vs base models
- Live YouTube caption comparison with fine-tuned STT
- Smoke test: **8** canary sentences (`tools/health_check.py --n-canaries 8`) + theological term audit

### Phase 6: Active Learning Feedback Loop

- Route low-confidence segments to operator review
- Human correction → `tools/merge_corrections.py` into training data
- Retrain on corrected data (repeat 2-4 cycles) — refresh runbook §6
- Target: 20-40% relative WER reduction per early cycle; stop when &lt; 2% for 2 cycles

### Phase 7: Live Demo Deployment ✅ shipped as v2026.6

- ✅ FastAPI + vanilla JS operator control plane (replaced the original Streamlit sketch)
- ✅ Pre-flight gating (GPU / models / mic / adapter manifest / llama-server)
- ✅ Mid-session controls (pause / resume / lang_flip / vad / fallback)
- ✅ Live observability sparklines (VRAM, CPU, latency p50/p95, confidence)
- ✅ Audio device enumeration + USB hotplug toast
- ✅ Verse highlights + post-session summary trigger
- ✅ systemd unit + launchd plist + bootstrap.sh for church PC install
- See [`operator_runbook.md`](./operator_runbook.md) for the day-of-event workflow

**Deferred patches** (will ship as v2026.6.x): multi-channel TTS routing (9.4.1),
live diarization on a rolling audio buffer (9.6.1), macOS Shortcuts for
voice-command triggers (10).

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

- ✅ Dedicated hardware auto-start (Phase 9.5 — systemd unit + launchd plist + bootstrap.sh)
- ✅ Post-sermon summary trigger (Phase 9.6 — `/api/features/summary`)
- ✅ Verse extraction wired into the live operator UI (Phase 9.6 — `/api/features/verses`)
- 9.4.1 — Multi-channel TTS routing (UI is ready, output dropdown disabled / preview-only; needs PiperTTSEngine output-device support + `--tts-output local` mode)
- 9.6.1 — Live diarization on a rolling audio buffer (UI surface deferred; existing `features/diarize.py` is offline-only)
- Continuous improvement loop: live inference → log diagnostics → retrain monthly (depends on Phase 6/8 active learning)

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
