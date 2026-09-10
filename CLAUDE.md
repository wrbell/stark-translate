# CLAUDE.md — Live Bilingual Speech-to-Text

> **Mac handoff, v2026.14 candidate:** See [implementation status](docs/mac_implementation_status.md)
> and [evaluation inputs/results](docs/evaluation/README.md). v2026.13 PRs #180–191
> are already merged. Operator reliability, timing schema 2, reproducible setup,
> live Review/export and opt-in experiments are implemented on
> `codex/mac-reliability-roadmap`. The [48-run screen](docs/evaluation/mac_v2026_14_screening/README.md)
> is complete and supports no combined configuration; E4B, 0.5 s silence and
> 0.6 s partial cadence remain default, and MTP stays off. Packaged offline VAD
> and automatic pinned Marian CT2 setup passed focused/real CPU checks. Separate
> [24-run synthetic routing probes](docs/evaluation/mac_v2026_14_routing/README.md)
> passed; the final CPU suite is 1,790 passed/4 skipped (59.07% coverage).
> [Local wheel/sdist/Mac ZIP validation](docs/evaluation/mac_v2026_14_installation.md)
> and installed EN/ES runtime checks are complete; publication remains pending. Archived
> `e2e_latency_ms` is processing time, not speech-end-to-display. Human natural
> Spanish/two-speaker/bilingual review and physical output tests remain gates.
> PyPI publishing is pending by user choice. Do not restart the older roadmap
> from scratch or overwrite the user's working `stt_env`.

A fully on-device, live bidirectional speech-to-text system (English/Spanish) for church outreach at Stark Road Gospel Hall (Farmington Hills, MI). Supports `--lang en` (EN→ES) and `--lang es` (ES→EN). Includes Piper TTS (`--tts`). All Python, MLX on Apple Silicon for inference, CUDA on NVIDIA for training.

**Two-pass pipeline** for fast partials and high-quality finals:
- **Partials (while speaking):** STT + MarianMT — italicised in UI; Mac uses Parakeet for English and Whisper for Spanish, with cached Marian CT2 preferred.
- **Finals (on silence):** STT + Gemma 4 translation — replaces partial.
- **Mac (MLX):** English Parakeet / Spanish mlx-whisper + Gemma 4 OptiQ E4B finals (TranslateGemma opt-out via `--model-family translategemma`); Marian CT2 on CPU with HF fallback. Schema 2 now measures estimated speech end to payload readiness and visible-browser acknowledgment separately. The sub-second median caption-delivery goal remains unachieved in the current baseline; historical processing timings do not establish that goal.
- **CUDA (v2026.8+):** faster-whisper + W16 LoRA via CT2 (~353ms p50 / ~413ms p95 STT, A2000 Ada) + MarianMT-CT2 int8_float16 (~57ms p50 / ~116ms p95 partial) + Gemma 4 E4B Q4_K_M via llama.cpp (~470ms final). Partial p50 ≈ 410 ms (STT 353 + Marian 57); final p50 ≈ 820 ms.
- **CUDA model options:** Gemma 4 E2B Q4_K_M (3.5 GB VRAM, ~280ms) for low-VRAM, E4B Q4_K_M (4.9 GB, ~470ms, 7/8 canary) for best quality
- **Optimization wins:** v2026.7 Whisper drops STT WER 19% relative overall, 43% on theological terms (`docs/archive/v2026.7/STT_BENCHMARK.md`). v2026.8 Marian drops partial-translate latency 66% relative at p50 vs HF CUDA (`docs/archive/v2026.8/MARIAN_BENCHMARK.md`).

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

## Recent Additions (2026-09-09, v2026.13 — Mac latency program)

- **Three production bugs on the Mac final path, all fixed:** Gemma 4 stop set replaced with `{<eos>, <unk>}` → every final ran to `max_tokens` (#172 → #181, `ensure_stop_tokens`); first Gemma forward on a pool worker under mlx 0.32 thread-local streams → `There is no Stream(gpu, 1) in current thread` on every final (#181, `warm_mlx_model` on the load thread); `qe_b` UnboundLocalError in the single-model finalize path (#184, re-applied in #187 after a squash merge dropped it). Together they explain the empty 2026-08-30 session CSVs.
- **Replay benchmark** (#182): `dry_run_ab.py --audio-file` + `tools/replay_bench.py` play real sermon clips through VAD → partials → finals and report p50/p95. Archived `e2e_latency_ms` measures processing after submission, not speech end → final. Pick VAD-dense clip offsets (the 10-min mark is a hymn → music hold). Results: `docs/archive/v2026.13/MAC_LATENCY.md`.
- **Parakeet TDT 0.6B v3 on MLX** (#186 + live dispatch): `--stt-backend parakeet-mlx`; **auto default for English sessions** after the Piper→STT roundtrip gate (`tools/stt_roundtrip_compare.py`: WER 0.6 % vs whisper-turbo 0.2 %, Tier-1 term recall 1.0, p50 143 vs 643 ms); Spanish stays on whisper-turbo. Live: STT p50 1.0–1.2 s → 0.24 s and Gemma decode recovers (GPU no longer contended).
- **Marian CT2 on Mac**: `scripts/convert_marian_ct2.py --quantization int8` → `adapters/marian_ct2/<dir>/active/` (auto-preferred; factory forces `int8` + 4 threads off-CUDA). Partial p50 270–380 ms → 52–94 ms. Working env override is `STARK_TRANSLATE_MARIAN_BACKEND` (single underscore).
- **Archived measurements (legacy processing p50, two 150 s clips):** 2026-08-30 code 4.8 / 3.7 s → fixes 4.0 / 2.9 s → + Parakeet 1.6 / 1.0 s; partial processing ~1.0 s → ~0.2–0.26 s. Fast mode `--gemma4-size e2b`: 1.2 / 0.86 s (canary 11/18 vs 13/18). Isolated E4B translate: 2139/2176/3242 → 473/1393/1677 ms. VAD 0.35 s / 0.4 s cadence tried and rejected (more GPU contention). These numbers retain their historical definitions and do not measure speech-end-to-display.
- **Gemma 4 MTP drafter ("MTS")** (#183, #177): `engines/mlx_spec.py` over mlx-optiq `runtime.spec`; probe: byte-identical output but 31 % acceptance → ≤ 14 % latency win; RoPE-offset suspect rejected; stays off.
- **CUDA proposal** (#185): `docs/cuda_latency_proposal.md` + `scripts/cuda/*.sh` (llama.cpp b10883, Gemma 4 MTP with f16 KV, `-fa` retest, W16→HF fp16, Parakeet EN+ES); `start_server.sh` default `--no-draft`, `--mtp` opt-in; not yet executed on the A2000.
- **Part D:** TTS per-language routing (#188, 9.4.1) and live diarization behind `--diarize` (#189, 9.6.1, gate pending). Default ES Piper voice fixed (`es_MX-claude-high`; `es_ES-carlfm-high` does not exist).

## Earlier Additions (2026-09-09, v2026.12 close-out)

- **v2026.12 — restart review + close-out.** Repo was dormant 2026-05-03 → 2026-08-14. Aug-14: security hygiene (#139/#148/#154/#160, pip-audit 0, issues #131–#138 opened). Aug-30/31 (Cursor-agent burst): #162 pipeline refresh (garbage filter, Phase 4 runner, W17 DoRA script, Parakeet-NeMo engine, AL tools), #164 WSL runbook, #165 TurboQuant/adapter_path/Mac health_check, #166 hymn corpus, #167 coverage 31→50 %, #168 Mac STT∥translation overlap (MLX ≥0.31.2 thread-local streams, `max_workers=2`), #169 OptiQ/MTS accel matrix (`tools/benchmark_mlx_accel.py`), #170 absorbed the closed v2026.8–11 latency PRs onto main, #171 Gemma 4 OptiQ E4B as Mac default with `enable_thinking=False`. Version bumped 2026.7.0.0 → 2026.12.0.0 to cover all of it (no tag existed after v2026.7.0.0). Stale branches pruned; #131–#138 triaged (kept open); tracking issues #172–#179 filed.
- **Gemma 4 EOS bug (Mac) — root cause of the ~2.2 s finals (#172).** `load_mlx_gemma`, `MLXGemmaEngine.load` and `tools/health_check.py` applied the TranslateGemma `<end_of_turn>` EOS patch to Gemma 4, which has no such token: `convert_tokens_to_ids` returns `<unk>` (3) and the code *replaced* `tokenizer._eos_token_ids` with `{1, 3}`, discarding `<turn|>`=106 / `<|tool_response>`=50 that mlx-lm had loaded from `config.json`. Every Gemma 4 call ran to `max_tokens` (bench: 64/64/~100 tokens ≈ 2139/2176/3242 ms at ~30 tok/s) and emitted `<turn|><|channel>thought…` junk that `clean_translation` stripped. The E2B "0/8 canary collapse" in `docs/mac_pipeline_refresh.md` is the same bug plus thinking-on, not PLE garbage. Fix: `ensure_stop_tokens()` in `engines/translation_prompts.py` (branch `feat/gemma4-eos-fix`).
- **Known stale bits found in review:** `--mts` has never loaded (mlx-lm has no `gemma4_assistant`; the error is swallowed) — #177; `--multiprocess` (`workers.py`) still builds TranslateGemma prompts — #176; `start_server.sh` enables the E2B draft by default despite the v2026.5 single-GPU loss — #173; Dockerfile pins llama.cpp b8783 vs b9022 in `start_server.sh` — #174; `docs/archive/v2026.9/FOLLOWUPS.md` was referenced but never existed — folded into `docs/roadmap.md` (#175); `features/live_diarize.py` exited 1 without `HF_TOKEN` instead of falling back to fake labels (fixed here).
- **Latency program (v2026.13, in progress):** replay harness (#179), Gemma 4 MTP drafter via the mlx-optiq runtime (#177), Parakeet TDT v3 STT on MLX (#178), Marian CT2 on Mac (ctranslate2 4.7.1 arm64 no longer links libomp), then a CUDA proposal (`docs/cuda_latency_proposal.md`). Verified env: mlx 0.32.2, mlx-lm 0.31.3, mlx-optiq 0.4.34.

## Earlier Additions (2026-05-03)

- **v2026.11 — Imatrix-calibrated quantization** (`docs/archive/v2026.11/IMATRIX_CALIBRATION.md`). Built `training/build_imatrix_corpus.py` (sermon corpus + 25× canary oversampling, K=32 interleave). Ran `llama-imatrix` on bf16 E2B (21s) + E4B (31s) on clean GPU, no fallback ladder needed. Re-quantized Q4_K_M and IQ4_XS for both with `--imatrix`. **Result: imatrix on E4B IQ4_XS recovered 1 of 3 lost canary items (Jacobo) — 5/8 → 6/8.** Still 1 below Q4_K_M-no-imatrix baseline (7/8) — `pacto`→alianza and `partimiento`→fracción are synonym-level shifts that activation-side calibration can't constrain (corpus has the EN source only, can't bias the ES output token). Default stays `gemma-4-e4b-it-q4km.gguf` (no imatrix). Ships the infrastructure + 4 calibrated GGUFs + 4 new CONFIGS in `bench_translate_t1_t4.py`. Imatrix recipe captured for the next fine-tune cycle, where expected payoff is higher.

- **v2026.10 — IQ4_XS quantization sweep** (`docs/archive/v2026.10/IQ4_XS_BENCHMARK.md`). Re-quantized E2B + E4B from HF bf16 to IQ4_XS via `llama-quantize`. **E4B IQ4_XS shows -9% p50 latency BUT regressed canary 7/8 → 5/8** (loses `pacto`→alianza, `Jacobo`→James untranslated, `partimiento`→fracción). Default stays Q4_K_M; PR3 ships the artifact + bench infra (new `t2-iq4xs`, `t3-iq4xs` configs in `bench_translate_t1_t4.py`) for PR4 imatrix calibration to attempt rescue. **E2B IQ4_XS is acceptable** (-4% p50, canary holds at 6/8). GGUFs produced locally; not yet hosted on HF.

- **v2026.9 — Gemma 4 latency optimization (Phase 2 cheap wins)** (`docs/archive/v2026.9/GEMMA_OPTIM_PHASE2.md`). Cumulative-flags sweep over four llama.cpp optimizations. Net: ~3% partial p50 reduction with canary unchanged. Added `-ctv q8_0` to `start_server.sh` (V-cache quantization, free win). Pinned llama.cpp to `d8794eecd` / build `b9022` (2026-05-04, +240 commits upstream). Skipped: `GGML_CUDA_GRAPH_OPT=1` env var (already compiled in via `USE_GRAPHS=1`); `--flash-attn on` (regressed E4B by +56% on b8782 — Gemma 4 SWA-aware FA kernels needed; flagged for retry on b9022 in a future phase). Misses the 20% gate, so PR3 (IQ4_XS quantization) is mandatory next.

- **v2026.8 — MarianMT partial-translation acceleration** (`docs/archive/v2026.8/MARIAN_BENCHMARK.md`). Wires opus-mt-{en-es,es-en} onto CTranslate2 for the partial-translation path, mirroring the v2026.7 STT recipe. New `scripts/convert_marian_ct2.py` (vendor-model converter, no LoRA — Marian is off-the-shelf). New `engines/cuda_engine.py::MarianCT2Engine` — CT2 `Translator` + HF `MarianTokenizer`, internally thread-safe so the engine does not hold `_pytorch_lock`. The HF path moved to `engines/marian_hf_engine.py::MarianHFEngine` (extracted from `engines/mlx_engine.py`); both engines now share `engines/_locks.py::_pytorch_lock` with Silero VAD (latent thread-safety bug fix — they used to be independent locks). `engines/factory.py` auto-prefers `adapters/marian_ct2/{en-es,es-en}/active/` when present; falls back to HF when absent or when ctranslate2 fails to import. New `STARK_TRANSLATE__MARIAN_BACKEND` and `STARK_TRANSLATE__MARIAN_COMPUTE_TYPE` env vars. **Bench result on A2000 Ada (`tools/benchmark_translate_engines.py` over 48 stratified clips):** Marian partial p50 drops 66% relative (167 ms HF CUDA → 57 ms CT2 int8_float16), p95 drops 54% (253 ms → 116 ms), peak VRAM drops 19% (1.96 GB → 1.58 GB), canary unchanged at 14/16 (the 2 misses are pre-existing opus-mt Jacobo/Santiago and "fracción"/"partimiento" limitations, not quantization artifacts). New CSV/JSONL field `marian_backend` so operators can see which path is hot. Setup: run `scripts/convert_marian_ct2.py` once per direction.

- **v2026.7 — Whisper STT latency optimization** (`docs/archive/v2026.7/STT_BENCHMARK.md`). Wires the W16 fine-tune (7.25% fresh-eval WER) into the production CUDA path via `training/export_ct2.py` (LoRA merge → CTranslate2 conversion). `engines/factory.py` auto-prefers `adapters/whisper_turbo_ct2/active/` when present, falls back to off-the-shelf `large-v3-turbo` otherwise. **Bench result on A2000 Ada:** WER drops 19% relative overall (13.55% → 11.00%) and 43% relative on theological terms (15.22% → 8.70%) at ~zero latency cost (p95 96% of baseline). Default `compute_type` bumped from `int8` → `int8_float16` (CTranslate2 docs recommend it for Ampere/Ada; bench saw 0% latency / 0% VRAM delta — escape hatch via `STARK_CUDA__COMPUTE_TYPE=int8`). New `--stt-backend {auto,faster-whisper,hf,mlx}` CLI flag + `STARK_STT__BACKEND` env var; `HFWhisperEngine` gained `compile_mode` + `warmup_seconds` constructor args. **Spec-decode default draft removed** — distil-large-v3.5 + whisper-turbo is broken (different decoder layer counts → 10× slower with hallucinated output, see `docs/archive/v2026.5/spec_decode_research.md`); factory now raises `ValueError` instead of silently producing garbage. New `tools/benchmark_stt_engines.py` benchmark harness using the 41-clip manifest at `tools/stt_bench_manifest.json` and reusing `scripts/benchmarks/vram_sampler.py` (extracted from `bench_translate_t1_t4.py`).

## Earlier Additions (2026-04-25)

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
| **Test** (`test.yml`) | push / PR | pytest (Python 3.11 + 3.12), coverage ≥50%, Codecov, PR comment |
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
- [ ] **Phase 4 — Preprocessing:** Run the 10-step audio cleaning pipeline on all collected data — orchestrator: `training/run_phase4_preprocess.sh` / `training/run_phase4_corpus.py` (writes `phase4_status.json`). Requires sermon WAVs on WSL. **WSL runbook:** [`docs/wsl_pipeline_refresh.md`](./docs/wsl_pipeline_refresh.md) §1.
- [x] **Phase 5 — Re-transcribe:** Generate clean labels with Whisper large-v3 (not YouTube auto-captions) — Deepgram Nova-3 with 50 theological keyterms (35 sermons)
- [x] **Phase 6 — Fine-tune (round 1):** TranslateGemma S1-S9 complete (S6 winner: balanced ratio), Whisper W12 data scaling (198K chunks, 21.41% baseline WER), W15 hard mining bug fixed, **W16 corrective run = 7.25% fresh-eval WER**, **v2026.5 Phase 1A: Gemma 4 E4B Q4_K_M wins production default (5–9× speedup, 4× less VRAM)**
- [ ] **Phase 7 — Evaluate:** Transfer adapters to Mac, re-run A/B with fine-tuned models + live YT comparison — **WSL→Mac handoff:** [`docs/wsl_pipeline_refresh.md`](./docs/wsl_pipeline_refresh.md) §5 · **Mac-only path:** [`docs/mac_pipeline_refresh.md`](./docs/mac_pipeline_refresh.md) (`--adapter-dir`, `--turboquant`, `health_check --backend mlx`).
- [ ] **Phase 8 — Feedback loop:** Route flagged segments to correction → retrain (repeat 2–4 more cycles) — **WSL loop:** [`docs/wsl_pipeline_refresh.md`](./docs/wsl_pipeline_refresh.md) §6 (`merge_corrections.py`, `deploy_adapters.py`).
- [x] **Phase 9 — Demo:** **Shipped as v2026.6 operator control plane** (FastAPI + vanilla JS, not Streamlit as originally sketched). Pre-flight, mid-session controls, live observability, audio hotplug, verse highlights, summary trigger, systemd/launchd, bootstrap.sh. See [`docs/operator_runbook.md`](./docs/operator_runbook.md).
- [ ] **Phase 10 — Integrate & polish:** Deferred 9.4.1 (multi-channel TTS routing) and 9.6.1 (live diarization on rolling buffer). Continuous improvement loop (live → log → retrain monthly). macOS Shortcuts for voice-command triggers if useful. systemd auto-start at the church PC is already in place via Phase 9.5.
