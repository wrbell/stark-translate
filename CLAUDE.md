# CLAUDE.md — Live Bilingual Speech-to-Text: Project Overview

## Project Summary

A fully on-device, live English speech-to-text system with dual English/Spanish output, built for church outreach at Stark Road Gospel Hall (Farmington Hills, MI). Designed for low-latency local transcription that outperforms YouTube's built-in captions, with automated quality monitoring and an iterative fine-tuning loop.

A **two-pass pipeline** provides fast partials and high-quality finals:

- **Partials (while speaking):** mlx-whisper STT + MarianMT PyTorch translation (~380ms) -- displayed in italics
- **Finals (on silence):** mlx-whisper STT + TranslateGemma 4B/12B 4-bit translation (~1.3s / ~1.9s) -- replaces partial

Two translation models run in parallel for A/B comparison:

- **Approach A (Lightweight):** TranslateGemma-4B 4-bit via mlx-lm -- ~650ms translation, ~1.3s end-to-end
- **Approach B (High-Accuracy):** TranslateGemma-12B 4-bit via mlx-lm -- ~1.4s translation, ~1.9s end-to-end
- **Speculative decoding:** 4B serves as draft model for 12B via `mlx_lm.generate(draft_model=)`, configurable via `--num-draft-tokens`

All Python, MLX-accelerated on Apple Silicon for inference (mlx-whisper for STT, mlx-lm for translation), CUDA-accelerated on NVIDIA for training.

### Goals

- Real-time mic input with <500ms partial outputs
- Side-by-side A/B comparison in browser displays (served over LAN via HTTP + WebSocket)
- Automated quality monitoring against YouTube livestream captions
- Confidence-based flagging to minimize human review effort
- Reference-free translation quality estimation (no Spanish ground truth needed)
- Iterative fine-tuning feedback loop converging in 3–5 cycles
- Exportable metrics for post-test analysis

---

## Environment Split

This project spans two machines. Each has its own detailed setup doc:

| Machine | Role | Doc |
|---------|------|-----|
| **MacBook** (M3 Pro, 18GB, 18-core GPU, Metal 4) | Inference, live demos, monitoring, UI | [`CLAUDE-macbook.md`](./CLAUDE-macbook.md) |
| **Windows Desktop** (WSL2, A2000 Ada 16GB, 64GB RAM) | Audio preprocessing, fine-tuning, training loops | [`CLAUDE-windows.md`](./CLAUDE-windows.md) |

Model transfer path: WSL shared folder `/mnt/c/Users/YourName/Projects/` → copy LoRA adapters to Mac project root.

---

## Full Project Structure

```
project_dir/
├── CLAUDE.md                       # This file — project overview
├── CLAUDE-macbook.md               # Mac inference environment guide
├── CLAUDE-windows.md               # Windows/WSL training environment guide
├── README.md                       # Quick-start guide and architecture summary
├── pyproject.toml                  # Project metadata, ruff/mypy/pytest config, CalVer version
├── .pre-commit-config.yaml         # Pre-commit hooks (ruff check + format)
├── .commitlintrc.yml               # Conventional commit enforcement config
├── codecov.yml                     # Codecov coverage config (ignore engines, training)
├── todo.md                         # Phased task list
│
├── .github/
│   ├── dependabot.yml              # Automated dependency updates (pip + actions)
│   ├── labeler.yml                 # PR auto-labeling rules by path
│   └── workflows/
│       ├── lint.yml                # Ruff, mypy, bandit, vulture, HTML tidy
│       ├── test.yml                # pytest + coverage threshold (≥18%) + PR comment + Codecov
│       ├── release.yml             # GitHub Release on version tags
│       ├── security.yml            # pip-audit weekly + on push
│       ├── label.yml               # Auto-label PRs by changed paths
│       ├── commitlint.yml          # Conventional commit format check
│       └── stale.yml               # Auto-close stale issues/PRs
│
├── tests/                          # 150+ tests (conftest.py + 18 test_*.py files)
│   └── ...                         # engines, settings, glossary, pipeline, QE, dry-run, features
│
├── dry_run_ab.py                   # Main pipeline: mic → VAD → STT → translate → WebSocket + HTTP
├── settings.py                     # Unified pydantic-settings config (STARK_ env prefix, .env)
├── setup_models.py                 # One-command model download + verification
├── build_glossary.py               # EN→ES theological glossary builder (229 terms)
├── download_sermons.py             # yt-dlp sermon downloader
│
├── displays/
│   ├── audience_display.html       # Projector display (EN/ES side-by-side, QR overlay, fullscreen)
│   ├── ab_display.html             # A/B/C operator comparison (Gemma 4B / MarianMT / 12B)
│   ├── mobile_display.html         # Phone/tablet responsive view (model toggle, Spanish-only)
│   ├── church_display.html         # Simplified church-oriented layout
│   └── obs_overlay.html            # OBS streaming overlay
│
├── engines/
│   ├── __init__.py
│   ├── base.py                     # ABCs: STTEngine, TranslationEngine, result dataclasses
│   ├── mlx_engine.py               # MLXWhisperEngine, MLXGemmaEngine, MarianEngine
│   ├── cuda_engine.py              # FasterWhisperEngine, CUDAGemmaEngine
│   ├── factory.py                  # create_stt_engine(), create_translation_engine()
│   └── active_learning.py          # Fallback event JSONL logger
│
├── tools/
│   ├── live_caption_monitor.py     # YouTube caption comparison system
│   ├── translation_qe.py           # Reference-free translation quality estimation
│   ├── benchmark_latency.py        # End-to-end latency benchmarking
│   ├── stt_benchmark.py            # STT-specific benchmarking
│   ├── convert_models_to_both.py   # Model format conversion (MLX ↔ CUDA)
│   └── test_adaptive_model.py      # Adaptive model testing
│
├── features/
│   ├── diarize.py                  # Speaker diarization via pyannote-audio
│   ├── summarize_sermon.py         # Post-sermon 5-sentence summary generator
│   └── extract_verses.py           # Bible verse reference extraction from transcripts
│
├── training/                       # All WSL — 14 scripts
│   ├── preprocess_audio.py         # 10-step audio cleaning pipeline
│   ├── transcribe_church.py        # Whisper large-v3 pseudo-labeling
│   ├── prepare_*.py                # Bible corpus, Whisper dataset, Piper dataset builders
│   ├── train_*.py                  # Whisper LoRA, Gemma QLoRA, MarianMT, Piper TTS
│   ├── evaluate_*.py               # SacreBLEU/chrF++/COMET, Piper quality
│   └── assess_quality.py           # Baseline transcript quality assessment
│
├── docs/                           # 13 docs — training plans, roadmaps, troubleshooting
│   ├── training_plan.md            # Full training schedule + go/no-go gates
│   ├── macos_libomp_fix.md         # libomp conflict diagnosis + fix
│   └── ...                         # seattle run, accent tuning, multilingual, RTX 2070, OBS integration
│
├── fine_tuned_whisper_mi/          # LoRA adapters for Whisper (post fine-tune)
├── fine_tuned_gemma_mi_A/          # LoRA adapters for Gemma 4B
├── fine_tuned_gemma_mi_B/          # LoRA adapters for Gemma 12B
│
├── stark_data/                     # Church audio WAVs + JSON transcripts
│   ├── raw/                        # Original downloads from yt-dlp
│   ├── cleaned/                    # Post-preprocessing audio
│   ├── transcripts/                # JSON transcription labels
│   ├── corrections/                # Human-corrected segments (feedback loop)
│   └── live_sessions/              # Per-chunk WAV files from live runs (for fine-tuning)
│
├── bible_data/                     # Biblical parallel text for translation fine-tuning
│   ├── en/                         # English verse text (KJV, ASV, WEB, BBE, YLT)
│   ├── es/                         # Spanish verse text (RVR1909, Español Sencillo)
│   ├── aligned/                    # Verse-aligned JSONL pairs (~155K)
│   ├── glossary/                   # Theological term glossary (EN→ES, 229 terms)
│   └── holdout/                    # Stratified test set (~3.1K verses by genre)
│
├── metrics/
│   ├── ab_metrics.csv              # A/B test output logs
│   ├── caption_comparison.jsonl    # YouTube vs. local WER logs
│   ├── confidence_flags.jsonl      # Low-confidence segment queue
│   └── translation_qe.jsonl        # Translation quality scores
│
├── requirements-mac.txt            # Mac pip dependencies
├── requirements-nvidia.txt         # NVIDIA/CUDA inference dependencies
├── requirements-windows.txt        # Windows/WSL training pip dependencies
└── stt_env/                        # Python 3.11 virtualenv (gitignored)
```

---

## Architecture: The Robust Pipeline

The pipeline has six quality layers. Each layer catches different failure modes, and together they form a self-improving system.

### Layer 1: Audio Preprocessing (Windows/WSL)

A strict 10-step pipeline transforms raw YouTube church audio into clean training data. Order matters — each step's output feeds the next.

1. **Download** — `yt-dlp` pulls highest-quality audio from Stark Road Gospel Hall YouTube
2. **Format** — Convert to 16kHz mono WAV (Whisper's required format) via ffmpeg
3. **Initial quality gate** — Measure SNR (reject < 10 dB), detect clipping (reject > 1% samples at ±0.99)
4. **Classify segments** — `inaSpeechSegmenter` tags speech / music / noise regions
5. **Source separation** — `demucs` (`htdemucs`, `--two-stems vocals`) isolates vocals from music segments
6. **Denoise** — ffmpeg bandpass (80Hz–8kHz) then `noisereduce` (non-stationary, `prop_decrease=0.6–0.8`)
7. **Normalize** — `pyloudnorm` to -16 LUFS, true peak limit -1 dBTP
8. **VAD chunking** — `silero-vad` detects speech; WhisperX-style cut-and-merge to 1–30s segments with 100ms padding
9. **Speaker diarization** (optional) — `pyannote-audio` to filter for primary speaker
10. **Final quality gate** — Reject chunks with SNR < 15 dB, duration outside 1–30s, or silence ratio > 50%

**Critical insight:** Don't over-clean. Whisper was trained on noisy, diverse audio. Moderate noise in training data improves inference robustness. Match training noise profile to expected demo conditions.

### Layer 2: Data Quality Assessment (Windows/WSL)

Before committing to fine-tuning, establish a baseline:

- Sample 50–100 segments stratified by speaker, audio quality, and topic
- Manually transcribe the sample (~5–8x real-time effort)
- Compute WER using `jiwer`

| Baseline WER | Strategy |
|-------------|----------|
| < 10% | Use directly with confidence-based filtering |
| 10–20% | Filter out worst segments by `avg_logprob` |
| 20–30% | Weakly supervised pretraining on all noisy data, then fine-tune on small clean subset |
| > 30% | Re-transcribe everything with Whisper large-v3 instead of YouTube auto-captions |

**Strong recommendation:** Re-transcribe all audio with Whisper large-v3 (~7% WER) rather than using YouTube auto-captions (~15% WER) as training labels. Use YouTube captions only as a live monitoring signal.

### Layer 3: Confidence-Based Flagging (Mac — Inference Time)

Whisper exposes three segment-level quality signals:

| Metric | Good | Flag for review | Auto-reject |
|--------|------|----------------|-------------|
| `avg_logprob` | > -0.3 | < -0.5 | < -1.0 |
| `no_speech_prob` | < 0.1 | > 0.3 | > 0.6 (with low logprob) |
| `compression_ratio` | < 1.8 | > 2.0 | > 2.4 (hallucination) |

Word-level confidence via `faster-whisper` (`word_timestamps=True`) or `whisper-timestamped`. Flag any word with probability < 0.3. Route bottom 5–15% of segments to human review queue.

**Caveat:** Token confidence mixes language model and acoustic signals. High-frequency function words may score high even when misrecognized. Use segment-level aggregation over individual word scores.

### Layer 4: Live YouTube Caption Comparison (Mac — Inference Time)

Run local Whisper transcription simultaneously with YouTube livestream caption extraction. This is not ground truth — it's a noisy reference signal for trend monitoring.

**Architecture:**
- Local channel: `streamlink` captures live audio → Whisper processes in real time
- YouTube channel: Poll InnerTube timed-text endpoint every 5–10s, or use `youtube-transcript-api` / `yt-dlp --write-auto-subs` post-stream
- Alignment: 30-second windowed comparison with sliding-window offset search to handle latency drift
- Metrics: `jiwer` computes per-window WER, CER with word-level diffs

**Interpreting cross-system WER:** 5–15% = normal disagreement, 15–25% = potential issues, >25% = likely degradation. Track trends over sessions, not absolute values. Log everything to JSONL.

### Layer 5: Translation Quality Estimation (Mac — Inference Time)

Monitor English→Spanish quality without ground-truth Spanish references. Three tiers, escalating:

**Tier 1 (always-on, per-segment, ~150–300ms):**
- **CometKiwi** (`Unbabel/wmt22-cometkiwi-da`) — source + translation → 0–1 score. Good: > 0.85, review: 0.50–0.70, critical: < 0.50
- **LaBSE** cross-lingual cosine similarity — good translations > 0.85
- Length ratio check (Spanish typically 15–25% longer than English)
- Untranslated content detection via regex

**Tier 2 (triggered by Tier 1 flags):**
- **Back-translation** via MarianMT (`Helsinki-NLP/opus-mt-es-en`, ~75MB) → compare back-translated English to original via BERTScore (F1 > 0.90 = good)
- **LanguageTool** (`language_tool_python`, `es` locale) for Spanish grammar checks

**Tier 3 (offline batch):**
- BLASER 2.0 QE (Meta SONAR) for deep fluency analysis
- Spanish LM perplexity via BETO (`dccuchile/bert-base-spanish-wwm-cased`)

### Layer 6: Active Learning Feedback Loop (Both Machines)

Four stages per cycle: **infer → flag → correct → retrain**

1. Mac runs inference with confidence scoring, flags low-quality segments
2. Flagged segments routed to human review queue (Label Studio or Prodigy)
3. Corrections stored in versioned dataset (`stark_data/corrections/`)
4. WSL desktop retrains with LoRA on updated data

**Convergence:** 3–5 cycles typical. First cycle often yields 20–40% relative WER reduction.

**Anti-forgetting safeguards:**
- LoRA inherently preserves base weights (disable adapter = instant baseline recovery)
- Replay buffer: mix 70% general-domain + 30% church-domain samples
- Learning rate 40x lower than pretraining (e.g., ~6.25e-5 for Whisper-large)
- Elastic Weight Consolidation if forgetting persists

**Curriculum learning:** Clean → medium → hard samples across training phases. Yields ~5–7% additional WER reduction vs. random ordering.

---

## Fine-Tuning Strategy: Biblical Text & Church Audio

This section covers the complete domain adaptation plan for both models. Training runs on the Windows/WSL desktop; adapters transfer to the Mac for inference.

### Biblical Parallel Text Corpus (~155K Verse Pairs)

The Bible translation community provides some of the best-organized parallel corpora in NLP. Three sources provide immediate training data:

**Primary datasets:**
- `bible-nlp/biblenlp-corpus` on HuggingFace — 833 languages, verse-aligned, per-translation license metadata, CC-BY-4.0
- `Helsinki-NLP/bible_para` — Christodoulopoulos/Steedman Bible corpus, standard parallel format, CC0-1.0
- `scrollmapper/bible_databases` on GitHub — ASV, KJV, WEB, BBE, YLT (English) + RVR1909, Español Moderno (Spanish) in SQL/JSON/CSV with numeric verse IDs (e.g., `01001001` = Genesis 1:1) for trivial cross-translation alignment

The Protestant Bible contains ~31,102 verses per translation. Pairing five public-domain English translations against two freely licensed Spanish translations yields ~155,000 parallel verse pairs:

| Pair | Register | License |
|------|----------|---------|
| KJV ↔ RVR1909 | Formal-to-formal, archaic | Public domain |
| ASV ↔ RVR1909 | Formal-to-formal, slightly modern | Public domain |
| WEB ↔ Español Sencillo | Modern-to-modern, accessible | PD / CC BY-SA 4.0 |
| BBE ↔ RVR1909 | Simplified English to formal Spanish | Public domain |
| YLT ↔ RVR1909 | Hyper-literal English to formal Spanish | Public domain |

**Copyright warning:** Do NOT use ESV, NASB, NIV, NLT, NVI, LBLA, RVR1960, or DHH as training data. Their "fair use" policies cap at ~500 verses for quotation — not bulk ML training. Stick to pre-1923 or explicitly dedicated public-domain translations.

**Supplementary data:** Theological glossary (229 terms, covering all 66 books, 31 proper names, theological concepts), bilingual catechism excerpts (Westminster Shorter, Heidelberg), and any bilingual sermon transcripts the church produces.

### Sermon Audio Data via Pseudo-Labeling

No dedicated English church sermon ASR dataset exists publicly. Build your own using the pseudo-labeling strategy (same approach used to train Whisper large-v3 itself on 4M hours):

1. Collect 20–50 hours of church audio (soundboard recordings preferred over room mics)
2. Segment into 5–30 second chunks via Silero VAD
3. Run base Distil-Whisper to generate initial transcriptions
4. Filter by confidence: discard segments with compression ratio > 2.4 or avg log-prob < -1.0
5. Human-correct the bottom 20% by confidence (prioritize theological terms and biblical names)
6. Format as HuggingFace Dataset with `audio` and `sentence` columns

Church audio challenges: reverberant sanctuaries (>2 second RT60), PA system artifacts (comb filtering, delay arrivals), unique background noise patterns (congregation responses, worship music transitions, HVAC, page-turning).

**Data volume thresholds:** 5–10 hours for vocabulary improvement, 20–50 hours for strong domain adaptation (sweet spot), 50–100+ hours for production-grade results.

### Whisper LoRA Configuration

Apply LoRA to both encoder (acoustic domain shift) and decoder (vocabulary patterns). Target `q_proj` and `v_proj` at minimum; expand to `k_proj`, `out_proj`, `fc1`, `fc2` for maximum adaptation.

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Rank (r) | 32 | Most validated across Whisper LoRA studies |
| Alpha | 64 | 2× rank, standard scaling |
| Target modules | `q_proj`, `v_proj` | Minimum viable; expand if needed |
| Dropout | 0.05 | Light regularization |
| Learning rate | 1e-4 | Standard for LoRA (higher than full FT) |
| Batch size | 4 (effective 16 via grad accum) | Conservative for 16GB VRAM |
| Precision | BF16 | Ada GPUs support natively |
| Gradient checkpointing | Enabled | Essential memory savings |
| Max steps | 4,000 (or 3–5 epochs) | First cycle; adjust by convergence |
| VRAM usage | ~8–10 GB | Comfortable on A2000 Ada |

**Catastrophic forgetting is minimal with LoRA** — base weights are frozen, preserving general English recognition. For extra safety, mix 70–80% domain data with 20–30% general English ASR data (LibriSpeech/Common Voice).

### TranslateGemma QLoRA Configuration

The 4B variant loads at ~2.6 GB in 4-bit, leaving ample headroom for training on 16GB VRAM. Uses HuggingFace TRL's SFTTrainer with bitsandbytes 4-bit quantization.

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Rank (r) | 16 | Validated for domain-specific translation FT |
| Alpha | 16 | 1× rank for QLoRA |
| Target modules | `all-linear` | All linear layers for best adaptation quality |
| Quantization | NF4, double quant, BF16 compute | Standard QLoRA recipe |
| Batch size | 1 (effective 4 via grad accum) | Safe for 16GB |
| Learning rate | 2e-4 | Standard QLoRA rate |
| Epochs | 3 | ~155K verse pairs |
| Packing | Enabled | Multiple short verses per sequence |
| Optimizer | `paged_adamw_32bit` | Memory-efficient |
| Max seq length | 512 | Bible verses rarely exceed 200 tokens |
| VRAM usage | ~10–12 GB | Comfortable on A2000 Ada |

**TranslateGemma chat template** must be followed exactly — it requires `source_lang_code` and `target_lang_code` fields in the user message content.

**MarianMT for partials:** `Helsinki-NLP/opus-mt-en-es` runs as PyTorch (~298MB, ~80ms) for fast partial translations during speech, while TranslateGemma handles final translations on silence. MarianMT also supports full fine-tuning without LoRA for lower quality ceiling but dramatically higher iteration speed.

### Theological Vocabulary Challenges

Critical translations that require context-dependent disambiguation:

| English | Spanish Options | Context Rule |
|---------|----------------|--------------|
| Atonement | *expiación* vs *propiciación* | Removal of sin vs. appeasing wrath |
| Covenant | *pacto* (Protestant) vs *alianza* (Catholic) | Match target audience |
| Righteousness | *justicia* | Also means "justice" — theological context needed |
| James (apostle) | *Jacobo* | Person context (Mark 3:17) |
| James (epistle) | *Santiago* | Book/letter context |
| Sanctification | *santificación* | Consistent across contexts |
| Grace | *gracia* | Also means "humor/charm" — theological context needed |

**Mitigation strategies:** A 229-term theological glossary has been built covering all 66 books, 31 proper names, theological concepts, and liturgical terms. Used for soft constraint training (append target terms to source during training), dictionary augmentation (inject glossary pairs as training examples), and theological term spot-checking in evaluation.

### Evaluation Strategy

**Automatic metrics:** SacreBLEU (n-gram precision), chrF++ (character-level, better for Spanish morphology), COMET (neural, highest human correlation). Use all three together.

**Holdout test set:** ~3,100 verses (10%) stratified by biblical genre — Pentateuch, History, Poetry, Prophecy, Gospels, Epistles, Apocalyptic. Use multiple Spanish reference translations (RVR1909 + Español Sencillo) for more robust scoring.

**Human evaluation dimensions:** Adequacy (full meaning preserved?), fluency (natural Spanish?), theological precision (doctrinal terms correct?). The last dimension is the most critical and cannot be captured by automatic metrics.

**Expected improvement targets post fine-tuning:**
- SacreBLEU: +3–8 points on biblical text
- chrF++: +2–5 points
- COMET: +0.02–0.05
- Theological term accuracy: 80%+ (vs. ~40–60% base model)
- Whisper WER on church audio: 10–30% relative reduction

### Related Work

Bible NMT is an active research area, primarily for low-resource languages. The eBible Corpus paper (2023) found NLLB-600M outperformed both SMT and OpenNMT on Bible translation. "From Priest to Doctor" (COLING 2025) confirmed that Bible-trained models carry strong domain transfer. The BibleNLP community (`biblenlp.github.io`) maintains the richest ecosystem with 833-language coverage.

No published work exists on fine-tuning Whisper for church/religious speech — making this a genuine gap. Domain-adapted Whisper studies in aviation (WER 70% → 28% with LoRA) and industrial jargon (~1–2% WER after 5 epochs) demonstrate the techniques transfer directly.

---

## A/B Testing Protocol

1. **Sessions:** 5-minute reads — MI news, sermon excerpts, conversational English
2. **Compare:** Latency histograms, error rates per phrase length
3. **Edge cases:** Test accents, background noise (fan hum, coffee shop ambiance)
4. **Metrics columns:** `approach`, `latency_ms`, `stt_wer`, `bleu`, `en_text`, `es_text`
5. **Measured baselines:** Partials ~380ms (MarianMT PyTorch), finals A ~1.3s (4B), finals B ~1.9s (12B)
6. **Scaling:** Wrap in loop for 10+ runs; use `timeit` for precise RTF
7. **Serving:** HTTP on `0.0.0.0:8080` (LAN/phone access), WebSocket on `0.0.0.0:8765`

---

## CI/CD Pipeline

Seven GitHub Actions workflows enforce quality on every push and PR to `main`:

| Workflow | Trigger | What it does |
|----------|---------|-------------|
| **Lint** (`lint.yml`) | push / PR | Ruff check + format, mypy, bandit security scan, vulture dead code (advisory), HTML tidy |
| **Test** (`test.yml`) | push / PR | pytest (150+ tests, Python 3.11 + 3.12), coverage threshold (≥18%), Codecov upload, PR comment |
| **Release** (`release.yml`) | `v*` tag push | Creates GitHub Release from tag |
| **Security** (`security.yml`) | push / PR / weekly | pip-audit on both requirements files |
| **Label** (`label.yml`) | PR | Auto-labels PRs by changed paths (engines, displays, training, etc.) |
| **Commitlint** (`commitlint.yml`) | PR | Enforces conventional commit format (`feat:`, `fix:`, `docs:`, `ci:`, etc.) |
| **Stale** (`stale.yml`) | scheduled | Auto-closes stale issues and PRs |
| **Dependabot** (`dependabot.yml`) | weekly | Opens PRs for pip (minor/patch) and Actions version updates |

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

CalVer format: `YYYY.M.W.PATCH` (e.g., `2026.2.4.0` = 2026, February, week 4, first release). Single source of truth in `pyproject.toml`.

---

## Key Library Reference

| Library | Role | Mac | WSL |
|---------|------|-----|-----|
| `mlx-whisper` | STT inference (large-v3-turbo primary, distil-large-v3.5 fallback) | ✅ (MLX) | — |
| `mlx-lm` | TranslateGemma 4B/12B 4-bit inference | ✅ (MLX) | — |
| `Helsinki-NLP/opus-mt-en-es` | MarianMT PyTorch (298MB) | ✅ (CPU) | ✅ (CUDA) |
| `distil-whisper/distil-large-v3.5` | STT model (training) | — | ✅ (CUDA) |
| `google/translategemma-4b` / `12b` | Translation (training) | — | ✅ (CUDA) |
| `silero-vad` | Voice activity detection | ✅ | ✅ |
| `noisereduce` | Spectral noise reduction | ✅ | ✅ |
| `demucs` v4 | Source separation | ✅ (MPS/CPU) | ✅ (CUDA) |
| `pyloudnorm` | Loudness normalization | ✅ | ✅ |
| `inaSpeechSegmenter` | Speech/music classification | ✅ | ✅ |
| `pyannote-audio` 3.1 | Speaker diarization | ✅ (CPU) | ✅ (CUDA) |
| `jiwer` | WER/CER computation | ✅ | ✅ |
| `faster-whisper` | Confidence scores (CTranslate2 backend) | ✅ (CPU) | ✅ |
| `whisper-timestamped` | Word-level confidence | ✅ | ✅ |
| `Unbabel/wmt22-cometkiwi-da` | Translation QE | ✅ (CPU) | — |
| `sentence-transformers/LaBSE` | Cross-lingual similarity | ✅ | — |
| `Helsinki-NLP/opus-mt-es-en` | Back-translation | ✅ | — |
| `language_tool_python` | Spanish grammar | ✅ | — |
| `peft` (LoRA) | Parameter-efficient fine-tuning | — | ✅ (CUDA) |
| `trl` (SFTTrainer) | Supervised fine-tuning w/ packing | — | ✅ (CUDA) |
| `bitsandbytes` | 4-bit quantization (QLoRA, CUDA-only) | — | ✅ (CUDA) |
| `sacrebleu` + `unbabel-comet` | Translation eval (BLEU/chrF++/COMET) | ✅ | ✅ |
| `Label Studio` | Annotation / correction UI | ✅ (web) | ✅ (web) |
| `streamlit` | Dashboard UI | ✅ | — |
| `youtube-transcript-api` | YT caption extraction | ✅ | — |

---

## Future Features

> **Note:** Initial implementations of the summary and verse extraction features now exist as `features/summarize_sermon.py` and `features/extract_verses.py`. Speaker diarization is implemented in `features/diarize.py`. These are ready for integration testing with live session data.

### 1. Post-Sermon 5-Sentence Summary

After each service, the pipeline generates a structured 5-sentence summary from the full transcript:

- **Sentence 1:** Overall gist covering both speakers' themes
- **Sentences 2–3:** Summary of Speaker 1's key points
- **Sentences 4–5:** Summary of Speaker 2's key points

**Implementation:** Feed the full diarized transcript (Speaker A segments + Speaker B segments) to a local LLM (Gemma 3 4B or similar) with a structured prompt constraining output to exactly 5 sentences in the format above. Speaker diarization from `pyannote-audio` (Layer 1, Step 9) provides the per-speaker segmentation. Output as both English and Spanish summaries. Can run as a batch job post-service — no real-time constraint.

**Dependencies:** Requires speaker diarization to be enabled (currently optional in the preprocessing pipeline). Two-speaker assumption can be enforced via `pyannote` with `min_speakers=2, max_speakers=2`.

### 2. Verse Reference Extraction

Return a per-speaker list of every Bible verse referenced during the sermon. Two-pass approach: (1) regex + rule-based extraction for explicit citations ("Romans 8:28"), (2) LLM-assisted extraction for spoken-form references ("turn to Romans chapter eight"). Outputs per-speaker JSON with reference, timestamp, and context snippet. Stretch goal: cross-reference extracted verses against translation output for theological accuracy checks.

---

## Compute Timeline — Training & Fine-Tuning

All training runs on the **Windows Desktop** (A2000 Ada 16GB, ~16 TFLOPS FP16, ~2× T4). Total project estimate: **~48–73 GPU-hrs**, **~33–53 human-hrs** over ~5 weeks.

| Task | Wall Clock | VRAM |
|------|-----------|------|
| Bible corpus download + alignment | ~15 min | CPU |
| Sermon audio download (50 hrs) | ~1–2 hrs | Network |
| Audio preprocessing (10-step pipeline) | ~4–6 hrs | GPU (demucs bottleneck) |
| Pseudo-labeling (Whisper large-v3, 50 hrs) | ~3–5 hrs | ~8 GB |
| Human correction (bottom 20%, ~12K segs) | ~15–25 hrs | Manual |
| **Distil-Whisper LoRA** (20 hrs audio, 3 epochs) | ~5–8 hrs | ~8–10 GB |
| **Distil-Whisper LoRA** (50 hrs audio, 3 epochs) | ~11–15 hrs | ~8–10 GB |
| **TranslateGemma 4B QLoRA** (155K pairs, 3 epochs) | ~8–12 hrs | ~10–12 GB |
| **TranslateGemma 12B QLoRA** (155K pairs, 3 epochs) | ~18–27 hrs | ~14–15 GB |
| Evaluation (BLEU/WER/COMET on holdout) | ~30–60 min | ~6 GB |

**Cycle timing:** Cycle 1 ~40–62 hrs (includes one-time data prep). Cycles 2–5 ~17–30 hrs each (re-label → correct → retrain → eval).

**Key bottlenecks:** Human correction dominates cycle 1 (15–25 hrs, shrinks each cycle). TranslateGemma 12B is tight on VRAM (~14–15 GB peak; reduce batch to 1 + grad accum 8 if OOM). Demucs is the data-prep GPU bottleneck — run overnight. Both LoRA and QLoRA fit within a single overnight run (8–15 hrs).

---

## Next Steps (Ordered)

- [x] **Phase 0 — Setup:** Configure both environments per machine-specific docs
- [x] **Phase 1 — Baseline:** Run base A/B test (no fine-tuning) to establish latency and WER baselines
- [x] **Phase 1.5 — CI/CD:** GitHub Actions (lint, test, security, release), pre-commit, CalVer versioning, 130+ tests
- [ ] **Phase 2 — Data collection:** Download and sample 10–20 hours of Stark Road audio via `yt-dlp`
- [ ] **Phase 3 — Quality assessment:** Manually transcribe 50–100 sample segments, compute baseline WER
- [ ] **Phase 4 — Preprocessing:** Run the 10-step audio cleaning pipeline on all collected data
- [ ] **Phase 5 — Re-transcribe:** Generate clean labels with Whisper large-v3 (not YouTube auto-captions)
- [ ] **Phase 6 — Fine-tune (round 1):** LoRA training for Whisper + Gemma on WSL desktop
- [ ] **Phase 7 — Evaluate:** Transfer adapters to Mac, re-run A/B with fine-tuned models + live YT comparison
- [ ] **Phase 8 — Feedback loop:** Route flagged segments to correction → retrain (repeat 2–4 more cycles)
- [ ] **Phase 9 — Demo:** Deploy Streamlit dashboard for Farmington Hills coffee shop outreach event
- [ ] **Phase 10 — Integrate:** macOS Shortcuts for voice-command triggers, `streamlit-webrtc` for true live mic
