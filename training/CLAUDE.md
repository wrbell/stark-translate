# training/ — Fine-Tuning & Data Preparation (Windows/WSL)

All training runs on the **Windows Desktop** (WSL2, A2000 Ada 16GB, 64GB RAM). Adapters transfer to Mac for inference. See `CLAUDE-windows.md` for environment setup.

## Overview

Domain adaptation for two models:
1. **Whisper LoRA** — church sermon STT (English acoustic domain)
2. **TranslateGemma QLoRA** — biblical EN↔ES translation

## Audio Preprocessing (10-Step Pipeline)

Raw YouTube church audio → clean training data. Order matters.

1. **Download** — `yt-dlp` from Stark Road Gospel Hall YouTube
2. **Format** — 16kHz mono WAV via ffmpeg
3. **Initial quality gate** — SNR (reject < 10 dB), clipping (reject > 1% at ±0.99)
4. **Classify segments** — `inaSpeechSegmenter` tags speech/music/noise
5. **Source separation** — `demucs` (`htdemucs`, `--two-stems vocals`)
6. **Denoise** — ffmpeg bandpass (80Hz–8kHz) + `noisereduce` (non-stationary, `prop_decrease=0.6–0.8`)
7. **Normalize** — `pyloudnorm` to -16 LUFS, true peak -1 dBTP
8. **VAD chunking** — `silero-vad`, 1–30s segments, 100ms padding
9. **Speaker diarization** (optional) — `pyannote-audio` for primary speaker
10. **Final quality gate** — SNR > 15 dB, duration 1–30s, silence ratio < 50%

**Critical insight:** Don't over-clean. Whisper was trained on noisy audio. Match training noise to expected demo conditions.

## Data Quality Assessment

Before fine-tuning, establish a baseline on 50–100 stratified segments:

| Baseline WER | Strategy |
|-------------|----------|
| < 10% | Use directly with confidence-based filtering |
| 10–20% | Filter worst segments by `avg_logprob` |
| 20–30% | Weakly supervised pretraining, then fine-tune on clean subset |
| > 30% | Re-transcribe with Whisper large-v3 instead of YouTube captions |

**Recommendation:** Re-transcribe all audio with Whisper large-v3 (~7% WER) rather than YouTube auto-captions (~15% WER).

## Biblical Parallel Text Corpus (~155K Verse Pairs)

**Primary datasets:**
- `bible-nlp/biblenlp-corpus` (HuggingFace, 833 languages, CC-BY-4.0)
- `Helsinki-NLP/bible_para` (CC0-1.0)
- `scrollmapper/bible_databases` (GitHub, SQL/JSON/CSV, numeric verse IDs)

| Pair | Register | License |
|------|----------|---------|
| KJV ↔ RVR1909 | Formal-to-formal, archaic | Public domain |
| ASV ↔ RVR1909 | Formal-to-formal, slightly modern | Public domain |
| WEB ↔ Español Sencillo | Modern-to-modern | PD / CC BY-SA 4.0 |
| BBE ↔ RVR1909 | Simplified EN to formal ES | Public domain |
| YLT ↔ RVR1909 | Hyper-literal EN to formal ES | Public domain |

**COPYRIGHT WARNING:** Do NOT use ESV, NASB, NIV, NLT, NVI, LBLA, RVR1960, or DHH. Fair use caps at ~500 verses — not bulk ML training. Stick to pre-1923 or public-domain translations.

**Supplementary data:** 229-term theological glossary, bilingual catechism excerpts, bilingual sermon transcripts.

## Sermon Audio via Pseudo-Labeling

1. Collect 20–50 hours of church audio (soundboard preferred)
2. Segment into 5–30s chunks via Silero VAD
3. Run base Distil-Whisper for initial transcriptions
4. Filter by confidence (compression ratio > 2.4 or avg log-prob < -1.0 → discard)
5. Human-correct bottom 20% (prioritize theological terms, biblical names)
6. Format as HuggingFace Dataset (`audio` + `sentence` columns)

**Data volume thresholds:** 5–10h = vocabulary improvement, 20–50h = strong adaptation (sweet spot), 50–100h+ = production-grade.

## Whisper LoRA Configuration

Target `q_proj` + `v_proj` (minimum); expand to `k_proj`, `out_proj`, `fc1`, `fc2` for maximum adaptation.

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Rank (r) | 32 | Most validated across Whisper LoRA studies |
| Alpha | 64 | 2× rank, standard scaling |
| Target modules | `q_proj`, `v_proj` | Minimum viable |
| Dropout | 0.05 | Light regularization |
| Learning rate | 1e-4 | Standard LoRA |
| Batch size | 4 (effective 16 via grad accum) | Conservative for 16GB VRAM |
| Precision | BF16 | Ada native |
| Gradient checkpointing | Enabled | Essential memory savings |
| Max steps | 4,000 (or 3–5 epochs) | First cycle |
| VRAM usage | ~8–10 GB | Comfortable on A2000 Ada |

Mix 70–80% domain data with 20–30% general English (LibriSpeech/Common Voice) for catastrophic forgetting safety.

## TranslateGemma QLoRA Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Rank (r) | 16 | Validated for domain-specific translation |
| Alpha | 16 | 1× rank for QLoRA |
| Target modules | `all-linear` | All linear layers |
| Quantization | NF4, double quant, BF16 compute | Standard QLoRA |
| Batch size | 1 (effective 4 via grad accum) | Safe for 16GB |
| Learning rate | 2e-4 | Standard QLoRA rate |
| Epochs | 3 | ~155K verse pairs |
| Packing | Enabled | Multiple verses per sequence |
| Optimizer | `paged_adamw_32bit` | Memory-efficient |
| Max seq length | 512 | Verses rarely exceed 200 tokens |
| VRAM usage | ~10–12 GB | Comfortable on A2000 Ada |

**TranslateGemma chat template** requires `source_lang_code` and `target_lang_code` fields exactly.

**MarianMT** (`Helsinki-NLP/opus-mt-en-es`, ~298MB, ~80ms) supports full fine-tuning without LoRA — lower quality ceiling but faster iteration.

## Theological Vocabulary Challenges

| English | Spanish Options | Context Rule |
|---------|----------------|--------------|
| Atonement | *expiación* vs *propiciación* | Removal of sin vs. appeasing wrath |
| Covenant | *pacto* (Protestant) vs *alianza* (Catholic) | Match audience |
| Righteousness | *justicia* | Also "justice" — theological context needed |
| James (apostle) | *Jacobo* | Person context (Mark 3:17) |
| James (epistle) | *Santiago* | Book/letter context |
| Sanctification | *santificación* | Consistent |
| Grace | *gracia* | Also "humor/charm" — theological context needed |

**Mitigation:** 229-term glossary for soft constraint training, dictionary augmentation, and spot-checking.

## Evaluation Strategy

**Automatic metrics:** SacreBLEU + chrF++ + COMET (use all three).

**Holdout:** ~3,100 verses (10%) stratified by genre — Pentateuch, History, Poetry, Prophecy, Gospels, Epistles, Apocalyptic. Multiple Spanish references (RVR1909 + Español Sencillo).

**Human evaluation:** Adequacy, fluency, theological precision (most critical, not captured by auto metrics).

**Expected improvement targets:**
- SacreBLEU: +3–8 points | chrF++: +2–5 | COMET: +0.02–0.05
- Theological term accuracy: 80%+ (vs. ~40–60% base)
- Whisper WER: 10–30% relative reduction

## Anti-Forgetting Safeguards

- LoRA preserves base weights (disable adapter = instant baseline recovery)
- Replay buffer: 70% general-domain + 30% church-domain
- Learning rate 40× lower than pretraining (~6.25e-5 for Whisper-large)
- Elastic Weight Consolidation if forgetting persists
- **Curriculum learning:** Clean → medium → hard samples. Yields ~5–7% additional WER reduction.

## Compute Timeline

Total estimate: **~48–73 GPU-hrs**, **~33–53 human-hrs** over ~5 weeks.

| Task | Wall Clock | VRAM |
|------|-----------|------|
| Bible corpus download + alignment | ~15 min | CPU |
| Sermon audio download (50 hrs) | ~1–2 hrs | Network |
| Audio preprocessing (10-step) | ~4–6 hrs | GPU (demucs bottleneck) |
| Pseudo-labeling (Whisper large-v3) | ~3–5 hrs | ~8 GB |
| Human correction (bottom 20%) | ~15–25 hrs | Manual |
| Distil-Whisper LoRA (20h audio) | ~5–8 hrs | ~8–10 GB |
| Distil-Whisper LoRA (50h audio) | ~11–15 hrs | ~8–10 GB |
| TranslateGemma 4B QLoRA | ~8–12 hrs | ~10–12 GB |
| TranslateGemma 12B QLoRA | ~18–27 hrs | ~14–15 GB |
| Evaluation (BLEU/WER/COMET) | ~30–60 min | ~6 GB |

**Cycle timing:** Cycle 1 ~40–62 hrs (includes data prep). Cycles 2–5 ~17–30 hrs each.

**Key bottlenecks:** Human correction dominates cycle 1. TranslateGemma 12B tight on VRAM (~14–15 GB peak). Demucs is the data-prep GPU bottleneck — run overnight.

## Related Work

- eBible Corpus (2023): NLLB-600M outperformed SMT and OpenNMT on Bible translation
- "From Priest to Doctor" (COLING 2025): Bible-trained models carry strong domain transfer
- BibleNLP community (`biblenlp.github.io`): 833-language ecosystem
- No published Whisper fine-tuning for church/religious speech — genuine research gap
- Domain-adapted Whisper: aviation WER 70%→28% (LoRA), industrial jargon ~1–2% WER after 5 epochs

---

## Adapter Export & Transfer

After training, LoRA adapters must be transferred to inference machines:

1. **Location**: Adapters saved to `fine_tuned_*/` dirs on WSL (safetensors format)
2. **Transfer to Mac**: Copy adapter dir to `adapters/{model}/active/` (see `docs/deploy.md` for automated rsync pipeline)
3. **Validate on Mac**: Run health check — 5 canonical sentences, check expected substrings + latency < 5s per sentence
4. **Version naming**: `cycle{N}_{YYYYMMDD}_{sha256[:8]}` (SHA-256 over `adapter_model.safetensors`)
5. **Manifest**: `adapters/manifest.json` tracks all versions with training metrics (final loss, eval WER, epochs, data size)
6. **Rollback**: Two-slot system — `active/` + `previous/`. Swap back if health check fails post-activation.

## Adding a New Language Corpus

When adding Hindi or Chinese translation training (see `docs/multi_lingual.md`):

1. **Find aligned verse pairs** — `bible-nlp/biblenlp-corpus` has 833 languages (CC-BY-4.0). Hindi IRV: `hin2017` (~31K verses). Chinese CUV-S: `cmn-cu89s` (~31K verses, public domain).
2. **Prepare with `prepare_bible_corpus.py`** — input: two translation dirs, output: JSONL pairs with `source_lang_code`, `target_lang_code`, `source_text`, `target_text`, `verse_id`.
3. **Build theological glossary** (100–150 terms minimum) using `build_glossary.py` as template. Include honorifics (Hindi तू for divine address) and denomination-specific terms (Chinese 圣灵 not 圣神).
4. **QLoRA config**: Same as Spanish but **r=32** (new language direction needs higher rank). `max_seq_length=768` for Hindi (2.5–3.5x token fertility), 512 for Chinese.
5. **Evaluation**: chrF++ (primary — handles morphology and no-space scripts), COMET, theological term accuracy. Use `--tokenize zh` for Chinese SacreBLEU.
6. **Copyright**: Same rules apply — only pre-1923 or explicitly public domain translations. No ESV, NASB, NIV, NLT, NVI, LBLA, RVR1960, DHH.

## Go/No-Go Gates

Quick reference for training convergence criteria (from `docs/accent_tuning_plan.md` and evaluation strategy above):

- **WER improvement**: > 10% relative (minimum), > 20% relative (target)
- **BLEU improvement**: > +2 points (minimum), > +4 points (target)
- **Theological term accuracy**: > 65% (minimum), > 80% (target)
- **Accent fairness gap**: < 10% absolute (minimum), < 5% absolute (target)
- **Stop condition**: Improvement < 2% relative for 2 consecutive cycles on worst-performing metric
