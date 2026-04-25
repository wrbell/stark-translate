# tools/ — Monitoring, Benchmarking & Validation

## Tool Inventory

| File | Purpose |
|------|---------|
| `live_caption_monitor.py` | YouTube caption comparison (live, post-session, trend analysis) |
| `translation_qe.py` | Reference-free translation quality estimation |
| `validate_session.py` | Post-session validation vs YouTube captions |
| `benchmark_latency.py` | End-to-end latency profiling |
| `stt_benchmark.py` | STT-only benchmarking |
| `roundtrip_test.py` | End-to-end STT + translation roundtrip quality test |
| `prepare_finetune_data.py` | Fine-tuning data export from live sessions |
| `download_roundtrip_texts.py` | Download test texts for roundtrip testing |
| `convert_models_to_both.py` | Model format conversion (MLX ↔ CUDA) |
| `test_adaptive_model.py` | Adaptive model selection testing |
| `batch_translate.py` | Batch translation processing |
| `export_review.py` | Export review queue data |
| `glossary.py` | Tiered glossary: Tier 1 (50 boost for Deepgram) + Tier 2 (229 master) |
| `sort_sermons.py` | Sermon cataloging by type/year with manifest + training cutoff |
| `lock_data.py` | SHA-256 data lockfile for training data versioning |
| `build_eval_sets.py` | Stratified eval set builder (500 verse holdout + sermon eval) |
| `health_check.py` | Adapter health verification (5 canary sentences) |
| `manage_adapters.py` | Adapter lifecycle: register, activate, rollback, export |

## YouTube Caption Comparison (Layer 4)

Runs local Whisper STT simultaneously with YouTube livestream caption extraction. Not ground truth — a noisy reference for trend monitoring.

**Architecture:**
- Local channel: `streamlink` captures live audio → Whisper processes in real time
- YouTube channel: Poll InnerTube timed-text endpoint every 5–10s, or `youtube-transcript-api` / `yt-dlp --write-auto-subs` post-stream
- Alignment: 30-second windowed comparison with sliding-window offset search
- Metrics: `jiwer` per-window WER, CER with word-level diffs

**Interpreting cross-system WER:** 5–15% = normal disagreement, 15–25% = potential issues, >25% = likely degradation. Track trends over sessions. Log to JSONL.

## Text-Anchor Alignment (CRITICAL)

Live sessions use wall-clock timestamps; YouTube starts from stream beginning. The offset can be 10–30+ minutes. The narrow ±5s search (`find_best_offset`) will always fail for this.

**Solution:** `find_global_offset_by_text()` fuzzy-matches phrases from the live session against YouTube transcript to find the correct offset.

**The fallback condition must check `_wer is None`** — not just `> 0.6`. When there's zero overlapping text (large offset), `find_best_offset` returns `wer=None`.

**2026-03-01 session reference:** Video `UF0QFnoWZJ4`, offset = -533.8s (~8.9 min), combined WER = 19.6% across 76/81 windows. YouTube segments cached at `stark_data/live_sessions/20260301/UF0QFnoWZJ4_segments.json`.

## Translation Quality Estimation (Layer 5)

Monitor EN↔ES quality without ground-truth references. Three tiers, escalating:

**Tier 1 (always-on, per-segment, ~150–300ms):**
- **CometKiwi** (`Unbabel/wmt22-cometkiwi-da`) — source + translation → 0–1 score. Good: > 0.85, review: 0.50–0.70, critical: < 0.50
- **LaBSE** cross-lingual cosine similarity — good > 0.85
- Length ratio check (Spanish typically 15–25% longer than English)
- Untranslated content detection via regex

**Tier 2 (triggered by Tier 1 flags):**
- **Back-translation** via MarianMT (`Helsinki-NLP/opus-mt-es-en`, ~75MB) → BERTScore (F1 > 0.90 = good)
- **LanguageTool** (`language_tool_python`, `es` locale) for Spanish grammar

**Tier 3 (offline batch):**
- BLASER 2.0 QE (Meta SONAR) for deep fluency analysis
- Spanish LM perplexity via BETO (`dccuchile/bert-base-spanish-wwm-cased`)

## Validation Pipeline

`validate_session.py` downloads YouTube segments and caches as JSON. **Engine must call `.load()` before `.transcribe()`.**

YouTube transcript API may return "subtitles disabled" — the pipeline downloads audio and re-transcribes with Whisper instead (more reliable, apples-to-apples comparison).

## Known Issues

- 3 flaky tests in `test_caption_monitor_utils.py` (`TestComputeWerCerExtra`) fail when run alongside `test_validate_session.py` due to jiwer mock cleanup ordering. Pass fine in isolation.

---

## Adapter Deployment Pipeline

Planned 6-phase deployment from training desktop to inference endpoints (full spec in `docs/deploy.md`):

1. **Version**: SHA-256 hash adapter weights, store as `adapters/{model}/cycle{N}_{date}_{hash}/`, update `adapters/manifest.json`
2. **Convert** (NVIDIA only): Merge LoRA into base model, requantize to GGUF Q4_K_M. Mac skips this — MLX loads adapters directly.
3. **Transfer**: `rsync` adapter files to endpoint staging directory (~260 MB total for all three adapters, <3s on LAN)
4. **Health check**: SSH → run `tools/health_check.py` on staging adapter. 5 test sentences with expected substrings + latency < 5s. If any check fails → abort, current active adapter untouched.
5. **Activate**: Atomic swap `staging/` → `active/`, move old `active/` → `previous/`. Signal reload: SIGUSR1 (Mac, ~2-3s hot-reload) or `systemctl restart` (NVIDIA, ~10-20s).
6. **Verify**: Re-run health check on now-active adapter. If fails → automatic rollback (swap `active/` ↔ `previous/`).

**Rollback**: `python tools/deploy_adapters.py --rollback --endpoints mac-dev`. Base model fallback always available if both slots are corrupted — load without `adapter_path=`.

## Active Learning Feedback Loop

Cross-tool data flow for the flag → correct → retrain cycle (Phases 7–8):

1. **Capture**: `dry_run_ab.py` saves per-chunk WAVs + diagnostics JSONL during live session (confidence scores, latency, low-confidence words)
2. **Extract**: `prepare_finetune_data.py` identifies low-confidence segments → `stark_data/corrections/review_queue_{date}.tsv`
3. **Correct**: Human reviews in Label Studio (or TSV editor), saves corrections to `stark_data/corrections/{session_id}.jsonl`
4. **Merge**: `merge_corrections.py` (planned) merges corrections into training dataset
5. **Retrain**: Transfer to WSL, retrain with corrected data mixed in (70/30 replay buffer to prevent forgetting)
6. **Deploy**: Transfer adapters back to Mac via the deployment pipeline above
7. **Composite quality score**: `0.45 * neural_qe + 0.35 * stt_confidence + 0.20 * marian_agreement` — used to prioritize segments for review

**Target**: 3–5 cycles. First cycle yields 20–40% relative WER reduction. Stop when improvement < 2% relative for 2 consecutive cycles.

## Data Integrity & Adapter Management

### Data Lockfile (`tools/lock_data.py`)

Records SHA-256 hashes of all training data files. Run before training to snapshot, before eval to verify no drift.

```bash
python tools/lock_data.py generate   # Create/update lockfile
python tools/lock_data.py verify     # Check files match lockfile
```

Output: `bible_data/data_lockfile.json`

### Sermon Sorting (`tools/sort_sermons.py`)

Classifies sermons by type (gospel/ministry/conference/throwback) and date from metadata JSONs. Produces `stt-data/manifest.json` and organizes WAVs into `stt-data/{type}/{year}/`.

```bash
python tools/sort_sermons.py --input stark_data/raw/midwest --output-dir stt-data --cutoff 2026-03-14
```

Training cutoff: 2026-03-14. Data before = train, on/after = eval.

### Tiered Glossary (`tools/glossary.py`)

Two-tier glossary for Deepgram keyterm boosting and training normalization:
- **Tier 1 (Boost):** 50 terms, <420 tokens, for Deepgram `keyterm` parameter
- **Tier 2 (Master):** 229 terms, full EN→ES theological glossary

```bash
python training/build_glossary.py --build-tiers  # Generates tier1_boost.json + tier2_master.json
```

### Adapter Management (`tools/manage_adapters.py`)

Lifecycle management with manifest at `adapters/manifest.json`:

```bash
python tools/manage_adapters.py register --adapter hybrid_runs/S8_deepl_only --model gemma_4b
python tools/manage_adapters.py activate --model gemma_4b --version S8_deepl_only  # runs health check first
python tools/manage_adapters.py rollback --model gemma_4b
python tools/manage_adapters.py list --model gemma_4b
```

### Health Check (`tools/health_check.py`)

Verifies adapter produces sane translations before deployment. 5 canary sentences covering theological terms (atonement, James/Santiago, propitiation, breaking of bread, resurrection).

```bash
python tools/health_check.py --adapter hybrid_runs/S8_deepl_only
```

Exit code 0 = pass, 1 = fail. Used automatically by `manage_adapters.py activate`.

### Evaluation Sets (`tools/build_eval_sets.py`)

Builds proper evaluation sets with stratification and provenance:
- 500 verse pairs stratified by genre (Pentateuch/History/Poetry/Prophets/Gospels/Epistles)
- Sermon eval chunks filtered to post-cutoff sermons only
- Registry at `bible_data/eval_registry.json`
- **Fresh eval set:** 4 post-cutoff sermons (Gospel+Teaching 3/22 and 3/29), 2,706 examples — used for W12+ Whisper ablation evaluation

```bash
python tools/build_eval_sets.py --dry-run   # Preview
python tools/build_eval_sets.py             # Build (modifies verse_pairs_train.jsonl)
```

### Gemma 4 Benchmark (`training/benchmark_gemma4.py`)

Compares TranslateGemma 4B/12B vs Gemma 4 E2B/E4B on EN→ES translation (HF NF4 only). Three evaluation tiers: Bible verse holdout (BLEU/chrF++/COMET), Deepgram sermon chunks (COMET-QE + hallucination ratio), 8 theological canary sentences (term accuracy).

```bash
python training/benchmark_gemma4.py --models tg4b,e2b --max-samples 50 --skip-comet
python training/benchmark_gemma4.py --models all  # Full 4-model comparison
```

> **Note (v2026.5):** the VRAM column in `metrics/gemma4_benchmark/comparison.json` uses `torch.cuda.max_memory_allocated()`, which undercounts Gemma 4 by ~2×. For accurate VRAM and a head-to-head with llama.cpp Q4_K_M, see Phase 1A below.

### Phase 1A llama.cpp Benchmark (`bench_translate_t1_t4.py`)

Extends the Gemma 4 comparison with three llama.cpp/GGUF configs (T2 E2B, T3 E4B, T4 E4B+E2B spec decode), continuous `nvidia-smi` VRAM sampling (drop-in replacement for the broken PyTorch counter), and server-side timing parser. Output JSON is shape-compatible with `gemma4_benchmark/comparison.json`. Gate 1A: best llama.cpp config tok/s ≥ 1.30× HF baseline AND canary ≥ 6/8.

```bash
# T1 (HF E2B baseline) and other HF configs
python bench_translate_t1_t4.py --config t1 --n-sermon 125 --out metrics/phase1a_t1.json
python bench_translate_t1_t4.py --config tg4b_hf --out metrics/phase1a_tg4b_hf.json    # add tg12b_hf, e4b_hf

# T2/T3/T4 — start llama-server first
./start_server.sh --model models/gemma-4-e2b-it-q4km.gguf --no-draft &
python bench_translate_t1_t4.py --config t2 --server-log /tmp/llama_t2.log --out metrics/phase1a_t2.json
pkill -f llama-server

# Merge full 7-config matrix
python bench_translate_t1_t4.py --config merge --inputs metrics/phase1a_t*.json metrics/phase1a_*_hf.json --out metrics/phase1a_benchmark.json
```

Result: T2 (E2B GGUF) 8.89× T1, T3 (E4B GGUF) 5.46× T1 with 7/8 canary. See `docs/archive/v2026.5/BENCHMARK.md`.

## Per-Tool Quick Reference

Invocation examples for tools not covered above:

```bash
# Batch translation processing
python tools/batch_translate.py input.txt --lang en --output out.tsv

# Export review queue from diagnostics
python tools/export_review.py metrics/diagnostics_*.jsonl --min-priority 3

# Convert models between MLX and CUDA formats
python tools/convert_models_to_both.py --model gemma-4b --format gguf

# Test adaptive model selection logic
python tools/test_adaptive_model.py --sentences 20

# Download test texts for roundtrip quality testing
python tools/download_roundtrip_texts.py --count 50
```
