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
