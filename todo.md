# TODO — Stark Road Bilingual Speech-to-Text

> **Machine:** M3 Pro MacBook, 18GB unified, MLX + PyTorch
> **Remote:** Windows Desktop (WSL2, A2000 Ada 16GB) for training

---

## DONE

- [x] Environment setup (Python 3.11, venv, brew deps)
- [x] All Windows pipeline scripts written (10 files)
- [x] Directory structure with .gitkeep
- [x] Git repo + GitHub remote
- [x] Hardware profiled (18GB, 18-core GPU, Metal 4)
- [x] Docs updated (CLAUDE.md, CLAUDE-macbook.md)
- [x] README written
- [x] HuggingFace auth configured
- [x] PyTorch models downloaded (Whisper, Gemma 4B, 12B)
- [x] MLX installed (mlx 0.30.6, mlx-lm 0.30.6, mlx-whisper 0.4.3)
- [x] MLX models downloaded (4B-4bit: 2.2GB, 12B-4bit: 6.6GB, distil-whisper)
- [x] TranslateGemma EOS fix (add `<end_of_turn>` to `_eos_token_ids`)
- [x] Verified: both models fit in 9GB, 4B ~650ms, 12B ~1.4s
- [x] dry_run_ab.py rewritten for MLX
- [x] setup_models.py rewritten for MLX
- [x] ab_display.html created

---

## IN PROGRESS

### P0 — End-to-End Live Demo (Today)

- [ ] **Run full dry_run_ab.py live** — mic → STT → translate → browser display
  - Verify mlx-whisper works with real mic audio
  - Verify WebSocket → browser connection
  - Verify both 4B and 12B running simultaneously
  - Run 5+ min, 10+ utterances
  - Review CSV output

- [ ] **Latency measurement system** — proper timing breakdown
  - STT latency: time from audio chunk ready → text output
  - Translation latency: time from English text → Spanish text (per model)
  - End-to-end latency: audio chunk ready → WebSocket broadcast
  - Display update latency: WebSocket send → browser render
  - Log all timings to CSV and display in browser footer
  - Add token/sec reporting from MLX generate
  - Running p50/p95 percentile display in browser footer (per component)

### P1 — Mac Pipeline Hardening

- [ ] **Confidence scoring** — extract Whisper segment-level quality
  - `avg_logprob`, `no_speech_prob`, `compression_ratio` from mlx-whisper
  - Word-level probabilities via `word_timestamps=True`
  - Flag segments below threshold for review
  - Log to confidence_flags.jsonl

- [ ] **Translation quality estimation (QE)** — reference-free
  - CometKiwi: source + translation → score (0-1)
  - LaBSE cross-lingual cosine similarity
  - Length ratio check (Spanish 15-25% longer than English)
  - Back-translation via MarianMT → BERTScore
  - Log to translation_qe.jsonl

- [ ] **Error handling** — graceful recovery
  - Mic disconnect/reconnect
  - Model OOM → unload/reload
  - WebSocket client drop/reconnect
  - Empty/silence audio → skip gracefully

- [ ] **Streaming translation** — use mlx-lm's `stream_generate()`
  - Show partial translations as tokens arrive
  - Lower perceived latency
  - Update browser display incrementally

- [ ] **Prompt caching** — reuse TranslateGemma prompt prefix
  - The chat template prefix is constant (~90 tokens)
  - Cache with `mlx_lm.batch_generate()` prompt_caches
  - Could cut translation latency by ~30-40%

### P2 — Display & UX

- [ ] **Projected display mode** — church-optimized layout
  - Spanish top (large), English bottom (smaller)
  - Black background, high contrast
  - Configurable font sizes
  - ProPresenter/PowerPoint integration research

- [ ] **Mobile display** — phone/tablet view
  - Responsive layout for phone viewing
  - QR code to connect
  - Spanish-only mode option

- [ ] **Display polish**
  - Smooth text transitions (CSS animations)
  - Show confidence indicator (green/yellow/red per segment)
  - Show translation speed comparison live
  - Fullscreen toggle button (not just keyboard)
  - Chunk history / scroll back

### P3 — Data Collection & Prep (Windows)

- [ ] **Download sermon audio** — yt-dlp from Stark Road YouTube
  - Target: 20-50 hours
  - Prefer soundboard recordings over room mics
  - Store in stark_data/raw/

- [ ] **Run audio preprocessing pipeline**
  - preprocess_audio.py (10-step pipeline)
  - Output to stark_data/cleaned/

- [ ] **Pseudo-label with Whisper large-v3**
  - transcribe_church.py
  - Output JSON transcripts to stark_data/transcripts/

- [ ] **Download Bible parallel corpus**
  - prepare_bible_corpus.py
  - ~155K verse pairs from scrollmapper/BibleNLP
  - Output to bible_data/aligned/

- [ ] **Build theological glossary**
  - build_glossary.py
  - ~200-500 terms with soft constraints
  - Output to bible_data/glossary/

- [ ] **Quality assessment baseline**
  - assess_quality.py
  - Sample 50-100 segments
  - Manual transcription for WER baseline

### P4 — Fine-Tuning (Windows)

- [ ] **Whisper LoRA fine-tuning**
  - train_whisper.py (r=32, α=64, encoder+decoder)
  - 20-50 hrs church audio
  - 3-5 epochs on A2000 Ada

- [ ] **TranslateGemma QLoRA fine-tuning**
  - train_gemma.py (r=16, 4-bit NF4)
  - ~155K Bible verse pairs
  - 3 epochs, packing enabled

- [ ] **MarianMT fallback fine-tune**
  - train_marian.py
  - Full fine-tune (298MB model)
  - Quick iteration baseline

- [ ] **Evaluation**
  - evaluate_translation.py
  - SacreBLEU, chrF++, COMET
  - Per-genre breakdown (narrative, poetry, epistles, etc.)
  - Theological term spot-check

- [ ] **LoRA adapter transfer**
  - Export from Windows → Mac
  - Test with MLX (mlx_lm supports adapter_path=)
  - A/B test: base vs fine-tuned

### P5 — Monitoring & Feedback Loop

- [ ] **YouTube caption comparison**
  - live_caption_monitor.py
  - Compare local Whisper vs YouTube auto-captions
  - Windowed WER tracking

- [ ] **Active learning pipeline**
  - Flag low-confidence segments → human review queue
  - Label Studio / Prodigy integration
  - Versioned corrections in stark_data/corrections/
  - Re-train loop (3-5 cycles target)

### P6 — Future Features

- [ ] **Post-sermon 5-sentence summary**
  - Full transcript → Gemma 3 4B → structured summary
  - Per-speaker summary (requires diarization)

- [ ] **Verse reference extraction**
  - Regex + LLM for explicit/implicit verse citations
  - Per-speaker verse list with timestamps

- [ ] **Multi-language support**
  - TranslateGemma supports 36 languages
  - French, Portuguese, Chinese for broader outreach

---

## Architecture Reference

```
Mac Inference (MLX):
  Mic → Silero VAD (PyTorch) → mlx-whisper STT → TranslateGemma 4B+12B (MLX 4-bit) → WebSocket → Browser

Windows Training (CUDA):
  Raw audio → 10-step preprocessing → Whisper large-v3 pseudo-labels → LoRA fine-tune
  Bible corpus → aligned JSONL → QLoRA fine-tune TranslateGemma
```

| Component | Model | Framework | Memory |
|-----------|-------|-----------|--------|
| VAD | Silero VAD | PyTorch | ~2 MB |
| STT | distil-whisper-large-v3 | mlx-whisper | ~1.5 GB |
| Translate A | TranslateGemma 4B 4-bit | mlx-lm | ~2.5 GB |
| Translate B | TranslateGemma 12B 4-bit | mlx-lm | ~7 GB |
| **Total** | | | **~11 GB / 18 GB** |
