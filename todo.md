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

- [x] **Latency measurement system** — proper timing breakdown
  - STT latency: time from audio chunk ready → text output
  - Translation latency: time from English text → Spanish text (per model)
  - End-to-end latency: audio chunk ready → WebSocket broadcast
  - Log all timings to CSV and display in browser footer
  - Token/sec reporting from MLX stream_generate
  - Running p50/p95 percentile display in browser footer (per component)

### P1 — Mac Pipeline Hardening ✓

- [x] **Confidence scoring** — extract Whisper segment-level quality
  - `avg_logprob` from mlx-whisper segments → 0-1 confidence score
  - Displayed as green/yellow/red dot per chunk in browser
  - Logged to CSV per chunk

- [x] **Translation quality estimation (QE)** — reference-free
  - Real-time: length ratio + untranslated detection (in dry_run_ab.py)
  - Offline: translation_qe.py with 3 tiers (lightweight, MarianMT back-translation, LaBSE)
  - QE scores logged to CSV per chunk (qe_a, qe_b)

- [x] **Error handling** — graceful recovery
  - Mic disconnect/reconnect (auto-retry with 2s backoff)
  - WebSocket client drop (dead client cleanup in broadcast)
  - Empty/silence audio → skip gracefully
  - process_chunk wrapped in try/except

- [x] **Streaming translation** — using mlx-lm's `stream_generate()`
  - Token-by-token generation with generation_tps metrics
  - Average tokens/sec displayed in browser footer
  - Foundation ready for real-time partial display (future)

- [~] **Prompt caching** — deferred (prefix is only ~30-40 tokens, <50ms savings)
  - The variable text is embedded in the JSON, so true prefix is small
  - Not worth the complexity vs ~650ms total translation time

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

- [x] **Profanity word filter** — client-side regex filter for church display
  - Blocks common profanity (ass, fuck, shit, etc.) — "hell" is allowed
  - Replaces with asterisks of same length
  - Applied to both English and Spanish panels

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

- [ ] **Speaker identification** — who is speaking on a panel of 8
  - `pyannote-audio` speaker diarization (segment → speaker cluster)
  - Speaker enrollment: record 30s samples per person → speaker embeddings
  - Real-time speaker ID: compare live embedding vs enrolled speakers
  - Display speaker name alongside translation (e.g., "Pastor Jim: ...")
  - Support 2-8 speakers in panel discussions
  - Persistent speaker profiles stored in `stark_data/speakers/`
  - Could use `speechbrain` or `resemblyzer` for lightweight embeddings on Mac

### P7 — Latency Reduction Roadmap

- [ ] **Parallel A/B translation** — run 4B and 12B simultaneously
  - Currently sequential (~650ms + ~1400ms = ~2050ms)
  - Use `asyncio.gather()` with thread pool or multiprocessing
  - MLX supports concurrent model inference on Apple Silicon
  - Target: E2E ~1500ms (limited by slower model)

- [ ] **Shorter chunk duration** — reduce from 3s to 1.5-2s
  - Faster time-to-first-output for live speech
  - Trade-off: shorter context → slightly lower STT accuracy
  - Test with `--chunk-duration 1.5`

- [ ] **VAD-triggered instant processing** — process on silence gap, not timer
  - Current: accumulate 3s of speech, then process
  - Better: start processing as soon as speaker pauses (0.3-0.5s silence)
  - Already partially implemented (silence_frames threshold)
  - Tune max_silence_frames lower for faster trigger

- [ ] **Speculative decoding** — use 4B as draft model for 12B
  - mlx_lm supports `draft_model` parameter in stream_generate()
  - 4B generates candidate tokens, 12B verifies in batch
  - Could speed up 12B by 2-3x with ~90% acceptance rate
  - Target: 12B latency from ~1400ms → ~600-800ms

- [ ] **Smaller/faster translation model exploration**
  - Test `mlx-community/Qwen2.5-3B-Instruct-4bit` with translation prompt
  - Test `mlx-community/gemma-2-2b-it-4bit` as ultra-light translator
  - MarianMT (`Helsinki-NLP/opus-mt-en-es`, 298MB) as fallback — ~50ms/translation
  - Benchmark quality vs TranslateGemma on theological test set

- [ ] **KV cache quantization** — reduce memory, allow larger batches
  - `kv_bits=4` in stream_generate() — ~75% cache memory reduction
  - Enables larger prefill_step_size for faster prompt processing
  - May allow fitting additional models in memory

- [ ] **Whisper model optimization**
  - Test `mlx-community/whisper-tiny.en` for speed (~50ms) vs accuracy trade-off
  - Test `mlx-community/whisper-small.en` as middle ground
  - Streaming Whisper: process audio in 1s overlapping windows
  - Word-level timestamps: get partial results before full chunk completes

- [ ] **WebSocket batching** — reduce browser update overhead
  - Batch multiple small updates into single WebSocket frame
  - Use requestAnimationFrame on client for smooth rendering
  - Reduce DOM operations per update

- [ ] **Pre-warm models between chunks** — keep GPU hot
  - Run a dummy forward pass during silence to prevent GPU clock throttling
  - MLX lazy evaluation: ensure graph is compiled before speech arrives

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
