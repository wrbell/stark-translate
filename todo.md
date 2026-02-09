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
- [x] Latency measurement system (STT/translation/E2E, p50/p95, tokens/sec)
- [x] Confidence scoring (avg_logprob → green/yellow/red dots)
- [x] Translation QE (length ratio + untranslated detection)
- [x] Error handling (mic reconnect, WebSocket cleanup, silence skip)
- [x] Profanity filter (removed "ass", "damn", "whore" — biblical terms)
- [x] VAD fixed — Silero streaming API (`model(tensor, sr).item()`)
- [x] WebSocket fixed — `difference_update()`, `ping_interval=None`
- [x] Two-pass STT — fast partials (1s) + quality finals (on silence)
- [x] MarianMT for partials (~80ms) + TranslateGemma for finals (~650ms)
- [x] Audience display (side-by-side EN/ES, Calibri, fading context, fullscreen)
- [x] A/B/C comparison display (Gemma 4B / MarianMT or 12B / Audience hybrid)
- [x] MacBook Pro mic only (no external mic selection)
- [x] Whisper theological prompt (sin, reign, bow, mediator, etc.)
- [x] Previous-text context passed to Whisper for cross-chunk accuracy
- [x] Default 4B only (`--ab` flag for both models)
- [x] Parallel A/B translation via `run_in_executor`
- [x] `generate` instead of `stream_generate` (eliminates callback overhead)
- [x] Dynamic `max_tokens` based on input word count
- [x] Automated diagnostics (homophones, bad splits, Marian divergence, durations)
- [x] Per-chunk audio WAV saving for fine-tuning (`stark_data/live_sessions/`)
- [x] Structured JSONL review queue with priority scoring
- [x] Word-level timestamps + per-word confidence logging
- [x] Segment metadata (compression_ratio, no_speech_prob, avg_logprob)
- [x] Hallucination detection (compression_ratio > 2.4)
- [x] Hardware profiling + per-chunk resource monitoring (CPU, RAM)
- [x] CTranslate2 int8 MarianMT (76MB, 3.3x faster than PyTorch, identical output)
- [x] Parallel PyTorch/CT2 MarianMT comparison in live testing (logged to CSV)
- [~] Prompt caching — deferred (prefix ~30-40 tokens, <50ms savings)
- [x] Mobile display — responsive phone/tablet view with model toggle + Spanish-only mode
- [x] Mobile display profanity filter fixed (allow "ass", "damn", "whore" — biblical terms)
- [x] Mobile display WebSocket auto-detect (connects via LAN, not just localhost)
- [x] Audience display QR code overlay (click header → QR for phone URL)
- [x] Audience display scroll history (all sentences kept, scrollable, auto-scroll with pause)
- [x] Fullscreen toggle button on audience + A/B/C displays (bottom-right, auto-hide)
- [x] HTTP static server in dry_run_ab.py (`--http-port 8080`) for phone/LAN access
- [x] WebSocket binds 0.0.0.0 (phones can connect over LAN)
- [x] All displays auto-detect WebSocket host (work over LAN, not just localhost)
- [x] Theological glossary expanded to 229 terms (66 books, 31 proper names, theological concepts, liturgical terms, sermon phrases)

---

## IN PROGRESS

### P0 — Live Testing (Requires Running Sessions)

- [ ] **Run full dry_run_ab.py live** — mic → STT → translate → browser display
  - Verify MarianMT partials + TranslateGemma finals working end-to-end
  - Verify A/B/C display renders correctly
  - Verify audio WAVs and diagnostics JSONL saving
  - Run 5+ min, 10+ utterances
  - Review CSV output + diagnostics summary

- [ ] **Tune VAD parameters** — need data from live sessions
  - VAD_THRESHOLD (currently 0.3) — too low catches breaths, too high drops words
  - SILENCE_TRIGGER (0.8s) — too short splits mid-sentence, too long delays output
  - MAX_UTTERANCE (8s) — find sweet spot for Whisper context vs lag

- [ ] **Expand Whisper prompt** — run 5-10 live sessions
  - Review homophone flags in diagnostics summary
  - Add every misrecognized theological word to WHISPER_PROMPT
  - This list becomes the fine-tuning evaluation checklist
  - Currently tracking: reign/rain, mediator/media, profit/prophet, alter/altar

- [ ] **Tune sentence boundary splitting** — review bad_split flags in CSV
  - Utterances ending with function words = mid-phrase cuts
  - Adjust SILENCE_TRIGGER and MAX_UTTERANCE to find natural pauses
  - Consider smarter boundary detection beyond simple silence gap

- [ ] **Evaluate MarianMT vs TranslateGemma divergence** — review marian_similarity
  - Identify systematic patterns where MarianMT misleads before Gemma corrects
  - If certain phrase types consistently diverge, assess UX impact

- [ ] **Calibrate QE thresholds** — need qe_a/qe_b data from 5+ sessions
  - Tune so flagged scores actually correlate with bad translations
  - Consider adding CometKiwi as Tier 1 QE when ready

- [ ] **Fix singing/hymns breaking VAD and STT**
  - Singing causes VAD confusion and STT garbage
  - Options: lower gain during music, inaSpeechSegmenter to detect & skip music,
    or RMS + tonal content heuristic

### P1 — Physical Testing

- [ ] **Test in actual church environment**
  - Mic gain calibration in the room (PA system, HVAC, congregation)
  - Projection display readability from pews
  - Multiple speakers switching — does silence gap handle handoffs?
  - Font size, fade timing, partial→final replacement feedback from users

- [ ] **Test audience display readability with real users**
  - Get feedback from non-English speakers
  - Is 5 sentences of fading context right?
  - How disorienting is italic→regular replacement?
  - Side-by-side EN/ES vs stacked layout?

### P2 — Display & UX

- [ ] **ProPresenter/PowerPoint integration research**
  - Investigate NDI output, text overlay APIs, or MIDI control
  - Alternative: OBS overlay via browser source

### P3 — Data Collection & Prep

- [ ] **Download sermon audio** — yt-dlp from Stark Road YouTube *(Mac or Windows)*
  - Target: 20-50 hours
  - Prefer soundboard recordings over room mics
  - Store in stark_data/raw/

- [ ] **Run audio preprocessing pipeline** *(Windows — needs demucs GPU)*
  - preprocess_audio.py (10-step pipeline)
  - Output to stark_data/cleaned/

- [ ] **Pseudo-label with Whisper large-v3** *(Windows — needs CUDA)*
  - transcribe_church.py
  - Output JSON transcripts to stark_data/transcripts/

- [ ] **Download Bible parallel corpus** *(Mac or Windows)*
  - prepare_bible_corpus.py
  - ~155K verse pairs from scrollmapper/BibleNLP
  - Output to bible_data/aligned/

- [ ] **Quality assessment baseline** *(Windows)*
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
  - Theological term spot-check against 229-term glossary

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

- [ ] **Speaker diarization** — tag which speaker is talking
  - pyannote-audio with min_speakers=2, max_speakers=2
  - Store speaker labels in diagnostics JSONL
  - Display speaker name alongside translation

- [ ] **Speaker identification** — who is speaking on a panel of 8
  - Speaker enrollment: record 30s samples → embeddings
  - Real-time speaker ID: compare live embedding vs enrolled
  - Support 2-8 speakers in panel discussions
  - Persistent profiles in `stark_data/speakers/`

- [ ] **Post-sermon 5-sentence summary**
  - Full transcript → Gemma 3 4B → structured summary
  - Per-speaker summary (requires diarization)

- [ ] **Verse reference extraction**
  - Regex + LLM for explicit/implicit verse citations
  - Per-speaker verse list with timestamps

- [ ] **Multi-language support**
  - TranslateGemma supports 36 languages
  - French, Portuguese, Chinese for broader outreach

- [ ] **Hardware portability** — port to church PCs (Windows/Intel)
  - MLX → CTranslate2/ONNX for non-Apple hardware
  - Models and fine-tuning are portable, runtime changes
  - Use hardware_*.json + per-chunk resource logs to spec requirements
  - 4B fits in 2.5GB RAM (most PCs), 12B needs 16GB+
  - MarianMT (298MB) runs anywhere

### P7 — Latency Reduction Roadmap

- [x] **Parallel A/B translation** — `run_in_executor` for simultaneous inference
- [x] **Shorter chunk duration** — 2s chunks + two-pass partial/final architecture
- [x] **VAD-triggered instant processing** — process on silence gap (0.8s)
- [x] **MarianMT for fast partials** — ~80ms vs ~650ms TranslateGemma

- [ ] **Speculative decoding** — use 4B as draft model for 12B
  - `mlx_lm.generate` supports `draft_model` parameter
  - Both models already loaded in A/B mode — zero additional memory
  - Expected 1.5-3x speedup on 12B (1400ms → 600-900ms)
  - Zero quality degradation (rejected drafts replaced by target model tokens)
  - Tune `num_draft_tokens` (4-16, start at 8)
  - **Implementation:** modify `translate_mlx()` to pass `draft_model=mlx_a_model` when translating with 12B

- [ ] **Pre-warm models during silence** — dummy forward pass to keep Metal GPU warm
  - Fire a 1-token forward pass when silence detected (>2s gap)
  - Prevents ~50-150ms cold-start latency on first token after idle
  - Add to `audio_loop()` during silence gap detection
  - Low complexity, high impact for perceived responsiveness

- [~] **KV cache quantization** — `kv_bits=4` — deferred
  - Not beneficial for short translations (10-50 tokens)
  - Would help for long-context features (summarization, 3K+ tokens)
  - Revisit when post-sermon summary feature is implemented

- [ ] **Smaller/faster translation model exploration**
  - Test Qwen2.5-3B, gemma-2-2b as ultra-light translators
  - Benchmark quality vs TranslateGemma on theological test set

- [ ] **Whisper model optimization**
  - Test whisper-tiny.en (~50ms) vs whisper-small.en as speed/accuracy trade-offs
  - Streaming Whisper: process audio in overlapping windows

---

## Architecture Reference

```
Mac Inference (MLX) — Two-Pass Architecture:
  Mic → Silero VAD → [Partial: mlx-whisper STT + MarianMT (~580ms, italic)]
                    → [Final: mlx-whisper STT + TranslateGemma 4B (~1.3s, regular)]
                    → WebSocket → Browser (A/B/C display + audience display + mobile)

  Serving:
    → WebSocket on 0.0.0.0:8765 (all displays connect here)
    → HTTP server on 0.0.0.0:8080 (serves HTML to phones over LAN)

  Per-chunk data saved:
    → Audio WAV (stark_data/live_sessions/)
    → Diagnostics JSONL (metrics/) with review priority scoring
    → CSV metrics (metrics/)
    → Hardware profile (metrics/)

Windows Training (CUDA):
  Raw audio → 10-step preprocessing → Whisper large-v3 pseudo-labels → LoRA fine-tune
  Bible corpus → aligned JSONL → QLoRA fine-tune TranslateGemma
```

| Component | Model | Framework | Memory |
|-----------|-------|-----------|--------|
| VAD | Silero VAD | PyTorch | ~2 MB |
| STT | distil-whisper-large-v3 | mlx-whisper | ~1.5 GB |
| Translate (fast) | MarianMT opus-mt-en-es | PyTorch fp32 | ~300 MB |
| Translate (fast) | MarianMT opus-mt-en-es | CT2 int8 | ~76 MB |
| Translate A | TranslateGemma 4B 4-bit | mlx-lm | ~2.5 GB |
| Translate B | TranslateGemma 12B 4-bit | mlx-lm | ~7 GB |
| **Total (4B only)** | | | **~4.3 GB / 18 GB** |
| **Total (A/B)** | | | **~11.3 GB / 18 GB** |
