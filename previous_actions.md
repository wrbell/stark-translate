# Previous Actions — Stark Road Bilingual Speech-to-Text

> Completed work moved here from todo.md for reference.

---

## Environment & Setup

- [x] Environment setup (Python 3.11, venv, brew deps)
- [x] All Windows pipeline scripts written (10 files)
- [x] Directory structure with .gitkeep
- [x] Git repo + GitHub remote
- [x] Hardware profiled (18GB, 18-core GPU, Metal 4)
- [x] Docs updated (CLAUDE.md, CLAUDE-macbook.md)
- [x] README written
- [x] HuggingFace auth configured

## Models & Inference

- [x] PyTorch models downloaded (Whisper, Gemma 4B, 12B)
- [x] MLX installed (mlx 0.30.6, mlx-lm 0.30.6, mlx-whisper 0.4.3)
- [x] MLX models downloaded (4B-4bit: 2.2GB, 12B-4bit: 6.6GB, distil-whisper)
- [x] TranslateGemma EOS fix (add `<end_of_turn>` to `_eos_token_ids`)
- [x] Verified: both models fit in 9GB, 4B ~650ms, 12B ~1.4s
- [x] dry_run_ab.py rewritten for MLX
- [x] setup_models.py rewritten for MLX
- [x] CTranslate2 int8 MarianMT (76MB, 3.3x faster than PyTorch, identical output)

## Pipeline Features

- [x] Two-pass STT — fast partials (1s) + quality finals (on silence)
- [x] MarianMT for partials (~80ms) + TranslateGemma for finals (~650ms)
- [x] Parallel A/B translation via `run_in_executor`
- [x] `generate` instead of `stream_generate` (eliminates callback overhead)
- [x] Dynamic `max_tokens` based on input word count
- [x] Default 4B only (`--ab` flag for both models)
- [x] Whisper theological prompt (sin, reign, bow, mediator, etc.)
- [x] Previous-text context passed to Whisper for cross-chunk accuracy
- [x] Speculative decoding wired — 4B drafts tokens for 12B (`--num-draft-tokens`)
- [x] HTTP static server in dry_run_ab.py (`--http-port 8080`) for phone/LAN access
- [x] WebSocket binds 0.0.0.0 (phones can connect over LAN)

## Quality & Diagnostics

- [x] Latency measurement system (STT/translation/E2E, p50/p95, tokens/sec)
- [x] Confidence scoring (avg_logprob → green/yellow/red dots)
- [x] Translation QE (length ratio + untranslated detection)
- [x] Automated diagnostics (homophones, bad splits, Marian divergence, durations)
- [x] Per-chunk audio WAV saving for fine-tuning (`stark_data/live_sessions/`)
- [x] Structured JSONL review queue with priority scoring
- [x] Word-level timestamps + per-word confidence logging
- [x] Segment metadata (compression_ratio, no_speech_prob, avg_logprob)
- [x] Hallucination detection (compression_ratio > 2.4)
- [x] Hardware profiling + per-chunk resource monitoring (CPU, RAM)
- [x] Parallel PyTorch/CT2 MarianMT comparison in live testing (logged to CSV)

## Error Handling & Stability

- [x] Error handling (mic reconnect, WebSocket cleanup, silence skip)
- [x] Profanity filter (removed "ass", "damn", "whore" — biblical terms)
- [x] VAD fixed — Silero streaming API (`model(tensor, sr).item()`)
- [x] WebSocket fixed — `difference_update()`, `ping_interval=None`
- [x] MacBook Pro mic only (no external mic selection)

## Display & UX

- [x] ab_display.html created (A/B/C comparison, Gemma 4B / MarianMT or 12B / Audience hybrid)
- [x] Audience display (side-by-side EN/ES, Calibri, fading context, fullscreen)
- [x] Mobile display — responsive phone/tablet view with model toggle + Spanish-only mode
- [x] Mobile display profanity filter fixed (allow biblical terms)
- [x] Mobile display WebSocket auto-detect (connects via LAN, not just localhost)
- [x] Audience display QR code overlay (click header → QR for phone URL)
- [x] Audience display scroll history (all sentences kept, scrollable, auto-scroll with pause)
- [x] Fullscreen toggle button on audience + A/B/C displays (bottom-right, auto-hide)
- [x] All displays auto-detect WebSocket host (work over LAN, not just localhost)

## Data & Glossary

- [x] Theological glossary expanded to 229 terms (66 books, 31 proper names, theological concepts, liturgical terms, sermon phrases)

## Latency (P7 — completed items)

- [x] Parallel A/B translation — `run_in_executor` for simultaneous inference
- [x] Shorter chunk duration — 2s chunks + two-pass partial/final architecture
- [x] VAD-triggered instant processing — process on silence gap (0.8s)
- [x] MarianMT for fast partials — ~80ms vs ~650ms TranslateGemma

## P2 — Display & UX

- [x] OBS overlay (`obs_overlay.html`) — transparent lower-third for OBS Browser Source, URL params for model/english/lines
- [x] Projection integration research (`docs/projection_integration.md`) — OBS, NDI, ProPresenter, PowerPoint comparison
- [x] Church display fixes — profanity filter, WS auto-detect, fullscreen button

## P5 — Monitoring

- [x] `live_caption_monitor.py` — YouTube caption comparison (post-stream, live, trend report modes)

## P6 — Future Features (scripts built)

- [x] `diarize.py` — pyannote-audio 3.1 speaker diarization with optional enrollment-based speaker ID
- [x] `summarize_sermon.py` — 5-sentence structured summary via MLX LLM, English + Spanish output
- [x] `extract_verses.py` — Bible verse extraction (66 book variants, spoken numbers, context tracking)

## P7 — Latency (Phase 1 + Phase 2 implemented)

- [x] 1A: Disable `word_timestamps` for partials (100-200ms savings)
- [x] 4A: Pre-warm models during silence (50-150ms savings)
- [x] 2C: Reduce `max_tokens` multiplier 2.5→1.8 (20-50ms savings)
- [x] 1E: Trim Whisper prompt to ~40 words, cap prev_text at 100 chars (20-30ms savings)
- [x] 5D: Move I/O (WAV/JSONL/CSV) to background ThreadPoolExecutor (10-30ms savings)
- [x] 5A: Replace `scipy.signal.resample` with `decimate` (5-10ms savings)
- [x] 2B: Prompt caching via `make_prompt_cache()` for fixed chat template prefix (50-80ms savings)
- [x] 4B: Increase `mx.set_cache_limit` from 100MB to 256MB (20-40ms savings)
- [x] 2E: EOS verification — confirmed `<end_of_turn>` stops generation early (30-50ms savings)
- [x] 3B: Pre-tokenize translation suffix tokens at startup (30-50ms savings)
- [x] 5B: Move VAD to dedicated thread with queue (5-10ms jitter reduction)
- [x] 1C: lightning-whisper-mlx researched — 1.8x SLOWER, not viable (see `docs/fast_stt_options.md`)
- [x] STT benchmark tool (`stt_benchmark.py`) — mlx-whisper vs alternatives, profiling mode

## Windows Training Pipeline (reviewed/fixed)

- [x] All training scripts updated: `--resume` support, newer transformers API (`processing_class=`)
- [x] `transcribe_church.py` rewritten with dual backend (transformers + faster-whisper)
- [x] `train_marian.py` fixed: Seq2SeqTrainer, DataCollatorForSeq2Seq, eval during training
- [x] `evaluate_translation.py` — added MarianMT evaluation via `--marian` flag
- [x] `requirements-windows.txt` updated with all CUDA training deps
- [x] `requirements-mac.txt` cleaned up (removed bitsandbytes, added ctranslate2 + jiwer)

## YouTube / Data

- [x] Playlist researched: 275 videos, 249.8 hours, March 2020–Feb 2026, two-speaker format
- [x] `download_sermons.py` updated with default playlist URL, metadata extraction, ETA tracking

## Documentation

- [x] CLAUDE.md updated for MLX architecture, two-pass pipeline, all new files
- [x] CLAUDE-macbook.md updated for MLX, browser displays, HTTP server, latency optimizations

## Deferred

- [~] KV cache quantization (`kv_bits=4`) — not beneficial for short translations (10-50 tokens), revisit for summarization
- [~] lightning-whisper-mlx — stagnant since Apr 2024, 1.8x slower than mlx-whisper, missing key features
- [~] WhisperKit — Swift-only, wrong architecture for numpy-array pipeline
