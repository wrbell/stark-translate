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

## Deferred

- [~] Prompt caching — deferred (prefix ~30-40 tokens, <50ms savings) → now reconsidered in P7
- [~] KV cache quantization (`kv_bits=4`) — not beneficial for short translations (10-50 tokens), revisit for summarization
