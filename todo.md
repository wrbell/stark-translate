# TODO — Stark Road Bilingual Speech-to-Text

> **Mac:** M3 Pro, 18GB unified, MLX + PyTorch
> **Windows:** WSL2, A2000 Ada 16GB, CUDA — for training
> **Completed work:** see [previous_actions.md](previous_actions.md)

---

## P0 — Live Testing (Requires Running Sessions)

- [ ] **Run full dry_run_ab.py live** — 5+ min, 10+ utterances
  - Verify MarianMT partials + TranslateGemma finals end-to-end
  - Verify A/B/C display renders correctly
  - Verify audio WAVs and diagnostics JSONL saving
  - Review CSV output + diagnostics summary

- [ ] **Tune VAD parameters** — need data from live sessions
  - VAD_THRESHOLD (0.3) — too low catches breaths, too high drops words
  - SILENCE_TRIGGER (0.8s) — too short splits mid-sentence, too long delays output
  - MAX_UTTERANCE (8s) — find sweet spot for Whisper context vs lag

- [ ] **Expand Whisper prompt** — run 5-10 live sessions
  - Review homophone flags in diagnostics
  - Add every misrecognized theological word to WHISPER_PROMPT
  - Tracking: reign/rain, mediator/media, profit/prophet, alter/altar

- [ ] **Tune sentence boundary splitting** — review bad_split flags in CSV

- [ ] **Evaluate MarianMT vs TranslateGemma divergence** — review marian_similarity

- [ ] **Calibrate QE thresholds** — need qe_a/qe_b data from 5+ sessions

- [ ] **Fix singing/hymns breaking VAD and STT**
  - Options: lower gain during music, inaSpeechSegmenter, RMS heuristic

---

## P1 — Physical Testing

- [ ] **Test in actual church environment**
  - Mic gain, projection readability, speaker handoffs, font/fade feedback

- [ ] **Test audience display readability with real users**
  - Feedback from non-English speakers on fading context, layout, partials

---

## P2 — Display & UX

- [ ] **ProPresenter/PowerPoint integration research**
  - NDI output, text overlay APIs, OBS browser source overlay

---

## P3 — Data Collection & Prep

- [ ] **Download sermon audio** — yt-dlp from Stark Road YouTube *(Mac or Windows)*
  - Target: 20-50 hours, prefer soundboard recordings

- [ ] **Download Bible parallel corpus** *(Mac or Windows)*
  - Clone scrollmapper, run `prepare_bible_corpus.py`
  - ~155K verse pairs → `bible_data/aligned/`

- [ ] **Run audio preprocessing pipeline** *(Windows — needs demucs GPU)*

- [ ] **Pseudo-label with Whisper large-v3** *(Windows — needs CUDA)*

- [ ] **Quality assessment baseline** *(Windows)*
  - Sample 50-100 segments, manual transcription for WER baseline

---

## P4 — Fine-Tuning (Windows)

- [ ] **Whisper LoRA** — train_whisper.py (r=32, α=64, encoder+decoder, 3-5 epochs)
- [ ] **TranslateGemma QLoRA** — train_gemma.py (r=16, 4-bit NF4, ~155K verse pairs, 3 epochs)
- [ ] **MarianMT fallback fine-tune** — train_marian.py (full fine-tune, 298MB)
- [ ] **Evaluation** — SacreBLEU, chrF++, COMET, per-genre, theological term spot-check
- [ ] **LoRA adapter transfer** — Windows → Mac, test with MLX `adapter_path=`

---

## P5 — Monitoring & Feedback Loop

- [ ] **YouTube caption comparison** — live_caption_monitor.py, windowed WER tracking
- [ ] **Active learning pipeline** — Label Studio, versioned corrections, 3-5 retrain cycles

---

## P6 — Future Features

- [ ] **Speaker diarization** — pyannote-audio, 2 speakers, per-speaker labels in display
- [ ] **Speaker identification** — enrollment embeddings, 2-8 speaker panel support
- [ ] **Post-sermon 5-sentence summary** — Gemma 3 4B, per-speaker
- [ ] **Verse reference extraction** — regex + LLM, per-speaker verse list with timestamps
- [ ] **Multi-language support** — TranslateGemma supports 36 languages
- [ ] **Hardware portability** — RTX 2070 endpoints (see `docs/rtx2070_feasibility.md`)

---

## P7 — Sub-1s Latency Roadmap

**Target:** 4B path under 1s (currently ~1.15s), 12B path under 1.5s (currently ~1.9s)

```
Current budget:
  4B:   STT ~500ms + Translate ~650ms = ~1150ms
  12B:  STT ~500ms + Translate ~1400ms = ~1900ms
```

### Phase 1 — Quick wins (~3 hrs, 4B → ~850ms)

| # | Task | Savings | Effort |
|---|------|---------|--------|
| 1A | Disable `word_timestamps` for partials | 100-200ms | 30 min |
| 4A | Pre-warm models during silence (1-token forward pass) | 50-150ms | 1 hr |
| 2C | Reduce `max_tokens` multiplier (2.5 → 1.8) | 20-50ms | 5 min |
| 1E | Trim Whisper prompt to ~40 words, cap prev_text at 100 chars | 20-30ms | 15 min |
| 5D | Move I/O (WAV/JSONL/CSV) to background threads | 10-30ms | 30 min |
| 5A | Replace `scipy.signal.resample` with `decimate` or record at 16kHz | 5-10ms | 15 min |

### Phase 2 — Medium effort (~6 hrs, 4B → ~650ms, 12B → ~1100ms)

| # | Task | Savings | Effort |
|---|------|---------|--------|
| 2A | Benchmark speculative decoding, tune `num_draft_tokens` 3-16 | 400-700ms (12B) | 2 hrs |
| 2B | Prompt caching via `make_prompt_cache()` for fixed chat template prefix | 50-80ms | 1 hr |
| 1C | Benchmark `lightning-whisper-mlx` (claims 4x faster) | 100-200ms | 2 hrs |
| 4B | Increase `mx.set_cache_limit` from 100MB to 256MB | 20-40ms | 5 min |

### Phase 3 — Architectural (~16 hrs, 4B → ~500ms, 12B → ~800ms)

| # | Task | Savings | Effort |
|---|------|---------|--------|
| 6A | Streaming translation display (tokens → browser as generated) | 300-500ms perceived | 4 hrs |
| 6C | Pipeline N/N-1 overlap (translate chunk N-1 while STT runs on N) | 300-500ms actual | 8 hrs |
| 6B | Adaptive model selection (MarianMT for short, Gemma for complex) | 200-400ms on simple | 4 hrs |

### Phase 4 — Advanced (~44 hrs, 4B → ~300ms, 12B → ~600ms)

| # | Task | Savings | Effort |
|---|------|---------|--------|
| 1D | Lightning-SimulWhisper (MLX+CoreML streaming, AlignAtt) | 200-300ms | 16 hrs |
| 1F | WhisperKit CoreML native (Swift bridge needed) | 200-300ms | 24 hrs |
| 4D | 4-bit quantized Whisper (`whisper-large-v3-mlx-4bit`) | 50-100ms | 2 hrs |

```
Projected:        4B path    12B path
Phase 1:          ~850ms     ~1700ms
Phase 2:          ~650ms     ~1100ms
Phase 3:          ~500ms     ~800ms
Phase 4:          ~300ms     ~600ms
```

### Other P7 items

- [ ] **1B. whisper-large-v3-turbo for partials** — ~50-100ms, benchmark WER vs distil first
- [ ] **2D. Smaller translation models** — gemma-3-1b-it, NLLB-600M, fine-tuned MarianMT
- [ ] **2E. Early stopping / stream_generate** — ~30-50ms, verify EOS already stops `generate()`
- [ ] **3A. Overlap STT and translation** — partial already does this; full overlap via 3B/6C
- [ ] **3B. Pre-compute translation prompt tokens during STT** — ~30-50ms
- [ ] **4E. Increase prefill chunk size** — ~10-20ms
- [ ] **5B. Move VAD to separate thread** — ~5-10ms jitter reduction
- [ ] **5C. Use `asyncio.to_thread`** — ~1-2ms micro-optimization
- [ ] **6D. Batch multiple short utterances** — ~100-200ms amortized

### Research Sources

- [mlx-lm: speculative decoding, prompt caching, stream_generate](https://github.com/ml-explore/mlx-lm)
- [Apple ReDrafter: 1.37-2.3x speedup on Apple Silicon](https://machinelearning.apple.com/research/recurrent-drafter)
- [Lightning-SimulWhisper: MLX/CoreML streaming](https://github.com/altalt-org/Lightning-SimulWhisper)
- [lightning-whisper-mlx: 10x faster Whisper](https://github.com/mustafaaljadery/lightning-whisper-mlx)
- [WhisperKit: 0.45s streaming latency](https://github.com/argmaxinc/WhisperKit)
- [mlx-whisper word_timestamps overhead](https://github.com/ml-explore/mlx-examples/issues/1254)

---

## Architecture Reference

```
Mac Inference (MLX) — Two-Pass Architecture:
  Mic → Silero VAD → [Partial: mlx-whisper + MarianMT (~580ms, italic)]
                    → [Final: mlx-whisper + TranslateGemma 4B (~1.3s, regular)]
                    → WebSocket → Browser (A/B/C + audience + mobile)

  Serving:
    WebSocket on 0.0.0.0:8765 | HTTP on 0.0.0.0:8080

Windows Training (CUDA):
  Raw audio → 10-step preprocessing → Whisper pseudo-labels → LoRA fine-tune
  Bible corpus → aligned JSONL → QLoRA fine-tune TranslateGemma
```

| Component | Model | Framework | Memory |
|-----------|-------|-----------|--------|
| VAD | Silero VAD | PyTorch | ~2 MB |
| STT | distil-whisper-large-v3 | mlx-whisper | ~1.5 GB |
| Translate (fast) | MarianMT opus-mt-en-es | CT2 int8 | ~76 MB |
| Translate A | TranslateGemma 4B 4-bit | mlx-lm | ~2.5 GB |
| Translate B | TranslateGemma 12B 4-bit | mlx-lm | ~7 GB |
| **Total (4B only)** | | | **~4.3 GB / 18 GB** |
| **Total (A/B)** | | | **~11.3 GB / 18 GB** |
