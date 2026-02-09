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

## P3 — Data Collection & Prep

- [ ] **Download sermon audio** — `python download_sermons.py -n 30` *(Mac or Windows)*
  - 275 videos available, 249.8 hours total
  - Start with 30 most recent (~30 hrs), expand to 50-80 if quality is good

- [ ] **Download Bible parallel corpus** *(Mac or Windows)*
  - Clone scrollmapper (in progress), run `prepare_bible_corpus.py`
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

See `docs/training_time_estimates.md` for detailed time projections.

---

## P5 — Monitoring & Feedback Loop

- [ ] **Run first YouTube caption comparison** — test `live_caption_monitor.py post` with a downloaded sermon
- [ ] **Active learning pipeline** — Label Studio, versioned corrections, 3-5 retrain cycles

---

## P6 — Future Features

- [ ] **Test speaker diarization** — run `diarize.py` on a sermon WAV, verify 2-speaker output
- [ ] **Test verse extraction** — run `extract_verses.py` on a session CSV, review accuracy
- [ ] **Test sermon summary** — run `summarize_sermon.py` on a session CSV
- [ ] **Multi-language support** — TranslateGemma supports 36 languages
- [ ] **Hardware portability** — RTX 2070 endpoints (see `docs/rtx2070_feasibility.md`)

---

## P7 — Sub-1s Latency Roadmap

**Status:** Phase 1 + Phase 2 implemented. Projected 4B path: ~1150ms → ~650ms.

```
Post Phase 1+2 budget (estimated):
  4B:   STT ~300ms + Translate ~350ms = ~650ms
  12B:  STT ~300ms + Translate ~800ms = ~1100ms
```

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

### Remaining P7 items

- [ ] **2A. Benchmark speculative decoding** — tune `num_draft_tokens` 3-16, measure actual 12B speedup
- [ ] **1B. whisper-large-v3-turbo for partials** — ~50-100ms, benchmark WER vs distil first
- [ ] **2D. Smaller translation models** — gemma-3-1b-it, NLLB-600M, fine-tuned MarianMT
- [ ] **4E. Increase prefill chunk size** — ~10-20ms
- [ ] **5C. Use `asyncio.to_thread`** — ~1-2ms micro-optimization
- [ ] **6D. Batch multiple short utterances** — ~100-200ms amortized
- [ ] **Upgrade mlx-whisper to 0.4.3** — alignment computation speedup

### Research Sources

- [mlx-lm: speculative decoding, prompt caching, stream_generate](https://github.com/ml-explore/mlx-lm)
- [Apple ReDrafter: 1.37-2.3x speedup on Apple Silicon](https://machinelearning.apple.com/research/recurrent-drafter)
- [Lightning-SimulWhisper: MLX/CoreML streaming](https://github.com/altalt-org/Lightning-SimulWhisper)
- [WhisperKit: 0.45s streaming latency](https://github.com/argmaxinc/WhisperKit)
- [Fast STT Options feasibility study](docs/fast_stt_options.md)

---

## Architecture Reference

```
Mac Inference (MLX) — Two-Pass Architecture:
  Mic → Silero VAD (threaded) → [Partial: mlx-whisper + MarianMT (~480ms, italic)]
                               → [Final: mlx-whisper + TranslateGemma 4B (~650ms, regular)]
                               → WebSocket → Browser (A/B/C + audience + mobile + OBS)

  Optimizations: prompt caching, model pre-warming, background I/O, fast resampling

  Serving:
    WebSocket on 0.0.0.0:8765 | HTTP on 0.0.0.0:8080

Windows Training (CUDA):
  Raw audio → 10-step preprocessing → Whisper pseudo-labels → LoRA fine-tune
  Bible corpus → aligned JSONL → QLoRA fine-tune TranslateGemma
```

| Component | Model | Framework | Memory |
|-----------|-------|-----------|--------|
| VAD | Silero VAD | PyTorch (threaded) | ~2 MB |
| STT | distil-whisper-large-v3 | mlx-whisper | ~1.5 GB |
| Translate (fast) | MarianMT opus-mt-en-es | CT2 int8 | ~76 MB |
| Translate A | TranslateGemma 4B 4-bit | mlx-lm | ~2.5 GB |
| Translate B | TranslateGemma 12B 4-bit | mlx-lm | ~7 GB |
| **Total (4B only)** | | | **~4.3 GB / 18 GB** |
| **Total (A/B)** | | | **~11.3 GB / 18 GB** |
