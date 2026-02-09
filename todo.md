# TODO — Stark Road Bilingual Speech-to-Text

> **Mac:** M3 Pro, 18GB unified, MLX + PyTorch
> **Windows:** WSL2, A2000 Ada 16GB, CUDA — for training
> **Completed work:** see [docs/previous_actions.md](docs/previous_actions.md)

---

## Phase A — Before Seattle (Feb 8-11)

### Done

- [x] Repo reorganization — scripts into `training/`, `tools/`, `features/`, `displays/`
- [x] Bible corpus generated — 269K verse pairs in `bible_data/aligned/`
- [x] Glossary expanded — 229 terms via `build_glossary.py`
- [x] Training scripts syntax-verified — all 8 pass AST parse
- [x] Seattle training run designed — see `docs/seattle_training_run.md`

### Mac (Feb 8-11)

1. [ ] **Run live pipeline test** — `python dry_run_ab.py`, 5+ min, 10+ utterances
   - Verify MarianMT partials + TranslateGemma finals end-to-end
   - Verify A/B/C display renders correctly (`displays/ab_display.html`)
   - Verify audio WAVs and diagnostics JSONL saving
   - Review CSV output + diagnostics summary
2. [ ] **Initial VAD/STT tuning** — review first session's diagnostics
   - Note VAD_THRESHOLD, SILENCE_TRIGGER, MAX_UTTERANCE behavior
   - Log any theological word misrecognitions for WHISPER_PROMPT
3. [ ] **Push final pre-Seattle state** — commit all changes, push to remote

### Windows (Feb 11 evening)

4. [ ] **Set up WSL environment** — install deps per `CLAUDE-windows.md`, run pre-flight checklist
5. [ ] **Download 30 sermons** — `python download_sermons.py -n 30` (~30 hrs, ~1.7 GB)
6. [ ] **Launch `seattle_run.sh` in tmux** — see `docs/seattle_training_run.md`

---

## Phase B — During Seattle (Feb 12-17, Mac)

> Windows training runs unattended (~50 GPU-hrs). Mac is free for code work.
> See `docs/seattle_training_run.md` for the Windows timeline.

### P7 Latency — Code Work (no models needed)

7. [ ] **P7-6A: Streaming translation display** — 300-500ms perceived savings
   - Show translation tokens as they generate (via `mlx_lm.stream_generate`)
   - WebSocket sends partial translation tokens → display renders incrementally
   - User sees first words ~200ms after STT, instead of waiting for full translation
8. [ ] **P7-6B: Adaptive model selection** — 200-400ms on simple utterances
   - Route short/simple utterances to MarianMT-only (skip TranslateGemma)
   - Heuristic: <8 words, no theological terms, high STT confidence → fast path
   - Falls back to TranslateGemma for complex sentences
9. [ ] **P7-6D: Batch multiple short utterances** — ~100-200ms amortized
   - Combine rapid-fire short phrases into single translation call
   - Reduces per-utterance overhead for back-and-forth dialogue

### P7 Latency — Benchmarking (needs models loaded)

10. [ ] **P7-1B: Benchmark whisper-large-v3-turbo** — potential ~50-100ms STT savings
    - `pip install mlx-whisper` already installed; download turbo variant
    - Compare WER and latency vs distil-whisper-large-v3
    - If WER is comparable, switch partials to turbo

### P5 — Monitoring

11. [ ] **Run YouTube caption comparison** — `tools/live_caption_monitor.py` on a sermon with good captions
    - Retest fixed methodology (repetition detection, alignment validation)
    - Establish baseline cross-system WER before fine-tuning

### P6 — Feature Testing (if sermon WAVs available on Mac)

12. [ ] **Test speaker diarization** — `features/diarize.py` on a sermon WAV
13. [ ] **Test verse extraction** — `features/extract_verses.py` on a session CSV
14. [ ] **Test sermon summary** — `features/summarize_sermon.py` on a session CSV

---

## Phase C — After Seattle (Feb 18+)

> Return, check training results, transfer adapters to Mac.

### Immediate (Feb 18-19)

15. [ ] **Check training results** — review `logs/seattle_run_*/progress.log`
    - Compare baseline vs post-training eval (BLEU, chrF++, COMET)
    - Check if 12B completed or OOMed
    - Go/No-Go: WER >10% relative improvement, BLEU >+2 points
16. [ ] **Transfer LoRA adapters** — `scp` from Windows → Mac
    - `fine_tuned_whisper_mi/`, `fine_tuned_marian_mi/`
    - `fine_tuned_gemma_mi_A/`, `fine_tuned_gemma_mi_B/` (if completed)
17. [ ] **Convert LoRA → MLX format** — ~30 min per model
    - Merge LoRA weights into base model, requantize for MLX
    - Verify output equivalence on test sentences

### Testing Fine-Tuned Models (Feb 19-21)

18. [ ] **A/B test: base vs fine-tuned** — `python dry_run_ab.py` with adapter paths
    - Compare translation quality side-by-side
    - Measure latency impact of fine-tuned adapters
19. [ ] **Tune VAD parameters** — with enough live session data
    - VAD_THRESHOLD (0.3), SILENCE_TRIGGER (0.8s), MAX_UTTERANCE (8s)
20. [ ] **Expand Whisper prompt** — add misrecognized theological words from sessions
21. [ ] **Calibrate QE thresholds** — using qe_a/qe_b data from 5+ sessions
22. [ ] **Evaluate MarianMT vs TranslateGemma divergence** — review marian_similarity scores

### P7 Latency — Post-Training

23. [ ] **P7-2D: Compare smaller translation models** — now including fine-tuned MarianMT
    - gemma-3-1b-it, NLLB-600M, fine-tuned MarianMT (from Seattle run)
    - If fine-tuned MarianMT closes quality gap, it may replace TranslateGemma for speed

### Ongoing

24. [ ] **Tune sentence boundary splitting** — review bad_split flags in CSV
25. [ ] **Fix singing/hymns breaking VAD and STT**
    - Options: lower gain during music, inaSpeechSegmenter, RMS heuristic
26. [ ] **Active learning pipeline** — Label Studio, versioned corrections, 3-5 retrain cycles
    - Route low-QE segments to human review → retrain on Windows

---

## P1 — Physical Testing (When Ready)

- [ ] **Test in actual church environment**
  - Mic gain, projection readability, speaker handoffs, font/fade feedback
- [ ] **Test audience display readability with real users**
  - Feedback from non-English speakers on fading context, layout, partials

---

## Future (P6)

- [ ] **Multi-language support** — TranslateGemma supports 36 languages
- [ ] **RTX 2070 edge deployment** — see `docs/roadmap.md` Phase 3

---

## P7 — Sub-1s Latency Reference

**Status:** Phase 1 + Phase 2 + 6C implemented. Pipeline overlap active.

```
Post Phase 1+2+6C budget (measured):
  4B:   STT ~300ms + Translate ~350ms = ~650ms (overlap hides translation)
  12B:  STT ~300ms + Translate ~800ms = ~1100ms
```

| # | Task | Savings | Status | When |
|---|------|---------|--------|------|
| 6C | Pipeline N/N-1 overlap | 300-500ms actual | DONE | — |
| 6A | Streaming translation display | 300-500ms perceived | TODO | Phase B |
| 6B | Adaptive model selection | 200-400ms on simple | TODO | Phase B |
| 6D | Batch short utterances | ~100-200ms amortized | TODO | Phase B |
| 1B | whisper-large-v3-turbo | ~50-100ms STT | TODO | Phase B |
| 2D | Smaller translation models | varies | TODO | Phase C |
| 2A | Speculative decoding | — | DEFERRED | benchmarked 18-90% slower on M3 Pro |

### Research Sources

- [mlx-lm: speculative decoding, prompt caching, stream_generate](https://github.com/ml-explore/mlx-lm)
- [Fast STT Options feasibility study](docs/fast_stt_options.md)

---

## Architecture Reference

```
Mac Inference (MLX) — Two-Pass Architecture:
  Mic → Silero VAD (threaded) → [Partial: mlx-whisper + MarianMT (~480ms, italic)]
                               → [Final: mlx-whisper + TranslateGemma 4B (~650ms, regular)]
                               → WebSocket → Browser (displays/)

  Optimizations: prompt caching, model pre-warming, background I/O,
                 fast resampling, pipeline overlap (6C)

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
