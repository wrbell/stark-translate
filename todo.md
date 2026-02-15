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

## Phase D — Accent-Diverse STT Tuning (Code Complete, Data Pending)

> **Code implemented:** `download_sermons.py --accent`, `preprocess_audio.py` accent propagation,
> `prepare_whisper_dataset.py` (new), `train_whisper.py` AccentBalancedTrainer.
> See [`docs/accent_tuning_plan.md`](docs/accent_tuning_plan.md) for full 4-week plan.

### D1. Data Collection (Week 1)

27. [ ] **Identify Scottish preacher YouTube playlists** — need 3-5 hrs, 5+ speakers
28. [ ] **Identify British preacher YouTube playlists** — need 2-3 hrs, 5+ speakers
29. [ ] **Identify Canadian preacher YouTube playlists** — need 1-2 hrs, 3+ speakers
30. [ ] **Download Midwest audio** — `python download_sermons.py --accent midwest -n 30`
31. [ ] **Download Scottish audio** — `python download_sermons.py --accent scottish "PLAYLIST_URL"`
32. [ ] **Download British audio** — `python download_sermons.py --accent british "PLAYLIST_URL"`
33. [ ] **Download Canadian audio** — `python download_sermons.py --accent canadian "PLAYLIST_URL"`
34. [ ] **Preprocess all accent audio** — `python training/preprocess_audio.py --input stark_data/raw --output stark_data/cleaned --resume`
35. [ ] **Pseudo-label all accent audio** — `python training/transcribe_church.py --backend faster-whisper --resume`

### D2. Training (Week 2)

36. [ ] **Human-correct bottom 10% of Scottish transcripts** — highest error rate, 4-6 hrs
37. [ ] **Build accent-balanced dataset** — `python training/prepare_whisper_dataset.py`
   - Verify `metadata.csv` accent distribution
   - Verify temperature balancing (T=0.5) rebalances Scottish up
38. [ ] **Train Whisper LoRA Round 1** — `python training/train_whisper.py --accent-balance --dataset stark_data/whisper_dataset`
39. [ ] **Evaluate Round 1** — check per-accent WER, accent gap metric
   - `wer_midwest`, `wer_scottish`, `wer_british`, `wer_canadian`, `wer_accent_gap`
40. [ ] **Human-correct Scottish segments with WER > 30%** — 3-5 hrs
41. [ ] **Train Whisper LoRA Round 2** — retrain with corrected data
42. [ ] **Evaluate Round 2** — compare accent gap improvement

### D3. Gap Closing (Week 3)

43. [ ] **Re-pseudo-label all accent data with Round 2 model**
44. [ ] **Active learning: flag bottom 5-15% per accent, correct**
45. [ ] **Train Whisper LoRA Round 3**
46. [ ] **Evaluate Round 3** — target: accent WER gap < 5%, Scottish WER < 10%

### D4. Integration (Week 4)

47. [ ] **Transfer accent-tuned adapters to Mac**
48. [ ] **Test live inference with Scottish test audio**
49. [ ] **Verify no Midwest WER regression** — primary domain must not degrade
50. [ ] **Record final per-accent WER in `metrics/accent_wer.csv`**

---

## Phase E — Multilingual: Hindi & Chinese (After Phase D)

> See [`docs/multi_lingual.md`](docs/multi_lingual.md) for detailed checklist
> and [`docs/multilingual_tuning_proposal.md`](docs/multilingual_tuning_proposal.md) for research.

### E0. Zero-Shot Baseline (1 day)

51. [ ] **Test TranslateGemma with `target_lang_code="hi"`** — 10 test sentences
52. [ ] **Test TranslateGemma with `target_lang_code="zh-Hans"`** — 10 test sentences
53. [ ] **Spot-check:** Hindi honorifics (तू vs आप for God), Chinese 神 vs 上帝 consistency
54. [ ] **Record baseline chrF++ and COMET** — 50 sample translations per language

### E1. Data Preparation (3-5 days)

55. [ ] **Download Hindi IRV Bible** — `bible-nlp/biblenlp-corpus`, ~31K verses, CC-BY-SA 4.0
56. [ ] **Download Chinese CUV Bible** — Simplified + Traditional, ~31K verses each, public domain
57. [ ] **Align verse pairs** — EN × Hindi = ~155K pairs, EN × Chinese = ~310K+ pairs
58. [ ] **Build Hindi theological glossary** — ~100-150 terms modeled on `build_glossary.py`
59. [ ] **Build Chinese theological glossary** — ~100-150 terms, enforce Protestant terminology
60. [ ] **Decision: 神 vs 上帝** — recommend 神

### E2. QLoRA Training (2 nights)

61. [ ] **Hindi QLoRA** — r=32, max_seq_length=768, 3 epochs on A2000 Ada (~6-9 hrs)
62. [ ] **Chinese QLoRA** — r=32, max_seq_length=512, 3 epochs on A2000 Ada (~6-9 hrs)
63. [ ] **Evaluate both** — chrF++, COMET, theological term accuracy, Hindi honorific accuracy

### E3. Pipeline Integration (1-2 days)

64. [ ] **Transfer adapters to Mac, implement adapter switching** in `dry_run_ab.py`
65. [ ] **Wire up partial models** — IndicTrans2-Dist for Hindi, opus-mt-en-zh for Chinese
66. [ ] **Extend WebSocket protocol** — multi-language translation object
67. [ ] **Hindi partial display strategy** — English partial + Hindi final only (SOV issue)

### E4. Display Updates (1 day)

68. [ ] **Add Devanagari/CJK fonts** — Noto Sans Devanagari (~200KB) + system CJK stack
69. [ ] **Mobile language selector** — [EN] [ES] [HI] [ZH] tabs in `mobile_display.html`
70. [ ] **Test rendering** — Devanagari line-height 1.6, CJK line-height 1.5

---

## Future (P6)

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
