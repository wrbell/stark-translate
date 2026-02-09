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

**Target:** Under 1s E2E for 4B path (currently ~1.3s), under 1.5s for 12B path (currently ~1.9s)

**Current latency budget:**
```
4B-only mode:                        A/B mode:
  STT (mlx-whisper):   ~500ms          STT:           ~500ms
  Translation (4B):    ~650ms          Translation A:  ~650ms
  ─────────────────────────            Translation B: ~1400ms
  Total serial:       ~1150ms          ────────────────────────
  Total overlap:      ~1150ms          Total (bottleneck): ~1900ms
```

#### Completed

- [x] **Parallel A/B translation** — `run_in_executor` for simultaneous inference
- [x] **Shorter chunk duration** — 2s chunks + two-pass partial/final architecture
- [x] **VAD-triggered instant processing** — process on silence gap (0.8s)
- [x] **MarianMT for fast partials** — ~80ms vs ~650ms TranslateGemma

---

#### Category 1: STT Optimizations

- [ ] **1A. Disable `word_timestamps` for partials** *(~100-200ms savings)*
  - Complexity: **Low** | Quality impact: **None** (partials don't need word-level data)
  - Mac-only: No (portable pattern)
  - `word_timestamps=True` causes significant overhead in mlx-whisper: a 40s audio
    file takes ~27s with timestamps vs ~3-4s without (7x slower). The overhead is
    smaller for 2-5s chunks but still ~100-200ms extra
  - Currently: `process_final()` uses `word_timestamps=True` (needed for confidence),
    but `process_partial()` also calls `mlx_whisper.transcribe()` without explicitly
    setting `word_timestamps=False`
  - **Implementation:**
    ```python
    # In process_partial():
    result = mlx_whisper.transcribe(
        audio_data,
        path_or_hf_repo=stt_pipe,
        language="en",
        condition_on_previous_text=False,
        initial_prompt=_whisper_prompt(),
        word_timestamps=False,  # ADD THIS — no per-word data needed for partials
    )
    ```
  - Also adds `word_timestamps=False` explicitly to be defensive against future
    default changes in mlx-whisper

- [ ] **1B. Switch to `whisper-large-v3-turbo` for partials** *(~50-100ms savings)*
  - Complexity: **Low** | Quality impact: **Minimal** (Turbo has ~1% higher WER than distil)
  - Mac-only: No (MLX model, but concept portable to any Whisper runtime)
  - Model: `mlx-community/whisper-large-v3-turbo` (fp16, ~809M params, 4 decoder layers)
  - vs current: `mlx-community/distil-whisper-large-v3` (fp16, ~756M params, 2 decoder layers)
  - Turbo has 4 decoder layers vs distil's 2, but benchmarks show comparable speed
  - Main advantage: better multilingual support and slightly different error profile
  - Could use turbo for partials (where speed matters most) and distil for finals
  - **Implementation:**
    ```python
    PARTIAL_STT_MODEL = "mlx-community/whisper-large-v3-turbo"
    FINAL_STT_MODEL = "mlx-community/distil-whisper-large-v3"
    # Warm up both at startup
    ```
  - **Benchmark first:** run both on 20 sermon clips, compare WER and latency

- [ ] **1C. Try `lightning-whisper-mlx` for faster STT** *(~100-200ms savings)*
  - Complexity: **Medium** | Quality impact: **None** (same underlying model, faster runtime)
  - Mac-only: Yes (MLX-specific implementation)
  - Claims 10x faster than whisper.cpp, 4x faster than standard mlx-whisper
  - Uses batched decoding and optimized attention kernels
  - Supports `distil-large-v3` model variant
  - `pip install lightning-whisper-mlx`
  - **Implementation:**
    ```python
    from lightning_whisper_mlx import LightningWhisperMLX
    whisper = LightningWhisperMLX(model="distil-large-v3", batch_size=12, quant=None)
    result = whisper.transcribe("/path/to/audio.wav")
    ```
  - **Risk:** may not support `initial_prompt` (theological vocabulary bias) — verify
  - **Risk:** may not support `word_timestamps` — verify for final pass

- [ ] **1D. Try `Lightning-SimulWhisper` for true streaming STT** *(~200-300ms savings)*
  - Complexity: **High** | Quality impact: **Minimal** (uses AlignAtt streaming policy)
  - Mac-only: Yes (MLX + CoreML hybrid)
  - Repo: `altalt-org/Lightning-SimulWhisper`
  - Uses CoreML encoder (Apple Neural Engine, up to 18x faster) + MLX decoder
  - Implements SimulStreaming's AlignAtt policy for simultaneous transcription
  - Can run large-v3-turbo in real-time on M2 MacBook Pro
  - Instead of waiting for full utterance, starts outputting text as audio streams in
  - Winner of IWSLT 2025 Simultaneous Speech Translation shared task
  - **Implementation:** major refactor — replaces the current batch-transcribe approach
    with a streaming API. Process_partial and process_final merge into a single
    continuous stream
  - **Dependency:** requires CoreML model conversion. May need to convert
    `distil-large-v3` to CoreML format using `coremltools`
  - Estimated latency: ~200ms for first hypothesis vs current ~500ms

- [ ] **1E. Reduce Whisper prompt length** *(~20-30ms savings)*
  - Complexity: **Low** | Quality impact: **Minimal risk** (test on live sessions)
  - Mac-only: No (portable)
  - Current `WHISPER_PROMPT` is ~85 words / ~120 tokens — Whisper must encode all of
    these as decoder prefix before generating
  - `prev_text` adds up to ~200 chars more (~50 tokens) — total prefix ~170 tokens
  - Each prefix token costs ~0.15-0.3ms on distil-large-v3 via MLX
  - Trim prompt to essential 40-50 words (most impactful theological terms only):
    ```python
    WHISPER_PROMPT = (
        "Sermon at Stark Road Gospel Hall. God, Christ, Holy Spirit. "
        "Salvation, atonement, mediator, covenant, righteousness, "
        "sanctification, redemption, reconciliation, grace, mercy, "
        "repentance, forgiveness, glory, reign, altar, prophet. "
        "Scripture, the Gospel, the Lord."
    )
    ```
  - Also: cap `prev_text` at 100 chars instead of 200

- [ ] **1F. WhisperKit (CoreML native) for maximum STT speed** *(~200-300ms savings)*
  - Complexity: **High** | Quality impact: **None** (same Whisper model, optimized runtime)
  - Mac-only: Yes (Swift/CoreML, requires bridging to Python)
  - WhisperKit achieves 0.45s mean latency for streaming hypothesis text
  - Uses Apple Neural Engine (ANE) for near-peak hardware utilization
  - 2.2% WER — matches or beats other implementations
  - **Problem:** WhisperKit is Swift — would need Python bridge via `subprocess`,
    `pyobjc`, or rewrite audio pipeline in Swift
  - Pre-converted CoreML models: `argmaxinc/whisperkit-coreml` on HuggingFace
  - Best considered as a future production optimization, not immediate

---

#### Category 2: Translation Optimizations

- [ ] **2A. Speculative decoding — 4B drafts for 12B** *(~400-700ms savings on 12B path)*
  - Complexity: **Low** (already partially implemented in code) | Quality impact: **None**
  - Mac-only: No (mlx-lm feature, concept portable to vLLM/TensorRT-LLM)
  - `mlx_lm.generate` supports `draft_model` parameter natively
  - TranslateGemma 4B and 12B share same Gemma 3 tokenizer (262K vocab, SentencePiece)
    so they are confirmed compatible for speculative decoding
  - Both models already loaded in A/B mode — zero additional memory cost
  - Apple's ReDrafter research shows 1.37-2.3x speedup on Apple Silicon via MLX
  - LM Studio users report 20-50% speed gains with speculative decoding
  - Expected: 12B from 1400ms to 600-900ms (1.5-2.3x speedup)
  - Tune `num_draft_tokens` parameter: start at 3 (current), sweep 4-16
    - Higher values = more aggressive speculation, better when draft acceptance is high
    - Translation tasks have high acceptance rates (structured output)
  - **Implementation:** already wired in `translate_mlx()` and `process_final()`:
    ```python
    # Already in dry_run_ab.py line 767:
    task_b = loop.run_in_executor(
        None, lambda: translate_mlx(mlx_b_model, mlx_b_tokenizer, english,
                                    draft_model=mlx_a_model))
    ```
  - **Remaining work:** benchmark different `num_draft_tokens` values, log acceptance
    rate, verify no quality regression on theological terms
  - **Known issue:** GitHub mlx-lm #250 reports speculative `_step` can be slower than
    non-speculative on some hardware — benchmark on M3 Pro specifically

- [ ] **2B. Prompt caching / KV cache reuse for translation** *(~50-80ms savings)*
  - Complexity: **Medium** | Quality impact: **None**
  - Mac-only: No (concept portable)
  - TranslateGemma chat template produces a fixed prefix for every translation:
    ```
    <bos><start_of_turn>user\n{"source_lang_code": "en", "target_lang_code": "es", ...}
    ```
  - This prefix is ~30-40 tokens and identical across all translation calls
  - mlx-lm supports `make_prompt_cache()` to pre-compute and reuse KV cache:
    ```python
    from mlx_lm import load, generate
    from mlx_lm.utils import make_prompt_cache

    model, tokenizer = load(MLX_MODEL_A)
    cache = make_prompt_cache(model)

    # First call populates cache with the fixed prefix
    # Subsequent calls skip re-computing the prefix KV states
    result = generate(model, tokenizer, prompt=full_prompt,
                      prompt_cache=cache, max_tokens=max_tok)
    ```
  - Saves ~50-80ms per call (prefix prefill at ~500 tokens/sec = 30-40 tokens * 2ms/token)
  - **Caveat:** cache must be reset between conversations or when prompt changes.
    For translation, the prefix is static, so cache is always valid
  - Previously deferred as "~30-40 tokens, <50ms savings" but with other optimizations
    stacking, every 50ms counts toward the sub-1s target

- [ ] **2C. Reduce `max_tokens` more aggressively** *(~20-50ms savings)*
  - Complexity: **Low** | Quality impact: **None** (if calibrated correctly)
  - Mac-only: No (portable)
  - Current formula: `max_tok = max(32, int(input_words * 2.5))`
  - For a typical 15-word utterance: max_tok = 37
  - For a typical 30-word utterance: max_tok = 75
  - Spanish is ~1.15-1.25x English in word count, ~1.3x in tokens
  - Tighter formula: `max_tok = max(24, int(input_words * 1.8))`
  - Saves ~20-50ms by reducing unnecessary generation overhead on the margin
  - **Implementation:** change one line in `translate_mlx()`:
    ```python
    max_tok = max(24, int(input_words * 1.8))
    ```
  - Monitor for truncation — if any translation gets cut off, loosen the multiplier

- [ ] **2D. Explore smaller translation models** *(~200-400ms savings on 4B path)*
  - Complexity: **Medium** | Quality impact: **Moderate** (likely lower BLEU on theological text)
  - Mac-only: No (portable)
  - Options to benchmark:
    - `google/gemma-3-1b-it` 4-bit via mlx-lm (~0.6GB) — general-purpose, not translation-specific
    - `Helsinki-NLP/opus-mt-en-es` already loaded as MarianMT — use as final translator
      for simple utterances (< 10 words), TranslateGemma for complex ones
    - `facebook/nllb-200-distilled-600M` — Meta's NLLB, strong on translation
    - Fine-tuned MarianMT (from P4 training) — could bridge quality gap
  - Tiered approach: use CT2 MarianMT (~30ms) for short phrases, 4B (~650ms) for
    complex sentences only
  - **Decision criteria:** run all on the 229-term theological glossary +
    50 sermon excerpts, compare BLEU/chrF++ scores

- [ ] **2E. Early stopping on translation** *(~30-50ms savings)*
  - Complexity: **Low** | Quality impact: **None**
  - Mac-only: No (portable)
  - Currently using `generate()` which runs to `max_tokens` or EOS
  - Switch to `stream_generate()` and stop as soon as `<end_of_turn>` is detected:
    ```python
    from mlx_lm import stream_generate
    tokens = []
    for response in stream_generate(model, tokenizer, prompt=prompt, max_tokens=max_tok):
        tokens.append(response.token)
        if response.token == eot_id:
            break
    ```
  - **Wait:** `generate()` already stops at EOS (we set `_eos_token_ids`), so this
    only saves time if there's overhead in post-EOS processing. Benchmark to confirm.
  - Alternative benefit: `stream_generate()` enables sending partial translation
    tokens to the browser as they're generated (streaming translation display)

---

#### Category 3: Pipeline Parallelism

- [ ] **3A. Overlap STT and translation via streaming** *(~200-400ms savings)*
  - Complexity: **High** | Quality impact: **None** (same models, different scheduling)
  - Mac-only: No (architectural pattern)
  - Currently: STT runs to completion (500ms), THEN translation starts (650ms) = serial
  - With streaming: start translating the first N words of STT output while Whisper
    continues decoding the rest
  - **Architecture:**
    ```
    Current (serial):
      [───── STT 500ms ─────][────── Translate 650ms ──────] = 1150ms

    Proposed (overlapped):
      [───── STT 500ms ─────]
                [── Translate (starts after first ~200ms of STT) ──]
      = ~800-900ms total
    ```
  - **Implementation approach:**
    1. Use `mlx_whisper.transcribe()` as-is (it doesn't support streaming output)
    2. Instead, run STT on the first 1s of audio (partial), start translation
       immediately, then run STT on full audio and re-translate if text changed
    3. This is essentially what the two-pass architecture already does — the partial
       pass IS the overlap mechanism
  - **Better approach:** run STT and translation on different threads. While STT
    processes chunk N, translate chunk N-1. Requires pipelining:
    ```python
    # Pseudo-code for pipeline overlap:
    async def audio_loop():
        prev_english = None
        while True:
            audio = await get_next_chunk()
            # Start translating previous chunk's text in parallel with current STT
            if prev_english:
                translate_task = asyncio.create_task(translate_async(prev_english))
            stt_result = await run_stt(audio)
            if prev_english:
                translation = await translate_task
                broadcast(prev_english, translation)
            prev_english = stt_result.text
    ```
  - **Latency impact:** removes ~200-400ms of serial waiting, but adds 1-chunk
    display delay (text appears one chunk later)

- [ ] **3B. Pre-compute translation prompt tokens during STT** *(~30-50ms savings)*
  - Complexity: **Low** | Quality impact: **None**
  - Mac-only: No (portable)
  - While STT is running (~500ms), pre-tokenize and pre-compute the fixed
    TranslateGemma chat template prefix
  - Currently, `tokenizer.apply_chat_template()` + tokenization happens after STT
    completes — can be done in parallel
  - **Implementation:** in `process_final()`, fire template preparation as a concurrent
    task before STT:
    ```python
    # Pre-compute the template prefix (everything except the actual text)
    template_prefix = tokenizer.apply_chat_template(
        [{"role": "user", "content": [{"type": "text",
          "source_lang_code": "en", "target_lang_code": "es",
          "text": "PLACEHOLDER"}]}],
        add_generation_prompt=True
    )
    # After STT completes, only need to swap in the actual text and re-tokenize
    # the variable part (much faster than full template + tokenize)
    ```
  - Small but free optimization when combined with other changes

---

#### Category 4: Model Optimizations

- [ ] **4A. Pre-warm models during silence** *(~50-150ms savings on first-after-idle)*
  - Complexity: **Low** | Quality impact: **None**
  - Mac-only: Yes (Metal GPU warm-up specific)
  - Metal GPU has cold-start latency when kernels haven't run recently
  - MLX compiles and caches Metal kernels on first use — this is a one-time cost
  - But after extended silence (>5-10s), Metal GPU can power down, adding ~50-150ms
    on next inference
  - Fire a 1-token forward pass on TranslateGemma when silence detected (>2s gap)
  - Also fire a tiny Whisper transcribe on 0.1s of silence to keep encoder warm
  - **Implementation:**
    ```python
    # In audio_loop(), when silence extends beyond 2 seconds:
    if silence_frames > int(2.0 * SAMPLE_RATE / 512) and not gpu_warm:
        # Warm up TranslateGemma (1-token generation)
        loop.run_in_executor(None, lambda: generate(
            mlx_a_model, mlx_a_tokenizer, prompt="translate", max_tokens=1, verbose=False))
        # Warm up Whisper (tiny audio)
        loop.run_in_executor(None, lambda: mlx_whisper.transcribe(
            np.zeros(1600, dtype=np.float32),
            path_or_hf_repo=stt_pipe, language="en"))
        gpu_warm = True
    if has_speech:
        gpu_warm = False
    ```
  - Low complexity, high impact for perceived responsiveness after pauses

- [ ] **4B. Increase `mx.set_cache_limit` for translation models** *(~20-40ms savings)*
  - Complexity: **Low** | Quality impact: **None**
  - Mac-only: Yes (MLX/Metal specific)
  - Currently set to `100 * 1024 * 1024` (100MB) to prevent memory growth
  - This limit applies to Metal shader cache and intermediate computation buffers
  - For short translations (10-50 tokens), 100MB is sufficient
  - But if cache is too small, MLX may re-allocate buffers between calls, adding latency
  - Try `256 * 1024 * 1024` (256MB) — still well within the 18GB budget:
    ```python
    mx.set_cache_limit(256 * 1024 * 1024)  # 256MB — balance speed vs memory growth
    ```
  - Monitor total memory usage across sessions to ensure no leak

- [~] **4C. KV cache quantization (`kv_bits=4`)** — deferred
  - Not beneficial for short translations (10-50 tokens)
  - Would help for long-context features (summarization, 3K+ tokens)
  - Revisit when post-sermon summary feature is implemented

- [ ] **4D. Quantize whisper model to 4-bit** *(~50-100ms savings, experimental)*
  - Complexity: **Medium** | Quality impact: **Moderate** (may increase WER 1-3%)
  - Mac-only: Yes (MLX quantization)
  - Model: `mlx-community/whisper-large-v3-mlx-4bit` (exists on HuggingFace)
  - Current distil-large-v3 is fp16 (~1.5GB) — 4-bit would be ~400MB
  - Faster encoder forward pass due to reduced memory bandwidth
  - **Risk:** Whisper 4-bit quantization is less mature than LLM quantization
  - **Benchmark:** compare WER on 20 sermon clips before committing
  - Only use for partials if quality drops; keep fp16 for finals

- [ ] **4E. Increase prefill chunk size** *(~10-20ms savings)*
  - Complexity: **Low** | Quality impact: **None**
  - Mac-only: Yes (MLX specific)
  - LM Studio found that increasing prefill chunk size from default 512 to 4096
    gives ~2x faster prompt processing on Apple Silicon
  - mlx-lm may have a similar internal parameter — check `mlx_lm.generate` kwargs
  - For short translation prompts (~40-80 tokens), impact is small but measurable

---

#### Category 5: System-Level Optimizations

- [ ] **5A. Eliminate scipy resampling overhead** *(~5-10ms per frame savings)*
  - Complexity: **Low** | Quality impact: **None**
  - Mac-only: No (portable)
  - Currently: mic records at 48kHz, `scipy.signal.resample()` downsamples to 16kHz
    in every audio callback (every ~32ms frame)
  - `scipy.signal.resample` uses FFT — overkill for fixed-ratio resampling
  - Replace with `librosa.resample()` (faster polyphase filter) or integer decimation:
    ```python
    # Simple 3:1 decimation (48kHz → 16kHz) with anti-alias filter
    from scipy.signal import decimate
    raw_16k = decimate(raw, 3).astype(np.float32)
    ```
  - Or record at 16kHz directly if mic supports it (many USB mics do):
    ```python
    stream = sd.InputStream(samplerate=16000, ...)  # skip resampling entirely
    ```
  - Saves ~5-10ms per frame * 30 frames/sec = CPU headroom for other tasks

- [ ] **5B. Move VAD to separate thread** *(~5-10ms latency reduction)*
  - Complexity: **Medium** | Quality impact: **None**
  - Mac-only: No (portable)
  - VAD (`is_speech()`) runs synchronously in the audio loop, blocking STT scheduling
  - Silero VAD takes ~1-3ms per frame on CPU, but any Python GIL contention adds jitter
  - Run VAD in a dedicated thread with a ring buffer:
    ```python
    vad_thread = threading.Thread(target=vad_worker, daemon=True)
    # vad_worker reads from audio_queue, writes speech/silence events to vad_event_queue
    ```
  - Eliminates frame-level jitter in the main asyncio loop

- [ ] **5C. Use `asyncio.to_thread` instead of `run_in_executor`** *(~1-2ms savings)*
  - Complexity: **Low** | Quality impact: **None**
  - Mac-only: No (portable, Python 3.9+)
  - `asyncio.to_thread()` is a simpler, slightly more efficient wrapper than
    `loop.run_in_executor(None, ...)` — avoids executor pool overhead
  - Micro-optimization but trivial to apply:
    ```python
    # Before:
    task_a = loop.run_in_executor(None, lambda: translate_mlx(...))
    # After:
    task_a = asyncio.create_task(asyncio.to_thread(translate_mlx, ...))
    ```

- [ ] **5D. Profile and optimize Python overhead** *(~10-30ms savings)*
  - Complexity: **Medium** | Quality impact: **None**
  - Mac-only: No (portable)
  - Profile the full pipeline with `py-spy` or `cProfile` to find Python-level bottlenecks:
    ```bash
    py-spy record -o profile.svg -- python dry_run_ab.py --4b-only
    ```
  - Known overhead candidates:
    - `tokenizer.apply_chat_template()` — uses Jinja2, can be ~5-10ms
    - `tokenizer.encode()` / `tokenizer.decode()` — ~1-3ms each
    - JSON serialization for WebSocket broadcast — ~1-2ms
    - `save_chunk_audio()` WAV write — ~5-10ms (move to background thread)
    - `write_diag_jsonl()` — ~2-5ms (move to background thread)
  - Move all I/O (WAV save, JSONL write, CSV write) to background threads:
    ```python
    asyncio.get_event_loop().run_in_executor(None, save_chunk_audio, audio_data, cid)
    asyncio.get_event_loop().run_in_executor(None, write_diag_jsonl, ...)
    ```

---

#### Category 6: Architectural Changes

- [ ] **6A. Streaming translation display** *(perceived latency reduction ~300-500ms)*
  - Complexity: **Medium** | Quality impact: **None**
  - Mac-only: No (portable)
  - Instead of waiting for full translation, stream tokens to the browser as generated
  - User sees Spanish words appearing one by one — perceived latency drops dramatically
    even though total compute time is the same
  - Use `stream_generate()` instead of `generate()`:
    ```python
    from mlx_lm import stream_generate

    async def translate_streaming(model, tokenizer, text, ws_clients):
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        spanish_tokens = []
        for response in stream_generate(model, tokenizer, prompt=prompt, max_tokens=max_tok):
            spanish_tokens.append(response.text)
            if len(spanish_tokens) % 3 == 0:  # broadcast every 3 tokens
                await broadcast({"type": "translation_stream",
                                 "partial_spanish": "".join(spanish_tokens)})
    ```
  - **Browser change:** audience_display.html must handle `translation_stream` events
    and update text incrementally
  - Previously tried `stream_generate` and switched to `generate` to "eliminate callback
    overhead" — re-evaluate whether the overhead was real or negligible

- [ ] **6B. Adaptive model selection based on utterance complexity** *(~200-400ms savings on simple utterances)*
  - Complexity: **Medium** | Quality impact: **Minimal** (simple utterances translate well with any model)
  - Mac-only: No (portable)
  - Not all utterances need TranslateGemma 4B — short phrases ("God is good",
    "Let us pray") translate perfectly with MarianMT
  - Use a heuristic to route:
    ```python
    def choose_translator(english_text):
        words = english_text.split()
        # Short, common phrases → MarianMT (30ms)
        if len(words) <= 8:
            return "marian"
        # Complex theological sentences → TranslateGemma 4B (650ms)
        return "gemma_4b"
    ```
  - Could also use STT confidence: high-confidence simple text → MarianMT,
    complex/ambiguous text → TranslateGemma
  - **Risk:** MarianMT handles theological terms worse — needs glossary fine-tuning first

- [ ] **6C. Pipeline N/N-1 overlap** *(~300-500ms savings)*
  - Complexity: **High** | Quality impact: **None**
  - Mac-only: No (portable)
  - While translating chunk N-1's text, run STT on chunk N's audio simultaneously
  - Requires restructuring the audio loop to decouple STT and translation scheduling:
    ```
    Time:    |--STT chunk 1--|--STT chunk 2--|--STT chunk 3--|
                             |--Translate 1--|--Translate 2--|
    ```
  - Currently STT and translation for the same chunk are sequential
  - With N/N-1 overlap, the effective E2E latency for the steady-state path becomes:
    `max(STT_time, Translate_time)` instead of `STT_time + Translate_time`
  - For 4B: max(500, 650) = 650ms instead of 1150ms
  - For 12B: max(500, 1400) = 1400ms instead of 1900ms
  - **Trade-off:** adds one chunk of display delay (text appears one utterance later)
  - **Implementation:** requires async task management with careful ordering:
    ```python
    pending_translation = None
    while True:
        audio = await get_next_utterance()
        if pending_translation:
            translation_result = await pending_translation
            await broadcast(translation_result)
        english = await run_stt(audio)
        pending_translation = asyncio.create_task(run_translation(english))
    ```
  - **Compatibility:** does NOT conflict with the two-pass partial/final architecture.
    Partials still use MarianMT in real-time. This optimizes the final pass only.

- [ ] **6D. Batch multiple short utterances** *(~100-200ms savings per batch)*
  - Complexity: **Medium** | Quality impact: **None**
  - Mac-only: No (portable)
  - When speaker pauses briefly between short phrases, accumulate 2-3 phrases and
    translate as a single batch
  - TranslateGemma prefill cost is mostly fixed — translating "Hello. How are you?"
    takes roughly the same time as translating "Hello." alone
  - Reduces per-utterance translation overhead from ~650ms to ~350ms amortized
  - **Implementation:** hold final translation for 200ms after silence — if another
    utterance starts within that window, batch them together
  - **Trade-off:** adds up to 200ms extra display delay on the first phrase in a batch

---

#### Priority-Ordered Implementation Plan

**Phase 1 — Quick wins (target: 4B path from 1150ms to ~900ms)**
1. 1A. Disable `word_timestamps` for partials *(100-200ms, 30 min)*
2. 4A. Pre-warm models during silence *(50-150ms, 1 hr)*
3. 2C. Reduce `max_tokens` multiplier *(20-50ms, 5 min)*
4. 1E. Reduce Whisper prompt length *(20-30ms, 15 min)*
5. 5D. Move I/O to background threads *(10-30ms, 30 min)*
6. 5A. Eliminate scipy resampling overhead *(5-10ms, 15 min)*

**Phase 2 — Medium effort (target: 4B path to ~750ms, 12B to ~1200ms)**
7. 2A. Speculative decoding benchmark + tune `num_draft_tokens` *(400-700ms on 12B, 2 hrs)*
8. 2B. Prompt caching / KV cache reuse *(50-80ms, 1 hr)*
9. 1C. Benchmark lightning-whisper-mlx *(100-200ms, 2 hrs)*
10. 4B. Increase `mx.set_cache_limit` *(20-40ms, 5 min)*

**Phase 3 — Architectural (target: 4B path to ~650ms, 12B to ~900ms)**
11. 6A. Streaming translation display *(perceived 300-500ms, 4 hrs)*
12. 6C. Pipeline N/N-1 overlap *(300-500ms actual, 8 hrs)*
13. 6B. Adaptive model selection *(200-400ms on simple utterances, 4 hrs)*

**Phase 4 — Advanced (production optimization)**
14. 1D. Lightning-SimulWhisper streaming STT *(200-300ms, 16 hrs)*
15. 1F. WhisperKit CoreML native *(200-300ms, 24 hrs)*
16. 4D. Quantize Whisper to 4-bit *(50-100ms, 2 hrs + quality validation)*

**Projected latency after each phase:**
```
              4B path    12B path    Effort
Current:      ~1150ms    ~1900ms     —
Phase 1:      ~850ms     ~1700ms     ~3 hrs
Phase 2:      ~650ms     ~1100ms     ~6 hrs
Phase 3:      ~500ms     ~800ms      ~16 hrs
Phase 4:      ~300ms     ~600ms      ~44 hrs
```

---

#### Research Sources

- [MLX framework and optimizations](https://github.com/ml-explore/mlx)
- [mlx-lm speculative decoding, prompt caching, stream_generate](https://github.com/ml-explore/mlx-lm)
- [Apple ReDrafter — up to 2.3x speedup on Apple Silicon](https://machinelearning.apple.com/research/recurrent-drafter)
- [Lightning-SimulWhisper — MLX/CoreML streaming](https://github.com/altalt-org/Lightning-SimulWhisper)
- [lightning-whisper-mlx — 10x faster Whisper on Apple Silicon](https://github.com/mustafaaljadery/lightning-whisper-mlx)
- [WhisperKit — 0.45s streaming latency, 2.2% WER](https://github.com/argmaxinc/WhisperKit)
- [SimulStreaming — IWSLT 2025 winner](https://github.com/ufal/SimulStreaming)
- [WWDC 2025 — MLX on Apple Silicon](https://developer.apple.com/videos/play/wwdc2025/298/)
- [MLX prefill chunk size optimization](https://github.com/thornad/lmstudio-mlx-patch)
- [mlx-whisper word_timestamps memory/perf issues](https://github.com/ml-explore/mlx-examples/issues/1254)
- [Whisper large-v3-turbo vs distil-large-v3](https://huggingface.co/openai/whisper-large-v3-turbo/discussions/40)
- [Benchmarking ML on Apple Silicon with MLX (paper)](https://arxiv.org/abs/2510.18921)
- [LM Studio speculative decoding](https://lmstudio.ai/blog/lmstudio-v0.3.10)

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
