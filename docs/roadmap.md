# Roadmap — Stark Road Bilingual Speech-to-Text

> **Mac EN↔ES follow-up:** [PR #196](https://github.com/wrbell/stark-translate/pull/196) records this work.
> [EN↔ES evidence](evaluation/mac_followup_20260910/README.md) records completed
> source-accounted screens and silent hymn diagnostics. No screened arm qualified;
> defaults remain unchanged. Final artifact, service and merge status is recorded in
> [implementation status](mac_implementation_status.md).


> Living document tracking the project from Mac prototype through Windows training to
> production deployment.
>
> **Last updated:** 2026-09-10 (EN/ES screens complete; repaired-source checks recorded; defaults unchanged).
>
> **Remaining tasks (canonical):** [`backlog.json`](./backlog.json) · rendered
> [`backlog.md`](./backlog.md) · contracts [`current_architecture.md`](./current_architecture.md)
> · evidence [`mac_implementation_status.md`](./mac_implementation_status.md).
> Sections marked **historical** keep earlier measurements under their original dates;
> current numbers live only in the linked evidence documents.

---

## Current State (2026-09-10)

```
Mac (M3 Pro 18 GB, MLX) — the target exercised in this follow-up
  STT:        Parakeet TDT v3 (EN) · mlx-whisper large-v3-turbo (ES)
  Partials:   Marian CT2 int8 on CPU, every 0.6 s of speech (HF fallback)
  Finals:     Gemma 4 E4B OptiQ on 0.5 s silence (E2B opt-in, TranslateGemma opt-out)
  VAD:        packaged Silero 6.2.1; ONNX opt-in
  Capture:    PortAudio in a disposable child, 5 s no-input / 3 s idle timeouts (mic-stall fix)
  Operator:   FastAPI + vanilla JS control plane (:9000) for lay volunteers; readiness from
              the pipeline health channel; Review/export; support bundles; preflight
  Setup:      stark-translate setup / doctor, models.lock.json, managed Marian CT2 cache
  Timing:     schema 2 speech_end_to_final_ms + speech-end → visible-browser ACK upper bound
  MTP:        --mts rejected before load (#177); offline probe only

Lite (implemented; hardware performance pending)
  Profiles:   standard (default) · lite-cpu · lite-cpu-quality · lite-cuda-8gb
  Runtime:    Torch-free lite-cpu extra, stark-translate-lite, ONNX Silero, Whisper small CT2,
              Marian CT2 finals or Gemma 4 E2B via session-owned llama-server (b10883)
  Evidence:   isolated Mac CPU synthetic EN+ES caption/TTS smoke; E2B GGUF + native runtime
              CPU quality smoke passed — no x86 CPU, native Windows or RTX 2070 run yet

Windows / WSL (A2000 Ada 16 GB, CUDA)
  Inference:  W16 Whisper CT2 + Marian CT2 + Gemma 4 E4B Q4_K_M via llama.cpp b10883
  Training:   Phase 4 preprocess, E4B domain SFT, W17 — scripted, not run since 2026-04-30
  Latency:    CUDA proposal scripts header-marked unexecuted

Source and releases
  published:  v2026.14.0.0 (2026-09-11; GitHub Release + MSI; PRs #192–#203) — v2026.13 was the previous release
  source:     v2026.14 / 2026.14.0.0; integration history in PR #192, EN/ES follow-up in PR #196
  evidence:   per-feature acceptance and source/artifact identity are recorded separately
  publishing: tag, GitHub Release and MSI done 2026-09-11; PyPI pending trusted-publisher mapping; GHCR image pending the PR #203 rebuild
```

**Live microphone:** the 2026-09-09 built-in-microphone session (`20260909_233204_799019_en`)
stalled after "Listening..." — the operator showed RUNNING from the CSV header while no
audio frames arrived and the audience display stayed disconnected; a separate
`sounddevice` record probe stalled too. File-replay EN/ES sessions on the same build
completed. The fix (isolated capture with no-input timeouts, health-derived readiness,
owned-process cleanup) is **implemented and integrated**. With microphone permission
allowed, real EN and ES capture/readiness/stop and EN pause/resume passed on September 10.
The initial room had no detected speech. Later synthetic acoustic EN completed
with captions; Spanish and a traced retest failed on source loss. The retest lost
no parent-handoff frames but retained 160 ms upstream loss. Exact microphone
name/host-API selection now passes a separate bounded native identity probe,
which saved no audio and ran no STT. Sustained live capture remains unvalidated.
A separate controlled file replay reached a visibly observed Chrome caption. [Exact operator evidence](evaluation/attended_mic_20260910/README.md) keeps
those observations separate (`mac-live-mic-stall`, `issue-131-smoke`).

Day-of-event workflow: [`operator_runbook.md`](./operator_runbook.md) (with recorded UI evidence). First-time install:
[`packaging/macos.md`](./packaging/macos.md), `bootstrap.sh`.

---

## Active Work

Status, priority, dependencies and acceptance for every item below are in
[`backlog.json`](./backlog.json); this section is the narrative.

### Mac — source integrated, remaining certification

1. **PR #192 source merge completed** (`pr-192-integration`): [actual closeout records](evaluation/overnight_closeout_20260910/README.md) bind the merge and final-head CI. Earlier frozen source `752ab9a` passed 2,363 CPU-suite tests, four skips, 63.80% coverage and three GPU regressions; its wheel matches all 152 runtime members. The original Standard hour failed source-bound validation. The fresh full-service Standard hour completed with all 563 final spans/WAV headers and 2,814 preview spans consistent, 7,594 successful writes and observed cleanup. CPU Lite also completed with 468 final/271 preview spans consistent, all 1,979 writes complete and cleanup observed. Sparse first previews and large observed tails prevent a fast-production recommendation. Keep these functional cohorts, remote CI and source review separate from the completed screen. The unused 350-second slice is not a cohort. See [current evidence](mac_implementation_status.md) and PR #192 for actual source integration; publication remains separate.
2. **Live microphone** (`mac-live-mic-stall`, `issue-131-smoke`): the stall fix is implemented — `tools/isolated_audio.py` / `capture_worker.py` (disposable PortAudio child, 5 s no-input and 3 s idle timeouts), `tools/pipeline_health.py` readiness consumed by the operator. Real capture/readiness passed for EN/ES after microphone permission. Later synthetic acoustic EN completed, but Spanish retained upstream sample loss; exact device identity resolution passed its separate bounded probe. Preserve the failed receipts and resolve sustained capture before claiming the live gate. Further microphone/output testing is deferred by the current user instruction. #131 closes only on its original full acceptance.
3. **Sub-second caption delivery** (`caption-delivery-goal`, `overnight-latency-scheduling`): the [96-run screen](evaluation/overnight_screen_20260910/README.md) completed with 672 finals and 0/28 selected arms. The sub-second goal was not met on this 45-second English cohort; E4B defaults remain unchanged. No ordinary confirmation or combination of these arms is justified. Pursue [new measured hypotheses](latency_next_experiments.md), keeping tiny endpoint counts, control drift, unreviewed references and locked-native-screen/DOM telemetry separate from certification.
4. **Human and device gates:** locally reviewed Spanish church references, blinded bilingual review, natural two-speaker audio (#133) and second physical output. Public natural EN/ES read-speech references are now available for engineering comparisons; they do not supply church review. Original #132 permits independent physical or virtual routing, while the separate physical-device gate requires real outputs. The laptop runbook rehearsal (#134) is complete and closed, with full hymn/spoken input, setup-to-first-caption timing and a written note retained in the [closeout evidence](evaluation/overnight_closeout_20260910/README.md). #193/#194 track remaining hymn/quality work; live microphone testing remains under #131.
5. **Active learning evidence (#137):** Review/export is implemented and fixture-tested. A real human correction from a recorded Sunday must reach a dated corpus and be merged; the documented first retrain may be a dry run. Draft notes and generated text are not approved pairs.
6. **Separate R&D (#138):** the [offline Hindi church-audio baseline](./evaluation/overnight_hindi/README.md) is complete and archived. Human Hindi review, live integration and the QLoRA decision remain pending; no further Hindi work is scheduled in this EN↔ES latency program.

### Current follow-up evidence

The [Mac follow-up report](evaluation/mac_followup_20260910/README.md) retains the
original-acceptance audit, pinned public development/confirmation data, STT and
fixed-reference translation comparisons, decoder trials and an isolated
application dependency candidate. E2B was faster in isolated translation with
lower reference overlap and fewer passing lexical canaries; those results do not
measure caption delivery or establish bilingual meaning approval. Standard,
Spanish Parakeet, CPU Lite cadence and independent Lite deadline screens are now
complete, with no qualified arms. CPU Whisper-base failed its separate WER guard
in both languages. E4B, Spanish Whisper and the 0.6-second partial cadence remain
the defaults; no follow-up result establishes the sub-second caption goal.
Current [source checks](evaluation/mac_followup_20260910/final-760e948/source-validation.md)
bind repaired source `760e948`. The [delivery packet](evaluation/mac_followup_20260910/final-760e948/README.md)
keeps artifact and full-service acceptance separate from those checks and the
earlier c13 hymn diagnostics.

[Hymn source repairs](evaluation/mac_followup_20260910/hymn-source-repairs.md)
preserve existing text delimiters and accepted speech onset after music hold.
The 15-frame recovery threshold and 0.7-second final minimum are unchanged.
Both silent diagnostic packets have executed: the natural file control never
entered music hold, and the text comparison used supplied boundary hypotheses.
See the completed [natural control](evaluation/mac_followup_20260910/final-c13f51f/hymn-capture.md)
and [text packet](evaluation/mac_followup_20260910/final-c13f51f/hymn-boundary.md).
#193/#194 remain open for independently reviewed natural transition boundaries
and bilingual meaning; these diagnostics do not approve a detector or prompt change.

### Equal-priority deployment targets

- **Lite CPU inference** (`lite-cpu-inference`): **implemented and integrated** — `stark_translate/profiles.py` (`lite-cpu`, `lite-cpu-quality`), `operator_app/lite_preflight.py`, Torch-free `lite-cpu` extra, `stark-translate-lite`, pinned Whisper small / Marian / E2B / Silero ONNX artifacts, `tools/llama_runtime.py`. Evidence: isolated Mac CPU synthetic EN+ES caption/TTS smoke and actual installed CPU E2B quality inference ([`lite_profiles.md`](./lite_profiles.md)). The full-service CPU Lite replay completed on the Mac with consistent retained spans, writes and cleanup; see the [endurance report](evaluation/overnight_endurance_20260910/README.md). Its sparse previews and large tails remain performance limitations; this is not a controlled Standard/Lite comparison or a fast-production recommendation. Performance and natural-speech quality on an x86 CPU host: pending hardware.
- **Native Windows / RTX 2070** (`rtx2070-native-validation`): `lite-cuda-8gb` implemented (pinned Windows CUDA 12.4 llama.cpp archives, sm_75 Linux build option); nothing has run on a 2070 or native Windows; the v2026.13 MSI digest was verified without Windows execution and the MSI remains a scaffold plan.

### WSL — pipeline refresh (pending hardware)

Ordered runbook: [`wsl_pipeline_refresh.md`](./wsl_pipeline_refresh.md) — Phase 4 full
preprocess → Gemma 4 E4B domain SFT → GGUF (8-canary sanity) → W17 DoRA + hard-mix → CT2
(`benchmark_stt_engines.py` gate: W17 ≤ W16) → optional Parakeet EN bench → Mac transfer
and Phase 7 A/B (#135) → Phase 8 active learning. Then the
[CUDA latency proposal](./cuda_latency_proposal.md) (MTP drafter opt-in, `-fa on` retest,
W16 HF fp16 / Parakeet probes); the v2026.9 followups (`-fa`, `-c 2048`, prompt-cache
reuse, #175) are folded into it. Gemma 4 tuning results and next directions:
[`gemma4_tuning/v1_results.md`](./gemma4_tuning/v1_results.md),
[`gemma4_tuning/v3_directions.md`](./gemma4_tuning/v3_directions.md).

### Experiments kept opt-in

The [September 10 screen](evaluation/overnight_screen_20260910/README.md) completed
96/96 valid runs and selected 0/28 experiment/model arms. All 588 candidate final
comparisons against each control set retained identical text, without establishing
reference accuracy. [First-preview E4B](evaluation/overnight_screen_20260910/first_preview_e4b.md)
illustrates why control drift and repeat-level guards matter. These results do not
justify ordinary confirmations or combined settings; new hypotheses remain distinct.

After the [48-run frozen screen](./evaluation/mac_v2026_14_screening/README.md) no
combined configuration beat E4B + 0.5 s silence + 0.6 s cadence on both models, and the
[24 synthetic routing probes](./evaluation/mac_v2026_14_routing/README.md) only proved the
conservative Marian policy routes as designed. Shorter silence, final-aware partials,
idle-only warmup, ONNX VAD, terminology prompt, conservative routing and the Gemma 4
assistant drafter (#177) all stay opt-in. Any default change needs a matched both-model
gain plus human review.

---

The [next EN↔ES latency proposals](latency_next_experiments.md) rank subsequent
experiments from observed stage delays. They require profiling and confirmation;
they are not implemented default changes.

## Completed Work (historical)

### Phase 1: Infrastructure & Inference (done; v2026.1–v2026.6)

- **Engines package** — ABCs (`STTEngine`, `TranslationEngine`, `TTSEngine`), MLX + CUDA implementations, factory auto-detection
- **Whisper Large-V3-Turbo swap** — both partials and finals (Mac ES and CUDA still use it; Mac EN moved to Parakeet in v2026.13)
- **CUDA streaming runtime** — `CUDAGemmaStreamingEngine` with TextIteratorStreamer, prompt cache, 4B→12B speculative decoding, VRAM tier detection (superseded for finals by llama.cpp in v2026.5)
- **Dual-target inference** — `--backend auto|mlx|cuda`, `--no-ab`, `--low-vram`
- **Unified config** — `settings.py` (pydantic-settings, `STARK_` prefix)
- **Piper TTS** — EN + ES voices, ONNX runtime, WebSocket/WAV/local output, `--tts`
- **Bidirectional language support** — `--lang en` / `--lang es`
- **Pipeline overlap** — translation(N) concurrent with STT(N+1)
- **5 display modes** — Audience, A/B, Mobile, Church, OBS overlay
- **CI/CD** — GitHub Actions workflows (current count and names in `CLAUDE.md` § CI/CD), coverage gate in `test.yml`, Codecov, pre-commit, CalVer

### Phase 2: Data Collection (done; 2026-03)

- **333 sermons cataloged**, 160+ downloaded, organized into `stt-data/{type}/{year}/`
- **Deepgram Nova-3 oracle** — 35 sermons transcribed with 50 theological keyterms
- **Tiered glossary** — Tier 1 (50 boost terms), Tier 2 (229 master terms)
- **Data integrity** — SHA-256 lockfile, 2026-03-14 training cutoff, adapter health checks
- **Evaluation sets** — 500 stratified verse holdout + 422 sermon eval chunks + fresh eval set (4 post-cutoff sermons)

### Phase 3: TranslateGemma Fine-Tuning (done; superseded by Gemma 4 tuning)

- **S1–S9 ablation sweep**; **S6 winner** (balanced 1:1 verse/sermon) — details in [`archive/training/gemma_tuning_test_matrix.md`](archive/training/gemma_tuning_test_matrix.md)
- **Platense corpus misalignment** found 2026-04-29 and fixed in `verse_pairs_train_v2.jsonl` ([`platense_alignment_bug.md`](./platense_alignment_bug.md)); the sweep trained on partly misaligned pairs
- **Gemma 4 program** replaced it: v1 → v1.1 → v2-cpo reached parity with stock E4B, Jacobo canary still failing ([`gemma4_tuning/v1_results.md`](./gemma4_tuning/v1_results.md))

### Phase 4: Whisper LoRA (W16 deployed; W17 pending)

- **W0–W9 ablation**, **W12 data scaling** (198K Deepgram-aligned chunks from 328 sermons), **W15 hard example mining** — [`archive/research/hard_mining.md`](archive/research/hard_mining.md), [`archive/v2026.5/w15_postmortem.md`](archive/v2026.5/w15_postmortem.md)
- **W16** — production CUDA STT as CTranslate2 int8_float16; measurements and their definitions in [`archive/v2026.7/STT_BENCHMARK.md`](archive/v2026.7/STT_BENCHMARK.md)
- **W17** — DoRA + hard-mix curriculum scripted (`training/run_w17_curriculum.sh`), not trained
- **Alignment hardening** — sharded Arrow writes, memory cap, streaming to disk

### Mac latency program (v2026.13, on main)

PRs #180–191: measurement correction (legacy `e2e_latency_ms` is processing time),
Parakeet EN STT, Marian CT2 on Mac CPU, replay harness, thread-safe MLX overlap.
Numbers: [`archive/v2026.13/MAC_LATENCY.md`](archive/v2026.13/MAC_LATENCY.md).

### Mac reliability and evaluation (v2026.14 candidate, local branch)

Implemented: operator session identity and metric schemas, drained shutdown, schema 2
timing, backend-aware setup/preflight with the shared resolver, packaged VAD, managed
Marian CT2 cache, live/post-session Review with portable exports and evaluation/training
separation, frozen manifests and the 48-run screen, local wheel/sdist/Mac ZIP checks.
Validation counts, security scope and artifact hashes are recorded once in
[`mac_implementation_status.md`](./mac_implementation_status.md) and
[`evaluation/README.md`](./evaluation/README.md). Open gates are the human/device items
under Active Work.

---

## Upcoming Phases

### Phase 5: Adapter Evaluation & Transfer (pending WSL artifacts; #135)

- Transfer W16 CT2 and v2-cpo to the Mac — runbook §5
- Run the translation-only text-canary prerequisite with `tools/health_check.py --backend mlx --n-canaries 8`; it does not test STT or a live A/B
- Separately compare stock E4B versus v2-cpo finals and stock STT versus W16 on the Mac CPU faster-whisper path; keep the deliberate Parakeet English default as its own comparison
- Written ship/no-ship note; stock E4B stays default on no-ship

### Phase 6: Active Learning Feedback Loop (implemented path, evidence pending; #137)

- Route low-confidence segments to operator Review (implemented)
- Human correction → dated corrections corpus → `tools/merge_corrections.py` (dry run acceptable for the first cycle)
- Retrain on corrected data (2–4 cycles) — runbook §6; stop when the worst metric improves < 2 % relative for two cycles

### Phase 7: Live Demo Deployment — shipped as v2026.6

FastAPI + vanilla JS operator control plane, pre-flight gating, mid-session controls,
observability sparklines, device enumeration, verse highlights, summary trigger,
systemd/launchd/bootstrap install. Workflow: [`operator_runbook.md`](./operator_runbook.md).
Deferred: macOS Shortcuts voice triggers.

### Phase 8: Multilingual Expansion (Hindi & Chinese) — pending user decision

The separate [offline Hindi church-audio baseline](./evaluation/overnight_hindi/README.md)
is complete: Parakeet English transcripts and E4B/E2B Hindi predictions are archived.
It establishes offline availability, without Hindi reference scores, human approval,
a QLoRA decision or a live target. The earlier [text probe](./evaluation/mac_v2026_14_hindi/README.md)
remains separate evidence. Future steps below require a later language decision;
no Hindi/Chinese work is scheduled in the current EN↔ES latency program.

| Step | What |
|------|------|
| Zero-shot baseline | `target_lang_code="hi"` / `"zh-Hans"` through the live pipeline on church audio; note SOV partial garble |
| Data preparation | biblical verse pairs (`bible-nlp/biblenlp-corpus`), glossary ≥ 100 terms |
| Hindi / Chinese QLoRA | separate adapters, r=32; 768 / 512 max sequence |
| Evaluation + integration | chrF++/COMET, adapter switching, display labels |

Historical design proposals: English partial + Hindi final; Chinese → 神 (Shen) for God. These do not authorize the pending language expansion.

### Phase 9: Piper TTS Multi-Language (deferred)

Scripts ready: `prepare_piper_dataset.py`, `train_piper.py`, `export_piper_onnx.py`,
`evaluate_piper.py`. Multi-channel routing code exists (Phase 10 below).

### Phase 10: Production Polish

- Done — dedicated hardware auto-start (systemd unit, launchd plist, `bootstrap.sh`)
- Done — post-sermon summary trigger and live verse highlights in the operator UI (workflow evidence in the rehearsal; bilingual accuracy review pending)
- **Original routing acceptance supported** — 9.4.1 multi-channel TTS routing (#132): per-language operator selection/persistence, native host-output routing and engine tests have [retained evidence](evaluation/mac_followup_20260910/tts-routing-acceptance.md). Physical unplug/replug, human audibility and a complete caption-triggered native-TTS session remain separate pending checks; they do not extend the original issue’s physical-or-virtual routing acceptance. Issue closure follows reviewed merge.
- **Implemented, gate not run** — 9.6.1 live diarization (#133): `--diarize`, rolling buffer, separate daemon, `speaker` on finals. Needs a two-speaker clip and the +50 ms p95 check ([`live_diarization.md`](./live_diarization.md)).
- **Completed — laptop runbook rehearsal (#134).** The original 06:49 UTC Standard hour remains failed source-bound evidence. Repaired Standard and CPU Lite completed with consistent retained spans, required writes and cleanup; the [endurance report](evaluation/overnight_endurance_20260910/README.md) keeps their quality limits and selected waveform checks explicit. The actual source merge and #134 closure are recorded in the [closeout evidence](evaluation/overnight_closeout_20260910/README.md). #193/#194 retain hymn/quality follow-ups; live microphone certification remains under #131. Continuous improvement follows Phases 6/8.

---

## Observed Metrics (historical — see linked sources for definitions)

### STT (2026-03, Whisper LoRA program)

| Metric | Base (no fine-tune) | Target | Source |
|--------|--------------------|--------|--------|
| WER (church, fresh eval) | 21.41 % normalized | < 10 % | `training/CLAUDE.md` § W12 |
| WER (Scottish accent) | ~22–34 % | < 10 % | [`archive/research/accent_tuning_plan.md`](archive/research/accent_tuning_plan.md) |

W16 fresh-eval and 41-clip bench WER (different measurements, do not mix):
[`archive/v2026.7/STT_BENCHMARK.md`](archive/v2026.7/STT_BENCHMARK.md).

### Translation (2026-03, TranslateGemma S-sweep)

| Metric | S6 (fine-tuned 4B) vs 12B base |
|--------|--------------------------------|
| COMET | −0.0002 (tied) — [`archive/training/gemma_tuning_test_matrix.md`](archive/training/gemma_tuning_test_matrix.md) |

Gemma 4 tuning results: [`gemma4_tuning/v1_results.md`](./gemma4_tuning/v1_results.md).

### CUDA inference

HF NF4 era (2026-03): [`archive/training/benchmark_training.md`](archive/training/benchmark_training.md).
llama.cpp cutover and later tuning: [`archive/v2026.5/BENCHMARK.md`](archive/v2026.5/BENCHMARK.md),
[`archive/v2026.9/GEMMA_OPTIM_PHASE2.md`](archive/v2026.9/GEMMA_OPTIM_PHASE2.md),
[`archive/v2026.10/IQ4_XS_BENCHMARK.md`](archive/v2026.10/IQ4_XS_BENCHMARK.md),
[`archive/v2026.11/IMATRIX_CALIBRATION.md`](archive/v2026.11/IMATRIX_CALIBRATION.md).

### Mac inference

TranslateGemma-era tables (2026-03) and the 2026-08-30 Gemma 4 run that hit `max_tokens`
on every call (EOS bug #172) are superseded. Current measurements, with the
speech-end/processing distinction: [`archive/v2026.13/MAC_LATENCY.md`](archive/v2026.13/MAC_LATENCY.md)
and [`evaluation/README.md`](./evaluation/README.md).

---

## Key Decisions (Resolved)

| Decision | Resolution |
|----------|------------|
| Finals model family | Gemma 4 E4B — CUDA via llama.cpp Q4_K_M (v2026.5), Mac via OptiQ 4-bit (2026-08-30); TranslateGemma is opt-out |
| CUDA Gemma loading | llama.cpp GGUF; HF NF4 kept as legacy fallback (PLE embeddings make it 14–15 GB) |
| Mac EN STT | Parakeet TDT v3 MLX (v2026.13); Whisper large-v3-turbo stays for ES and CUDA (W16 CT2) |
| Partials | Marian CT2 int8 — CUDA (v2026.8) and Mac CPU (v2026.13); HF fallback |
| Mac defaults | E4B, 0.5 s silence, 0.6 s cadence retained after separate 48-run and 96-run screens; all experiments opt-in |
| E2B | Separately evaluated, opt-in `--gemma4-size e2b`; default promotion requires the complete performance/quality gates and bilingual review |
| Measurement | Schema 2 `speech_end_to_final_ms`; legacy `e2e_latency_ms` labeled processing time |
| MTP / assistant drafter | Off on both platforms (#177 experimental; CUDA `--mtp` opt-in, unbenchmarked) |
| Environments | pyproject extras `.[mlx]` / `.[cuda]` / `.[cpu]`; `requirements-*.txt` deprecated except the WSL training env |
| Pipeline threading | 2 workers on MLX (≥ 0.31.2 thread-local streams) and CUDA; `--multiprocess` escape hatch (#176 implemented) |
| Training data alignment | Deepgram word timestamps + faster-whisper chunk boundaries; corpus v2 after the Platense fix |
| Deployment targets | Mac primary; lite CPU and native Windows/RTX 2070 equal priority |
| Publishing | PR #192 merged and justified issues closed; PyPI/package/release tags pending |

## Key Decisions (Pending)

| Decision | When | Options / owner |
|----------|------|-----------------|
| Hindi/Chinese timing | Later user decision | Separate R&D; offline Hindi church-audio baseline archived (#138), live path not started; no overnight action |
| Church Spanish recording source | Local source and independent review pending | Public FLEURS Spanish already supports separately labeled engineering WER; church terminology and locally approved quality remain unvalidated |
| E2B as default | After a qualifying performance/quality comparison and blinded bilingual review | Current screens qualified no arm; retain the measured speed, meaning and terminology tradeoffs |
| Production hardware | Before future production-device certification | Dedicated church PC vs portable Mac; the laptop #134 rehearsal is already complete |
| PyPI publication and release tag | Deferred by decision (2026-09-11) | Tag `v2026.14.0.0`, GitHub Release, MSI and GHCR image are published; the PyPI publish job is gated on `PYPI_PUBLISH_ENABLED` and stays off |
| W17 curriculum iterations | After Phase 4 on WSL | 2–4 cycles typical |
| Scottish accent data sources | Before accent tuning | User provides playlist URLs |
| TTS voice fine-tuning | Phase 9 | Fine-tune from Piper base vs train from scratch |
| Lite vs RTX 2070 certification order | When the required x86 or RTX 2070 hardware is available | Profiles are implemented; run the documented device-specific gates on whichever host is available first |

---

## Reference Documents

| Doc | Contents |
|-----|----------|
| [`backlog.json`](./backlog.json) | Machine-readable remaining tasks with status, certification and acceptance |
| [`current_architecture.md`](./current_architecture.md) | v2026.14 candidate inference/operator contracts |
| [`overnight_status.md`](./overnight_status.md) | Overnight worktree deliverables and unfinished areas |
| [`mac_implementation_status.md`](./mac_implementation_status.md) | Local validation evidence (single source for test counts) |
| [`evaluation/README.md`](./evaluation/README.md) | Frozen manifests, measurement definitions, reports |
| [`mlx_cuda_parity.md`](./mlx_cuda_parity.md) | MLX ↔ CUDA semantic parity checklist |
| [`cuda_latency_proposal.md`](./cuda_latency_proposal.md) | Unexecuted CUDA latency scripts and gates |
| [`live_diarization.md`](./live_diarization.md) | Rolling-buffer diarization design and p95 budget |
| [`gemma4_tuning/overview.md`](./gemma4_tuning/overview.md) | Gemma 4 E2B/E4B tuning program |
| [`training_plan.md`](archive/training/training_plan.md) | Full training schedule, channel inventory, go/no-go gates |
| [`accent_tuning_plan.md`](archive/research/accent_tuning_plan.md) | Accent-diverse STT tuning plan |
| [`hard_mining.md`](archive/research/hard_mining.md) | W15 hard example mining design |
| [`multi_lingual.md`](archive/research/multi_lingual.md) | Hindi & Chinese actionable todo list |
| [`multilingual_tuning_proposal.md`](archive/research/multilingual_tuning_proposal.md) | Full Hindi/Chinese research |
| [`rtx2070_feasibility.md`](archive/research/rtx2070_feasibility.md) | RTX 2070 hardware analysis |
| [`fast_stt_options.md`](archive/research/fast_stt_options.md) | Lightning-whisper-mlx feasibility (not viable) |
| [`projection_integration.md`](archive/research/projection_integration.md) | OBS/NDI/ProPresenter integration |
| [`turbo_inference.md`](archive/research/turbo_inference.md) | Turbo model inference details |
| [`data_pipeline_status.md`](archive/training/data_pipeline_status.md) | Data pipeline state (2026-03) |
