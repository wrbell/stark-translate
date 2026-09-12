# EN↔ES latency: closed-arm registry and open attribution questions

This file is the single registry of every latency arm that has been screened and closed on the Mac. A closed arm
is **never re-run as a confirmation and never enters a combination** (`docs/evaluation/overnight_experiment_plan.md`,
`CLAUDE.md`). Each row links to the immutable evidence that closed it. New hypotheses must be distinct and
source-bound; pick them from measured overlap (see "Open attribution questions"), not from this list.

## Closed arms — never re-run or combined

| Arm(s) | Knob / env | Verdict | Evidence |
|---|---|---|---|
| 14 experiments × {E2B, E4B} (`allocation_512/1024`, `async_captions`, `clause_preview`, `first_preview`, `latest_partial`, `marian_memo`, `pause_preview`, `pause_speculation`, `prefix_cache`, `rolling_6s`, `streaming_stt`, `vad_worker_torch/onnx`) | `STARK_EXPERIMENT_*` (`MLX_CACHE_MB`, `ASYNC_CAPTIONS`, `CLAUSE_PREVIEW_S`, `FIRST_PREVIEW_S`, `LATEST_PARTIAL`, `MARIAN_MEMO`, `PAUSE_PREVIEW_MS`, `SPECULATE_PAUSE_MS`, `GEMMA_PREFIX_CACHE`, `INCREMENTAL_STT=rolling/stream`, `VAD_WORKER`) | 0/28 selected; every row failed a tail guard and the median gate | [overnight screen 2026-09-10](evaluation/overnight_screen_20260910/README.md) |
| Early clause cuts (2 s / 4 s × 160 / 240 ms) and partial-STT deadline margins (100 / 250 ms) × {E4B, E2B} × {EN, ES} | `STARK_EXPERIMENT_EARLY_CLAUSE_S` + `_EARLY_CLAUSE_PAUSE_MS`, `STARK_EXPERIMENT_PARTIAL_DEADLINE_MARGIN_MS` | 0/24 qualified (36/72 median-only) | [Standard screen](evaluation/mac_followup_20260910/standard-screen-result.md) |
| Spanish Parakeet STT (vs mlx-whisper large-v3-turbo) × {E4B, E2B} | `--stt-backend parakeet-mlx` for ES | 0/2 (queue-wait and update-gap tails) | [Spanish Parakeet screen](evaluation/mac_followup_20260910/spanish-parakeet-result.md) |
| CPU Lite partial cadence 0.9 s / 1.2 s × {EN, ES} | `--partial-interval` | 0/4 (preview coverage loss, tails) | [Lite cadence screen](evaluation/mac_followup_20260910/lite-cadence-result.md) |
| CPU Lite deadline margins 100 / 250 ms × {EN, ES} | `STARK_EXPERIMENT_PARTIAL_DEADLINE_MARGIN_MS` | 0/4 (STT stage-queue and delivery tails) | [Lite deadline screen](evaluation/mac_followup_20260910/lite-deadline-result.md) |
| CPU Whisper base (vs small) | Lite STT model size | rejected on WER in both languages | [CPU STT comparison](evaluation/mac_followup_20260910/cpu-stt-comparison.md) |
| Gemma 4 MTP / assistant drafter (`--mts`), incl. the RoPE-offset patch | `--mts`, `engines/mlx_spec.py` | 31 % acceptance, medium p50 ≈ 0.96×; off; #177 closed not-planned | [MTP notes](mlx_mtp_notes.md), [v2026.13 MAC_LATENCY §3](archive/v2026.13/MAC_LATENCY.md) |
| 0.4 s partial cadence; 12 CT2 threads | `--partial-interval 0.4`; CT2 `intra_threads` 12 | both worse (GPU/CPU contention) | [v2026.13 MAC_LATENCY §5](archive/v2026.13/MAC_LATENCY.md) |
| E2B OptiQ as speculative draft for E4B, γ = 1 / 2 / 3 (isolated text bench) | `STARK_EXPERIMENT_DRAFT_MODEL_ID`, `_DRAFT_TOKENS` | byte-identical but canary-length gain below the 15 %/150 ms gate | [overnight L2](evaluation/overnight_20260911/L2-e2b-draft/README.md) |
| `draft_g3` live (E2B draft γ=3) and `serial_finals` (no STT/translation overlap), 360 s clips | same draft knobs; `STARK_EXPERIMENT_SERIAL_FINALS` | draft: STT starved on the 18 GB budget, previews −30 %, p95 ×2–3; serial: overlap rare, waiting only hurts | [tail screen 2026-09-11](evaluation/followup_20260911/X-tail-screen/README.md) |
| Series 3 arms (2026-09-12): `marian_threads_2`, `max_utterance_6`, `partial_recheck_translation` | `STARK_TRANSLATE_MARIAN_INTRA_THREADS=2`; `STARK_VAD_MAX_UTTERANCE=6.0`; `STARK_EXPERIMENT_PARTIAL_RECHECK_TRANSLATION=true` | all REJECTED on G1 on both clips: threads-2 raised the Gemma p95 (+16 % A, +6 % B); the 6 s cap lowered medians but raised the clip A tail and failed its quality guard (20 % concatenated WER, duplicates); the partial re-check cut the clip A Gemma p95 by 8.8 % (below the 15 %/300 ms gate), nothing on B, previews −7…−13 % | [series 3 L-B](evaluation/series3_20260912/LB-tail-screen/README.md) |
| Series 4 (2026-09-12) engineering change, not an arm — P1 runtime fixes: keep-warm after the final, first stream token at 1, hash-pinned Parakeet joint scalar decode at load; Metal wired limit at load **rejected** (opt-in only) | `STARK_WARMUP_AFTER_FINAL`, `STARK_STREAM_FIRST_TOKEN_BATCH_SIZE`, `STARK_PARAKEET_JOINT_EVAL` (all on by default), `STARK_MLX_WIRED_LIMIT` (off) | merged on the paired identity screen (byte-identical finals 6/6 pairs, tail/preview/memory gates pass): Gemma-silence first-token p50 −10.8 % (A) / −7.5 % (B), final STT p50 −23 % / −13 %, payload-ready p50 −8.9 % / −2.4 %; the wired limit showed no speed effect and +3.8 GB peak RSS, so it stays off | [series 4 P1](evaluation/series4_20260912/P1-runtime/README.md) |
| Series 4 (2026-09-12) arm `partial_reuse` at 100 / 300 ms — the final reuses the last emitted partial's STT for silence finals | `STARK_EXPERIMENT_PARTIAL_REUSE_MS` (0 off; opt-in `partial_reuse_keep_confidence` variant merged in #219 on 2026-09-12, unscreened) | both REJECTED: Gemma-silence first-token p50 −18…−24 % and payload-ready p50 −12…−22 %, but the text guard failed on the denser clip (EN WER 2.7–6.5 %, ES 3.7–8.0 % vs the 3 % guard; +3 finals per run at 300 ms), reused finals lose the Marian route (all-silence median only −6 % / 0 % on clip A) and the clip A first-token p95 (+15 % / +24 %) and cut p95 (+12 % / +26 %) regressed; clip B tails improved | [series 4 P2](evaluation/series4_20260912/P2-arm-screen/README.md) |

Note on the allocator arms: `STARK_EXPERIMENT_MLX_CACHE_MB` was applied process-wide by the Gemma loader when
those arms ran (Gemma loads last), so they were real screens; PR #214 only made the STT loaders and workers
consistent with it.

## What is known about where the time goes

- Silence-final medians on the promoted runtime: fixed 0.5 s silence trigger + Parakeet ≈ 0.4 s + Gemma E4B
  ≈ 0.75–1.0 s for Gemma-routed finals (14–19 tokens at 31–35 tok/s); Marian-routed finals are already
  sub-second at the median ([L1 + correction](evaluation/overnight_20260911/L1-silence-stages/README.md),
  [tail screen](evaluation/followup_20260911/X-tail-screen/README.md)).
- The first translated tokens reach the wire about 1.1–1.2 s after speech end (p95 ≈ 1.3 s), versus 1.6–2.0 s
  for the payload-ready number the goal measures ([L-C](evaluation/series3_20260912/LC-first-token/README.md)).
- Control tails are translation-dominated (worst decile: Gemma call > 800 ms in 95–100 %, STT in 14 %); ~86 partial
  STT calls per 360 s run start while a Gemma translation is active.

## Attribution answers (series 3 L-A)

Answered by [series 3 L-A](evaluation/series3_20260912/STATUS.md) with `tools/stt_overlap_attribution.py` on traced
control runs: how much of slow Gemma decode overlaps partial STT; how much of slow final STT overlaps another
chunk's decode; isolated vs live Gemma tokens/s; process CPU during decode with Marian previews; the share and
cost of smart/hard-cut finals. Arms ran only where the mechanism was measured; all three admitted arms were rejected (row above). Every hypothesis that keeps the model, the token count and the goal's metric is now closed; the remaining levers are a smaller/faster final model with quality review, fewer output tokens, or restating the goal on the first-token measure. Series 4 (2026-09-12) added the output-identical P1 runtime fixes (first-token p50 −8…−11 %, final STT −13…−23 %, merged) and closed partial reuse (text guard) and a semantic end-of-utterance classifier (no separation); the confidence-preserving reuse variant is the one open branch, contingent on a human judgement of the listed text differences.

## Harness controls and attribution

`STARK_EXPERIMENT_PARTIAL_RECHECK_TRANSLATION` defaults to `false` and accepts
`true`, `false`, `1`, or `0`. When enabled, partial workers recheck translation
activity before physical STT and count `partial_suppressed_translation_running`
with the utterance identity. Queued work suppressed here emits no preview.
`STARK_EXPERIMENT_MLX_CACHE_MB` now reaches the STT loaders and both spawned
workers consistently; its default is still 256 MiB. No default is promoted.

With `STARK_EXPERIMENT_TRACE=true`, physical STT and final translation records
include calling-thread `cpu_ms` alongside wall time. This excludes GPU work and
CPU used by other native threads; multiprocess wrappers measure parent IPC work.
`first_stream_token` marks the first callback batch per chunk, not a browser ACK
or necessarily token 1. `tools/stt_overlap_attribution.py --diagnostics DIAG.jsonl
--output report.json --markdown report.md` reports union overlap durations,
slow/normal buckets, and endpoint cohorts. Repeat `--diagnostics` for comparable
runs; truncated traces undercount overlap, and overlap alone is not causation.

Clip preparation accepts `--prepare --source NAME.wav --clip-key KEY` to write a
single-entry `manifest_KEY.json`. Keys allow letters, digits, `_`, and `-`.
The Spanish fixture retains its zero-offset special case.
