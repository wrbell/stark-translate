# Issue closure audit — #131–#138, #176, #177

Independent evidence audit for an authorized overnight delivery. This document
does not close, comment, push, or open GitHub objects. Root owns GitHub
actions after integration.

| Field | Value |
|---|---|
| Audit date | 2026-09-09 |
| Auditor role | Isolated worktree evidence reviewer (no production edits) |
| Local HEAD | `5154fb9d6a4cebae7918d063f48df39814c6bc12` (`codex/overnight-issue-evidence`, message: `docs(delivery): record validated Mac artifacts and installed runtime evidence`) |
| `origin/main` | `09e4679a041fa9825d397436496d6f30cd49aa08` (message: `chore: drop accidentally committed stark_data/replay symlink; ignore replay clips (#191)`) |
| Relationship | `origin/main` is an ancestor of local HEAD. Local is **60 commits ahead**; `origin/main` has **0** commits not in HEAD. |
| Remote containing local HEAD | none observed (`git branch -r --contains 5154fb9` empty) |
| Versions | `origin/main` `2026.13.0.0`; local HEAD `2026.14.0.0` candidate |
| Publication | Pending by user choice. Not treated as a closure substitute. |
| Open on GitHub at audit time | #131, #132, #133, #134, #135, #136, #137, #138, #176, #177 |

GitHub objects cited below were read via `gh` (issues, comments, merged PR
metadata). Code citations use GitHub blob/commit URLs only when the SHA is on
`origin/main`. Local-only SHAs are labeled **unpushed**.

## How to read this audit

- **Original acceptance** is the issue body's Do / Done-when text. Later
  comments are evidence or scope notes, not a license to weaken the body.
- **Merged `origin/main`** is what GitHub `main` actually contains.
- **Local HEAD** is the Mac reliability/v2026.14 candidate branch. It is
  implementation, not a merge.
- **Automated fixtures** are tests and synthetic reports. They are not live
  mic, church hardware, second physical output, two-speaker audio, bilingual
  approval, WSL training, or CUDA box evidence.
- **Unsuccessful bounded experiment** is recorded work that failed a gate.
  **Unperformed work** is a missing original step.
- Closing by inventing substitute issues, or by treating a later narrower
  note as the original Done-when, is out of scope.

### Suggested overall close policy

| Issue | Close now on `origin/main`? | Close after merging local HEAD? | Notes |
|---|---|---|---|
| #176 multiprocess | No. Bug still on `main`. | **Yes**, as option (a) implemented. | Add tests; do not block original AC on tests that the issue did not require. |
| #177 MTP | **Conditional yes** as experimented-off. | Same MTP core already on `main`. | Speed gate **did not pass**. Live `--mts` still uses `mlx_lm.load`. |
| #131 startup | No. | **No** without a live-mic utterance. | Mac operator/UI/partials/finals exist locally; original Do asked for live mic. CUDA is `and/or`. |
| #134 rehearsal | No (note not on `main`). | **No** until time-to-first-caption is written. | Laptop stand-in is allowed. Hymn + speech captured. |
| #132 TTS | **No** against original Do (virtual cables). Code/tests are on `main`. | Still needs a second output / virtual-cable check. | Do not treat mocked devices as independent EN/ES routing. |
| #137 correction loop | No. Review UI not on `main`. | **No**. Dated notes ≠ approved caption pairs; no Sunday loop. | Retrain script is documented. |
| #135 W16 + v2-cpo | No. Blocked: WSL / live A/B. | No. | Default E4B is not a documented no-ship decision. |
| #136 Jacobo CPO | No. Blocked: WSL CPO continue. | No. | Church prompt is a precursor, not the AC. Jacobo still fails default E4B. |
| #138 Hindi | No. | **No**. Offline text probe ≠ live pipeline / church audio. | |
| #133 diarization | No. Code on `main`; gate not run. | No. | Two-speaker + p95 +50 ms is the original Done-when. |

## Merge topology and sources

`origin/main` already contains v2026.13 PRs **#180–#191**, including:

| PR | Merge commit on `origin/main` | Relevance |
|---|---|---|
| [#183](https://github.com/wrbell/stark-translate/pull/183) | [`1e5a630`](https://github.com/wrbell/stark-translate/commit/1e5a6300b3dcf39d0c9670194c26389fecb72939) | MTP wrapper `engines/mlx_spec.py` |
| [#188](https://github.com/wrbell/stark-translate/pull/188) | [`d9fab13`](https://github.com/wrbell/stark-translate/commit/d9fab1398a3ddfe24f8a5ca05904c059fbfcba24) | Per-language TTS routing |
| [#189](https://github.com/wrbell/stark-translate/pull/189) | [`c59d761`](https://github.com/wrbell/stark-translate/commit/c59d761cbb1deae86aabf8434ed226757aa05f54) | Opt-in live diarization |
| [#190](https://github.com/wrbell/stark-translate/pull/190) | [`8b0c36d`](https://github.com/wrbell/stark-translate/commit/8b0c36d) | Parakeet EN default, Marian CT2 Mac, `docs/archive/v2026.13/MAC_LATENCY.md` |
| [#191](https://github.com/wrbell/stark-translate/pull/191) | [`09e4679`](https://github.com/wrbell/stark-translate/commit/09e4679a041fa9825d397436496d6f30cd49aa08) | Replay symlink hygiene |

Local-only (not on `origin/main`, not observed on any remote): operator Review,
session lifecycle, evaluation pack (`docs/evaluation/mac_v2026_14_*`),
rehearsal/Hindi/installation reports, `workers.py` Gemma-4 routing
(`f58f54aba668688194a8c8f12dabcf91a25441f8`), packaged VAD/CT2 setup, and
related tests.

Primary local status write-up: [`docs/mac_implementation_status.md`](mac_implementation_status.md).
Evaluation index: [`docs/evaluation/README.md`](evaluation/README.md).

### Auditor CPU checks (this worktree)

Existing `stt_env` was used **read-only** as an interpreter. No installs, no
GPU, no playback, no original-data writes.

| Command | Result |
|---|---|
| Inspect `workers.translation_worker_main` | Loads; uses `MLXGemmaEngine`; `model_family` kwarg present; default still `"translategemma"`; STT worker still `mlx_whisper` only |
| `pytest` MTP/TTS/factory/settings/imports/preference-triples | **137 passed** |
| `pytest` review / live-diarize / Parakeet dispatch / correction-import | **67 passed** |

System `/usr/bin/python3` lacks FastAPI; those TTS API tests failed there and
passed under `stt_env`. That is an auditor-interpreter limit, not a new product
finding.

---

## #176 — `--multiprocess` TranslateGemma prompts

- GitHub: https://github.com/wrbell/stark-translate/issues/176
- State at audit: OPEN, no comments
- Labels: `bug`, `pipeline`
- Opened 2026-09-09

### Original acceptance

Issue body (no separate Done-when heading):

1. `workers.py` hardcodes the TranslateGemma structured chat template
   (~L245–290) and unconditionally builds a prompt cache.
2. Since PR #171, Mac default is Gemma 4 OptiQ E4B, so `--multiprocess`
   would feed Gemma 4 a TranslateGemma-shaped prompt.
3. Option **(a)**: route `workers.py` through
   `engines/translation_prompts.py::build_chat_messages` + `ensure_stop_tokens`
   and skip the prompt cache for gemma4.
4. Option **(b)**: deprecate `--multiprocess` now that MLX ≥ 0.31.2 thread-local
   streams and in-process `max_workers=2` are the production path (PR #168).
5. Low priority; not on the latency critical path.

### Observed evidence

**`origin/main` still has the bug.**
[`workers.py`](https://github.com/wrbell/stark-translate/blob/09e4679a041fa9825d397436496d6f30cd49aa08/workers.py)
last changed on `main` in
[`9e87715`](https://github.com/wrbell/stark-translate/commit/9e877159f255785358f380b49a2c98610eb59f19).
`translation_worker_main` still:

- documents TranslateGemma 4B/12B + prompt cache
- calls `_fix_eos` / `_build_prompt_cache` / `_translate_4b`
- builds `source_lang_code` / `target_lang_code` messages (lines 189–220,
  245–269, 303+)

`origin/main` `dry_run_ab.py` `_start_workers` passes only
`source_lang` / `target_lang` into the translation worker — no `model_family`.

**Local HEAD implements option (a), unpushed.**
Commit `f58f54aba668688194a8c8f12dabcf91a25441f8`
(`feat(mac): trace capture latency and share inference paths`) rewrites
`translation_worker_main` to construct `MLXGemmaEngine(..., model_family=model_family)`
and states that Gemma 4 never receives TG prompts. Prompt cache is not built in
the worker; `MLXGemmaEngine.load` only builds it when
`model_family == "translategemma"`. Parent `_start_workers` now passes
`model_family=MODEL_FAMILY` plus adapter and terminology kwargs
(`dry_run_ab.py` ~1500–1509).

`--multiprocess` is **not** deprecated (option b unused). Flag remains
(`dry_run_ab.py` `--multiprocess`; settings default false).

### Required additional evidence

1. Merge `f58f54a` (or equivalent) to `origin/main`. Until then GitHub `main`
   still ships the original bug.
2. Optional but important for root (not in the issue body): a CPU test that
   `_start_workers` forwards `model_family="gemma4"` and that the worker no
   longer contains `source_lang_code` message construction. **No test file
   imports `workers` today.**
3. Residual (not original AC): STT worker still loads `mlx-whisper` only;
   English production STT is Parakeet. Worker `model_family` default remains
   `"translategemma"` if a caller omits the kwarg.

### Suggested GitHub wording

**Do not close on current `origin/main`.**

After the worker rewrite is on `main`:

> Closing #176: option (a) landed. `workers.translation_worker_main` now
> constructs `MLXGemmaEngine` with the parent `model_family` and no longer
> builds TranslateGemma `source_lang_code` prompts or an unconditional Gemma-4
> prompt cache (`<merge SHA>`). `--multiprocess` remains an escape hatch, off
> by default; production overlap is still in-process `max_workers=2`. Not
> verified by a live `--multiprocess` GPU run in this audit.

---

## #177 — Gemma 4 MTP / MTS bounded experiment

- GitHub: https://github.com/wrbell/stark-translate/issues/177
- State at audit: OPEN
- Comment: https://github.com/wrbell/stark-translate/issues/177#issuecomment-5608923953
- Opened 2026-09-09

### Original acceptance

**Problem.** `--mts` / `STARK_TRANSLATE_MLX_MTS` has never actually run.
mlx-lm 0.31.3 has no `gemma4_assistant`; `load_mlx_gemma(MLX_DRAFT_MODEL_ID)`
raises; the error is swallowed at `dry_run_ab.py` ~L1218;
`tools/benchmark_mlx_accel.py` `e4b_mts` records `LOAD FAIL`.

**Plan.** Use mlx-optiq `optiq.runtime.spec`; new `engines/mlx_spec.py`;
route `MLXGemmaEngine.translate_streaming` and
`dry_run_ab.translate_mlx_streaming` through it; drop non-streaming MTS
fallback; gamma sweep + byte-identical greedy check; probe
`tools/mts_acceptance_probe.py`.

**Gate.** Identical output vs greedy; canary ≥ 7/8; medium p50 ≤ 0.85× the
post-EOS-fix E4B; acceptance ≥ 30% (llama.cpp 70–87%, so lower Metal
acceptance is an implementation issue to chase, **time-boxed**).
**Otherwise `mlx_mts` stays off.** Drafter:
`mlx-community/gemma-4-e4b-it-assistant-bf16`.

### Observed evidence

**Wrapper and probe are on `origin/main`** via
[#183](https://github.com/wrbell/stark-translate/pull/183)
([`1e5a630`](https://github.com/wrbell/stark-translate/commit/1e5a6300b3dcf39d0c9670194c26389fecb72939)):
[`engines/mlx_spec.py`](https://github.com/wrbell/stark-translate/blob/09e4679a041fa9825d397436496d6f30cd49aa08/engines/mlx_spec.py),
[`tools/mts_acceptance_probe.py`](https://github.com/wrbell/stark-translate/blob/09e4679a041fa9825d397436496d6f30cd49aa08/tools/mts_acceptance_probe.py),
mocked [`tests/test_mlx_spec.py`](https://github.com/wrbell/stark-translate/blob/09e4679a041fa9825d397436496d6f30cd49aa08/tests/test_mlx_spec.py).
Default `settings.translation.mlx_mts` is **False** on both trees.

**Bounded GPU probe was performed** (issue comment +
[`docs/archive/v2026.13/MAC_LATENCY.md`](https://github.com/wrbell/stark-translate/blob/09e4679a041fa9825d397436496d6f30cd49aa08/docs/archive/v2026.13/MAC_LATENCY.md)
§3, shipped in #190). E4B OptiQ + assistant-bf16, mlx-optiq 0.4.34, 3 runs,
8 canaries + 3 sentences:

| γ | p50 | tok/s | acceptance | vs greedy short/medium/long |
|---|---:|---:|---:|---|
| 1 | 655 ms | 20.9 | 31.3% | 453 / 1339 / 1440 ms |
| 2 | 645 ms | 21.1 | 21.6% | |
| 3 | 670 ms | 20.3 | 15.2% | |

Post-EOS-fix greedy medium p50 in the same doc is **1393 ms**.
1339 / 1393 ≈ **0.961**, not ≤ 0.85. Long −14%; short ≈ 0.
Byte-identical: comment 11/11; MAC_LATENCY 33/33 (11 items × 3 runs).
Acceptance ≥ 30% holds only at γ=1.

**Canary ≥ 7/8 was not published for the probe.** Byte-identical-to-greedy
implies the same misses as greedy E4B. The 8-slice health_check on Mac recon
was **6/8** (Santiago / partimiento). E4B `none` first eight 18-canary rows in
the local quality report are also 6/8 (canary_01 James, canary_03 partimiento).
This audit does **not** treat ≥ 7/8 as demonstrated.

**Time-boxed RoPE chase was performed and rejected** (MAC_LATENCY §3;
[`docs/mlx_mtp_notes.md`](https://github.com/wrbell/stark-translate/blob/09e4679a041fa9825d397436496d6f30cd49aa08/docs/mlx_mtp_notes.md)
§7 suspect 1): acceptance unchanged 31.3% / 21.6%; latency slightly worse.
Further suspects (bf16 target, batched-verify) are named, not run. That is
extra research, not a completed original gate.

**Live `--mts` still uses the failing loader** on both `origin/main` and local
HEAD (`dry_run_ab.py` ~1253–1259): `load_mlx_gemma(MLX_DRAFT_MODEL_ID)` inside
try/except, `WARNING: MTS drafter load failed`.
`MLXGemmaEngine` still `mlx_load`s `draft_model_id` and passes `draft_model=`
to `mlx_lm.generate`. **`spec_stream` is not called from `mlx_engine.py` or
`dry_run_ab.py`.** `docs/mlx_mtp_notes.md` still says integration into those
modules is a separate change. The original “`--mts` has never loaded” symptom
remains true for the CLI flag. The working path is the **probe**, not live
finals.

Auditor CPU: `tests/test_mlx_spec.py` and `tests/test_mlx_gemma4_factory.py`
passed (mocked; `mlx_mts is False` by default).

### Gate vs original numbers (no speed-gate claim)

| Original gate | Result |
|---|---|
| Identical vs greedy | Recorded pass (11/11 or 33/33) |
| Canary ≥ 7/8 | Not shown; likely fail if identical to greedy 6/8 |
| Medium p50 ≤ 0.85× 1393 ms (≤ ~1184 ms) | **Fail** (1339 ms) |
| Acceptance ≥ 30% | Pass at γ=1 only (31.3%) |
| Otherwise stay off | **Honored** (`mlx_mts` default false; screening/status keep MTP off) |

This is an unsuccessful bounded experiment, not unperformed work.

### Required additional evidence

None required to **decline shipping MTP**. To **close** under the original
“otherwise stays off” clause, root should state that explicitly and must not
claim the 0.85× gate.

If root instead keeps #177 open, the remaining original-plan item is wiring
`spec_stream` into live translate (or making `--mts` fail closed / documented
no-op). Do not open a substitute “MTP speed” issue to keep a failed gate
alive.

### Suggested GitHub wording

> Closing #177 as a completed bounded experiment that did **not** pass the
> speed gate. `engines/mlx_spec.py` and `tools/mts_acceptance_probe.py` landed
> in #183. On M3 Pro, γ=1 was byte-identical to greedy with 31.3% acceptance;
> medium p50 1339 ms vs 1393 ms greedy (0.96×, not ≤0.85×). RoPE-offset
> suspect rejected. `mlx_mts` remains default off. No caption-delivery or
> 0.85× claim. Residual: live `--mts` still calls `mlx_lm.load` and swallows
> `gemma4_assistant` failure; production path does not use `spec_stream`.
> Further Metal-acceptance research is new work, not this issue’s ship gate.

---

## #131 — Post-internship recon / operator smoke

- GitHub: https://github.com/wrbell/stark-translate/issues/131
- State at audit: OPEN
- Comments: 2026-08-14 recon; 2026-09-09 Mac recon; 2026-09-09 v2026.13 update

### Original acceptance

**Do**

1. Fresh `stark-translate setup` on Mac **and/or** CUDA box
2. `stark-translate operator` → pre-flight gates all green
3. `--dry-run-text` plus **one live mic utterance** EN→ES **and** ES→EN
4. Confirm W16 CT2 adapter still preferred by factory
   (`adapters/whisper_turbo_ct2/active/`)
5. Note any model-download, llama-server, or display breakage

**Done when**

Operator UI loads, one partial + one final appear on audience display, no
uncaught errors in the session log.

### Observed evidence

**Comments (not a close).**
2026-08-14 cloud pytest on `343e3c2`: 1091 passed / 3 skipped; live mic still
needed. Security hygiene #139. 2026-09-09 Mac recon on `main` @ `3ce16a4`:
operator `/operator/` and `/api/preflight` 200; health_check 6/8;
`--dry-run-text` EN and ES OK; left open for CUDA/church-PC.
https://github.com/wrbell/stark-translate/issues/131#issuecomment-5608300630
Later: Mac stack produces finals via replay #182; remaining named as CUDA
pre-flight.
https://github.com/wrbell/stark-translate/issues/131#issuecomment-5609517155

**`origin/main`:** operator app, preflight, displays, W16 CT2 autodetection
in [`engines/factory.py`](https://github.com/wrbell/stark-translate/blob/09e4679a041fa9825d397436496d6f30cd49aa08/engines/factory.py)
(`_WHISPER_CT2_ACTIVE_PATH`, tests in `tests/test_factory_stt_backend.py`).
No v2026.14 rehearsal/install pack on `main`.

**Local HEAD (unpushed) Mac evidence — not live mic:**

- Installed operator surfaces: [`docs/evaluation/mac_v2026_14_installation.md`](evaluation/mac_v2026_14_installation.md)
  — `/healthz`, `/operator/`, review script 200; **“No browser, microphone or
  physical playback was used.”** Synthetic EN/ES installed inference exit 0
  (file text → STT → E4B → Piper WAV).
- Controlled operator rehearsal: [`docs/evaluation/mac_v2026_14_rehearsal.md`](evaluation/mac_v2026_14_rehearsal.md)
  and [`mac_v2026_14_rehearsal_report.json`](evaluation/mac_v2026_14_rehearsal_report.json)
  — three sessions exit 0; audience reconnect; visible caption ACKs (8/9, 1/1,
  8/8 unique finals); operator UI; pause/resume; EN→ES→EN. Inputs are
  historical/synthetic **files**, not a microphone.
- Factory W16 preference: CPU tests passed here with a fake `model.bin`.
  `adapters/whisper_turbo_ct2/active` is **absent in this worktree**. Mac
  English default is Parakeet (`docs/mlx_cuda_parity.md`: W16 CT2 is CUDA-only).
- CUDA / llama-server / church-PC operator: **not evidenced**.
  `docs/cuda_latency_proposal.md` remains a proposal.

### Required additional evidence

1. One **live microphone** utterance EN→ES and one ES→EN with operator UI +
   audience partial and final and a session log without uncaught errors
   (original Do; still called out in 2026-08-14 comments).
2. Fresh setup on the CUDA box **only if** root treats “and/or” as requiring
   both. The body allows Mac alone for setup. Later comments expanded CUDA as
   remaining work; that expansion is not the original Done-when.
3. Live confirm that CUDA factory prefers `adapters/whisper_turbo_ct2/active/`
   when that directory exists on the A2000. Mac Parakeet default does not
   satisfy the W16 bullet.

Do not close on dry-run-text, replay, or installed-WAV sessions.

### Suggested GitHub wording

**Keep open.**

> Audit 2026-09-09: Mac operator UI, preflight HTTP 200, `--dry-run-text`, and
> a three-session file rehearsal with audience partials/finals are recorded on
> the v2026.14 candidate (`docs/evaluation/mac_v2026_14_rehearsal.md`), not yet
> on `origin/main`. Original Do still missing: live mic EN→ES and ES→EN.
> W16 CT2 preference is unit-tested; the adapter is not in this worktree and
> was not confirmed on CUDA. CUDA operator pre-flight not run. Not closing.

---

## #134 — Sunday dry-run with the operator runbook

- GitHub: https://github.com/wrbell/stark-translate/issues/134
- State at audit: OPEN

### Original acceptance

**Do**

1. Walk the runbook on church hardware **or a laptop stand-in**
2. Time setup → first caption; note operator-only vs helper-needed steps
3. Capture one full hymn + one spoken segment
4. File follow-ups for anything a non-technical operator would get stuck on

**Done when**

A written dry-run note exists (what worked, what broke, **time-to-first-caption**)
and any blocking UX holes have their own issues.

### Observed evidence

**`origin/main`:** runbook exists
([`docs/operator_runbook.md`](https://github.com/wrbell/stark-translate/blob/09e4679a041fa9825d397436496d6f30cd49aa08/docs/operator_runbook.md)).
No rehearsal report. Comments say the dry-run still needs a human; v2026.13
latency is a prerequisite, not the dry-run itself.

**Local HEAD (unpushed):** written note
[`docs/evaluation/mac_v2026_14_rehearsal.md`](evaluation/mac_v2026_14_rehearsal.md)
+ [`mac_v2026_14_rehearsal_report.json`](evaluation/mac_v2026_14_rehearsal_report.json).
Laptop Mac pipeline, 2026-09-09 America/Detroit.

| Session | Input | Finals | Notes |
|---|---|---:|---|
| `20260909_210408_972943_en` | Historical English hymn/speech + inserted silence; paused | 9 | Hymn + spoken segment |
| `20260909_210738_020946_es` | Synthetic Spanish | 1 | Language restart |
| `20260909_211201_787725_en` | Synthetic John 3:16 + historical English speech | 8 | Review / Stop / summary |

What worked (driver + file checks): startup held `starting` until production
header; pause/resume; new session identity on language flip; audience
reconnect/history reset; visible ACKs; verse from production CSV; live
review; per-language TTS preference persistence; normal Stop drained.

What broke: premature readiness, small-sample p95, stale language/TTS form
(fixed later in the local branch); summary first-run format failure retained
at `.cache/mac-roadmap/summary_format_failure_20260909_211201_787725_en.json`;
chunk 1 of the hymn session lacks a visible final ACK (missing evidence, not a
display-fail verdict).

**Time-to-first-caption is not recorded.** Session start/end timestamps exist;
no setup→first-caption duration; no operator-only vs helper-needed list.
No GitHub follow-up issues were filed for remaining holes; pending items live
in the rehearsal report (`pending_gates`) and
[`docs/mac_implementation_status.md`](mac_implementation_status.md)
“Explicit pending gates.” Those are not “their own issues” on GitHub.

This rehearsal is explicitly **not** a church-hardware Sunday and **not** a
latency acceptance run.

### Required additional evidence

1. Merge the rehearsal documents if they are to count as the written note on
   `main`.
2. A numeric **setup → first caption** (and, as originally asked,
   operator-only vs helper-needed).
3. GitHub issues for any remaining **blocking** non-technical UX holes, or a
   written statement that remaining items are non-blocking (human/device
   quality gates already tracked as #132/#133/#138 etc. — do not duplicate to
   game closure).
4. Church hardware is **not** required by the original “or laptop stand-in”
   clause.

### Suggested GitHub wording

**Keep open** until TTFC is in the written note.

> Laptop stand-in rehearsal is documented on the v2026.14 candidate
> (`docs/evaluation/mac_v2026_14_rehearsal.md`): hymn + speech, what worked /
> what broke, audience captions. Original Done-when still missing:
> time-to-first-caption. Remaining device/human gates are listed in that note
> and `docs/mac_implementation_status.md`; they are not filed as follow-up
> issues. Not a church Sunday. Not closing.

---

## #132 — 9.4.1 Multi-channel TTS routing

- GitHub: https://github.com/wrbell/stark-translate/issues/132
- State at audit: OPEN
- Comment after #188: remaining hands-on virtual cable / second output
  https://github.com/wrbell/stark-translate/issues/132#issuecomment-5609268563

### Original acceptance

**Do**

1. Add output-device support to `PiperTTSEngine`
2. Wire `--tts-output local` (and multi-channel routing via AudioFetch /
   **virtual cables**)
3. Enable the operator UI dropdown for real devices

**Done when**

Operator can send TTS to a chosen output device; EN and ES can be routed
independently. Tests cover the new engine path.

### Observed evidence

**On `origin/main` via [#188](https://github.com/wrbell/stark-translate/pull/188)
([`d9fab13`](https://github.com/wrbell/stark-translate/commit/d9fab1398a3ddfe24f8a5ca05904c059fbfcba24)):**

- `settings.tts.output_devices`
- `--tts-device-en` / `--tts-device-es`
- `engines/audio_devices.py` name match + hotplug re-resolve
- `/api/audio/output-devices`
- operator selects `#output-device-en` / `#output-device-es`
- [`tests/test_tts_multichannel.py`](https://github.com/wrbell/stark-translate/blob/09e4679a041fa9825d397436496d6f30cd49aa08/tests/test_tts_multichannel.py)
  (mocked devices)

Auditor: 137-test batch including `test_tts_multichannel.py` and
`test_phase9_4_1_tts_device.py` passed under `stt_env`.

**Device/human evidence is not independent routing.**

Local rehearsal TTS mode was **`wav`** (synthesis files, no playback-request
boundary). Separate builtin smoke (rehearsal.md / report
`tts_builtin_device_smoke`): both EN and ES played to **MacBook Pro Speakers**.
Other enumerated endpoint: Microsoft Teams Audio, not a second physical
output. **No unplug/replug. No virtual cable. No AudioFetch.**
Installation evidence: no physical playback.

### Required additional evidence

Original Do’s virtual-cable / independent EN vs ES output is **unperformed**,
not a failed experiment.

1. Choose two distinct outputs (e.g. built-in + BlackHole / transmitter).
2. Route EN and ES independently and confirm audible (or host-API) destinations.
3. Optional hotplug re-resolve, as implemented in code but not exercised.

Do not close by treating mocked `list_output_devices` or same-speaker dual
language smoke as independent routing.

### Suggested GitHub wording

**Keep open.**

> #188 merged to `main`: per-language device map, CLI, operator dropdowns, and
> engine-path tests. Hands-on independent routing is still missing (virtual
> cable / second output). Rehearsal only exercised MacBook Pro Speakers for
> both languages and WAV synthesis. Original Do is not complete. Not closing.

---

## #137 — Active learning / operator correction

- GitHub: https://github.com/wrbell/stark-translate/issues/137
- State at audit: OPEN
- Comment: blocked WSL/human for retrain half
  https://github.com/wrbell/stark-translate/issues/137#issuecomment-5608289789

### Original acceptance

**Do**

1. Flag low-confidence STT / translation in the operator UI
2. Correction workflow that writes training-shaped JSONL
3. One closed loop on a **recorded Sunday** (correct → merge → smoke retrain)

**Done when**

An operator can correct a caption and that **pair** lands in a **dated
corrections corpus**. Retrain script documented even if the first retrain is
a dry run.

### Observed evidence

**`origin/main`:** CLI tools exist (`tools/prepare_finetune_data.py`
`extract-review-queue`, `tools/merge_corrections.py`) and are documented in
[`tools/CLAUDE.md`](https://github.com/wrbell/stark-translate/blob/09e4679a041fa9825d397436496d6f30cd49aa08/tools/CLAUDE.md)
and the runbook. **No** `operator_app/review.py`, `tools/review_data.py`,
operator Review UI, or `tests/test_operator_review.py` on `main`. Flagging in
the live operator UI is not the v2026.14 Review surface.

**Local HEAD (unpushed):** Review API/UI/store
(`operator_app/review.py`, `displays/operator/review.js`,
`tools/review_data.py`) write dated sidecars
`stark_data/corrections/<session>.jsonl`. Default list filter
`flagged_only` uses `review_priority >= 1`. Tests
(`tests/test_operator_review.py`, 26 collected, passed here) cover revisions,
independent transcript/translation approvals, portable bundle roundtrip,
merge contracts — **fixtures**, including fabricated approvals in tmp dirs.

Rehearsal used that UI: notes in
`stark_data/corrections/20260909_210408_972943_en.jsonl` and
`..._211201_787725_en.jsonl`. Report states both approval flags remain
**false**; notes are rehearsal/draft-recovery text, **not** human-approved
caption pairs; **no** transcript/translation approval, training import, or
retrain was performed. Pending gate: “Approved portable training handoff from
real live data was not rehearsed; only automated fixtures exercised it.”

Retrain/merge commands are documented (runbook + `tools/CLAUDE.md` +
`docs/wsl_pipeline_refresh.md`). A dry-run retrain of a **real corrected
pair** is not recorded. Sunday closed loop is unperformed. WSL smoke retrain
is unperformed.

### Required additional evidence

1. Merge Review/export to `main` if GitHub `main` is the close basis.
2. Operator correction of a real caption to an approved **pair** in a dated
   `stark_data/corrections/` JSONL (not a rehearsal note).
3. Documented merge dry run into a training-shaped JSONL (CLI already exists).
4. Original Do’s Sunday loop: correct → merge → smoke retrain, even if the
   retrain is a documented dry run. Fixtures cannot replace that.

### Suggested GitHub wording

**Keep open.**

> Operator Review on the v2026.14 candidate can persist dated sidecars and
> reject unapproved exports; automated merge/export tests pass. Rehearsal
> notes are unapproved and are not caption pairs. No recorded-Sunday
> correct→merge→retrain loop. Retrain commands are documented. Not closing.

---

## #135 — Deploy W16 + v2-cpo to Mac and live A/B

- GitHub: https://github.com/wrbell/stark-translate/issues/135
- State at audit: OPEN
- Comment: re-scoped, blocked WSL
  https://github.com/wrbell/stark-translate/issues/135#issuecomment-5608289398

### Original acceptance

**Do**

1. Copy W16 CT2 + v2-cpo GGUF/adapters onto the M3 Pro
2. Live A/B: stock E4B vs v2-cpo, off-the-shelf turbo vs W16
3. 5-canary health check + theological term audit (`tools/health_check.py`)
4. Decide ship/no-ship for v2-cpo (latency win vs Jacobo miss)

**Done when**

A short A/B note with canary scores and a ship decision. If no-ship, stock
E4B stays default and this closes.

### Observed evidence

No A/B note of W16 vs turbo or v2-cpo vs stock E4B on Mac exists in
`docs/evaluation/` or issue comments.

Local/default Mac path: Parakeet EN + stock Gemma 4 E4B OptiQ; Spanish
whisper-turbo. `docs/mlx_cuda_parity.md` (on `main`): W16 LoRA→CT2 is
**CUDA-only**; mlx-whisper has no LoRA/CT2 load. `adapters/whisper_turbo_ct2/active`
absent here. `--adapter-dir` is wired, not used for a v2-cpo live A/B.

Stock E4B remaining default is a **status**, not a documented no-ship
decision from the required A/B. Closing as no-ship without canary scores
would weaken the Done-when.

Quality comparison (local, text, not W16/v2-cpo):
[`docs/evaluation/mac_v2026_14_quality/comparison.md`](evaluation/mac_v2026_14_quality/comparison.md)
E4B vs E2B, church vs none prompts — different experiment.

### Required additional evidence

WSL/Mac conversion and the live A/B (or a written no-ship with canary scores
from that A/B). Blocked on WSL as the 2026-09-09 comment states. Do not close
because Parakeet replaced the Mac STT question; that substitution is not the
original W16 A/B.

### Suggested GitHub wording

**Keep open.**

> No W16 vs turbo or v2-cpo vs stock E4B live A/B note exists. Stock E4B
> remains default without a ship/no-ship record. Mac STT default is now
> Parakeet; W16 CT2 is still CUDA-only. Blocked on WSL/adapter conversion.
> Not closing.

---

## #136 — Jacobo canary preference triples

- GitHub: https://github.com/wrbell/stark-translate/issues/136
- State at audit: OPEN
- Comment: blocked WSL; cheap precursor = few-shot prompt
  https://github.com/wrbell/stark-translate/issues/136#issuecomment-5608289613

### Original acceptance

**Do**

1. Hand-craft 50–100 preference triples for failing canaries / glossary
   disambiguation
2. One short CPO continue from v2 (`training/train_gemma4_cpo.py --init-adapter`)
3. Re-score the 8-canary set + 500-verse holdout

**Done when**

Jacobo canary **passes** (or we document why it cannot) **and** COMET-22 does
not regress vs v2-cpo.

### Observed evidence

`training/train_gemma4_cpo.py --init-adapter` and
`tools/build_preference_triples.py` exist (CPU tests for the triples tool
passed here). **No** 50–100 hand-crafted triples run, **no** CPO continue,
**no** 8-canary + 500-verse re-score vs v2-cpo, **no** COMET-22 comparison.

Local precursor (not the AC): `terminology_prompt="church"` in
`engines/translation_prompts.py` (“Jacobo for James the person and Santiago
for the epistle”). Quality report E4B **church** canary_01 output is still
“**Santiago** escribió sobre fe y obras.” (required term `santiago` passes
that row’s check; the **Jacobo** requirement of this issue does not).
E2B church emitted Jacobo and therefore **missed** the `santiago` canary
term. Default E4B `none` still emits untranslated “James”.

There is **no** document that Jacobo cannot be fixed. The issue remains a
training/CPO task.

### Required additional evidence

The CPO continue and the two scores in the Done-when, or an explicit “cannot”
write-up plus COMET-22 vs v2-cpo. Prompt-side church examples are not a
substitute.

### Suggested GitHub wording

**Keep open.**

> No preference-triple CPO continue or COMET-22 vs v2-cpo. Default E4B still
> does not emit Jacobo for “James wrote about faith and works” (church prompt:
> Santiago; none: James). `train_gemma4_cpo.py --init-adapter` is unimplemented
> work, not a failed experiment. Blocked on WSL. Not closing.

---

## #138 — Hindi zero-shot baseline

- GitHub: https://github.com/wrbell/stark-translate/issues/138
- State at audit: OPEN

### Original acceptance

**Do**

1. `target_lang_code="hi"` **through the live pipeline**
2. Note SOV partial-garble (plan: Hindi→EN partial + Hindi final)
3. 8-canary + a handful of verse pairs, chrF++/COMET if cheap

**Done when**

A short baseline note: does zero-shot Hindi even work **on church audio**,
and is a QLoRA week worth it this semester.

### Observed evidence

**`origin/main`:** `LANG_NAMES` includes `hi`; Piper `hi_IN-kusal-medium`.
`dry_run_ab.py --lang` choices are still `["en", "es"]` only (same on local
HEAD). No live Hindi session language, no SOV partial policy in the live
pipeline.

**Local HEAD (unpushed) offline text probe:**
[`docs/evaluation/mac_v2026_14_hindi/README.md`](evaluation/mac_v2026_14_hindi/README.md)
— `quality --target hi`, 43 English **text** inputs × 3, E4B/E2B, no
references, no chrF++/COMET, Spanish canary terms disabled. Example retained
English “surety” in E4B Hindi. README: does **not** add live Hindi STT,
language switching, or TTS; “Live Hindi and Chinese remain later feature
decisions.” Blank `blind_review.jsonl`.

This is an offline generation probe, **not** church audio and **not** the live
pipeline. QLoRA-week decision is not answered from church audio.

### Required additional evidence

Live `hi` through the pipeline (or a recorded live dry-run), SOV partial
policy note, and a baseline that addresses **church audio**. chrF++/COMET
only if cheap and references exist (they do not today). Do not close on the
offline text probe.

### Suggested GitHub wording

**Keep open.**

> Offline Gemma 4 EN→HI text generations are recorded on the v2026.14
> candidate (`docs/evaluation/mac_v2026_14_hindi/`). `--lang` is still en/es
> only. No live Hindi, no church-audio baseline, no chrF++/COMET, no QLoRA
> semester decision. Not closing.

---

## #133 — 9.6.1 Live diarization on a rolling buffer

- GitHub: https://github.com/wrbell/stark-translate/issues/133
- State at audit: OPEN
- Comment after #189: gate not run
  https://github.com/wrbell/stark-translate/issues/133#issuecomment-5609516989

### Original acceptance

**Do**

1. Run diarization on a rolling live audio buffer (not post-session)
2. Attribute captions to speaker turns in operator + audience displays
3. Keep off by default (`--diarize`)

**Done when**

Two-speaker dry-run shows distinct speaker labels on finals **without blowing
p95 caption latency**. Offline path still works.

Triage comment (not a replacement AC) specified p95 within +50 ms of
post-latency-program baseline; that matches
[`docs/live_diarization.md`](https://github.com/wrbell/stark-translate/blob/09e4679a041fa9825d397436496d6f30cd49aa08/docs/live_diarization.md)
Gate section.

### Observed evidence

**Code on `origin/main` via [#189](https://github.com/wrbell/stark-translate/pull/189)
([`c59d761`](https://github.com/wrbell/stark-translate/commit/c59d761cbb1deae86aabf8434ed226757aa05f54)):**
`--diarize` default off; rolling buffer; `features/live_diarize.py`;
`speaker_labels.py`; WebSocket/CSV/audience prefix; operator checkbox.
Offline `features/diarize.py` remains. Tests
`tests/test_phase9_6_1_live_diarize.py` passed here (fake embedder / no
models).

**Gate not run.** Doc and PR both say so. Rehearsal summary has no speaker
labels. No two-speaker clip, no SpeechBrain/HF_TOKEN soak, no
`replay_bench.py` with/without `--diarize` p95. Natural two-speaker remains
an explicit pending gate in `docs/mac_implementation_status.md`.

HF_TOKEN fallback for missing pyannote was a close-out fix (#180) for a CI
vs laptop test gap; that is not the two-speaker latency gate.

### Required additional evidence

Two-speaker dry-run with distinct labels on finals, and p95 caption latency
not blown (doc: +50 ms vs `--diarize` off on the same clip). Offline path
already has tests; keep it working during that run.

### Suggested GitHub wording

**Keep open.**

> #189 merged: opt-in rolling-buffer diarization, default off, operator +
> audience labels, unit tests. Original Done-when (two-speaker dry-run + p95)
> has not been run. Not closing.

---

## Repository gaps for root (do not edit production code in this worktree)

These are observations for the implementing worktrees. This audit did not
change product code.

1. **#176 untested.** No `tests/` module imports `workers`. Default
   `model_family="translategemma"` in `translation_worker_main` is a footgun
   if the parent omits the kwarg. STT worker ignores Parakeet.
2. **#177 live flag vs probe.** `engines/mlx_spec.spec_stream` is unused by
   `MLXGemmaEngine` / `dry_run_ab` finals. `--mts` still swallows
   `mlx_lm.load` failure. If #177 is closed as experimented-off, the CLI
   should not imply a working live drafter.
3. **#134 metric hole.** Rehearsal schema has no `time_to_first_caption`
   field; adding one to a future rehearsal report is evidence, not a
   substitute GitHub issue.
4. **#138 live language.** `--lang` remains `en`/`es`; Hindi probe is
   `mac_evaluation.py quality --target hi` only.
5. **Interpreter split.** Auditor system Python lacked FastAPI; `stt_env`
   ran the same TTS API tests cleanly. Not a new product defect.

## Conclusion

Nothing in this set is closable on **today’s `origin/main`** except a
**conditional** close of **#177** as a bounded negative MTP experiment
(speed gate not met; remains off). **#176** is fixed only on the unpushed
local branch. **#188/#189** shipped TTS and diarization **code** to `main`
without the original device/two-speaker evidence.

After local HEAD is merged, the only additional issue whose **original**
acceptance is then in reach without new human/device/WSL work is **#176**.
**#134** still needs a written time-to-first-caption. **#131** still needs
live mic. **#132, #133, #135, #136, #137, #138** remain blocked on physical
routing, two-speaker latency, WSL adapters/CPO, a real correction pair +
retrain dry run, and live Hindi/church audio respectively.

Do not close any issue by swapping fixtures for those gates. Package
publication remains pending and is not a closure criterion for this set.
