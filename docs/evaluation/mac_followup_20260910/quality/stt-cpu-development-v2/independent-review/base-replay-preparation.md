# Conditional CPU Whisper-base replay preparation — 2026-09-10

This is source-only contingency design, not an implemented interface or an executed plan. The recovery state remained `stage=quality-v2` when inspected. No base implementation is authorized unless the completed 12-worker/600-row quality cohort qualifies at least one language. If both languages fail, stop at this note. Existing frozen runtime, preparation helpers, active processes and completed evidence remain immutable.

## Eligibility before implementation

Use the completed, provenance-validated CPU Whisper small/base quality recovery report and `lite-followup-preparation/prepare.py:259-355` (`validate_quality`). For each language independently, all three matched repeats must satisfy:

- base aggregate WER <= small aggregate WER + max(0.01, 0.05 × small aggregate WER);
- base glossary term recall >= small recall whenever there are term opportunities;
- exact original development recording/reference identities, 50 successful recordings per worker, identical per-model pinned identity across repeats, no fallback, and the complete serial 12-worker cohort.

Eligibility authorizes only conditional implementation/replay preparation for that language. The quality workers use whole recordings, beam 5 and no prompt; live Lite uses beam 1 and its existing prompt. Quality qualification is not a live latency result. The reviewed validator explicitly leaves `live_latency_eligible` and `p95_claim_eligible` false.

## Existing gaps and precise touchpoints

Paths below are relative to the repository. Loader/harness line references were inspected in `.cache/mac-en-es-closeout/runtime/`; that copy must not be edited.

| Source | Existing behavior | Conditional change |
| --- | --- | --- |
| `stark_translate/profiles.py:50-107` | `resolve_profile` chooses `whisper-small`; `apply_profile` overwrites `settings.stt.whisper_cuda_model`. | Add a pure, explicit replay-only profile override helper; preserve both ordinary functions and every default. |
| `dry_run_ab.py:5679`, `:5868-5910` | CLI accepts an STT backend, not an STT model. Profile policy is applied after argument parsing. | Add `--research-stt-model` with only `whisper-base` as an explicit choice and `None` default. Validate scope before lifecycle/model creation; apply the selected model after ordinary profile policy. |
| `dry_run_ab.py:1219-1276` | CPU Lite resolves a registered model key locally; profile artifact provenance enumerates `RUNTIME_PROFILE.model_keys()`. | Keep this loader and its local-only/no-fallback path. The effective profile must carry the selected key, so artifact provenance and settings name the same model. |
| `dry_run_ab.py:5397` | Session metadata records profile, artifacts, STT settings and source provenance. | Record the explicit research request separately and retain effective profile/settings plus actual loader identity. Never relabel a small load as base. |
| `tools/mac_followup_latency.py:196-241` | `runtime_contract` independently freezes a small binding from the profile. | Use the same pure override helper when constructing intent; freeze base binding before launching a worker. |
| `tools/mac_followup_latency.py:379-433` | `configuration` emits backend/cadence flags only. | Validate a configuration field `research_stt_model: "whisper-base"`, then emit the explicit CLI flag. Absence preserves the exact control command. |
| `tools/mac_followup_latency.py:243-275` | Actual identity is checked against requested ID or resolved path, exact revision/config hash, full profile and source language. | Preserve those checks. Add an exact research-request metadata check; reject undeclared override, loaded small, primary substitution, missing identity and unexpected Gemma. |
| `models.lock.json` | `whisper-small` is registered; base is not. | Add an optional, unselected `whisper-base` HF snapshot entry with `required_for: []`, no default/optional setup group and no generic `base` alias. Do not modify existing pins. |
| `.cache/mac-en-es-closeout/lite-followup-preparation/prepare.py:235` | Existing preparation requires the small runtime contract. | Leave this immutable control validator unchanged. A new preparation artifact must explicitly admit only the new gated base experiment. |

Proposed pure helper contract: `resolve_replay_stt_profile(profile, *, research_stt_model=None, source_kind="file") -> RuntimeProfile`. Return the existing profile unchanged when omitted. For the explicit override require exactly `profile.name == "lite-cpu"`, `profile.backend == "cpu"`, and file replay; return `dataclasses.replace(profile, stt_model="whisper-base")`. The CLI then sets `settings.stt.whisper_cuda_model` to that effective profile value after `apply_profile`. Use the same helper in the harness; do not add a persistent setting, environment override, operator control, new product profile or setup option.

The CLI must additionally require an explicit `--audio-file` and disabled TTS for this research flag, and reject Standard, CPU-quality/CUDA profiles, microphone input and unknown/unpinned models before any lifecycle/model work. Existing source validation and Lite restrictions still apply. The per-language eligibility gate belongs to the frozen experiment preparation/validator, not to a general product preference. A manually invoked research command cannot by itself confer experiment eligibility.

## Exact model identity and local availability

Requested key: `whisper-base`; repository: `Systran/faster-whisper-base`; revision: `ebe41f70d5b6dfa9166e2c581c45c9c0cfc57b66`.

Already prepared quality source (path only, no weights read by this review):

`.cache/mac-en-es-closeout/base-model/hub/models--Systran--faster-whisper-base/snapshots/ebe41f70d5b6dfa9166e2c581c45c9c0cfc57b66`

Register the required CT2 files consistently with `whisper-small` (`config.json`, `model.bin`, `tokenizer.json`). Resolve through `resolve_profile_model`; do not accept an arbitrary path merely because its last component resembles the revision. Before freezing commands, prove that both small and base resolve under the captured local environment. The prepared base lives in a private HF cache, so adding the registry entry alone does not make the default lookup find it. Bind the already prepared cache through the new experiment environment/model directory, then verify both control and candidate resolution without downloading. Do not change the default setup cache or silently copy/download models.

Freeze key, repo, full revision, resolved path, config SHA-256 and the completed quality receipt/model-identity reference in the new spec. Retain requested intent separately from actual runtime identity. The existing loader may report the resolved path as its requested ID; the contract already permits that only when the path, revision and config identity match exactly. Reuse recorded model hashes for quality provenance; this preparation authorizes no weight hashing.

## New frozen spec and comparisons

Use a new output namespace and new reviewed runtime snapshot after implementation, not the current frozen snapshot or active preparation helpers. Re-run small controls and base candidates on that same new runtime. Keep `profile: "lite-cpu"`, CPU int8, 3 STT threads/1 worker, Marian CT2 finals, ONNX VAD, default 0.6 s cadence, no TTS/Gemma and all existing source-accounting/translation settings.

A candidate configuration adds only `research_stt_model: "whisper-base"`. Its spec binds a completed quality receipt hash, the exact eligible language, matching development clip/reference identity and the two pinned model bindings. Baseline and closing anchor omit the override. Reject a candidate for an ineligible or unspecified language; do not broaden English success to Spanish or vice versa. Use the accepted matched opening/candidate/closing repeat design and freeze its actual command count before launch. Do not combine base with a deadline arm merely because either passed separately.

Keep current source completeness, preview availability/loss, matched responsiveness, queue/memory, downstream WER/translation/terms and cleanup guards. Public confirmation remains conditional on a qualified live candidate and untouched approved references; no pooling endpoint samples across runs/languages. A p95 claim needs at least 100 eligible endpoints in each individual run. Sparse endpoints and whole-recording STT call timing cannot satisfy that gate.

## Focused tests to add only after authorization

- `tests/test_lite_profiles.py`: default small and default setup model sets unchanged; explicit helper returns base without changing final engine; reject every unsupported profile/backend/source; pinned lookup rejects missing/wrong revisions and missing required files.
- `tests/test_mac_followup_latency.py`: override emits one exact flag; control emits none; frozen contract and actual metadata agree; reject small-as-base, wrong revision/config/path, unknown fields, wrong language or failed/missing quality receipt. Preserve existing gain, no-TTS and no-Gemma checks.
- CLI-focused source/pure fixtures around `dry_run_ab.py` parsing/profile application: override survives `apply_profile`, invalid scope fails before native imports/model/lifecycle work, explicit model reaches `create_stt_engine`, local-only and fallback-disabled settings remain true/false respectively.
- `tests/test_factory_stt_backend.py`: retain explicit model binding behavior; do not change the factory unless the new integration exposes an actual gap.
- New preparation fixtures: completed cohort and all three language-specific quality pairs required; no arbitrary-language promotion, no omitted/reordered references or wrong manifest, no reuse/mutation of prior output roots, no unapproved cadence/deadline combination.

No production code, models, defaults, setup behavior or commands were changed or run for this note. No tests, native imports, audio/device calls, Git operations or weight reads were performed.
