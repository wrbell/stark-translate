# Live and offline model-source inventory — 2026-09-10

The Mac live pinning repair covers the selected MLX/Whisper/Parakeet wrappers,
`workers.py`, optional live SpeechBrain/pyannote loaders, post-session summaries
and HF setup validation. **It does not clear
repository-wide B615 or certify model/dependency safety.** The full
[source inventory](live-hf-source-inventory.json) records every observed call's
file, line, symbol, call text, keyword expressions, role, pin status and source
file SHA-256, plus the model registry snapshot.

This read-only source snapshot records committed source
`476e349d6887b9f4e2379f9e57646bdd2bb2c581`; its exact observation times are in JSON.
It includes pinning commit `97cfc9f`, the resolved-path test correction `fad0558`,
startup fallback repair `fb44a0f`, and the documentation-contract correction.
All 211 per-file hashes were rechecked and are unchanged since `fb44a0f`;
only these four documentation artifacts are dirty in the recorded status.
Later source edits need a new inventory.
This snapshot supersedes inventory `10c9b338347b9118946e770582e3030a8f5f1e0014b71f028f488cffb464c2df`
and pinning receipt `bf5fd0822d28e64a93635616d8fdd2256eb2f82ed95e426d92bc64c67a917385`.
Their exact bytes, paired Markdown and hash index remain in
`.cache/mac-en-es-closeout/live-pinning-preparation/before-startup-fallback-20260910T170347Z/`.
JSON preserves the earlier `bf0bcf43` snapshot reference as well.
No models, native imports, network requests, weights, devices, tests, builds or
installs were run for this inventory. Historical
[security evidence](../mac_v2026_14_security.md) and the
[earlier live-source scan](../security_feasibility_20260910/raw/bandit-live-model-sources.json)
remain unchanged.

The scan covers **211 tracked source/shell files and 173 loader call sites**, plus
two separately recorded direct/delegated acquisition boundaries. It excludes and
lists 169 test files; archived executable evidence helpers are included. The
additional excluded file is the now-tracked pinning test. Model pins and metadata
are unchanged. There
are **118 unpinned or conditional sites**, **54 sites without an identified
unpinned-source gap**, and **one unverified legacy named-voice API**. These are
call-site counts, not unique downloads, Bandit findings or exploitable defects.

| Source boundary | Observed lines | Acquisition policy |
|---|---|---|
| [engines/model_paths.py](../../../engines/model_paths.py) | 192 | Registered full 40-hex revision → complete local snapshot; existing local overrides remain explicit overrides. |
| [dry_run_ab.py](../../../dry_run_ab.py) | 1188, 1198, 2684, 3051 | Main/fallback Whisper sources resolve before load; primary acquisition failures enter the language-guarded fallback path, and later transcriptions reuse the selected path. |
| [engines/mlx_engine.py](../../../engines/mlx_engine.py) | 227, 245, 368, 445, 827, 846 | Whisper primary/fallback and Gemma/draft wrappers use the pinned loading boundary. Live MTS remains rejected. |
| [engines/parakeet_mlx_engine.py](../../../engines/parakeet_mlx_engine.py) / [workers.py](../../../workers.py) | 159 / 47, 65 | Parakeet and the optional multiprocess Whisper worker receive local resolved sources. |
| [features/live_diarize.py](../../../features/live_diarize.py) | 91, 94, 259, 301 | ECAPA pins both source and `pretrained_path`; version-specific SpeechBrain revision/FetchConfig handling is explicit. Pyannote resolves registered config and nested component checkpoints locally, preserving configured pipeline parameters. |
| [engines/marian_hf_engine.py](../../../engines/marian_hf_engine.py) / [engines/tts_engine.py](../../../engines/tts_engine.py) | 74, 75 / 42, 47, 167, 168 | Existing Marian HF and Piper acquisition uses registered pinned sources; local overrides are separate provenance. |
| [tools/vad_runtime.py](../../../tools/vad_runtime.py) | 55 | Installed Silero package asset; version and checksum are recorded. Managed ONNX and Marian CT2 are local artifacts. |
| [features/summarize_sermon.py](../../../features/summarize_sermon.py) | 271, 454 | Operator summary and the separate `--translate-with-gemma` option both resolve pinned/local sources before `mlx_lm.load`. |
| [operator_app/setup.py](../../../operator_app/setup.py) | 212 (download); 199 (guard) | Every selected HF revision must be full 40-hex before mkdir/cache/marker/download processing; the wrapper independently repeats the guard. |

Residual sites comprise 105 training/export/evaluation calls and 13 optional
live calls. The operator summary and HF setup revision gaps are repaired in
source. Exact calls and conditional local paths are classified in JSON.

At `dry_run_ab.py:1184`, primary resolution and warmup now share the guarded
`try`. English network/cache failures can reach the configured pinned or cached
fallback. Spanish fails its language guard before fallback resolution/inference;
`UnpinnedModelError` is re-raised for either language. The separate engine wrapper
already has the same distinction. This changes no fallback model or thresholds.

| Remaining scope | Exact examples / boundaries |
|---|---|
| Optional CPU/CUDA/HF/NeMo engines | [engines/cuda_engine.py](../../../engines/cuda_engine.py): 131, 327, 444, 445, 668, 669, 696, 1071; [engines/hf_whisper_engine.py](../../../engines/hf_whisper_engine.py): 102, 103, 114; [engines/parakeet_engine.py](../../../engines/parakeet_engine.py): 70. The managed Marian tokenizer is local-only; Lite's explicit local-only Whisper setting does not harden every generic Standard backend. |
| Batch pyannote/Whisper | [features/diarize.py](../../../features/diarize.py): 293, 362, 409, 575. This unmodified helper is no longer imported by the live diarization daemon. |
| Standalone caption monitor | [tools/live_caption_monitor.py](../../../tools/live_caption_monitor.py): 885, 920, 1546. Lines 885/920 are its live capture mode; 1546 is WAV comparison. |
| Offline Torch Hub | [tools/batch_translate.py](../../../tools/batch_translate.py): 161; [tools/validate_session.py](../../../tools/validate_session.py): 263; [tools/benchmark_latency.py](../../../tools/benchmark_latency.py): 293; [training/preprocess_audio.py](../../../training/preprocess_audio.py): 261. These still select an unpinned Silero repository. |
| Manual training, export, scoring and conversion | All HF/Unsloth/PEFT/COMET/LaBSE, draft-probe, conversion and corpus calls are enumerated in JSON, including shell/heredoc downloads. Local adapters/checkpoints and downstream COMET checkpoint loads are distinguished from acquisition calls. |

The live pipeline’s `dry_run_ab.py:1796` QE function computes only length and
untranslated-word heuristics. `tools/validate_session.py` imports the similarly
model-free `tier1_score`. Heavy Marian backtranslation, BERTScore
(`tools/translation_qe.py:115`) and LaBSE are optional tiers in the standalone
CSV scoring CLI; they are not live pipeline or operator-only QE services. The
BERTScore wrapper implicitly selects an encoder from `lang="en"` without a pin.

No explicit `trust_remote_code` argument was found in the scanned call sites;
that means **not passed**, not globally disabled. Two offline Torch Hub calls
explicitly pass `trust_repo=True` (batch translation and session validation).
Torch Hub repository execution, SpeechBrain hparams/custom modules, checkpoint
deserialization and third-party transitive loads are distinct trust boundaries.
Revision pinning identifies the selected files; it does not establish that their
contents are safe. Gated access and functional/human diarization acceptance
remain separate gates. No post-change native/model validation was executed for
this inventory; the paired [pinning note](live-hf-pinning.md) and
[receipts](live-hf-pinning-receipts.json) record the implementation agent’s
separate checks and gated-access limitations.

The original 16 stdlib checks remain historical evidence. Five subsequent
startup source checks passed with real resolver/language guards and fake native
APIs; their retained receipt and all four source hashes match this snapshot.
[Earlier Python 3.11 CI](https://github.com/wrbell/stark-translate/actions/runs/34504851183/job/102964373192)
on `fad0558` passed 2,809 tests, with six skipped and 66.24% coverage; Python 3.12
and lint also passed that head. The source-check harness did not execute pytest.
Subsequent [CI for `fb44a0f`](https://github.com/wrbell/stark-translate/actions/runs/34505780125)
passed all four new startup cases but failed a stale documentation assertion:
Python 3.11 recorded 2,812 passed, one failed, six skipped and 66.24% coverage.
The coordinator reported the same sole failure on Python 3.12. The documentation
assertion is repaired in `476e349`; full CI for that head remains pending here.
Neither the failed full suite nor source checks establish native/artifact validation.

## 2026-09-11 follow-up

The 13 `optional_live` entries with `residual_remote_risk: true` now use pinned
source resolution or a strict local-only loader. The JSON field is `roles`
(a list); filtering membership yields exactly the 13 sites below. Its earlier
text, hashes and JSON remain historical and are not refreshed by this append.
The legacy Piper site is separately classified under
`training_export_evaluation` with a null risk value.

| Original inventory sites | Follow-up policy |
| --- | --- |
| `engines/cuda_engine.py:131,327` | Primary and lazy fallback faster-whisper require an installed pinned snapshot or explicit local path; both pass `local_files_only=True`. |
| `engines/cuda_engine.py:444,445,668,669,696` | Basic/streaming Transformers tokenizer, target and assistant use `resolve_hf_model_source`: local-only path or exact manifest revision. The assistant is validated before any primary loader call. |
| `engines/hf_whisper_engine.py:102,103,114` | Processor, target and draft share the same source policy; all configured sources are validated before loading. |
| `engines/parakeet_engine.py:70` | NeMo requires exactly one local `.nemo` checkpoint and uses `restore_from`; it never calls the revision-less `from_pretrained`. |
| `tools/live_caption_monitor.py:885,920` | Live warmup and chunk transcription reuse the complete local snapshot from `resolve_model_for_loading`, after validating remote identity with `resolve_hf_model_source`. |
| `training/evaluate_piper.py:219` (legacy, additional) | `resolve_piper_voice` selects a local ONNX/config pair; names require a manifest pin and an installed copy. Piper receives only a local path. |

`resolve_hf_model_source` now checks a remote identity's immutable manifest entry
before looking in caches, so a cached moving-ref repo cannot bypass the missing
pin error. Explicit local overrides remain allowed. Faster-whisper, NeMo and
legacy Piper never acquire models at runtime; an unavailable local artifact
raises `UnpinnedModelError` naming the model and `models.lock.json`.

Existing entries cover faster-whisper Turbo/small, the monitor's
`wbell7/distil-whisper-large-v3.5-mlx`, and the EN/ES Piper voices. No new lock
entries were added: the permitted local evidence did not supply immutable
revisions for faster-whisper `large-v3`, `openai/whisper-large-v3[-turbo]`,
`google/translategemma-{4b,12b}-it`, or `nvidia/parakeet-tdt-0.6b-v3`.
These remote selections and arbitrary unregistered custom/draft IDs now fail
closed until registered, or explicitly configured as local artifacts.
Defaults, dependencies and existing lock entries are unchanged.

The [operator-only offline model path policy](../../security_offline_model_paths.md)
enumerates the separate 105 residual `training_export_evaluation` calls, explains
the CI B615 skip and scan-root limits, and gives the pinning procedure for
promotion into live code. The offline role has 130 entries total: 105 true,
24 false and the one legacy Piper null entry addressed above.

Regression tests use tiny file fixtures and mocked Transformers, faster-whisper,
NeMo, MLX, Piper and download APIs. They cover local/pinned selection, missing
and malformed pins, partial caches, target/draft/fallback rejection, and reuse
of the monitor's selected snapshot. No model download, native model execution,
server startup or hardware certification was performed for this follow-up.
