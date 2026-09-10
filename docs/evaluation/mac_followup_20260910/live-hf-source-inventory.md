# Live and offline model-source inventory — 2026-09-10

The Mac live pinning repair covers the selected MLX/Whisper/Parakeet wrappers,
`workers.py`, optional live SpeechBrain/pyannote loaders, post-session summaries
and HF setup validation. **It does not clear
repository-wide B615 or certify model/dependency safety.** The full
[source inventory](live-hf-source-inventory.json) records every observed call's
file, line, symbol, call text, keyword expressions, role, pin status and source
file SHA-256, plus the model registry snapshot.

This read-only source snapshot was taken at `2026-09-10T16:42:04.825174+00:00` on
`4b0144d1b15bcca2fac4b490d52db2ca257238fb` with the recorded working-tree changes.
Per-file hashes bind the observed source; later edits need a new inventory.
This snapshot supersedes the earlier `bf0bcf43655096cde46905c8076a7efe84481829dc484e3497020f03a363f7c9`
inventory, preserved with its Markdown and hash index in the recorded scratch archive.
No models, native imports, network requests, weights, devices, tests, builds or
installs were run for this inventory. Historical
[security evidence](../mac_v2026_14_security.md) and the
[earlier live-source scan](../security_feasibility_20260910/raw/bandit-live-model-sources.json)
remain unchanged.

The scan covers **211 tracked source/shell files and 173 loader call sites**, plus
two separately recorded direct/delegated acquisition boundaries. It excludes and
lists 168 test files; archived executable evidence helpers are included. There
are **118 unpinned or conditional sites**, **54 sites without an identified
unpinned-source gap**, and **one unverified legacy named-voice API**. These are
call-site counts, not unique downloads, Bandit findings or exploitable defects.

| Source boundary | Observed lines | Acquisition policy |
|---|---|---|
| [engines/model_paths.py](../../../engines/model_paths.py) | 192 | Registered full 40-hex revision → complete local snapshot; existing local overrides remain explicit overrides. |
| [dry_run_ab.py](../../../dry_run_ab.py) | 1188, 1196, 2682, 3049 | Main/fallback Whisper sources resolve before load; later transcriptions reuse the selected path. |
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

| Remaining scope | Exact examples / boundaries |
|---|---|
| Optional CPU/CUDA/HF/NeMo engines | [engines/cuda_engine.py](../../../engines/cuda_engine.py): 131, 327, 444, 445, 668, 669, 696, 1071; [engines/hf_whisper_engine.py](../../../engines/hf_whisper_engine.py): 102, 103, 114; [engines/parakeet_engine.py](../../../engines/parakeet_engine.py): 70. The managed Marian tokenizer is local-only; Lite's explicit local-only Whisper setting does not harden every generic Standard backend. |
| Batch pyannote/Whisper | [features/diarize.py](../../../features/diarize.py): 293, 362, 409, 575. This unmodified helper is no longer imported by the live diarization daemon. |
| Standalone caption monitor | [tools/live_caption_monitor.py](../../../tools/live_caption_monitor.py): 885, 920, 1546. Lines 885/920 are its live capture mode; 1546 is WAV comparison. |
| Offline Torch Hub | [tools/batch_translate.py](../../../tools/batch_translate.py): 161; [tools/validate_session.py](../../../tools/validate_session.py): 263; [tools/benchmark_latency.py](../../../tools/benchmark_latency.py): 293; [training/preprocess_audio.py](../../../training/preprocess_audio.py): 261. These still select an unpinned Silero repository. |
| Manual training, export, scoring and conversion | All HF/Unsloth/PEFT/COMET/LaBSE, draft-probe, conversion and corpus calls are enumerated in JSON, including shell/heredoc downloads. Local adapters/checkpoints and downstream COMET checkpoint loads are distinguished from acquisition calls. |

The live pipeline’s `dry_run_ab.py:1794` QE function computes only length and
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
