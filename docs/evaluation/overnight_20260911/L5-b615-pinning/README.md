# L5 — remaining optional-live Hugging Face download paths (B615)

**Lane state:** DONE — merged to `main` as `2cf0f7a` (PR #202, squash, CI green on Python 3.11/3.12, lint, security).

## What changed

- The 13 `optional_live` residual call sites listed in `docs/evaluation/mac_followup_20260910/live-hf-source-inventory.json` (`engines/cuda_engine.py` ×7, `engines/hf_whisper_engine.py` ×3, `engines/parakeet_engine.py` ×1, `tools/live_caption_monitor.py` ×2) and the legacy Piper named loader in `training/evaluate_piper.py` now resolve through `engines/model_paths.py`. A pinned `models.lock.json` revision or a complete local snapshot loads with `local_files_only` / `revision=`; any other identity raises `UnpinnedModelError` before a loader is called, instead of fetching an unpinned remote.
- NeMo `ASRModel.from_pretrained` (no revision argument) is replaced by a required local `.nemo` via `restore_from`.
- `engines/model_paths.py`: `resolve_hf_model_source` rejects unpinned remote identities before the cache lookup; new `resolve_local_model_for_loading` for loaders that need prepared local artifacts.
- New `docs/security_offline_model_paths.md` records the 105 training/export/evaluation call sites as operator-only offline paths (each inventory ID listed once) and the procedure for pinning one when it is promoted to live. The inventory Markdown received an append-only 2026-09-11 section; the inventory JSON and every existing lock entry are byte-identical to before.
- 38 mocked regression tests in `tests/test_optional_live_model_pinning.py`; no test loads a model or touches the network.

## What this does not claim

No new lock entries were added (no revision was invented). CI still skips Bandit B615; the unit tests are the guard. This is source-level pinning of the optional live paths; it is not a global B615 clearance, a dependency-advisory audit, or model/device certification. The gated Pyannote segmentation dependency (HTTP 403) is unchanged.

Backlog item: `security-b615-pinning` — acceptance ("optional/fallback HF paths pin revisions or are documented as operator-only, starting with live-path fallbacks") is met for the inventoried sites; the closeout lane updates the backlog entry.
