# training/AGENTS.md — Fine-Tuning & Data (Agent Guide)

> Paired with [`CLAUDE.md`](./CLAUDE.md) (programs and their state, data pipeline with
> verified flags, Whisper LoRA / W17, Gemma 4 tuning, historical TranslateGemma sweep,
> export and transfer). All training runs on **WSL/A2000**; exported artifacts move to the
> inference machines. No WSL job has run since 2026-04-30; every WSL item is pending
> hardware time. Native Windows / RTX 2070 **inference** is the separate Lite runtime
> ([`docs/lite_profiles.md`](../docs/lite_profiles.md)), not a training topic.

## Agent constraints

- **Training cutoff:** 2026-03-14 — never train on post-cutoff eval sermons or the four
  fresh-eval video ids listed in `CLAUDE.md` § Data organization.
- **Copyright:** public-domain Bible pairs only — no ESV/NASB/NIV/NLT/NVI/LBLA/RVR1960/DHH.
- **Corpus version:** `bible_data/aligned/verse_pairs_train_v2.jsonl`; v1 has the Platense
  misalignment. `run_gemma4_e4b_domain_sft.sh` now defaults to v2; explicitly selected
  missing corpora fail. Both production recipes support CPU-only `--dry-run` / `--preflight`
  through `tools/training_preflight.py`, before trainer/GPU imports. Missing WSL artifacts
  remain a failed preflight, never a completed CUDA gate.
- **Holdout safety:** Review/export must not overwrite local holdout; correction imports keep
  evaluation and training separate (`tests/test_correction_import_safety.py`).
- **Gemma 4:** train E2B and E4B separately; freeze PLE and vision/audio towers;
  `enable_thinking=False` on every example; QLoRA via Unsloth. `train_gemma4.py`,
  `train_gemma4_cpo.py`, `export_gguf.py` ran for the v1/v1.1/v2-cpo experiments but keep
  pre-run `UNTESTED` headers — cite them as "review before each run".
- **W17:** use Whisper's `out_proj`; local model config and source adapter rank, scale,
  full tensor coverage and shapes are checked before GPU imports. The recipe explicitly
  enables `--allow-target-expansion` for fresh added modules/DoRA magnitudes while retaining
  every W16 LoRA tensor. `--require-replay` forbids falling back to hard-only training.
  Hard-subset JSON is aligned into a separate audiofolder before training.
  Preflight, mining and alignment resolve literal source stems through the same WAV
  index; nested/mixed-case WAV extensions work, and ambiguous duplicate paths fail.
- **Jacobo candidates:** [`candidates/README.md`](candidates/README.md) records original
  synthetic, **unapproved** preference triples, provenance and overlap checks. The CPO
  loader rejects these before GPU imports. The missing v2 holdout and bilingual review
  remain gates; candidate preparation does not close #136 or #137.
- **Ship rule:** stock Gemma 4 E4B stays the default until a Mac A/B note (#135) says
  otherwise; v2-cpo is at parity and fails the Jacobo canary (#136).
- **Quoting flags:** every CLI flag in `CLAUDE.md` was read from the script's `argparse`
  block at `c5fb689`; re-check with `--help` after pulling.
- Do not paste WER/COMET/latency numbers into guides; link the evidence document that
  defines the measurement.

## Pipeline order (WSL runbook)

1. Phase 4 preprocess → 2. Gemma 4 E4B domain SFT → GGUF → 3. W17 Whisper DoRA + hard-mix →
CT2 → 4. optional Parakeet EN bench on CUDA → 5. Mac transfer + Phase 7 evaluate →
6. active learning cycle → 7. CUDA latency proposal.
Runbook: [`docs/wsl_pipeline_refresh.md`](../docs/wsl_pipeline_refresh.md).

## Artifacts and where their numbers live

| Artifact | State | Evidence |
|----------|-------|----------|
| W16 Whisper LoRA → CT2 | Deployed CUDA STT (`adapters/whisper_turbo_ct2/active`); Mac uses Parakeet/mlx-whisper without LoRA | [`docs/archive/v2026.7/STT_BENCHMARK.md`](../docs/archive/v2026.7/STT_BENCHMARK.md) |
| W17 | Scripted (`run_w17_curriculum.sh`), untrained | — |
| Gemma 4 spike / v1 / v1.1 / v2-cpo | Trained 2026-04-29/30; parity with stock E4B, Jacobo failing | [`docs/gemma4_tuning/v1_results.md`](../docs/gemma4_tuning/v1_results.md) |
| E4B domain SFT (`run_gemma4_e4b_domain_sft.sh`) | Scripted, not run; v2 default, CPU preflight implemented | — |
| TranslateGemma S1–S9 (S6 winner) | Historical, superseded | [`docs/archive/training/gemma_tuning_test_matrix.md`](../docs/archive/training/gemma_tuning_test_matrix.md) |
| Hindi / Chinese corpora | Training corpora pending user decision (#138); offline church-audio R&D baseline completed separately, no live path or further EN↔ES-program work | [`docs/evaluation/overnight_hindi/README.md`](../docs/evaluation/overnight_hindi/README.md) |

## Backlog

`wsl-phase4`, `wsl-e4b-domain-sft`, `wsl-w17-export`, `cuda-latency-proposal`,
`issue-135-mac-ab`, `issue-136-jacobo-cpo`, `issue-138-hindi-zero-shot` in
[`docs/backlog.json`](../docs/backlog.json).
