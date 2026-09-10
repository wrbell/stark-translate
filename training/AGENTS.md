# training/AGENTS.md — Fine-Tuning & Data (Agent Guide)

> Paired with [`CLAUDE.md`](./CLAUDE.md) (10-step preprocess, LoRA/QLoRA configs, hard
> mining, Gemma 4 tuning program, historical TranslateGemma sweep, go/no-go gates).
> All training runs on **WSL/A2000**; adapters transfer to Mac for inference only.
> Nothing in `training/` ran tonight; every WSL item is pending hardware.

## Agent constraints

- **Training cutoff:** 2026-03-14 (`CLAUDE.md` § Data Organization) — never train on post-cutoff eval sermons or the fresh-eval set.
- **Copyright:** public-domain Bible pairs only — no ESV/NASB/NIV/NLT/NVI/LBLA/RVR1960/DHH bulk training.
- **Corpus version:** use `verse_pairs_train_v2.jsonl`; v1 has the Platense misalignment (`docs/platense_alignment_bug.md`).
- **Holdout safety:** Review/export tests must not overwrite local holdout; write beside the target dataset. Correction imports keep evaluation and training separate (`tests/test_correction_import_safety.py`).
- **Gemma 4:** train E2B and E4B separately; freeze PLE and the vision/audio towers; `enable_thinking=False` on every example; QLoRA via Unsloth. `train_gemma4.py` / `train_gemma4_cpo.py` / `export_gguf.py` are marked `UNTESTED` for the full domain run — say so when citing them.
- **Ship rule:** stock Gemma 4 E4B stays the default until a Mac A/B note (#135) says otherwise; v2-cpo is at parity, not better, and fails the Jacobo canary (#136).
- **Phase 4:** requires sermon WAVs on WSL — status file `stark_data/cleaned/phase4_status.json`.
- Do not paste WER/COMET numbers into guides; link the evidence document that defines the measurement.

## Pipeline order (WSL runbook)

1. Phase 4 preprocess → 2. Gemma 4 E4B domain SFT → GGUF → 3. W17 Whisper DoRA + hard-mix → CT2 → 4. optional Parakeet EN bench → 5. Mac transfer + Phase 7 evaluate → 6. active learning cycle.
Runbook: [`docs/wsl_pipeline_refresh.md`](../docs/wsl_pipeline_refresh.md).

## Artifacts and where their numbers live

| Artifact | State | Evidence |
|----------|-------|----------|
| W16 Whisper LoRA → CT2 | Deployed CUDA STT (`adapters/whisper_turbo_ct2/active`) | [`docs/archive/v2026.7/STT_BENCHMARK.md`](../docs/archive/v2026.7/STT_BENCHMARK.md) |
| W17 | Scripted (`run_w17_curriculum.sh`), untrained | — |
| Gemma 4 v1 / v1.1 / v2-cpo | Trained on WSL; parity with stock E4B, Jacobo failing | [`docs/gemma4_tuning/v1_results.md`](../docs/gemma4_tuning/v1_results.md) |
| E4B domain SFT (`run_gemma4_e4b_domain_sft.sh`) | Scripted, pending Phase 4 corpus | — |
| TranslateGemma S1–S9 (S6 winner) | Historical, superseded | [`docs/archive/training/gemma_tuning_test_matrix.md`](../docs/archive/training/gemma_tuning_test_matrix.md) |

## Backlog

`wsl-phase4`, `wsl-e4b-domain-sft`, `wsl-w17-export`, `cuda-latency-proposal`,
`issue-135-mac-ab`, `issue-136-jacobo-cpo` in [`docs/backlog.json`](../docs/backlog.json).
