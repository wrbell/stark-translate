# training/AGENTS.md — Fine-Tuning & Data (Agent Guide)

> Paired with [`CLAUDE.md`](./CLAUDE.md). All training runs on **WSL/A2000**; adapters
> transfer to Mac for inference only.

## Agent constraints

- **Training cutoff:** 2026-03-14 — never train on post-cutoff eval sermons listed in CLAUDE.md.
- **Copyright:** Public-domain Bible pairs only — no ESV/NIV/NLT bulk training.
- **Holdout safety:** Export/review tests must not overwrite local holdout; write beside target dataset.
- **Phase 4:** Requires sermon WAVs on WSL — status in `phase4_status.json`.

## Pipeline order (WSL runbook)

1. Phase 4 preprocess → 2. Gemma E4B domain SFT → 3. W17 Whisper DoRA → 4. optional Parakeet bench → 5. Mac transfer → 6. active learning

Runbook: [`docs/wsl_pipeline_refresh.md`](../docs/wsl_pipeline_refresh.md)

## Key artifacts

| Run | Result |
|-----|--------|
| S6 TranslateGemma | Balanced 1:1 verse/sermon winner |
| W12 Whisper | 198K chunks; 21.41% baseline WER fresh eval |
| W16 | 7.25% fresh-eval WER — production CUDA STT |
| W17 | Scripted, not yet trained |

## Full reference

10-step preprocess, LoRA configs, hard mining, Gemma sweeps, go/no-go gates:
[`CLAUDE.md`](./CLAUDE.md)
