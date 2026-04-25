# W15/W15b Curriculum Learning Failure — Post-Mortem

> **Date:** 2026-04-12
> **Impact:** WER regressed from 7.61% to 64.76% (8.5x degradation)
> **Root cause:** `--init-from` code bug + hard-only training data
> **Status:** Bug fixed in `train_whisper.py`, W16 corrective run planned

---

## What Happened

W15 and W15b attempted curriculum learning: mine hard examples from W14 adapter, then train a new adapter on those hard examples with `--init-from W14` to inherit learned weights.

| Run | LR | Data | Steps | Train Loss | Fresh Eval WER |
|-----|-----|------|-------|------------|----------------|
| W14 (baseline) | 1e-4 | 40,947 mixed | 10,443 | 0.0988 | 7.42% |
| W15 | 5e-5 | 2,506 hard-only | 294 | 0.9488 | ~64.76% |
| W15b | 1e-5 | 9,778 hard-only | 1,165 | 0.5872 | 64.76% |

Both W15 and W15b catastrophically regressed. Lower learning rate (W15b) and more data did not help.

---

## Root Cause 1: `--init-from` Silently Failed to Load Weights

**File:** `training/train_whisper.py`, lines 320–333

The `--init-from` mechanism used `model.load_state_dict(adapter_weights, strict=False)`. With `strict=False`, missing keys are silently ignored. The code only checked for `unexpected_keys` (always empty), never `missing_keys`.

**Evidence — first-step training loss:**

| Run | Step 10 Loss | Expected (if weights loaded) | Interpretation |
|-----|-------------|------------------------------|----------------|
| W14 | ~0.5 | — | Normal start from random LoRA |
| W15 | **1.8174** | ~0.1 (continuing from W14's 0.0988) | **Weights not loaded** |
| W15b | **1.7971** | ~0.1 | **Same failure** |

If adapter weights had loaded correctly, step-10 loss should be near W14's final loss (~0.1). Instead both start at ~1.8, indicating random LoRA initialization — the model learned from scratch on hard-only data.

**Fix applied:** Added `missing_keys` validation for LoRA keys, plus a check that at least some `lora_*` tensors were found in the file.

---

## Root Cause 2: Hard-Only Training Data

Even if weights had loaded, training on 100% hard examples is destructive:

| Dataset | Examples | Composition | Result |
|---------|----------|-------------|--------|
| W14 | 40,947 | Mixed clean/medium/hard | 7.42% WER |
| W15 | 2,506 | **100% hard** (WER 0.15–0.80) | 64.76% WER |
| W15b | 9,778 | **100% hard** (WER 0.15–0.80) | 64.76% WER |

The hard subset composition (`ablation/sermon_whisper_chunks_w15_hard.json`):
- 10,000 chunks with avg_logprob range -4.30 to -0.02 (wide variance)
- Includes very noisy/garbage chunks that confuse the model
- No clean examples to anchor general transcription ability

From `whisper_ablation/w14_mined.jsonl` (52,928 chunks): median WER = 0.0 (majority correct), only 2.8% have WER > 0.5. Training on only the tail produces a model that can't handle the 97.2% of normal speech.

---

## What Was NOT a Factor

- **W14 adapter quality:** W14 eval = 7.42% WER (improvement over W13's 7.61%). Safe init source.
- **Learning rate:** W15b used 10x lower LR (1e-5 vs 5e-5), same failure. Not a hyperparameter issue.
- **Replay ratio:** Both used 0.5 (50% general replay). Replay helps but can't compensate for broken init + hard-only data.
- **LoRA config:** Same r=32, alpha=64, q_proj+v_proj as all winning runs.

---

## Corrective Actions

### 1. Bug Fix (Done)

`train_whisper.py` now validates `--init-from`:
- Checks `missing_keys` for any LoRA-related keys (raises `ValueError` if found)
- Verifies at least one `lora_*` tensor exists in the safetensors file
- Logs which specific LoRA tensors were loaded

### 2. W16 Corrective Run (Planned)

Key differences from W15:

| Parameter | W15 (failed) | W16 (planned) |
|-----------|-------------|---------------|
| Init source | W14 | **W7** (winner, 5.63% WER on old eval) |
| Data composition | 100% hard | **30% hard + 70% random** |
| Dataset size | 2,506 | **~10,000+** |
| Validation | None | **Check step-10 loss < 0.5** |

### 3. Monitoring (Planned)

- Log first-step loss after `--init-from` to training manifest
- Fail training if first-step loss > 1.0 (indicates fresh init)
- Record init-from source path and SHA-256 in manifest for traceability

---

## Files Referenced

| File | Evidence |
|------|----------|
| `training/train_whisper.py:320-333` | Bug location (strict=False) |
| `whisper_ablation/W15_hard_w14/checkpoint-294/trainer_state.json` | W15 step-10 loss = 1.8174 |
| `whisper_ablation/W15b_hard_w14/checkpoint-1165/trainer_state.json` | W15b step-10 loss = 1.7971 |
| `whisper_ablation/W14_combined50k_3ep/training_manifest.json` | W14 final loss = 0.0988 |
| `whisper_ablation/w14_mined.jsonl` | 52,928 mined chunks, median WER = 0.0 |
| `ablation/sermon_whisper_chunks_w15_hard.json` | Hard subset composition |
