#!/usr/bin/env bash
# Gemma 4 E4B domain SFT → GGUF Q4_K_M (Windows/WSL CUDA).
# --dry-run / --preflight validates actual selected data/config without GPU imports.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PYTHON="${STARK_TRAINING_PYTHON:-python}"
DRY_RUN=0
if [[ "${1:-}" == "--dry-run" || "${1:-}" == "--preflight" ]]; then DRY_RUN=1; shift; fi
if [[ $# -ne 0 ]]; then echo 'Usage: run_gemma4_e4b_domain_sft.sh [--dry-run|--preflight]' >&2; exit 2; fi
OUT_ADAPTER="${STARK_GEMMA4_ADAPTER:-fine_tuned_gemma4_e4b_domain}"
OUT_GGUF="${STARK_GEMMA4_GGUF:-models/gemma-4-e4b-it-q4km-domain.gguf}"
TRAIN_DATA="${STARK_GEMMA4_TRAIN:-}"
VERSE="${STARK_GEMMA4_VERSE:-bible_data/aligned/verse_pairs_train_v2.jsonl}"
SERMON="${STARK_GEMMA4_SERMON:-bible_data/sermon_pairs_train.jsonl}"
TRAIN_ARGS=(--base unsloth/gemma-4-E4B-it --output "$OUT_ADAPTER" --lora-r 8 --lora-alpha 8
  --epochs 2 --lr 2e-4 --packing)
# A selected path must exist. Never silently fall back after a typo or missing mount.
if [[ -n "$TRAIN_DATA" ]]; then
  TRAIN_ARGS+=(--train-data "$TRAIN_DATA")
else
  TRAIN_ARGS+=(--verse-pairs "$VERSE" --sermon-pairs "$SERMON")
fi
PREFLIGHT_ARGS=(gemma)
if [[ -n "${STARK_TRAINING_HOLDOUT:-}" ]]; then PREFLIGHT_ARGS+=(--holdout "$STARK_TRAINING_HOLDOUT"); fi
"$PYTHON" tools/training_preflight.py "${PREFLIGHT_ARGS[@]}" -- "${TRAIN_ARGS[@]}"
if [[ $DRY_RUN -eq 1 ]]; then exit 0; fi
if [[ -e "$OUT_ADAPTER" || -e "$OUT_GGUF" ]]; then echo 'Refusing to overwrite an existing Gemma output' >&2; exit 2; fi
"$PYTHON" training/train_gemma4.py "${TRAIN_ARGS[@]}"
"$PYTHON" training/export_gguf.py --adapter "$OUT_ADAPTER" --base unsloth/gemma-4-E4B-it \
  --output "$OUT_GGUF" --qtype Q4_K_M --sanity-test
echo 'E4B domain artifact exported; benchmark and Mac A/B gates remain pending before activation.'
