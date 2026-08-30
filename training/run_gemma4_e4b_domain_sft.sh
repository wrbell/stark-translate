#!/usr/bin/env bash
# Gemma 4 E4B domain SFT → GGUF Q4_K_M (Windows/WSL CUDA).
# Expects S6-style verse/sermon pair JSONL under bible_data/ or STARK_* paths.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

OUT_ADAPTER="${STARK_GEMMA4_ADAPTER:-fine_tuned_gemma4_e4b_domain}"
OUT_GGUF="${STARK_GEMMA4_GGUF:-models/gemma-4-e4b-it-q4km-domain.gguf}"
TRAIN_DATA="${STARK_GEMMA4_TRAIN:-}"
VERSE="${STARK_GEMMA4_VERSE:-bible_data/verse_pairs_train.jsonl}"
SERMON="${STARK_GEMMA4_SERMON:-bible_data/sermon_pairs_train.jsonl}"

TRAIN_ARGS=(
  --base unsloth/gemma-4-E4B-it
  --output "$OUT_ADAPTER"
  --lora-r 8
  --lora-alpha 8
  --epochs 2
  --lr 2e-4
  --packing
)

if [[ -n "$TRAIN_DATA" && -f "$TRAIN_DATA" ]]; then
  TRAIN_ARGS+=(--train-data "$TRAIN_DATA")
else
  [[ -f "$VERSE" ]] && TRAIN_ARGS+=(--verse-pairs "$VERSE")
  [[ -f "$SERMON" ]] && TRAIN_ARGS+=(--sermon-pairs "$SERMON")
fi

echo "==> Domain SFT: ${TRAIN_ARGS[*]}"
python training/train_gemma4.py "${TRAIN_ARGS[@]}"

echo "==> Export GGUF Q4_K_M + sanity canary"
python training/export_gguf.py \
  --adapter "$OUT_ADAPTER" \
  --base unsloth/gemma-4-E4B-it \
  --output "$OUT_GGUF" \
  --qtype Q4_K_M \
  --sanity-test

echo "Done. Activate with:"
echo "  python tools/manage_adapters.py register --model gemma_e4b_gguf --adapter $OUT_ADAPTER"
echo "  # Point llama-server at $OUT_GGUF"
