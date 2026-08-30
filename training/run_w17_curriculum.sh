#!/usr/bin/env bash
# W17 Whisper curriculum: expanded modules + DoRA + hard-mix (not hard-only).
# Run on Windows/WSL after Phase 4 preprocess + Deepgram alignment.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

DATASET="${STARK_WHISPER_DATASET:-stark_data/whisper_dataset_deepgram}"
INIT_FROM="${STARK_W16_ADAPTER:-adapters/whisper_turbo/active}"
HARD_MINED="${STARK_HARD_MINED:-stark_data/hard_examples_w17.jsonl}"
HARD_SUBSET="${STARK_HARD_SUBSET:-stark_data/hard_subset_w17}"
OUT="${STARK_W17_OUT:-fine_tuned_whisper_w17}"
CT2_OUT="${STARK_W17_CT2:-whisper_ct2/w17}"

MODULES=(q_proj v_proj k_proj o_proj fc1 fc2)

echo "==> Mine hard examples from ${INIT_FROM}"
python training/mine_hard_examples.py \
  --adapter "$INIT_FROM" \
  --chunks-json "${DATASET}/chunks.json" \
  --deepgram-dir "${STARK_DEEPGRAM_DIR:-stark_data/deepgram}" \
  --audio-dir "${STARK_AUDIO_DIR:-stark_data/cleaned/chunks}" \
  --output "$HARD_MINED" \
  --resume \
  "$@" || true

echo "==> Build hard subset (WER 0.15–0.80) + mix with replay via train --replay-ratio"
python training/build_hard_subset.py \
  --mined "$HARD_MINED" \
  --chunks-json "${DATASET}/chunks.json" \
  --output "$HARD_SUBSET" \
  --wer-min 0.15 \
  --wer-max 0.80 \
  --include-tier1

echo "==> Train W17 (DoRA + expanded modules + init-from W16 + replay 0.3)"
python training/train_whisper.py \
  --dataset "$HARD_SUBSET" \
  --output "$OUT" \
  --model openai/whisper-large-v3-turbo \
  --target-modules "${MODULES[@]}" \
  --lora-r 32 \
  --lora-alpha 64 \
  --epochs 1 \
  --lr 1e-4 \
  --replay-ratio 0.3 \
  --init-from "$INIT_FROM" \
  --use-dora

echo "==> Export CT2 + sanity WER gate"
python training/export_ct2.py \
  --adapter "$OUT" \
  --output "$CT2_OUT"

echo "==> Register CT2 adapter slot"
python tools/manage_adapters.py register \
  --model whisper_turbo_ct2 \
  --adapter "$CT2_OUT" || true

echo "W17 complete. Bench before activate:"
echo "  python tools/benchmark_stt_engines.py"
echo "  python tools/manage_adapters.py activate --model whisper_turbo_ct2 --version <id>"
