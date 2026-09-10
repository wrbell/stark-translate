#!/usr/bin/env bash
# W17 Whisper curriculum: expanded modules + DoRA + hard-mix (not hard-only).
# --dry-run / --preflight reads config, adapter headers and source data on CPU only.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PYTHON="${STARK_TRAINING_PYTHON:-python}"
DRY_RUN=0
if [[ "${1:-}" == "--dry-run" || "${1:-}" == "--preflight" ]]; then DRY_RUN=1; shift; fi
if [[ $# -ne 0 ]]; then echo 'Usage: run_w17_curriculum.sh [--dry-run|--preflight]' >&2; exit 2; fi

DATASET="${STARK_WHISPER_DATASET:-stark_data/whisper_dataset_deepgram}"
INIT_FROM="${STARK_W16_ADAPTER:-adapters/whisper_turbo/active}"
HARD_MINED="${STARK_HARD_MINED:-stark_data/hard_examples_w17.jsonl}"
HARD_SUBSET="${STARK_HARD_SUBSET:-stark_data/hard_subset_w17.json}"
W17_DATASET="${STARK_W17_DATASET:-stark_data/whisper_dataset_w17}"
DEEPGRAM="${STARK_DEEPGRAM_DIR:-stark_data/deepgram_transcripts}"
AUDIO="${STARK_AUDIO_DIR:-stark_data/raw/midwest}"
OUT="${STARK_W17_OUT:-fine_tuned_whisper_w17}"
CT2_OUT="${STARK_W17_CT2:-whisper_ct2/w17}"
MODULES=(q_proj v_proj k_proj out_proj fc1 fc2)
TRAIN_ARGS=(--dataset "$W17_DATASET" --output "$OUT" --model openai/whisper-large-v3-turbo
  --target-modules "${MODULES[@]}" --lora-r 32 --lora-alpha 64 --epochs 1 --lr 1e-4
  --replay-ratio 0.3 --require-replay --init-from "$INIT_FROM" --use-dora --allow-target-expansion)
if [[ -n "${STARK_WHISPER_MODEL_CONFIG:-}" ]]; then TRAIN_ARGS+=(--model-config "$STARK_WHISPER_MODEL_CONFIG"); fi
PREFLIGHT_ARGS=(w17 --chunks-json "${DATASET}/chunks.json" --deepgram-dir "$DEEPGRAM" --audio-dir "$AUDIO")
if [[ -n "${STARK_TRAINING_HOLDOUT:-}" ]]; then PREFLIGHT_ARGS+=(--holdout "$STARK_TRAINING_HOLDOUT"); fi

# Exact same config/data validation is mandatory for a real run and a dry-run.
if [[ $DRY_RUN -eq 1 ]]; then
  exec "$PYTHON" tools/training_preflight.py "${PREFLIGHT_ARGS[@]}" -- "${TRAIN_ARGS[@]}"
fi
for output in "$HARD_MINED" "$HARD_SUBSET" "$W17_DATASET" "$OUT" "$CT2_OUT"; do
  if [[ -e "$output" ]]; then echo "Refusing to overwrite existing W17 output: $output" >&2; exit 2; fi
done
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
TRAIN_CHUNKS="$WORK/train_chunks.json"
"$PYTHON" tools/training_preflight.py "${PREFLIGHT_ARGS[@]}" --training-chunks-output "$TRAIN_CHUNKS" -- "${TRAIN_ARGS[@]}"

# Both mining and lookup consume the identical filtered ordering. Eval chunks never enter mining.
"$PYTHON" training/mine_hard_examples.py --adapter "$INIT_FROM" --chunks-json "$TRAIN_CHUNKS" \
  --deepgram-dir "$DEEPGRAM" --audio-dir "$AUDIO" --output "$HARD_MINED"
"$PYTHON" training/build_hard_subset.py --mined "$HARD_MINED" --chunks-json "$TRAIN_CHUNKS" \
  --output "$HARD_SUBSET" --wer-min 0.15 --wer-max 0.80 --include-tier1
# Mining can yield an empty subset; revalidate it before alignment/trainer startup.
"$PYTHON" tools/training_preflight.py "${PREFLIGHT_ARGS[@]}" --chunks-json "$HARD_SUBSET" -- "${TRAIN_ARGS[@]}"
# build_hard_subset emits JSON, not an audiofolder. Materialize a separate dataset.
"$PYTHON" training/align_deepgram_chunks.py --whisper-chunks "$HARD_SUBSET" \
  --deepgram-dir "$DEEPGRAM" --audio-dir "$AUDIO" --output "$W17_DATASET"
"$PYTHON" training/train_whisper.py "${TRAIN_ARGS[@]}"
"$PYTHON" training/export_ct2.py --adapter "$OUT" --output "$CT2_OUT"
"$PYTHON" tools/manage_adapters.py register --model whisper_turbo_ct2 --adapter "$CT2_OUT"
echo 'W17 exported. CUDA benchmark, Mac A/B and activation gates remain separate.'
