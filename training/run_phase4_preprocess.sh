#!/usr/bin/env bash
# Phase 4 — full-corpus audio preprocess (Windows/WSL).
# Requires sermon WAVs under stark_data/raw (or STARK_RAW_DIR).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
INPUT="${STARK_RAW_DIR:-stark_data/raw}"
OUTPUT="${STARK_CLEANED_DIR:-stark_data/cleaned}"

if [[ ! -d "$INPUT" ]]; then
  echo "ERROR: input dir missing: $INPUT"
  echo "Place sermon WAVs there, or set STARK_RAW_DIR."
  exit 1
fi

python training/run_phase4_corpus.py \
  --input "$INPUT" \
  --output "$OUTPUT" \
  --resume \
  "$@"

echo "Phase 4 status: $OUTPUT/phase4_status.json"
