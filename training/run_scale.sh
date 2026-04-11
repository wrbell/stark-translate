#!/usr/bin/env bash
# run_scale.sh — Phase 3 Scale-Up: Synthetic sermon data + COMET-primary gates
#
# 3 runs (S1/S2/S3), testing different verse/sermon data mix ratios with the
# Phase 2.5 winning config. Uses 12B-distilled sermon data via --sermon-data.
#
# Usage:
#   bash training/run_scale.sh          # from project root
#   nohup bash training/run_scale.sh &  # background / overnight

set -euo pipefail

# --- Setup ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

VENV="/home/wbell/stt_train_env/bin/activate"
SCALE_DIR="scale_runs"
LOG="$SCALE_DIR/scale_log.txt"
SERMON_DATA="ablation/sermon_distilled_pairs.jsonl"

source "$VENV"
mkdir -p "$SCALE_DIR"

# ============================================================
# CONFIG: Phase 2.5 determines winner (A1 or B4)
# ============================================================
# --- IF Phase 2.5 selects A1 (COMET-optimal) ---
# WINNING_LR="1e-5"
# WINNING_STEPS="50"
# WINNING_EXTRAS=""
# --- IF Phase 2.5 selects B4 (BLEU-optimal) ---
WINNING_LR="1e-6"
WINNING_STEPS="1114"
WINNING_EXTRAS="--neftune 5"
# ============================================================

# Validate sermon data exists
if [[ ! -f "$SERMON_DATA" ]]; then
    echo "ERROR: Sermon data not found at $SERMON_DATA" >&2
    echo "Run the synthetic data pipeline first (see docs/scale_run.md)." >&2
    exit 1
fi

echo "=== Phase 3 Scale-Up: Synthetic Sermon Data — $(date) ===" | tee "$LOG"
echo "Project dir: $PROJECT_DIR" | tee -a "$LOG"
echo "Python: $(which python)" | tee -a "$LOG"
echo "Winning config: lr=$WINNING_LR, steps=$WINNING_STEPS, extras='$WINNING_EXTRAS'" | tee -a "$LOG"
echo "Sermon data: $SERMON_DATA" | tee -a "$LOG"
echo "" | tee -a "$LOG"

# --- Run definitions ---
declare -a RUNS=(
    "S1_verse65_sermon30"
    "S2_verse50_sermon45"
    "S3_sermon_only"
)

declare -A TRAIN_ARGS
TRAIN_ARGS[S1_verse65_sermon30]="--max-pairs 8000 --glossary-oversample 2 --lr $WINNING_LR --max-steps $WINNING_STEPS $WINNING_EXTRAS --sermon-data $SERMON_DATA"
TRAIN_ARGS[S2_verse50_sermon45]="--max-pairs 4000 --glossary-oversample 2 --lr $WINNING_LR --max-steps $WINNING_STEPS $WINNING_EXTRAS --sermon-data $SERMON_DATA"
TRAIN_ARGS[S3_sermon_only]="--max-pairs 0 --glossary-oversample 2 --lr $WINNING_LR --max-steps $WINNING_STEPS $WINNING_EXTRAS --sermon-data $SERMON_DATA"

PASSED=0
FAILED=0

for RUN in "${RUNS[@]}"; do
    OUTDIR="$SCALE_DIR/$RUN"
    METRICS="$SCALE_DIR/${RUN}_metrics.json"
    EXTRA="${TRAIN_ARGS[$RUN]}"

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" | tee -a "$LOG"
    echo "[$RUN] Starting at $(date)" | tee -a "$LOG"
    echo "  Train args: $EXTRA" | tee -a "$LOG"
    echo "  Output dir: $OUTDIR" | tee -a "$LOG"
    echo "" | tee -a "$LOG"

    # --- Training ---
    TRAIN_START=$(date +%s)
    if python training/train_gemma.py A $EXTRA -o "$OUTDIR" 2>&1 | tee -a "$LOG"; then
        TRAIN_END=$(date +%s)
        TRAIN_ELAPSED=$(( TRAIN_END - TRAIN_START ))
        echo "[$RUN] Training complete in ${TRAIN_ELAPSED}s" | tee -a "$LOG"
    else
        TRAIN_END=$(date +%s)
        TRAIN_ELAPSED=$(( TRAIN_END - TRAIN_START ))
        echo "[$RUN] TRAINING FAILED after ${TRAIN_ELAPSED}s — skipping eval" | tee -a "$LOG"
        FAILED=$((FAILED + 1))
        continue
    fi

    # --- Evaluation (corpus metrics + glossary with regression detection) ---
    EVAL_START=$(date +%s)
    if python training/evaluate_translation.py \
        --adapter "$OUTDIR" --max-samples 500 \
        --output-file "$METRICS" \
        --compare-base 2>&1 | tee -a "$LOG"; then
        EVAL_END=$(date +%s)
        EVAL_ELAPSED=$(( EVAL_END - EVAL_START ))
        echo "[$RUN] Eval complete in ${EVAL_ELAPSED}s" | tee -a "$LOG"
        PASSED=$((PASSED + 1))
    else
        EVAL_END=$(date +%s)
        EVAL_ELAPSED=$(( EVAL_END - EVAL_START ))
        echo "[$RUN] EVAL FAILED after ${EVAL_ELAPSED}s" | tee -a "$LOG"
        FAILED=$((FAILED + 1))
    fi

    echo "" | tee -a "$LOG"
done

# --- Summary ---
echo "" | tee -a "$LOG"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" | tee -a "$LOG"
echo "=== SCALE-UP SUMMARY — $(date) ===" | tee -a "$LOG"
echo "Passed: $PASSED / ${#RUNS[@]}   Failed: $FAILED" | tee -a "$LOG"
echo "" | tee -a "$LOG"

# Print results table (COMET-first)
printf "%-25s %8s %8s %8s\n" "Run" "COMET" "chrF++" "BLEU" | tee -a "$LOG"
printf "%-25s %8s %8s %8s\n" "---" "-----" "------" "----" | tee -a "$LOG"

for RUN in "${RUNS[@]}"; do
    METRICS="$SCALE_DIR/${RUN}_metrics.json"
    if [[ -f "$METRICS" ]]; then
        COMET=$(python -c "import json; d=json.load(open('$METRICS')); c=d.get('comet'); print(f'{c:.3f}' if c else 'N/A')")
        CHRF=$(python -c "import json; d=json.load(open('$METRICS')); print(f\"{d['chrf']:.1f}\")")
        BLEU=$(python -c "import json; d=json.load(open('$METRICS')); print(f\"{d['bleu']:.1f}\")")
        printf "%-25s %8s %8s %8s\n" "$RUN" "$COMET" "$CHRF" "$BLEU" | tee -a "$LOG"
    else
        printf "%-25s %8s %8s %8s\n" "$RUN" "FAIL" "FAIL" "FAIL" | tee -a "$LOG"
    fi
done

echo "" | tee -a "$LOG"
echo "Acceptance gates (COMET-primary):" | tee -a "$LOG"
echo "  Floor:   COMET > 0.740" | tee -a "$LOG"
echo "  Minimum: COMET > 0.752 (beat A1)" | tee -a "$LOG"
echo "  Target:  COMET > 0.770" | tee -a "$LOG"
echo "Metrics files: $SCALE_DIR/*_metrics.json" | tee -a "$LOG"
echo "Full log: $LOG" | tee -a "$LOG"
echo "=== Done ===" | tee -a "$LOG"
