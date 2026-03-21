#!/usr/bin/env bash
# run_scale.sh — Phase 3 Scale-Up: Test data scaling with winning config
#
# 4 runs (S2a/S2b conditional), testing 20K and 50K pairs with the winning
# ablation/B-series config. Fill in placeholders after B-series completes.
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

source "$VENV"
mkdir -p "$SCALE_DIR"

# ============================================================
# FILL IN AFTER B-SERIES COMPLETES
# ============================================================
WINNING_LR="1e-6"
WINNING_STEPS="TBD"           # e.g., 150 or 1114
SCALED_STEPS_20K="TBD"        # proportional to 20K epoch fraction
SCALED_STEPS_50K="TBD"        # proportional to 50K epoch fraction
WINNING_EXTRAS=""              # e.g., "--neftune 5 --lora-dropout 0"
# ============================================================

# Validate placeholders are filled in
if [[ "$WINNING_STEPS" == "TBD" ]]; then
    echo "ERROR: WINNING_STEPS is still TBD. Fill in placeholders after B-series completes." >&2
    exit 1
fi

echo "=== Phase 3 Scale-Up — $(date) ===" | tee "$LOG"
echo "Project dir: $PROJECT_DIR" | tee -a "$LOG"
echo "Python: $(which python)" | tee -a "$LOG"
echo "Winning config: lr=$WINNING_LR, steps=$WINNING_STEPS, extras='$WINNING_EXTRAS'" | tee -a "$LOG"
echo "" | tee -a "$LOG"

# --- Run definitions ---
declare -a RUNS=(
    "S1a_20k_same_steps"
    "S1b_20k_scaled"
    # Uncomment S2 runs if S1 shows BLEU improvement > +2 over ablation best
    # "S2a_50k_same_steps"
    # "S2b_50k_scaled"
)

declare -A TRAIN_ARGS
TRAIN_ARGS[S1a_20k_same_steps]="--max-pairs 20000 --glossary-oversample 2 --lr $WINNING_LR --max-steps $WINNING_STEPS $WINNING_EXTRAS"
TRAIN_ARGS[S1b_20k_scaled]="--max-pairs 20000 --glossary-oversample 2 --lr $WINNING_LR --max-steps $SCALED_STEPS_20K $WINNING_EXTRAS"
TRAIN_ARGS[S2a_50k_same_steps]="--max-pairs 50000 --glossary-oversample 2 --lr $WINNING_LR --max-steps $WINNING_STEPS $WINNING_EXTRAS"
TRAIN_ARGS[S2b_50k_scaled]="--max-pairs 50000 --glossary-oversample 2 --lr $WINNING_LR --max-steps $SCALED_STEPS_50K $WINNING_EXTRAS"

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

    # --- Evaluation ---
    EVAL_START=$(date +%s)
    if python training/evaluate_translation.py \
        --adapter "$OUTDIR" --max-samples 500 \
        --output-file "$METRICS" 2>&1 | tee -a "$LOG"; then
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

# Print results table
printf "%-25s %8s %8s %8s\n" "Run" "BLEU" "chrF++" "COMET" | tee -a "$LOG"
printf "%-25s %8s %8s %8s\n" "---" "----" "------" "-----" | tee -a "$LOG"

for RUN in "${RUNS[@]}"; do
    METRICS="$SCALE_DIR/${RUN}_metrics.json"
    if [[ -f "$METRICS" ]]; then
        BLEU=$(python -c "import json; d=json.load(open('$METRICS')); print(f\"{d['bleu']:.1f}\")")
        CHRF=$(python -c "import json; d=json.load(open('$METRICS')); print(f\"{d['chrf']:.1f}\")")
        COMET=$(python -c "import json; d=json.load(open('$METRICS')); c=d.get('comet'); print(f'{c:.3f}' if c else 'N/A')")
        printf "%-25s %8s %8s %8s\n" "$RUN" "$BLEU" "$CHRF" "$COMET" | tee -a "$LOG"
    else
        printf "%-25s %8s %8s %8s\n" "$RUN" "FAIL" "FAIL" "FAIL" | tee -a "$LOG"
    fi
done

echo "" | tee -a "$LOG"
echo "Acceptance gates:" | tee -a "$LOG"
echo "  Floor:   BLEU > 17.7, COMET > 0.720" | tee -a "$LOG"
echo "  Minimum: BLEU > ablation best, COMET > 0.740" | tee -a "$LOG"
echo "  Target:  BLEU > ablation best + 2, COMET > 0.770" | tee -a "$LOG"
echo "Metrics files: $SCALE_DIR/*_metrics.json" | tee -a "$LOG"
echo "Full log: $LOG" | tee -a "$LOG"
echo "=== Done ===" | tee -a "$LOG"
