#!/usr/bin/env bash
# run_ablation.sh — Phase 1 Ablation: 6-run train+eval sweep
#
# Runs all 6 ablation experiments sequentially, logging to ablation/ablation_log.txt.
# Launch and walk away (~9.5 hours total).
#
# Usage:
#   bash training/run_ablation.sh          # from project root
#   nohup bash training/run_ablation.sh &  # background / overnight

set -euo pipefail

# --- Setup ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

VENV="/home/wbell/stt_train_env/bin/activate"
ABLATION_DIR="ablation"
LOG="$ABLATION_DIR/ablation_log.txt"

source "$VENV"
mkdir -p "$ABLATION_DIR"

echo "=== Phase 1 Ablation — $(date) ===" | tee "$LOG"
echo "Project dir: $PROJECT_DIR" | tee -a "$LOG"
echo "Python: $(which python)" | tee -a "$LOG"
echo "" | tee -a "$LOG"

# --- Run definitions ---
# Format: NAME OUTDIR EXTRA_TRAIN_ARGS...
declare -a RUNS=(
    "A1_steps50"
    "A2_steps150"
    "A3_lr5e6"
    "A4_lr1e6"
    "A5_rank4"
    "A6_replay20"
)

declare -A TRAIN_ARGS
TRAIN_ARGS[A1_steps50]="--max-pairs 8000 --max-steps 50"
TRAIN_ARGS[A2_steps150]="--max-pairs 8000 --max-steps 150"
TRAIN_ARGS[A3_lr5e6]="--max-pairs 8000 --lr 5e-6"
TRAIN_ARGS[A4_lr1e6]="--max-pairs 8000 --lr 1e-6"
TRAIN_ARGS[A5_rank4]="--max-pairs 8000 --lora-r 4 --lora-alpha 8"
TRAIN_ARGS[A6_replay20]="--max-pairs 8000 --replay-ratio 0.2"

PASSED=0
FAILED=0

for RUN in "${RUNS[@]}"; do
    OUTDIR="$ABLATION_DIR/$RUN"
    METRICS="$ABLATION_DIR/${RUN}_metrics.json"
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
echo "=== ABLATION SUMMARY — $(date) ===" | tee -a "$LOG"
echo "Passed: $PASSED / ${#RUNS[@]}   Failed: $FAILED" | tee -a "$LOG"
echo "" | tee -a "$LOG"

# Print results table from metrics JSONs
printf "%-15s %8s %8s %8s\n" "Run" "BLEU" "chrF++" "COMET" | tee -a "$LOG"
printf "%-15s %8s %8s %8s\n" "---" "----" "------" "-----" | tee -a "$LOG"

for RUN in "${RUNS[@]}"; do
    METRICS="$ABLATION_DIR/${RUN}_metrics.json"
    if [[ -f "$METRICS" ]]; then
        BLEU=$(python -c "import json; d=json.load(open('$METRICS')); print(f\"{d['bleu']:.1f}\")")
        CHRF=$(python -c "import json; d=json.load(open('$METRICS')); print(f\"{d['chrf']:.1f}\")")
        COMET=$(python -c "import json; d=json.load(open('$METRICS')); c=d.get('comet'); print(f'{c:.3f}' if c else 'N/A')")
        printf "%-15s %8s %8s %8s\n" "$RUN" "$BLEU" "$CHRF" "$COMET" | tee -a "$LOG"
    else
        printf "%-15s %8s %8s %8s\n" "$RUN" "FAIL" "FAIL" "FAIL" | tee -a "$LOG"
    fi
done

echo "" | tee -a "$LOG"
echo "Base reference: BLEU 19.7 (threshold 17.7)" | tee -a "$LOG"
echo "Metrics files: $ABLATION_DIR/*_metrics.json" | tee -a "$LOG"
echo "Full log: $LOG" | tee -a "$LOG"
echo "=== Done ===" | tee -a "$LOG"
