#!/usr/bin/env bash
# run_b_series.sh — Phase 2 B-Series: Combine A-series winners
#
# 5 runs exploring the space between A1 (50 steps, lr=1e-5) and A4 (full epoch, lr=1e-6).
# All runs use lr=1e-6 (the dominant variable from Phase 1).
# Sequential train+eval, ~7.8 hours total. B1+B2 finish in ~1.8 hrs for early signal.
#
# Usage:
#   bash training/run_b_series.sh          # from project root
#   nohup bash training/run_b_series.sh &  # background / overnight

set -euo pipefail

# --- Setup ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

VENV="/home/wbell/stt_train_env/bin/activate"
ABLATION_DIR="ablation"
LOG="$ABLATION_DIR/b_series_log.txt"

source "$VENV"
mkdir -p "$ABLATION_DIR"

echo "=== Phase 2 B-Series — $(date) ===" | tee "$LOG"
echo "Project dir: $PROJECT_DIR" | tee -a "$LOG"
echo "Python: $(which python)" | tee -a "$LOG"
echo "" | tee -a "$LOG"

# --- Run definitions ---
declare -a RUNS=(
    "B1_lr1e6_50"
    "B2_lr1e6_150"
    "B3_lr1e6_nodrop"
    "B4_lr1e6_neftune"
    "B5_lr1e6_replay"
)

declare -A TRAIN_ARGS
TRAIN_ARGS[B1_lr1e6_50]="--max-pairs 8000 --lr 1e-6 --max-steps 50"
TRAIN_ARGS[B2_lr1e6_150]="--max-pairs 8000 --lr 1e-6 --max-steps 150"
TRAIN_ARGS[B3_lr1e6_nodrop]="--max-pairs 8000 --lr 1e-6 --lora-dropout 0"
TRAIN_ARGS[B4_lr1e6_neftune]="--max-pairs 8000 --lr 1e-6 --neftune 5"
TRAIN_ARGS[B5_lr1e6_replay]="--max-pairs 8000 --lr 1e-6 --replay-ratio 0.2"

PASSED=0
FAILED=0

# --- Check A6 results for B5 conditional ---
A6_METRICS="$ABLATION_DIR/A6_replay20_metrics.json"
A6_NOTE=""
if [[ -f "$A6_METRICS" ]]; then
    A6_BLEU=$(python -c "import json; d=json.load(open('$A6_METRICS')); print(f\"{d['bleu']:.1f}\")")
    echo "A6 replay BLEU: $A6_BLEU (threshold: 17.7)" | tee -a "$LOG"
    A6_GOOD=$(python -c "import json; d=json.load(open('$A6_METRICS')); print('yes' if d['bleu'] > 17.7 else 'no')")
    if [[ "$A6_GOOD" == "no" ]]; then
        A6_NOTE="A6 replay scored $A6_BLEU (below 17.7) — B5 replay run included but may not help"
        echo "WARNING: $A6_NOTE" | tee -a "$LOG"
    fi
else
    A6_NOTE="A6 metrics not found — B5 replay run included but A6 results unknown"
    echo "NOTE: $A6_NOTE" | tee -a "$LOG"
fi
echo "" | tee -a "$LOG"

for RUN in "${RUNS[@]}"; do
    OUTDIR="$ABLATION_DIR/$RUN"
    METRICS="$ABLATION_DIR/${RUN}_metrics.json"
    EXTRA="${TRAIN_ARGS[$RUN]}"

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" | tee -a "$LOG"
    echo "[$RUN] Starting at $(date)" | tee -a "$LOG"
    echo "  Train args: $EXTRA" | tee -a "$LOG"
    echo "  Output dir: $OUTDIR" | tee -a "$LOG"

    # Log A6 context for B5
    if [[ "$RUN" == "B5_lr1e6_replay" && -n "$A6_NOTE" ]]; then
        echo "  NOTE: $A6_NOTE" | tee -a "$LOG"
    fi
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
echo "=== B-SERIES SUMMARY — $(date) ===" | tee -a "$LOG"
echo "Passed: $PASSED / ${#RUNS[@]}   Failed: $FAILED" | tee -a "$LOG"
echo "" | tee -a "$LOG"

# Print results table from metrics JSONs
printf "%-20s %8s %8s %8s\n" "Run" "BLEU" "chrF++" "COMET" | tee -a "$LOG"
printf "%-20s %8s %8s %8s\n" "---" "----" "------" "-----" | tee -a "$LOG"

# Include A-series winners for comparison
for RUN in "A1_steps50" "A4_lr1e6" "${RUNS[@]}"; do
    METRICS="$ABLATION_DIR/${RUN}_metrics.json"
    if [[ -f "$METRICS" ]]; then
        BLEU=$(python -c "import json; d=json.load(open('$METRICS')); print(f\"{d['bleu']:.1f}\")")
        CHRF=$(python -c "import json; d=json.load(open('$METRICS')); print(f\"{d['chrf']:.1f}\")")
        COMET=$(python -c "import json; d=json.load(open('$METRICS')); c=d.get('comet'); print(f'{c:.3f}' if c else 'N/A')")
        printf "%-20s %8s %8s %8s\n" "$RUN" "$BLEU" "$CHRF" "$COMET" | tee -a "$LOG"
    else
        printf "%-20s %8s %8s %8s\n" "$RUN" "FAIL" "FAIL" "FAIL" | tee -a "$LOG"
    fi
done

echo "" | tee -a "$LOG"
echo "Base reference: BLEU 19.7 (threshold 17.7)" | tee -a "$LOG"
echo "A1 reference:   BLEU 20.4 (50 steps, lr=1e-5)" | tee -a "$LOG"
echo "A4 reference:   BLEU 20.7 (full epoch, lr=1e-6)" | tee -a "$LOG"
echo "Metrics files: $ABLATION_DIR/B*_metrics.json" | tee -a "$LOG"
echo "Full log: $LOG" | tee -a "$LOG"
echo "=== Done ===" | tee -a "$LOG"
