#!/usr/bin/env bash
# run_whisper_ablation.sh — Whisper LoRA Ablation: W1-W6 sweep + W7-W9 scaling
#
# Phase W1: 6-run ablation to find dominant variable (lr, targets, replay, data size).
# Phase W2: 3-run scaling with winner config (epochs, rank).
#
# Each run trains a Whisper LoRA adapter and extracts WER from all_results.json.
# Launch and walk away (~22 hours total for all 9 runs).
#
# Usage:
#   bash training/run_whisper_ablation.sh
#   nohup bash training/run_whisper_ablation.sh &

set -euo pipefail

# --- Setup ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

VENV="/home/wbell/stt_train_env/bin/activate"
source "$VENV"

ABLATION_DIR="whisper_ablation"
LOG="$ABLATION_DIR/ablation_log.txt"
DS="stark_data/whisper_dataset_deepgram"
DS_5K="stark_data/whisper_dataset_deepgram_5k"

mkdir -p "$ABLATION_DIR"

echo "=== Whisper LoRA Ablation (W1-W6 + W7-W9) — $(date) ===" | tee "$LOG"
echo "Project dir: $PROJECT_DIR" | tee -a "$LOG"
echo "Python: $(which python)" | tee -a "$LOG"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>&1 | tee -a "$LOG"
echo "" | tee -a "$LOG"

PIPELINE_START=$(date +%s)

# --- VRAM logging helper ---
log_vram() {
    local label="$1"
    echo "[VRAM] $label — $(date '+%H:%M:%S')" | tee -a "$LOG"
    nvidia-smi --query-gpu=memory.used,memory.free,memory.total \
        --format=csv,noheader,nounits 2>/dev/null | \
        awk -F', ' '{printf "[VRAM]   Used: %s MiB / %s MiB total (%.0f%% free)\n", $1, $3, $2/$3*100}' | \
        tee -a "$LOG"
}

# --- WER extraction helper ---
extract_wer() {
    local results_file="$1"
    if [[ -f "$results_file" ]]; then
        python -c "
import json
d = json.load(open('$results_file'))
wer = d.get('eval_wer', d.get('wer', None))
if wer is not None:
    print(f'{wer:.2f}')
else:
    print('N/A')
"
    else
        echo "FAIL"
    fi
}

PASSED=0
FAILED=0

# ============================================================
# Phase W1: Ablation Sweep (W1-W6)
# ============================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" | tee -a "$LOG"
echo "PHASE W1: Ablation Sweep (W1-W6) — $(date)" | tee -a "$LOG"
echo "" | tee -a "$LOG"

# --- Run definitions ---
declare -a W1_RUNS=(
    "W1_baseline"
    "W2_lr5e5"
    "W3_lr1e5"
    "W4_qkvo"
    "W5_noreplay"
    "W6_5k"
)

declare -A W1_ARGS
W1_ARGS[W1_baseline]="--dataset $DS --lr 1e-4 --epochs 1 --target-modules q_proj v_proj --lora-r 32 --lora-alpha 64 --replay-ratio 0.3"
W1_ARGS[W2_lr5e5]="--dataset $DS --lr 5e-5 --epochs 1 --target-modules q_proj v_proj --lora-r 32 --lora-alpha 64 --replay-ratio 0.3"
W1_ARGS[W3_lr1e5]="--dataset $DS --lr 1e-5 --epochs 1 --target-modules q_proj v_proj --lora-r 32 --lora-alpha 64 --replay-ratio 0.3"
W1_ARGS[W4_qkvo]="--dataset $DS --lr 1e-4 --epochs 1 --target-modules q_proj k_proj v_proj out_proj --lora-r 32 --lora-alpha 64 --replay-ratio 0.3"
W1_ARGS[W5_noreplay]="--dataset $DS --lr 1e-4 --epochs 1 --target-modules q_proj v_proj --lora-r 32 --lora-alpha 64 --replay-ratio 0"
W1_ARGS[W6_5k]="--dataset $DS_5K --lr 1e-4 --epochs 1 --target-modules q_proj v_proj --lora-r 32 --lora-alpha 64 --replay-ratio 0.3"

declare -A W1_DESC
W1_DESC[W1_baseline]="lr=1e-4, 1 epoch, qv, replay=0.3 (baseline LoRA config)"
W1_DESC[W2_lr5e5]="lr=5e-5, 1 epoch, qv, replay=0.3 (half lr)"
W1_DESC[W3_lr1e5]="lr=1e-5, 1 epoch, qv, replay=0.3 (ultra-conservative lr)"
W1_DESC[W4_qkvo]="lr=1e-4, 1 epoch, qkvo, replay=0.3 (expanded targets)"
W1_DESC[W5_noreplay]="lr=1e-4, 1 epoch, qv, replay=0 (no anti-forgetting)"
W1_DESC[W6_5k]="lr=1e-4, 1 epoch, qv, replay=0.3, dataset=5k subset"

for RUN in "${W1_RUNS[@]}"; do
    OUTDIR="$ABLATION_DIR/$RUN"
    EXTRA="${W1_ARGS[$RUN]}"
    DESC="${W1_DESC[$RUN]}"

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" | tee -a "$LOG"
    echo "[$RUN] Starting at $(date)" | tee -a "$LOG"
    echo "  Config: $DESC" | tee -a "$LOG"
    echo "  Args: $EXTRA" | tee -a "$LOG"
    echo "  Output dir: $OUTDIR" | tee -a "$LOG"
    echo "" | tee -a "$LOG"

    # --- Training ---
    TRAIN_START=$(date +%s)
    if python training/train_whisper.py $EXTRA -o "$OUTDIR" 2>&1 | tee -a "$LOG"; then
        TRAIN_END=$(date +%s)
        TRAIN_ELAPSED=$(( TRAIN_END - TRAIN_START ))
        echo "[$RUN] Training complete in ${TRAIN_ELAPSED}s ($(( TRAIN_ELAPSED / 60 )) min)" | tee -a "$LOG"
        log_vram "After $RUN training"
        PASSED=$((PASSED + 1))
    else
        TRAIN_END=$(date +%s)
        TRAIN_ELAPSED=$(( TRAIN_END - TRAIN_START ))
        echo "[$RUN] TRAINING FAILED after ${TRAIN_ELAPSED}s — skipping" | tee -a "$LOG"
        log_vram "After $RUN failure"
        FAILED=$((FAILED + 1))
    fi

    # Extract WER from all_results.json
    RESULTS_FILE="$OUTDIR/all_results.json"
    WER=$(extract_wer "$RESULTS_FILE")
    echo "[$RUN] WER: $WER" | tee -a "$LOG"
    echo "" | tee -a "$LOG"
done

# ============================================================
# Phase W1 Summary
# ============================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" | tee -a "$LOG"
echo "PHASE W1 SUMMARY — $(date)" | tee -a "$LOG"
echo "Passed: $PASSED / ${#W1_RUNS[@]}   Failed: $FAILED" | tee -a "$LOG"
echo "" | tee -a "$LOG"

printf "%-20s %10s %12s\n" "Run" "WER" "Description" | tee -a "$LOG"
printf "%-20s %10s %12s\n" "---" "---" "-----------" | tee -a "$LOG"

for RUN in "${W1_RUNS[@]}"; do
    RESULTS_FILE="$ABLATION_DIR/$RUN/all_results.json"
    WER=$(extract_wer "$RESULTS_FILE")
    DESC="${W1_DESC[$RUN]}"
    printf "%-20s %10s  %s\n" "$RUN" "$WER" "$DESC" | tee -a "$LOG"
done

echo "" | tee -a "$LOG"

# Print best run
python -c "
import json, os

ablation_dir = '$ABLATION_DIR'
runs = ['W1_baseline', 'W2_lr5e5', 'W3_lr1e5', 'W4_qkvo', 'W5_noreplay', 'W6_5k']

results = {}
for run in runs:
    path = os.path.join(ablation_dir, run, 'all_results.json')
    if os.path.exists(path):
        d = json.load(open(path))
        wer = d.get('eval_wer', d.get('wer'))
        if wer is not None:
            results[run] = wer

if results:
    best = min(results, key=results.get)
    print(f'BEST W1-W6: {best} (WER = {results[best]:.2f}%)')
    print()
    # Decision logic
    if best == 'W1_baseline':
        print('  --> Default config works. Scale to 3 epochs (W7).')
    elif best in ('W2_lr5e5', 'W3_lr1e5'):
        print(f'  --> LR-limited: {best} won. Same dynamic as TranslateGemma.')
    elif best == 'W4_qkvo':
        print('  --> Expanded targets help. Combine with lr winner.')
    elif best == 'W5_noreplay':
        print('  --> No replay needed. Domain close enough to general English.')
    elif best == 'W6_5k':
        print('  --> 5K matches full data. Diminishing returns — focus on quality.')
    print()
    print('ACTION: Set WINNER_* variables below for Phase W2 scaling.')
else:
    print('BEST W1-W6: (no results available)')
print()
" 2>&1 | tee -a "$LOG"

# ============================================================
# Phase W2: Scale Winner (W7-W9) — TBD
# ============================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" | tee -a "$LOG"
echo "PHASE W2: Scale Winner (W7-W9) — TBD" | tee -a "$LOG"
echo "" | tee -a "$LOG"

# --- W1-W6 WINNER CONFIG (fill after Phase W1 completes) ---
WINNER_LR="TBD"
WINNER_TARGETS="TBD"
WINNER_REPLAY="TBD"
WINNER_RANK=32

if [[ "$WINNER_LR" == "TBD" || "$WINNER_TARGETS" == "TBD" || "$WINNER_REPLAY" == "TBD" ]]; then
    echo "SKIPPING Phase W2: Winner config not set." | tee -a "$LOG"
    echo "  1. Review W1-W6 results above" | tee -a "$LOG"
    echo "  2. Edit WINNER_LR, WINNER_TARGETS, WINNER_REPLAY in this script" | tee -a "$LOG"
    echo "  3. Re-run this script (W1-W6 will be skipped if output dirs exist)" | tee -a "$LOG"
    echo "" | tee -a "$LOG"
    echo "Phase W2 runs (once winner is set):" | tee -a "$LOG"
    echo "  W7: winner + 3 epochs (standard training duration)" | tee -a "$LOG"
    echo "  W8: winner + 5 epochs (overfitting check)" | tee -a "$LOG"
    echo "  W9: winner + r=64, alpha=128 (higher rank capacity)" | tee -a "$LOG"
    echo "" | tee -a "$LOG"
else
    echo "Winner config: lr=$WINNER_LR, targets=$WINNER_TARGETS, replay=$WINNER_REPLAY, rank=$WINNER_RANK" | tee -a "$LOG"
    echo "" | tee -a "$LOG"

    declare -a W2_RUNS=(
        "W7_3epochs"
        "W8_5epochs"
        "W9_rank64"
    )

    declare -A W2_ARGS
    W2_ARGS[W7_3epochs]="--dataset $DS --lr $WINNER_LR --epochs 3 --target-modules $WINNER_TARGETS --lora-r $WINNER_RANK --lora-alpha $(( WINNER_RANK * 2 )) --replay-ratio $WINNER_REPLAY"
    W2_ARGS[W8_5epochs]="--dataset $DS --lr $WINNER_LR --epochs 5 --target-modules $WINNER_TARGETS --lora-r $WINNER_RANK --lora-alpha $(( WINNER_RANK * 2 )) --replay-ratio $WINNER_REPLAY"
    W2_ARGS[W9_rank64]="--dataset $DS --lr $WINNER_LR --epochs 3 --target-modules $WINNER_TARGETS --lora-r 64 --lora-alpha 128 --replay-ratio $WINNER_REPLAY"

    declare -A W2_DESC
    W2_DESC[W7_3epochs]="winner + 3 epochs (standard training duration)"
    W2_DESC[W8_5epochs]="winner + 5 epochs (overfitting check)"
    W2_DESC[W9_rank64]="winner + r=64, alpha=128 (higher rank capacity)"

    W2_PASSED=0
    W2_FAILED=0

    for RUN in "${W2_RUNS[@]}"; do
        OUTDIR="$ABLATION_DIR/$RUN"
        EXTRA="${W2_ARGS[$RUN]}"
        DESC="${W2_DESC[$RUN]}"

        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" | tee -a "$LOG"
        echo "[$RUN] Starting at $(date)" | tee -a "$LOG"
        echo "  Config: $DESC" | tee -a "$LOG"
        echo "  Args: $EXTRA" | tee -a "$LOG"
        echo "  Output dir: $OUTDIR" | tee -a "$LOG"
        echo "" | tee -a "$LOG"

        TRAIN_START=$(date +%s)
        if python training/train_whisper.py $EXTRA -o "$OUTDIR" 2>&1 | tee -a "$LOG"; then
            TRAIN_END=$(date +%s)
            TRAIN_ELAPSED=$(( TRAIN_END - TRAIN_START ))
            echo "[$RUN] Training complete in ${TRAIN_ELAPSED}s ($(( TRAIN_ELAPSED / 60 )) min)" | tee -a "$LOG"
            log_vram "After $RUN training"
            W2_PASSED=$((W2_PASSED + 1))
        else
            TRAIN_END=$(date +%s)
            TRAIN_ELAPSED=$(( TRAIN_END - TRAIN_START ))
            echo "[$RUN] TRAINING FAILED after ${TRAIN_ELAPSED}s" | tee -a "$LOG"
            log_vram "After $RUN failure"
            W2_FAILED=$((W2_FAILED + 1))
        fi

        RESULTS_FILE="$OUTDIR/all_results.json"
        WER=$(extract_wer "$RESULTS_FILE")
        echo "[$RUN] WER: $WER" | tee -a "$LOG"
        echo "" | tee -a "$LOG"
    done

    # Phase W2 Summary
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" | tee -a "$LOG"
    echo "PHASE W2 SUMMARY — $(date)" | tee -a "$LOG"
    echo "Passed: $W2_PASSED / ${#W2_RUNS[@]}   Failed: $W2_FAILED" | tee -a "$LOG"
    echo "" | tee -a "$LOG"

    printf "%-20s %10s %12s\n" "Run" "WER" "Description" | tee -a "$LOG"
    printf "%-20s %10s %12s\n" "---" "---" "-----------" | tee -a "$LOG"

    for RUN in "${W2_RUNS[@]}"; do
        RESULTS_FILE="$ABLATION_DIR/$RUN/all_results.json"
        WER=$(extract_wer "$RESULTS_FILE")
        DESC="${W2_DESC[$RUN]}"
        printf "%-20s %10s  %s\n" "$RUN" "$WER" "$DESC" | tee -a "$LOG"
    done
    echo "" | tee -a "$LOG"
fi

# ============================================================
# Final Summary: All Runs (W1-W9)
# ============================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" | tee -a "$LOG"
echo "FULL SUMMARY — $(date)" | tee -a "$LOG"
echo "" | tee -a "$LOG"

python -c "
import json, os

ablation_dir = '$ABLATION_DIR'
all_runs = [
    ('W1_baseline',  'lr=1e-4, qv, replay=0.3'),
    ('W2_lr5e5',     'lr=5e-5, qv, replay=0.3'),
    ('W3_lr1e5',     'lr=1e-5, qv, replay=0.3'),
    ('W4_qkvo',      'lr=1e-4, qkvo, replay=0.3'),
    ('W5_noreplay',  'lr=1e-4, qv, replay=0'),
    ('W6_5k',        'lr=1e-4, qv, replay=0.3, 5k'),
    ('W7_3epochs',   'winner + 3 epochs'),
    ('W8_5epochs',   'winner + 5 epochs'),
    ('W9_rank64',    'winner + r=64'),
]

print('=' * 80)
print('WHISPER LoRA ABLATION — FULL RESULTS')
print('=' * 80)
print()

printf_fmt = '%-20s %10s  %s'
print(f\"{'Run':<20} {'WER':>10}  {'Config'}\")
print(f\"{'---':<20} {'---':>10}  {'------'}\")

results = {}
for run_name, desc in all_runs:
    path = os.path.join(ablation_dir, run_name, 'all_results.json')
    if os.path.exists(path):
        d = json.load(open(path))
        wer = d.get('eval_wer', d.get('wer'))
        if wer is not None:
            results[run_name] = wer
            print(f'{run_name:<20} {wer:>10.2f}  {desc}')
        else:
            print(f'{run_name:<20} {\"N/A\":>10}  {desc}')
    else:
        print(f'{run_name:<20} {\"--\":>10}  {desc}')

print()

if results:
    best = min(results, key=results.get)
    worst = max(results, key=results.get)
    print(f'BEST:  {best} (WER = {results[best]:.2f}%)')
    print(f'WORST: {worst} (WER = {results[worst]:.2f}%)')
    print()

    # Check W1-W6 vs W7-W9
    w1_runs = {k: v for k, v in results.items() if k.startswith(('W1', 'W2', 'W3', 'W4', 'W5', 'W6'))}
    w2_runs = {k: v for k, v in results.items() if k.startswith(('W7', 'W8', 'W9'))}

    if w1_runs:
        best_w1 = min(w1_runs, key=w1_runs.get)
        print(f'Best Phase W1: {best_w1} (WER = {w1_runs[best_w1]:.2f}%)')
    if w2_runs:
        best_w2 = min(w2_runs, key=w2_runs.get)
        print(f'Best Phase W2: {best_w2} (WER = {w2_runs[best_w2]:.2f}%)')
        # Scaling interpretation
        if w1_runs:
            delta = w2_runs[best_w2] - w1_runs[best_w1]
            if delta < -1.0:
                print(f'  --> Scaling improved WER by {abs(delta):.2f}% abs. Ship {best_w2}.')
            elif delta < 1.0:
                print(f'  --> Scaling had minimal effect ({delta:+.2f}%). Ship best overall.')
            else:
                print(f'  --> Scaling REGRESSED WER by {delta:.2f}% abs. Overfitting likely.')
    print()

    # Go/No-Go gates
    print('GO/NO-GO GATES (vs base):')
    print('  Floor:   WER < base')
    print('  Minimum: > 10% relative WER reduction')
    print('  Target:  > 20% relative WER reduction')
    print('  Kill:    > 5% absolute WER regression on general English')
    print()
print()
" 2>&1 | tee -a "$LOG"

PIPELINE_END=$(date +%s)
TOTAL_ELAPSED=$(( PIPELINE_END - PIPELINE_START ))
echo "" | tee -a "$LOG"
echo "=== Whisper ablation complete — total ${TOTAL_ELAPSED}s ($(( TOTAL_ELAPSED / 60 )) min) — $(date) ===" | tee -a "$LOG"
echo "" | tee -a "$LOG"
echo "Results: $ABLATION_DIR/*/all_results.json" | tee -a "$LOG"
echo "Full log: $LOG" | tee -a "$LOG"
echo "=== Done ===" | tee -a "$LOG"
