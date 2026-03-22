#!/usr/bin/env bash
# run_hybrid_ratio.sh — Phase 3b: Sermon-Dominant Ratio Sweep (S4/S5/S6)
#
# Uses the WINNING training config from S1/S2/S3, varies only verse/sermon ratio.
# Hypothesis: model needs sermon-dominant data, verse pairs may dilute signal.
#
# Data mix (all use 1800 sermon pairs from expanded hybrid generation):
#   S4: 0 verse + 507 glossary + 1800 sermon = ~2,307 (78% sermon)
#       Hypothesis: If verse pairs are the poison, removing them entirely
#       should unlock the sermon signal. S1's +0.0019 DeepL gain with only
#       11.5% sermon → what happens at 78%?
#   S5: 500 verse + 507 glossary + 1800 sermon = ~2,807 (64% sermon)
#       Hypothesis: A small verse anchor prevents catastrophic forgetting
#       of general translation patterns while keeping sermon dominant.
#   S6: 1800 verse + 507 glossary + 1800 sermon = ~4,107 (44% sermon)
#       Hypothesis: Control — equal ratio. If this matches S1/S2, ratio
#       doesn't matter and we just need more sermon data.
#
# Requires: DEEPL_KEY env var, S1/S2/S3 winner config set below
#
# Usage:
#   export DEEPL_KEY="your-key-here"
#   bash training/run_hybrid_ratio.sh
#   nohup bash training/run_hybrid_ratio.sh &

set -euo pipefail

# --- Setup ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

VENV="/home/wbell/stt_train_env/bin/activate"
source "$VENV"

HYBRID_DIR="hybrid_runs"
LOG="$HYBRID_DIR/ratio_sweep_log.txt"
SERMON_DATA="bible_data/synthetic/hybrid_sermon_pairs_1800.jsonl"
CHUNKS="ablation/sermon_test_chunks_v2.json"
CACHE="ablation/sermon_12b_translations_v2.json"

mkdir -p "$HYBRID_DIR"

# --- S1/S2/S3 WINNER CONFIG ---
# S1 won: COMET prox 12B=-0.0036 (WARN), DeepL=+0.0019 (best of 3)
# S2 tied on 12B but lost theo_terms (4/8 vs 5/8). S3 KILL on both ceilings.
WINNING_LR="1e-5"
WINNING_STEPS="50"
WINNING_EXTRAS=""

# --- Validate ---
if [[ -z "${DEEPL_KEY:-}" ]]; then
    echo "ERROR: DEEPL_KEY environment variable is not set." >&2
    echo "  export DEEPL_KEY='your-deepl-pro-key'" >&2
    exit 1
fi

if [[ "$WINNING_LR" == "TBD" || "$WINNING_STEPS" == "TBD" ]]; then
    echo "ERROR: S1/S2/S3 winner config not set." >&2
    echo "  Edit WINNING_LR and WINNING_STEPS in this script first." >&2
    echo "  Run 'bash training/run_hybrid_scale.sh' and check results." >&2
    exit 1
fi

echo "=== Phase 3b: Sermon-Dominant Ratio Sweep (S4/S5/S6) — $(date) ===" | tee "$LOG"
echo "Project dir: $PROJECT_DIR" | tee -a "$LOG"
echo "Python: $(which python)" | tee -a "$LOG"
echo "Winner config: lr=$WINNING_LR, steps=$WINNING_STEPS, extras='$WINNING_EXTRAS'" | tee -a "$LOG"
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

# ============================================================
# Step 1: Generate 1800-chunk hybrid data (60/40 12B/DeepL)
# ============================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" | tee -a "$LOG"
echo "[Step 1/5] Generate hybrid synthetic data — $(date)" | tee -a "$LOG"
echo "  60% 12B + 40% DeepL, max 1800 chunks" | tee -a "$LOG"
echo "" | tee -a "$LOG"

GEN_START=$(date +%s)
python training/generate_hybrid_synthetic.py \
    --deepl-key "$DEEPL_KEY" \
    --max-chunks 1800 \
    --ratio-deepl 0.40 \
    --output "$SERMON_DATA" 2>&1 | tee -a "$LOG"
GEN_END=$(date +%s)
GEN_ELAPSED=$(( GEN_END - GEN_START ))
echo "" | tee -a "$LOG"
echo "[Step 1/5] Data generation complete in ${GEN_ELAPSED}s ($(( GEN_ELAPSED / 60 )) min)" | tee -a "$LOG"
log_vram "After data generation (12B model)"
echo "" | tee -a "$LOG"

# Validate output
if [[ ! -f "$SERMON_DATA" ]]; then
    echo "ERROR: Synthetic data not generated at $SERMON_DATA" >&2
    exit 1
fi
PAIR_COUNT=$(wc -l < "$SERMON_DATA")
echo "Generated $PAIR_COUNT hybrid sermon pairs (target: ~1800)" | tee -a "$LOG"
echo "" | tee -a "$LOG"

# Build training args from winner config
TRAIN_ARGS="--lr $WINNING_LR"
if [[ "$WINNING_STEPS" != "-1" ]]; then
    TRAIN_ARGS="$TRAIN_ARGS --max-steps $WINNING_STEPS"
fi
if [[ -n "$WINNING_EXTRAS" ]]; then
    TRAIN_ARGS="$TRAIN_ARGS $WINNING_EXTRAS"
fi

# ============================================================
# Step 2: Train S4 — sermon-only (0 verse pairs, 78% sermon)
# Hypothesis: verse pairs are poison → removing them unlocks sermon signal
# ============================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" | tee -a "$LOG"
echo "[Step 2/5] Train S4 — sermon-only (0 verse, 78% sermon) — $(date)" | tee -a "$LOG"
echo "  Hypothesis: verse pairs are poison; pure sermon should unlock signal" | tee -a "$LOG"
echo "  Args: --max-pairs 0 --glossary-oversample 1 $TRAIN_ARGS" | tee -a "$LOG"
echo "" | tee -a "$LOG"

S4_START=$(date +%s)
python training/train_gemma.py A \
    --max-pairs 0 --glossary-oversample 1 \
    $TRAIN_ARGS \
    --sermon-data "$SERMON_DATA" \
    -o "$HYBRID_DIR/S4_sermon_only" 2>&1 | tee -a "$LOG"
S4_END=$(date +%s)
S4_ELAPSED=$(( S4_END - S4_START ))
echo "[Step 2/5] S4 training complete in ${S4_ELAPSED}s" | tee -a "$LOG"
log_vram "After S4 training (4B QLoRA)"
echo "" | tee -a "$LOG"

# ============================================================
# Step 3: Train S5 — sermon-heavy (500 verse, 64% sermon)
# Hypothesis: small verse anchor prevents catastrophic forgetting while sermon stays dominant
# ============================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" | tee -a "$LOG"
echo "[Step 3/5] Train S5 — sermon-heavy (500 verse, 64% sermon) — $(date)" | tee -a "$LOG"
echo "  Hypothesis: small verse anchor prevents catastrophic forgetting" | tee -a "$LOG"
echo "  Args: --max-pairs 500 --glossary-oversample 1 $TRAIN_ARGS" | tee -a "$LOG"
echo "" | tee -a "$LOG"

S5_START=$(date +%s)
python training/train_gemma.py A \
    --max-pairs 500 --glossary-oversample 1 \
    $TRAIN_ARGS \
    --sermon-data "$SERMON_DATA" \
    -o "$HYBRID_DIR/S5_sermon_heavy" 2>&1 | tee -a "$LOG"
S5_END=$(date +%s)
S5_ELAPSED=$(( S5_END - S5_START ))
echo "[Step 3/5] S5 training complete in ${S5_ELAPSED}s" | tee -a "$LOG"
log_vram "After S5 training (4B QLoRA)"
echo "" | tee -a "$LOG"

# ============================================================
# Step 4: Train S6 — balanced (1800 verse, 44% sermon)
# Hypothesis: control — if this matches S1/S2, ratio doesn't matter; need more data
# ============================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" | tee -a "$LOG"
echo "[Step 4/5] Train S6 — balanced (1800 verse, 44% sermon) — $(date)" | tee -a "$LOG"
echo "  Hypothesis: control — equal ratio; if ~S1/S2, need more data not ratio change" | tee -a "$LOG"
echo "  Args: --max-pairs 1800 --glossary-oversample 1 $TRAIN_ARGS" | tee -a "$LOG"
echo "" | tee -a "$LOG"

S6_START=$(date +%s)
python training/train_gemma.py A \
    --max-pairs 1800 --glossary-oversample 1 \
    $TRAIN_ARGS \
    --sermon-data "$SERMON_DATA" \
    -o "$HYBRID_DIR/S6_balanced" 2>&1 | tee -a "$LOG"
S6_END=$(date +%s)
S6_ELAPSED=$(( S6_END - S6_START ))
echo "[Step 4/5] S6 training complete in ${S6_ELAPSED}s" | tee -a "$LOG"
log_vram "After S6 training (4B QLoRA)"
echo "" | tee -a "$LOG"

# ============================================================
# Step 5: Evaluate each — dual ceiling (12B + DeepL) + glossary
# ============================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" | tee -a "$LOG"
echo "[Step 5/5] Evaluate all runs — $(date)" | tee -a "$LOG"
echo "" | tee -a "$LOG"

for RUN in S4_sermon_only S5_sermon_heavy S6_balanced; do
    echo "--- Evaluating $RUN ---" | tee -a "$LOG"

    # Sermon eval with dual ceiling (12B + DeepL)
    EVAL_START=$(date +%s)
    python training/evaluate_sermon.py \
        --chunks "$CHUNKS" \
        --adapter "$HYBRID_DIR/$RUN" \
        --ceiling-cache "$CACHE" \
        --deepl-key "$DEEPL_KEY" \
        --output "$HYBRID_DIR/${RUN}_sermon_eval.json" 2>&1 | tee -a "$LOG"
    EVAL_END=$(date +%s)
    EVAL_ELAPSED=$(( EVAL_END - EVAL_START ))
    echo "[$RUN] Sermon eval complete in ${EVAL_ELAPSED}s" | tee -a "$LOG"
    log_vram "After $RUN sermon eval (4B base + 4B adapter + 12B ceiling)"
    echo "" | tee -a "$LOG"

    # Glossary regression check
    GLOSS_START=$(date +%s)
    python training/evaluate_translation.py \
        --adapter "$HYBRID_DIR/$RUN" \
        --glossary-only --compare-base 2>&1 | tee -a "$LOG"
    GLOSS_END=$(date +%s)
    GLOSS_ELAPSED=$(( GLOSS_END - GLOSS_START ))
    echo "[$RUN] Glossary eval complete in ${GLOSS_ELAPSED}s" | tee -a "$LOG"
    log_vram "After $RUN glossary eval (4B base + 4B adapter)"
    echo "" | tee -a "$LOG"
done

# ============================================================
# Summary comparison table (S1-S6 combined)
# ============================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" | tee -a "$LOG"
echo "COMPARISON SUMMARY — $(date)" | tee -a "$LOG"
echo "" | tee -a "$LOG"

python -c "
import json, os

# S1-S3 (training config sweep) + S4-S6 (ratio sweep)
s1s3_runs = ['S1_lr1e5_50steps', 'S2_lr3e6_100steps', 'S3_lr1e6_neftune']
s4s6_runs = ['S4_sermon_only', 'S5_sermon_heavy', 'S6_balanced']
all_runs = s1s3_runs + s4s6_runs
hybrid_dir = '$HYBRID_DIR'

print('=' * 110)
print('FULL RESULTS: S1-S3 (config sweep) + S4-S6 (ratio sweep)')
print('=' * 110)
print()

# Data mix context
print('DATA MIX:')
print('  S1/S2/S3: 8000 verse + 507 glossary(2x) + 1200 sermon (11.5% sermon)')
print('  S4:       0 verse    + 507 glossary(1x) + 1800 sermon (78% sermon)')
print('  S5:       500 verse  + 507 glossary(1x) + 1800 sermon (64% sermon)')
print('  S6:       1800 verse + 507 glossary(1x) + 1800 sermon (44% sermon)')
print()

# Load all results
data = {}
for run in all_runs:
    path = os.path.join(hybrid_dir, f'{run}_sermon_eval.json')
    if os.path.exists(path):
        data[run] = json.load(open(path))
    else:
        data[run] = None

labels = {
    'S1_lr1e5_50steps': 'S1(cons)',
    'S2_lr3e6_100steps': 'S2(mid)',
    'S3_lr1e6_neftune': 'S3(aggr)',
    'S4_sermon_only': 'S4(0v)',
    'S5_sermon_heavy': 'S5(500v)',
    'S6_balanced': 'S6(1800v)',
}

def fmt(d, key, dp=4):
    if d is None:
        return 'FAIL'
    m = d.get('metrics', {})
    v = m.get(key)
    return f'{v:.{dp}f}' if v is not None else 'N/A'

# Header
cols = [labels[r] for r in all_runs]
header = f\"{'Metric':<35}\" + ''.join(f'{c:>12}' for c in cols)
print(header)
print('-' * 110)

# COMET proximity (12B)
for key, label in [
    ('comet_proximity_gain', 'COMET prox. gain (12B)'),
    ('comet_adapter_vs_ceiling', '  Adapter vs 12B'),
    ('comet_base_vs_ceiling', '  Base vs 12B'),
]:
    row = f'{label:<35}'
    for run in all_runs:
        row += f'{fmt(data.get(run), key):>12}'
    print(row)
print()

# COMET proximity (DeepL)
for key, label in [
    ('comet_proximity_gain_deepl', 'COMET prox. gain (DeepL)'),
    ('comet_deepl_adapter_vs_ceiling', '  Adapter vs DeepL'),
    ('comet_deepl_base_vs_ceiling', '  Base vs DeepL'),
]:
    row = f'{label:<35}'
    for run in all_runs:
        row += f'{fmt(data.get(run), key):>12}'
    print(row)
print()

# chrF++
for key, label in [
    ('chrf_proximity_gain', 'chrF++ prox. gain (12B)'),
    ('chrf_deepl_proximity_gain', 'chrF++ prox. gain (DeepL)'),
]:
    row = f'{label:<35}'
    for run in all_runs:
        row += f'{fmt(data.get(run), key, 1):>12}'
    print(row)
print()

# Theological terms
for run in all_runs:
    d = data.get(run)
    if d:
        t = d.get('metrics', {}).get('theological_terms', {})
        label = labels[run]
        print(f'Theo terms ({label:<8}): base={t.get(\"base\",\"?\")}/8  adapter={t.get(\"adapter\",\"?\")}/8  12B={t.get(\"ceiling\",\"?\")}/8')
print()

# Verdicts
print('VERDICTS:')
vcols = [labels[r] for r in all_runs]
vheader = f\"  {'Gate':<30}\" + ''.join(f'{c:>12}' for c in vcols)
print(vheader)
print('  ' + '-' * 100)
all_gates = set()
for run in all_runs:
    d = data.get(run)
    if d:
        all_gates.update(d.get('verdicts', {}).keys())
for gate in sorted(all_gates):
    row = f'  {gate:<30}'
    for run in all_runs:
        d = data.get(run)
        if d:
            v = d.get('verdicts', {}).get(gate, 'N/A')
            s = 'KILL' if 'KILL' in v else ('WARN' if 'WARN' in v else ('PASS' if 'PASS' in v else 'SKIP'))
        else:
            s = 'FAIL'
        row += f'{s:>12}'
    print(row)
print()

# Decision
print('DECISION:')
for run in all_runs:
    d = data.get(run)
    label = labels[run]
    if d is None:
        print(f'  {label}: FAIL -- evaluation did not complete')
        continue
    m = d.get('metrics', {})
    comet_12b = m.get('comet_proximity_gain')
    comet_deepl = m.get('comet_proximity_gain_deepl')

    if comet_12b is not None and comet_12b < -0.01:
        print(f'  {label}: KILL (12B) -- COMET prox {comet_12b:+.4f}')
        continue
    if comet_deepl is not None and comet_deepl < -0.01:
        print(f'  {label}: KILL (DeepL) -- COMET prox {comet_deepl:+.4f}')
        continue
    status_12b = 'PASS' if (comet_12b is not None and comet_12b >= 0.005) else 'WARN'
    status_deepl = 'PASS' if (comet_deepl is not None and comet_deepl >= 0.005) else ('SKIP' if comet_deepl is None else 'WARN')
    c12 = f'{comet_12b:+.4f}' if comet_12b is not None else 'N/A'
    cdl = f'{comet_deepl:+.4f}' if comet_deepl is not None else 'N/A'
    print(f'  {label}: 12B={status_12b} ({c12}), DeepL={status_deepl} ({cdl})')
print()

# Ratio sweep interpretation
print('RATIO SWEEP INTERPRETATION:')
s4 = data.get('S4_sermon_only')
s5 = data.get('S5_sermon_heavy')
s6 = data.get('S6_balanced')
if s4 and s5 and s6:
    gains = {}
    for name, d in [('S4', s4), ('S5', s5), ('S6', s6)]:
        g = d.get('metrics', {}).get('comet_proximity_gain')
        gains[name] = g
    best = max(gains, key=lambda k: gains[k] if gains[k] is not None else -999)
    print(f'  Best ratio run: {best} (COMET prox gain = {gains[best]:+.4f})')
    if best == 'S4':
        print('  --> Verse pairs appear HARMFUL. Drop them entirely, scale up sermon data (Phase B).')
    elif best == 'S5':
        print('  --> Light verse anchor helps. Keep ~500 verse pairs, scale up sermon data (Phase B).')
    elif best == 'S6':
        print('  --> Balance matters. Maintain ~1:1 ratio, scale up both pools.')
    # Check if all similar (within 0.005)
    vals = [v for v in gains.values() if v is not None]
    if len(vals) == 3 and (max(vals) - min(vals)) < 0.005:
        print('  --> All three similar (<0.005 spread). Ratio does not matter much; sermon signal is key.')
    # Check if all KILL
    if all(v is not None and v < -0.01 for v in vals):
        print('  --> All three KILL. 1800 sermon pairs insufficient. Need Phase B (18,000+ chunks).')
else:
    print('  (Incomplete results -- cannot interpret)')
print()
" 2>&1 | tee -a "$LOG"

PIPELINE_END=$(date +%s)
TOTAL_ELAPSED=$(( PIPELINE_END - PIPELINE_START ))
echo "" | tee -a "$LOG"
echo "=== Phase 3b complete — total ${TOTAL_ELAPSED}s ($(( TOTAL_ELAPSED / 60 )) min) — $(date) ===" | tee -a "$LOG"
echo "" | tee -a "$LOG"
echo "Acceptance gates (same as S1-S3):" | tee -a "$LOG"
echo "  12B proximity:  KILL < -0.01, WARN < 0.005, PASS >= 0.005" | tee -a "$LOG"
echo "  DeepL proximity: KILL < -0.01, WARN < 0.005, PASS >= 0.005" | tee -a "$LOG"
echo "" | tee -a "$LOG"
echo "Next steps:" | tee -a "$LOG"
echo "  If any S4-S6 passes: use winning ratio + Phase B expanded data" | tee -a "$LOG"
echo "  If all KILL: run Phase B transcription first (bash training/transcribe_sermons.sh)" | tee -a "$LOG"
echo "" | tee -a "$LOG"
echo "Eval files: $HYBRID_DIR/*_sermon_eval.json" | tee -a "$LOG"
echo "Full log: $LOG" | tee -a "$LOG"
echo "=== Done ===" | tee -a "$LOG"
