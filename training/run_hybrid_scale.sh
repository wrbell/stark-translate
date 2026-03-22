#!/usr/bin/env bash
# run_hybrid_scale.sh — Phase 3: Hybrid 12B + DeepL Synthetic Sermon Pipeline
#
# End-to-end pipeline:
#   1. Generate hybrid synthetic data (12B + DeepL) from whisper chunks
#   2. Train S1/S2/S3 sweep with different training configs
#   3. Evaluate each with dual ceiling (12B + DeepL)
#   4. Print comparison summary
#
# Requires: DEEPL_KEY environment variable
#
# Usage:
#   export DEEPL_KEY="your-key-here"
#   bash training/run_hybrid_scale.sh
#   nohup bash training/run_hybrid_scale.sh &

set -euo pipefail

# --- Setup ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

VENV="/home/wbell/stt_train_env/bin/activate"
source "$VENV"

HYBRID_DIR="hybrid_runs"
LOG="$HYBRID_DIR/hybrid_log.txt"
SERMON_DATA="bible_data/synthetic/hybrid_sermon_pairs.jsonl"
CHUNKS="ablation/sermon_test_chunks_v2.json"
CACHE="ablation/sermon_12b_translations_v2.json"

mkdir -p "$HYBRID_DIR"

# --- Validate DEEPL_KEY ---
if [[ -z "${DEEPL_KEY:-}" ]]; then
    echo "ERROR: DEEPL_KEY environment variable is not set." >&2
    echo "  export DEEPL_KEY='your-deepl-pro-key'" >&2
    exit 1
fi

echo "=== Phase 3: Hybrid 12B + DeepL Pipeline — $(date) ===" | tee "$LOG"
echo "Project dir: $PROJECT_DIR" | tee -a "$LOG"
echo "Python: $(which python)" | tee -a "$LOG"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>&1 | tee -a "$LOG"
echo "" | tee -a "$LOG"

PIPELINE_START=$(date +%s)

# ============================================================
# Step 1: Generate hybrid synthetic data (~65 min)
# ============================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" | tee -a "$LOG"
echo "[Step 1/6] Generate hybrid synthetic data — $(date)" | tee -a "$LOG"
echo "  70% 12B + 30% DeepL, max 1200 chunks" | tee -a "$LOG"
echo "" | tee -a "$LOG"

GEN_START=$(date +%s)
python training/generate_hybrid_synthetic.py \
    --deepl-key "$DEEPL_KEY" \
    --max-chunks 1200 2>&1 | tee -a "$LOG"
GEN_END=$(date +%s)
GEN_ELAPSED=$(( GEN_END - GEN_START ))
echo "" | tee -a "$LOG"
echo "[Step 1/6] Data generation complete in ${GEN_ELAPSED}s ($(( GEN_ELAPSED / 60 )) min)" | tee -a "$LOG"
echo "" | tee -a "$LOG"

# Validate output
if [[ ! -f "$SERMON_DATA" ]]; then
    echo "ERROR: Synthetic data not generated at $SERMON_DATA" >&2
    exit 1
fi
PAIR_COUNT=$(wc -l < "$SERMON_DATA")
echo "Generated $PAIR_COUNT hybrid sermon pairs" | tee -a "$LOG"
echo "" | tee -a "$LOG"

# ============================================================
# Step 2: Train S1 — A1 config (lr=1e-5, 50 steps, conservative)
# ============================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" | tee -a "$LOG"
echo "[Step 2/6] Train S1 (lr=1e-5, 50 steps) — $(date)" | tee -a "$LOG"
echo "" | tee -a "$LOG"

S1_START=$(date +%s)
python training/train_gemma.py A \
    --max-pairs 8000 --glossary-oversample 2 \
    --lr 1e-5 --max-steps 50 \
    --sermon-data "$SERMON_DATA" \
    -o "$HYBRID_DIR/S1_lr1e5_50steps" 2>&1 | tee -a "$LOG"
S1_END=$(date +%s)
S1_ELAPSED=$(( S1_END - S1_START ))
echo "[Step 2/6] S1 training complete in ${S1_ELAPSED}s" | tee -a "$LOG"
echo "" | tee -a "$LOG"

# ============================================================
# Step 3: Train S2 — middle config (lr=3e-6, 100 steps)
# ============================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" | tee -a "$LOG"
echo "[Step 3/6] Train S2 (lr=3e-6, 100 steps) — $(date)" | tee -a "$LOG"
echo "" | tee -a "$LOG"

S2_START=$(date +%s)
python training/train_gemma.py A \
    --max-pairs 8000 --glossary-oversample 2 \
    --lr 3e-6 --max-steps 100 \
    --sermon-data "$SERMON_DATA" \
    -o "$HYBRID_DIR/S2_lr3e6_100steps" 2>&1 | tee -a "$LOG"
S2_END=$(date +%s)
S2_ELAPSED=$(( S2_END - S2_START ))
echo "[Step 3/6] S2 training complete in ${S2_ELAPSED}s" | tee -a "$LOG"
echo "" | tee -a "$LOG"

# ============================================================
# Step 4: Train S3 — B4 config (lr=1e-6, full epoch, neftune=5)
# ============================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" | tee -a "$LOG"
echo "[Step 4/6] Train S3 (lr=1e-6, neftune=5, full epoch) — $(date)" | tee -a "$LOG"
echo "" | tee -a "$LOG"

S3_START=$(date +%s)
python training/train_gemma.py A \
    --max-pairs 8000 --glossary-oversample 2 \
    --lr 1e-6 --neftune 5 \
    --sermon-data "$SERMON_DATA" \
    -o "$HYBRID_DIR/S3_lr1e6_neftune" 2>&1 | tee -a "$LOG"
S3_END=$(date +%s)
S3_ELAPSED=$(( S3_END - S3_START ))
echo "[Step 4/6] S3 training complete in ${S3_ELAPSED}s" | tee -a "$LOG"
echo "" | tee -a "$LOG"

# ============================================================
# Step 5: Evaluate each — dual ceiling (12B + DeepL)
# ============================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" | tee -a "$LOG"
echo "[Step 5/6] Evaluate all runs — $(date)" | tee -a "$LOG"
echo "" | tee -a "$LOG"

for RUN in S1_lr1e5_50steps S2_lr3e6_100steps S3_lr1e6_neftune; do
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
    echo "" | tee -a "$LOG"

    # Glossary regression check
    GLOSS_START=$(date +%s)
    python training/evaluate_translation.py \
        --adapter "$HYBRID_DIR/$RUN" \
        --glossary-only --compare-base 2>&1 | tee -a "$LOG"
    GLOSS_END=$(date +%s)
    GLOSS_ELAPSED=$(( GLOSS_END - GLOSS_START ))
    echo "[$RUN] Glossary eval complete in ${GLOSS_ELAPSED}s" | tee -a "$LOG"
    echo "" | tee -a "$LOG"
done

# ============================================================
# Step 6: Summary comparison table
# ============================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" | tee -a "$LOG"
echo "[Step 6/6] COMPARISON SUMMARY — $(date)" | tee -a "$LOG"
echo "" | tee -a "$LOG"

python -c "
import json, os

runs = ['S1_lr1e5_50steps', 'S2_lr3e6_100steps', 'S3_lr1e6_neftune']
hybrid_dir = '$HYBRID_DIR'

print('=' * 90)
print('PHASE 3 RESULTS: Hybrid 12B + DeepL — S1/S2/S3 Sweep')
print('=' * 90)
print()

header = f\"{'Metric':<40} {'S1 (cons.)':>15} {'S2 (mid)':>15} {'S3 (aggr.)':>15}\"
print(header)
print('-' * 90)

data = {}
for run in runs:
    path = os.path.join(hybrid_dir, f'{run}_sermon_eval.json')
    if os.path.exists(path):
        data[run] = json.load(open(path))
    else:
        data[run] = None

def fmt(d, key, dp=4):
    if d is None:
        return 'FAIL'
    m = d.get('metrics', {})
    v = m.get(key)
    return f'{v:.{dp}f}' if v is not None else 'N/A'

# COMET proximity to 12B
print(f\"{'COMET prox. gain (12B)':<40} {fmt(data.get(runs[0]), 'comet_proximity_gain'):>15} {fmt(data.get(runs[1]), 'comet_proximity_gain'):>15} {fmt(data.get(runs[2]), 'comet_proximity_gain'):>15}\")
print(f\"{'  Base vs 12B':<40} {fmt(data.get(runs[0]), 'comet_base_vs_ceiling'):>15} {fmt(data.get(runs[1]), 'comet_base_vs_ceiling'):>15} {fmt(data.get(runs[2]), 'comet_base_vs_ceiling'):>15}\")
print(f\"{'  Adapter vs 12B':<40} {fmt(data.get(runs[0]), 'comet_adapter_vs_ceiling'):>15} {fmt(data.get(runs[1]), 'comet_adapter_vs_ceiling'):>15} {fmt(data.get(runs[2]), 'comet_adapter_vs_ceiling'):>15}\")
print()

# COMET proximity to DeepL
print(f\"{'COMET prox. gain (DeepL)':<40} {fmt(data.get(runs[0]), 'comet_proximity_gain_deepl'):>15} {fmt(data.get(runs[1]), 'comet_proximity_gain_deepl'):>15} {fmt(data.get(runs[2]), 'comet_proximity_gain_deepl'):>15}\")
print(f\"{'  Base vs DeepL':<40} {fmt(data.get(runs[0]), 'comet_deepl_base_vs_ceiling'):>15} {fmt(data.get(runs[1]), 'comet_deepl_base_vs_ceiling'):>15} {fmt(data.get(runs[2]), 'comet_deepl_base_vs_ceiling'):>15}\")
print(f\"{'  Adapter vs DeepL':<40} {fmt(data.get(runs[0]), 'comet_deepl_adapter_vs_ceiling'):>15} {fmt(data.get(runs[1]), 'comet_deepl_adapter_vs_ceiling'):>15} {fmt(data.get(runs[2]), 'comet_deepl_adapter_vs_ceiling'):>15}\")
print()

# chrF++ proximity
print(f\"{'chrF++ prox. gain (12B)':<40} {fmt(data.get(runs[0]), 'chrf_proximity_gain', 1):>15} {fmt(data.get(runs[1]), 'chrf_proximity_gain', 1):>15} {fmt(data.get(runs[2]), 'chrf_proximity_gain', 1):>15}\")
print(f\"{'chrF++ prox. gain (DeepL)':<40} {fmt(data.get(runs[0]), 'chrf_deepl_proximity_gain', 1):>15} {fmt(data.get(runs[1]), 'chrf_deepl_proximity_gain', 1):>15} {fmt(data.get(runs[2]), 'chrf_deepl_proximity_gain', 1):>15}\")
print()

# Theological terms
for run in runs:
    d = data.get(run)
    if d:
        t = d.get('metrics', {}).get('theological_terms', {})
        print(f\"{'Theo terms (' + run.split('_')[0] + ')':<40} base={t.get('base','?')}/8  adapter={t.get('adapter','?')}/8  12B={t.get('ceiling','?')}/8\")
print()

# Verdicts
print('VERDICTS:')
print(f\"  {'Gate':<35} {'S1':>15} {'S2':>15} {'S3':>15}\")
print('  ' + '-' * 80)
all_gates = set()
for run in runs:
    d = data.get(run)
    if d:
        all_gates.update(d.get('verdicts', {}).keys())
for gate in sorted(all_gates):
    vals = []
    for run in runs:
        d = data.get(run)
        if d:
            v = d.get('verdicts', {}).get(gate, 'N/A')
            s = 'KILL' if 'KILL' in v else ('WARN' if 'WARN' in v else ('PASS' if 'PASS' in v else 'SKIP'))
        else:
            s = 'FAIL'
        vals.append(s)
    print(f'  {gate:<35} {vals[0]:>15} {vals[1]:>15} {vals[2]:>15}')
print()

# Decision
print('DECISION:')
for run in runs:
    d = data.get(run)
    if d is None:
        print(f'  {run}: FAIL — evaluation did not complete')
        continue
    m = d.get('metrics', {})
    comet_12b = m.get('comet_proximity_gain')
    comet_deepl = m.get('comet_proximity_gain_deepl')
    label = run.split('_')[0]

    # Check 12B
    if comet_12b is not None and comet_12b < -0.01:
        print(f'  {label}: KILL (12B) — COMET prox {comet_12b:+.4f}')
        continue
    # Check DeepL
    if comet_deepl is not None and comet_deepl < -0.01:
        print(f'  {label}: KILL (DeepL) — COMET prox {comet_deepl:+.4f}')
        continue
    # Both pass/warn
    status_12b = 'PASS' if (comet_12b is not None and comet_12b >= 0.005) else 'WARN'
    status_deepl = 'PASS' if (comet_deepl is not None and comet_deepl >= 0.005) else ('SKIP' if comet_deepl is None else 'WARN')
    print(f'  {label}: 12B={status_12b} ({comet_12b:+.4f if comet_12b else \"N/A\"}), DeepL={status_deepl} ({comet_deepl:+.4f if comet_deepl else \"N/A\"})')
print()
" 2>&1 | tee -a "$LOG"

PIPELINE_END=$(date +%s)
TOTAL_ELAPSED=$(( PIPELINE_END - PIPELINE_START ))
echo "" | tee -a "$LOG"
echo "=== Phase 3 complete — total ${TOTAL_ELAPSED}s ($(( TOTAL_ELAPSED / 60 )) min) — $(date) ===" | tee -a "$LOG"
echo "" | tee -a "$LOG"
echo "Acceptance gates (COMET-primary, dual ceiling):" | tee -a "$LOG"
echo "  Floor:   COMET > 0.740" | tee -a "$LOG"
echo "  Minimum: COMET > 0.752 (beat base)" | tee -a "$LOG"
echo "  Target:  COMET > 0.780" | tee -a "$LOG"
echo "  12B proximity:  KILL < -0.01, WARN < 0.005, PASS >= 0.005" | tee -a "$LOG"
echo "  DeepL proximity: KILL < -0.01, WARN < 0.005, PASS >= 0.005" | tee -a "$LOG"
echo "" | tee -a "$LOG"
echo "Eval files: $HYBRID_DIR/*_sermon_eval.json" | tee -a "$LOG"
echo "Full log: $LOG" | tee -a "$LOG"
echo "=== Done ===" | tee -a "$LOG"
