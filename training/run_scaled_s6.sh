#!/usr/bin/env bash
# run_scaled_s6.sh — S7 Pipeline: Scaled Balanced (5000 chunks, 60/40 12B/DeepL)
#
# End-to-end pipeline:
#   1. Generate 5000-chunk hybrid synthetic data (60% 12B + 40% DeepL)
#   2. Train S7 with S6 balanced config at 5000-pair scale
#   3. Evaluate with dual ceiling (12B + DeepL) + glossary regression
#   4. Print S6 vs S7 comparison
#
# Requires: DEEPL_KEY environment variable
#
# Usage:
#   export DEEPL_KEY="your-key-here"
#   bash training/run_scaled_s6.sh
#   nohup bash training/run_scaled_s6.sh &

set -euo pipefail

# --- Setup ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

VENV="/home/wbell/stt_train_env/bin/activate"
source "$VENV"

HYBRID_DIR="hybrid_runs"
LOG="$HYBRID_DIR/s7_scaled_log.txt"
SERMON_DATA="bible_data/synthetic/hybrid_sermon_pairs_5000.jsonl"
CHUNKS="ablation/sermon_test_chunks_v2.json"
CACHE="ablation/sermon_12b_translations_v2.json"
EXPANDED_CHUNKS="ablation/sermon_whisper_chunks_expanded.json"

mkdir -p "$HYBRID_DIR"

# --- Validate DEEPL_KEY ---
if [[ -z "${DEEPL_KEY:-}" ]]; then
    echo "ERROR: DEEPL_KEY environment variable is not set." >&2
    echo "  export DEEPL_KEY='your-deepl-pro-key'" >&2
    exit 1
fi

echo "=== S7 Pipeline: Scaled Balanced (5000 chunks) — $(date) ===" | tee "$LOG"
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

# ============================================================
# Step 1: Generate 5000-chunk hybrid data (60/40 12B/DeepL)
# ============================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" | tee -a "$LOG"
echo "[Step 1/4] Generate hybrid synthetic data — $(date)" | tee -a "$LOG"
echo "  60% 12B + 40% DeepL, max 5000 chunks (expanded pool)" | tee -a "$LOG"
echo "" | tee -a "$LOG"

GEN_START=$(date +%s)
python training/generate_hybrid_synthetic.py \
    --deepl-key "$DEEPL_KEY" \
    --whisper-chunks "$EXPANDED_CHUNKS" \
    --max-chunks 5000 --ratio-deepl 0.40 \
    --output "$SERMON_DATA" 2>&1 | tee -a "$LOG"
GEN_END=$(date +%s)
GEN_ELAPSED=$(( GEN_END - GEN_START ))
echo "" | tee -a "$LOG"
echo "[Step 1/4] Data generation complete in ${GEN_ELAPSED}s ($(( GEN_ELAPSED / 60 )) min)" | tee -a "$LOG"
log_vram "After data generation (12B model)"
echo "" | tee -a "$LOG"

# Validate output
if [[ ! -f "$SERMON_DATA" ]]; then
    echo "ERROR: Synthetic data not generated at $SERMON_DATA" >&2
    exit 1
fi
PAIR_COUNT=$(wc -l < "$SERMON_DATA")
echo "Generated $PAIR_COUNT hybrid sermon pairs (target: ~5000)" | tee -a "$LOG"
echo "" | tee -a "$LOG"

# ============================================================
# Step 2: Train S7 — scaled balanced (5000 pairs, lr=1e-5, 75 steps)
# ============================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" | tee -a "$LOG"
echo "[Step 2/4] Train S7 — scaled balanced (5000 pairs) — $(date)" | tee -a "$LOG"
echo "  Config: lr=1e-5, 75 steps, glossary-oversample=1" | tee -a "$LOG"
echo "  Hypothesis: S6 balanced ratio + 2.8x more data → signal emergence" | tee -a "$LOG"
echo "" | tee -a "$LOG"

S7_START=$(date +%s)
python training/train_gemma.py A \
    --max-pairs 5000 --glossary-oversample 1 \
    --lr 1e-5 --max-steps 75 \
    --sermon-data "$SERMON_DATA" \
    -o "$HYBRID_DIR/S7_scaled_balanced" 2>&1 | tee -a "$LOG"
S7_END=$(date +%s)
S7_ELAPSED=$(( S7_END - S7_START ))
echo "[Step 2/4] S7 training complete in ${S7_ELAPSED}s" | tee -a "$LOG"
log_vram "After S7 training (4B QLoRA)"
echo "" | tee -a "$LOG"

# ============================================================
# Step 3: Evaluate — dual ceiling (12B + DeepL) + glossary
# ============================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" | tee -a "$LOG"
echo "[Step 3/4] Evaluate S7 — $(date)" | tee -a "$LOG"
echo "" | tee -a "$LOG"

# Sermon eval with dual ceiling (12B + DeepL)
EVAL_START=$(date +%s)
python training/evaluate_sermon.py \
    --chunks "$CHUNKS" \
    --adapter "$HYBRID_DIR/S7_scaled_balanced" \
    --ceiling-cache "$CACHE" \
    --deepl-key "$DEEPL_KEY" \
    --output "$HYBRID_DIR/S7_scaled_balanced_sermon_eval.json" 2>&1 | tee -a "$LOG"
EVAL_END=$(date +%s)
EVAL_ELAPSED=$(( EVAL_END - EVAL_START ))
echo "[S7] Sermon eval complete in ${EVAL_ELAPSED}s" | tee -a "$LOG"
log_vram "After S7 sermon eval (4B base + 4B adapter + 12B ceiling)"
echo "" | tee -a "$LOG"

# Glossary regression check
GLOSS_START=$(date +%s)
python training/evaluate_translation.py \
    --adapter "$HYBRID_DIR/S7_scaled_balanced" \
    --glossary-only --compare-base 2>&1 | tee -a "$LOG"
GLOSS_END=$(date +%s)
GLOSS_ELAPSED=$(( GLOSS_END - GLOSS_START ))
echo "[S7] Glossary eval complete in ${GLOSS_ELAPSED}s" | tee -a "$LOG"
log_vram "After S7 glossary eval (4B base + 4B adapter)"
echo "" | tee -a "$LOG"

# ============================================================
# Step 4: S6 vs S7 comparison
# ============================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" | tee -a "$LOG"
echo "[Step 4/4] S6 vs S7 COMPARISON — $(date)" | tee -a "$LOG"
echo "" | tee -a "$LOG"

python -c "
import json, os

hybrid_dir = '$HYBRID_DIR'
runs = ['S6_balanced', 'S7_scaled_balanced']
labels = {
    'S6_balanced': 'S6 (1800)',
    'S7_scaled_balanced': 'S7 (5000)',
}

print('=' * 80)
print('S6 vs S7: Effect of Scaling Data (1800 -> 5000 sermon pairs)')
print('=' * 80)
print()

# Data mix context
print('DATA MIX:')
print('  S6: 1800 verse + 507 glossary(1x) + 1800 sermon = ~4,107 (44% sermon)')
print('  S7: 5000 verse + 507 glossary(1x) + 5000 sermon = ~10,507 (48% sermon)')
print()

# Load results
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

# Header
header = f\"{'Metric':<40} {'S6 (1800)':>15} {'S7 (5000)':>15}\"
print(header)
print('-' * 80)

# COMET proximity (12B)
for key, label in [
    ('comet_proximity_gain', 'COMET prox. gain (12B)'),
    ('comet_adapter_vs_ceiling', '  Adapter vs 12B'),
    ('comet_base_vs_ceiling', '  Base vs 12B'),
]:
    row = f'{label:<40} {fmt(data.get(runs[0]), key):>15} {fmt(data.get(runs[1]), key):>15}'
    print(row)
print()

# COMET proximity (DeepL)
for key, label in [
    ('comet_proximity_gain_deepl', 'COMET prox. gain (DeepL)'),
    ('comet_deepl_adapter_vs_ceiling', '  Adapter vs DeepL'),
    ('comet_deepl_base_vs_ceiling', '  Base vs DeepL'),
]:
    row = f'{label:<40} {fmt(data.get(runs[0]), key):>15} {fmt(data.get(runs[1]), key):>15}'
    print(row)
print()

# chrF++
for key, label in [
    ('chrf_proximity_gain', 'chrF++ prox. gain (12B)'),
    ('chrf_deepl_proximity_gain', 'chrF++ prox. gain (DeepL)'),
]:
    row = f'{label:<40} {fmt(data.get(runs[0]), key, 1):>15} {fmt(data.get(runs[1]), key, 1):>15}'
    print(row)
print()

# Theological terms
for run in runs:
    d = data.get(run)
    if d:
        t = d.get('metrics', {}).get('theological_terms', {})
        label = labels[run]
        print(f'Theo terms ({label:<10}): base={t.get(\"base\",\"?\")}/8  adapter={t.get(\"adapter\",\"?\")}/8  12B={t.get(\"ceiling\",\"?\")}/8')
print()

# Verdicts
print('VERDICTS:')
vheader = f\"  {'Gate':<35} {'S6 (1800)':>15} {'S7 (5000)':>15}\"
print(vheader)
print('  ' + '-' * 65)
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
    print(f'  {gate:<35} {vals[0]:>15} {vals[1]:>15}')
print()

# Decision
print('SCALING INTERPRETATION:')
s6 = data.get('S6_balanced')
s7 = data.get('S7_scaled_balanced')
if s6 and s7:
    s6_gain = s6.get('metrics', {}).get('comet_proximity_gain')
    s7_gain = s7.get('metrics', {}).get('comet_proximity_gain')
    s6_deepl = s6.get('metrics', {}).get('comet_proximity_gain_deepl')
    s7_deepl = s7.get('metrics', {}).get('comet_proximity_gain_deepl')

    if s7_gain is not None and s6_gain is not None:
        delta = s7_gain - s6_gain
        print(f'  12B proximity delta (S7 - S6): {delta:+.4f}')
        if delta > 0.005:
            print('  --> Scaling HELPS. More data = better signal. Consider further scaling.')
        elif delta > -0.005:
            print('  --> Scaling NEUTRAL. Diminishing returns at 5000 pairs.')
        else:
            print('  --> Scaling HURTS. Possible noise dilution. Review data quality.')

    if s7_deepl is not None and s6_deepl is not None:
        delta_d = s7_deepl - s6_deepl
        print(f'  DeepL proximity delta (S7 - S6): {delta_d:+.4f}')

    # Kill check
    for name, d in [('S6', s6), ('S7', s7)]:
        g = d.get('metrics', {}).get('comet_proximity_gain')
        gd = d.get('metrics', {}).get('comet_proximity_gain_deepl')
        if g is not None and g < -0.01:
            print(f'  {name}: KILL (12B) -- COMET prox {g:+.4f}')
        elif gd is not None and gd < -0.01:
            print(f'  {name}: KILL (DeepL) -- COMET prox {gd:+.4f}')
        else:
            s12 = 'PASS' if (g is not None and g >= 0.005) else 'WARN'
            sdl = 'PASS' if (gd is not None and gd >= 0.005) else ('SKIP' if gd is None else 'WARN')
            c12 = f'{g:+.4f}' if g is not None else 'N/A'
            cdl = f'{gd:+.4f}' if gd is not None else 'N/A'
            print(f'  {name}: 12B={s12} ({c12}), DeepL={sdl} ({cdl})')
else:
    print('  (Incomplete results -- cannot interpret)')
print()
" 2>&1 | tee -a "$LOG"

PIPELINE_END=$(date +%s)
TOTAL_ELAPSED=$(( PIPELINE_END - PIPELINE_START ))
echo "" | tee -a "$LOG"
echo "=== S7 pipeline complete — total ${TOTAL_ELAPSED}s ($(( TOTAL_ELAPSED / 60 )) min) — $(date) ===" | tee -a "$LOG"
echo "" | tee -a "$LOG"
echo "Acceptance gates (same as S1-S6):" | tee -a "$LOG"
echo "  12B proximity:  KILL < -0.01, WARN < 0.005, PASS >= 0.005" | tee -a "$LOG"
echo "  DeepL proximity: KILL < -0.01, WARN < 0.005, PASS >= 0.005" | tee -a "$LOG"
echo "" | tee -a "$LOG"
echo "Next steps:" | tee -a "$LOG"
echo "  If S7 PASS: ship adapter, begin active learning cycle 2" | tee -a "$LOG"
echo "  If S7 WARN: try S8 with neftune or higher steps" | tee -a "$LOG"
echo "  If S7 KILL: re-examine data quality in expanded pool" | tee -a "$LOG"
echo "" | tee -a "$LOG"
echo "Eval files: $HYBRID_DIR/S7_scaled_balanced_sermon_eval.json" | tee -a "$LOG"
echo "Full log: $LOG" | tee -a "$LOG"
echo "=== Done ===" | tee -a "$LOG"
