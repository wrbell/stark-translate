#!/usr/bin/env bash
# run_phase25.sh — Phase 2.5: Full sermon smoke test (A1 + B4)
#
# Runs everything end-to-end:
#   1. A1 eval (3 models, generates 12B translations)
#   2. Extract 12B cache from A1 output
#   3. B4 eval (2 models, uses cached 12B)
#   4. Print comparison summary
#
# Usage:
#   nohup bash training/run_phase25.sh &
#   tail -f ablation/phase25_log.txt

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

VENV="/home/wbell/stt_train_env/bin/activate"
source "$VENV"

CHUNKS="ablation/sermon_test_chunks_v2.json"
CACHE="ablation/sermon_12b_translations_v2.json"
LOG="ablation/phase25_log.txt"

echo "=== Phase 2.5 Sermon Smoke Test — $(date) ===" | tee "$LOG"
echo "Chunks: $CHUNKS" | tee -a "$LOG"
echo "Python: $(which python)" | tee -a "$LOG"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>&1 | tee -a "$LOG"
echo "" | tee -a "$LOG"

# ============================================================
# Step 1: A1 eval (3 models — base, adapter, 12B)
# ============================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" | tee -a "$LOG"
echo "[Step 1/4] A1 eval (COMET-optimal) — $(date)" | tee -a "$LOG"
echo "  3 models sequential, 520 chunks + 8 sermon sentences" | tee -a "$LOG"
echo "" | tee -a "$LOG"

A1_START=$(date +%s)
python training/evaluate_sermon.py \
    --chunks "$CHUNKS" \
    --adapter ablation/A1_steps50 \
    --output ablation/sermon_eval_A1_v2.json 2>&1 | tee -a "$LOG"
A1_END=$(date +%s)
A1_ELAPSED=$(( A1_END - A1_START ))
echo "" | tee -a "$LOG"
echo "[Step 1/4] A1 eval complete in ${A1_ELAPSED}s ($(( A1_ELAPSED / 60 )) min)" | tee -a "$LOG"
echo "" | tee -a "$LOG"

# ============================================================
# Step 2: Extract 12B cache from A1 output
# ============================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" | tee -a "$LOG"
echo "[Step 2/4] Extracting 12B cache — $(date)" | tee -a "$LOG"

python -c "
import json
d = json.load(open('ablation/sermon_eval_A1_v2.json'))
cache = []
for ct in d['chunk_translations']:
    cache.append({'en': ct['en'], 'es_12b': ct['ceiling']})
for td in d['theological_details']:
    cache.append({'en': td['sentence'], 'es_12b': td['ceiling_translation']})
if 'segment' in d:
    cache.append({'en': d['segment']['en'], 'es_12b': d['segment']['ceiling']})
with open('$CACHE', 'w') as f:
    json.dump(cache, f, indent=2, ensure_ascii=False)
print(f'Cached {len(cache)} 12B translations to $CACHE')
" 2>&1 | tee -a "$LOG"
echo "" | tee -a "$LOG"

# ============================================================
# Step 3: B4 eval (2 models — base + adapter, 12B from cache)
# ============================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" | tee -a "$LOG"
echo "[Step 3/4] B4 eval (BLEU-optimal) — $(date)" | tee -a "$LOG"
echo "  2 models + ceiling cache, 520 chunks + 8 sermon sentences" | tee -a "$LOG"
echo "" | tee -a "$LOG"

B4_START=$(date +%s)
python training/evaluate_sermon.py \
    --chunks "$CHUNKS" \
    --adapter ablation/B4_lr1e6_neftune \
    --ceiling-cache "$CACHE" \
    --output ablation/sermon_eval_B4_v2.json 2>&1 | tee -a "$LOG"
B4_END=$(date +%s)
B4_ELAPSED=$(( B4_END - B4_START ))
echo "" | tee -a "$LOG"
echo "[Step 3/4] B4 eval complete in ${B4_ELAPSED}s ($(( B4_ELAPSED / 60 )) min)" | tee -a "$LOG"
echo "" | tee -a "$LOG"

# ============================================================
# Step 4: Summary comparison
# ============================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" | tee -a "$LOG"
echo "[Step 4/4] COMPARISON SUMMARY — $(date)" | tee -a "$LOG"
echo "" | tee -a "$LOG"

python -c "
import json

a1 = json.load(open('ablation/sermon_eval_A1_v2.json'))
b4 = json.load(open('ablation/sermon_eval_B4_v2.json'))

a1m = a1['metrics']
b4m = b4['metrics']

print('=' * 70)
print('PHASE 2.5 RESULTS: A1 vs B4 on 520 sermon chunks')
print('=' * 70)
print()
print(f'{\"Metric\":<35} {\"A1 (COMET-opt)\":>15} {\"B4 (BLEU-opt)\":>15}')
print('-' * 70)

def fmt(v, dp=4):
    return f'{v:.{dp}f}' if v is not None else 'N/A'

print(f'{\"COMET proximity gain\":<35} {fmt(a1m.get(\"comet_proximity_gain\")):>15} {fmt(b4m.get(\"comet_proximity_gain\")):>15}')
print(f'{\"  Base vs 12B\":<35} {fmt(a1m.get(\"comet_base_vs_ceiling\")):>15} {fmt(b4m.get(\"comet_base_vs_ceiling\")):>15}')
print(f'{\"  Adapter vs 12B\":<35} {fmt(a1m.get(\"comet_adapter_vs_ceiling\")):>15} {fmt(b4m.get(\"comet_adapter_vs_ceiling\")):>15}')
print(f'{\"chrF++ proximity gain\":<35} {fmt(a1m.get(\"chrf_proximity_gain\"), 1):>15} {fmt(b4m.get(\"chrf_proximity_gain\"), 1):>15}')
print(f'{\"  Base vs 12B\":<35} {fmt(a1m.get(\"chrf_base_vs_ceiling\"), 1):>15} {fmt(b4m.get(\"chrf_base_vs_ceiling\"), 1):>15}')
print(f'{\"  Adapter vs 12B\":<35} {fmt(a1m.get(\"chrf_adapter_vs_ceiling\"), 1):>15} {fmt(b4m.get(\"chrf_adapter_vs_ceiling\"), 1):>15}')
print(f'{\"Clean chunks used\":<35} {str(a1m.get(\"clean_chunks_used\", \"N/A\")):>15} {str(b4m.get(\"clean_chunks_used\", \"N/A\")):>15}')
print()

# Hallucination
a1h = a1m.get('hallucination') or {}
b4h = b4m.get('hallucination') or {}
print(f'{\"Hallucination ratio (adapter)\":<35} {fmt(a1h.get(\"avg_adapter_ratio\"), 2):>15} {fmt(b4h.get(\"avg_adapter_ratio\"), 2):>15}')
print(f'{\"Hallucination ratio increase\":<35} {fmt(a1h.get(\"ratio_increase\"), 2):>15} {fmt(b4h.get(\"ratio_increase\"), 2):>15}')
print()

# Theo terms
a1t = a1m.get('theological_terms', {})
b4t = b4m.get('theological_terms', {})
print(f'{\"Sermon theo terms (base)\":<35} {str(a1t.get(\"base\", \"?\")) + \"/8\":>15} {str(b4t.get(\"base\", \"?\")) + \"/8\":>15}')
print(f'{\"Sermon theo terms (adapter)\":<35} {str(a1t.get(\"adapter\", \"?\")) + \"/8\":>15} {str(b4t.get(\"adapter\", \"?\")) + \"/8\":>15}')
print(f'{\"Sermon theo terms (12B)\":<35} {str(a1t.get(\"ceiling\", \"?\")) + \"/8\":>15} {str(b4t.get(\"ceiling\", \"?\")) + \"/8\":>15}')
print()

# Archaic
a1a = a1m.get('archaic_markers', {})
b4a = b4m.get('archaic_markers', {})
print(f'{\"Archaic markers (base)\":<35} {str(a1a.get(\"base\", \"?\")):>15} {str(b4a.get(\"base\", \"?\")):>15}')
print(f'{\"Archaic markers (adapter)\":<35} {str(a1a.get(\"adapter\", \"?\")):>15} {str(b4a.get(\"adapter\", \"?\")):>15}')
print(f'{\"Archaic markers (12B)\":<35} {str(a1a.get(\"ceiling\", \"?\")):>15} {str(b4a.get(\"ceiling\", \"?\")):>15}')
print()

# Verdicts
print('VERDICTS:')
print(f'  {\"Gate\":<30} {\"A1\":>18} {\"B4\":>18}')
print('  ' + '-' * 66)
all_gates = set(list(a1['verdicts'].keys()) + list(b4['verdicts'].keys()))
for gate in sorted(all_gates):
    a1v = a1['verdicts'].get(gate, 'N/A')
    b4v = b4['verdicts'].get(gate, 'N/A')
    a1s = 'KILL' if 'KILL' in a1v else ('WARN' if 'WARN' in a1v else ('PASS' if 'PASS' in a1v else 'SKIP'))
    b4s = 'KILL' if 'KILL' in b4v else ('WARN' if 'WARN' in b4v else ('PASS' if 'PASS' in b4v else 'SKIP'))
    print(f'  {gate:<30} {a1s:>18} {b4s:>18}')
print()

# Decision
a1_comet = a1m.get('comet_proximity_gain')
b4_comet = b4m.get('comet_proximity_gain')
print('DECISION:')
if a1_comet is not None and a1_comet > 0.005:
    print('  A1: PASS — COMET proximity gain > 0.005')
elif a1_comet is not None and a1_comet < -0.01:
    print('  A1: KILL — COMET proximity gain < -0.01')
elif a1_comet is not None:
    print(f'  A1: WARN — COMET proximity gain {a1_comet:+.4f} in noise range')
else:
    print('  A1: SKIP — no COMET data')

if b4_comet is not None and b4_comet > 0.005:
    print('  B4: PASS — COMET proximity gain > 0.005')
elif b4_comet is not None and b4_comet < -0.01:
    print('  B4: KILL — COMET proximity gain < -0.01')
elif b4_comet is not None:
    print(f'  B4: WARN — COMET proximity gain {b4_comet:+.4f} in noise range')
else:
    print('  B4: SKIP — no COMET data')

print()
if (a1_comet is not None and a1_comet > 0.005) or (b4_comet is not None and b4_comet > 0.005):
    winner = 'A1' if (a1_comet or -999) > (b4_comet or -999) else 'B4'
    print(f'  >>> OUTCOME A: {winner} shows improvement. Lock winner → Phase 3. <<<')
elif (a1_comet is not None and a1_comet >= -0.01) or (b4_comet is not None and b4_comet >= -0.01):
    print('  >>> OUTCOME B: Both in noise range. Proceed to Phase 3 with synthetic data. <<<')
else:
    print('  >>> OUTCOME C: Both KILL. Deploy base model. Try Phase 3 with distilled-only. <<<')
print()
print(f'Total GPU time: A1=${A1_ELAPSED}s + B4=${B4_ELAPSED}s = {int(\"${A1_ELAPSED}\") + int(\"${B4_ELAPSED}\")}s')
" 2>&1 | tee -a "$LOG"

TOTAL_END=$(date +%s)
TOTAL_ELAPSED=$(( TOTAL_END - A1_START ))
echo "" | tee -a "$LOG"
echo "=== Phase 2.5 complete — total ${TOTAL_ELAPSED}s ($(( TOTAL_ELAPSED / 60 )) min) — $(date) ===" | tee -a "$LOG"
