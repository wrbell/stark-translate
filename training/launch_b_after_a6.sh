#!/usr/bin/env bash
# launch_b_after_a6.sh — Wait for A6 eval to finish, then launch B-series
#
# Polls every 30s for ablation/A6_replay20_metrics.json or for the A6 eval
# process to exit, then immediately starts run_b_series.sh.
#
# Usage:
#   nohup bash training/launch_b_after_a6.sh &
#   # Go to sleep. B-series runs overnight.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

A6_METRICS="ablation/A6_replay20_metrics.json"
A6_EVAL_PID=269615  # Current A6 eval PID — update if restarted

echo "=== B-Series Launcher — $(date) ==="
echo "Waiting for A6 eval to complete..."
echo "  Watching for: $A6_METRICS"
echo "  Watching PID: $A6_EVAL_PID"
echo ""

while true; do
    # Check if metrics file appeared (eval finished successfully)
    if [[ -f "$A6_METRICS" ]]; then
        echo "[$(date)] A6 metrics found!"
        source /home/wbell/stt_train_env/bin/activate
        python -c "
import json
d = json.load(open('$A6_METRICS'))
bleu = d.get('bleu', 'N/A')
chrf = d.get('chrf', 'N/A')
comet = d.get('comet')
comet_s = f'{comet:.3f}' if comet else 'N/A'
print(f'  A6 results: BLEU={bleu:.1f}  chrF++={chrf:.1f}  COMET={comet_s}')
"
        break
    fi

    # Check if eval process exited (could have failed)
    if ! kill -0 "$A6_EVAL_PID" 2>/dev/null; then
        echo "[$(date)] A6 eval process ($A6_EVAL_PID) exited."
        if [[ -f "$A6_METRICS" ]]; then
            echo "  Metrics file found — eval succeeded."
        else
            echo "  WARNING: Metrics file NOT found — eval may have failed."
            echo "  B-series will still launch (B5 replay note will reflect missing A6 data)."
        fi
        break
    fi

    sleep 30
done

echo ""
echo "=== Launching B-Series — $(date) ==="
echo ""

# Small delay to let run_ablation.sh finish its summary output
sleep 5

exec bash training/run_b_series.sh
