#!/bin/zsh
# Declared p95-tail screen tail_screen_20260911: 2 clips x (ctl, draft_g3, serial_finals) x 3 repeats, GPU serial.
# Usage: PY=/abs/venv/bin/python zsh run_screen.sh   (PY defaults to the promoted venv)
set -o pipefail
ROOT=/Users/willem/Code/vibes/SRTranslate; X=$ROOT/.cache/followup-20260911/X
PY=${PY:-$ROOT/venv/bin/python}
cd $ROOT
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 STARK_EXPERIMENT_TRACE=true STARK_EXPERIMENT_TRACE_CAPACITY=131072
EXTRA='--profile standard --no-tts --stt-backend parakeet-mlx --model-family gemma4 --gemma4-size e4b --vad-backend torch --no-mts --gain 1 --replay-speed 1'
METAL_BUDGET=16642998272   # 15.5 GiB
pageouts() { vm_stat | awk '/Pageouts/ {gsub(/\./,"",$2); print $2}'; }
run_one() {  # clip arm repeat order
  local clip=$1 arm=$2 rep=$3 order=$4 tag
  tag=ts0911_${clip}_${arm}_r${rep}
  if grep -q "\"tag\":\"$tag\",.*\"rc\":0" $X/runs.jsonl 2>/dev/null; then echo "=== $tag already recorded; skip"; return 0; fi
  if pgrep -f dry_run_ab.py >/dev/null; then echo "BUSY: dry_run_ab.py running; abort"; exit 3; fi
  if [ "$arm" = draft_g3 ] && [ -f $X/draft_blocked ]; then echo "=== $tag SKIPPED (draft arm blocked on memory)"; return 0; fi
  local before=$(pageouts)
  echo "=== $tag order=$order start $(date -u +%FT%TZ) py=$PY"
  case $arm in
    ctl) env -u STARK_EXPERIMENT_DRAFT_MODEL_ID -u STARK_EXPERIMENT_DRAFT_TOKENS -u STARK_EXPERIMENT_SERIAL_FINALS \
           $PY -m tools.replay_bench --manifest $X/manifest_${clip}.json --tag $tag --configs "$arm=$EXTRA" 2>&1 | tail -3 ;;
    draft_g3) STARK_EXPERIMENT_DRAFT_MODEL_ID=mlx-community/gemma-4-e2b-it-OptiQ-4bit STARK_EXPERIMENT_DRAFT_TOKENS=3 \
           $PY -m tools.replay_bench --manifest $X/manifest_${clip}.json --tag $tag --configs "$arm=$EXTRA" 2>&1 | tail -3 ;;
    serial_finals) STARK_EXPERIMENT_SERIAL_FINALS=true \
           $PY -m tools.replay_bench --manifest $X/manifest_${clip}.json --tag $tag --configs "$arm=$EXTRA" 2>&1 | tail -3 ;;
  esac
  local rc=$?; local after=$(pageouts)
  printf '{"tag":"%s","clip":"%s","arm":"%s","repeat":%s,"order":%s,"rc":%s,"pageouts":{"before":%s,"after":%s},"ended":"%s"}\n' \
    "$tag" "$clip" "$arm" "$rep" "$order" "$rc" "$before" "$after" "$(date -u +%FT%TZ)" >> $X/runs.jsonl
  echo "rc=$rc pageouts $before -> $after"
  if [ "$arm" = draft_g3 ]; then
    local lc=$(ls -t metrics/session_lifecycle_${tag}_*.json 2>/dev/null | head -1)
    local peak=$(python3 -c "import json,sys; print(json.load(open('$lc'))['memory'].get('peak_metal_bytes',0))" 2>/dev/null || echo 0)
    echo "draft peak_metal_bytes=$peak"
    local ctl_delta=$(python3 -c "
import json,statistics
rows=[json.loads(l) for l in open('$X/runs.jsonl') if l.strip()]
d=[r['pageouts']['after']-r['pageouts']['before'] for r in rows if r['arm']=='ctl' and r['clip']=='$clip' and r['rc']==0]
print(int(statistics.median(d)) if d else 0)")
    local delta=$((after-before)); local limit=$(( ctl_delta*5 > 20000 ? ctl_delta*5 : 20000 ))
    echo "draft pageout delta=$delta (control median delta=$ctl_delta, limit=$limit)"
    if [ "$peak" -gt "$METAL_BUDGET" ] || [ "$delta" -gt "$limit" ]; then echo "DRAFT BLOCKED: peak=$peak pageout delta=$delta limit=$limit" | tee $X/draft_blocked; fi
  fi
}
order=0
for clip in A B; do
  for run in ctl:0 draft_g3:0 serial_finals:0 serial_finals:1 draft_g3:1 ctl:1 ctl:2 serial_finals:2 draft_g3:2; do
    run_one $clip ${run%%:*} ${run##*:} $order; order=$((order+1))
  done
done
echo "=== screen done $(date -u +%FT%TZ) ==="
