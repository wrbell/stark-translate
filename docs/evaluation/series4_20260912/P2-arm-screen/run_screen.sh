#!/bin/zsh
# P2 arm screen: control vs partial_reuse_ms=100 vs =300, one checkout (P2 branch rebased on main), same venv python; 2 clips x 3 repeats.
set -o pipefail
ROOT=/Users/willem/Code/vibes/SRTranslate; P2=$ROOT/.cache/series4-20260912/P2; X=$ROOT/.cache/followup-20260911/X
DIR=${P2_CHECKOUT:-/Users/willem/Code/vibes/SRTranslate-wt-p2}; PY=$ROOT/venv/bin/python
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 STARK_EXPERIMENT_TRACE=true STARK_EXPERIMENT_TRACE_CAPACITY=131072
EXTRA='--profile standard --no-tts --stt-backend parakeet-mlx --model-family gemma4 --gemma4-size e4b --vad-backend torch --no-mts --gain 1 --replay-speed 1'
pageouts() { vm_stat | awk '/Pageouts/ {gsub(/\./,"",$2); print $2}'; }
run_one() {
  local clip=$1 arm=$2 rep=$3 order=$4 tag
  tag=p2s0912_${clip}_${arm}_r${rep}
  if grep -q "\"tag\":\"$tag\",.*\"rc\":0" $P2/runs.jsonl 2>/dev/null; then echo "=== $tag already recorded; skip"; return 0; fi
  if pgrep -f dry_run_ab.py >/dev/null; then echo "BUSY: dry_run_ab.py running; abort"; exit 3; fi
  local before=$(pageouts)
  echo "=== $tag order=$order start $(date -u +%FT%TZ)"
  case $arm in
    ctl) (cd $DIR && env -u STARK_EXPERIMENT_PARTIAL_REUSE_MS $PY -m tools.replay_bench --manifest $X/manifest_${clip}.json --tag $tag --configs "$arm=$EXTRA" 2>&1 | tail -3) ;;
    reuse100) (cd $DIR && STARK_EXPERIMENT_PARTIAL_REUSE_MS=100 $PY -m tools.replay_bench --manifest $X/manifest_${clip}.json --tag $tag --configs "$arm=$EXTRA" 2>&1 | tail -3) ;;
    reuse300) (cd $DIR && STARK_EXPERIMENT_PARTIAL_REUSE_MS=300 $PY -m tools.replay_bench --manifest $X/manifest_${clip}.json --tag $tag --configs "$arm=$EXTRA" 2>&1 | tail -3) ;;
  esac
  local rc=$?; local after=$(pageouts)
  printf '{"tag":"%s","clip":"%s","arm":"%s","repeat":%s,"order":%s,"rc":%s,"checkout":"%s","pageouts":{"before":%s,"after":%s},"ended":"%s"}\n' "$tag" "$clip" "$arm" "$rep" "$order" "$rc" "$DIR" "$before" "$after" "$(date -u +%FT%TZ)" >> $P2/runs.jsonl
  echo "rc=$rc pageouts $before -> $after"
}
order=0
for clip in A B; do
  for run in ctl:0 reuse100:0 reuse300:0 reuse300:1 reuse100:1 ctl:1 ctl:2 reuse300:2 reuse100:2; do
    run_one $clip ${run%%:*} ${run##*:} $order; order=$((order+1))
  done
done
echo "=== P2 screen done $(date -u +%FT%TZ) ==="
