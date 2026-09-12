#!/bin/zsh
# P1 paired identity screen: control checkout (5227a73) vs P1 worktree, same venv python, 2 clips x 3 alternating pairs.
set -o pipefail
ROOT=/Users/willem/Code/vibes/SRTranslate; P1=$ROOT/.cache/series4-20260912/P1; X=$ROOT/.cache/followup-20260911/X
BASE=/Users/willem/Code/vibes/SRTranslate-wt-base; CAND=/Users/willem/Code/vibes/SRTranslate-wt-p1; PY=$ROOT/venv/bin/python
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 STARK_EXPERIMENT_TRACE=true STARK_EXPERIMENT_TRACE_CAPACITY=131072
EXTRA='--profile standard --no-tts --stt-backend parakeet-mlx --model-family gemma4 --gemma4-size e4b --vad-backend torch --no-mts --gain 1 --replay-speed 1'
pageouts() { vm_stat | awk '/Pageouts/ {gsub(/\./,"",$2); print $2}'; }
run_one() {
  local clip=$1 arm=$2 rep=$3 order=$4 tag dir
  tag=p1s0912_${clip}_${arm}_r${rep}
  if [ "$arm" = ctl ]; then dir=$BASE; else dir=$CAND; fi
  if grep -q "\"tag\":\"$tag\",.*\"rc\":0" $P1/runs.jsonl 2>/dev/null; then echo "=== $tag already recorded; skip"; return 0; fi
  if pgrep -f dry_run_ab.py >/dev/null; then echo "BUSY: dry_run_ab.py running; abort"; exit 3; fi
  local before=$(pageouts)
  echo "=== $tag order=$order dir=$dir start $(date -u +%FT%TZ)"
  (cd $dir && $PY -m tools.replay_bench --manifest $X/manifest_${clip}.json --tag $tag --configs "$arm=$EXTRA" 2>&1 | tail -3)
  local rc=$?; local after=$(pageouts)
  printf '{"tag":"%s","clip":"%s","arm":"%s","repeat":%s,"order":%s,"rc":%s,"checkout":"%s","pageouts":{"before":%s,"after":%s},"ended":"%s"}\n' "$tag" "$clip" "$arm" "$rep" "$order" "$rc" "$dir" "$before" "$after" "$(date -u +%FT%TZ)" >> $P1/runs.jsonl
  echo "rc=$rc pageouts $before -> $after"
}
order=0
for clip in A B; do
  for run in ctl:0 cand:0 cand:1 ctl:1 ctl:2 cand:2; do
    run_one $clip ${run%%:*} ${run##*:} $order; order=$((order+1))
  done
done
echo "=== P1 screen done $(date -u +%FT%TZ) ==="
