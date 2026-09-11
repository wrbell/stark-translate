#!/bin/zsh
# L-B screen tail_screen_series3_20260912: 2 clips x (ctl, marian_threads_2, max_utterance_6, partial_recheck_translation) x 3 repeats.
set -o pipefail
ROOT=/Users/willem/Code/vibes/SRTranslate; LB=$ROOT/.cache/series3-20260912/LB; X=$ROOT/.cache/followup-20260911/X
PY=$ROOT/venv/bin/python; cd $ROOT
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 STARK_EXPERIMENT_TRACE=true STARK_EXPERIMENT_TRACE_CAPACITY=131072
EXTRA='--profile standard --no-tts --stt-backend parakeet-mlx --model-family gemma4 --gemma4-size e4b --vad-backend torch --no-mts --gain 1 --replay-speed 1'
pageouts() { vm_stat | awk '/Pageouts/ {gsub(/\./,"",$2); print $2}'; }
run_one() {
  local clip=$1 arm=$2 rep=$3 order=$4 tag
  tag=lb0912_${clip}_${arm}_r${rep}
  if grep -q "\"tag\":\"$tag\",.*\"rc\":0" $LB/runs.jsonl 2>/dev/null; then echo "=== $tag already recorded; skip"; return 0; fi
  if pgrep -f dry_run_ab.py >/dev/null; then echo "BUSY: dry_run_ab.py running; abort"; exit 3; fi
  local before=$(pageouts)
  echo "=== $tag order=$order start $(date -u +%FT%TZ)"
  case $arm in
    ctl) env -u STARK_TRANSLATE_MARIAN_INTRA_THREADS -u STARK_VAD_MAX_UTTERANCE -u STARK_EXPERIMENT_PARTIAL_RECHECK_TRANSLATION $PY -m tools.replay_bench --manifest $X/manifest_${clip}.json --tag $tag --configs "$arm=$EXTRA" 2>&1 | tail -3 ;;
    marian_threads_2) STARK_TRANSLATE_MARIAN_INTRA_THREADS=2 $PY -m tools.replay_bench --manifest $X/manifest_${clip}.json --tag $tag --configs "$arm=$EXTRA" 2>&1 | tail -3 ;;
    max_utterance_6) STARK_VAD_MAX_UTTERANCE=6.0 $PY -m tools.replay_bench --manifest $X/manifest_${clip}.json --tag $tag --configs "$arm=$EXTRA" 2>&1 | tail -3 ;;
    partial_recheck_translation) STARK_EXPERIMENT_PARTIAL_RECHECK_TRANSLATION=true $PY -m tools.replay_bench --manifest $X/manifest_${clip}.json --tag $tag --configs "$arm=$EXTRA" 2>&1 | tail -3 ;;
  esac
  local rc=$?; local after=$(pageouts)
  printf '{"tag":"%s","clip":"%s","arm":"%s","repeat":%s,"order":%s,"rc":%s,"pageouts":{"before":%s,"after":%s},"ended":"%s"}\n' "$tag" "$clip" "$arm" "$rep" "$order" "$rc" "$before" "$after" "$(date -u +%FT%TZ)" >> $LB/runs.jsonl
  echo "rc=$rc pageouts $before -> $after"
}
order=0
for clip in A B; do
  for run in ctl:0 marian_threads_2:0 max_utterance_6:0 partial_recheck_translation:0 partial_recheck_translation:1 max_utterance_6:1 marian_threads_2:1 ctl:1 ctl:2 partial_recheck_translation:2 marian_threads_2:2 max_utterance_6:2; do
    run_one $clip ${run%%:*} ${run##*:} $order; order=$((order+1))
  done
done
echo "=== L-B done $(date -u +%FT%TZ) ==="
