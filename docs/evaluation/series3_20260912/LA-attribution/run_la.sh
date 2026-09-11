#!/bin/zsh
# L-A attribution inputs: isolated E4B text bench + 3 traced control replays on clip A (with cpu_ms fields) + process CPU sampling.
set -o pipefail
ROOT=/Users/willem/Code/vibes/SRTranslate; LA=$ROOT/.cache/series3-20260912/LA; X=$ROOT/.cache/followup-20260911/X
PY=$ROOT/venv/bin/python; cd $ROOT
if pgrep -f "dry_run_ab.py|operator_app.cli operator" >/dev/null; then echo BUSY; exit 3; fi
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "=== source $(git rev-parse --short HEAD) $(date -u +%FT%TZ)"
echo "=== isolated E4B text bench"
$PY tools/benchmark_mlx_accel.py --configs e4b --sentences all --runs 5 --warmup 2 --no-stt --output $LA/textbench_e4b.json 2>&1 | grep -vE '^\s*$' | tail -8; echo "rc=$?"
export STARK_EXPERIMENT_TRACE=true STARK_EXPERIMENT_TRACE_CAPACITY=131072
EXTRA='--profile standard --no-tts --stt-backend parakeet-mlx --model-family gemma4 --gemma4-size e4b --vad-backend torch --no-mts --gain 1 --replay-speed 1'
for rep in 0 1 2; do
  tag=la0912_A_ctl_r${rep}
  if pgrep -f dry_run_ab.py >/dev/null; then echo BUSY; exit 3; fi
  echo "=== $tag $(date -u +%FT%TZ)"
  # process CPU sampler (per-process tree, 5 s) for the replay child, started once the pipeline appears
  ( for i in $(seq 1 60); do sleep 1; pid=$(pgrep -f "dry_run_ab.py" | head -1); [ -n "$pid" ] && break; done
    [ -n "$pid" ] && while kill -0 $pid 2>/dev/null; do ps -o pid=,%cpu=,rss= -p $pid | awk -v t=$(date +%s) '{print t","$1","$2","$3}' >> $LA/cpu_${tag}.csv; sleep 5; done ) &
  $PY -m tools.replay_bench --manifest $X/manifest_A.json --tag $tag --configs "ctl=$EXTRA" 2>&1 | tail -3; echo "rc=$?"
  wait
done
echo "=== attribution $(date -u +%FT%TZ)"
$PY tools/stt_overlap_attribution.py $(for f in metrics/diagnostics_ts0911_{A,B}_ctl_r{0,1,2}_*.jsonl metrics/diagnostics_la0912_A_ctl_r{0,1,2}_*.jsonl; do echo --diagnostics $f; done) --output $LA/attribution.json --markdown $LA/attribution.md 2>&1 | tail -5; echo "rc=$?"
echo "=== L-A done $(date -u +%FT%TZ)"
