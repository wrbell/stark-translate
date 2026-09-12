#!/bin/zsh
# P1 micro-bench: same venv interpreter, control checkout (5227a73) vs P1 worktree; text bench then Parakeet joint eval.
set -o pipefail
ROOT=/Users/willem/Code/vibes/SRTranslate; P1=$ROOT/.cache/series4-20260912/P1; BASE=/Users/willem/Code/vibes/SRTranslate-wt-base; CAND=/Users/willem/Code/vibes/SRTranslate-wt-p1
PY=$ROOT/venv/bin/python
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
if pgrep -f "dry_run_ab.py|benchmark_mlx_accel|parakeet_joint_eval|parakeet_profile" >/dev/null; then echo "BUSY"; exit 3; fi
for pair in "ctl:$BASE" "cand:$CAND" "ctl2:$BASE" "cand2:$CAND"; do
  arm=${pair%%:*}; dir=${pair#*:}
  echo "=== textbench $arm from $dir start $(date -u +%FT%TZ)"
  (cd $dir && $PY tools/benchmark_mlx_accel.py --configs e4b --sentences all --runs 5 --warmup 2 --no-stt --output $P1/textbench_${arm}.json 2>&1 | grep -vE '^\s*$' | tail -4)
  echo "rc=$? end $(date -u +%FT%TZ)"
done
echo "=== parakeet joint eval (P1 checkout: stock engine control vs qualified joint method) start $(date -u +%FT%TZ)"
(cd $CAND && $PY tools/parakeet_joint_eval.py --manifest docs/evaluation/mac_followup_20260910/public_data/manifest.json --audio-root $ROOT/.cache/mac-en-es-closeout/fleurs-v1 --profile-receipt $ROOT/.cache/mac-en-es-closeout/parakeet-profile/development-r1.json --profile-sha256 68478068c956f296cfe8b6e505af6aa87368ebd963551ea9ba7add53b4cc5190 --languages en es --limit 3 --repeats 3 --timeout-seconds 1800 --output $P1/joint_eval_cand.json 2>&1 | tail -5)
echo "rc=$? end $(date -u +%FT%TZ)"
echo "=== parakeet profile: control checkout (stock) start $(date -u +%FT%TZ)"
(cd $BASE && $PY tools/parakeet_profile.py --manifest docs/evaluation/mac_followup_20260910/public_data/manifest.json --audio-root $ROOT/.cache/mac-en-es-closeout/fleurs-v1 --languages en --limit 3 --repeats 3 --timeout-seconds 1800 --output $P1/profile_ctl.json 2>&1 | tail -3)
echo "rc=$? end $(date -u +%FT%TZ)"
echo "=== parakeet profile: P1 checkout (joint decode active by default) start $(date -u +%FT%TZ)"
(cd $CAND && $PY tools/parakeet_profile.py --manifest docs/evaluation/mac_followup_20260910/public_data/manifest.json --audio-root $ROOT/.cache/mac-en-es-closeout/fleurs-v1 --languages en --limit 3 --repeats 3 --timeout-seconds 1800 --output $P1/profile_cand.json 2>&1 | tail -3)
echo "rc=$? end $(date -u +%FT%TZ)"
echo "=== microbench done $(date -u +%FT%TZ) ==="
