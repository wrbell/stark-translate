#!/bin/zsh
# P1-E: 3,640 s file-replay service run through the operator on the promoted runtime (no mic, no TTS, no recording).
set -o pipefail
ROOT=/Users/willem/Code/vibes/SRTranslate; E=$ROOT/.cache/series3-20260912/P1E; DATA=$E/root; PORT=9014
mkdir -p $DATA/metrics; cd $ROOT
if pgrep -f "dry_run_ab.py|operator_app.cli operator" >/dev/null; then echo "BUSY"; exit 3; fi
export STARK_PROJECT_ROOT=$DATA STARK_MODELS_DIR=$ROOT/.cache/mac-roadmap/ct2-setup-validation/managed
export STARK_AUDIO_SOURCE=file STARK_AUDIO_FILE="$ROOT/stark_data/raw/Gospel_Message_(12_14_25)_5D2rOMvkwrk.wav"
export STARK_REPLAY_SPEED=1 STARK_SESSION_KIND=replay HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 STARK_OPERATOR_LOG_DIR=$DATA/metrics
echo "=== operator start $(date -u +%FT%TZ)"
$ROOT/venv/bin/python -m operator_app.cli operator --host 127.0.0.1 --port $PORT --no-browser --profile standard > $E/operator.log 2>&1 &
OPID=$!; echo $OPID > $E/operator.pid
for i in $(seq 1 60); do sleep 1; curl -fsS http://127.0.0.1:$PORT/healthz >/dev/null 2>&1 && break; done
echo "healthz after ${i}s"
curl -fsS "http://127.0.0.1:$PORT/api/preflight?backend=mlx&lang=en" > $E/preflight.json; echo "preflight rc=$?"
echo "=== session start $(date -u +%FT%TZ)"
curl -fsS -X POST http://127.0.0.1:$PORT/api/session/start -H 'Content-Type: application/json' \
  -d '{"lang":"en","profile":"standard","backend":"mlx","stt_backend":"parakeet-mlx","gemma4_size":"e4b","record_audio":false,"tts":false,"run_ab":false,"vad_threshold":0.3}' > $E/start.json; echo "start rc=$?"; cat $E/start.json | head -c 400; echo
SID=$(python3 -c "import json; print(json.load(open('$E/start.json')).get('session_id',''))"); PID=$(python3 -c "import json; print(json.load(open('$E/start.json')).get('pid',''))")
echo "session=$SID pid=$PID"
sleep 20
$ROOT/venv/bin/python -m tools.endurance_monitor --root $DATA --session $SID --pid $PID --output $E/monitor --duration-seconds 3900 > $E/monitor.log 2>&1 &
MPID=$!; echo "monitor pid=$MPID"
# wait until the session is running, then until it leaves running (replay ends -> idle/error) or 3,900 s
state() { curl -fsS http://127.0.0.1:$PORT/api/session/status 2>/dev/null | python3 -c "import json,sys; print(json.load(sys.stdin).get('state'))" 2>/dev/null; }
for i in $(seq 1 90); do st=$(state); [ "$st" = running ] && break; sleep 2; done; echo "state=$st after ${i}x2s"
end=$((SECONDS+3900))
while [ $SECONDS -lt $end ]; do
  st=$(state)
  case "$st" in running|starting|paused|stopping) sleep 30 ;; *) echo "session state=$st at $(date -u +%TZ)"; break ;; esac
done
curl -fsS http://127.0.0.1:$PORT/api/session/status > $E/status_final.json 2>/dev/null
curl -fsS -X POST http://127.0.0.1:$PORT/api/session/stop > $E/stop.json 2>/dev/null; echo "stop rc=$?"
sleep 15; wait $MPID 2>/dev/null; echo "monitor rc=$?"
kill -TERM $OPID 2>/dev/null; sleep 3; kill -0 $OPID 2>/dev/null && kill -KILL $OPID; echo "=== operator stopped $(date -u +%FT%TZ)"
ls $DATA/metrics | head; ls $E/monitor
