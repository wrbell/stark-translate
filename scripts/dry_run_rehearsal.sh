#!/usr/bin/env bash
# dry_run_rehearsal.sh — automated end-to-end check of the operator surface.
#
# Walks every endpoint a real day-of-event session touches:
#   1. /healthz returns ok
#   2. /api/preflight has no failed checks (warn is fine)
#   3. /api/session/start launches a session
#   4. /api/control/lang_flip flips, then flips back (round trip)
#   5. /api/session/stop returns to idle
#   6. /api/features/verses responds with a (possibly empty) list
#
# Designed to be runnable by a non-technical volunteer the day before
# an event. Exits 0 on green, 1 on any check that fails.
#
# Usage:
#     ./scripts/dry_run_rehearsal.sh                 # localhost:9000
#     OPERATOR_URL=http://10.0.0.5:9000 ./scripts/dry_run_rehearsal.sh
#
# Dependencies: bash, curl, python3 (used only for tiny JSON parsing).

set -euo pipefail

URL="${OPERATOR_URL:-http://localhost:9000}"
PASS_PREFIX="  ✓"
FAIL_PREFIX="  ✗"
WARN_PREFIX="  !"

green() { printf '\033[32m%s\033[0m' "$*"; }
red()   { printf '\033[31m%s\033[0m' "$*"; }
yellow(){ printf '\033[33m%s\033[0m' "$*"; }

step() { printf "\n[%s] %s\n" "$1" "$2"; }

ok=0
fail=0

check() {
    local label="$1"; local result="$2"
    if [ "$result" = "ok" ]; then
        printf '%s %s\n' "$(green "$PASS_PREFIX")" "$label"
        ok=$((ok + 1))
    elif [ "$result" = "warn" ]; then
        printf '%s %s\n' "$(yellow "$WARN_PREFIX")" "$label"
    else
        printf '%s %s\n' "$(red "$FAIL_PREFIX")" "$label"
        fail=$((fail + 1))
    fi
}

# Tiny helpers
http_get()  { curl -sS --max-time 5 "$URL$1"; }
http_post() { curl -sS --max-time 30 -X POST -H "Content-Type: application/json" -d "$2" "$URL$1"; }
http_status_get()  { curl -sS --max-time 5 -o /dev/null -w '%{http_code}' "$URL$1"; }

json_field() { python3 -c "import sys, json; d=json.load(sys.stdin); print(d.get('$1', ''))" 2>/dev/null || true; }

# -- 1. /healthz --------------------------------------------------------------
step 1 "Health probe"
status="$(http_get /healthz | json_field status || true)"
check "/healthz returns status=ok" "$([ "$status" = "ok" ] && echo ok || echo fail)"

# -- 2. /api/preflight --------------------------------------------------------
step 2 "Pre-flight"
preflight=$(http_get /api/preflight)
counts=$(printf '%s' "$preflight" | python3 -c "import sys,json; d=json.load(sys.stdin); c=d['status_counts']; print(f'{c[\"pass\"]} pass, {c[\"warn\"]} warn, {c[\"fail\"]} fail')" 2>/dev/null || echo "(parse error)")
echo "    $counts"
fail_count=$(printf '%s' "$preflight" | python3 -c "import sys,json; print(json.load(sys.stdin)['status_counts']['fail'])" 2>/dev/null || echo 1)
check "no red checks" "$([ "$fail_count" = "0" ] && echo ok || echo fail)"

# -- 3. /api/session/start ----------------------------------------------------
step 3 "Start session (lang=en, default args)"
start_resp=$(http_post /api/session/start '{"lang":"en"}')
state=$(printf '%s' "$start_resp" | json_field state)
session_id=$(printf '%s' "$start_resp" | json_field session_id)
case "$state" in
    starting|running) check "session state=$state, id=$session_id" ok ;;
    *)                check "session start (state=$state)" fail ;;
esac

# Wait briefly for the subprocess to flip running.
for _ in $(seq 1 20); do
    state=$(http_get /api/session/status | json_field state)
    [ "$state" = "running" ] && break
    sleep 0.5
done
check "session reaches running state" "$([ "$state" = "running" ] && echo ok || echo fail)"

# -- 4. /api/control/lang_flip (round trip) -----------------------------------
step 4 "Language-flip round trip"
flip_a=$(http_post /api/control/lang_flip '{}' | json_field state)
sleep 0.5
flip_b=$(http_post /api/control/lang_flip '{}' | json_field state)
check "first flip succeeded (state=$flip_a)" "$([ "$flip_a" != "" ] && [ "$flip_a" != "error" ] && echo ok || echo fail)"
check "second flip restored (state=$flip_b)" "$([ "$flip_b" != "" ] && [ "$flip_b" != "error" ] && echo ok || echo fail)"

# -- 5. /api/session/stop -----------------------------------------------------
step 5 "Stop session"
stop_state=$(http_post /api/session/stop '{}' | json_field state)
check "session returned to idle (state=$stop_state)" "$([ "$stop_state" = "idle" ] && echo ok || echo fail)"

# -- 6. /api/features/verses --------------------------------------------------
step 6 "Verse highlights endpoint"
verses_status=$(http_status_get /api/features/verses)
check "/api/features/verses responds 200" "$([ "$verses_status" = "200" ] && echo ok || echo fail)"

# -- summary ------------------------------------------------------------------
echo
if [ "$fail" -eq 0 ]; then
    echo "$(green '✓ rehearsal passed') — $ok checks green"
    exit 0
else
    echo "$(red '✗ rehearsal failed') — $fail check(s) red, $ok green"
    exit 1
fi
