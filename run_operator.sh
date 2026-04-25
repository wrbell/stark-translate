#!/usr/bin/env bash
# Launch the operator control plane (Phase 9).
#
# This wraps `uvicorn operator_app.main:app` with sensible defaults so a
# non-technical operator can start the service from a single command. The
# systemd unit and launchd plist both shell out to this script in
# production; running it directly is the same behavior.
#
# Usage:
#     ./run_operator.sh                                # default 0.0.0.0:9000
#     PORT=9001 ./run_operator.sh                      # custom port
#     STARK_OPERATOR_LOG_DIR=... ./run_operator.sh     # custom log dir

set -euo pipefail

ROOT="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-9000}"
VENV="${VENV:-$ROOT/venv}"

export STARK_PROJECT_ROOT="$ROOT"
export STARK_OPERATOR_LOG_DIR="${STARK_OPERATOR_LOG_DIR:-$ROOT/metrics}"

mkdir -p "$STARK_OPERATOR_LOG_DIR"

UVICORN="$VENV/bin/uvicorn"
if [ ! -x "$UVICORN" ]; then
    UVICORN="$(command -v uvicorn || true)"
fi
if [ -z "$UVICORN" ]; then
    echo "ERROR: uvicorn not found. Activate venv or run ./bootstrap.sh." >&2
    exit 2
fi

echo "starting operator at http://$HOST:$PORT (logs: $STARK_OPERATOR_LOG_DIR)"
exec "$UVICORN" operator_app.main:app --host "$HOST" --port "$PORT"
