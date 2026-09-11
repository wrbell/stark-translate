#!/usr/bin/env bash
# Launch the operator control plane (Phase 9).
#
# This wraps `uvicorn operator_app.main:app` with sensible defaults so a
# non-technical operator can start the service from a single command. The
# systemd unit and launchd plist both shell out to this script in
# production; running it directly is the same behavior.
#
# Usage:
#     ./run_operator.sh                                # default 127.0.0.1:9000
#     PORT=9001 ./run_operator.sh                      # custom port
#     STARK_OPERATOR_LOG_DIR=... ./run_operator.sh     # custom log dir

set -euo pipefail

ROOT="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-9000}"
source "$ROOT/scripts/runtime_env.sh"
stark_apply_python_pointer "$ROOT"
PYTHON="$(stark_resolve_python "$ROOT")"

export STARK_PROJECT_ROOT="$ROOT"
export STARK_OPERATOR_LOG_DIR="${STARK_OPERATOR_LOG_DIR:-$ROOT/metrics}"

mkdir -p "$STARK_OPERATOR_LOG_DIR"

if ! "$PYTHON" -c 'import uvicorn' >/dev/null 2>&1; then
    echo "ERROR: uvicorn missing in $PYTHON. Run ./bootstrap.sh." >&2
    exit 2
fi

echo "starting operator at http://$HOST:$PORT (logs: $STARK_OPERATOR_LOG_DIR; python: $PYTHON $("$PYTHON" --version 2>&1 || true))"
exec "$PYTHON" -m operator_app.cli operator --no-browser --host "$HOST" --port "$PORT"
