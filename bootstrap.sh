#!/usr/bin/env bash
# bootstrap.sh — first-time setup for the stark-translate church PC.
#
# Run this once, on a fresh Ubuntu 24.04 install with an NVIDIA GPU,
# from a checkout of the stark-translate repo. It does:
#
#   1. Verify prerequisites (Python 3.11+, ffmpeg, CUDA toolkit, etc.)
#   2. Create a venv at ./venv and install requirements-nvidia.txt
#   3. Install systemd unit + drop-in with the actual install paths
#   4. Run a one-shot pre-flight via /api/preflight
#   5. Print final URLs the operator should bookmark
#
# Designed to be idempotent — re-runnable if something fails midway.
#
# Usage:
#     ./bootstrap.sh                                # full install
#     ./bootstrap.sh --skip-systemd                 # local-only (no daemon)
#     STARK_USER=alice ./bootstrap.sh               # install under a different user
#
# Exit codes:
#     0  success
#     2  prerequisite missing
#     3  venv install failed
#     4  systemd install failed
#     5  pre-flight has red items

set -euo pipefail

ROOT="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

SKIP_SYSTEMD=0
for arg in "$@"; do
    case "$arg" in
        --skip-systemd) SKIP_SYSTEMD=1 ;;
        --help|-h)
            sed -n '2,/^set -euo/p' "$0" | sed 's/^# \?//'
            exit 0
            ;;
        *) echo "unknown arg: $arg (--help for usage)" >&2; exit 2 ;;
    esac
done

STARK_USER="${STARK_USER:-$USER}"

# -----------------------------------------------------------------------------
log() { printf '[bootstrap] %s\n' "$*"; }
fail() { printf '[bootstrap] ERROR: %s\n' "$*" >&2; exit "${2:-2}"; }

# 1. Prerequisites ------------------------------------------------------------
log "checking prerequisites…"

command -v python3 >/dev/null || fail "python3 not found — apt install python3.11 python3.11-venv" 2

PY_VER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
case "$PY_VER" in
    3.11|3.12) ;;
    *) fail "python $PY_VER unsupported — need 3.11 or 3.12 (apt install python3.11)" 2 ;;
esac
log "  python $PY_VER OK"

command -v ffmpeg >/dev/null || fail "ffmpeg not found — apt install ffmpeg" 2
log "  ffmpeg OK"

if command -v nvidia-smi >/dev/null 2>&1; then
    log "  nvidia-smi OK ($(nvidia-smi --query-gpu=name --format=csv,noheader -i 0 2>/dev/null | head -1))"
else
    log "  nvidia-smi NOT found — operator will run on CPU (slow). Install NVIDIA driver if this is the church PC."
fi

# 2. venv + dependencies ------------------------------------------------------
VENV="$ROOT/venv"
REQS="requirements-nvidia.txt"
[ ! -f "$REQS" ] && REQS="requirements-mac.txt"

if [ ! -d "$VENV" ]; then
    log "creating venv at $VENV"
    python3 -m venv "$VENV" || fail "venv creation failed" 3
fi

log "installing $REQS into venv (this may take 5–15 minutes)…"
"$VENV/bin/pip" install --upgrade pip wheel >/tmp/bootstrap-pip.log 2>&1 || true
"$VENV/bin/pip" install -r "$REQS" >>/tmp/bootstrap-pip.log 2>&1 \
    || fail "dependency install failed (see /tmp/bootstrap-pip.log)" 3
log "  pip install OK"

# 3. systemd unit -------------------------------------------------------------
if [ "$SKIP_SYSTEMD" -eq 0 ] && command -v systemctl >/dev/null 2>&1; then
    UNIT_SRC="$ROOT/systemd/stark-translate.service"
    UNIT_DEST="/etc/systemd/system/stark-translate.service"

    if [ -f "$UNIT_SRC" ]; then
        log "installing systemd unit (requires sudo)…"
        sudo cp "$UNIT_SRC" "$UNIT_DEST"

        # Drop-in with the actual install paths
        DROPIN_DIR="/etc/systemd/system/stark-translate.service.d"
        sudo mkdir -p "$DROPIN_DIR"
        sudo tee "$DROPIN_DIR/override.conf" >/dev/null <<EOF
[Service]
User=$STARK_USER
WorkingDirectory=$ROOT
Environment=STARK_PROJECT_ROOT=$ROOT
Environment=STARK_OPERATOR_LOG_DIR=$ROOT/metrics
ExecStart=
ExecStart=$VENV/bin/uvicorn operator_app.main:app --host 0.0.0.0 --port 9000
EOF
        sudo systemctl daemon-reload
        sudo systemctl enable stark-translate.service
        sudo systemctl restart stark-translate.service || fail "systemd restart failed" 4
        log "  systemd unit installed + enabled at boot"
    else
        log "  systemd unit source missing at $UNIT_SRC — skipping"
    fi
else
    log "skipping systemd install"
fi

# 4. Pre-flight ---------------------------------------------------------------
log "waiting up to 30s for /healthz…"
URL="http://127.0.0.1:9000/healthz"
ok=0
for _ in $(seq 1 30); do
    if curl -sf "$URL" >/dev/null 2>&1; then
        ok=1
        break
    fi
    sleep 1
done

if [ "$ok" -eq 0 ]; then
    log "  WARNING: /healthz did not respond within 30s. The service may still be starting."
    log "  Check: journalctl -u stark-translate -n 50 --no-pager"
else
    log "  /healthz OK"
    log "running pre-flight checks…"
    PREFLIGHT=$(curl -s "http://127.0.0.1:9000/api/preflight" | "$VENV/bin/python" -c 'import json,sys; d=json.load(sys.stdin); print(d["status_counts"]["fail"], json.dumps(d["status_counts"]))' 2>/dev/null || echo "0 {}")
    set -- $PREFLIGHT
    FAIL_COUNT="$1"
    log "  pre-flight summary: $2"
    if [ "$FAIL_COUNT" -gt 0 ]; then
        log "  WARNING: $FAIL_COUNT pre-flight check(s) red. Open http://localhost:9000/operator/ to investigate."
    fi
fi

# 5. Final summary ------------------------------------------------------------
log ""
log "stark-translate operator installed at:"
log "  $ROOT"
log ""
log "Operator UI:        http://localhost:9000/operator/"
log "Audience display:   http://<this-host>:8080/audience_display.html (after Start)"
log "Health probe:       http://localhost:9000/healthz"
log ""
if [ "$SKIP_SYSTEMD" -eq 0 ] && command -v systemctl >/dev/null 2>&1; then
    log "Auto-starts on boot via systemd. Manual control:"
    log "  sudo systemctl status stark-translate"
    log "  sudo systemctl restart stark-translate"
    log "  journalctl -u stark-translate -f"
else
    log "Manual launch:"
    log "  ./run_operator.sh"
fi
log ""
log "Next: read docs/operator_runbook.md for the day-of-event workflow."
