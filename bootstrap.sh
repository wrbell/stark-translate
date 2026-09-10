#!/usr/bin/env bash
# bootstrap.sh — first-time setup for the stark-translate church PC.
#
# Run this once, on a fresh Ubuntu 24.04 install with an NVIDIA GPU,
# from a checkout of the stark-translate repo. It does:
#
#   1. Verify prerequisites (Python 3.11+, ffmpeg, CUDA toolkit, etc.)
#   2. Create a venv at ./venv and install stark-translate[cuda|cpu] from pyproject
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
INSTALL_LAUNCHD=0
for arg in "$@"; do
    case "$arg" in
        --skip-systemd) SKIP_SYSTEMD=1 ;;
        --install-launchd) INSTALL_LAUNCHD=1 ;;
        --help|-h)
            sed -n '2,/^set -euo/p' "$0" | sed 's/^# \?//'
            exit 0
            ;;
        *) echo "unknown arg: $arg (--help for usage)" >&2; exit 2 ;;
    esac
done

STARK_USER="${STARK_USER:-$USER}"
source "$ROOT/scripts/runtime_env.sh"
PYTHON="$(stark_resolve_python "$ROOT")"

# -----------------------------------------------------------------------------
log() { printf '[bootstrap] %s\n' "$*"; }
fail() { printf '[bootstrap] ERROR: %s\n' "$*" >&2; exit "${2:-2}"; }

# 1. Prerequisites ------------------------------------------------------------
log "checking prerequisites…"

EXTRA="cpu"
if [ "$(uname -s)" = "Darwin" ] && [ "$(uname -m)" = "arm64" ]; then
    EXTRA="mlx"
    log "  Apple Silicon detected — operator will use MLX/Metal. NVIDIA CUDA is not required."
    PYTHON_HINT="install Python 3.11+ (for example: brew install python@3.11)"
    AUDIO_HINT="brew install ffmpeg portaudio"
else
    PYTHON_HINT="install Python 3.11+ (Ubuntu: apt install python3.11 python3.11-venv)"
    AUDIO_HINT="install ffmpeg and PortAudio using your system package manager"
    if command -v nvidia-smi >/dev/null 2>&1; then
        EXTRA="cuda"
        log "  NVIDIA GPU detected — operator will use CUDA."
    else
        log "  NVIDIA GPU not detected — operator will use the CPU backend."
    fi
fi

command -v python3 >/dev/null || fail "python3 not found — $PYTHON_HINT" 2

PY_VER=$("$PYTHON" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
case "$PY_VER" in
    3.11|3.12|3.13|3.14) ;;
    *) fail "python $PY_VER unsupported — $PYTHON_HINT" 2 ;;
esac
log "  python $PY_VER OK"

command -v ffmpeg >/dev/null || fail "ffmpeg not found — $AUDIO_HINT" 2
log "  ffmpeg OK"

# 2. venv + dependencies ------------------------------------------------------
# Prefer the pyproject.toml extras (v2026.7+); fall back to the legacy
# requirements files for environments that pin against the frozen versions.
if [ -n "${STARK_PYTHON:-}" ]; then
    VENV="$("$PYTHON" -c 'import sys; print(sys.prefix if sys.prefix != sys.base_prefix else "")')"
    [ -n "$VENV" ] || VENV="$ROOT/venv"
elif [ -n "${VENV:-}" ]; then
    :
elif [ -n "${VIRTUAL_ENV:-}" ]; then
    VENV="$VIRTUAL_ENV"
elif [ -n "${CONDA_PREFIX:-}" ]; then
    VENV="$CONDA_PREFIX"
elif [ -x "$ROOT/stt_env/bin/python" ]; then
    VENV="$ROOT/stt_env"
else
    VENV="$ROOT/venv"
fi
if [ ! -d "$VENV" ]; then
    log "creating venv at $VENV"
    "$PYTHON" -m venv "$VENV" || fail "venv creation failed" 3
fi

log "installing stark-translate[$EXTRA] into venv (this may take 5–15 minutes)…"
"$VENV/bin/python" -m pip install --upgrade pip wheel >/tmp/bootstrap-pip.log 2>&1 || true
"$VENV/bin/python" -m pip install ".[$EXTRA]" >>/tmp/bootstrap-pip.log 2>&1 \
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

# Explicit launchd setup (never install a login service implicitly).
if [ "$INSTALL_LAUNCHD" -eq 1 ]; then
    "$VENV/bin/python" -m operator_app.cli launchd install --project-root "$ROOT" || fail "launchd install failed" 4
fi

# 4. Model setup and preflight: works even when no server is running.
log "downloading models for $EXTRA (existing snapshots are reused)…"
"$VENV/bin/python" -m operator_app.cli setup --backend "$EXTRA" || fail "model setup failed" 3
"$VENV/bin/python" -m operator_app.cli doctor --backend "$EXTRA" || fail "pre-flight failed; inspect missing models/dependencies/audio" 5

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
