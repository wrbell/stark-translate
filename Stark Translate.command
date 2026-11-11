#!/bin/bash
# Double-click this file in Finder to start Stark Translate.
# The first time, it installs the program and downloads the models.
# Keep the window open while you use Stark Translate. Close it to stop.
#
# Test mode: "Stark Translate.command" --dry-run (or STARK_LAUNCHER_DRY_RUN=1)
# prints the commands it would run and stops.
set -uo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "${ROOT}" || exit 1
URL="http://localhost:9000/operator/"
LOG="${ROOT}/metrics/install.log"
DRY_RUN="${STARK_LAUNCHER_DRY_RUN:-0}"
[ "${1:-}" = "--dry-run" ] && DRY_RUN=1
# A double-click must not install into a conda or virtualenv from the shell profile.
unset VIRTUAL_ENV CONDA_PREFIX
# shellcheck source=scripts/runtime_env.sh
source "${ROOT}/scripts/runtime_env.sh"

note() { printf '\n'; printf '%s\n' "$@"; }
run() {
    if [ "${DRY_RUN}" = "1" ]; then printf 'would run: %s\n' "$*"; else "$@"; fi
}
stop() {
    note "$1" "Press Return to close this window."
    [ "${DRY_RUN}" = "1" ] || read -r _
    exit 1
}
# Print the Python that can start the operator (pointer, then ./venv). Fail if none.
runtime_python() (
    stark_apply_python_pointer "${ROOT}" || exit 1
    python="$(stark_resolve_python "${ROOT}")" || exit 1
    # Without Xcode tools, /usr/bin/python3 only opens an install dialog.
    [ "${python}" != "/usr/bin/python3" ] || exit 1
    "${python}" -c 'import uvicorn, operator_app.cli' || exit 1
    printf '%s\n' "${python}"
)

if curl -fs --max-time 2 "http://127.0.0.1:9000/healthz" >/dev/null 2>&1; then
    note "Stark Translate is already open. The browser shows it now."
    run open "${URL}"
    exit 0
fi
[ "$(uname -m)" = "arm64" ] || stop "Stark Translate needs a Mac with Apple Silicon (M1 or newer)."

if ! runtime_python >/dev/null 2>&1; then
    if [ -z "${STARK_PYTHON:-}${VENV:-}" ] && [ ! -e "${ROOT}/.stark-python" ]; then
        command -v python3.11 >/dev/null 2>&1 \
            || stop "This Mac does not have Python 3.11. Ask the setup owner to install it, then double-click this file again."
        STARK_PYTHON="$(command -v python3.11)"
        export STARK_PYTHON
        CHOSEN_PYTHON=1
    fi
    note "Stark Translate is not installed yet. The installation starts now." \
        "Keep this window open. Keep the Mac connected to power and to the internet."
    if [ "${DRY_RUN}" = "1" ]; then
        run "${ROOT}/bootstrap.sh" --skip-systemd
    else
        mkdir -p "${ROOT}/metrics"
        PYTHONUNBUFFERED=1 "${ROOT}/bootstrap.sh" --skip-systemd 2>&1 | tee "${LOG}"
        case "${PIPESTATUS[0]}" in
            0) ;;
            2) stop "Installation stopped: this Mac is missing a program, or a setting is wrong. The full log is in ${LOG}." ;;
            3) stop "Installation stopped: the program or model download did not finish. The full log is in ${LOG}." ;;
            5) stop "Installation stopped: the final check found a problem. The full log is in ${LOG}." ;;
            *) stop "Installation stopped because of an error. The full log is in ${LOG}." ;;
        esac
        [ -z "${CHOSEN_PYTHON:-}" ] || unset STARK_PYTHON
        runtime_python >/dev/null 2>&1 || stop "The installation finished, but Stark Translate cannot start. The full log is in ${LOG}."
    fi
fi

note "Stark Translate starts now. The browser opens the operator page." \
    "Keep this window open. Close it to stop Stark Translate."
if [ "${DRY_RUN}" = "1" ]; then
    run "${ROOT}/run_operator.sh" "(browser opens)"
    exit 0
fi
# Exit status 130 means Control-C, which is a normal way to stop.
STARK_OPEN_BROWSER=1 HOST=127.0.0.1 PORT=9000 "${ROOT}/run_operator.sh" \
    || [ "$?" -eq 130 ] || stop "Stark Translate stopped because of an error. Read the messages above."
