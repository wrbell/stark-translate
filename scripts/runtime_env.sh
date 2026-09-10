#!/usr/bin/env bash
# Sourced by launch/bootstrap scripts. Explicit override > activated env > repo env.
stark_resolve_python() {
    local root="$1" candidate
    if [ -n "${STARK_PYTHON:-}" ]; then
        candidate="$STARK_PYTHON"
    elif [ -n "${VENV:-}" ]; then
        candidate="$VENV/bin/python"
    elif [ -n "${VIRTUAL_ENV:-}" ]; then
        candidate="$VIRTUAL_ENV/bin/python"
    elif [ -n "${CONDA_PREFIX:-}" ]; then
        candidate="$CONDA_PREFIX/bin/python"
    elif [ -x "$root/stt_env/bin/python" ]; then
        candidate="$root/stt_env/bin/python"
    elif [ -x "$root/venv/bin/python" ]; then
        candidate="$root/venv/bin/python"
    else
        candidate="$(command -v python3 || true)"
    fi
    if [ ! -x "$candidate" ]; then
        printf 'ERROR: selected Python is not executable: %s\n' "$candidate" >&2
        return 2
    fi
    printf '%s\n' "$candidate"
}
