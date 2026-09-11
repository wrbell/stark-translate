#!/usr/bin/env bash
# STARK_PYTHON > VENV > .stark-python > VIRTUAL_ENV > CONDA_PREFIX > stt_env > venv > python3.
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

# Apply in the caller's shell before resolving Python in a command substitution.
stark_apply_python_pointer() {
    local root="$1" line candidate=""
    if [ -n "${STARK_PYTHON:-}" ] || [ -n "${VENV:-}" ]; then
        if [ -e "$root/.stark-python" ]; then
            printf '%s\n' 'notice: .stark-python ignored because STARK_PYTHON/VENV is set' >&2
        fi
        return 0
    fi
    [ -e "$root/.stark-python" ] || return 0
    while IFS= read -r line || [ -n "$line" ]; do
        line="${line%$'\r'}"
        line="${line#"${line%%[![:space:]]*}"}"
        line="${line%"${line##*[![:space:]]}"}"
        case "$line" in
            ""|\#*) continue ;;
            \~/*) candidate="$HOME/${line#\~/}" ;;
            /*) candidate="$line" ;;
            *) candidate="$root/$line" ;;
        esac
        break
    done < "$root/.stark-python"
    if [ -z "$candidate" ]; then
        printf 'ERROR: %s/.stark-python contains no interpreter path\n' "$root" >&2
        return 2
    fi
    if [ ! -f "$candidate" ] || [ ! -x "$candidate" ]; then
        printf 'ERROR: interpreter from %s/.stark-python is not executable: %s\n' "$root" "$candidate" >&2
        return 2
    fi
    export STARK_PYTHON="$candidate"
    return 0
}
