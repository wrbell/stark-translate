#!/usr/bin/env bash
# Audit the selected Mac runtime without installing packages or loading models.
set -euo pipefail

ROOT="$(cd -- "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT=""
AUDITOR=""
fail() { printf 'ERROR: %s\n' "$*" >&2; exit 2; }
while [ "$#" -gt 0 ]; do
    case "$1" in
        --output|--auditor)
            [ "$#" -ge 2 ] && [ -n "$2" ] || fail "$1 requires a path"
            case "$1" in
                --output) OUT="$2" ;;
                --auditor) AUDITOR="$2" ;;
            esac
            shift 2
            ;;
        --help|-h)
            printf 'Usage: %s --output DIR [--auditor PATH]\n' "$0"
            exit 0
            ;;
        *) fail "unknown argument: $1" ;;
    esac
done
[ -n "$OUT" ] || fail "--output DIR is required"
[ ! -e "$OUT" ] && [ ! -L "$OUT" ] || fail "output must not exist: $OUT"
if [ -z "$AUDITOR" ]; then
    AUDITOR="$(command -v pip-audit || true)"
    if [ -z "$AUDITOR" ]; then
        AUDITOR="$ROOT/.cache/package-smoke/bin/pip-audit"
    fi
fi
[ -f "$AUDITOR" ] && [ -x "$AUDITOR" ] || fail "pip-audit is not executable: $AUDITOR"
AUDITOR="$(cd -- "$(dirname "$AUDITOR")" && pwd)/$(basename "$AUDITOR")"

source "$ROOT/scripts/runtime_env.sh"
stark_apply_python_pointer "$ROOT"
PYTHON="$(stark_resolve_python "$ROOT")"
mkdir -p -- "$(dirname "$OUT")"
mkdir -- "$OUT"
OUT="$(cd -- "$OUT" && pwd)"
SITE_EXIT=0
AUDIT_EXIT=null
VALIDATION_EXIT=null
SITE="$("$PYTHON" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')" || SITE_EXIT=$?
if [ "$SITE_EXIT" -eq 0 ] && [ -z "$SITE" ]; then
    printf 'ERROR: selected interpreter returned no site-packages path\n' >&2
    SITE_EXIT=2
fi
if [ "$SITE_EXIT" -eq 0 ]; then
    AUDIT_EXIT=0
    "$AUDITOR" --path "$SITE" --format json --output "$OUT/installed-audit.json" || AUDIT_EXIT=$?
    VALIDATION_EXIT=0
    "$PYTHON" "$ROOT/tools/check_dependency_audit.py" --runtime mac "$OUT/installed-audit.json" || VALIDATION_EXIT=$?
fi
"$PYTHON" - "$OUT/receipt.json" "$PYTHON" "$SITE" "$AUDITOR" "$SITE_EXIT" "$AUDIT_EXIT" "$VALIDATION_EXIT" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

output, interpreter, site, auditor, site_exit, audit_exit, validation_exit = sys.argv[1:]
receipt = {
    "interpreter": interpreter,
    "site_packages": site,
    "auditor": auditor,
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "exit_codes": {
        "site_packages": json.loads(site_exit),
        "audit": json.loads(audit_exit),
        "validation": json.loads(validation_exit),
    },
}
Path(output).write_text(json.dumps(receipt, indent=2) + "\n")
PY
if [ "$SITE_EXIT" -ne 0 ]; then exit "$SITE_EXIT"; fi
if [ "$AUDIT_EXIT" -ne 0 ]; then exit "$AUDIT_EXIT"; fi
exit "$VALIDATION_EXIT"
