#!/usr/bin/env bash
# Retest llama.cpp -fa on for Gemma 4 E4B on the b10883+ pin.
#
# NOT YET EXECUTED ON HARDWARE. Proposal-only until the next WSL session.
# See docs/cuda_latency_proposal.md.
#
# Isolates flash-attn from MTP: A/B t3 with FA off vs on, q8 KV, no draft.
# Optional FA+MTP pass if FA_WITH_MTP=1 and the assistant GGUF exists.
#
# b8782 regressed E4B +56% p50 with -fa on (docs/archive/v2026.9/GEMMA_OPTIM_PHASE2.md).
#
# Usage:
#   scripts/cuda/retest_flash_attn.sh
#   FA_WITH_MTP=1 SPEC_N=3 scripts/cuda/retest_flash_attn.sh
#   scripts/cuda/retest_flash_attn.sh --help

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"

LLAMA_DIR="${LLAMA_DIR:-$HOME/llama.cpp}"
SERVER="${SERVER:-$LLAMA_DIR/build/bin/llama-server}"
MODEL_DIR="${MODEL_DIR:-$ROOT/models}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8090}"
PYTHON="${PYTHON:-python3}"
N_SERMON="${N_SERMON:-125}"
METRICS_DIR="${METRICS_DIR:-$ROOT/metrics/cuda_fa}"
LOG_DIR="${LOG_DIR:-/tmp}"
HEALTH_TIMEOUT_S="${HEALTH_TIMEOUT_S:-120}"
FA_WITH_MTP="${FA_WITH_MTP:-0}"
SPEC_N="${SPEC_N:-3}"

TARGET="${TARGET:-$MODEL_DIR/gemma-4-e4b-it-q4km.gguf}"
ASSISTANT="${ASSISTANT:-$MODEL_DIR/gemma-4-e4b-it-assistant-q4_0.gguf}"

usage() {
    cat <<EOF
A/B Gemma 4 E4B with -fa off vs -fa on (no MTP by default).

NOT YET EXECUTED ON HARDWARE.

Env / defaults:
  SERVER / TARGET / PORT / PYTHON / N_SERMON
  FA_WITH_MTP   1 = extra run: -fa on + draft-mtp (f16 KV)   [${FA_WITH_MTP}]
  SPEC_N        n-max for the optional MTP+FA run            [${SPEC_N}]
  METRICS_DIR                                                [${METRICS_DIR}]

Gate: FA-on p50 <= FA-off p50, canary 7/8, VRAM <= baseline+0.2 GB.
If FA+MTP accept drops >5 pp vs MTP-only, ship MTP without FA.
EOF
}

for arg in "$@"; do
    case "$arg" in
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown arg: $arg (try --help)" >&2; exit 1 ;;
    esac
done

if [ ! -x "${SERVER}" ]; then
    echo "ERROR: llama-server not found at ${SERVER}" >&2
    exit 1
fi
if [ ! -f "${TARGET}" ]; then
    echo "ERROR: missing ${TARGET}" >&2
    exit 1
fi

mkdir -p "${METRICS_DIR}"

wait_health() {
    local i
    for i in $(seq 1 "${HEALTH_TIMEOUT_S}"); do
        if curl -s --max-time 1 "http://${HOST}:${PORT}/health" 2>/dev/null | grep -q '"status":"ok"'; then
            return 0
        fi
        sleep 1
    done
    echo "ERROR: llama-server not healthy within ${HEALTH_TIMEOUT_S}s" >&2
    return 1
}

stop_server() {
    if [ -n "${SERVER_PID:-}" ] && kill -0 "${SERVER_PID}" 2>/dev/null; then
        kill -TERM "${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
    fi
    SERVER_PID=""
    if (timeout 1 bash -c "exec 3<>/dev/tcp/${HOST}/${PORT}" 2>/dev/null); then
        pkill -f "llama-server.*--port ${PORT}" || true
        sleep 2
    fi
}

start_server() {
    local log="$1"
    shift
    stop_server
    echo "==> ${SERVER} $*  (log ${log})"
    "${SERVER}" --host "${HOST}" --port "${PORT}" -ngl 999 -c 512 "$@" >"${log}" 2>&1 &
    SERVER_PID=$!
    if ! wait_health; then
        tail -50 "${log}" >&2 || true
        stop_server
        exit 3
    fi
}

run_bench() {
    local cfg="$1"
    local log="$2"
    local out="$3"
    "${PYTHON}" scripts/benchmarks/bench_translate_t1_t4.py \
        --config "${cfg}" \
        --server-url "http://${HOST}:${PORT}" \
        --server-log "${log}" \
        --n-sermon "${N_SERMON}" \
        --out "${out}"
}

trap 'stop_server' EXIT INT TERM

echo "NOT YET EXECUTED ON HARDWARE."

LOG="${LOG_DIR}/llama_t3_fa_off.log"
start_server "${LOG}" -m "${TARGET}" -ctk q8_0 -ctv q8_0
run_bench t3 "${LOG}" "${METRICS_DIR}/t3_fa_off.json"

LOG="${LOG_DIR}/llama_t3_fa_on.log"
start_server "${LOG}" -m "${TARGET}" -ctk q8_0 -ctv q8_0 -fa on
run_bench t3 "${LOG}" "${METRICS_DIR}/t3_fa_on.json"
grep -E "flash_attn|flash attn" "${LOG}" | head -5 || true

if [ "${FA_WITH_MTP}" = "1" ]; then
    if [ ! -f "${ASSISTANT}" ]; then
        echo "ERROR: FA_WITH_MTP=1 but missing ${ASSISTANT}" >&2
        exit 1
    fi
    LOG="${LOG_DIR}/llama_t3_fa_mtp.log"
    start_server "${LOG}" \
        -m "${TARGET}" \
        -md "${ASSISTANT}" \
        --spec-type draft-mtp \
        --spec-draft-n-max "${SPEC_N}" \
        -fa on
    run_bench t3-mtp "${LOG}" "${METRICS_DIR}/t3_fa_mtp.json"
    grep -E "draft acceptance|statistics +draft-mtp" "${LOG}" | tail -10 || true
fi

echo "==> results in ${METRICS_DIR}"
ls -l "${METRICS_DIR}"
echo "NOT YET EXECUTED ON HARDWARE."
