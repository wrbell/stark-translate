#!/usr/bin/env bash
# A/B llama.cpp E4B (and optional E2B) with/without --spec-type draft-mtp.
#
# NOT YET EXECUTED ON HARDWARE. Proposal-only until the next WSL session.
# See docs/cuda_latency_proposal.md.
#
# Starts llama-server itself (kills on exit). Requires port 8090 free.
# For each n-max in SPEC_N_LIST (default 2 3 4): f16 KV, Q4_0 assistant,
# runs bench_translate_t1_t4.py --config t3-mtp (and t2-mtp if RUN_T2=1).
# Baseline t3 (no MTP, q8 KV) runs first.
#
# Usage:
#   scripts/cuda/bench_mtp.sh
#   SPEC_N_LIST="2 3 4" scripts/cuda/bench_mtp.sh
#   scripts/cuda/bench_mtp.sh --help

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
SPEC_N_LIST="${SPEC_N_LIST:-2 3 4}"
RUN_T2="${RUN_T2:-0}"
METRICS_DIR="${METRICS_DIR:-$ROOT/metrics/cuda_mtp}"
LOG_DIR="${LOG_DIR:-/tmp}"
HEALTH_TIMEOUT_S="${HEALTH_TIMEOUT_S:-120}"

TARGET_E4B="${TARGET_E4B:-$MODEL_DIR/gemma-4-e4b-it-q4km.gguf}"
ASSIST_E4B="${ASSIST_E4B:-$MODEL_DIR/gemma-4-e4b-it-assistant-q4_0.gguf}"
TARGET_E2B="${TARGET_E2B:-$MODEL_DIR/gemma-4-e2b-it-q4km.gguf}"
ASSIST_E2B="${ASSIST_E2B:-$MODEL_DIR/gemma-4-e2b-it-assistant-q4_0.gguf}"

usage() {
    cat <<EOF
Sweep Gemma 4 MTP n-max against bench_translate_t1_t4.py.

NOT YET EXECUTED ON HARDWARE.

Env / defaults:
  SERVER / LLAMA_DIR   llama-server binary
  MODEL_DIR            GGUF dir
  SPEC_N_LIST          n-max values (space-separated)     [${SPEC_N_LIST}]
  N_SERMON             sermon chunks                      [${N_SERMON}]
  RUN_T2               1 = also t2 / t2-mtp               [${RUN_T2}]
  PORT                 llama-server port                  [${PORT}]
  PYTHON               interpreter                        [${PYTHON}]
  METRICS_DIR          JSON output dir                    [${METRICS_DIR}]

Collects server timings + draft acceptance via --server-log (parse_server_timings
understands both n_drafted/n_accept and 'draft acceptance = … (acc / gen)').
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
    echo "Run scripts/cuda/build_llamacpp.sh first." >&2
    exit 1
fi
if [ ! -f "${TARGET_E4B}" ]; then
    echo "ERROR: missing target ${TARGET_E4B}" >&2
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
    # leftover from a previous crash
    if (timeout 1 bash -c "exec 3<>/dev/tcp/${HOST}/${PORT}" 2>/dev/null); then
        echo "WARN: ${HOST}:${PORT} still bound — pkill llama-server" >&2
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
    echo "==> bench --config ${cfg} -> ${out}"
    "${PYTHON}" scripts/benchmarks/bench_translate_t1_t4.py \
        --config "${cfg}" \
        --server-url "http://${HOST}:${PORT}" \
        --server-log "${log}" \
        --n-sermon "${N_SERMON}" \
        --out "${out}"
}

trap 'stop_server' EXIT INT TERM

echo "NOT YET EXECUTED ON HARDWARE — this script will occupy the GPU for ~tens of minutes."

# --- baseline T3: q8 KV, no draft, no FA (production-without-MTP) ----------
LOG="${LOG_DIR}/llama_t3_baseline.log"
start_server "${LOG}" -m "${TARGET_E4B}" -ctk q8_0 -ctv q8_0
run_bench t3 "${LOG}" "${METRICS_DIR}/t3.json"

if [ "${RUN_T2}" = "1" ] && [ -f "${TARGET_E2B}" ]; then
    LOG="${LOG_DIR}/llama_t2_baseline.log"
    start_server "${LOG}" -m "${TARGET_E2B}" -ctk q8_0 -ctv q8_0
    run_bench t2 "${LOG}" "${METRICS_DIR}/t2.json"
fi

# --- MTP sweep: f16 KV (omit -ctk/-ctv), --spec-type draft-mtp --------------
if [ ! -f "${ASSIST_E4B}" ]; then
    echo "ERROR: missing assistant ${ASSIST_E4B}" >&2
    echo "Run scripts/cuda/convert_gemma4_assistant_gguf.sh first." >&2
    exit 1
fi

for N in ${SPEC_N_LIST}; do
    LOG="${LOG_DIR}/llama_t3_mtp_n${N}.log"
    start_server "${LOG}" \
        -m "${TARGET_E4B}" \
        -md "${ASSIST_E4B}" \
        --spec-type draft-mtp \
        --spec-draft-n-max "${N}"
    run_bench t3-mtp "${LOG}" "${METRICS_DIR}/t3-mtp-n${N}.json"
    echo "==> acceptance / timings: grep the log"
    grep -E "draft acceptance|statistics +draft-mtp" "${LOG}" | tail -20 || true
done

if [ "${RUN_T2}" = "1" ] && [ -f "${TARGET_E2B}" ] && [ -f "${ASSIST_E2B}" ]; then
    for N in ${SPEC_N_LIST}; do
        LOG="${LOG_DIR}/llama_t2_mtp_n${N}.log"
        start_server "${LOG}" \
            -m "${TARGET_E2B}" \
            -md "${ASSIST_E2B}" \
            --spec-type draft-mtp \
            --spec-draft-n-max "${N}"
        run_bench t2-mtp "${LOG}" "${METRICS_DIR}/t2-mtp-n${N}.json"
        grep -E "draft acceptance|statistics +draft-mtp" "${LOG}" | tail -20 || true
    done
fi

echo "==> results in ${METRICS_DIR}"
ls -l "${METRICS_DIR}"
echo "Gate: t3-mtp p50 <= 300 ms, canary >= 7/8, accept >= 0.45, VRAM(E4B+STT+Marian) <= 12 GB."
echo "NOT YET EXECUTED ON HARDWARE."
