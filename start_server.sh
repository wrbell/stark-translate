#!/bin/bash
# Start llama-server for translation.
#
# Usage:
#   ./start_server.sh                    # E4B target only (production default)
#   ./start_server.sh --no-draft         # same, explicit
#   ./start_server.sh --mtp              # E4B + official Gemma 4 MTP assistant
#   ./start_server.sh --e2b-draft        # legacy T4 (measured LOSS — not recommended)
#   ./start_server.sh --model <path>     # Custom target GGUF
#   ./start_server.sh --help
#
# llama.cpp pin: tag b10883 (2026-09-09). Must match Dockerfile ARG LLAMA_CPP_REF
# and scripts/cuda/build_llamacpp.sh. Includes Gemma 4 MTP (#23398, #24282) and
# gemma4-assistant fix (#28183). Older pins: d8794eecd / b9022 (v2026.9).
# See docs/cuda_latency_proposal.md.
#
# Prerequisites:
#   scripts/cuda/build_llamacpp.sh
#   scripts/cuda/convert_gemma4_assistant_gguf.sh   # only for --mtp

set -euo pipefail

LLAMA_DIR="${LLAMA_DIR:-$HOME/llama.cpp}"
SERVER="${SERVER:-$LLAMA_DIR/build/bin/llama-server}"
MODEL_DIR="${MODEL_DIR:-models}"

# Defaults: E4B target, no speculative draft (T4 E2B→E4B was a single-GPU loss).
TARGET="${MODEL_DIR}/gemma-4-e4b-it-q4km.gguf"
DRAFT="${MODEL_DIR}/gemma-4-e2b-it-q4km.gguf"
ASSISTANT="${ASSISTANT:-}"
HOST="127.0.0.1"
PORT="8090"
NO_DRAFT=true
MTP=false
E2B_DRAFT=false
FLASH_ATTN="${FLASH_ATTN:-off}"
SPEC_N="${SPEC_N:-3}"

usage() {
    cat <<EOF
Start llama-server for Stark Road translation.

  ./start_server.sh                 E4B Q4_K_M, q8 KV, no draft (default)
  ./start_server.sh --no-draft      same
  ./start_server.sh --mtp           official Gemma 4 MTP assistant, f16 KV
  ./start_server.sh --e2b-draft     legacy E2B spec decode (NOT recommended)
  ./start_server.sh --flash-attn    add -fa on (retest on b10883; was +56% on b8782)
  ./start_server.sh --model PATH    override target GGUF
  ./start_server.sh --assistant PATH
  ./start_server.sh --draft PATH    E2B GGUF for --e2b-draft
  ./start_server.sh --port N

Env: LLAMA_DIR, SERVER, MODEL_DIR, ASSISTANT, SPEC_N (default 3), FLASH_ATTN=on|off
Pin: llama.cpp b10883 — see docs/cuda_latency_proposal.md
EOF
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help) usage; exit 0 ;;
        --no-draft) NO_DRAFT=true; MTP=false; E2B_DRAFT=false; shift ;;
        --mtp) MTP=true; NO_DRAFT=true; E2B_DRAFT=false; shift ;;
        --e2b-draft) E2B_DRAFT=true; NO_DRAFT=false; MTP=false; shift ;;
        --flash-attn|--fa) FLASH_ATTN=on; shift ;;
        --model) TARGET="$2"; shift 2 ;;
        --draft) DRAFT="$2"; shift 2 ;;
        --assistant) ASSISTANT="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        *) echo "Unknown arg: $1 (try --help)" >&2; exit 1 ;;
    esac
done

if [ -z "${ASSISTANT}" ]; then
    if [[ "${TARGET}" == *e2b* ]]; then
        ASSISTANT="${MODEL_DIR}/gemma-4-e2b-it-assistant-q4_0.gguf"
    else
        ASSISTANT="${MODEL_DIR}/gemma-4-e4b-it-assistant-q4_0.gguf"
    fi
fi

if [ ! -f "$SERVER" ]; then
    echo "ERROR: llama-server not found at $SERVER"
    echo "Build it: scripts/cuda/build_llamacpp.sh"
    echo "  (or: cd $LLAMA_DIR && cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 && cmake --build build --config Release -j\$(nproc) --target llama-server)"
    exit 1
fi

if [ ! -f "$TARGET" ]; then
    echo "ERROR: Model not found at $TARGET"
    echo "Export GGUF first — see docs/cuda_latency_proposal.md"
    exit 1
fi

if [ "$MTP" = true ] && [ ! -f "$ASSISTANT" ]; then
    echo "ERROR: MTP assistant not found at $ASSISTANT"
    echo "Convert it: scripts/cuda/convert_gemma4_assistant_gguf.sh"
    exit 1
fi

if [ "$E2B_DRAFT" = true ] && [ ! -f "$DRAFT" ]; then
    echo "ERROR: --e2b-draft requested but draft not found at $DRAFT" >&2
    exit 1
fi

if [ "$MTP" = true ]; then
    echo "Starting llama-server: target + Gemma 4 MTP assistant (f16 KV)"
    echo "  Assistant: $ASSISTANT"
    echo "  spec-draft-n-max: $SPEC_N"
elif [ "$E2B_DRAFT" = true ]; then
    echo "Starting llama-server: target + E2B draft (LEGACY spec decode — measured loss)"
    echo "  Draft:  $DRAFT"
else
    echo "Starting llama-server: target only (--no-draft default)"
fi

echo "  Target: $TARGET"
echo "  URL:    http://$HOST:$PORT"
echo ""

# Pre-flight: refuse to start if port is already in use. /dev/tcp is bash-built-in
# so no extra dependency. Honors HOST so we can run two servers on different ports.
if (timeout 2 bash -c "exec 3<>/dev/tcp/$HOST/$PORT" 2>/dev/null); then
    echo "ERROR: $HOST:$PORT is already in use — another llama-server (or other process)" >&2
    echo "       is bound there. Stop it first: pkill -f 'llama-server.*--port $PORT'" >&2
    exit 2
fi

# Launch llama-server in the background and emit READY when it answers /health.
# Caller scripts can `until grep -q READY <log>` to gate startup. SIGINT/SIGTERM
# in the foreground kill the child so Ctrl-C still works as expected.
#
# Flag policy (docs/cuda_latency_proposal.md):
#   default     — -ctk/-ctv q8_0, no -md  (v2026.9; T4 E2B draft is opt-in --e2b-draft)
#   --mtp       — omit q8 KV (f16 default). Quantized KV → ~0% MTP accept (llama.cpp #23398).
#   -fa         — off unless --flash-attn. b8782 E4B +56%; retest on b10883.
# GGML_CUDA_GRAPH_OPT is compiled in (USE_GRAPHS=1); no env-var needed.
ARGS=(
    -m "$TARGET"
    --host "$HOST"
    --port "$PORT"
    -ngl 999
    -c 512
)
if [ "$MTP" = true ]; then
    ARGS+=(--spec-type draft-mtp --spec-draft-n-max "$SPEC_N" -md "$ASSISTANT")
elif [ "$E2B_DRAFT" = true ]; then
    ARGS+=(-ctk q8_0 -ctv q8_0 -md "$DRAFT" --draft 16 --draft-min 5)
else
    ARGS+=(-ctk q8_0 -ctv q8_0)
fi
if [ "$FLASH_ATTN" = on ]; then
    ARGS+=(-fa on)
fi

"$SERVER" "${ARGS[@]}" &
SERVER_PID=$!
trap 'kill -TERM $SERVER_PID 2>/dev/null; wait $SERVER_PID 2>/dev/null; exit 0' INT TERM

# Wait up to 120s for /health to return ok (E4B cold-load takes ~30-90s on 16GB).
for _ in $(seq 1 120); do
    sleep 1
    if curl -s --max-time 1 "http://$HOST:$PORT/health" 2>/dev/null | grep -q '"status":"ok"'; then
        echo "READY"
        wait $SERVER_PID
        exit $?
    fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "ERROR: llama-server died before reaching ready state" >&2
        exit 3
    fi
done

echo "ERROR: llama-server did not become healthy within 120s" >&2
kill -TERM $SERVER_PID 2>/dev/null || true
exit 4
