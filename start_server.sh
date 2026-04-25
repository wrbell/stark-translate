#!/bin/bash
# Start llama-server for translation with speculative decoding.
#
# Usage:
#   ./start_server.sh                    # E4B target + E2B draft (best quality)
#   ./start_server.sh --no-draft         # E2B standalone (simpler, less VRAM)
#   ./start_server.sh --model <path>     # Custom model
#
# Prerequisites:
#   1. Build llama.cpp: cd ~/llama.cpp && cmake -B build -DGGML_CUDA=ON && cmake --build build -j
#   2. Export GGUF: python ~/llama.cpp/convert_hf_to_gguf.py ...
#   3. Quantize: ~/llama.cpp/build/bin/llama-quantize ... Q4_K_M

set -e

LLAMA_DIR="${LLAMA_DIR:-$HOME/llama.cpp}"
SERVER="${LLAMA_DIR}/build/bin/llama-server"
MODEL_DIR="${MODEL_DIR:-models}"

# Defaults: E4B target + E2B draft
TARGET="${MODEL_DIR}/gemma-4-e4b-it-q4km.gguf"
DRAFT="${MODEL_DIR}/gemma-4-e2b-it-q4km.gguf"
HOST="127.0.0.1"
PORT="8090"
NO_DRAFT=false

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --no-draft) NO_DRAFT=true; shift ;;
        --model) TARGET="$2"; shift 2 ;;
        --draft) DRAFT="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# Verify server binary
if [ ! -f "$SERVER" ]; then
    echo "ERROR: llama-server not found at $SERVER"
    echo "Build it: cd $LLAMA_DIR && cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 && cmake --build build --config Release -j\$(nproc)"
    exit 1
fi

# Verify model
if [ ! -f "$TARGET" ]; then
    echo "ERROR: Model not found at $TARGET"
    echo "Export GGUF first — see docs/april_squeeze/spec_decode_research.md"
    exit 1
fi

if [ "$NO_DRAFT" = false ] && [ -f "$DRAFT" ]; then
    echo "Starting llama-server: target + draft (spec decode)"
elif [ "$NO_DRAFT" = false ] && [ ! -f "$DRAFT" ]; then
    echo "WARNING: Draft model not found at $DRAFT — running without spec decode"
    echo "Starting llama-server: target only"
else
    echo "Starting llama-server: target only (--no-draft)"
fi

echo "  Target: $TARGET"
[ "$NO_DRAFT" = false ] && [ -f "$DRAFT" ] && echo "  Draft:  $DRAFT"
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
"$SERVER" -m "$TARGET" --host "$HOST" --port "$PORT" -ngl 999 -c 512 -ctk q8_0 \
    $([ "$NO_DRAFT" = false ] && [ -f "$DRAFT" ] && echo "-md $DRAFT --draft 16 --draft-min 5") &
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
kill -TERM $SERVER_PID 2>/dev/null
exit 4
