#!/usr/bin/env bash
# Entrypoint for the stark-translate container. Dispatches based on the
# command (CMD or `docker compose run … <command>`):
#
#   operator       → uvicorn operator_app.main:app --host 0.0.0.0 --port 9000
#   llama-server   → ./start_server.sh (auto-detects E4B + E2B GGUFs in $MODEL_DIR)
#   audio-bridge   → tools/audio_bridge.py (Phase 9.4 fallback; not yet built)
#   bash           → drop into a shell for debugging
#
# All commands run as PID 1 inside the container; SIGTERM from `docker stop`
# propagates to the right process.

set -euo pipefail

cmd="${1:-operator}"
shift || true

case "$cmd" in
    operator)
        echo "[entrypoint] launching operator (uvicorn) on 0.0.0.0:9000"
        # Pre-flight: check that the GGUFs exist when the operator path expects them.
        if [ -n "${STARK_CUDA__LLAMACPP_URL:-}" ] && [ ! -d "${MODEL_DIR:-/app/models}" ]; then
            echo "[entrypoint] WARN: \$MODEL_DIR ($MODEL_DIR) not mounted; preflight will be yellow."
        fi
        exec /opt/venv/bin/uvicorn operator_app.main:app \
             --host 0.0.0.0 --port 9000 \
             --log-level "${STARK_LOG_LEVEL:-info}"
        ;;

    llama-server)
        : "${LLAMA_PORT:=8090}"
        : "${MODEL_DIR:=/app/models}"
        echo "[entrypoint] launching llama-server on 0.0.0.0:${LLAMA_PORT} (models: ${MODEL_DIR})"
        cd /app
        # start_server.sh already handles --no-draft / --port / --model
        if [ -f "${MODEL_DIR}/gemma-4-e4b-it-q4km.gguf" ]; then
            exec ./start_server.sh --port "${LLAMA_PORT}"
        elif [ -f "${MODEL_DIR}/gemma-4-e2b-it-q4km.gguf" ]; then
            exec ./start_server.sh --port "${LLAMA_PORT}" --no-draft \
                 --model "${MODEL_DIR}/gemma-4-e2b-it-q4km.gguf"
        else
            echo "[entrypoint] FATAL: no GGUFs in ${MODEL_DIR}. Mount STARK_MODELS_DIR." >&2
            echo "[entrypoint] Expected gemma-4-e4b-it-q4km.gguf or gemma-4-e2b-it-q4km.gguf." >&2
            exit 2
        fi
        ;;

    audio-bridge)
        echo "[entrypoint] audio-bridge profile is a Phase 9.4.2 stub; nothing to run yet."
        # Stay alive so docker compose treats this as healthy until 9.4.2 lands.
        exec sleep infinity
        ;;

    bash|sh)
        exec bash "$@"
        ;;

    *)
        echo "[entrypoint] unknown command: $cmd"
        echo "[entrypoint] valid: operator | llama-server | audio-bridge | bash"
        exit 64
        ;;
esac
