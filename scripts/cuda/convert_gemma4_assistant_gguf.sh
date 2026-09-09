#!/usr/bin/env bash
# Download Google Gemma 4 MTP assistants and convert to GGUF (f16 + Q4_0).
#
# NOT YET EXECUTED ON HARDWARE. Proposal-only until the next WSL session.
# See docs/cuda_latency_proposal.md.
#
# Recipe from llama.cpp PR #24282:
#   convert_hf_to_gguf.py <hf-dir> --outfile <f16.gguf> --outtype f16
#   llama-quantize <f16.gguf> <q4_0.gguf> Q4_0
#
# Usage:
#   scripts/cuda/convert_gemma4_assistant_gguf.sh
#   SIZE=E4B scripts/cuda/convert_gemma4_assistant_gguf.sh
#   scripts/cuda/convert_gemma4_assistant_gguf.sh --help

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
LLAMA_DIR="${LLAMA_DIR:-$HOME/llama.cpp}"
MODEL_DIR="${MODEL_DIR:-$ROOT/models}"
HF_CACHE="${HF_CACHE:-$MODEL_DIR/hf}"
# Comma-separated: E4B, E2B (Google repo ids use capital E).
SIZES="${SIZES:-E4B,E2B}"
PYTHON="${PYTHON:-python3}"

usage() {
    cat <<EOF
Convert google/gemma-4-{E4B,E2B}-it-assistant to GGUF f16 + Q4_0.

NOT YET EXECUTED ON HARDWARE.

Env / defaults:
  LLAMA_DIR    llama.cpp checkout (needs convert_hf_to_gguf.py + llama-quantize)
  MODEL_DIR    output dir for GGUFs                    [${MODEL_DIR}]
  HF_CACHE     HF snapshot dir                         [${HF_CACHE}]
  SIZES        comma list E4B and/or E2B               [${SIZES}]
  PYTHON       interpreter with huggingface_hub        [${PYTHON}]

Outputs (lowercase filenames to match start_server.sh / CONFIGS):
  models/gemma-4-e4b-it-assistant-f16.gguf
  models/gemma-4-e4b-it-assistant-q4_0.gguf
  models/gemma-4-e2b-it-assistant-f16.gguf
  models/gemma-4-e2b-it-assistant-q4_0.gguf
EOF
}

for arg in "$@"; do
    case "$arg" in
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown arg: $arg (try --help)" >&2; exit 1 ;;
    esac
done

CONVERT="${LLAMA_DIR}/convert_hf_to_gguf.py"
QUANTIZE="${LLAMA_DIR}/build/bin/llama-quantize"
if [ ! -f "${CONVERT}" ]; then
    echo "ERROR: missing ${CONVERT} — run scripts/cuda/build_llamacpp.sh first" >&2
    exit 1
fi
if [ ! -x "${QUANTIZE}" ]; then
    echo "ERROR: missing ${QUANTIZE} — run scripts/cuda/build_llamacpp.sh first" >&2
    exit 1
fi

mkdir -p "${MODEL_DIR}" "${HF_CACHE}"

download_hf() {
    local repo="$1"
    local dest="$2"
    if [ -d "${dest}" ] && [ -f "${dest}/config.json" ]; then
        echo "==> reuse HF snapshot ${dest}"
        return 0
    fi
    echo "==> download ${repo} -> ${dest}"
    if command -v huggingface-cli >/dev/null 2>&1; then
        huggingface-cli download "${repo}" --local-dir "${dest}"
    else
        "${PYTHON}" - <<PY
from huggingface_hub import snapshot_download
snapshot_download(repo_id="${repo}", local_dir="${dest}")
PY
    fi
}

IFS=',' read -r -a size_arr <<< "${SIZES}"
for SIZE in "${size_arr[@]}"; do
    SIZE="$(echo "${SIZE}" | tr -d '[:space:]')"
    [ -n "${SIZE}" ] || continue
    SIZE_LC="$(echo "${SIZE}" | tr '[:upper:]' '[:lower:]')"
    REPO="google/gemma-4-${SIZE}-it-assistant"
    SRC="${HF_CACHE}/gemma-4-${SIZE}-it-assistant"
    F16="${MODEL_DIR}/gemma-4-${SIZE_LC}-it-assistant-f16.gguf"
    Q40="${MODEL_DIR}/gemma-4-${SIZE_LC}-it-assistant-q4_0.gguf"

    download_hf "${REPO}" "${SRC}"

    echo "==> convert ${REPO} -> ${F16}"
    "${PYTHON}" "${CONVERT}" "${SRC}" --outfile "${F16}" --outtype f16

    echo "==> quantize Q4_0 -> ${Q40}"
    "${QUANTIZE}" "${F16}" "${Q40}" Q4_0

    ls -lh "${F16}" "${Q40}"
done

echo "NOT YET EXECUTED ON HARDWARE — GGUFs are inputs to start_server.sh --mtp / bench_mtp.sh."
