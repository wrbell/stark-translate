#!/usr/bin/env bash
# Build llama.cpp (CUDA, Ada sm_89) for the WSL A2000 box.
#
# NOT YET EXECUTED ON HARDWARE. Proposal-only until the next WSL session.
# See docs/cuda_latency_proposal.md.
#
# Usage:
#   scripts/cuda/build_llamacpp.sh
#   LLAMA_CPP_REF=b10883 LLAMA_DIR=$HOME/llama.cpp scripts/cuda/build_llamacpp.sh
#   scripts/cuda/build_llamacpp.sh --help

set -euo pipefail

LLAMA_CPP_REF="${LLAMA_CPP_REF:-b10883}"
LLAMA_DIR="${LLAMA_DIR:-$HOME/llama.cpp}"
LLAMA_REPO="${LLAMA_REPO:-https://github.com/ggml-org/llama.cpp.git}"
CMAKE_BUILD_DIR="${CMAKE_BUILD_DIR:-build}"
CMAKE_CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCHITECTURES:-89}"
JOBS="${JOBS:-$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)}"

usage() {
    cat <<EOF
Build llama-server, llama-quantize, and llama-bench with GGML_CUDA=ON.

NOT YET EXECUTED ON HARDWARE.

Env / defaults:
  LLAMA_CPP_REF              git tag/commit to checkout   [${LLAMA_CPP_REF}]
  LLAMA_DIR                  clone path                   [${LLAMA_DIR}]
  LLAMA_REPO                 remote URL                   [${LLAMA_REPO}]
  CMAKE_BUILD_DIR            cmake -B dir (relative)      [${CMAKE_BUILD_DIR}]
  CMAKE_CUDA_ARCHITECTURES   nvcc sm list (Ada = 89)      [${CMAKE_CUDA_ARCHITECTURES}]
  JOBS                       parallel compile             [${JOBS}]

Must stay in lockstep with Dockerfile ARG LLAMA_CPP_REF and start_server.sh pin.
EOF
}

for arg in "$@"; do
    case "$arg" in
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown arg: $arg (try --help)" >&2; exit 1 ;;
    esac
done

echo "==> llama.cpp pin ${LLAMA_CPP_REF}  CUDA arch ${CMAKE_CUDA_ARCHITECTURES}  dir ${LLAMA_DIR}"

if [ ! -d "${LLAMA_DIR}/.git" ]; then
    git clone "${LLAMA_REPO}" "${LLAMA_DIR}"
fi

git -C "${LLAMA_DIR}" fetch --tags origin
git -C "${LLAMA_DIR}" checkout "${LLAMA_CPP_REF}"

cmake -S "${LLAMA_DIR}" -B "${LLAMA_DIR}/${CMAKE_BUILD_DIR}" \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCHITECTURES}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLAMA_BUILD_SERVER=ON

cmake --build "${LLAMA_DIR}/${CMAKE_BUILD_DIR}" --config Release -j"${JOBS}" \
    --target llama-server llama-quantize llama-bench

BIN="${LLAMA_DIR}/${CMAKE_BUILD_DIR}/bin"
echo "==> built:"
ls -l "${BIN}/llama-server" "${BIN}/llama-quantize" "${BIN}/llama-bench"
"${BIN}/llama-server" --version || true
echo "NOT YET EXECUTED ON HARDWARE — verify sm_89 / b10883 on the WSL box before serving."
