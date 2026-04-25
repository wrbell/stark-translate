# syntax=docker/dockerfile:1.7
# stark-translate Linux/CUDA image for v2026.7+.
# Multi-stage build: builder compiles llama.cpp; runtime carries only the binary
# + python venv. Cuts the final image from ~9 GB to ~3 GB (validated against
# Markaicode's 2025 LLM Docker article and llama.cpp's official server variant).

# ---------------------------------------------------------------------------
# Stage 1: build llama-server + Python wheels
# ---------------------------------------------------------------------------
ARG CUDA_VERSION=12.6.3
ARG UBUNTU_VERSION=24.04
ARG CUDA_ARCHS=89

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS builder

ARG CUDA_ARCHS
# Pin llama.cpp to a known-working release tag; bump intentionally with a
# Dockerfile change so we can reason about behavior across releases.
# (b8782 was skipped in llama.cpp's tag sequence — sequence jumps b8781→b8783.)
ARG LLAMA_CPP_REF=b8783

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    ca-certificates \
    python3.12 \
    python3.12-venv \
    python3-pip \
 && rm -rf /var/lib/apt/lists/*

# --- llama.cpp build ---------------------------------------------------------
# The GHCR builder has no GPU, so we link against the CUDA driver stubs that
# ship inside the nvidia/cuda:devel image. Without this, ld(1) bails on
# `cuMemMap`/`cuGetErrorString`/etc. The stub is replaced by the real
# /usr/lib/x86_64-linux-gnu/libcuda.so.1 at container runtime via the NVIDIA
# Container Toolkit's library injection.
WORKDIR /opt
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs:${LD_LIBRARY_PATH}
RUN ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 \
 && git clone https://github.com/ggml-org/llama.cpp.git \
 && cd llama.cpp \
 && git checkout ${LLAMA_CPP_REF} \
 && cmake -B build \
        -DGGML_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHS} \
        -DCMAKE_BUILD_TYPE=Release \
        -DLLAMA_BUILD_SERVER=ON \
        -DCMAKE_EXE_LINKER_FLAGS="-L/usr/local/cuda/lib64/stubs" \
        -DCMAKE_SHARED_LINKER_FLAGS="-L/usr/local/cuda/lib64/stubs" \
 && cmake --build build --config Release -j"$(nproc)" --target llama-server

# --- Python venv with stark-translate[cuda] ----------------------------------
COPY pyproject.toml README.md /tmp/build/
COPY operator_app /tmp/build/operator_app
COPY engines /tmp/build/engines
COPY tools /tmp/build/tools
COPY features /tmp/build/features
COPY displays /tmp/build/displays
COPY scripts /tmp/build/scripts
COPY settings.py dry_run_ab.py models.lock.json start_server.sh run_operator.sh /tmp/build/

WORKDIR /tmp/build
RUN python3.12 -m venv /opt/venv \
 && /opt/venv/bin/pip install --no-cache-dir --upgrade pip wheel \
 && /opt/venv/bin/pip install --no-cache-dir '.[cuda]'

# ---------------------------------------------------------------------------
# Stage 2: slim runtime
# ---------------------------------------------------------------------------
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION} AS runtime

ARG CUDA_VERSION
ARG UBUNTU_VERSION

LABEL org.opencontainers.image.source="https://github.com/wrbell/stark-translate" \
      org.opencontainers.image.description="Live bilingual STT + translation operator stack (CUDA path)" \
      org.opencontainers.image.licenses="Proprietary" \
      org.opencontainers.image.vendor="Stark Road Gospel Hall"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PATH=/opt/venv/bin:$PATH \
    STARK_PROJECT_ROOT=/app \
    STARK_OPERATOR_LOG_DIR=/app/metrics

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    ffmpeg \
    libsndfile1 \
    libportaudio2 \
    pulseaudio-utils \
    curl \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/llama.cpp/build/bin/llama-server /usr/local/bin/llama-server

WORKDIR /app
COPY operator_app /app/operator_app
COPY engines /app/engines
COPY tools /app/tools
COPY features /app/features
COPY displays /app/displays
COPY scripts /app/scripts
COPY docker/entrypoint.sh /app/docker/entrypoint.sh
COPY settings.py dry_run_ab.py models.lock.json start_server.sh run_operator.sh /app/

RUN mkdir -p /app/metrics /app/models /app/stark_data \
 && chmod +x /app/docker/entrypoint.sh /app/run_operator.sh /app/start_server.sh \
              /app/scripts/dry_run_rehearsal.sh

# llama-server is a thin native binary; expose its config knobs as env vars
ENV LLAMA_DIR=/usr/local \
    SERVER=/usr/local/bin/llama-server \
    MODEL_DIR=/app/models

EXPOSE 9000 8080 8765 8766 8090

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl --fail --silent http://localhost:9000/healthz || exit 1

ENTRYPOINT ["/app/docker/entrypoint.sh"]
CMD ["operator"]
