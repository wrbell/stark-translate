# Docker Containerization Plan

Single-command deployment: `docker compose up` brings up live bilingual church translation on any NVIDIA GPU.

---

## Architecture

Two containers, one network:

| Container | Purpose | Base Image |
|-----------|---------|------------|
| **stark-inference** | Runtime inference (Whisper + TranslateGemma + MarianMT + Piper TTS) | `nvidia/cuda:12.4.1-runtime-ubuntu22.04` |
| **stark-training** | Optional training environment (LoRA fine-tuning, Deepgram oracle, alignment) | `nvidia/cuda:12.4.1-devel-ubuntu22.04` |

The training container is gated behind a Docker Compose profile and only starts when explicitly requested.

---

## Inference Container (stark-inference)

### Base Image

`nvidia/cuda:12.4.1-runtime-ubuntu22.04` — minimal CUDA runtime, no compiler toolchain.

### Dependencies

- **faster-whisper** — CTranslate2-backed Whisper STT (replaces mlx-whisper for CUDA)
- **transformers + bitsandbytes** — TranslateGemma 4B/12B INT4 translation (vllm as optional upgrade)
- **sentencepiece** — MarianMT tokenization
- **piper-tts** — Piper neural TTS for ES/EN audio output
- **websockets** — live text streaming to browser displays
- **sounddevice** — audio capture (when PulseAudio passthrough is available)
- **uvicorn + starlette** — HTTP server for display pages

### Volumes

| Mount | Container Path | Purpose |
|-------|---------------|---------|
| `./models` | `/app/models` | Quantized model weights (GPTQ/AWQ), cached after first download |
| `./config` | `/app/config` | `settings.py` overrides, `.env` |

### Ports

| Port | Protocol | Purpose |
|------|----------|---------|
| 8765 | WebSocket | Live text stream to browser displays |
| 8080 | HTTP | Display pages, LAN/phone access |
| 8766 | WebSocket | TTS audio stream |

### Entrypoint Sequence

1. **Model check** — verify model weights exist in `/app/models`, download from HuggingFace Hub if missing
2. **Health check** — run 5 canary sentences through the full pipeline (STT + translate + TTS) to confirm GPU availability and model loading
3. **Launch** — start `dry_run_ab.py` with the configured backend and hardware profile

---

## Training Container (stark-training)

### Base Image

`nvidia/cuda:12.4.1-devel-ubuntu22.04` — includes `nvcc` and CUDA headers for compiling Flash Attention, bitsandbytes, and custom kernels.

### Dependencies

Full `requirements-windows.txt` plus:

- **deepgram-sdk** — oracle transcription with 50 theological keyterms
- **peft** — LoRA/QLoRA adapter training
- **trl** — SFTTrainer for translation fine-tuning
- **datasets** — HuggingFace dataset loading
- **wandb** — experiment tracking (optional)
- **jiwer** — WER computation for eval

### Volumes

| Mount | Container Path | Purpose |
|-------|---------------|---------|
| `./stt-data` | `/app/stt-data` | Audio files sorted by `type/year/` |
| `./bible_data` | `/app/bible_data` | Verse pairs, synthetic sermon pairs, glossary |
| `./whisper_ablation` | `/app/whisper_ablation` | Whisper training outputs and checkpoints |
| `./hybrid_runs` | `/app/hybrid_runs` | TranslateGemma sweep results |
| `./models` | `/app/models` | Shared model cache with inference container |

---

## Hardware Profiles

| Profile | GPU Examples | VRAM | Models Loaded | Compose Flag |
|---------|-------------|------|---------------|-------------|
| `full` | A2000 Ada / RTX 3060 12GB / RTX 4070 | 12+ GB | Whisper large-v3-turbo + TranslateGemma 12B INT4 + MarianMT + Piper | default |
| `standard` | RTX 3060 8GB / RTX 3070 | 8+ GB | Whisper large-v3-turbo + TranslateGemma 4B INT4 + MarianMT + Piper | `--profile standard` |
| `low-vram` | GTX 1660 6GB / GTX 1070 | 6 GB | Whisper large-v3-turbo + MarianMT only (no Gemma, no TTS) | `--profile low-vram` |

The hardware profile is selected via environment variable `STARK_PROFILE` and controls which models are loaded at startup. The `full` profile uses TranslateGemma 12B as the final translation engine with 4B as a speculative draft. The `standard` profile uses 4B only. The `low-vram` profile falls back to MarianMT for both partials and finals.

---

## docker-compose.yml Structure

```yaml
version: "3.9"

services:
  inference:
    build:
      context: .
      dockerfile: Dockerfile.nvidia
    runtime: nvidia
    container_name: stark-inference
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - STARK_BACKEND=cuda
      - STARK_PROFILE=${STARK_PROFILE:-full}
    env_file:
      - ./docker/.env.docker
    ports:
      - "8765:8765"   # WebSocket live text
      - "8080:8080"   # HTTP display pages
      - "8766:8766"   # TTS audio WebSocket
    volumes:
      - ./models:/app/models
      - ./config:/app/config
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    healthcheck:
      test: ["CMD", "python", "-c", "import torch; assert torch.cuda.is_available()"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: unless-stopped

  training:
    build:
      context: .
      dockerfile: Dockerfile.training
    runtime: nvidia
    container_name: stark-training
    profiles: ["training"]
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
    env_file:
      - ./docker/.env.docker
    volumes:
      - ./stt-data:/app/stt-data
      - ./bible_data:/app/bible_data
      - ./whisper_ablation:/app/whisper_ablation
      - ./hybrid_runs:/app/hybrid_runs
      - ./models:/app/models
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    stdin_open: true
    tty: true
```

---

## Dockerfile.nvidia (Inference)

```dockerfile
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3-pip \
    libsndfile1 ffmpeg pulseaudio-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
COPY requirements-windows.txt .

RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir faster-whisper piper-tts bitsandbytes

COPY . /app
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8765 8080 8766

ENTRYPOINT ["/entrypoint.sh"]
CMD ["inference"]
```

---

## Dockerfile.training (Training)

```dockerfile
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3-pip \
    libsndfile1 ffmpeg git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-windows.txt .

RUN pip install --no-cache-dir -r requirements-windows.txt \
    && pip install --no-cache-dir deepgram-sdk peft trl wandb jiwer

COPY . /app

ENTRYPOINT ["bash"]
```

---

## docker/entrypoint.sh

```bash
#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="/app/models"
PROFILE="${STARK_PROFILE:-full}"

echo "=== Stark Translate — Docker Entrypoint ==="
echo "Profile: $PROFILE"
echo "GPU:     $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'not detected')"

# --- Step 1: Model download ---
download_if_missing() {
    local model_id="$1"
    local target_dir="$2"
    if [ ! -d "$target_dir" ] || [ -z "$(ls -A "$target_dir" 2>/dev/null)" ]; then
        echo "Downloading $model_id ..."
        python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('$model_id', local_dir='$target_dir')
"
        echo "Downloaded $model_id -> $target_dir"
    else
        echo "Found $target_dir (cached)"
    fi
}

# Always needed
download_if_missing "openai/whisper-large-v3-turbo" "$MODEL_DIR/whisper-large-v3-turbo"
download_if_missing "Helsinki-NLP/opus-mt-en-es"     "$MODEL_DIR/opus-mt-en-es"
download_if_missing "Helsinki-NLP/opus-mt-es-en"     "$MODEL_DIR/opus-mt-es-en"

if [ "$PROFILE" = "full" ] || [ "$PROFILE" = "standard" ]; then
    download_if_missing "google/translate-gemma-4b-gptq" "$MODEL_DIR/translategemma-4b-gptq"
fi

if [ "$PROFILE" = "full" ]; then
    download_if_missing "google/translate-gemma-12b-gptq" "$MODEL_DIR/translategemma-12b-gptq"
fi

if [ "$PROFILE" != "low-vram" ]; then
    download_if_missing "rhasspy/piper-voices" "$MODEL_DIR/piper-voices"
fi

# --- Step 2: Health check (5 canary sentences) ---
echo ""
echo "=== Running health check ==="
python3 -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available'
print(f'CUDA OK: {torch.cuda.get_device_name(0)}')
print(f'VRAM:    {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
"

python3 -c "
canary = [
    ('en', 'The grace of the Lord Jesus Christ be with you all.'),
    ('es', 'La gracia del Senor Jesucristo sea con todos ustedes.'),
    ('en', 'For God so loved the world that he gave his only begotten Son.'),
    ('es', 'Porque de tal manera amo Dios al mundo que dio a su Hijo unigenito.'),
    ('en', 'The Lord is my shepherd; I shall not want.'),
]
print(f'Canary sentences loaded: {len(canary)} OK')
# Full pipeline health check runs after model loading in the main process
"

echo "Health check passed."

# --- Step 3: Launch ---
echo ""
echo "=== Starting Stark Translate ($PROFILE profile) ==="

if [ "${1:-}" = "inference" ]; then
    exec python3 dry_run_ab.py --backend cuda --profile "$PROFILE"
else
    exec "$@"
fi
```

---

## Environment Variables

Stored in `docker/.env.docker`:

```bash
# Backend
STARK_BACKEND=cuda
STARK_PROFILE=full

# STT
STARK_STT__WHISPER_MODEL=openai/whisper-large-v3-turbo
STARK_STT__COMPUTE_TYPE=float16
STARK_STT__BEAM_SIZE=5

# Translation
STARK_TRANSLATE__CUDA_MODEL_4B=models/translategemma-4b-gptq
STARK_TRANSLATE__CUDA_MODEL_12B=models/translategemma-12b-gptq
STARK_TRANSLATE__MARIAN_EN_ES=models/opus-mt-en-es
STARK_TRANSLATE__MARIAN_ES_EN=models/opus-mt-es-en

# TTS
STARK_TTS__ENABLED=true
STARK_TTS__VOICE_ES=es_MX-claude-high
STARK_TTS__VOICE_EN=en_US-amy-medium

# Display
STARK_DISPLAY__WS_PORT=8765
STARK_DISPLAY__HTTP_PORT=8080
STARK_DISPLAY__TTS_PORT=8766

# Optional API keys (for training container)
STARK_DEEPGRAM__API_KEY=
STARK_DEEPL__API_KEY=

# Logging
STARK_LOG_LEVEL=INFO
```

---

## Model Download Strategy

| Model | Approx Size | Profile |
|-------|-------------|---------|
| Whisper large-v3-turbo (CTranslate2) | ~1.5 GB | all |
| MarianMT opus-mt-en-es + es-en | ~600 MB | all |
| TranslateGemma 4B GPTQ | ~2.5 GB | `full`, `standard` |
| TranslateGemma 12B GPTQ | ~7 GB | `full` only |
| Piper TTS voices (ES + EN) | ~100 MB | `full`, `standard` |
| **Total (full profile)** | **~11.7 GB** | |

Downloads happen on first `docker compose up`. Subsequent runs use the cached `models/` volume. To force re-download, delete the specific model subdirectory.

---

## Audio Input

Docker containers have no direct microphone access. Three supported strategies, in order of preference:

### Option 1: PulseAudio Socket Passthrough (recommended for Linux hosts)

```yaml
# Add to inference service in docker-compose.yml
volumes:
  - /run/user/1000/pulse:/run/user/1000/pulse
  - ~/.config/pulse/cookie:/root/.config/pulse/cookie
environment:
  - PULSE_SERVER=unix:/run/user/1000/pulse/native
```

This gives the container direct access to the host PulseAudio server and all audio devices.

### Option 2: Host Audio Capture via WebSocket

Run a lightweight audio capture script on the host that streams PCM audio to the container:

```bash
# On host (outside Docker)
python3 tools/audio_bridge.py --ws-url ws://localhost:8765/audio
```

The container receives audio over WebSocket and processes it identically to local mic input. This is the most portable option and works on Windows/macOS hosts.

### Option 3: File/Stream URL Input

Point the container at a pre-recorded file or a live stream URL:

```bash
docker compose run inference python3 dry_run_ab.py --input rtsp://camera:8554/live
docker compose run inference python3 dry_run_ab.py --input /app/stt-data/test.wav
```

Useful for testing, batch processing, and CI pipelines.

---

## Quick Start

```bash
# Clone and run (full profile, default)
git clone https://github.com/wrbell/stark-translate.git
cd stark-translate
docker compose up

# Open browser for live display
# http://localhost:8080

# Standard profile (8GB VRAM)
STARK_PROFILE=standard docker compose up

# Low-VRAM profile (6GB)
STARK_PROFILE=low-vram docker compose up

# Start training container alongside inference
docker compose --profile training up

# Training shell (interactive)
docker compose --profile training run training

# Run a specific training script
docker compose --profile training run training \
    python3 training/train_whisper.py --config whisper_ablation/W0_baseline.yaml
```

---

## Files to Create

| File | Purpose |
|------|---------|
| `Dockerfile.nvidia` | Inference container (CUDA runtime, faster-whisper, TranslateGemma, Piper) |
| `Dockerfile.training` | Training container (CUDA devel, full training deps, deepgram-sdk) |
| `docker-compose.yml` | Service orchestration with hardware profiles |
| `docker/entrypoint.sh` | Model download, health check, and launch sequence |
| `docker/.env.docker` | Runtime environment variable template |
| `docker/.dockerignore` | Exclude `stt-data/`, large files, and venvs from build context |

---

## .dockerignore

```
stt-data/
stark_data/
whisper_ablation/
hybrid_runs/
ablation/
models/
fine_tuned_*
*.wav
*.mp3
*.flac
*.pt
*.bin
*.safetensors
__pycache__/
.git/
.venv/
*.egg-info/
```

---

## Networking

Both containers share a default Docker Compose network. The training container can reach the inference container at `stark-inference:8765` for live evaluation during training runs.

External access from the LAN (for phone/tablet displays in the church):

```bash
# Find host IP
ip addr show | grep "inet " | grep -v 127.0.0.1

# Phones connect to http://<host-ip>:8080
# WebSocket at ws://<host-ip>:8765
```

No additional firewall rules needed beyond the published ports.

---

## Production Considerations

- **Restart policy**: `unless-stopped` on the inference container ensures it survives host reboots.
- **GPU memory limits**: Docker does not enforce VRAM limits natively. The hardware profile controls model selection to stay within budget.
- **Log rotation**: Add `logging.driver: json-file` with `max-size: 50m` and `max-file: 3` to prevent log bloat during long services.
- **Secrets**: API keys (`DEEPGRAM`, `DEEPL`) should use Docker secrets or a `.env` file excluded from version control. The template `docker/.env.docker` ships with empty values.
- **Updates**: Pull new images with `docker compose build --no-cache && docker compose up -d`. Model weights persist in the `models/` volume.
