# Linux/CUDA Docker — v2026.7+

Single-command deployment for the Linux/NVIDIA target audience: church PCs with
an Ada/Ampere GPU and a fresh Ubuntu install.

```bash
# Set where models live on the host (anywhere with ~10 GB free)
export STARK_MODELS_DIR="$HOME/.cache/stark-translate/models"
export STARK_DATA_DIR="$HOME/.local/share/stark-translate/sessions"

# Pull the pre-built image (Track 1 publishes to GHCR on every v* tag)
docker compose pull

# Bring up operator + llama-server. First boot loads E4B which takes ~90 s.
docker compose up
```

Open <http://localhost:9000/operator/> and the operator UI loads.

---

## What ships

| Service | Image | Purpose |
|---|---|---|
| `operator` | `ghcr.io/wrbell/stark-translate:${TAG:-latest}` | FastAPI control plane on `:9000` + audience HTTP on `:8080` + transcript WS on `:8765` + TTS WS on `:8766` |
| `llama-server` | same image, different `command` | llama.cpp HTTP API on `:8090` (auto-picks E4B if present, falls back to E2B with `--no-draft`) |
| `audio-bridge` | same image, profile-gated | Phase 9.4.2 stub. Run with `docker compose --profile audio-bridge up` when host PulseAudio passthrough doesn't work. |

The image is built from a multi-stage `Dockerfile` against
`nvidia/cuda:12.6.3-devel-ubuntu24.04` (builder) and
`nvidia/cuda:12.6.3-runtime-ubuntu24.04` (runtime). Final size is ~3 GB —
llama.cpp is built once in the builder and only the binary copies through.

GGUFs are **bind-mounted, never baked** — same pattern OpenWebUI, LocalAI,
LM Studio, vLLM all use. Run `stark-translate setup` (Track 4 PyPI install) or
manually copy the GGUFs into `$STARK_MODELS_DIR` first; see
[`models.md`](./models.md) for the lockfile format.

---

## NVIDIA Container Toolkit setup

The `docker-compose.yml` declares CDI mode (`nvidia.com/gpu=all`), which is
the default since toolkit v1.18.0 (mid-2025).

```bash
# One-time host setup
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml

# Verify
docker run --rm --device nvidia.com/gpu=all \
    nvidia/cuda:12.6.3-base-ubuntu24.04 nvidia-smi
```

For older toolkits (≤1.17), replace the `devices:` block in `docker-compose.yml`
with:

```yaml
runtime: nvidia
environment:
  NVIDIA_VISIBLE_DEVICES: all
```

---

## Audio I/O — host mic capture

Two paths, in order of preference:

1. **PulseAudio socket bind-mount** (works on most Linux desktops). The compose
   file already mounts `/run/user/1000/pulse` and the cookie. If your UID is
   not 1000, edit both lines.
2. **`audio-bridge` profile** (host has no Pulse, e.g. Wayland-only or remote
   head). `docker compose --profile audio-bridge up` runs a sidecar that taps
   `/dev/snd` directly and forwards over the operator's `/audio` WebSocket.

If neither works, install via the PyPI/uv path on the same host instead — the
operator runs natively without the container indirection.

---

## Observability

`/metrics` on port `:9000` exposes Prometheus exposition format:

```text
# HELP stark_uptime_seconds Operator process uptime in seconds.
# TYPE stark_uptime_seconds gauge
stark_uptime_seconds 142.3
# HELP stark_vram_mib Current GPU 0 VRAM usage in MiB (nvidia-smi).
# TYPE stark_vram_mib gauge
stark_vram_mib 4912
…
```

Wire it into Prometheus by scraping `http://operator:9000/metrics`. The same
data is also available as JSON at `/api/metrics` and pushed at ~1 Hz over
`/ws/control`.

---

## Building locally

For dev-loop changes that don't go through CI:

```bash
docker buildx build \
    --platform linux/amd64 \
    --build-arg CUDA_ARCHS=89 \
    -t stark-translate:dev .
```

`CUDA_ARCHS` defaults to `89` (Ada — RTX 40 series, A2000 Ada). Use
`"75;80;86;89;90"` for a portable image at the cost of a ~30% longer compile.

---

## CI / GHCR

`.github/workflows/docker.yml` builds and pushes on every `v*` tag and on
pushes to `main`. Tag scheme:

| Trigger | Tags |
|---|---|
| `git push origin main` | `:main`, `:latest` |
| `git tag v2026.7.1.0 && git push --tags` | `:v2026.7.1.0`, `:2026.7.1.0`, `:2026.7` |
| `workflow_dispatch` | configurable via the `tag` input |

The workflow uses `pypa`-style `id-token: write` permissions so future cosign
keyless signing can attach without a per-tag KMS key.

---

## Troubleshooting

**`/healthz` returns 503 with `llama-server` red.** Check
`docker compose logs llama-server`. The most common cause is GGUFs missing
from `$STARK_MODELS_DIR` — the entrypoint will exit code 2 with a message
listing the expected filenames.

**`pulseaudio: connection refused`.** Either your UID isn't 1000 or the host
isn't running Pulse. Switch to the `audio-bridge` profile or set
`STARK_NO_PULSE=1` and use the bridge.

**Cold-start health check times out.** First load of E4B is ~90 s on a
3060/A2000 Ada. The compose `start_period: 120s` already accounts for this
but if you're on a slower disk, bump it.

**`nvidia-smi` works on host but `docker run --device nvidia.com/gpu=all` fails.**
Regenerate CDI: `sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml`.
The CDI file is regenerated per kernel/driver bump, not automatically.

---

## See also

- [`pypi.md`](./pypi.md) — `uv tool install` path for developers and Mac users
- [`models.md`](./models.md) — `models.lock.json` format and bootstrap flow
- [`windows.md`](./windows.md) — Windows MSI path (planned for v2026.7.2)
- [`../operator_runbook.md`](../operator_runbook.md) — operator UI walkthrough
