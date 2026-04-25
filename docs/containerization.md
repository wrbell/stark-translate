# Containerization & Distribution

> **v2026.7 rewrite.** This document used to describe a single-target NVIDIA
> Docker layout against TranslateGemma 4B/12B GPTQ. That setup was retired
> in v2026.5 when we moved to Gemma 4 GGUF + llama.cpp, and again in v2026.6
> when the operator FastAPI app on `:9000` became the entry point. The full
> v2026.7 plan covers four channels: Linux Docker, Windows MSI, macOS via
> `uv tool install`, and PyPI for developers.

---

## Channels

| Audience | Channel | Doc |
|---|---|---|
| Linux/NVIDIA operator (church PC, dedicated server) | `docker compose up` against the GHCR image | [`packaging/linux-docker.md`](./packaging/linux-docker.md) |
| Developers + Mac volunteers | `uv tool install 'stark-translate[cuda\|mlx\|cpu]'` | [`packaging/pypi.md`](./packaging/pypi.md) |
| Windows operator (non-technical) | unsigned `.msi` from GitHub Releases (PyApp under the hood); planned for v2026.7.2 | [`packaging/windows.md`](./packaging/windows.md) |
| Mac native `.app` | deferred to v2026.8 (Apple Developer enrollment is the gate) | [`packaging/macos.md`](./packaging/macos.md) |

Every channel installs from the same source tree and the same
`pyproject.toml`; the only differences are extras and per-channel build
config (Dockerfile, PyApp config, Briefcase config).

---

## Quick reference — Linux/CUDA Docker

```bash
export STARK_MODELS_DIR="$HOME/.cache/stark-translate/models"
docker compose pull
docker compose up
open http://localhost:9000/operator/
```

Three services on one network:

- `operator` — FastAPI control plane on `:9000`, audience HTTP on `:8080`,
  transcript WS on `:8765`, TTS WS on `:8766`
- `llama-server` — llama.cpp HTTP API on `:8090` (auto-picks E4B if present,
  falls back to E2B with `--no-draft`)
- `audio-bridge` — profile-gated stub for hosts without PulseAudio passthrough

GGUFs are **bind-mounted from the host**, not baked into the image. This is
the same pattern OpenWebUI / LocalAI / LM Studio / vLLM all use.

See [`packaging/linux-docker.md`](./packaging/linux-docker.md) for the full
walkthrough including NVIDIA Container Toolkit setup, audio passthrough, and
the `/metrics` Prometheus endpoint.

---

## Models — bootstrap and lockfile

The model manifest lives at `models.lock.json` at the repo root. It pins
URLs, sizes, and SHA-256 for every artifact the operator needs (GGUFs,
Whisper Turbo, MarianMT, Piper voices). Both the PyPI/uv path and the Docker
path read it via `operator_app.setup`:

```bash
stark-translate setup                # download to default cache dir
stark-translate setup --refresh      # re-download even if .installed sidecar matches
```

Once Track 1 ships, the Linux Docker path delegates to the same module —
preflight blocks Start when models are missing, instead of the old "implicit
download on first inference" anti-pattern. See [`packaging/models.md`](./packaging/models.md).

---

## Tracks (release sequencing)

| Track | Status | Tag |
|---|---|---|
| Track 4 — PyPI wheel + uv | shipped | `v2026.7.0.0` |
| Track 1 — Linux Docker / GHCR | this PR | `v2026.7.1.0` |
| Track 3 — Windows MSI (unsigned) | next | `v2026.7.2.0` |
| Track 3 follow-up — sign the MSI | deferred (cert procurement) | `v2026.7.2.1` |
| Track 2 — Mac native `.app` | deferred (Apple Developer enrollment) | `v2026.8.0.0` |

---

## Migration from the old document

If you arrived here from a stale link to the GPTQ-era setup:

- The "stark-inference" container is now a single multi-stage build with
  `nvidia/cuda:12.6.3-{devel,runtime}-ubuntu24.04` as base layers.
- The "stark-training" container has not been re-shipped — training stays on
  the WSL2 desktop per `CLAUDE-windows.md`. There's no v2026.7 training image.
- Streamlit is gone; the operator UI is plain HTML/JS served by FastAPI on
  `:9000` and audience displays are the static pages on `:8080`.
- Models are bind-mounted, not baked, so a 4 GB image hosts 10+ GB of model
  weights without being fat.
