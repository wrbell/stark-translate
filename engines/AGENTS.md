# engines/AGENTS.md — STT + Translation + TTS (Agent Guide)

> Paired with [`CLAUDE.md`](./CLAUDE.md). Read both; this file emphasizes agent
> constraints and navigation.

## Critical rules

- Call `.load()` before `.transcribe()` / `.translate()` — factory does not auto-load.
- **MLX ≥ 0.31.2:** max_workers=2 overlap OK; materialize weights + first Gemma forward on load thread (`warm_mlx_model`).
- **PyTorch lock:** Marian HF + Silero VAD share `_pytorch_lock`; VAD stays on asyncio thread.
- **Gemma stop tokens:** `ensure_stop_tokens()` after load — never replace Gemma 4 EOS with `{1,3}` only.
- **Do not recreate `stt_env`** on Mac; use setup resolver and cached models.

## Mac defaults (v2026.14 candidate)

| Role | Path |
|------|------|
| STT EN | `ParakeetMLXEngine` / `--stt-backend parakeet-mlx` |
| STT ES | mlx-whisper large-v3-turbo |
| Partial | Marian CT2 int8 CPU (`adapters/marian_ct2/` or managed cache) |
| Final | Gemma 4 E4B OptiQ; E2B via `--gemma4-size e2b` only after review |
| MTP | Off (#177 experimental) |

Env: `STARK_TRANSLATE_MARIAN_BACKEND` (single underscore) for Marian backend override.

## CUDA / llama.cpp

Production finals via `LlamaCppEngine` + `start_server.sh`. HF NF4 Gemma is not recommended
on 16 GB — use Q4_K_M GGUF. W16 Whisper CT2 auto-preferred when present.

## Adding backends / languages

Follow 4-step pattern in [`CLAUDE.md`](./CLAUDE.md): ABC → file → `factory.py` branch → `tests/conftest.py` mock.

## Full reference

Model IDs, VRAM tiers, confidence thresholds, spec-decode caveats, adapter loading:
[`CLAUDE.md`](./CLAUDE.md)
