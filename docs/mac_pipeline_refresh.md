# Mac / MLX Pipeline Refresh Runbook

> **Audience:** Willem on the MacBook (M3 Pro, 18GB) — inference, live demos, Phase 7 scaffolding  
> **Purpose:** Same-day Mac checklist that uses the pipeline refresh **without WSL**. Companion to [`wsl_pipeline_refresh.md`](./wsl_pipeline_refresh.md).  
> **Related:** [`CLAUDE-macbook.md`](../CLAUDE-macbook.md) · [`docs/operator_runbook.md`](./operator_runbook.md) · [`CLAUDE.md`](../CLAUDE.md)

> **Last updated:** 2026-08-30

---

## What you can do today (no WSL)

| Already live | How |
|--------------|-----|
| Garbage filter | Any `python dry_run_ab.py --backend mlx` session |
| File-side AL | `mine_hallucination_phrases.py`, `prepare_finetune_data.py`, `merge_corrections.py`, `deploy_adapters.py --dry-run` |
| Live A/B + operator UI | `dry_run_ab.py --ab`, `uvicorn operator_app.main:app --port 9000` |
| TurboQuant + LoRA path | `--turboquant`, `--adapter-dir` (this Mac refresh) |
| Gemma 4 OptiQ + MTS | Default finals; `--mts` for drafter — see [`mlx_cuda_parity.md`](./mlx_cuda_parity.md) |
| TranslateGemma opt-out | `--model-family translategemma` |
| Phase 7 health gate on M3 | `tools/health_check.py --backend mlx` (defaults to Gemma4 OptiQ) |

## Do **not** run on Mac

These need the A2000 / WSL box — see [`wsl_pipeline_refresh.md`](./wsl_pipeline_refresh.md):

- Phase 4 audio preprocess (`training/run_phase4_preprocess.sh`)
- W17 Whisper DoRA train + CT2 export
- Gemma 4 E4B domain SFT + GGUF export
- Parakeet / NeMo EN bench
- Loading Whisper LoRA in live mlx-whisper (mlx-whisper has no LoRA; CT2 is CUDA-only)

**Whisper LoRA on Mac:** evaluate with PyTorch/PEFT offline if needed; live STT stays base mlx-whisper until CT2 adapters land on a CUDA church PC.

---

## 0. Env (once)

```bash
cd ~/stark-translate   # or your clone
source stt_env/bin/activate
pip install -U '.[mlx]'          # includes mlx-optiq for TurboQuant
git pull origin main
```

---

## 1. Validate the live garbage filter

```bash
python dry_run_ab.py --backend mlx --dry-run-text "Subtítulos realizados por la comunidad de Amara.org"
# Expect filter drop / no translation spam for known phantoms

python dry_run_ab.py --backend mlx   # short mic session; check metrics/diagnostics_*.jsonl
```

Optional: mine more phantoms from prior sessions:

```bash
python tools/mine_hallucination_phrases.py --glob 'metrics/diagnostics_*.jsonl'
```

---

## 2. Baseline A/B latency (Gemma 4 OptiQ ± TranslateGemma)

```bash
# Default: Gemma 4 E4B OptiQ
python dry_run_ab.py --backend mlx

# TranslateGemma 4B opt-out
python dry_run_ab.py --backend mlx --model-family translategemma

# A/B TranslateGemma 4B vs 12B (~11 GB)
python dry_run_ab.py --ab --backend mlx --model-family translategemma
```

Model IDs come from settings (`STARK_TRANSLATE_MLX_MODEL_GEMMA4_E4B` / TG `STARK_TRANSLATE_MLX_MODEL_4B`). For MTS / size overrides:

```bash
python dry_run_ab.py --backend mlx --mts
python dry_run_ab.py --backend mlx --gemma4-size e2b
python tools/benchmark_mlx_accel.py --quick --configs tg4b,e4b,e4b_mts
```

See [`mlx_cuda_parity.md`](./mlx_cuda_parity.md). Naïve uniform Gemma 4 MLX 4-bit quants (PLE quantized) produce garbage — use OptiQ / PLE-safe builds only.

### Mac latency notes (2026-08-30 soak)

| Config | medium p50 | Canary | Role |
|--------|------------|--------|------|
| **e4b OptiQ (default)** | ~2176 ms | **7/8** | **Quality mode / Mac default** |
| tg4b (opt-out) | ~899 ms | 6/8 | Faster legacy finals |
| e2b OptiQ | ~1624 ms | **0/8** | Not usable — PLE/canary collapse on this build |
| e4b + MTS (`-assistant`) | LOAD FAIL | — | `gemma4_assistant` unsupported in current mlx-lm; skip `--mts` until mlx-lm catches up |

**Demo mode vs quality mode:** keep **E4B OptiQ** for demos that need theological register; use `--model-family translategemma` when you need sub-second finals and can accept TG canary gaps. Do not ship E2B OptiQ until canaries recover.

### TurboQuant (optional KV compression)

`--turboquant` is soft-disabled against **mlx-optiq 0.4.x**: that package imports as `optiq` and does not expose a drop-in `TurboQuantKVCache` for the `mlx_lm.generate` path (KV mixed-precision is serve/runtime-only). The flag remains for forward compatibility; the pipeline logs a warning and continues without TQ.

```bash
# Will warn and continue without TQ on mlx-optiq 0.4.x:
python dry_run_ab.py --ab --backend mlx --turboquant
# or: export STARK_TRANSLATION__TURBOQUANT=true
```

Disable with `--no-turboquant`. Bits: `STARK_TRANSLATION__TURBOQUANT_KEY_BITS` (default 3) / `_VAL_BITS` (default 4).

### LoRA adapters (after WSL transfer)

```bash
# Place adapter under adapters/.../active/ (adapter_config.json + weights)
python dry_run_ab.py --backend mlx \
  --adapter-dir adapters/translategemma_4b/active

# A/B with optional 12B adapter
python dry_run_ab.py --ab --backend mlx \
  --adapter-dir adapters/translategemma_4b/active \
  --adapter-dir-b adapters/translategemma_12b/active
```

Factory path (tools / validate): `create_translation_engine(backend="mlx", adapter_path=..., use_turboquant=True)`.

---

## 3. Phase 7 gate — MLX health check (8 canaries)

```bash
# Base model only
python tools/health_check.py --backend mlx --n-canaries 8 --verbose

# With adapter (after transfer from WSL)
python tools/health_check.py --backend mlx --n-canaries 8 \
  --base-model mlx-community/translategemma-4b-it-4bit \
  --adapter adapters/translategemma_4b/active \
  --max-latency 5 --output metrics/health_mlx.json
```

**Gate:** overall PASS on 8 canaries (including *partimiento del pan*). CUDA/bnb path remains: `--backend cuda` on WSL.

Deploy dry-run (no SSH required):

```bash
python tools/deploy_adapters.py --cycle N --models translategemma_4b --endpoints mac-dev --dry-run
```

---

## 4. YouTube caption compare (Layer 4)

After a live or file session, use the YT comparison tools in [`tools/CLAUDE.md`](../tools/CLAUDE.md) (windowed WER / text-anchor alignment). Trend should not regress vs your pre-refresh Mac baseline.

---

## 5. When WSL adapters arrive

1. Copy LoRA / GGUF / CT2 folders into `adapters/` (see WSL runbook §5).
2. Run §3 health_check on Mac for MT adapters.
3. Re-run §2 with `--adapter-dir`.
4. Church CUDA PC: prefer CT2 Whisper + llama.cpp Gemma path (not this Mac doc).

---

## Quick command index

| Goal | Command |
|------|---------|
| Live Gemma4 OptiQ | `python dry_run_ab.py --backend mlx` |
| Live A/B + TQ (TG) | `python dry_run_ab.py --ab --backend mlx --model-family translategemma --turboquant` |
| Live Gemma 4 + MTS | `python dry_run_ab.py --backend mlx --mts` |
| Accel matrix | `python tools/benchmark_mlx_accel.py --quick` |
| Live + LoRA | `python dry_run_ab.py --backend mlx --adapter-dir <path>` |
| Health (MLX) | `python tools/health_check.py --backend mlx --n-canaries 8` |
| Mine phantoms | `python tools/mine_hallucination_phrases.py` |
| Deploy dry-run | `python tools/deploy_adapters.py ... --dry-run` |
| Operator UI | `uvicorn operator_app.main:app --port 9000` → `/operator/` |

---

## Honesty notes

- Mac default translation is **Gemma 4 OptiQ E4B**. TranslateGemma remains available via `--model-family translategemma`.
- Domain adapters from WSL still improve theological canaries (Santiago / partimiento); base OptiQ alone is ~7/8.
- Operator control plane does not yet expose TurboQuant / adapter-dir toggles — use CLI/env for those.
- SIGUSR1 hot-reload of adapters is still a deploy.md TODO; restart the process after swapping `active/`.
- `--turboquant` is soft-disabled on mlx-optiq 0.4.x (no drop-in TurboQuantKVCache for mlx_lm.generate).
