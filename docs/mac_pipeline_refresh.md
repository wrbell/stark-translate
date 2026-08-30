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
| Phase 7 health gate on M3 | `tools/health_check.py --backend mlx` |

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

## 2. Baseline A/B latency (TranslateGemma 4B ± 12B)

```bash
# 4B only (~4.3 GB)
python dry_run_ab.py --backend mlx

# A/B (~11 GB) — use when you want 4B vs 12B side-by-side
python dry_run_ab.py --ab --backend mlx
```

Model IDs come from settings (`STARK_TRANSLATION__MLX_MODEL_4B` / `_12B`). Defaults remain community TranslateGemma 4-bit; do **not** assume Gemma 4 E4B MLX weights exist yet.

### TurboQuant (optional KV compression)

```bash
pip install mlx-optiq   # if not already via '.[mlx]'
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
| Live 4B | `python dry_run_ab.py --backend mlx` |
| Live A/B + TQ | `python dry_run_ab.py --ab --backend mlx --turboquant` |
| Live + LoRA | `python dry_run_ab.py --backend mlx --adapter-dir <path>` |
| Health (MLX) | `python tools/health_check.py --backend mlx --n-canaries 8` |
| Mine phantoms | `python tools/mine_hallucination_phrases.py` |
| Deploy dry-run | `python tools/deploy_adapters.py ... --dry-run` |
| Operator UI | `uvicorn operator_app.main:app --port 9000` → `/operator/` |

---

## Honesty notes

- Mac default translation remains **TranslateGemma** MLX 4-bit until a proven Gemma 4 MLX community build + domain adapter exists.
- Operator control plane does not yet expose TurboQuant / adapter-dir toggles — use CLI/env for those.
- SIGUSR1 hot-reload of adapters is still a deploy.md TODO; restart the process after swapping `active/`.
