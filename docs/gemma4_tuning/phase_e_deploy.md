# Phase E — Deploy

**Goal:** roll the v1 (or v2) adapters into production with a one-step rollback path.

**Wall clock:** 1–2 days.

## E1. Merge → GGUF Q4_K_M for both E4B and E2B

```bash
python training/export_gguf.py --adapter fine_tuned_gemma4_e4b_v1 --output models/gemma-4-e4b-it-q4km-v1.gguf
python training/export_gguf.py --adapter fine_tuned_gemma4_e2b_v1 --output models/gemma-4-e2b-it-q4km-v1.gguf
```

(Re-uses `training/export_gguf.py` from Phase A. The verify-chat-template step inside the script is mandatory; do not skip.)

## E2. Register adapters via `tools/manage_adapters.py`

The script's model registry currently knows `gemma_4b` / `gemma_12b` / `whisper_turbo`. Extend to add:

- `gemma_4_e4b`
- `gemma_4_e2b`

Add a `gguf_path` field to the manifest schema to track the deployed GGUF artifact (currently only tracks `safetensors` adapters).

```bash
python tools/manage_adapters.py register \
    --adapter fine_tuned_gemma4_e4b_v1 \
    --model gemma_4_e4b \
    --gguf models/gemma-4-e4b-it-q4km-v1.gguf \
    --eval-file metrics/v1_e4b_diff.json
```

Version naming convention: `cycle{N}_{YYYYMMDD}_{sha256[:8]}` (e.g. `cycle1_20260512_abc12345`).

## E3. Extend `tools/health_check.py` from 5 to 8 canary sentences

Match `scripts/benchmarks/bench_translate_t1_t4.py` Tier B canary set. This gives consistent canary scores between the dev benchmark and the deploy gate.

Bandit-acceptable extension: add 3 sentences to the existing `CANARY_SENTENCES` list, each with:
- `source` (English)
- `expected_substrings` (Spanish term list — at least one must appear)
- `latency_max_s = 10.0` (unchanged from existing)

## E4. Activate

```bash
python tools/manage_adapters.py activate --model gemma_4_e4b --version cycle1_20260512_abc12345
```

This auto-runs `tools/health_check.py`. Activation is atomic — `active/` and `previous/` symlinks swap together.

## E5. A/B in production for one Sunday service

Operator control plane already has runtime engine swap UI (Phase 9 / v2026.6). The volunteer running the service can toggle between the v1 adapter and stock E4B mid-session if anything sounds off.

### Post-service review checklist

User (subjective, not scored):
- [ ] Did Spanish translations sound natural and contemporaneous?
- [ ] Did theological terms come out correctly (atonement → expiación, James epistle → Santiago, covenant → pacto, sanctification → santificación)?
- [ ] Any latency hiccups vs. baseline?
- [ ] Any hallucinations or repeated phrases?

System (logged):
- [ ] p50 / p95 latency in operator dashboard.
- [ ] CometKiwi-XL on-the-fly QE (if enabled) — average score across the session.
- [ ] Operator-flagged segments (operator UI has a flag button on each translation).

### Rollback

```bash
python tools/manage_adapters.py rollback --model gemma_4_e4b
```

One-step, atomic — swaps `active/` and `previous/` symlinks.

## E6. Cleanup of runtime workaround (only after one full successful service)

Once v1 is in production for at least one full Sunday service with positive review, the `enable_thinking: false` workaround at `engines/llamacpp_engine.py:132-138` becomes redundant — the trained adapter bakes this in.

**Do NOT remove prematurely.** The workaround is harmless when not needed and removing it before the adapter is permanent breaks rollback.

When ready to remove: delete the `payload["chat_template_kwargs"] = {"enable_thinking": False}` lines and the surrounding comment, then re-test the canary against both the v1 and stock GGUFs to verify both still produce direct (non-thinking) output.
