# Gemma 4 Tuning — Phase B + C v1 Results (2026-04-29)

End-to-end results from the Phase B spike and Phase C v1 ship attempt. The
plan documents (`overview.md`, `phase_b_spike.md`, `phase_c_domain_sft.md`)
remain the as-written plan; this doc records what actually happened so the
next iteration is informed by it.

## Phase B — Spike (3,400 pairs, 200 steps, 1 epoch — smoke test)

**Verdict: PASS the smoke-test gates, FAIL the canary parity gate.**

| metric | prod E4B (no FT) | spike | delta |
|---|---|---|---|
| sermon p50 latency | 469 ms | 404 ms | **−14%** |
| sermon p95 | 640 ms | 566 ms | −12% |
| sermon tok/s | 40.6 | 44.8 | +10% |
| hallucination ratio | 0.951 | 0.946 | tied |
| canary | 7/8 | **6/8** | −1 |
| VRAM peak | 4.94 GB | 5.06 GB | +0.12 |

**The end-to-end pipeline works** — Unsloth QLoRA → bf16 merge → GGUF Q4_K_M → llama-server → translation. Loss converged 9.5 → 1.66 over 200 steps. Final adapter loaded cleanly via llama-server and produced sensible Spanish output.

**No-thinking behavior is baked into the adapter.** Even without the runtime `enable_thinking: false` override, the merged spike GGUF goes straight to translation (no chain-of-thought leak into `reasoning_content`). This validates the Phase E6 cleanup plan to retire the runtime workaround once a permanent adapter ships.

**The −1 canary regression** ("James and John, the sons of Zebedee" → spike said "Santiago" instead of "Jacobo") foreshadows the Phase C glossary-disambiguation problem.

## Phase C0 — Corpus + holdout prep

- **Eval holdout rebuilt** at `bible_data/aligned/verse_pairs_test_v2.jsonl` (500 verses, stratified across 6 buckets) via the v2-aware extension to `tools/build_eval_sets.py`. Replaces the broken 2-line stub. Training corpus reduced 251,504 → 251,004 (500 removed for holdout).
- **CometKiwi-XL filter** on the 21,828 DeepL sermon pairs at threshold 0.85 → kept 3,587 (16.4%). DeepL sermon translations cluster in 0.7–0.85 (median 0.74), so 0.85 is harsh-but-honest.
- **v1 corpus** built via `tools/build_v1_corpus.py`: 9,233 rows total. Composition: sermon (CometKiwi ≥0.85) 39%, bible (v2, modern-EN biased) 39%, glossary-tagged 9%, OPUS-100 replay 13%. Glossary yield was lower than the planned ~30% — only 866 of the source pool had Tier-2 glossary terms present on both EN and ES sides.

## Phase C2 — v1 training (9,233 pairs × 2 epochs)

**Wall clock:** ~2 hours (1,156 steps at ~6.5 s/step on A2000 Ada). Model load 141 s (Unsloth cache warm).

**Loss trajectory:** 3.88 → 2.68 → 2.35 → 1.50 → ~1.16-1.35 (second-epoch oscillation). Final reported avg = **1.508**. Smooth monotonic descent, grad norm steady 1.0–1.5. Inside the 1–3 sanity gate; better than the spike's 1.66.

**No incidents.** No OOM, no NaN, no chat-template mismatch.

## Phase C3 — Export + bench

GGUF export ran cleanly: bf16 merge → GGUF f16 → Q4_K_M (~7 min total). Metadata verified: `architecture=gemma4`, `eos_token_id=106`, `chat_template_len=11926`, `bos_token_id=2`. Same profile as the spike GGUF.

### Latency / throughput / canary (n=125 sermon, n=8 canary)

| metric | prod E4B (no FT) | v1 (9.2K, 2ep) | delta |
|---|---|---|---|
| sermon p50 latency | 469 ms | 394 ms | **−16%** |
| sermon p95 | 640 ms | 570 ms | −11% |
| sermon tok/s | 40.6 | 45.9 | +13% |
| hallucination ratio | 0.951 | 0.947 | tied |
| canary | 7/8 | **6/8** | −1 |
| VRAM peak | 4.94 GB | 3.88 GB | −1.06 (worth verifying) |
| median completion tokens | 19 | 19 | tied |

The same canary failure as the spike: "James and John, the sons of Zebedee" → v1 said "Santiago" instead of "Jacobo". Despite training on 866 glossary-tagged pairs with `<g>James||Santiago</g>` and `<g>James||Jacobo</g>`, the model didn't learn the apostle-vs-epistle disambiguation. This isn't a data-volume issue — it's that the glossary tagging gives the model two valid mappings without the contextual signal needed to choose between them.

### COMET-22 head-to-head (n=500 verse holdout, reference-based)

| metric | prod E4B (no FT) | v1 | delta |
|---|---|---|---|
| **COMET-22 mean** | **0.7515** | **0.7448** | **−0.0068** ❌ |
| chrF++ mean | 46.71 | 46.56 | −0.15 |
| per-row winners | 273 / 500 | 219 / 500 | prod wins more often |
| ties | — | 8 | — |

**Plan target: ≥ +2.0 COMET-22. Actual: −0.0068.** The Phase C ship-gate fails on the quality dimension. The −0.0068 delta is within ~1 standard-error band (stdev 0.106, SE ≈ 0.005, 95% CI ±0.009) so isn't formally significant in the mean, but the per-row winner count (273 vs 219) is a clear binomial signal (p < 0.01 against equal-quality null) — v1 produces meaningfully different translations than prod, and prod wins the head-to-head.

### Why v1 didn't beat the base

This was an SFT-only run on a generalist instruct model. The plan predicted exactly this risk pattern (overview.md, the same shape as the S6 plateau) and offered preference optimization (CPO / X-ALMA in Phase D) as the actual lever. v1 confirms that hypothesis empirically:

1. **Stock E4B is already a strong translator** (COMET-22 = 0.75 baseline). There's not much headroom for a 9K-pair SFT to improve over it.
2. **Mixed-register training corpus may have shifted output style toward sermon-colloquial Spanish**, which a formal-Spanish (RVR1909 + Platense) reference set penalizes via COMET. Future runs should probably reduce the sermon share or stratify the eval to test sermon and verse domains separately.
3. **Glossary tags don't disambiguate context-dependent terms** like James → Jacobo/Santiago. The current `<g>EN||ES</g>` annotation approach gives the model conflicting evidence rather than contextual cues. Phase D preference optimization addresses this naturally — given two candidates ("Jacobo" vs "Santiago" for the same source), CometKiwi-XL can rank them based on context.

## What we got right (worth keeping)

- **Latency win is real and reproducible** (B1, C2 both show ~14–16% p50 latency reduction). The fine-tune produces tighter outputs that finish faster. Useful even if quality is flat.
- **No-thinking behavior bakes in via training.** The runtime `enable_thinking: false` workaround (`engines/llamacpp_engine.py:138`) becomes redundant once a tuned adapter is permanent — exactly as Phase E6 predicted.
- **The Unsloth + Gemma 4 + LoRA r=8/alpha=8 + frozen vision/PLE config is solid.** No instabilities, no OOMs, clean convergence. Re-use this base for any future SFT.
- **CometKiwi-XL filtering integrates cleanly** into the corpus pipeline. The 0.85 threshold is harsh but produces cleaner training data.
- **Platense alignment fix (PR #96) was prerequisite.** Without it the bible mix would have been ~50% noise. Confirmed by spot-checking the v2 corpus during build.

## What needs to change for v2

Three options, in order of expected leverage:

1. **Skip ahead to Phase D — preference optimization.** Generate 4 candidates per source from v1, score with CometKiwi-XL, build (chosen, rejected) triples, train CPO. Plan target +0.5–1.5 COMET-22 over v1, which would land us at break-even-or-better vs prod. **Most plan-aligned next step.**
2. **Re-tune SFT with corpus rebalanced** — drop sermon to ~15%, bump verse to ~60%, keep glossary at ~15%. May restore verse quality without losing all sermon learning. Lower expected gain than CPO.
3. **Ship v1 as-is for production A/B.** Latency win is real, COMET delta is within noise band, operator UI has instant rollback. Lowest effort but commits to a small quality regression for material latency gain.

The plan's `phase_d_preference_optimization.md` is the natural next stop.

## Artifacts produced (not all tracked)

| Path | Tracked? | Description |
|---|---|---|
| `tools/build_v1_corpus.py` | yes | 4-source corpus mixer |
| `tools/score_comet22.py` | yes | Reference-based COMET-22 head-to-head scorer |
| `tools/build_eval_sets.py` | yes | Extended with `--train-path` / `--test-path` overrides for v2 |
| `bible_data/aligned/verse_pairs_test_v2.jsonl` | no (gitignored) | 500-verse stratified holdout, regenerable via `build_eval_sets.py` |
| `bible_data/synthetic/deepl_sermon_pairs_kiwi85.jsonl` | no (gitignored) | 3,587 sermon pairs above CometKiwi-XL 0.85 |
| `bible_data/v1_corpus/mixed.jsonl` | no (gitignored) | 9,233-row v1 training corpus |
| `bible_data/v1_corpus/mixed.manifest.json` | yes | Composition + provenance |
| `bible_data/eval_registry.json` | yes | Updated to point at the v2 holdout |
| `fine_tuned_gemma4_e4b_v1/` | no (artifact) | LoRA adapter from C2 (~75 MB safetensors) |
| `models/gemma-4-e4b-it-q4km-v1.gguf` | no (artifact) | Merged + quantized v1 GGUF (~5 GB) |
| `metrics/v1_e4b_verses_sermon.jsonl` | no (gitignored) | Per-translation hyps from v1 (500 holdout verses) |
| `metrics/prod_e4b_verses_sermon.jsonl` | no (gitignored) | Same for stock E4B |
| `metrics/comet22_v1_vs_prod.json` | no (gitignored) | The head-to-head COMET-22 result |
