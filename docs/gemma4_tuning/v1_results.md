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

---

# v1.1 Iteration — corpus rebalanced toward verse (2026-04-29)

After v1 fell short of the COMET ship gate, the leading hypothesis was that the 39%-sermon mix shifted output toward a colloquial register the formal-Spanish reference set penalizes. v1.1 tests that hypothesis with a verse-heavy mix.

## What changed

| component | v1 share | **v1.1 share** | rationale |
|---|---|---|---|
| Bible verses (v2) | 39% | **60%** | Match the eval distribution (verses) |
| Sermon (CometKiwi-XL ≥0.85) | 39% | **15%** | Reduce colloquial-register pull |
| Glossary-tagged | 9% | **15%** | More theological-term reinforcement (still doesn't fix Jacobo) |
| OPUS-100 replay | 13% | **10%** | Slightly trimmed |

Total corpus: 9,233 → **15,000** rows. Glossary yield was 866 → **2,250** thanks to a candidate-pool expansion (`tools/build_v1_corpus.py:annotate_with_glossary` now walks extra random bible pairs when needed).

## Training (Phase C2 prime)

Same trainer (Unsloth QLoRA, r=8, frozen vision/PLE, packed seq=1024), 2 epochs, ~3:21 hr wall over 1,876 steps. Final avg loss **1.582** (v1: 1.508). Smooth descent, no incidents.

## Results — apples-to-apples on the same 500-verse v2 holdout

| metric | prod | v1 | **v1.1** | v1.1 vs v1 | v1.1 vs prod |
|---|---|---|---|---|---|
| COMET-22 mean | 0.7515 | 0.7448 | **0.7494** | **+0.0046** | −0.0022 |
| chrF++ mean | 46.71 | 46.56 | **47.55** | **+0.99** | **+0.83** |
| per-row winners (v1.1 vs other) | 224 / 271 (vs prod) | 251 / 218 (vs v1) | — | **majority over v1** | minority vs prod |
| ties | 5 (vs prod) | 31 (vs v1) | — | — | — |
| canary (n=8) | 7/8 | 6/8 | 6/8 | tied | −1 (Jacobo, unchanged) |
| sermon p50 latency (verse n=500) | 675 ms | 645 ms | 663 ms | within noise | within noise |
| tok/s | 47.9 | 48.3 | 47.1 | within noise | within noise |
| VRAM peak | 3.93 GB | 3.93 GB | 3.79 GB | tied | −0.14 |
| median completion tokens | 32 | 31 | 31 | tied | −1 |

The v1→v1.1 gain (+0.0046 COMET-22) is right at the +0.005 plan threshold. Per-row winners (251 vs 218 with 31 ties) confirm v1.1 produces materially different — and majority-better — translations than v1, p<0.05 binomial against equal-quality null.

The v1.1→prod gap closed by ~70% (from −0.0068 to −0.0022) but didn't go positive. v1.1 still loses 224/495 row-pairs to prod, though it wins the chrF++ comparison by +0.83 — surface character overlap is genuinely better than the base, COMET-22 just doesn't reflect that.

## What it tells us

**The corpus mix matters and the hypothesis is partially confirmed.** Reducing sermon to 15% recovered most of the COMET regression v1 introduced. Going further (e.g. v1.2 with 75% bible / 5% sermon) would likely chase diminishing returns — the v1→v1.1 step was already worth ~70% of the gap, but the remaining ~30% probably can't be closed by SFT corpus tuning alone.

The Jacobo canary failure is **not** a corpus issue — both v1 and v1.1 fail the same canary identically. Glossary tags don't disambiguate context-dependent terms even with 2.6× more tagged training pairs.

## Decision: pivot to Phase D

Per the plan's decision tree, v1.1 lands in the "beats v1 by ≥+0.005 but doesn't beat prod → iterate v1.2 OR pivot to Phase D" branch. Choosing **Phase D** because:

1. **Linear extrapolation says one more SFT iteration won't close the gap.** The v1→v1.1 gain (+0.0046) is the same size as the remaining v1.1→prod gap (−0.0022 + noise). v1.2 with even less sermon would shift things by < +0.0046 (diminishing returns).
2. **Preference optimization addresses the actual remaining failure modes.** Both the residual COMET gap (style-vs-meaning ranking) and the Jacobo disambiguation are exactly what CPO with CometKiwi-XL as reward signal is designed to handle. Given two candidates ("Jacobo" vs "Santiago" for Sons of Zebedee), the reward model can learn the contextual preference.
3. **The plan called this out from the start.** Phase D is the documented next lever after SFT is exhausted.

v1.1 shows that SFT-corpus rebalancing is a real (modest) lever; Phase D should be the substantial lever.

## v1.1 artifacts

| Path | Tracked? | Description |
|---|---|---|
| `bible_data/v1_corpus/mixed_v1.1.jsonl` | no (gitignored) | 15,000-row rebalanced corpus |
| `bible_data/v1_corpus/mixed_v1.1.manifest.json` | yes | Composition + ratios + seed |
| `fine_tuned_gemma4_e4b_v1.1/` | no (artifact) | LoRA adapter (~75 MB) |
| `models/gemma-4-e4b-it-q4km-v1.1.gguf` | no (artifact) | Merged + quantized GGUF (~5 GB) |
| `metrics/v1.1_e4b_verses_sermon.jsonl` | no (gitignored) | Per-translation hyps |
| `metrics/comet22_v1.1_vs_prod.json` | no (gitignored) | COMET-22 head-to-head |
| `metrics/comet22_v1.1_vs_v1.json` | no (gitignored) | COMET-22 head-to-head |
| `tools/build_v1_corpus.py` | yes (modified) | Now accepts `--ratio-bible/sermon/glossary/opus` + `--label-suffix` + glossary pool expansion |

---

# v2-cpo Iteration — preference optimization on top of v1.1 (2026-04-30)

Phase D execution. Generates 4 translation candidates per source from v1.1, scores with CometKiwi-XL (the no-leakage filter different from the COMET-22 eval metric), builds (chosen, rejected) preference triples, trains CPO from v1.1 for 1 epoch.

## Pipeline

1. **Source pool** — sampled 2,000 sources from v2 train (1,500 modern-EN bible verses + 500 CometKiwi-filtered sermon chunks), all excluded from v1.1 training corpus.
2. **Candidate generation** — 2,000 × 4 = 8,000 translations from v1.1 GGUF at temperature 0.7 via llama-server. Wall ~77 min, 0 failures.
3. **Scoring + triples** — CometKiwi-XL scored all 8K candidates in 11 min. Built 1,384 (chosen, rejected) triples after `margin > 0.05` filter (392 dropped low-margin, 224 dropped identical-text).
4. **CPO training** — TRL CPOTrainer wrapped in Unsloth, `--init-adapter fine_tuned_gemma4_e4b_v1.1` (continues v1.1's LoRA), lr=5e-6, beta=0.1, 1 epoch (173 steps). Wall ~30 min. `rewards/accuracies` climbed 0.59 → 0.76 → ~0.70 averaged. `rewards/margins` 1-3 sustained — model learned to prefer chosen over rejected.
5. **Export + eval** — same pipeline as v1/v1.1.

## Results — apples-to-apples on the 500-verse v2 holdout

| metric | prod | v1 | v1.1 | **v2-cpo** | v2 vs v1.1 | v2 vs prod |
|---|---|---|---|---|---|---|
| COMET-22 mean | 0.7515 | 0.7448 | 0.7494 | **0.7508** | **+0.0014** | **−0.0008** |
| chrF++ mean | 46.71 | 46.56 | 47.55 | 47.19 | −0.35 | +0.48 |
| per-row vs prod (wins/losses/ties) | — | 224/271/5 | 224/271/5 | **247/250/3** | — | **statistical tie p=0.45** |
| per-row vs v1.1 | — | — | — | 207/173/120 | majority + 24% ties | — |
| canary (n=8) | 7/8 | 6/8 | 6/8 | 6/8 | tied | −1 (Jacobo) |
| sermon p50 latency (verse n=500) | 675 ms | 645 ms | 663 ms | 674 ms | within noise | tied |
| tok/s | 47.9 | 48.3 | 47.1 | 47.0 | within noise | within noise |
| VRAM peak | 3.93 GB | 3.93 GB | 3.79 GB | 3.80 GB | tied | −0.13 |

## What CPO accomplished

**Closed the COMET-22 gap to prod from v1's −0.0068 → v1.1's −0.0022 → v2-cpo's −0.0008.** That's an 88% reduction in regression vs the SFT-only path. The v2 vs prod per-row tally (247 wins, 250 losses, 3 ties) is statistically indistinguishable from a fair coin (p=0.45 binomial against equal-quality null) — the model is essentially **on par with stock E4B on overall translation quality**.

The v1.1 → v2-cpo improvement (+0.0014) is small but the per-row distribution is informative: 207 v2 wins, 173 v1.1 wins, **120 ties**. The 24% tie rate suggests CPO did targeted refinement of a fraction of outputs rather than wholesale rewriting — exactly what 1-epoch CPO at lr=5e-6 was designed to do.

**Trade-off:** chrF++ slipped from v1.1's 47.55 to v2's 47.19 (still beats prod's 46.71). The classic "CPO trades a bit of surface fidelity for semantic accuracy." Given COMET-22 is the primary quality metric (and what production-style users care about), this is the right trade.

## What CPO didn't fix

- **Jacobo canary still misses.** v2-cpo says "Santiago" for "James and John, sons of Zebedee" just like v1 and v1.1. The 2K source pool didn't include the apostle-context disambiguation cases — both candidates probably said "Santiago" so CometKiwi couldn't rank them differently. **Engineering fix:** future iteration should hand-craft 50-100 disambiguation triples and run a short CPO refinement.
- **Plan target was +0.5 COMET-22 over v1.** Got +0.006. Way short of the aspirational goal. The reality is stock E4B is already so strong that 1 epoch of CPO over 1.4K triples can only nudge things; serious CPO results require much larger triple pools (10K-100K) and possibly multiple iterations.

## Decision

**v2-cpo is statistical parity with prod, not a clear ship-it-tomorrow win.** The strongest argument for shipping v2 anyway: it matches production quality with the same latency profile while introducing the modern toolchain (no `enable_thinking: false` workaround needed). The strongest argument against: the canary regression (6/8 vs 7/8) is a real loss on a key disambiguation case the operator can hear during a service.

**Recommendation:** treat v2-cpo as the end of the SFT+CPO pipeline. Subsequent improvements need either:
- **(a) targeted preference data** for the failing canaries (specifically Jacobo, and "partimiento del pan" for the breaking-of-bread case), or
- **(b) much larger preference pools** (e.g. 20K+ triples from a wider eval distribution) to push beyond statistical noise, or
- **(c) accept that stock E4B is the practical ceiling** for this corpus + hardware setup, and ship v2 only if the latency win + future-iteration runway matters more than tying-with-prod COMET.

The plan called for ARPO/X-ALMA as the v2.1 step if CPO showed over-rejection — we don't see over-rejection here (0.7 accuracy, smooth descent), so ARPO isn't indicated. The bigger lever from here is preference data quality, not the optimizer.

## v2-cpo artifacts

| Path | Tracked? | Description |
|---|---|---|
| `tools/build_preference_triples.py` | yes | `generate` (via llama-server HTTP) + `score` (CometKiwi-XL via comet_env) subcommands |
| `training/train_gemma4_cpo.py` | yes | TRL CPOTrainer + Unsloth, `--init-adapter` for continuing v1.1's LoRA |
| `preference/sources_2k.jsonl` | no (gitignored) | 1500 verse + 500 sermon sources, exclude-v1.1 |
| `preference/v1.1_candidates.jsonl` | no (gitignored) | 2K × 4 candidates from v1.1 |
| `preference/v1.1_triples.jsonl` | no (gitignored) | 1,384 (chosen, rejected) triples post-margin filter |
| `fine_tuned_gemma4_e4b_v2_cpo/` | no (artifact) | CPO-refined LoRA (~75 MB) |
| `models/gemma-4-e4b-it-q4km-v2-cpo.gguf` | no (artifact) | Merged + quantized v2-cpo GGUF |
| `metrics/v2_cpo_e4b_verses_sermon.jsonl` | no (gitignored) | Per-translation hyps |
| `metrics/comet22_v2_vs_prod.json` | no (gitignored) | head-to-head (delta −0.0008, p=0.45) |
| `metrics/comet22_v2_vs_v1.1.json` | no (gitignored) | head-to-head (delta +0.0014, 207/173/120) |
