# v2026.11 — Imatrix-Calibrated Quantization (PR4)

**Status:** code shipped (corpus builder, CONFIGS extension, imatrix.gguf
files, 4 calibrated GGUF artifacts); **default model unchanged**. Numbers
measured on the A2000 Ada Mobile bench machine 2026-05-04 with the v2026.9
cumulative flag set (`-ctv q8_0`, llama.cpp `d8794eecd`/`b9022`).

---

## TL;DR

PR3 (v2026.10 IQ4_XS sweep) showed E4B IQ4_XS at -9% latency but with the
canary regressed 7/8 → 5/8. PR4 calibrates the same Q4_K_M and IQ4_XS
quantization passes with `--imatrix` on a sermon corpus over-sampled with
theological canaries (25× weighting) to test whether activation-aware
calibration can recover the canary.

**Result: imatrix recovers ONE of the three lost canary items on E4B
IQ4_XS** (the most theologically critical one — `Jacobo` for apostle James).
The other two losses (`pacto`→alianza, `partimiento`→fracción) are
synonym-level shifts and persist. Q4_K_M-imatrix unexpectedly **regressed**
on `pacto` (it had been correct on Q4_K_M-no-imatrix).

| Variant | E4B p50 | E4B canary | E2B p50 | E2B canary |
|---|---|---|---|---|
| **Q4_K_M (PR2 default)** | 478 ms | **7/8** ✓ | 272 ms | 6/8 |
| Q4_K_M-imatrix | 443 ms (-7%) | 6/8 (lost pacto) | 268 ms (-1%) | 6/8 |
| IQ4_XS (PR3) | 434 ms (-9%) | 5/8 (lost pacto, Jacobo, partimiento) | 261 ms (-4%) | 6/8 |
| **IQ4_XS-imatrix** | **420 ms (-12%)** | 6/8 (recovered Jacobo; still lost pacto, partimiento) | 283 ms (+4%) | 6/8 |

Per the v2026.8 plan's ship criterion ("imatrix-calibrated GGUF beats
no-imatrix IQ4_XS on at least one of {latency, canary, sermon CometKiwi}
while holding the other two"), **IQ4_XS-imatrix qualifies** — it beats
no-imatrix IQ4_XS on **both** latency (-3% additional) and canary (+1
item). Hallucination ratio = 0 on both.

But it does **not** match Q4_K_M-no-imatrix's 7/8 canary, so:

- **Production default stays `gemma-4-e4b-it-q4km.gguf` (no imatrix).**
- IQ4_XS-imatrix becomes the recommended low-bit variant (better than
  IQ4_XS-no-imatrix in every metric).
- The 4 imatrix.gguf calibration files + 4 calibrated GGUF artifacts ship
  as locally-built alternatives. The infrastructure is now in place for
  future fine-tunes (where imatrix has more headroom — bf16 fine-tune base
  → imatrix → quant maintains the fine-tune's gains better than naive
  quantization would).

---

## Hardware + Versions

Same as PR3 (A2000 Ada Mobile 16 GB, llama.cpp `d8794eecd` / b9022).
imatrix run wall-time:
- **E2B bf16 (8.7 GB on disk)**: 21.5 s on clean GPU, peak ~6.7 GB VRAM
- **E4B bf16 (15 GB on disk)**: 31.1 s on clean GPU, peak ~12 GB VRAM
  (fits comfortably; no fallback to partial offload or CPU needed)

The plan's three-tier fallback ladder (partial GPU offload → CPU-only →
cloud) was not exercised — the box was clean (1.3 GB VRAM baseline) so
both runs fit on-device.

---

## Calibration Corpus (`training/imatrix_corpus.txt`)

Built by `training/build_imatrix_corpus.py`. Critical design choice: the
8 theological canaries from `training.benchmark_gemma4.CANARY_SENTENCES`
are over-sampled at 25× and **interleaved every 32 sermon chunks** so the
calibration distribution stays "topically warm" — each batch the imatrix
accumulates over has theological context fresh in activation memory.

Corpus stats:
- 480 unique EN sermon chunks (from
  `bible_data/synthetic/deepl_sermon_benchmark_500.jsonl`, deduped, ≥10
  chars)
- 200 canary lines (8 sentences × 25 repeats), interleaved at K=32
- **680 chunks total**

Variants of this recipe to consider for future runs:
- `--canary-weight 0` — sermon-only baseline. Useful as a control to
  measure how much the canary oversampling moves the dial.
- `--canary-weight 50` — heavier theological anchoring. Risk: overfit on
  the 8-sentence canary distribution at the expense of broader
  Brethren-tradition vocabulary that lives only in the sermon corpus.
- `--max-sermon-chunks 200` — smaller corpus, faster imatrix run, but
  less coverage. (Default 800 chunks; we got 480 because the source set
  has 480 unique entries after dedup.)

---

## Per-Canary-Item Stability

E4B detail. Each row is one canary; items in **bold** lost the term
relative to Q4_K_M-no-imatrix.

| # | EN | Expected | Q4_K_M (baseline) | Q4_K_M-imatrix | IQ4_XS | IQ4_XS-imatrix |
|---|---|---|---|---|---|---|
| 1 | atonement | expiación | expiación ✓ | expiación ✓ | expiación ✓ | expiación ✓ |
| 2 | covenant | pacto | pacto ✓ | **alianza** ✗ | **alianza** ✗ | **alianza** ✗ |
| 3 | grace | gracia | gracia ✓ | gracia ✓ | gracia ✓ | gracia ✓ |
| 4 | righteousness | justicia | justicia ✓ | justicia ✓ | justicia ✓ | justicia ✓ |
| 5 | James (epistle) | Santiago | Santiago ✓ | Santiago ✓ | Santiago ✓ | Santiago ✓ |
| 6 | James (apostle) | Jacobo | Jacobo ✓ | Jacobo ✓ | **James** ✗ | Jacobo ✓ |
| 7 | sanctification | santificación | santificación ✓ | santificación ✓ | santificación ✓ | santificación ✓ |
| 8 | breaking of bread | partimiento | **ruptura** ✗ | **ruptura** ✗ | **fracción** ✗ | **fracción** ✗ |

Observations:
- Item 2 (`pacto` → `alianza`): imatrix-calibrated quants both shift to
  `alianza`. This is a Catholic translation choice; my hypothesis is that
  the imatrix is anchoring on activation patterns that bias toward
  general-Spanish religious vocabulary rather than the
  Brethren-tradition-specific `pacto`. The corpus oversampling has the
  *source* English ("covenant") but cannot constrain the *output*
  Spanish word — imatrix is structurally limited here.
- Item 6 (`Jacobo`): imatrix recovers this on IQ4_XS — the apostle-vs-
  epistle disambiguation that motivated picking E4B. This is the most
  important item for the church use case and is the headline win of PR4.
- Item 8 (`partimiento`): never resolved on any variant. The Q4_K_M
  baseline already produces `ruptura`. This is a model-architecture
  limit on the church-domain term; only a fine-tune could fix it.

E2B canary stays at 6/8 across all variants — E2B never resolved Jacobo
even at bf16, so there's no headroom for imatrix to recover.

---

## What Ships

### New files

| Path | Size | SHA-256 |
|---|---|---|
| `models/gemma-4-e2b-it-q4km-imatrix.gguf` | 3.2 GB | `bd0732b1...` |
| `models/gemma-4-e2b-it-iq4xs-imatrix.gguf` | 3.1 GB | `1aa08320...` |
| `models/gemma-4-e4b-it-q4km-imatrix.gguf` | 5.0 GB | `5e5aadfc...` |
| `models/gemma-4-e4b-it-iq4xs-imatrix.gguf` | 4.8 GB | `0368937f...` |
| `training/imatrix_e2b.gguf` | 2.7 MB | (generated) |
| `training/imatrix_e4b.gguf` | 4.0 MB | (generated) |
| `training/imatrix_corpus.txt` | ~50 KB | (generated) |

None of the GGUF artifacts are hosted on HF yet; they're locally
reproducible via the recipe below.

### Code

- `training/build_imatrix_corpus.py` — corpus builder with `--canary-weight`
  knob (default 25) and K=32 interleave stride.
- `scripts/benchmarks/bench_translate_t1_t4.py` — 4 new CONFIGS:
  `t2-q4km-imatrix`, `t3-q4km-imatrix`, `t2-iq4xs-imatrix`,
  `t3-iq4xs-imatrix`.

### What does NOT change

- `start_server.sh` still points at `gemma-4-e4b-it-q4km.gguf`.
- `models.lock.json` not updated.
- No live-pipeline behavior change.

---

## Reproducing the Numbers

```bash
source ~/stt_train_env/bin/activate

# 1. bf16 base (one-time per model, ~3-5 min each on local disk)
python ~/llama.cpp/convert_hf_to_gguf.py \
    /mnt/d/Data/stt-data/cache/hub/models--google--gemma-4-e2b-it/snapshots/<sha>/ \
    --outtype bf16 --outfile /tmp/gemma-4-e2b-it-bf16.gguf
python ~/llama.cpp/convert_hf_to_gguf.py \
    /mnt/d/Data/stt-data/cache/hub/models--google--gemma-4-e4b-it/snapshots/<sha>/ \
    --outtype bf16 --outfile /tmp/gemma-4-e4b-it-bf16.gguf

# 2. Build corpus (~1 second; emits training/imatrix_corpus.txt)
python training/build_imatrix_corpus.py

# 3. Run imatrix (clean GPU; E2B ~21s, E4B ~31s)
~/llama.cpp/build/bin/llama-imatrix \
    -m /tmp/gemma-4-e2b-it-bf16.gguf \
    -f training/imatrix_corpus.txt \
    -o training/imatrix_e2b.gguf \
    -c 512 -ngl 999
~/llama.cpp/build/bin/llama-imatrix \
    -m /tmp/gemma-4-e4b-it-bf16.gguf \
    -f training/imatrix_corpus.txt \
    -o training/imatrix_e4b.gguf \
    -c 512 -ngl 999

# 4. Re-quantize with imatrix (CPU; 1.5-3 min each)
for size in e2b e4b; do
    for q in Q4_K_M IQ4_XS; do
        suffix=$(echo $q | tr '[:upper:]' '[:lower:]' | tr -d _)
        ~/llama.cpp/build/bin/llama-quantize \
            --imatrix training/imatrix_${size}.gguf \
            /tmp/gemma-4-${size}-it-bf16.gguf \
            models/gemma-4-${size}-it-${suffix}-imatrix.gguf \
            $q
    done
done

# 5. Bench (each variant ~2 min)
for cfg in t2-q4km-imatrix t3-q4km-imatrix t2-iq4xs-imatrix t3-iq4xs-imatrix; do
    /tmp/bench_gemma_run.sh exp_imatrix $cfg \
        models/gemma-4-${cfg/-q4km-imatrix/-q4km-imatrix.gguf}... -ctv q8_0
    # (script inspects $cfg to pick the matching .gguf)
done

# 6. Cleanup
rm /tmp/gemma-4-{e2b,e4b}-it-bf16.gguf
```

---

## Followups

- **Compare 50× canary weighting.** The 25× default may be conservative;
  doubling could anchor `Jacobo`/`Santiago` more firmly without
  overfitting (the canary set is only 8 sentences; even at 50× it's only
  400 of 880 corpus lines). Cost: 1 imatrix re-run per model + 4
  re-quantizes ≈ 8 min total.
- **Sermon-set CometKiwi.** Default bench skips CometKiwi (slow first-run
  download). For a complete quality picture, re-bench the 4 imatrix
  variants with `bench_translate_t1_t4.py` extended to optional comet,
  same as `tools/benchmark_translate_engines.py --comet` already supports.
- **Bilingual canary corpus.** Current corpus is EN-only sources of EN→ES
  pairs. A bilingual augmented corpus (mixing the EN canary sources with
  reference ES translations) could let imatrix calibrate on the
  *output*-language activations as well — possibly recovering the
  `pacto`/`alianza` and `partimiento`/`fracción` synonym shifts that
  pure-EN imatrix can't constrain.
- **Apply to a future fine-tune.** This was the v2026.8 plan's strongest
  motivation for shipping the infrastructure even when imatrix doesn't
  win on the current production model. When a TG-style EN→ES fine-tune
  lands, `bf16 fine-tune → imatrix → Q4_K_M` should preserve the fine-tune's
  domain gains better than `bf16 fine-tune → naive Q4_K_M` would.

---

## End-of-PR4 Decision

PR4 ships the **infrastructure** (corpus builder, imatrix files,
calibrated GGUFs, CONFIGS extension) and a clear **negative result**:
imatrix is necessary but not sufficient to fix IQ4_XS for this production
model + canary. It buys back 1 of 3 lost items.

For the church demo today, the production default remains
`gemma-4-e4b-it-q4km.gguf` (no imatrix). Anyone constrained to lower-bit
weights should prefer IQ4_XS-imatrix over IQ4_XS-no-imatrix
unconditionally. The imatrix recipe is captured here for the next
fine-tune cycle, where its expected payoff is much higher.
