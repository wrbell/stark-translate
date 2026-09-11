# L2 — Gemma 4 E2B as a full-model speculative draft for E4B finals (MLX)

**Lane state:** REJECTED on the pre-declared promotion gate; the paired real-time replay (L2b) was therefore **not run**. Byte-identical output was established, and the medium/long-sentence speed-up is real and retained as a distinct follow-up hypothesis. No default changed; the draft path stays opt-in (`STARK_EXPERIMENT_DRAFT_MODEL_ID`, `STARK_EXPERIMENT_DRAFT_TOKENS`, PR #200).

This is a different mechanism from the rejected built-in MTP head (#177): the draft here is the separately loaded `mlx-community/gemma-4-e2b-it-OptiQ-4bit` model (same tokenizer as E4B, byte-identical `tokenizer.json`), verified by the E4B target through mlx-lm speculative decoding. It had never been measured on this Mac.

## Method

- Source `main` @ `50f81c6`, `stt_env`, HF offline, `STARK_EXPERIMENT_GEMMA_PREFIX_CACHE=false`, GPU otherwise idle. `tools/benchmark_mlx_accel.py --no-stt --sentences all --runs 5 --warmup 2`, greedy decoding (engine default), EN→ES.
- Sentences: the three fixed short/medium/long test sentences plus all 18 theological canary sentences (21 total, 5 runs each per config).
- Order: opening `e4b` → `e4b_e2b_draft_g1` → `g2` → `g3` in one process (so the identity map is computed across configs), then a separate closing `e4b` run as the drift anchor. Raw: `textbench_open_draft.json`, `textbench_close_e4b.json`, `textbench.log`.

## Identity

- `identity`: **21/21 sentences byte-identical across all four configs** (every run of every config produced the same SHA256 of the full output text); `all_identical` is true for every sentence within every config.
- The opening and closing E4B anchors are identical to each other on all 21 sentences (the pipeline is deterministic in this setting).
- Canaries: 13/18 for every config (unchanged from the E4B baseline; the same five canaries miss). All generations finished with `stop`.

## Latency (p50 of 5 runs, ms)

| config | short | medium | long | canary aggregate p50 | canary p95 | draft acceptance | Metal peak |
|---|---:|---:|---:|---:|---:|---:|---:|
| e4b (opening) | 532.7 | 1451.2 | 1673.6 | 696.9 | 909.0 | — | 6.4 GiB |
| e4b + E2B draft γ=1 | 435.4 | 1287.2 | 1582.9 | 670.8 | 838.8 | 0.50 | 10.5 GiB |
| e4b + E2B draft γ=2 | 459.3 | **1181.9** | **1397.5** | 648.5 | 800.5 | 0.67 | 10.5 GiB |
| e4b + E2B draft γ=3 | 397.3 | **1119.6** | 1398.3 | 674.7 | 838.0 | 0.75 | 10.5 GiB |
| e4b (closing) | 541.4 | 1464.1 | 1705.0 | 752.4 | — | — | 6.4 GiB |

Relative to both E4B anchors: γ=2 medium −18.6 % / −19.3 %, long −16.5 % / −18.0 %; γ=3 medium −22.8 % / −23.5 %, long −16.4 % / −18.0 %. On the canary aggregate (13–20-token outputs, the length of a typical live final — L1 measured 14 generated tokens at the median) γ=2 is −7.0 % / −13.8 % and γ=3 −3.2 % / −10.3 %, i.e. 22–104 ms.

## Gate

Pre-declared: some γ must reach p50 ≤ 0.85× (or −150 ms) versus **both** E4B anchors on **medium, long and the canary aggregate**, with 100 % byte identity and all `stop` finishes.

- Identity, canaries, finish reason: **met**.
- Medium and long: **met** at γ=2 and γ=3 against both anchors.
- Canary aggregate: **not met** at any γ (best −13.8 % / −104 ms against the closing anchor, −7 % against the opening one).

Result: **REJECTED** for promotion and for the paired replay tonight (whose entry condition was a full pass of this gate; the frozen memory guard would in any case have rejected the extra 4.1 GiB of Metal). Consistent with L1's arithmetic: a typical silence final spends ~390 ms in Gemma, so even the best short-sentence gain here is under 100 ms of a 1.4 s median.

## Retained hypothesis (for a separately declared experiment)

Speculative decoding with the E2B draft pays off where finals are long — smart/hard-cut finals and the p95 tail of `translation_call` (1.26–1.29 s at p95 in L1). A future bounded screen should target those endpoints and the p95 guard, not the silence-final median, and must budget ~16 GiB of the 18 GiB unified memory with Parakeet loaded (a live run was not attempted tonight for that reason). Any such run remains opt-in and quality-gated; byte identity here does not replace bilingual review.
