# Phase B — Spike

**Goal:** end-to-end smoke test with the new tooling before committing a full training run.

**Wall clock:** 1–2 days.

## B1. Replicate S6 scale on Gemma 4 E4B

Take the same data S6 used (1,800 verse + 1,800 sermon = 3,600 pairs, no glossary tags, no preference) and run it through the new trainer:

```bash
python training/train_gemma4.py \
    --base unsloth/gemma-4-E4B-it \
    --verse-pairs hybrid_runs/data/verse_1800.jsonl \
    --sermon-pairs hybrid_runs/data/sermon_1800.jsonl \
    --output fine_tuned_gemma4_e4b_spike \
    --max-steps 200
```

Expected wall clock on A2000 Ada with packing: **~30–45 min** (faster than S6's TranslateGemma run because Unsloth + packing).

## B2. Merge → GGUF → serve

```bash
python training/export_gguf.py \
    --adapter fine_tuned_gemma4_e4b_spike \
    --base unsloth/gemma-4-E4B-it \
    --output models/gemma-4-e4b-it-q4km-spike.gguf

# Start a test llama-server on a non-production port
~/llama.cpp/build/bin/llama-server -m models/gemma-4-e4b-it-q4km-spike.gguf --port 8092 &
```

## B3. Run benchmark in compare mode

```bash
python scripts/benchmarks/bench_translate_t1_t4.py \
    --config t3_spike --server-url http://localhost:8092 \
    --n-sermon 50 \
    --out metrics/spike_e4b.json

python scripts/benchmarks/bench_translate_t1_t4.py \
    --config merge \
    --inputs metrics/phase1a_t3.json metrics/spike_e4b.json \
    --out metrics/spike_diff.json
```

## Pass criteria (any failure = stop and debug, do not advance to Phase C)

- [ ] **Trainer doesn't crash.** Loss settles between 1–3 (NOT 13–15).
- [ ] **GGUF export succeeds** and `llama-gguf-tool show` reports valid `chat_template` with no thinking-mode artifacts.
- [ ] **Q4_K_M ↔ bf16 in-process delta** ≤ 0.3 COMET on the 8 canary sentences.
- [ ] **Latency unchanged** vs. stock E4B Q4_K_M baseline (within ±10%). If it regresses, the GGUF metadata or chat template is wrong.
- [ ] **Benchmark numbers are sensible** — even with only 3,600 pairs and no preference, expect either a small COMET gain or no regression. A large negative delta (> −1 COMET) indicates a prompt format mismatch — abort and recheck.

## What this phase explicitly does NOT need to prove

- It does not need to beat the v1 ship target. That's Phase C.
- It does not need full sermon-tier numbers; 50-chunk sub-sample is fine for plumbing verification.
- It does not need a passing canary 7/8. The goal is "end-to-end pipeline works", not "model is good".
