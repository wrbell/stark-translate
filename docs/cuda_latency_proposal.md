# CUDA inference-latency proposal (A2000 Ada 16 GB / WSL2)

> **Status:** proposal + ready-to-run scripts. **Not executed on hardware** (unchanged as of
> 2026-09-10 — the WSL box has not run a job since 2026-04-30; backlog `cuda-latency-proposal`).
> **Audience:** next WSL session on the RTX A2000 Ada 16 GB box (CUDA 12.6).
> **Pin:** llama.cpp tag **`b10883`** (2026-09-09) in `start_server.sh`, `Dockerfile` and `tools/llama_runtime.py`.
> **Policy:** Google (Gemma 4 + official assistants), NVIDIA (Parakeet), OpenAI Whisper, Helsinki-NLP opus-mt only.
> **Scope:** the A2000 (Ada, sm_89) CUDA path only. It is **not** the RTX 2070 Lite target
> (`lite-cuda-8gb`, Turing sm_75, E2B only — [`lite_profiles.md`](./lite_profiles.md)) and
> not the Mac: the MLX live path rejects `--mts` before any model loads
> (`validate_live_mts` in `dry_run_ab.py`; MTP stays an offline experiment,
> [`mlx_mtp_notes.md`](./mlx_mtp_notes.md), #177). Every number in §0 is a v2026.5–9
> archive measurement with its source column; §1–§6 figures are arithmetic projections.

Scripts (all header-marked unexecuted): `scripts/cuda/build_llamacpp.sh`, `convert_gemma4_assistant_gguf.sh`, `bench_mtp.sh`, `retest_flash_attn.sh`. Client changes in `engines/` / `dry_run_ab.py` are specified here, **not** implemented. Server-side switches that do exist: `start_server.sh --no-draft` (default), `--mtp` (opt-in, `SPEC_N=3`, f16 KV), `--flash-attn`, `--e2b-draft` (legacy loss); Docker `STARK_LLAMA_MTP=1` (`docker/entrypoint.sh`).

---

## 0. Current production (measured)

| Path | p50 | p95 | tok/s | Canary | VRAM | Source |
|------|-----|-----|-------|--------|------|--------|
| E4B Q4_K_M final (llama.cpp b9022, `-ctk/-ctv q8_0`, **no FA**) | 473 ms | 610 ms | 42.8 | 7/8 | 4.75 GB | [v2026.9](archive/v2026.9/GEMMA_OPTIM_PHASE2.md) |
| E2B Q4_K_M | 263 ms | 375 ms | ~70 | 6/8 | 3.31 GB | same |
| T4 E2B→E4B spec (`-md` `--draft 16`) | 559 ms | 1023 ms | 36.4 | 7/8 | 8.52 GB | [v2026.5 T4 vs T3](archive/v2026.5/BENCHMARK.md) |
| STT W16 CT2 `int8_float16` | 353 ms | 413 ms | — | WER 11.00% / tier-1 8.70% | 2.65 GB | [v2026.7](archive/v2026.7/STT_BENCHMARK.md) |
| STT HF fp16 turbo (off-the-shelf, not W16) | 193 ms | 261 ms | — | WER 14.07% | 4.59 GB | same |
| Marian CT2 partial | 57 ms | 116 ms | — | 14/16 | 1.58 GB | [v2026.8](archive/v2026.8/MARIAN_BENCHMARK.md) |
| **Live partial ≈ STT+Marian** | **~410 ms** | | | | | 353+57 |
| **Live final ≈ STT+E4B** | **~820 ms** | | | 7/8 | ~7.4 GB | 353+473 |

Launch today (until this patch): `start_server.sh` L85–86 `-ngl 999 -c 512 -ctk q8_0 -ctv q8_0` and, **if** `models/gemma-4-e2b-it-q4km.gguf` exists, `-md <E2B> --draft 16 --draft-min 5` (issue #173). Dockerfile still pinned `b8783` vs script `b9022`. `-fa` omitted (b8782 E4B **+56%** p50). Gemma 4 SWA → llama-server re-prefills every request (~50 ms). IQ4_XS / imatrix rejected (canary).

This patch: **default `NO_DRAFT=true`** (kills the T4 loss), **`--mtp` opt-in**, pin **`b10883`**.

---

## 1. Gemma 4 MTP drafter (rank 1 — only ~2× lever)

**What / why.** Official Google assistant is a ~77 M 4-layer MTP head that cross-attends the target KV — not a second E2B. T4 E2B draft was a **single-GPU loss** (559 vs 469 ms). MTP is the architecture Google trained for this. llama.cpp: [#23398](https://github.com/ggml-org/llama.cpp/pull/23398) merged 2026-06-07 (31B/26B); [#24282](https://github.com/ggml-org/llama.cpp/pull/24282) 2026-06-08 (E2B/E4B); [#28183](https://github.com/ggml-org/llama.cpp/pull/28183) 2026-09-01 (`gemma4-assistant` load fix). Flags ([speculative.md](https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md), [server README](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)):

```text
--spec-type draft-mtp --spec-draft-n-max N -md <assistant.gguf>
# optional multi-GPU: --spec-draft-device CUDA0
```

**Quantized KV (`-ctk/-ctv q8_0`) → ~0% draft acceptance** ([#23398](https://github.com/ggml-org/llama.cpp/pull/23398) thread: Q8 KV 0 accepted / 1090 generated; f16 KV restores ~55–60%). `--mtp` **drops q8 KV** (llama.cpp default **f16**). At `-c 512` V-cache was ~5 MiB even f16 ([v2026.9](archive/v2026.9/GEMMA_OPTIM_PHASE2.md)) — latency cost of f16 KV is noise; acceptance is not.

**Expected gain (arithmetic, not measured).** E4B wall 473 ms ≈ **~50 ms SWA re-prefill + ~423 ms decode** at 42.8 tok/s (~19 completion tokens). Upstream 31B Q8 mtp-bench in [#23398](https://github.com/ggml-org/llama.cpp/pull/23398): **40.0 → 83.3 tok/s (1.97×) at 57.65% accept, n-max=4**. E-series is smaller (drafter ~149 MiB F16 / ~40–80 MiB Q4_0; [PR #22738](https://github.com/ggml-org/llama.cpp/pull/22738) llama-bench) so draft is cheap; E4B accept on Hexagon was **47.9%** at n-max=3 ([#24282](https://github.com/ggml-org/llama.cpp/pull/24282)); M1 Pro E2B **71.6%** at n-max=3 ([#22735](https://github.com/ggml-org/llama.cpp/discussions/22735)). Band:

| Accept / n-max | Decode speedup | Decode ms | +50 ms prefill | **E4B p50** |
|----------------|----------------|-----------|----------------|-------------|
| ~48% (E-series floor) | ~1.5–1.6× | ~260–280 | 310–330 | **~310 ms** |
| ~57% (31B n-max=4 analog) | ~1.9–2.1× | ~200–220 | 250–270 | **~250–270 ms** |

**Target: E4B final p50 ~250–300 ms** (live final ~600–650 ms with W16 STT, ~440–500 ms if HF-W16 STT also lands). Drafter VRAM ≈ +0.05–0.15 GB.

**Commands (WSL):**

```bash
scripts/cuda/build_llamacpp.sh          # LLAMA_CPP_REF=b10883, sm_89
scripts/cuda/convert_gemma4_assistant_gguf.sh   # E4B + E2B, f16 and Q4_0
scripts/cuda/bench_mtp.sh               # t3 baseline, then t3-mtp n-max=2,3,4
# equivalent launch:
./start_server.sh --mtp                 # SPEC_N=3 default; f16 KV; -md assistant Q4_0
```

Checkpoints: [`google/gemma-4-E4B-it-assistant`](https://huggingface.co/google/gemma-4-E4B-it-assistant), [`google/gemma-4-E2B-it-assistant`](https://huggingface.co/google/gemma-4-E2B-it-assistant) ([Google MTP docs](https://ai.google.dev/gemma/docs/mtp/mtp)). Convert recipe from [#24282](https://github.com/ggml-org/llama.cpp/pull/24282): `convert_hf_to_gguf.py … --outtype f16` then `llama-quantize … Q4_0`.

**Gate (adopt `--mtp` as production):**

| Metric | Must |
|--------|------|
| E4B sermon p50 | **≤ 300 ms** (stretch 250) vs 473 ms baseline on **same** 125-chunk + 8-canary harness |
| Canary | **≥ 7/8** (temp 0, `enable_thinking: false` — MTP is lossless if verify is on) |
| Draft accept (server log) | **≥ 0.45** (else n-max is wasting verify) |
| VRAM peak (E4B+assistant+W16 STT+Marian) | **≤ 12 GB** (4 GB headroom on 16 GB) |
| p95 | **≤ 450 ms** (T3 was 610) |

Sweep `SPEC_N=2,3,4`; pick fastest p50 that holds canary + accept. If Q4_0 accept < 0.45, retry **f16** assistant (~149 MiB).

**Risk.** (1) 0% accept if q8 KV accidentally left on. (2) [#24758](https://github.com/ggml-org/llama.cpp/issues/24758) silent MTP disable on OOM — watch for `Gemma4Assistant requires ctx_other` then continue-without-MTP. (3) [#28286](https://github.com/ggml-org/llama.cpp/issues/28286) **draft-mtp + `-np > 1` cross-slot contamination** — keep **`-np 1`** while MTP is on. (4) Pin jump b9022 → b10883 is ~1800 commits; `enable_thinking:false` and Gemma 4 chat template must still work (canary is the tripwire). (5) E-series speedup may be closer to 1.5× than 2×.

**Rollback.** `./start_server.sh --no-draft` (q8 KV, no `-md`). Docker: omit `STARK_LLAMA_MTP`. Keep b9022 tree as `~/llama.cpp-b9022` until gate passes.

---

## 2. `-fa on` retest on b10883 (rank 2 — 5 min, isolate from MTP)

**What / why.** b8782 FA **+56% E4B p50** (478 → 744 ms) on hybrid SWA+global / RoPE-on-K ([v2026.9 Exp B](archive/v2026.9/GEMMA_OPTIM_PHASE2.md)). Never retested on b9022. Upstream since: CUDA FA `d_head=512` + `gqa_ratio=2` fix for Gemma4 MTP ([#24526](https://github.com/ggml-org/llama.cpp/pull/24526), closed E4B FA crash [#24400](https://github.com/ggml-org/llama.cpp/issues/24400)); ROCm reporter saw FA **erase MTP gains**. CUDA Ada is unknown — must A/B **without MTP first**.

**Expected gain.** 0 to −10% p50 if SWA FA kernels landed; 0 if auto-fallback; **+56% if still broken**. Do not bake into the 250–300 ms MTP target.

```bash
scripts/cuda/retest_flash_attn.sh    # t3, FLASH_ATTN=off vs on, no MTP
# then, only if FA ≤ baseline: FLASH_ATTN=on ./start_server.sh --mtp
```

**Gate.** FA-on p50 **≤** FA-off p50 (no regression) **and** canary 7/8 **and** VRAM ≤ baseline+0.2 GB. If MTP+FA accept drops >5 pp vs MTP-only, ship MTP **without** FA.

**Risk / rollback.** Same +56% regression; ROCm-style MTP kill. Rollback: omit `-fa on` (script default `FLASH_ATTN=off`).

---

## 3. STT options (rank 3 — independent of llama.cpp pin)

### 3a. W16 → HF fp16 (`hf_fp16_w16`)

Off-the-shelf HF turbo was **193 ms p50 vs 353 ms CT2 (−45%)** at worse WER (14.07% vs 11.00%) because W16 exists only as CT2 ([v2026.7 finding 4](archive/v2026.7/STT_BENCHMARK.md)). `training/export_ct2.py --keep-intermediate` already writes merged **bf16 HF**; that dir loads with `HFWhisperEngine`.

```bash
python training/export_ct2.py --adapter adapters/whisper_turbo/active \
  --output adapters/whisper_turbo_ct2/active \
  --keep-intermediate --intermediate whisper_hf/W16_mixed_w7
python tools/benchmark_stt_engines.py --variant hf_fp16_w16 \
  --model-id whisper_hf/W16_mixed_w7 --manifest tools/stt_bench_manifest.json
```

**Expected.** If W16 quality transfers: **STT p50 ~193 ms (−160 ms)**; live final **−160 ms** on top of MTP; VRAM **+1.9 GB** (4.59 vs 2.65). Combined with E4B+MTP+Marian ≈ **~9.5–10 GB**.

**Gate.** 41-clip WER **≤ 11.00%** overall **and** tier-1 **≤ 8.70%** **and** p95 **≤ 261 ms** **and** peak VRAM (HF-W16 + E4B+MTP + Marian) **≤ 12 GB**.

**Risk.** Merge-to-HF may not match CT2 WER; fp16 VRAM squeeze with MTP f16 KV. **Rollback:** keep `fw_int8float16_w16` CT2 default.

### 3b. Parakeet TDT 0.6B v3 (EN **and** ES)

[`nvidia/parakeet-tdt-0.6b-v3`](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) is **25 European languages including Spanish** (auto LID) — the `benchmark_parakeet_en.py` “EN-only” framing is outdated. Engine already exists (`engines/parakeet_engine.py`, `--stt-backend parakeet`). ~600 M, CC-BY-4.0, NVIDIA (allow-list).

```bash
# WSL CUDA venv only — do not install on the Mac
pip install 'nemo_toolkit[asr]'
python tools/benchmark_parakeet_en.py --manifest tools/stt_bench_manifest.json
# Also score Spanish clips in the same manifest (do not adopt on EN-only numbers)
```

**Expected.** Unknown on Stark Road audio. Public: Open ASR avg WER 6.34%, LibriSpeech-clean 1.93% (model card). Church/theological WER is the actual question. VRAM likely **< 2 GB**.

**Gate to adopt (either `--lang en` or bilingual).** Same 41-clip harness, **both EN and ES** slices: WER ≤ W16 **and** p95 ≤ 413 ms **and** tier-1 theological WER ≤ 8.70%. Fail any → Whisper stays default.

**Risk.** No W16-style church fine-tune; LID errors on code-switch; NeMo stack weight. **Rollback:** `STARK_STT__BACKEND=faster-whisper`.

---

## 4. Client / server plumbing (rank 4 — implement in a later PR; **do not edit `engines/` here**)

| Change | Where | Expected | Notes |
|--------|-------|----------|-------|
| `max_tokens=dynamic_max_tokens(text)` | `engines/llamacpp_engine.py` | **0–20 ms** typical (EOS already stops); **insurance** vs 256-token runaway | Median completion ~19 ([v2026.5](archive/v2026.5/BENCHMARK.md)); floor 64 / ratio 3.0 already in `translation_prompts.py`. MLX/CUDA HF already call it; llama.cpp does not. |
| `"cache_prompt": true` | request body | **0–50 ms**; likely **~0** | Server default is on ([README](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)). Gemma 4 **SWA re-prefills every request** (~50 ms, [v2026.5](archive/v2026.5/BENCHMARK.md)). Shared prefix is ~20 tok (~20 ms at 1k tok/s prefill) **if** SWA allows reuse — probe via `n_past` in server log. |
| HTTP keep-alive | replace per-request `urllib` with `http.client`/`urllib3` pool | **1–5 ms** on localhost | Handshake is noise vs 473 ms; still drop reconnect jitter on p95. |
| `"n_probs": 0` | request body | **~0** | Default already 0; set explicitly so a future template cannot enable top-N softmax. |
| `-np 2 --cont-batching` | `start_server.sh` | **Do not enable with MTP** | `--cont-batching` is **already default on**. `-np 2` without raising `-c` **halves per-slot context** (512/2=256) and [#28286](https://github.com/ggml-org/llama.cpp/issues/28286) contaminates MTP slots. Live pipeline is STT(N+1) ∥ translate(N) → **one** llama request at a time. Keep **`-np 1`**. |

**Gate (plumbing PR):** canary still 7/8; E4B p50 not worse than post-MTP baseline; no empty `content`.

**Rollback.** Revert the client PR; server flags unchanged.

---

## 5. Not recommended (with reasons)

| Idea | Why not |
|------|---------|
| **E2B drafting E4B** (`-md` `--draft 16`) | Measured **loss** on this GPU (559 vs 469 ms, 8.5 vs 4.9 GB). Default **off** (issue #173). |
| **IQ4_XS / imatrix GGUF** | Canary 7/8 → 5/8 then 6/8 ([v2026.10](archive/v2026.10/IQ4_XS_BENCHMARK.md), [v2026.11](archive/v2026.11/IMATRIX_CALIBRATION.md)). Stay Q4_K_M. |
| **DFlash (z-lab)** | Public drafts for Gemma 4 **12B / 26B / 31B only** ([dflash README](https://github.com/z-lab/dflash/blob/main/README.md)) — **no E4B**. 12B+ does not fit with Whisper on 16 GB. Not in the Google/NVIDIA/OpenAI/Helsinki allow-list. |
| **EAGLE-3** | Merged [#18039](https://github.com/ggml-org/llama.cpp/pull/18039) 2026-06-12; Gemma 4 drafts are **RedHatAI 31B/26B** only ([speculative.md](https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md)). No E4B speculator. |
| **vLLM for E4B** | Gemma 4 heterogeneous heads force `TRITON_ATTN`; **~9 tok/s on RTX 4090** vs ~100+ for Llama-3B ([vllm#38887](https://github.com/vllm-project/vllm/issues/38887)). Google lists E4B **bf16 at 17.9 GB** ([Gemma 4 overview](https://ai.google.dev/gemma/docs/core)) — no room for Whisper on 16 GB. |
| **TensorRT-LLM for E4B** | Examples exist ([TRT-LLM gemma](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/models/core/gemma)) but E4B **OOM on 22 GB L4**; FlashInfer **unsupported on SM89** ([TRT-LLM#14942](https://github.com/NVIDIA/TensorRT-LLM/issues/14942)). A2000 Ada is SM89. Operational cost vs llama.cpp for a church volunteer box. |
| **Gemma 4 12B/26B/31B live + Whisper** | Q4 12B ~7 GB + Whisper ~3 GB + Marian + KV; 26B/31B do not co-reside on 16 GB. |
| **Qwen / DeepSeek / other CN-origin** | Out of policy. |

---

## 6. Combined before / after budget (proposal, not measured)

Decode-only MTP band 1.6–2.0×; STT HF-W16 only if §3a gate holds. Partials stay Marian.

| Config | STT p50 | Marian | E4B p50 | **Live final** | **Live partial** | Canary | VRAM (STT+MT+E4B) |
|--------|---------|--------|---------|----------------|------------------|--------|-------------------|
| **Now (W16 CT2 + E4B, no spec)** | 353 | 57 | 473 | **~820** | **~410** | 7/8 | ~7.4 GB |
| A. MTP n-max=3, f16 KV, W16 CT2 | 353 | 57 | **250–310** | **~600–660** | ~410 | 7/8 | ~7.6 GB |
| B. A + FA if gate | 353 | 57 | 250–310 (−0–10%) | ~590–650 | ~410 | 7/8 | ~7.6 GB |
| C. B + HF-W16 (if WER gate) | **~193** | 57 | 250–310 | **~440–500** | **~250** | 7/8 | ~9.5 GB |
| Plumbing (keep-alive + dynamic max) | — | — | −0–20 ms | noise | noise | 7/8 | — |

**Success bar for the WSL session:** land **row A** (MTP) with canary 7/8. Rows B–C are extras. Do not ship C if WER regresses.

---

## 7. WSL execution checklist

Run after or instead of [wsl_pipeline_refresh.md](./wsl_pipeline_refresh.md) §§1–6. GPU-only; do not run these on the Mac.

1. **Env.** `source ~/stt_train_env/bin/activate`; `nvidia-smi` (A2000 Ada, 16 GB, sm_89); `cd` repo.
2. **Pin + build.** `LLAMA_CPP_REF=b10883 scripts/cuda/build_llamacpp.sh` → `llama-server`, `llama-quantize`, `llama-bench`. Keep `~/llama.cpp-b9022` as rollback.
3. **Baseline (new binary, old flags).** `./start_server.sh --no-draft` (q8 KV, no draft). `python scripts/benchmarks/bench_translate_t1_t4.py --config t3 --server-log /tmp/llama_t3.log --n-sermon 125 --out metrics/cuda_lat_t3.json`. Expect ~473 ms / 7/8 — if canary or p50 blows up, **stop** (pin regression).
4. **Assistants.** `scripts/cuda/convert_gemma4_assistant_gguf.sh` → `models/gemma-4-e{2,4}b-it-assistant-{f16,q4_0}.gguf`.
5. **MTP sweep.** `scripts/cuda/bench_mtp.sh` (n-max 2/3/4, f16 KV, Q4_0 assistant). Record p50/p95/tok/s/canary/VRAM/accept from `parse_server_timings` (`draft acceptance = …`). Pick winner. If accept < 0.45, rerun with **f16** assistant.
6. **FA.** `scripts/cuda/retest_flash_attn.sh`. Keep FA only if §2 gate.
7. **STT (optional, separate GPU occupancy).** Export HF W16 (`--keep-intermediate`) → `--variant hf_fp16_w16`. Parakeet EN+ES on the 41-clip manifest. Adopt only on §3 gates.
8. **Plumbing.** Separate PR (`engines/llamacpp_engine.py`): `dynamic_max_tokens`, keep-alive, `n_probs: 0`, `cache_prompt` probe. **Not this patch.**
9. **Adopt.** If §1 gate: production `./start_server.sh --mtp` (or Docker `STARK_LLAMA_MTP=1`). Else stay `--no-draft`.
10. **Rollback.** `--no-draft`; if pin is bad, `LLAMA_DIR=~/llama.cpp-b9022` and revert `LLAMA_CPP_REF`.

---

## Sources (upstream)

| Claim | URL |
|-------|-----|
| MTP merge 2026-06-07, flags, 0% accept on q8 KV, ~2× dense | https://github.com/ggml-org/llama.cpp/pull/23398 |
| E2B/E4B assistant convert (`f16`→`Q4_0`), n-max=3, 47.9% accept | https://github.com/ggml-org/llama.cpp/pull/24282 |
| `gemma4-assistant` fix 2026-09-01 | https://github.com/ggml-org/llama.cpp/pull/28183 |
| Nightly **b10883** (2026-09-09) | https://github.com/ggml-org/llama.cpp/releases/tag/b10883 |
| `--spec-type draft-mtp` / EAGLE-3 / DFlash types | https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md |
| `-np`, `--cont-batching` (default on), `--cache-prompt`, `n_probs`, `--spec-draft-device` | https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md |
| 31B Q8 40.0→83.3 tok/s @ 57.65% (n-max=4) | https://github.com/ggml-org/llama.cpp/pull/23398 (mtp-bench in PR) |
| M1 Pro E2B accept 71.6% n-max=3 | https://github.com/ggml-org/llama.cpp/discussions/22735 |
| Official assistants | https://huggingface.co/google/gemma-4-E4B-it-assistant · https://ai.google.dev/gemma/docs/mtp/mtp |
| EAGLE-3 (31B/26B only) | https://github.com/ggml-org/llama.cpp/pull/18039 |
| DFlash Gemma 4 sizes | https://github.com/z-lab/dflash/blob/main/README.md |
| FA crash fix / MTP+FA regression report | https://github.com/ggml-org/llama.cpp/pull/24526 |
| MTP + `-np>1` contamination | https://github.com/ggml-org/llama.cpp/issues/28286 |
| Silent MTP drop on OOM | https://github.com/ggml-org/llama.cpp/issues/24758 |
| vLLM E4B ~9 tok/s | https://github.com/vllm-project/vllm/issues/38887 |
| TRT-LLM E4B OOM / SM89 FlashInfer | https://github.com/NVIDIA/TensorRT-LLM/issues/14942 |
| E4B bf16 17.9 GB | https://ai.google.dev/gemma/docs/core |
| Parakeet TDT v3 multilingual (incl. `es`) | https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3 |
| Ada = CC 8.9 → `CMAKE_CUDA_ARCHITECTURES=89` | https://docs.nvidia.com/cuda/ada-tuning-guide/index.html |
| `cache_prompt` semantics | https://github.com/ggml-org/llama.cpp/discussions/13606 |
