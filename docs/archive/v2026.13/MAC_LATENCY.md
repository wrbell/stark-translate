# v2026.13 — Mac inference latency program (M3 Pro 18 GB, MLX)

> **Status:** in progress (2026-09-09). Numbers below are measured on the M3 Pro with
> `tools/benchmark_mlx_accel.py` (isolated) and `tools/replay_bench.py` (live pipeline
> replaying real sermon audio). Every claim links to the PR that produced it.

## 0. What was wrong (three production bugs on `main`)

All three made **every live Mac final fail or run long**; together they explain the
empty 2026-08-30 session CSVs. Found with the replay harness (#182).

| # | Bug | Effect | Fix |
|---|---|---|---|
| #172 | TranslateGemma-era EOS patch applied to Gemma 4: `convert_tokens_to_ids("<end_of_turn>")` → `<unk>` (3), then `tokenizer._eos_token_ids = {1, 3}` — dropped `<turn|>`=106 | every Gemma 4 call ran to `max_tokens` (~2.2 s), emitted `<turn|><|channel>thought…` junk | `ensure_stop_tokens()` (#181) |
| #181 | first Gemma forward pass on a pool worker (warm-up / first chunk) with mlx 0.32.2 thread-local streams | `RuntimeError: There is no Stream(gpu, 1) in current thread` on every final | `warm_mlx_model()` on the load thread (#181) |
| #184 | `qe_b` only assigned in A/B branches | `UnboundLocalError` after every single-model final | prologue init (#184) |

## 1. Isolated translation bench (`benchmark_mlx_accel.py --quick`, 3 runs, E4B OptiQ-4bit)

| Length (words) | before p50 | after #181 p50 | tokens | finish |
|---|---|---|---|---|
| short (6) | 2139 ms | **473 ms** | 8 | stop |
| medium (21) | 2176 ms | **1393 ms** | 36 | stop |
| long (33) | 3242 ms | **1677 ms** | 43 | stop |

Decode ≈ 33 tok/s, prefill ≈ 90–125 ms, TTFT ≈ 230–330 ms. E2B OptiQ after fix:
306 / 860 / 1000 ms at ~62 tok/s, canary 11/18 vs E4B 13/18 (its earlier "0/8" was the
same bug plus thinking-on, not PLE garbage). health_check 8-slice: 6/8 both before and after
(Santiago / partimiento lexical gaps).

## 2. Live pipeline replay (`replay_bench.py`, real sermon clips, VAD → partials → finals)

Clips: 150 s of `Gospel_Message_(12_14_25)` @ 20:30 and `Gospel_Message_(2_8_26)` @ 19:30
(speech-dense windows chosen with Silero VAD; the 10-minute mark is a hymn and trips the
music hold). `e2e_latency_ms` = speech end → final on display. `true_e2e_ms` also counts the
utterance's own duration (up to 8 s) and is not the user-facing number.

| Clip | metric | pre-fix (main + #181 warm only) | post #181 + #184 | Δ |
|---|---|---|---|---|
| 12_14_25 (26 finals) | e2e p50 / p95 | 4803 / 8939 | **4016 / 7944** | −16 % |
| | STT p50 (whisper turbo) | 1207 | 1179 | −2 % |
| | Gemma p50 (tokens, TTFT) | 2464 | 2147 (21 tok, TTFT 605 ms) | −13 % |
| | Marian HF partial p50 | 380 | 335 | |
| | partial total p50 | 1079 | 1074 | |
| 2_8_26 (30 finals) | e2e p50 / p95 | 3742 / 6641 | **2858 / 4365** | −24 % |
| | STT p50 | 982 | 999 | |
| | Gemma p50 (tokens, TTFT) | 1556 | 1202 (16 tok, TTFT 314 ms) | −23 % |
| | Marian HF partial p50 | 345 | 281 | |

**Why the live gain is smaller than the bench gain:** under load the GPU is shared with
whisper partials that re-transcribe the growing buffer every 0.6 s (STT p50 ≈ 1.0–1.2 s for
5–8 s utterances). Gemma decodes at 14–16 tok/s instead of 33 and TTFT doubles. STT is now
the biggest lever, not the translator.

## 3. Gemma 4 MTP drafter ("MTS") — `tools/mts_acceptance_probe.py` (#183, #177)

E4B OptiQ + `gemma-4-e4b-it-assistant-bf16` through mlx-optiq 0.4.34 `runtime.spec`, 3 runs,
8 canaries + 3 sentences. Output byte-identical to greedy in 33/33 cases.

| γ | p50 | tok/s | acceptance | vs greedy (short / medium / long) |
|---|---|---|---|---|
| 1 | 655 ms | 20.9 | **31.3 %** | 453 / 1339 / 1440 ms (≈0 / −4 % / −14 %) |
| 2 | 645 ms | 21.1 | 21.6 % | |
| 3 | 670 ms | 20.3 | 15.2 % | |

Not a win at 31 % acceptance (llama.cpp gets 70–87 % on the same drafter). RoPE-offset
patch experiment (`docs/mlx_mtp_notes.md` §7 suspect 1: `position = offset` and mask
`first_cached = position - k_len`): acceptance unchanged at 31.3 % / 21.6 % (γ=1/2), latency
slightly worse → suspect 1 rejected. Next suspects are the 4-bit target hidden states vs the
bf16-trained drafter (test against `gemma-4-e4b-it-bf16`) and batched-verify drift. `mlx_mts`
stays off; #177 stays open.

## 4. Parakeet TDT 0.6B v3 on MLX (#186, #178)

Real-model smoke: EN 3 s **98 ms**, EN 8 s **149 ms**, ES 7 s 424 ms (2 Cor 1:3 exact),
vs whisper-large-v3-turbo ≈ 505 ms for 3 s. Streaming in 0.6 s chunks ≈ 235 ms/chunk, so
partials keep batch re-transcription.

Live replay A/B (`--stt-backend parakeet-mlx`, same clips, HF Marian partials, `feat/parakeet-live-path`):

| Clip | metric | whisper-turbo (post #181/#184) | **parakeet-mlx** | Δ |
|---|---|---|---|---|
| 12_14_25 (26 finals) | e2e p50 / p95 | 4016 / 7944 | **1631 / 2708** | −59 % |
| | STT p50 | 1179 | **247** | −79 % |
| | Gemma p50 (TTFT, tokens) | 2147 (605, 21) | **1128 (328, 22)** | −47 % — GPU no longer contended |
| | partial total p50 (count) | 1074 (76) | **446 (206)** | −58 %, 2.7× more partials |
| 2_8_26 (30 finals) | e2e p50 / p95 | 2858 / 4365 | **1008 / 1394** | −65 % |
| | STT p50 | 999 | **240** | −76 % |
| | Gemma p50 (TTFT, tokens) | 1202 (314, 16) | **726 (273, 17)** | −40 % |
| | partial total p50 (count) | 917 (63) | **380 (201)** | −59 % |

Speech-end → final is now ≈ 1.0–1.6 s p50 (was 3.7–4.8 s on the 2026-08-30 code). The partial
path is now bounded by HF Marian on CPU (270–300 ms); see §5. WER gate vs whisper on a
Mac-runnable set — `tools/stt_roundtrip_compare.py` (Piper → STT, identical audio for both engines):

| Set | engine | WER mean / median | exact | Tier-1 term recall | p50 / p95 |
|---|---|---|---|---|---|
| EN: 18 canaries + 3 bench sentences + 30 Tier-1 terms (51) | whisper-turbo | 0.002 / 0.000 | 50/51 | 1.00 | 643 / 703 ms |
| | **parakeet-mlx** | 0.006 / 0.000 | 48/51 | 1.00 | **143 / 179 ms** |
| ES: 12 RVR1909 verses + 5 term carriers (17) | whisper-turbo | 0.068 / 0.038 | 7/17 | 1.00 | 625 / 743 ms |
| | parakeet-mlx | 0.093 / 0.077 | 6/17 | 1.00 | 138 / 192 ms |

(The ES references carry the 1909 orthography `á`, which Piper reads as "acentuada", inflating
both engines' WER equally.) **Decision:** `auto` STT on Mac = `parakeet-mlx` for English sessions,
whisper-turbo for Spanish; `--stt-backend mlx` forces Whisper. A human-verified church-audio WER
set (or the WSL 41-clip manifest) is still owed before calling this final.

## 5. Marian CT2 on Mac

`scripts/convert_marian_ct2.py` (int8) for en-es and es-en → `adapters/marian_ct2/*/active`
(sanity 6/8 each, same as HF). ctranslate2 4.7.1 arm64 links Accelerate only (no libomp),
so the historical PyTorch/libomp crash does not apply. Factory now downgrades
`int8_float16` → `int8` off-CUDA (CT2 rejects int8_float16 on CPU) and caps `intra_threads` at 4
off-CUDA: with the default thread count (12) CT2 saturated the CPU at 0.4 s partial cadence and
starved VAD/asyncio/MLX dispatch (run D below: TTFT p95 2.4–5.1 s, e2e p95 6–7 s).

| Run (parakeet STT) | Marian | threads | VAD | 12_14_25 e2e p50/p95 | partial p50 | 2_8_26 e2e p50/p95 | partial p50 |
|---|---|---|---|---|---|---|---|
| B | HF CPU | — | 0.5 / 0.6 | 1631 / 2708 | 446 | 1008 / 1394 | 380 |
| C | CT2 int8 | default | 0.5 / 0.6 | 1671 / 3836 | **262** | 1266 / 2294 | **195** |
| D | CT2 int8 | default | 0.35 / 0.4 | 2319 / 6076 | 264 | 1820 / 7368 | 221 |
| E (**E2B** finals) | CT2 int8 | 4 | 0.35 / 0.4 | **1208 / 2274** | **213** | **863 / 2115** | **162** |
| F (E4B) | CT2 int8 | 4 | 0.35 / 0.4 | 2349 / 4175 | 246 | 1271 / 2710 | 169 |
| **G (E4B, shipped default)** | CT2 int8 | 4 | 0.5 / 0.6 | **1544 / 2576** | **237** | **1222 / 2851** | **191** |

Marian partial p50 dropped 270–300 ms (HF) → 52–94 ms (CT2). **VAD verdict:** the 0.4 s partial
cadence (D/F) adds ~30 % more Parakeet calls and pushes E4B TTFT/e2e up (GPU contention); the 0.35 s
silence trigger only helps utterances that end by silence (clip 2 silence_delay −450 ms). Defaults stay
0.5 s / 0.6 s. E2B finals (canary 11/18 vs E4B 13/18) tolerate the faster cadence and are the
fast-mode option (`--gemma4-size e2b`: 863–1208 ms e2e p50).

## 6. Reproduce

```bash
python tools/benchmark_mlx_accel.py --configs e4b,e2b --no-stt --quick
python tools/replay_bench.py --prepare --seconds 150 --offset 1290   # then edit manifest offsets per clip
python tools/replay_bench.py --configs 'whisper=' 'parakeet=--stt-backend parakeet-mlx' --tag <tag>
STARK_RUN_GPU_TESTS=1 pytest tests_gpu -q
python tools/mts_acceptance_probe.py --gammas 1,2,3 --runs 3
```
