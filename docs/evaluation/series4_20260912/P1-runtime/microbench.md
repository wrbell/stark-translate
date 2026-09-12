# P1 micro-bench — same `venv` interpreter, control checkout `5227a73` vs the P1 branch

Run 2026-09-11T23:57Z → 2026-09-12T00:15Z on the idle GPU, one process at a time, HF offline. Text bench:
`tools/benchmark_mlx_accel.py --configs e4b --sentences all --runs 5 --warmup 2 --no-stt` (Gemma 4 E4B OptiQ, 3 length
classes + 18 theological canaries, 5 timed runs each); the bench calls the engine's non-streaming path, so of the four
P1 changes only the load-time wired limit is exercised here. Parakeet: `tools/parakeet_joint_eval.py` (stock engine
control vs the qualified joint method, FLEURS development items, en+es × 3 × 3 repeats) and `tools/parakeet_profile.py`
(readback profile of the stock decode). Receipts: `textbench_*.json`, `joint_eval_cand.json`, `profile_*.json`
(SHA256 prefixes in the index below).

## Text bench (p50 ms per sentence class; canary mean of 18 p50s; ttft = canary TTFT median)

| run | checkout | env | short | medium | long | canary mean | ttft | tok/s (medium) | Metal peak MiB | outputs identical to run 1 | canaries |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ctl | control 5227a73 | default | 525 | 1408 | 1610 | 699 | 330 | 33.8 | 6383 | yes | 13/18 |
| cand | P1 branch | default | 548 | 1438 | 1647 | 743 | 325 | 33.0 | 6383 | yes | 13/18 |
| ctl2 | control 5227a73 | default | 532 | 1428 | 1652 | 745 | 326 | 33.5 | 6383 | yes | 13/18 |
| cand2 | P1 branch | default | 519 | 1455 | 1647 | 744 | 326 | 32.4 | 6383 | yes | 13/18 |
| cand_nowired | P1 branch | STARK_MLX_WIRED_LIMIT=0 | 536 | 1428 | 1648 | 748 | 335 | 33.5 | 6383 | yes | 13/18 |
| ctl3 | control 5227a73 | default | 535 | 1435 | 1653 | 749 | 329 | 33.3 | 6383 | yes | 13/18 |
| cand_nowired2 | P1 branch | STARK_MLX_WIRED_LIMIT=0 | 533 | 1432 | 1657 | 744 | 325 | 33.2 | 6383 | yes | 13/18 |

- Every run produced byte-identical text for all 21 sentences (per-run `text_sha256s`), across checkouts and with the
  wired limit on or off.
- The first control run was the fastest in every class; every later run, whichever arm, sits ≈ 5 % above it: a warm-up
  / thermal ordering effect, not an arm effect. Pooled p50 sums: control 16133, 17026, 17104; P1 branch
  wired 17014, 17018; P1 branch with `STARK_MLX_WIRED_LIMIT=0` 17070, 17009.
  P1 (wired) vs the warm controls: -0.3 %; P1 with the limit off vs warm controls:
  -0.2 %. The wired-limit change is neutral in isolation, as expected: mlx-lm 0.31.3 sets the
  same recommended limit inside every `stream_generate` call and only restores the previous value afterwards, so the
  load-time setting removes the toggle without changing the limit during decode.

## Parakeet joint scalar decode (`joint_eval_cand.json`, P1 branch, `venv`)

18/18 pairs complete; **outputs exact True** (changed outputs: []); control wall p50 180.0 ms / p95 216.4 ms, joint p50 162.1 ms / p95 192.0 ms; paired saving p50 17.6 ms (9.6 %), no pooled tail regression (True). Per language: en 9.8 %, es 9.1 %, both exact. The receipt's `production_integration: not authorized by this report` line is the tool's standing statement; the promotion is decided by the paired identity screen, not by this receipt.

## Parakeet readback profile (`tools/parakeet_profile.py`, en × 3 × 3)

- control checkout: completed; outputs exact True; profiled vs control wall -0.5 % p50; readback share 11.4 %
- P1 branch, first attempt with the joint decode installed at load: failed_pairs (ValueError: Expected one undecorated installed function) → the profiler's instrumented arm parses and re-installs the stock `decode_greedy` source, so it cannot instrument the joint method; fixed on the branch (`77b4998`: the profiler always loads the engine with `joint_scalar_eval=False`, `--baseline-decode` is a compatibility no-op).
- P1 branch after the fix: completed; outputs exact True; profiled vs control wall +0.3 % p50; readback share 11.5 %

## Receipt index

- `joint_eval_cand.json` 469e20523993
- `microbench_compare.json` 9fce3b1df2d1
- `profile_cand.json` a76a777591be
- `profile_cand2.json` 365576009f04
- `profile_ctl.json` 128eec3e7d74
- `protocol.json` cefbfa12a2ba
- `textbench_cand.json` 06ebcdc4cb8d
- `textbench_cand2.json` 7f1b5834c2fb
- `textbench_cand_nowired.json` 09e87f1bb98f
- `textbench_cand_nowired2.json` a264e1cea425
- `textbench_ctl.json` ffdd6ee09ffd
- `textbench_ctl2.json` 1d65b35fd7f2
- `textbench_ctl3.json` 00620e6e3257
