# L-A — where the time still goes on the promoted runtime (2026-09-11)

**Verdict: all three candidate arms have a measured mechanism and were admitted to the L-B screen.** The STT side
is not the tail on these clips; the Gemma decode is, and roughly a third of it runs concurrently with preview STT.

Inputs: the six traced control runs of `tail_screen_20260911` (clips A and B, `main` `6f160bb`) plus three new traced
control replays on clip A with the P2-H `cpu_ms` fields (`la0912_A_ctl_r{0,1,2}`, `main` `a46649e`), all in the
promoted `venv`; `tools/stt_overlap_attribution.py` (PR #214) over the nine diagnostics files
([`attribution.md`](attribution.md), [`attribution.json`](attribution.json)); the isolated E4B text bench
([`textbench_e4b.json`](textbench_e4b.json)); 5 s process CPU samples of the pipeline during the three new replays
([`raw/cpu_*.csv`](raw/)).

## Pooled attribution (9 runs, 606 finals, 480 Gemma-routed)

| stage | bucket | count | overlapped by another chunk's Gemma decode (median share) | overlapped by partial STT (median share) |
|---|---|---:|---:|---:|
| final STT | slow (> 800 ms) | 15 | 0.00 | 0.37 |
| final STT | normal | 591 | 0.00 | 0.00 |
| Gemma decode | slow (> 800 ms) | 363 | — | 0.33 |
| Gemma decode | normal | 117 | — | 0.00 |

- **34.6 % of all Gemma decode time is overlapped by partial STT**; 613 of 4815 partial STT calls
  started while a translation was active (≈ 68 per 360 s run). Dispatch-time suppression cannot catch a partial
  that was already queued when the final's translation began.
- Final STT is rarely slow (15/606) and never overlaps another chunk's Gemma decode on these clips;
  the `[P7-6C]` overlap the pipeline designs for happened 3–5 times per run.
- Endpoint mix (pooled): silence 330 (p50 / p95 1565.6 / 2303.6 ms, tokens p95 34),
  smart_cut 240 (3084.7 / 6923.1 ms, tokens p95 41), hard_cut 36
  (2031.9 / 3245.4 ms, tokens p95 43). Smart cuts are 40 % of finals and cost +1.5 s at the median.

## Isolated vs live Gemma throughput

Isolated (text bench, no STT, `venv`, Metal peak 6.4 GiB): 33.0 tok/s at 36 tokens (medium), 32.6 at 43 (long),
33–37 on 8–19-token canaries; decode p50 1,117 ms for 36 tokens. **Live** Gemma-routed finals on clip A
(`la0912` traced controls, n=159): **29.2 tok/s p50, 33.2 p95** — about 13 % below isolated at the median. The
MLX host thread spends 40 % of a core during decode (`cpu_ms/elapsed` p50 0.40); the pipeline process as a whole
averaged 59.8 %, 65.0 %, 70.8 % of one core over the three replays with p95 115.2 %, 116.7 %, 143.4 % and peaks of
140.2 %, 140.5 %, 395.8 % — i.e. more than one core at the tail while Marian CT2 previews (4 intra-threads), Silero VAD and the
host thread share the P-cores. Partial STT calls take 155 ms p50 / 380 ms p95 and use little CPU (0.2 of a core).

## Admission decisions (criteria declared in the plan)

| arm | criterion | measured | verdict |
|---|---|---|---|
| `marian_threads_2` | live tok/s ≥ 15 % below isolated **or** process CPU > ~1 core while previews run | −13 %; p95 115–143 % of a core | **admitted** (CPU criterion) |
| `max_utterance_6` | cuts ≥ 20 % of finals and p50 ≥ +500 ms over silence finals | 40 %; +1,519 ms | **admitted** |
| `partial_recheck_translation` | ≥ 25 % of slow Gemma decode time overlapped by partial STT | 33 % (slow median), 34.6 % overall | **admitted** |

Not admitted / not a hypothesis: STT-side arms (final STT is not the tail); the allocator knob (already screened
and rejected; see the registry). Overlap is observational, not causal — that is what the screen tests.

This is engineering attribution on machine-timed replays; it certifies no human quality, visible delivery or
production default and changes none.
