# Compiled Parakeet decoder: negative development result

The isolated compiled decoder/joint candidate is rejected. All 18 original-control versus joint-evaluation outputs remain exact. The compiled variant matches 12/18 full outputs; the other six comparisons differ only in confidence and derived `avg_logprob` values. Text, token IDs, token/sentence timestamps and durations are identical in all 18 triplets. The predeclared gate required exact confidence too; it is not relaxed after seeing this result.

The same two source recordings differ in every repeat: `fleurs-en-development-1603-561237249582086895` and `fleurs-es-development-1624-8750634895258922350`. Each has one changed token confidence, with corresponding sentence/STT confidence and derived log-probability changes. The largest absolute difference is 0.000000059604644775390625. Full field paths and before/after values are in [the derived receipt](compile-review.json).

| Actual arm | Calls | Call-wall p50 | Call-wall p95 |
|---|---:|---:|---:|
| control | 18 | 189.398 ms | 228.958 ms |
| joint_eval | 18 | 169.681 ms | 231.517 ms |
| compiled_joint | 18 | 174.284 ms | 930.462 ms |

These are 18 triplets / 54 calls over three development recordings per language and three alternating repeats, with cold calls included. Percentiles are descriptive median/nearest-rank values for this small sample, not a production p95 certification. Median paired savings versus the uncompiled joint-evaluation control are -0.978 ms (-0.564%): the compiled candidate is slightly slower on that comparison. It therefore adds no measured median benefit to the earlier exact joint-evaluation candidate.

The first compiled STT call takes 930.462 ms. The separate first-signature dispatch is 0.313 ms and wrapper creation is 0.005 ms; those exclude deferred device execution and are **not compiler-exclusive cost estimates**. All 834 steady-state compiled step calls ran; 78 bootstrap steps stayed eager under the declared contract. One shape signature was used. The raw warm-only subset remains separate and still fails exactness; it cannot replace the primary cold-inclusive result. Process memory high-water values include all arms and model loading.

[The unchanged raw receipt](compile-development-r1.json) has SHA-256 `9f73bcda37e3ac8857a531e6f3a51149dc856478f34bf348313454a623414fa9`. Its nested reused joint-assessor field says `compile_experiment: not implemented; conditional follow-up only`; that is a stale inherited label. The outer `compilation_contract`, source hash, `compile_stats`, and `control` / `joint_eval` / `compiled_joint` call records describe the experiment actually executed. No raw value or label was rewritten.

The tensor-only decoder/joint graph was compiled in the isolated helper, with hidden/cell state passed explicitly and greedy decisions outside the graph. Upstream files and production engines remain unchanged. No runtime default, integration, human approval, or training eligibility changes follow.
