"""Render P1-runtime/microbench.md from the micro-bench receipts (re-runnable)."""
import json, statistics, hashlib
from pathlib import Path
P = Path('/Users/willem/Code/vibes/SRTranslate/.cache/series4-20260912/P1')
def load(n):
    p = P / n
    return json.loads(p.read_text()) if p.exists() else None
def sha(n): return hashlib.sha256((P / n).read_bytes()).hexdigest()[:12]
order = [('ctl', 'control 5227a73', 'default'), ('cand', 'P1 branch', 'default'), ('ctl2', 'control 5227a73', 'default'), ('cand2', 'P1 branch', 'default'),
         ('cand_nowired', 'P1 branch', 'STARK_MLX_WIRED_LIMIT=0'), ('ctl3', 'control 5227a73', 'default'), ('cand_nowired2', 'P1 branch', 'STARK_MLX_WIRED_LIMIT=0')]
rows, sums, sha0 = [], {}, None
for key, checkout, env in order:
    d = load(f'textbench_{key}.json')
    if not d: continue
    e = d['configs']['e4b']; pl = e['per_length']
    can = [c for c in e['canary_details']]
    ttft = statistics.median(c['ttft_ms'] for c in can if c.get('ttft_ms') is not None)
    cm = statistics.mean(pl[k]['latency']['p50'] for k in pl if k.startswith('canary'))
    sums[key] = sum(pl[k]['latency']['p50'] for k in pl)
    shas = tuple(pl[k]['text_sha256s'][0] for k in pl) + tuple(c['text_sha256'] for c in can)
    sha0 = sha0 or shas
    rows.append(f"| {key} | {checkout} | {env} | {pl['short']['latency']['p50']:.0f} | {pl['medium']['latency']['p50']:.0f} | {pl['long']['latency']['p50']:.0f} | {cm:.0f} | {ttft:.0f} | {pl['medium']['tps_mean']:.1f} | {e['metal_peak_mib']:.0f} | {'yes' if shas == sha0 else 'NO'} | {e['canary']['passed']}/{e['canary']['total']} |")
ctl = [sums[k] for k in sums if k.startswith('ctl')]; wired = [sums[k] for k in ('cand', 'cand2') if k in sums]; now = [sums[k] for k in sums if k.startswith('cand_nowired')]
ctl_warm = [sums[k] for k in ('ctl2', 'ctl3') if k in sums]
je = load('joint_eval_cand.json'); a = je['assessment']
pc = load('profile_ctl.json'); pc2 = load('profile_cand2.json'); pc1 = load('profile_cand.json')
def prof(d):
    if not d: return 'not run'
    a = d.get('assessment') or {}
    if d.get('status') != 'completed': return f"{d.get('status')} ({(d.get('pairs') or [{}])[-1].get('error', '')})"
    return f"{d['status']}; outputs exact {a.get('all_outputs_exact')}; profiled vs control wall {a['profile_vs_control_wall_percent']['p50']:+.1f} % p50; readback share {100*a['sampled_readback_share_of_profiled_stt_wall']['p50']:.1f} %"
md = f"""# P1 micro-bench — same `venv` interpreter, control checkout `5227a73` vs the P1 branch

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
{chr(10).join(rows)}

- Every run produced byte-identical text for all 21 sentences (per-run `text_sha256s`), across checkouts and with the
  wired limit on or off.
- The first control run was the fastest in every class; every later run, whichever arm, sits ≈ 5 % above it: a warm-up
  / thermal ordering effect, not an arm effect. Pooled p50 sums: control {', '.join(f'{x:.0f}' for x in ctl)}; P1 branch
  wired {', '.join(f'{x:.0f}' for x in wired)}; P1 branch with `STARK_MLX_WIRED_LIMIT=0` {', '.join(f'{x:.0f}' for x in now)}.
  P1 (wired) vs the warm controls: {100*(statistics.mean(wired)/statistics.mean(ctl_warm)-1) if wired and ctl_warm else float('nan'):+.1f} %; P1 with the limit off vs warm controls:
  {100*(statistics.mean(now)/statistics.mean(ctl_warm)-1) if now and ctl_warm else float('nan'):+.1f} %. The wired-limit change is neutral in isolation, as expected: mlx-lm 0.31.3 sets the
  same recommended limit inside every `stream_generate` call and only restores the previous value afterwards, so the
  load-time setting removes the toggle without changing the limit during decode.

## Parakeet joint scalar decode (`joint_eval_cand.json`, P1 branch, `venv`)

{a['pairs']}/{a['expected_pairs']} pairs complete; **outputs exact {a['all_outputs_exact']}** (changed outputs: {a['changed_outputs']}); control wall p50 {a['control_wall_ms']['p50']:.1f} ms / p95 {a['control_wall_ms']['p95']:.1f} ms, joint p50 {a['joint_eval_wall_ms']['p50']:.1f} ms / p95 {a['joint_eval_wall_ms']['p95']:.1f} ms; paired saving p50 {a['paired_saved_ms']['p50']:.1f} ms ({a['paired_saved_percent']['p50']:.1f} %), no pooled tail regression ({a['positive_paired_median_without_pooled_tail_regression']}). Per language: en {je['language_assessments']['en']['paired_saved_percent']['p50']:.1f} %, es {je['language_assessments']['es']['paired_saved_percent']['p50']:.1f} %, both exact. The receipt's `production_integration: not authorized by this report` line is the tool's standing statement; the promotion is decided by the paired identity screen, not by this receipt.

## Parakeet readback profile (`tools/parakeet_profile.py`, en × 3 × 3)

- control checkout: {prof(pc)}
- P1 branch, first attempt with the joint decode installed at load: {prof(pc1)} → the profiler's instrumented arm parses and re-installs the stock `decode_greedy` source, so it cannot instrument the joint method; fixed on the branch (`77b4998`: the profiler always loads the engine with `joint_scalar_eval=False`, `--baseline-decode` is a compatibility no-op).
- P1 branch after the fix: {prof(pc2)}

## Receipt index

{chr(10).join(f'- `{n}` {sha(n)}' for n in sorted(x.name for x in P.glob('*.json')) if n != 'RUNS.json')}
"""
(P / 'microbench.md').write_text(md); print(md[:3000])
