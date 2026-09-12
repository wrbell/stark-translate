"""Per-run / per-arm screen metrics beyond tail_screen_report: first-token and payload-ready per route and endpoint,
generation lock wait, final STT, previews, Metal peak, reuse counters. Usage: screen_metrics.py RUNS.json OUT.json"""
import json, math, sys, statistics
from pathlib import Path
from collections import defaultdict
runs_path, out_path = Path(sys.argv[1]), Path(sys.argv[2])
R = json.loads(runs_path.read_text())
def pct(xs, q):
    xs = sorted(x for x in xs if x is not None)
    return xs[max(0, math.ceil(q * len(xs)) - 1)] if xs else None
def dist(xs): return {"n": len([x for x in xs if x is not None]), "p50": pct(xs, .5), "p95": pct(xs, .95)}
def run_metrics(r):
    rows = [json.loads(l) for l in open(r['diagnostics_jsonl'])]
    fin = [x for x in rows if x.get('english') is not None and x.get('chunk_id') is not None]; last = rows[-1]
    partials = r['diagnostics_jsonl'].replace('diagnostics_', 'partials_')
    n_prev = sum(1 for _ in open(partials)) if Path(partials).exists() else None
    hw = r['diagnostics_jsonl'].replace('diagnostics_', 'hardware_').replace('.jsonl', '.json')
    metal = None
    if Path(hw).exists():
        h = json.load(open(hw)); metal = h.get('metal_peak_bytes') or h.get('metal_peak_mib') or (h.get('peak') or {}).get('metal') or h.get('max_metal_bytes')
        if metal is None:
            def walk(o):
                if isinstance(o, dict):
                    for k, v in o.items():
                        if 'metal' in k.lower() and 'peak' in k.lower() and isinstance(v, (int, float)): return v
                        w = walk(v)
                        if w is not None: return w
                if isinstance(o, list):
                    for v in o:
                        w = walk(v)
                        if w is not None: return w
            metal = walk(h)
    def ft(x):
        t = x.get('timing_stages_ms') or {}
        if x.get('ttft_ms_a') is None or t.get('translation_started') is None or t.get('speech_end') is None: return None
        return t['translation_started'] + x['ttft_ms_a'] - t['speech_end']
    coh = {}
    for name, pred in (('gemma_silence', lambda x: x.get('final_translation_route') == 'gemma' and x.get('endpoint_reason') == 'silence'),
                       ('marian_silence', lambda x: x.get('final_translation_route') == 'marian' and x.get('endpoint_reason') == 'silence'),
                       ('all_silence', lambda x: x.get('endpoint_reason') == 'silence'), ('all', lambda x: True),
                       ('cuts', lambda x: x.get('endpoint_reason') in ('smart_cut', 'hard_cut', 'max_utterance'))):
        xs = [x for x in fin if pred(x)]
        coh[name] = {'n': len(xs), 'first_token_ms': dist([ft(x) for x in xs]), 'speech_end_to_final_ms': dist([x.get('speech_end_to_final_ms') for x in xs]),
                     'stt_latency_ms': dist([x.get('stt_latency_ms') for x in xs]), 'ttft_ms_a': dist([x.get('ttft_ms_a') for x in xs]),
                     'generation_lock_wait_ms_a': dist([x.get('generation_lock_wait_ms_a') for x in xs]), 'gen_tokens_a': dist([x.get('gen_tokens_a') for x in xs]),
                     'bad_split': sum(1 for x in xs if x.get('bad_split'))}
    routes = defaultdict(int)
    for x in fin: routes[str(x.get('final_stt_route'))] += 1
    counters = {k: v for k, v in (last.get('latency_experiment_counters') or {}).items() if 'reuse' in k or k in ('final_stt_full', 'partial_emitted')}
    return {'tag': r['tag'], 'clip': r['clip'], 'arm': r['arm'], 'repeat': r['repeat'], 'finals': len(fin), 'previews': n_prev, 'metal_peak': metal,
            'endpoints': dict((k, sum(1 for x in fin if x.get('endpoint_reason') == k)) for k in ('silence', 'smart_cut', 'hard_cut', 'max_utterance')),
            'routes': dict((k, sum(1 for x in fin if x.get('final_translation_route') == k)) for k in ('gemma', 'marian')), 'final_stt_route': dict(routes), 'counters': counters, 'cohorts': coh}
per_run = [run_metrics(r) for r in R['runs']]
arms = sorted({r['arm'] for r in R['runs']}); clips = sorted({r['clip'] for r in R['runs']})
def pooled(clip, arm):
    rs = [r for r in R['runs'] if r['clip'] == clip and r['arm'] == arm]
    rows = []
    for r in rs:
        for x in (json.loads(l) for l in open(r['diagnostics_jsonl'])):
            if x.get('english') is not None and x.get('chunk_id') is not None: rows.append(x)
    def ft(x):
        t = x.get('timing_stages_ms') or {}
        return None if x.get('ttft_ms_a') is None or t.get('translation_started') is None or t.get('speech_end') is None else t['translation_started'] + x['ttft_ms_a'] - t['speech_end']
    out = {'runs': len(rs), 'finals': len(rows), 'previews_mean': statistics.mean([m['previews'] for m in per_run if m['clip'] == clip and m['arm'] == arm and m['previews'] is not None] or [0])}
    for name, pred in (('gemma_silence', lambda x: x.get('final_translation_route') == 'gemma' and x.get('endpoint_reason') == 'silence'), ('all_silence', lambda x: x.get('endpoint_reason') == 'silence'), ('all', lambda x: True)):
        xs = [x for x in rows if pred(x)]
        out[name] = {'n': len(xs), 'first_token_ms': dist([ft(x) for x in xs]), 'speech_end_to_final_ms': dist([x.get('speech_end_to_final_ms') for x in xs]), 'stt_latency_ms': dist([x.get('stt_latency_ms') for x in xs]), 'generation_lock_wait_ms_a': dist([x.get('generation_lock_wait_ms_a') for x in xs]), 'bad_split': sum(1 for x in xs if x.get('bad_split'))}
    return out
pool = {c: {a: pooled(c, a) for a in arms} for c in clips}
comp = {}
for c in clips:
    ctl = pool[c].get('ctl')
    for a in arms:
        if a == 'ctl' or not ctl: continue
        d = {}
        for coh in ('gemma_silence', 'all_silence', 'all'):
            for m in ('first_token_ms', 'speech_end_to_final_ms'):
                for q in ('p50', 'p95'):
                    b, v = ctl[coh][m][q], pool[c][a][coh][m][q]
                    d[f'{coh}.{m}.{q}'] = {'ctl': b, 'arm': v, 'delta_ms': (v - b) if (b is not None and v is not None) else None, 'delta_pct': round(100 * (v - b) / b, 1) if (b and v is not None) else None}
        d['previews_delta_pct'] = round(100 * (pool[c][a]['previews_mean'] / ctl['previews_mean'] - 1), 1) if ctl['previews_mean'] else None
        comp[f'{c}:{a}'] = d
out = {'screen': R.get('screen'), 'per_run': per_run, 'pooled': pool, 'comparisons': comp}
out_path.write_text(json.dumps(out, indent=1))
for m in per_run:
    g = m['cohorts']['gemma_silence']
    print(f"{m['tag']:22s} finals {m['finals']:3d} prev {m['previews']} metal {m['metal_peak']} gemma-silence n={g['n']} ft p50 {g['first_token_ms']['p50'] and round(g['first_token_ms']['p50'])} ready p50 {g['speech_end_to_final_ms']['p50'] and round(g['speech_end_to_final_ms']['p50'])} p95 {g['speech_end_to_final_ms']['p95'] and round(g['speech_end_to_final_ms']['p95'])} lock p95 {g['generation_lock_wait_ms_a']['p95'] and round(g['generation_lock_wait_ms_a']['p95'],1)} stt p50 {g['stt_latency_ms']['p50'] and round(g['stt_latency_ms']['p50'])} routes {m['final_stt_route']} {m['counters']}")
for k, d in comp.items():
    print(k, {kk: v['delta_pct'] for kk, v in d.items() if isinstance(v, dict)}, 'previews', d['previews_delta_pct'])
