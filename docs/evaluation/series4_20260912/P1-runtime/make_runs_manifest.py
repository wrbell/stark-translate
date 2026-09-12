"""RUNS.json for tools/tail_screen_report.py from run_screen.sh's runs.jsonl (two checkouts, metrics per checkout)."""
import glob, json, os, pathlib, sys
ROOT = pathlib.Path('/Users/willem/Code/vibes/SRTranslate'); P1 = ROOT / '.cache/series4-20260912/P1'
proto = json.loads((P1 / os.environ.get('PROTOCOL_JSON', 'protocol.json')).read_text())
runs = [json.loads(l) for l in (P1 / os.environ.get('RUNS_JSONL', 'runs.jsonl')).read_text().splitlines() if l.strip()]
def one(checkout, pattern):
    hits = sorted(glob.glob(str(pathlib.Path(checkout) / 'metrics' / pattern)))
    if len(hits) != 1: sys.exit(f'expected exactly one {pattern} under {checkout}, found {hits}')
    return hits[0]
out = {'screen': proto['id'], 'clips': {k: {'path': str(ROOT / v['path']), 'sha256': v['sha256']} for k, v in proto['clips'].items()},
       'arms': {'ctl': {'kind': 'control'}, 'cand': {'kind': 'candidate', 'memory_rule': 'relative'}}, 'runs': []}
for r in runs:
    if r['rc'] != 0: continue
    t, c = r['tag'], r['checkout']
    out['runs'].append({'tag': t, 'clip': r['clip'], 'arm': r['arm'], 'repeat': r['repeat'], 'order': r['order'],
        'replay_json': one(c, f'replay_{t}_*.json'), 'diagnostics_jsonl': one(c, f'diagnostics_{t}_*.jsonl'), 'csv': one(c, f'ab_metrics_{t}_*.csv'),
        'lifecycle_json': one(c, f'session_lifecycle_{t}_*.json'), 'session_metadata_json': one(c, f'session_metadata_{t}_*.json'), 'pageouts': r['pageouts']})
(P1 / 'RUNS.json').write_text(json.dumps(out, indent=2) + '\n'); print('runs', len(out['runs']))
