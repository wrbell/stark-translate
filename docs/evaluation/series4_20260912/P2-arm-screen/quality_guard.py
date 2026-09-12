"""Text guard for a screen: pooled English WER and concatenated Spanish WER of each candidate run vs the same-repeat control,
bad_split counts, differing finals (sequence-aligned on English), and the arm's own counters from the diagnostics summary row.
Usage: quality_guard.py RUNS.json OUT.json"""
import csv, json, sys
from difflib import SequenceMatcher
from pathlib import Path
import jiwer
runs_path, out_path = Path(sys.argv[1]), Path(sys.argv[2])
runs = json.loads(runs_path.read_text())['runs']
def finals(path):
    return [((r.get('english') or '').strip(), (r.get('spanish_a') or '').strip()) for r in csv.DictReader(open(path, newline=''))]
def diag(path):
    rows = [json.loads(l) for l in open(path)]
    fin = [r for r in rows if r.get('english') is not None and r.get('chunk_id') is not None]
    return fin, rows[-1]
ctl = {(r['clip'], r['repeat']): r for r in runs if r['arm'] == 'ctl'}
out = []
for r in runs:
    if r['arm'] == 'ctl': continue
    c = ctl[(r['clip'], r['repeat'])]; a = finals(r['csv']); b = finals(c['csv'])
    es_a = ' '.join(x[1] for x in a if x[1]); es_b = ' '.join(x[1] for x in b if x[1]); en_a = ' '.join(x[0] for x in a if x[0]); en_b = ' '.join(x[0] for x in b if x[0])
    fa, last_a = diag(r['diagnostics_jsonl']); fb, _ = diag(c['diagnostics_jsonl'])
    counters = last_a.get('latency_experiment_counters') or {}
    m = SequenceMatcher(None, [x[0] for x in a], [x[0] for x in b], autojunk=False)
    aligned = [(i + k, j + k) for i, j, n in m.get_matching_blocks() for k in range(n)]
    matched_a = {i for i, _ in aligned}; matched_b = {j for _, j in aligned}
    diffs = [{'candidate_index': i, 'candidate': a[i]} for i in range(len(a)) if i not in matched_a] + [{'control_index': j, 'control': b[j]} for j in range(len(b)) if j not in matched_b]
    spanish_diffs = sum(1 for i, j in aligned if a[i][1] != b[j][1])
    routes = {}
    for x in fa: routes[x.get('final_stt_route') or 'n/a'] = routes.get(x.get('final_stt_route') or 'n/a', 0) + 1
    out.append({'tag': r['tag'], 'arm': r['arm'], 'control': c['tag'], 'finals': len(a), 'control_finals': len(b), 'empty_finals': sum(1 for x in a if not x[1]),
                'duplicate_finals': len(a) - len({x[1] for x in a}), 'bad_split': sum(1 for x in fa if x.get('bad_split')), 'control_bad_split': sum(1 for x in fb if x.get('bad_split')),
                'english_wer_vs_control_pct': round(100 * jiwer.wer(en_b, en_a), 2) if en_b and en_a else None,
                'spanish_wer_vs_control_pct': round(100 * jiwer.wer(es_b, es_a), 2) if es_b and es_a else None,
                'aligned_english_finals': len(aligned), 'aligned_with_different_spanish': spanish_diffs, 'unmatched_finals': diffs,
                'final_stt_route_counts': routes, 'counters': {k: v for k, v in counters.items() if 'reuse' in k or k == 'final_stt_full'},
                'experiment_configuration': {k: v for k, v in (last_a.get('latency_experiment_configuration') or {}).items() if 'reuse' in k}})
out_path.write_text(json.dumps(out, indent=2) + '\n')
for o in out:
    print(o['tag'], 'finals', o['finals'], '/', o['control_finals'], 'bad_split', o['bad_split'], '/', o['control_bad_split'], 'EN WER%', o['english_wer_vs_control_pct'], 'ES WER%', o['spanish_wer_vs_control_pct'], 'aligned', o['aligned_english_finals'], 'es-diff', o['aligned_with_different_spanish'], 'unmatched', len(o['unmatched_finals']), o['final_stt_route_counts'], o['counters'])
