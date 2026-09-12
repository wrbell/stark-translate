"""Declared-gate verdict for the partial_reuse screen from report.json (tool gates), metrics.json and quality_guard.json."""
import json, sys
from pathlib import Path
P2 = Path('/Users/willem/Code/vibes/SRTranslate/.cache/series4-20260912/P2')
report = json.loads((P2 / 'report.json').read_text()); metrics = json.loads((P2 / 'metrics.json').read_text()); guard = json.loads((P2 / 'quality_guard.json').read_text())
arms = [a for a in report['clips']['A']['arms'] if report['clips']['A']['arms'][a].get('gates')]
verdicts = {}
for arm in arms:
    g = {}
    for clip in [c for c in 'AB' if f'{c}:{arm}' in metrics['comparisons']]:
        comp = metrics['comparisons'][f'{clip}:{arm}']
        ft = comp['gemma_silence.first_token_ms.p50']['delta_pct']; ready = comp['gemma_silence.speech_end_to_final_ms.p50']['delta_pct']
        g[f'{clip}.G_median'] = {'first_token_p50_delta_pct': ft, 'ready_p50_delta_pct': ready, 'pass': ft is not None and ready is not None and ft <= -15 and ready <= -15}
        tails = {}
        # Declared reading (fixed 2026-09-12T03:15Z before clip B): both screened metrics, p95, three cohorts.
        for coh, metric in (('gemma_silence', 'first_token_ms'), ('all_silence', 'first_token_ms'), ('gemma_silence', 'speech_end_to_final_ms'), ('all_silence', 'speech_end_to_final_ms'), ('all', 'speech_end_to_final_ms')):
            c = comp[f'{coh}.{metric}.p95']; allow = max(0.05 * c['ctl'], 100) if c['ctl'] else None
            tails[f'{coh}.{metric}'] = {'ctl': c['ctl'], 'arm': c['arm'], 'delta_ms': c['delta_ms'], 'allowance_ms': allow, 'pass': c['delta_ms'] is not None and allow is not None and c['delta_ms'] <= allow}
        tool_g2 = report['clips'][clip]['arms'][arm]['gates']['G2']
        g[f'{clip}.G_tail'] = {'cohorts': tails, 'tool_G2_cuts_p95': tool_g2.get('cuts'), 'pass': all(t['pass'] for t in tails.values())}
        g[f'{clip}.G_preview'] = {'previews_delta_pct': comp['previews_delta_pct'], 'pass': comp['previews_delta_pct'] is not None and comp['previews_delta_pct'] >= -2}
        tool = report['clips'][clip]['arms'][arm]['gates']
        g[f'{clip}.G_memory_tool_G6'] = {'pass': tool['G6']['pass']}
        g[f'{clip}.tool_G7'] = {'pass': tool['G7']['pass']}
        qs = [q for q in guard if q['arm'] == arm and q['tag'].split('_')[1] == clip]
        g[f'{clip}.G_text'] = {'runs': [{'tag': q['tag'], 'en_wer': q['english_wer_vs_control_pct'], 'es_wer': q['spanish_wer_vs_control_pct'], 'bad_split': q['bad_split'], 'control_bad_split': q['control_bad_split'], 'unmatched': len(q['unmatched_finals']), 'reuse_counts': q['final_stt_route_counts']} for q in qs],
                              'pass': all((q['english_wer_vs_control_pct'] or 0) <= 3 and (q['spanish_wer_vs_control_pct'] or 0) <= 3 and q['bad_split'] <= q['control_bad_split'] for q in qs) and bool(qs)}
    verdicts[arm] = {'gates': g, 'PASS': all(v['pass'] for v in g.values())}
(P2 / 'verdicts.json').write_text(json.dumps(verdicts, indent=1))
for arm, v in verdicts.items():
    print(arm, 'PASS' if v['PASS'] else 'REJECTED', {k: vv['pass'] for k, vv in v['gates'].items()})
    for k, vv in v['gates'].items():
        if k.endswith('G_median'): print('   ', k, vv)
