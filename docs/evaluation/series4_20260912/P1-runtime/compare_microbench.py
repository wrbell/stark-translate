"""Compare the P1 micro-bench receipts: text bench (ctl vs cand, two rounds each), Parakeet joint eval and profiles."""
import json, math, statistics
from pathlib import Path
P1 = Path('/Users/willem/Code/vibes/SRTranslate/.cache/series4-20260912/P1')
def load(n):
    p = P1 / n
    return json.loads(p.read_text()) if p.exists() else None
out = {"textbench": {}, "joint_eval": None, "profile": {}}
tb = {k: load(f"textbench_{k}.json") for k in ("ctl", "cand", "ctl2", "cand2")}
def per_length(d):
    e = d["configs"]["e4b"]; return e["per_length"], e["canary_details"], e.get("metal_peak_mib"), e.get("rss_delta_mb"), e["canary"]
rows = []
for k, d in tb.items():
    if not d: continue
    pl, cd, metal, rss, can = per_length(d)
    out["textbench"][k] = {"metal_peak_mib": metal, "rss_delta_mb": rss, "canary": can,
        "per_length": {name: {"p50_ms": v["latency"]["p50"], "p95_ms": v["latency"]["p95"], "mean_ms": v["latency"]["mean"], "tps_mean": v.get("tps_mean"), "n": v["latency"]["n"], "outputs_stable": len(set(v.get("text_sha256s", []))) <= 1, "text_sha256": (v.get("text_sha256s") or [None])[0]} for name, v in pl.items()},
        "outputs_sha256": {c["id"]: c["text_sha256"] for c in cd}, "canary_latency_ms": {c["id"]: c["latency_ms"] for c in cd}, "canary_ttft_ms": {c["id"]: c.get("ttft_ms") for c in cd}}
if out["textbench"].get("ctl") and out["textbench"].get("cand"):
    a, b = out["textbench"]["ctl"]["outputs_sha256"], out["textbench"]["cand"]["outputs_sha256"]
    out["textbench"]["canary_outputs_identical"] = a == b
    out["textbench"]["canary_output_diffs"] = [k for k in a if a.get(k) != b.get(k)]
    pa, pb = out["textbench"]["ctl"]["per_length"], out["textbench"]["cand"]["per_length"]
    out["textbench"]["per_length_outputs_identical"] = all(pa[k]["text_sha256"] == pb[k]["text_sha256"] for k in pa)
    out["textbench"]["per_length_delta_pct"] = {k: {m: round(100 * (pb[k][m] - pa[k][m]) / pa[k][m], 1) for m in ("p50_ms", "p95_ms", "mean_ms")} for k in pa}
    if out["textbench"].get("ctl2") and out["textbench"].get("cand2"):
        pa2, pb2 = out["textbench"]["ctl2"]["per_length"], out["textbench"]["cand2"]["per_length"]
        out["textbench"]["per_length_delta_pct_round2"] = {k: {m: round(100 * (pb2[k][m] - pa2[k][m]) / pa2[k][m], 1) for m in ("p50_ms", "p95_ms", "mean_ms")} for k in pa2}
        out["textbench"]["pooled_delta_pct"] = {m: round(100 * ((sum(pb[k][m] for k in pa) + sum(pb2[k][m] for k in pa2)) / (sum(pa[k][m] for k in pa) + sum(pa2[k][m] for k in pa2)) - 1), 1) for m in ("p50_ms", "p95_ms", "mean_ms")}
je = load("joint_eval_cand.json")
if je:
    a = je.get("assessment") or {}
    out["joint_eval"] = {"status": je.get("status"), "all_outputs_exact": a.get("all_outputs_exact"), "pairs": a.get("pairs"), "expected_pairs": a.get("expected_pairs"),
                         "summary": {k: v for k, v in a.items() if k not in ("pairs", "expected_pairs")}}
for k in ("ctl", "cand"):
    d = load(f"profile_{k}.json")
    if d:
        a = d.get("assessment") or {}
        out["profile"][k] = {"status": d.get("status"), "baseline_decode": d.get("baseline_decode"), **{kk: a.get(kk) for kk in ("profile_vs_control_wall_percent", "sampled_readback_share_of_profiled_stt_wall", "all_outputs_exact", "pairs")}}
(P1 / "microbench_compare.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1)[:6000])
