"""P2 attribution: can the last emitted partial's STT stand in for the final's STT on silence finals?

Inputs: the six series-3 control runs (lb0912_{A,B}_ctl_r{0,1,2}); per run the diagnostics rows (finals +
embedded latency trace) and the partials jsonl. Coordinates are capture samples at 48 kHz.
"""
from __future__ import annotations
import json, re, statistics, sys, hashlib
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path("/Users/willem/Code/vibes/SRTranslate")
M = ROOT / "metrics"
CLIPS = {"A": "Gospel_Message__12_14_25__5D2rOMvkwrk", "B": "Gospel_Message__2_8_26__BpIELcuh8O0"}
TAGS = [f"lb0912_{c}_ctl_r{r}" for c in "AB" for r in range(3)]

def norm(t):
    t = (t or "").casefold()
    t = re.sub(r"[^\w\s']", " ", t)
    return " ".join(t.split())

def pct(xs, q):
    xs = sorted(xs)
    if not xs: return None
    import math
    return xs[max(0, math.ceil(q * len(xs)) - 1)]

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def run(tag):
    clip = tag.split("_")[1]
    diag = M / f"diagnostics_{tag}_ctl_{CLIPS[clip]}.jsonl"
    part = M / f"partials_{tag}_ctl_{CLIPS[clip]}.jsonl"
    rows = [json.loads(l) for l in open(diag)]
    finals = [r for r in rows if r.get("english") is not None and r.get("chunk_id") is not None]
    trace = rows[-1]["latency_trace"]["events"]
    partials = [json.loads(l) for l in open(part)]
    last_partial = {}
    for p in partials:
        u = p["utterance_id"]
        if u not in last_partial or p["emitted_at_ms"] > last_partial[u]["emitted_at_ms"]:
            last_partial[u] = p
    submitted = {e["utterance_id"]: e["at_ms"] for e in trace if e.get("event") == "final_queue_submitted"}
    pstt_done = {}
    for e in trace:
        if e.get("event") == "physical_stt_finished" and e.get("kind") == "partial":
            pstt_done[(e["utterance_id"], e["sample_end"])] = e
    fstt = {e["utterance_id"]: e for e in trace if e.get("event") == "physical_stt_finished" and e.get("kind") == "final"}
    out = []
    for f in finals:
        u = f["utterance_id"]; lp = last_partial.get(u)
        rec = {"utterance_id": u, "chunk_id": f["chunk_id"], "endpoint": f.get("endpoint_reason"), "route": f.get("final_translation_route"),
               "final_stt_ms": (fstt.get(u) or {}).get("elapsed_ms", f.get("stt_latency_ms")), "speech_end_to_final_ms": f.get("speech_end_to_final_ms"),
               "has_partial": lp is not None}
        if lp is not None and f.get("speech_end_sample") is not None:
            gap = f["speech_end_sample"] - lp["sample_end"]
            rec["gap_ms"] = gap / 48.0
            rec["partial_covers_speech_end"] = lp["sample_end"] >= f["speech_end_sample"]
            done = pstt_done.get((u, lp["sample_end"]))
            rec["lead_stt_ms"] = (submitted[u] - done["at_ms"]) if (done and u in submitted) else None
            rec["lead_emit_ms"] = (submitted[u] - lp["emitted_at_ms"]) if u in submitted else None
            pe, fe, fc = lp.get("text_en") or "", f.get("english") or "", f.get("corrected_english") or f.get("english") or ""
            rec["text_exact"] = pe == fe
            rec["text_norm_equal"] = norm(pe) == norm(fe)
            rec["text_norm_equal_corrected"] = norm(pe) == norm(fc)
            rec["text_prefix"] = (not rec["text_norm_equal"]) and norm(fe).startswith(norm(pe)) and bool(norm(pe))
            rec["partial_words"], rec["final_words"] = len(norm(pe).split()), len(norm(fe).split())
            if not rec["text_norm_equal"]:
                rec["partial_text"], rec["final_text"] = pe, fe
        out.append(rec)
    return {"tag": tag, "clip": clip, "diagnostics": str(diag), "diagnostics_sha256": sha(diag), "partials": str(part), "partials_sha256": sha(part),
            "n_finals": len(finals), "endpoints": dict(Counter(r["endpoint"] for r in out)), "finals": out}

def summarize(recs):
    s = [r for r in recs if r["endpoint"] == "silence"]
    withp = [r for r in s if r.get("gap_ms") is not None]
    def share(pred): return round(100 * sum(1 for r in withp if pred(r)) / len(withp), 1) if withp else None
    leads = [r["lead_stt_ms"] for r in withp if r.get("lead_stt_ms") is not None]
    return {"n_silence": len(s), "n_with_partial": len(withp),
            "gap_le_100_pct": share(lambda r: r["gap_ms"] <= 100), "gap_le_300_pct": share(lambda r: r["gap_ms"] <= 300), "gap_le_600_pct": share(lambda r: r["gap_ms"] <= 600),
            "gap_ms_p50": pct([r["gap_ms"] for r in withp], 0.5), "gap_ms_p95": pct([r["gap_ms"] for r in withp], 0.95),
            "partial_result_before_finalize_pct": (round(100 * sum(1 for l in leads if l > 0) / len(leads), 1) if leads else None),
            "lead_stt_ms_min": min(leads) if leads else None, "lead_stt_ms_p50": pct(leads, 0.5),
            "text_exact_pct": share(lambda r: r["text_exact"]), "text_norm_equal_pct": share(lambda r: r["text_norm_equal"]),
            "text_norm_equal_corrected_pct": share(lambda r: r["text_norm_equal_corrected"]),
            "text_prefix_pct": share(lambda r: r["text_prefix"]), "text_differs_pct": share(lambda r: not (r["text_norm_equal"] or r["text_prefix"])),
            "final_stt_ms_p50_silence": pct([r["final_stt_ms"] for r in s if r.get("final_stt_ms") is not None], 0.5),
            "final_stt_ms_p95_silence": pct([r["final_stt_ms"] for r in s if r.get("final_stt_ms") is not None], 0.95),
            "gemma_routed_silence": sum(1 for r in s if r["route"] == "gemma"),
            "reuse_at_100_gemma": sum(1 for r in withp if r["gap_ms"] <= 100 and r["route"] == "gemma"),
            "reuse_at_300_gemma": sum(1 for r in withp if r["gap_ms"] <= 300 and r["route"] == "gemma")}

def main():
    runs = [run(t) for t in TAGS]
    per_run = {r["tag"]: summarize(r["finals"]) for r in runs}
    per_clip = {c: summarize([f for r in runs if r["clip"] == c for f in r["finals"]]) for c in "AB"}
    pooled = summarize([f for r in runs for f in r["finals"]])
    differing = [{"tag": r["tag"], **{k: f[k] for k in ("utterance_id", "gap_ms", "partial_text", "final_text")}} for r in runs for f in r["finals"] if f["endpoint"] == "silence" and f.get("partial_text") is not None]
    result = {"runs": runs, "per_run": per_run, "per_clip": per_clip, "pooled": pooled, "differing_silence_finals": differing,
              "method": {"coordinates": "capture samples at 48 kHz; gap_ms = (final.speech_end_sample - last_partial.sample_end)/48",
                         "lead_stt_ms": "final_queue_submitted.at_ms - physical_stt_finished(kind=partial, same utterance and sample_end).at_ms; positive = the partial's STT result existed before the pipeline finalized",
                         "text": "partial text_en vs final english (raw STT) and corrected_english; norm = casefold, punctuation stripped, whitespace collapsed"}}
    out = ROOT / ".cache/series4-20260912/P2"
    (out / "attribution.json").write_text(json.dumps(result, indent=1))
    lines = ["# P2 attribution — last emitted partial vs final STT (silence finals)", "",
             "| cohort | silence finals | with partial | gap ≤100 ms | ≤300 ms | ≤600 ms | gap p50/p95 ms | partial STT done before finalize | lead p50 ms (min) | text equal (norm) | prefix | differs | final STT p50/p95 ms | Gemma-routed | reuse@100 / @300 (Gemma) |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    def row(name, s):
        return f"| {name} | {s['n_silence']} | {s['n_with_partial']} | {s['gap_le_100_pct']} % | {s['gap_le_300_pct']} % | {s['gap_le_600_pct']} % | {s['gap_ms_p50']:.0f}/{s['gap_ms_p95']:.0f} | {s['partial_result_before_finalize_pct']} % | {s['lead_stt_ms_p50']:.0f} ({s['lead_stt_ms_min']:.0f}) | {s['text_norm_equal_pct']} % | {s['text_prefix_pct']} % | {s['text_differs_pct']} % | {s['final_stt_ms_p50_silence']:.0f}/{s['final_stt_ms_p95_silence']:.0f} | {s['gemma_routed_silence']} | {s['reuse_at_100_gemma']} / {s['reuse_at_300_gemma']} |"
    for t in TAGS: lines.append(row(t, per_run[t]))
    for c in "AB": lines.append(row(f"clip {c} pooled", per_clip[c]))
    lines.append(row("all pooled", pooled))
    lines += ["", f"Differing silence finals (partial text vs raw final STT, not equal and not a prefix): {len(differing)}", ""]
    for d in differing:
        lines.append(f"- `{d['tag']}` u{d['utterance_id']} gap {d['gap_ms']:.0f} ms — partial: “{d['partial_text']}” / final: “{d['final_text']}”")
    (out / "attribution.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:14]))
    print("pooled", json.dumps(pooled))

if __name__ == "__main__":
    main()
