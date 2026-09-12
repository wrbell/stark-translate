"""P3 offline feasibility: does Smart Turn v3 separate true utterance ends from internal pauses on our sermon replays?

Exact reconstruction from the six traced series-3 control runs. Coordinates: capture samples at 48 kHz; the replay
clips are 16 kHz mono PCM16, so clip16k index = capture sample // 3. Positives = silence finals cut at speech end +
pad; negatives = internal VAD-negative runs (>= 6 frames of 32 ms) that resumed speech inside the same utterance and
started >= 0.7 s into it (the minimum-final guard), cut at run start + pad. Classifier per upstream inference.py
(pinned; see model/pin.json). No threshold is tuned: 0.5 as shipped.
"""
from __future__ import annotations
import json, math, time, hashlib, statistics
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np, soundfile as sf, onnxruntime as ort
from transformers import WhisperFeatureExtractor

ROOT = Path("/Users/willem/Code/vibes/SRTranslate"); M = ROOT / "metrics"; OUT = ROOT / ".cache/series4-20260912/P3"
CLIPS = {"A": "Gospel_Message__12_14_25__5D2rOMvkwrk", "B": "Gospel_Message__2_8_26__BpIELcuh8O0"}
TAGS = [f"lb0912_{c}_ctl_r{r}" for c in "AB" for r in range(3)]
PADS_MS = (200, 300, 480); NEG_PADS_MS = (200, 300); FRAME = 1536; MIN_RUN = 6; MIN_FINAL_S = 0.7; SR16 = 16000
ONNX = OUT / "model/smart-turn-v3.2-cpu.onnx"

so = ort.SessionOptions(); so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL; so.inter_op_num_threads = 1
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
session = ort.InferenceSession(str(ONNX), sess_options=so, providers=["CPUExecutionProvider"])
fe = WhisperFeatureExtractor(chunk_length=8)

def truncate(a, n=8 * SR16):
    return a[-n:] if len(a) > n else np.pad(a, (n - len(a), 0))

def predict(audio):
    t0 = time.perf_counter()
    x = fe(truncate(audio), sampling_rate=SR16, return_tensors="np", padding="max_length", max_length=8 * SR16, truncation=True, do_normalize=True)
    feats = np.expand_dims(x.input_features.squeeze(0).astype(np.float32), 0)
    prob = float(session.run(None, {"input_features": feats})[0][0].item())
    return prob, (time.perf_counter() - t0) * 1000

def pct(xs, q):
    xs = sorted(xs); return xs[max(0, math.ceil(q * len(xs)) - 1)] if xs else None

def run(tag):
    clip = tag.split("_")[1]
    diag = M / f"diagnostics_{tag}_ctl_{CLIPS[clip]}.jsonl"
    rows = [json.loads(l) for l in open(diag)]
    finals = [r for r in rows if r.get("english") is not None and r.get("chunk_id") is not None]
    trace = rows[-1]["latency_trace"]["events"]
    audio, sr = sf.read(rows[0]["input_audio_path"], dtype="float32"); assert sr == SR16 and audio.ndim == 1
    frames = sorted((e["sample_start"], e["sample_end"], bool(e["vad_positive"])) for e in trace if e.get("event") == "vad_complete")
    pos, neg = [], []
    for f in finals:
        ss, se, send = f["sample_start"], f["sample_end"], f.get("speech_end_sample")
        if send is None: continue
        if f.get("endpoint_reason") == "silence":
            for pad in PADS_MS:
                end = min(len(audio), send // 3 + pad * SR16 // 1000)
                p, ms = predict(audio[ss // 3:end])
                pos.append({"tag": tag, "utterance_id": f["utterance_id"], "pad_ms": pad, "prob": p, "cpu_ms": ms, "dur_s": round((end - ss // 3) / SR16, 2), "route": f.get("final_translation_route")})
        # internal pauses inside this utterance's speech span
        inside = [fr for fr in frames if fr[0] >= ss and fr[1] <= send]
        i = 0
        while i < len(inside):
            if inside[i][2]: i += 1; continue
            j = i
            while j < len(inside) and not inside[j][2]: j += 1
            run_len = j - i; resumed = j < len(inside)
            run_start = inside[i][0]
            if resumed and run_len >= MIN_RUN and (run_start - ss) >= MIN_FINAL_S * 48000:
                for pad in NEG_PADS_MS:
                    end = min(len(audio), run_start // 3 + pad * SR16 // 1000)
                    p, ms = predict(audio[ss // 3:end])
                    neg.append({"tag": tag, "utterance_id": f["utterance_id"], "endpoint": f.get("endpoint_reason"), "run_frames": run_len, "run_ms": run_len * 32, "pad_ms": pad, "prob": p, "cpu_ms": ms, "dur_s": round((end - ss // 3) / SR16, 2)})
            i = j
    return {"tag": tag, "clip": clip, "clip_path": rows[0]["input_audio_path"], "clip_sha256": rows[0]["input_audio_sha256"], "n_finals": len(finals),
            "n_silence_finals": sum(1 for f in finals if f.get("endpoint_reason") == "silence"), "positives": pos, "negatives": neg}

def bin_of(frames):
    return "6-8" if frames <= 8 else ("9-11" if frames <= 11 else ">=12")

def summarize(pos, neg):
    out = {"positives": {}, "negatives": {}, "cpu_ms": {"n": len(pos) + len(neg), "p50": pct([x["cpu_ms"] for x in pos + neg], 0.5), "p95": pct([x["cpu_ms"] for x in pos + neg], 0.95)}}
    for pad in PADS_MS:
        xs = [x for x in pos if x["pad_ms"] == pad]
        out["positives"][f"pad_{pad}"] = {"n": len(xs), "complete_pct": round(100 * sum(1 for x in xs if x["prob"] > 0.5) / len(xs), 1) if xs else None, "prob_p50": pct([x["prob"] for x in xs], 0.5), "prob_p05": pct([x["prob"] for x in xs], 0.05)}
    for pad in NEG_PADS_MS:
        xs = [x for x in neg if x["pad_ms"] == pad]
        d = {"n": len(xs), "complete_pct": round(100 * sum(1 for x in xs if x["prob"] > 0.5) / len(xs), 1) if xs else None, "prob_p50": pct([x["prob"] for x in xs], 0.5), "prob_p95": pct([x["prob"] for x in xs], 0.95), "bins": {}}
        for b in ("6-8", "9-11", ">=12"):
            ys = [x for x in xs if bin_of(x["run_frames"]) == b]
            d["bins"][b] = {"n": len(ys), "complete_pct": round(100 * sum(1 for x in ys if x["prob"] > 0.5) / len(ys), 1) if ys else None}
        # pauses long enough to reach a countdown of `pad` ms (they would be the actual false-final exposure)
        reach = [x for x in xs if x["run_ms"] >= pad]
        d["reaching_countdown"] = {"n": len(reach), "complete_n": sum(1 for x in reach if x["prob"] > 0.5)}
        out["negatives"][f"pad_{pad}"] = d
    # rank AUC at matched pads
    for pad in NEG_PADS_MS:
        P = [x["prob"] for x in pos if x["pad_ms"] == pad]; N = [x["prob"] for x in neg if x["pad_ms"] == pad]
        if P and N:
            wins = sum(1 for p in P for n in N if p > n) + 0.5 * sum(1 for p in P for n in N if p == n)
            out[f"auc_pad_{pad}"] = round(wins / (len(P) * len(N)), 3)
    return out

def main():
    runs = [run(t) for t in TAGS]
    pos = [x for r in runs for x in r["positives"]]; neg = [x for r in runs for x in r["negatives"]]
    result = {"model": json.load(open(OUT / "model/pin.json")), "upstream_inference_sha256": hashlib.sha256((OUT / "upstream/inference.py").read_bytes()).hexdigest(),
              "threshold": 0.5, "pads_ms": PADS_MS, "neg_pads_ms": NEG_PADS_MS, "min_run_frames": MIN_RUN, "frame_ms": 32, "min_final_s": MIN_FINAL_S,
              "runs": [{k: v for k, v in r.items() if k not in ("positives", "negatives")} | {"n_pos": len(r["positives"]) // len(PADS_MS), "n_neg": len(r["negatives"]) // len(NEG_PADS_MS)} for r in runs],
              "per_clip": {c: summarize([x for x in pos if x["tag"].split("_")[1] == c], [x for x in neg if x["tag"].split("_")[1] == c]) for c in "AB"},
              "pooled": summarize(pos, neg), "positives": pos, "negatives": neg}
    # implied trade at a classifier-shortened countdown (complete → finalize at pad ms instead of ~480 ms)
    implied = {}
    n_runs = len(TAGS); silence_per_run = sum(r["n_silence_finals"] for r in runs) / n_runs; finals_per_run = sum(r["n_finals"] for r in runs) / n_runs
    for pad in NEG_PADS_MS:
        hit = result["pooled"]["positives"][f"pad_{pad}"]["complete_pct"] or 0
        extra = result["pooled"]["negatives"][f"pad_{pad}"]["reaching_countdown"]["complete_n"] / n_runs
        implied[f"countdown_{pad}"] = {"silence_finals_per_run": round(silence_per_run, 1), "finals_per_run": round(finals_per_run, 1),
                                       "early_finals_per_run": round(silence_per_run * hit / 100, 1), "ms_saved_each": 480 - pad,
                                       "spurious_finals_per_run": round(extra, 1), "spurious_pct_of_finals": round(100 * extra / finals_per_run, 1)}
    result["implied"] = implied
    (OUT / "eou_offline.json").write_text(json.dumps(result, indent=1))
    print(json.dumps({"runs": result["runs"], "pooled": result["pooled"], "implied": implied}, indent=1))

if __name__ == "__main__":
    main()
