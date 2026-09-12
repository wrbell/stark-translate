"""P3 addendum: extra positives from the 2026-09-10 endurance service replay's retained silence-final chunks."""
from __future__ import annotations
import json, math, os, time, hashlib
from pathlib import Path
import numpy as np, soundfile as sf, onnxruntime as ort
from transformers import WhisperFeatureExtractor
ROOT = Path("/Users/willem/Code/vibes/SRTranslate"); OUT = ROOT / ".cache/series4-20260912/P3"
BASE = ROOT / ".cache/overnight-20260910/installed-standard-fixed-endurance"
DIAG = BASE / "metrics/diagnostics_20260910_043120_839144_en.jsonl"
PADS = (200, 300, 480); SR16 = 16000
so = ort.SessionOptions(); so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL; so.inter_op_num_threads = 1; so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
session = ort.InferenceSession(str(OUT / "model/smart-turn-v3.2-cpu.onnx"), sess_options=so, providers=["CPUExecutionProvider"]); fe = WhisperFeatureExtractor(chunk_length=8)
def truncate(a, n=8 * SR16): return a[-n:] if len(a) > n else np.pad(a, (n - len(a), 0))
def predict(audio):
    t0 = time.perf_counter(); x = fe(truncate(audio), sampling_rate=SR16, return_tensors="np", padding="max_length", max_length=8 * SR16, truncation=True, do_normalize=True)
    return float(session.run(None, {"input_features": np.expand_dims(x.input_features.squeeze(0).astype(np.float32), 0)})[0][0].item()), (time.perf_counter() - t0) * 1000
def pct(xs, q): xs = sorted(xs); return xs[max(0, math.ceil(q * len(xs)) - 1)] if xs else None
rows = [json.loads(l) for l in open(DIAG)]
fin = [r for r in rows if r.get("english") is not None and r.get("chunk_id") is not None and r.get("endpoint_reason") == "silence"]
res, missing, bad = [], 0, 0
for r in fin:
    p = BASE / r["audio_path"]
    if not p.exists(): missing += 1; continue
    a, sr = sf.read(p, dtype="float32")
    if sr != SR16 or a.ndim != 1: bad += 1; continue
    off = (r["speech_end_sample"] - r["sample_start"]) // 3
    tail_ms = (len(a) - off) * 1000 / SR16
    for pad in PADS:
        end = min(len(a), off + pad * SR16 // 1000)
        prob, ms = predict(a[:end]); res.append({"chunk_id": r["chunk_id"], "pad_ms": pad, "prob": prob, "cpu_ms": ms, "dur_s": round(end / SR16, 2), "tail_ms": round(tail_ms), "route": r.get("final_translation_route")})
summary = {"source": {"diagnostics": str(DIAG.relative_to(ROOT)), "sha256": hashlib.sha256(DIAG.read_bytes()).hexdigest(), "session": "20260910_043120_839144_en"}, "n_silence_finals": len(fin), "missing_audio": missing, "bad_format": bad,
           "tail_ms_p50": pct([x["tail_ms"] for x in res if x["pad_ms"] == 200], 0.5),
           "positives": {f"pad_{pad}": {"n": sum(1 for x in res if x["pad_ms"] == pad), "complete_pct": round(100 * sum(1 for x in res if x["pad_ms"] == pad and x["prob"] > 0.5) / max(1, sum(1 for x in res if x["pad_ms"] == pad)), 1), "prob_p50": pct([x["prob"] for x in res if x["pad_ms"] == pad], 0.5), "prob_p05": pct([x["prob"] for x in res if x["pad_ms"] == pad], 0.05)} for pad in PADS},
           "cpu_ms": {"p50": pct([x["cpu_ms"] for x in res], 0.5), "p95": pct([x["cpu_ms"] for x in res], 0.95)}}
(OUT / "eou_endurance.json").write_text(json.dumps({"summary": summary, "items": res}, indent=1)); print(json.dumps(summary, indent=1))
