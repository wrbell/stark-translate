#!/usr/bin/env python3
"""Mac-runnable STT quality gate: Piper TTS → STT engines → WER + theological-term recall.

No human-verified church audio exists on the Mac, so this synthesises the theological
canaries, Tier-1 glossary terms (in carrier sentences) and the bench sentences with the
production Piper voices, transcribes them with every requested STT engine, and reports
normalised WER plus per-term recall. It is a *relative* gate (whisper-turbo vs
parakeet-mlx on identical audio), not an absolute WER claim.

    python tools/stt_roundtrip_compare.py --engines mlx,parakeet-mlx --lang en
    python tools/stt_roundtrip_compare.py --engines mlx,parakeet-mlx --lang es --es-verses 12

Loads Metal models — Mac only, never in CI.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time
import unicodedata
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

ROOT = Path(__file__).resolve().parents[1]
CARRIER_EN = "Today the message is about {term}, and we will read about it together."
CARRIER_ES = "Hoy el mensaje trata sobre {term}, y lo leeremos juntos."


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"[^\w\s']", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def wer(ref: str, hyp: str) -> float:
    import jiwer

    r, h = normalize(ref), normalize(hyp)
    if not r:
        return 0.0
    return float(jiwer.wer(r, h if h else " "))


def build_items(lang: str, n_terms: int, es_verses: int) -> list[dict]:
    from training.theological_canaries import THEOLOGICAL_CANARIES

    items: list[dict] = []
    if lang == "en":
        for i, c in enumerate(THEOLOGICAL_CANARIES):
            items.append({"id": f"canary_{i}", "text": c["en"], "terms": []})
        try:
            from tools.benchmark_mlx_accel import TEST_SENTENCES

            for k, v in TEST_SENTENCES.items():
                items.append({"id": f"sent_{k}", "text": v, "terms": []})
        except Exception:
            pass
        try:
            from tools.glossary import load_boost_keyterms

            for t in load_boost_keyterms()[:n_terms]:
                items.append({"id": f"term_{t}", "text": CARRIER_EN.format(term=t), "terms": [t]})
        except Exception as exc:  # glossary files are optional on a dev checkout
            print(f"  (glossary unavailable: {exc})")
    else:
        path = ROOT / "bible_data" / "es" / "rvr1909.jsonl"
        if path.exists():
            with path.open() as f:
                for line in f:
                    rec = json.loads(line)
                    text = rec.get("text") or rec.get("es") or rec.get("target") or ""
                    if 8 <= len(text.split()) <= 30:
                        items.append(
                            {"id": rec.get("ref") or rec.get("id") or f"verse_{len(items)}", "text": text, "terms": []}
                        )
                    if len(items) >= es_verses:
                        break
        for t in [
            "la expiación",
            "el partimiento del pan",
            "la justificación por la fe",
            "Santiago y Juan",
            "el propiciatorio",
        ]:
            items.append({"id": f"term_{t}", "text": CARRIER_ES.format(term=t), "terms": [t]})
    return items


def synthesize(items: list[dict], lang: str) -> list[np.ndarray]:
    from scipy.signal import resample_poly

    from engines.factory import create_tts_engine
    from settings import settings

    tts = create_tts_engine(voices={lang: settings.tts.voices[lang]})  # only the voice we need
    tts.load()
    audios = []
    for it in items:
        res = tts.synthesize(it["text"], language=lang)
        audio = np.asarray(res.audio, dtype=np.float32)
        if res.sample_rate != 16000:
            from math import gcd

            g = gcd(int(res.sample_rate), 16000)
            audio = resample_poly(audio, 16000 // g, int(res.sample_rate) // g).astype(np.float32)
        # 300 ms of leading/trailing silence, like a VAD-cut utterance
        pad = np.zeros(int(0.3 * 16000), dtype=np.float32)
        audios.append(np.concatenate([pad, audio, pad]))
    return audios


def make_engine(name: str):
    if name == "mlx":
        from engines.mlx_engine import MLXWhisperEngine

        return MLXWhisperEngine(fallback_on_low_conf=False)
    if name == "parakeet-mlx":
        from engines.parakeet_mlx_engine import ParakeetMLXEngine

        return ParakeetMLXEngine()
    raise SystemExit(f"unknown engine {name}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engines", default="mlx,parakeet-mlx")
    ap.add_argument("--lang", default="en", choices=["en", "es"])
    ap.add_argument("--n-terms", type=int, default=30, help="Tier-1 glossary terms to embed in carrier sentences (en)")
    ap.add_argument("--es-verses", type=int, default=12)
    ap.add_argument("--runs", type=int, default=1, help="Timed repetitions per item (latency only)")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    items = build_items(args.lang, args.n_terms, args.es_verses)
    print(f"{len(items)} items ({args.lang}); synthesising with Piper …")
    audios = synthesize(items, args.lang)
    report: dict = {"timestamp": datetime.now().isoformat(), "lang": args.lang, "n_items": len(items), "engines": {}}
    for name in args.engines.split(","):
        eng = make_engine(name)
        t0 = time.perf_counter()
        from tools.benchmark_identity import load_primary_model

        identity = load_primary_model(eng, eng.model_id)
        load_s = time.perf_counter() - t0
        eng.transcribe(audios[0], language=args.lang)  # warm
        rows = []
        for it, audio in zip(items, audios):
            lats = []
            for _ in range(args.runs):
                res = eng.transcribe(audio, language=args.lang)
                lats.append(res.latency_ms)
            hyp = res.text
            term_hits = [t for t in it["terms"] if normalize(t) in normalize(hyp)]
            rows.append(
                {
                    "id": it["id"],
                    "ref": it["text"],
                    "hyp": hyp,
                    "wer": wer(it["text"], hyp),
                    "latency_ms": statistics.median(lats),
                    "audio_s": round(len(audio) / 16000, 2),
                    "term_hit": (len(term_hits) == len(it["terms"])) if it["terms"] else None,
                    "confidence": res.confidence,
                }
            )
        wers = [r["wer"] for r in rows]
        term_rows = [r for r in rows if r["term_hit"] is not None]
        summary = {
            "model_identity": identity,
            "load_s": round(load_s, 1),
            "wer_mean": round(statistics.mean(wers), 4),
            "wer_median": round(statistics.median(wers), 4),
            "items_exact": sum(1 for w in wers if w == 0.0),
            "term_recall": round(sum(1 for r in term_rows if r["term_hit"]) / len(term_rows), 3) if term_rows else None,
            "latency_p50_ms": round(statistics.median(r["latency_ms"] for r in rows)),
            "latency_p95_ms": round(sorted(r["latency_ms"] for r in rows)[max(0, int(0.95 * len(rows)) - 1)]),
            "rows": rows,
        }
        report["engines"][name] = summary
        print(
            f"== {name}: WER mean {summary['wer_mean']:.3f} / median {summary['wer_median']:.3f} | exact {summary['items_exact']}/{len(rows)} "
            f"| term recall {summary['term_recall']} | p50 {summary['latency_p50_ms']} ms p95 {summary['latency_p95_ms']} ms | load {summary['load_s']} s"
        )
        for r in rows:
            if r["wer"] > 0.15 or r["term_hit"] is False:
                print(f"     {r['id'][:28]:28s} wer={r['wer']:.2f}  hyp={r['hyp'][:90]!r}")
        eng.unload()
    out = Path(args.output or ROOT / "metrics" / f"stt_roundtrip_{args.lang}_{datetime.now():%Y%m%d_%H%M%S}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
