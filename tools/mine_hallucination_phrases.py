#!/usr/bin/env python3
"""Mine new Whisper hallucination phrases from session diagnostics JSONL.

Scans ``metrics/diagnostics_*.jsonl`` for suppressed / high-priority short
utterances and proposes additions to ``_HALLUCINATION_PHRASES`` in dry_run_ab.py.

Usage::

    python tools/mine_hallucination_phrases.py
    python tools/mine_hallucination_phrases.py --metrics-dir metrics --min-count 3 --json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dry_run_ab import _HALLUCINATION_PHRASES, _normalize_suppress_text

_WORD_RE = re.compile(r"[a-záéíóúüñ]+(?:\s+[a-záéíóúüñ]+){0,5}", re.IGNORECASE)


def load_records(metrics_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(metrics_dir.glob("diagnostics_*.jsonl")):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "event" in rec:
                    continue
                rows.append(rec)
    return rows


def candidate_phrases(records: list[dict], *, max_words: int = 6) -> Counter:
    counts: Counter = Counter()
    for rec in records:
        text = (rec.get("english") or "").strip()
        if not text:
            continue
        # Prefer flagged / low-confidence / hallucination-marked rows
        priority = int(rec.get("review_priority") or 0)
        conf = rec.get("stt_confidence")
        is_hall = bool(rec.get("is_hallucination"))
        try:
            conf_f = float(conf) if conf is not None else 1.0
        except (TypeError, ValueError):
            conf_f = 1.0
        if not (is_hall or priority >= 3 or conf_f < 0.55):
            continue
        norm = _normalize_suppress_text(text)
        words = norm.split()
        if 1 <= len(words) <= max_words:
            counts[norm] += 1
    return counts


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metrics-dir", type=Path, default=Path("metrics"))
    p.add_argument("--min-count", type=int, default=2)
    p.add_argument("--suggest-threshold", type=float, default=0.70)
    p.add_argument("--json", action="store_true", help="Emit JSON suggestions")
    args = p.parse_args(argv)

    records = load_records(args.metrics_dir)
    counts = candidate_phrases(records)
    known = set(_HALLUCINATION_PHRASES)
    suggestions = []
    for phrase, count in counts.most_common():
        if count < args.min_count:
            continue
        if phrase in known:
            continue
        suggestions.append(
            {
                "phrase": phrase,
                "count": count,
                "suggested_threshold": args.suggest_threshold,
            }
        )

    if args.json:
        print(json.dumps({"suggestions": suggestions, "scanned": len(records)}, indent=2))
    else:
        print(f"Scanned {len(records)} chunk records")
        if not suggestions:
            print("No new hallucination phrase candidates.")
        else:
            print("Suggested additions to _HALLUCINATION_PHRASES:")
            for s in suggestions:
                print(f'  "{s["phrase"]}": {s["suggested_threshold"]},  # n={s["count"]}')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
