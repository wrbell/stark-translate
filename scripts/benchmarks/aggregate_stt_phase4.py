#!/usr/bin/env python3
"""aggregate_stt_phase4.py — read metrics/stt_phase4_*.json and emit the
markdown table for docs/archive/v2026.7/STT_BENCHMARK.md.

Run after the Phase 4 sweep finishes:

    python scripts/benchmarks/aggregate_stt_phase4.py

Prints the table to stdout. Hand-paste into the BENCHMARK.md doc.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
METRICS_DIR = PROJECT_ROOT / "metrics"

VARIANT_ORDER = [
    "fw_int8_offshelf",
    "fw_int8float16_offshelf",
    "fw_int8_w16",
    "fw_int8float16_w16",
    "hf_fp16_spec",
    "hf_fp16_spec_compiled",
    "hf_fp16_nospec",
    "hf_fp16_nospec_compiled",
]

PRETTY = {
    "fw_int8_offshelf": "fw_int8_offshelf (Phase 0 baseline)",
    "fw_int8float16_offshelf": "fw_int8float16_offshelf",
    "fw_int8_w16": "fw_int8_w16",
    "fw_int8float16_w16": "**fw_int8float16_w16 (production target)**",
    "hf_fp16_spec": "hf_fp16_spec",
    "hf_fp16_spec_compiled": "hf_fp16_spec_compiled (Phase 1 win)",
    "hf_fp16_nospec": "hf_fp16_nospec",
    "hf_fp16_nospec_compiled": "hf_fp16_nospec_compiled",
}


def fmt(v, n=1, suffix=""):
    if v is None:
        return "—"
    if isinstance(v, float):
        if n == 0:
            return f"{round(v)}{suffix}"
        return f"{v:.{n}f}{suffix}"
    return str(v) + suffix


def fmt_pct(v):
    if v is None:
        return "—"
    return f"{v * 100:.2f}%"


def main() -> int:
    rows = []
    for key in VARIANT_ORDER:
        path = METRICS_DIR / f"stt_phase4_{key}.json"
        if not path.exists():
            rows.append((key, None))
            continue
        rows.append((key, json.loads(path.read_text())))

    print(
        "| Variant | Load (s) | Cold start (ms) | Latency p50 (ms) | Latency p95 (ms) | RTF p95 (short) | VRAM peak (MiB) | WER overall | WER tier1 |"
    )
    print("|---|---|---|---|---|---|---|---|---|")
    for key, doc in rows:
        label = PRETTY.get(key, key)
        if doc is None:
            print(f"| {label} | _missing_ | — | — | — | — | — | — | — |")
            continue
        s = doc["summary"]
        short_tier = s["tiers"].get("short", {})
        print(
            f"| {label} | {fmt(s['load_seconds'])} | {fmt(s.get('cold_start_ms'))} | "
            f"{fmt(s.get('latency_ms_overall_median'))} | {fmt(s.get('latency_ms_overall_p95'))} | "
            f"{fmt(short_tier.get('rtf_p95'), n=3)} | {fmt(s['vram']['max_mib'], n=0)} | "
            f"{fmt_pct(s.get('wer_normalized_overall'))} | {fmt_pct(s.get('wer_normalized_tier1_only'))} |"
        )

    # Compact takeaway block — easier to scan than the table
    print()
    print("---")
    print()
    print("## Headline numbers")
    print()
    baseline = rows[0][1]
    if baseline is None:
        print("(no baseline; can't compare)")
        return 0
    base_p95 = baseline["summary"].get("latency_ms_overall_p95")
    base_wer = baseline["summary"].get("wer_normalized_overall")
    for key, doc in rows[1:]:
        if doc is None:
            continue
        s = doc["summary"]
        p95 = s.get("latency_ms_overall_p95")
        wer = s.get("wer_normalized_overall")
        if p95 is None or base_p95 is None:
            continue
        ratio = p95 / base_p95
        wer_delta = (wer - base_wer) * 100 if wer is not None and base_wer is not None else None
        print(
            f"- **{key}**: p95 {fmt(p95)} ms ({ratio * 100:.0f}% of baseline), "
            f"WER {fmt_pct(wer)} ({wer_delta:+.2f} pts vs baseline)"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
