#!/usr/bin/env python3
"""Post-session KPI scorecard generator.

Reads CSV and diagnostics JSONL from live sessions and produces a markdown
scorecard grading 5 KPIs: latency, WER proxy, translation QE, word stability,
and processing rate.

Usage:
    python tools/kpi_report.py metrics/ab_metrics_20260301_180046_en.csv
    python tools/kpi_report.py --latest
    python tools/kpi_report.py metrics/ab_metrics_*.csv --output report.md
    python tools/kpi_report.py metrics/ab_metrics_*.csv --comet
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import statistics
import sys
from datetime import datetime


def load_csv(path: str) -> list[dict]:
    """Load a CSV file and return rows as list of dicts."""
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def find_diagnostics_for_csv(csv_path: str) -> str | None:
    """Infer diagnostics JSONL path from CSV naming convention.

    CSV: metrics/ab_metrics_20260301_180046_en.csv
    JSONL: metrics/diagnostics_20260301_180046_en.jsonl
    """
    base = os.path.basename(csv_path)
    if base.startswith("ab_metrics_"):
        session_part = base[len("ab_metrics_") :].rsplit(".", 1)[0]
        diag_name = f"diagnostics_{session_part}.jsonl"
        diag_path = os.path.join(os.path.dirname(csv_path), diag_name)
        if os.path.exists(diag_path):
            return diag_path
    return None


def load_diagnostics(path: str) -> list[dict]:
    """Load a JSONL diagnostics file and return records as list of dicts."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _safe_float(value: str | float | None) -> float | None:
    """Convert a CSV value to float, returning None for empty/invalid."""
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def _percentile(values: list[float], pct: int) -> float:
    """Compute percentile without numpy dependency."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    k = (len(sorted_vals) - 1) * pct / 100
    f = int(k)
    c = f + 1
    if c >= len(sorted_vals):
        return sorted_vals[-1]
    return sorted_vals[f] + (k - f) * (sorted_vals[c] - sorted_vals[f])


STALL_THRESHOLD_MS = 10000  # >10s indicates queue stall, not normal latency


def compute_latency_kpis(rows: list[dict]) -> dict:
    """KPI 1: Latency percentiles from CSV data.

    Uses true_e2e_ms when available, falls back to e2e_latency_ms.
    Outliers >10s are counted separately as stalls.
    """
    e2e_values = []
    stt_values = []
    trans_a_values = []
    trans_b_values = []
    stall_count = 0

    for row in rows:
        # Prefer true_e2e_ms, fall back to e2e_latency_ms
        e2e = _safe_float(row.get("true_e2e_ms")) or _safe_float(row.get("e2e_latency_ms"))
        if e2e is not None:
            if e2e > STALL_THRESHOLD_MS:
                stall_count += 1
            else:
                e2e_values.append(e2e)

        stt = _safe_float(row.get("stt_latency_ms"))
        if stt is not None:
            stt_values.append(stt)

        lat_a = _safe_float(row.get("latency_a_ms"))
        if lat_a is not None and lat_a > 0:
            trans_a_values.append(lat_a)

        lat_b = _safe_float(row.get("latency_b_ms"))
        if lat_b is not None and lat_b > 0:
            trans_b_values.append(lat_b)

    return {
        "e2e_p50": _percentile(e2e_values, 50),
        "e2e_p95": _percentile(e2e_values, 95),
        "e2e_p99": _percentile(e2e_values, 99),
        "e2e_count": len(e2e_values),
        "stt_p50": _percentile(stt_values, 50),
        "stt_p95": _percentile(stt_values, 95),
        "trans_a_p50": _percentile(trans_a_values, 50),
        "trans_a_p95": _percentile(trans_a_values, 95),
        "trans_b_p50": _percentile(trans_b_values, 50) if trans_b_values else None,
        "trans_b_p95": _percentile(trans_b_values, 95) if trans_b_values else None,
        "stall_count": stall_count,
    }


def compute_wer_kpis(rows: list[dict], diag_records: list[dict]) -> dict:
    """KPI 2: WER proxy from confidence distribution and drop rates."""
    confidences = []
    for row in rows:
        conf = _safe_float(row.get("stt_confidence"))
        if conf is not None:
            confidences.append(conf)

    # Count empty STT drops and hallucination events from diagnostics
    empty_stt_count = sum(1 for r in diag_records if r.get("event") == "empty_stt_drop")
    hallucination_count = sum(1 for r in diag_records if r.get("is_hallucination") is True)
    low_conf_count = sum(1 for c in confidences if c < 0.5)
    total_chunks = len(rows)

    return {
        "conf_mean": statistics.mean(confidences) if confidences else None,
        "conf_p10": _percentile(confidences, 10) if confidences else None,
        "conf_p25": _percentile(confidences, 25) if confidences else None,
        "conf_count": len(confidences),
        "empty_stt_count": empty_stt_count,
        "hallucination_count": hallucination_count,
        "low_conf_count": low_conf_count,
        "low_conf_pct": round(low_conf_count / total_chunks * 100, 1) if total_chunks > 0 else 0,
        "total_chunks": total_chunks,
    }


def compute_translation_kpis(rows: list[dict], use_comet: bool = False) -> dict:
    """KPI 3: Translation quality from QE scores."""
    qe_a_values = []
    qe_b_values = []

    for row in rows:
        qe_a = _safe_float(row.get("qe_a"))
        if qe_a is not None:
            qe_a_values.append(qe_a)
        qe_b = _safe_float(row.get("qe_b"))
        if qe_b is not None:
            qe_b_values.append(qe_b)

    result = {
        "qe_a_mean": statistics.mean(qe_a_values) if qe_a_values else None,
        "qe_a_median": statistics.median(qe_a_values) if qe_a_values else None,
        "qe_a_p10": _percentile(qe_a_values, 10) if qe_a_values else None,
        "qe_a_count": len(qe_a_values),
        "qe_b_mean": statistics.mean(qe_b_values) if qe_b_values else None,
        "qe_b_count": len(qe_b_values),
        "comet_mean": None,
        "comet_median": None,
        "comet_p10": None,
    }

    if use_comet:
        comet_scores = _run_cometkiwi(rows)
        if comet_scores:
            result["comet_mean"] = statistics.mean(comet_scores)
            result["comet_median"] = statistics.median(comet_scores)
            result["comet_p10"] = _percentile(comet_scores, 10)

    return result


def _run_cometkiwi(rows: list[dict]) -> list[float] | None:
    """Run CometKiwi reference-free QE on (english, spanish_a) pairs.

    Returns list of scores, or None if unbabel-comet is not installed.
    """
    try:
        from comet import download_model, load_from_checkpoint
    except ImportError:
        print("  [kpi] unbabel-comet not installed — skipping CometKiwi", file=sys.stderr)
        print("  [kpi] Install with: pip install unbabel-comet", file=sys.stderr)
        return None

    pairs = []
    for row in rows:
        english = row.get("english", "")
        spanish = row.get("spanish_a", "")
        if english and spanish:
            pairs.append({"src": english, "mt": spanish})

    if not pairs:
        return None

    print(f"  [kpi] Running CometKiwi on {len(pairs)} pairs...", file=sys.stderr)
    model_path = download_model("Unbabel/wmt22-cometkiwi-da")
    model = load_from_checkpoint(model_path)
    output = model.predict(pairs, batch_size=32, gpus=0)
    return output.scores


def compute_stability_kpis(rows: list[dict], diag_records: list[dict]) -> dict:
    """KPI 4: Word stability from word_stability_pct or marian_similarity."""
    stability_values = []
    marian_sim_values = []

    for row in rows:
        ws = _safe_float(row.get("word_stability_pct"))
        if ws is not None:
            stability_values.append(ws)
        ms = _safe_float(row.get("marian_similarity"))
        if ms is not None:
            marian_sim_values.append(ms)

    # Also check diagnostics for word_stability_pct
    for rec in diag_records:
        if "word_stability_pct" in rec and rec["word_stability_pct"] is not None:
            val = _safe_float(rec["word_stability_pct"])
            if val is not None and not stability_values:
                # Only use diag if CSV doesn't have the column
                stability_values.append(val)

    has_stability = len(stability_values) > 0
    primary = stability_values if has_stability else marian_sim_values
    source = "word_stability_pct" if has_stability else "marian_similarity"

    return {
        "mean": statistics.mean(primary) if primary else None,
        "median": statistics.median(primary) if primary else None,
        "p10": _percentile(primary, 10) if primary else None,
        "count": len(primary),
        "source": source,
    }


def compute_reliability_kpis(rows: list[dict], diag_records: list[dict]) -> dict:
    """KPI 5: Processing success rate from completed vs attempted chunks.

    Uses diagnostics events to count drops. If no diagnostics available,
    estimates from CSV row count vs chunk_id gaps.
    """
    # Count events from diagnostics
    empty_stt = sum(1 for r in diag_records if r.get("event") == "empty_stt_drop")

    # Count completed (rows in CSV = successfully completed chunks)
    completed = len(rows)

    # Estimate attempted from chunk_ids — gaps indicate dropped chunks
    chunk_ids = []
    for row in rows:
        cid = row.get("chunk_id")
        if cid is not None and cid != "":
            try:
                chunk_ids.append(int(cid))
            except ValueError:
                pass

    if chunk_ids:
        max_id = max(chunk_ids)
        min_id = min(chunk_ids)
        # Attempted ≈ range of IDs (since IDs are sequential)
        estimated_attempted = max_id - min_id + 1
    else:
        estimated_attempted = completed

    # Also count from session summary record in diagnostics
    session_summary = None
    for rec in diag_records:
        if rec.get("event") == "session_summary":
            session_summary = rec
            break

    if session_summary:
        attempted = session_summary.get("chunks_attempted", estimated_attempted)
        actual_completed = session_summary.get("chunks_completed", completed)
        empty_stt = session_summary.get("chunks_empty_stt", empty_stt)
        hallucination = session_summary.get("chunks_hallucination", 0)
        dedup = session_summary.get("chunks_dedup", 0)
    else:
        attempted = estimated_attempted
        actual_completed = completed
        hallucination = 0
        dedup = 0

    rate = actual_completed / attempted * 100 if attempted > 0 else 0

    return {
        "attempted": attempted,
        "completed": actual_completed,
        "empty_stt": empty_stt,
        "hallucination": hallucination,
        "dedup": dedup,
        "rate": round(rate, 1),
    }


def grade(value: float | None, excellent: float, target: float, higher_is_better: bool = True) -> str:
    """Grade a KPI value as EXCELLENT / PASS / NEEDS WORK."""
    if value is None:
        return "N/A"
    if higher_is_better:
        if value >= excellent:
            return "EXCELLENT"
        if value >= target:
            return "PASS"
        return "NEEDS WORK"
    else:
        # Lower is better (e.g., latency)
        if value <= excellent:
            return "EXCELLENT"
        if value <= target:
            return "PASS"
        return "NEEDS WORK"


def generate_scorecard(
    latency: dict,
    wer: dict,
    translation: dict,
    stability: dict,
    reliability: dict,
) -> str:
    """Generate a markdown scorecard table."""
    lines = []
    lines.append("| # | KPI | Value | Target | Grade |")
    lines.append("|---|-----|-------|--------|-------|")

    # 1. Latency
    e2e_p95 = latency["e2e_p95"]
    e2e_grade = grade(e2e_p95, 2000, 3000, higher_is_better=False)
    lines.append(f"| 1 | E2E Latency P95 | {e2e_p95:.0f}ms | <3000ms | {e2e_grade} |")

    # 2. WER proxy (confidence)
    conf_mean = wer["conf_mean"]
    conf_grade = grade(conf_mean, 0.80, 0.65)
    conf_str = f"{conf_mean:.2f}" if conf_mean is not None else "N/A"
    lines.append(f"| 2 | STT Confidence (mean) | {conf_str} | >0.65 | {conf_grade} |")

    # 3. Translation QE
    qe_mean = translation["qe_a_mean"]
    qe_grade = grade(qe_mean, 0.95, 0.85)
    qe_str = f"{qe_mean:.2f}" if qe_mean is not None else "N/A"
    lines.append(f"| 3 | Translation QE (mean) | {qe_str} | >0.85 | {qe_grade} |")

    # 4. Word Stability
    stab_mean = stability["mean"]
    stab_grade = grade(stab_mean, 0.98, 0.95)
    stab_str = f"{stab_mean:.2f}" if stab_mean is not None else "N/A"
    stab_label = f"Word Stability ({stability['source']})"
    lines.append(f"| 4 | {stab_label} | {stab_str} | >0.95 | {stab_grade} |")

    # 5. Reliability
    rate = reliability["rate"]
    rel_grade = grade(rate, 99.5, 98.0)
    lines.append(f"| 5 | Processing Rate | {rate:.1f}% | >98% | {rel_grade} |")

    return "\n".join(lines)


def generate_report(
    csv_paths: list[str],
    latency: dict,
    wer: dict,
    translation: dict,
    stability: dict,
    reliability: dict,
    scorecard: str,
) -> str:
    """Generate full markdown KPI report."""
    lines = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines.append(f"# KPI Report — {now}")
    lines.append("")
    sources = ", ".join(f"`{os.path.basename(p)}`" for p in csv_paths)
    lines.append(f"**Source:** {sources}")
    lines.append("")

    # Scorecard
    lines.append("## Scorecard")
    lines.append("")
    lines.append(scorecard)
    lines.append("")

    # Detail: Latency
    lines.append("## 1. Latency Detail")
    lines.append("")
    lines.append(
        f"- E2E: P50={latency['e2e_p50']:.0f}ms, P95={latency['e2e_p95']:.0f}ms, P99={latency['e2e_p99']:.0f}ms ({latency['e2e_count']} chunks)"
    )
    lines.append(f"- STT: P50={latency['stt_p50']:.0f}ms, P95={latency['stt_p95']:.0f}ms")
    lines.append(f"- Translation A: P50={latency['trans_a_p50']:.0f}ms, P95={latency['trans_a_p95']:.0f}ms")
    if latency["trans_b_p50"] is not None:
        lines.append(f"- Translation B: P50={latency['trans_b_p50']:.0f}ms, P95={latency['trans_b_p95']:.0f}ms")
    if latency["stall_count"] > 0:
        lines.append(f"- Queue stalls (>10s): {latency['stall_count']}")
    lines.append("")

    # Detail: WER
    lines.append("## 2. WER Proxy Detail")
    lines.append("")
    if wer["conf_mean"] is not None:
        lines.append(
            f"- Confidence: mean={wer['conf_mean']:.2f}, P10={wer['conf_p10']:.2f}, P25={wer['conf_p25']:.2f} ({wer['conf_count']} chunks)"
        )
    lines.append(f"- Low confidence (<0.5): {wer['low_conf_count']}/{wer['total_chunks']} ({wer['low_conf_pct']}%)")
    lines.append(f"- Empty STT drops: {wer['empty_stt_count']}")
    lines.append(f"- Hallucinations detected: {wer['hallucination_count']}")
    lines.append("")

    # Detail: Translation
    lines.append("## 3. Translation Quality Detail")
    lines.append("")
    if translation["qe_a_mean"] is not None:
        lines.append(
            f"- QE A: mean={translation['qe_a_mean']:.2f}, median={translation['qe_a_median']:.2f}, P10={translation['qe_a_p10']:.2f} ({translation['qe_a_count']} chunks)"
        )
    if translation["qe_b_mean"] is not None:
        lines.append(f"- QE B: mean={translation['qe_b_mean']:.2f} ({translation['qe_b_count']} chunks)")
    if translation["comet_mean"] is not None:
        lines.append(
            f"- CometKiwi: mean={translation['comet_mean']:.3f}, median={translation['comet_median']:.3f}, P10={translation['comet_p10']:.3f}"
        )
    lines.append("")

    # Detail: Stability
    lines.append("## 4. Word Stability Detail")
    lines.append("")
    if stability["mean"] is not None:
        lines.append(f"- Source: `{stability['source']}`")
        lines.append(
            f"- Mean={stability['mean']:.2f}, Median={stability['median']:.2f}, P10={stability['p10']:.2f} ({stability['count']} chunks)"
        )
    else:
        lines.append("- No stability data available")
    lines.append("")

    # Detail: Reliability
    lines.append("## 5. Processing Rate Detail")
    lines.append("")
    lines.append(f"- Attempted: {reliability['attempted']}")
    lines.append(f"- Completed: {reliability['completed']}")
    lines.append(f"- Rate: {reliability['rate']}%")
    if reliability["empty_stt"] or reliability["hallucination"] or reliability["dedup"]:
        lines.append(
            f"- Drops: empty_stt={reliability['empty_stt']}, hallucination={reliability['hallucination']}, dedup={reliability['dedup']}"
        )
    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate KPI scorecard from session CSV data")
    parser.add_argument("csv_files", nargs="*", help="CSV file(s) to analyze")
    parser.add_argument("--latest", action="store_true", help="Use the most recent CSV file")
    parser.add_argument("--output", "-o", help="Write report to file instead of stdout")
    parser.add_argument("--comet", action="store_true", help="Run CometKiwi QE (requires unbabel-comet)")
    args = parser.parse_args()

    # Resolve CSV files
    csv_paths = []
    if args.latest:
        pattern = "metrics/ab_metrics_*.csv"
        matches = sorted(glob.glob(pattern))
        if not matches:
            print(f"No CSV files matching {pattern}", file=sys.stderr)
            sys.exit(1)
        csv_paths = [matches[-1]]
        print(f"Using latest: {csv_paths[0]}", file=sys.stderr)
    elif args.csv_files:
        csv_paths = args.csv_files
    else:
        parser.print_help()
        sys.exit(1)

    # Load all CSV rows
    all_rows = []
    for path in csv_paths:
        if not os.path.exists(path):
            print(f"File not found: {path}", file=sys.stderr)
            sys.exit(1)
        all_rows.extend(load_csv(path))

    if not all_rows:
        print("No data rows found in CSV files", file=sys.stderr)
        sys.exit(1)

    # Load diagnostics
    all_diag = []
    for path in csv_paths:
        diag_path = find_diagnostics_for_csv(path)
        if diag_path:
            all_diag.extend(load_diagnostics(diag_path))

    # Compute KPIs
    latency = compute_latency_kpis(all_rows)
    wer = compute_wer_kpis(all_rows, all_diag)
    translation = compute_translation_kpis(all_rows, use_comet=args.comet)
    stability = compute_stability_kpis(all_rows, all_diag)
    reliability = compute_reliability_kpis(all_rows, all_diag)

    # Generate report
    scorecard = generate_scorecard(latency, wer, translation, stability, reliability)
    report = generate_report(csv_paths, latency, wer, translation, stability, reliability, scorecard)

    if args.output:
        with open(args.output, "w") as f:
            f.write(report)
        print(f"Report saved to {args.output}", file=sys.stderr)
    else:
        print(report)


if __name__ == "__main__":
    main()
