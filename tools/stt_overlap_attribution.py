"""Attribute recorded STT/translation overlap on the shared session ms clock.

Stdlib only; no inference. Overlap is union duration, not a causal attribution.
CPU time is the calling thread's time (not GPU time or native worker CPU).
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict, deque
from itertools import pairwise
from pathlib import Path


def _number(value):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else None


def _pct(values, percentile):
    values = sorted(v for v in values if _number(v) is not None)
    return values[max(0, math.ceil(len(values) * percentile / 100) - 1)] if values else None


def _interval(start, end):
    return (start, end) if _number(start) is not None and _number(end) is not None and end >= start else None


def overlap_ms(interval, others):
    """Intersect with the union of other intervals, counting concurrent work once."""
    if interval is None:
        return None
    pieces = sorted(
        (max(interval[0], a), min(interval[1], b)) for a, b in others if a < interval[1] and b > interval[0]
    )
    total, end = 0.0, interval[0]
    for a, b in pieces:
        total += max(0, b - max(a, end))
        end = max(end, b)
    return total


def _physical_intervals(events):
    pending = defaultdict(deque)
    paired = []
    unmatched_finishes = 0
    for event in events:
        name, kind = event.get("event"), event.get("kind")
        if name not in {"physical_stt_started", "physical_stt_finished"} or kind not in {"partial", "final"}:
            continue
        identity = (
            ("chunk", event.get("chunk_id"))
            if kind == "final"
            else (
                ("request", event["request_sequence"])
                if event.get("request_sequence") is not None
                else ("utterance", event.get("utterance_id"))
            )
        )
        key = (kind, *identity)
        if name.endswith("_started"):
            pending[key].append(event)
        elif pending[key]:
            start = pending[key].popleft()
            interval = _interval(start.get("at_ms"), event.get("at_ms"))
            if interval is not None:
                paired.append({"start": start, "finish": event, "interval": interval})
            else:
                unmatched_finishes += 1
        else:
            unmatched_finishes += 1
    return paired, {
        "unmatched_stt_starts": sum(map(len, pending.values())),
        "unmatched_stt_finishes": unmatched_finishes,
    }


def _buckets(rows, elapsed_key, overlap_keys):
    result = {}
    for bucket in ("slow", "normal"):
        selected = [
            r for r in rows if _number(r.get(elapsed_key)) is not None and (r[elapsed_key] > 800) == (bucket == "slow")
        ]
        medians = {}
        for key in overlap_keys:
            shares = [
                r[key] / r[elapsed_key] for r in selected if r[elapsed_key] > 0 and _number(r.get(key)) is not None
            ]
            medians[key.removesuffix("_ms") + "_share"] = statistics.median(shares) if shares else None
        result[bucket] = {"count": len(selected), "median_overlap_shares": medians}
    return result


def aggregate(files):
    finals = [r for f in files for r in f["finals"]]
    gemma = [r for f in files for r in f["gemma_finals"]]
    decode_total = sum(r["decode_elapsed_ms"] for r in gemma if r["decode_elapsed_ms"] is not None)
    endpoints = {}
    for endpoint in ("silence", "smart_cut", "hard_cut"):
        rows = [r for r in finals if r["endpoint_reason"] == endpoint]
        endpoints[endpoint] = {
            "count": len(rows),
            "speech_end_to_final_ms": {f"p{p}": _pct([r["speech_end_to_final_ms"] for r in rows], p) for p in (50, 95)},
            "gen_tokens_a_p95": _pct([r["gen_tokens_a"] for r in rows], 95),
        }
    return {
        "counts": {
            "files": len(files),
            "finals": len(finals),
            "gemma_finals": len(gemma),
            "final_stt_intervals": sum(r["stt_elapsed_ms"] is not None for r in finals),
            "gemma_intervals": sum(r["decode_elapsed_ms"] is not None for r in gemma),
            "partial_stt_calls": sum(f["partial_stt_calls"] for f in files),
            "partial_stt_calls_translation_active": sum(f["partial_stt_calls_translation_active"] for f in files),
        },
        "stt_buckets": _buckets(finals, "stt_elapsed_ms", ("gemma_overlap_ms", "partial_stt_overlap_ms")),
        "gemma_buckets": _buckets(gemma, "decode_elapsed_ms", ("partial_stt_overlap_ms", "next_final_stt_overlap_ms")),
        "gemma_decode_partial_overlap_share": sum(r["partial_stt_overlap_ms"] or 0 for r in gemma) / decode_total
        if decode_total
        else None,
        "endpoint_mix": endpoints,
    }


def analyze_file(path: Path):
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    summaries = [r for r in records if r.get("event") == "session_summary"]
    if len(summaries) != 1:
        raise ValueError(f"Expected one session_summary in {path}")
    trace = summaries[0].get("latency_trace", {})
    if trace.get("schema_version", 2) != 2:
        raise ValueError(f"Trace does not use the shared schema-2 clock: {path}")
    events = trace.get("events", [])
    rows = [r for r in records if r.get("endpoint_reason") and r.get("timing_schema_version") == 2]
    physical, incomplete = _physical_intervals(events)
    final_stt = {p["start"].get("chunk_id"): p for p in physical if p["start"]["kind"] == "final"}
    partials = [p["interval"] for p in physical if p["start"]["kind"] == "partial"]
    translations = {}
    for row in rows:
        stages = row.get("timing_stages_ms") or {}
        if row.get("final_translation_route") == "gemma":
            translations[row.get("chunk_id")] = _interval(
                stages.get("translation_started"), stages.get("translation_finished")
            )
    cpu = defaultdict(list)
    for event in events:
        if (
            event.get("event") == "physical_translation_finished"
            and event.get("route") == "gemma"
            and _number(event.get("cpu_ms")) is not None
        ):
            cpu[event.get("chunk_id")].append(event["cpu_ms"])
    order = sorted(final_stt, key=lambda cid: final_stt[cid]["interval"][0])
    next_stt = {cid: final_stt[next_cid]["interval"] for cid, next_cid in pairwise(order)}
    finals, gemma = [], []
    for row in rows:
        cid = row.get("chunk_id")
        physical_row = final_stt.get(cid, {})
        interval = physical_row.get("interval")
        final = {
            "chunk_id": cid,
            "utterance_id": row.get("utterance_id"),
            "endpoint_reason": row["endpoint_reason"],
            "speech_end_to_final_ms": row.get("speech_end_to_final_ms"),
            "gen_tokens_a": row.get("gen_tokens_a"),
            "stt_elapsed_ms": interval[1] - interval[0] if interval else None,
            "gemma_overlap_ms": overlap_ms(
                interval, [v for k, v in translations.items() if k != cid and v is not None]
            ),
            "partial_stt_overlap_ms": overlap_ms(interval, partials),
            **{
                key: physical_row.get("start", {}).get(key)
                for key in ("translation_active", "concurrent_partial", "concurrent_final")
            },
            "cpu_ms": physical_row.get("finish", {}).get("cpu_ms"),
        }
        finals.append(final)
        if cid in translations:
            interval = translations[cid]
            tokens, decode_ms = _number(row.get("gen_tokens_a")), _number(row.get("decode_ms_a"))
            gemma.append(
                {
                    "chunk_id": cid,
                    "utterance_id": row.get("utterance_id"),
                    "decode_elapsed_ms": interval[1] - interval[0] if interval else None,
                    "gen_tokens_a": tokens,
                    "tokens_per_second": tokens / (decode_ms / 1000)
                    if tokens is not None and decode_ms is not None and decode_ms > 0
                    else None,
                    "partial_stt_overlap_ms": overlap_ms(interval, partials),
                    "next_final_stt_overlap_ms": overlap_ms(interval, [next_stt[cid]] if cid in next_stt else []),
                    "cpu_ms": sum(cpu[cid]) if cpu[cid] else None,
                }
            )
    partial_starts = [e for e in events if e.get("event") == "physical_stt_started" and e.get("kind") == "partial"]
    result = {
        "diagnostics": str(path),
        "finals": finals,
        "gemma_finals": gemma,
        "partial_stt_calls": len(partial_starts),
        "partial_stt_calls_translation_active": sum(e.get("translation_active") is True for e in partial_starts),
        "trace_quality": {
            "enabled": trace.get("enabled"),
            "discarded_old_events": trace.get("discarded_old_events", 0),
            **incomplete,
        },
        "manifest_sha256": summaries[0].get("manifest_sha256"),
    }
    result["aggregates"] = aggregate([result])
    return result


def analyze(paths):
    files = [analyze_file(Path(path)) for path in paths]
    hashes = {f["manifest_sha256"] for f in files if f["manifest_sha256"] is not None}
    if len(hashes) > 1:
        raise ValueError("Cannot pool cohorts with different manifest hashes")
    return {
        "schema_version": 1,
        "percentile_method": "nearest_rank",
        "slow_threshold_ms": 800,
        "notes": "Overlap is observational, not causal. Missing/truncated trace intervals undercount overlap. "
        "CPU measures only the calling thread; multiprocess wrappers measure parent IPC time. "
        "Pooled values describe supplied files; absent manifest hashes do not establish cohort comparability.",
        "files": files,
        "pooled": aggregate(files),
    }


def render_markdown(report):
    lines = [
        "# STT overlap attribution",
        "",
        report["notes"],
        "",
        "Slow means > 800 ms; percentiles use nearest rank.",
        "",
    ]

    def table(rows, keys):
        lines.extend(["| " + " | ".join(keys) + " |", "| " + " | ".join("---" for _ in keys) + " |"])
        for row in rows:
            values = [row.get(k) for k in keys]
            lines.append(
                "| "
                + " | ".join("—" if v is None else f"{v:.3f}" if isinstance(v, float) else str(v) for v in values)
                + " |"
            )
        lines.append("")

    for label, data in [(f["diagnostics"], f["aggregates"]) for f in report["files"]] + [("Pooled", report["pooled"])]:
        lines.extend([f"## {label}", ""])
        table([data["counts"]], list(data["counts"]))
        buckets = [
            {"stage": stage, "bucket": bucket, "count": values["count"], **values["median_overlap_shares"]}
            for stage in ("stt", "gemma")
            for bucket, values in data[stage + "_buckets"].items()
        ]
        table(buckets, list(dict.fromkeys(k for row in buckets for k in row)))
        lines.extend([f"Gemma time overlapped by partial STT: {data['gemma_decode_partial_overlap_share']}", ""])
        table(
            [
                {
                    "endpoint": k,
                    "count": v["count"],
                    **v["speech_end_to_final_ms"],
                    "gen_tokens_a_p95": v["gen_tokens_a_p95"],
                }
                for k, v in data["endpoint_mix"].items()
            ],
            ["endpoint", "count", "p50", "p95", "gen_tokens_a_p95"],
        )
    for file in report["files"]:
        lines.extend([f"## Chunk detail: {file['diagnostics']}", "", f"Trace quality: {file['trace_quality']}", ""])
        for key in ("finals", "gemma_finals"):
            lines.extend([f"### {key}", ""])
            if file[key]:
                table(file[key], list(file[key][0]))
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostics", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--markdown", required=True, type=Path)
    args = parser.parse_args(argv)
    paths = [*args.diagnostics, args.output, args.markdown]
    for i, path in enumerate(paths):
        for other in paths[:i]:
            if path.resolve() == other.resolve() or (path.exists() and other.exists() and path.samefile(other)):
                parser.error("Inputs and outputs must be distinct paths")
    report = analyze(args.diagnostics)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.markdown.write_text(render_markdown(report), encoding="utf-8")


if __name__ == "__main__":
    main()
