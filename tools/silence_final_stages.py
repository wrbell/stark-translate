"""Attribute schema-2 final-caption latency without inference or third-party imports.

The selected endpoint is reported at the top level. ``--all-endpoints`` adds
``endpoint_cohorts`` with the same structure for each observed endpoint. Checksums
are always route-specific; no latency distribution pools Gemma and Marian.
JSON retains numeric precision; Markdown displays one decimal place.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from datetime import UTC, datetime
from pathlib import Path

STAGES = {
    "silence_wait_and_vad_decision": ("speech_end", "vad_finalized"),
    "submit": ("vad_finalized", "submitted"),
    "queue": ("submitted", "dequeued"),
    "stt_dispatch": ("dequeued", "stt_started"),
    "stt_call": ("stt_started", "stt_finished"),
    "to_translation": ("stt_finished", "translation_requested"),
    "translation_lock_wait": ("translation_requested", "translation_lock_acquired"),
    "translation_prepare": ("translation_lock_acquired", "translation_started"),
    "translation_call": ("translation_started", "translation_finished"),
    "finalize": ("translation_finished", "final_ready"),
    "broadcast": ("final_ready", "broadcast_finished"),
    "total_speech_end_to_final": ("speech_end", "final_ready"),
}
SCALARS = (
    "stt_latency_ms",
    "prefill_ms_a",
    "ttft_ms_a",
    "decode_ms_a",
    "gen_tokens_a",
    "tps_a",
    "generation_lock_wait_ms_a",
    "finalization_overhead_ms",
    "broadcast_ms",
    "utterance_dur",
)
TOTAL = "total_speech_end_to_final"
SHARE_NOTE = (
    "share_of_total_p50 is an approximation: stage p50 / total p50; medians do not sum. "
    "Non-millisecond scalar rows have no share. Broadcast follows final readiness and is outside the total."
)


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def nearest_rank(values: list[float], percentile: int) -> float | None:
    """Use ascending[ceil(percentile / 100 * n) - 1], including for p50."""
    if not values:
        return None
    return sorted(values)[math.ceil(percentile / 100 * len(values)) - 1]


def _stats(values: list[float | None]) -> dict:
    present = [value for value in values if value is not None]
    return {
        "n": len(present),
        "missing": len(values) - len(present),
        "mean": statistics.mean(present) if present else None,
        "p50": nearest_rank(present, 50),
        "p95": nearest_rank(present, 95),
    }


def _delta(row: dict, start: str, end: str) -> float | None:
    stamps = row.get("timing_stages_ms")
    if not isinstance(stamps, dict):
        return None
    first, last = _number(stamps.get(start)), _number(stamps.get(end))
    # Preserve observed deltas, including negative values, rather than clamp to zero.
    return None if first is None or last is None else last - first


def _route(row: dict) -> str:
    if "final_translation_route" in row:
        route = row["final_translation_route"]
        return route if route in ("gemma", "marian") else "unknown"
    if row.get("tps_a") in (0, None) and not row.get("spanish_gemma") and row.get("spanish_marian"):
        return "marian"
    return "gemma"


def _cohort(rows: list[dict]) -> dict:
    stages = {name: _stats([_delta(row, start, end) for row in rows]) for name, (start, end) in STAGES.items()}
    stages.update({name: _stats([_number(row.get(name)) for row in rows]) for name in SCALARS})
    total_p50 = stages[TOTAL]["p50"]
    for name, stats in stages.items():
        stats["unit"] = {"gen_tokens_a": "tokens", "tps_a": "tokens/s", "utterance_dur": "s"}.get(name, "ms")
        stats["share_of_total_p50"] = (
            stats["p50"] / total_p50
            if total_p50 not in (None, 0) and stats["p50"] is not None and stats["unit"] == "ms"
            else None
        )
    own = _stats([_number(row.get("speech_end_to_final_ms")) for row in rows])
    return {
        "jsonl_rows": len(rows),
        "route_source": (
            "mixed"
            if 0 < sum("final_translation_route" in row for row in rows) < len(rows)
            else "final_translation_route"
            if any("final_translation_route" in row for row in rows)
            else "tps_a_proxy"
        ),
        "stages": stages,
        "checksum": {
            "total_speech_end_to_final_p50": total_p50,
            "speech_end_to_final_ms_p50": own["p50"],
            "stamp_n": stages[TOTAL]["n"],
            "stamp_missing": stages[TOTAL]["missing"],
            "scalar_n": own["n"],
            "scalar_missing": own["missing"],
            "tolerance_ms": 0.5,
            "checksum_ok": (
                abs(total_p50 - own["p50"]) <= 0.5 if total_p50 is not None and own["p50"] is not None else None
            ),
        },
    }


def _endpoint_report(rows: list[dict]) -> dict:
    cohorts = {route: _cohort([row for row in rows if _route(row) == route]) for route in ("gemma", "marian")}
    if any(_route(row) == "unknown" for row in rows):
        cohorts["unknown"] = _cohort([row for row in rows if _route(row) == "unknown"])
    ranked = [{"stage": name, **cohorts["gemma"]["stages"][name]} for name in STAGES]
    ranked.sort(key=lambda row: (row["p50"] is None, -row["p50"] if row["p50"] is not None else 0))
    return {
        "jsonl_rows": len(rows),
        "cohorts": cohorts,
        "ranked": ranked,
        "checksum": {route: cohort["checksum"] for route, cohort in cohorts.items()},
    }


def analyze(
    diagnostics: Path,
    label: str,
    *,
    endpoint: str = "silence",
    all_endpoints: bool = False,
    csv_path: Path | None = None,
) -> dict:
    """Read inputs only; hash the exact bytes that supplied the observations."""
    raw = diagnostics.read_bytes()
    captions = []
    for line_number, line in enumerate(raw.splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except (ValueError, UnicodeError) as exc:
            raise ValueError(f"{diagnostics}:{line_number}: invalid JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{diagnostics}:{line_number}: expected a JSON object")
        if "event" not in row and type(row.get("timing_schema_version")) is int and row["timing_schema_version"] == 2:
            captions.append(row)

    def reason(row: dict) -> str:
        return str(row.get("endpoint_reason") or "unknown")

    report = {
        "label": label,
        "source_file": str(diagnostics),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "filters": {
            "timing_schema_version": 2,
            "caption": "no event key",
            "endpoint_reason": endpoint,
            "all_endpoints": all_endpoints,
        },
        **_endpoint_report([row for row in captions if reason(row) == endpoint]),
        "share_note": SHARE_NOTE,
        "generated_at_utc": datetime.now(UTC).isoformat(),
    }
    if all_endpoints:
        report["endpoint_cohorts"] = {
            name: _endpoint_report([row for row in captions if reason(row) == name])
            for name in sorted({reason(row) for row in captions})
        }
    if csv_path is not None:
        with csv_path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if "endpoint_reason" not in (reader.fieldnames or []):
                raise ValueError(f"{csv_path}: CSV requires endpoint_reason column")
            report["csv_rows"] = sum(row.get("endpoint_reason") == "silence" for row in reader)
        report["csv_endpoint_reason"] = "silence"
        report["jsonl_silence_rows"] = sum(reason(row) == "silence" for row in captions)
    return report


def _format(value: float | None) -> str:
    return "—" if value is None else f"{value:.1f}"


def _escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def render_markdown(report: dict) -> str:
    """Render distributions and a Gemma stamp-stage ranking for each endpoint."""
    lines = [
        f"# {_escape(report['label'])}",
        "",
        f"Source: `{_escape(report['source_file'])}`",
        f"SHA256: `{report['sha256']}`",
        f"Generated at UTC: {report['generated_at_utc']}",
        "",
        "Nearest-rank percentiles. Durations are milliseconds except utterance_dur (seconds); "
        "gen_tokens_a is tokens and tps_a is tokens/second. Missing statistics are shown as —.",
        "",
        SHARE_NOTE,
        "",
    ]
    if "csv_rows" in report:
        lines.extend(
            [f"Silence rows: CSV {_format(report['csv_rows'])}; JSONL {_format(report['jsonl_silence_rows'])}.", ""]
        )

    def tables(endpoint: str, data: dict) -> None:
        lines.extend([f"## Endpoint: {_escape(endpoint)}", "", f"JSONL rows: {_format(data['jsonl_rows'])}", ""])
        for route, cohort in data["cohorts"].items():
            lines.extend(
                [
                    f"### {route.capitalize()}",
                    "",
                    f"Route source: {cohort['route_source']}",
                    "",
                    "| stage | n | missing | mean | p50 | p95 | share (approx.) |",
                    "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
                ]
            )
            for name, stats in cohort["stages"].items():
                values = [_format(stats[key]) for key in ("n", "missing", "mean", "p50", "p95", "share_of_total_p50")]
                lines.append(f"| {name} | " + " | ".join(values) + " |")
            checksum = cohort["checksum"]
            lines.extend(
                [
                    "",
                    f"Checksum: stamp total p50 {_format(checksum['total_speech_end_to_final_p50'])} ms; "
                    f"reported p50 {_format(checksum['speech_end_to_final_ms_p50'])} ms; "
                    f"checksum_ok: {json.dumps(checksum['checksum_ok'])} (tolerance 0.5 ms).",
                    "",
                ]
            )
        lines.extend(
            [
                "### Ranked Gemma stages (p50 descending)",
                "",
                "Total is included as a reference; scalar rows are excluded from this ranking.",
                "",
                "| stage | p50 (ms) |",
                "| --- | ---: |",
            ]
        )
        lines.extend(f"| {row['stage']} | {_format(row['p50'])} |" for row in data["ranked"])
        lines.append("")

    selected = report["filters"]["endpoint_reason"]
    tables(selected, report)
    for endpoint, data in report.get("endpoint_cohorts", {}).items():
        if endpoint != selected:
            tables(endpoint, data)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--csv", type=Path, help="Count silence rows in an optional CSV")
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path)
    parser.add_argument("--endpoint", default="silence")
    parser.add_argument("--all-endpoints", action="store_true")
    args = parser.parse_args(argv)
    try:
        inputs = [args.diagnostics] + ([args.csv] if args.csv is not None else [])
        outputs = [args.output] + ([args.markdown] if args.markdown is not None else [])
        for index, output in enumerate(outputs):
            for other in inputs + outputs[:index]:
                if output.resolve() == other.resolve() or (
                    output.exists() and other.exists() and output.samefile(other)
                ):
                    raise ValueError("Output paths must be distinct from each other and from inputs")
        report = analyze(
            args.diagnostics,
            args.label,
            endpoint=args.endpoint,
            all_endpoints=args.all_endpoints,
            csv_path=args.csv,
        )
        output_json = json.dumps(report, indent=2, allow_nan=False) + "\n"
        markdown = render_markdown(report) if args.markdown is not None else None
        args.output.write_text(output_json, encoding="utf-8")
        if args.markdown is not None:
            args.markdown.write_text(markdown, encoding="utf-8")
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
