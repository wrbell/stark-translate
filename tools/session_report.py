"""Offline report for one finished or crashed session. It reads files only and loads no model.

    python -m tools.session_report --session <sid> [--metrics-dir metrics] [--data-dir stark_data] [--top 20]
    python -m tools.session_report --latest [--stdout]

Writes metrics/session_report_<sid>.json and .md. Every value comes from a file the session wrote;
a missing input gives null and a reason in "missing".
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from tools.kpi_report import _percentile, _safe_float
from tools.pipeline_timing import TIMING_COLUMNS
from tools.review_data import ReviewStore, read_jsonl
from tools.session_lifecycle import _path, _read, _write, session_status

SCHEMA_VERSION = 1
# Schema 2 server timing stays apart from the legacy processing measures (tools/AGENTS.md).
SCHEMA2_LATENCY = tuple(c for c in TIMING_COLUMNS if c.endswith("_ms"))
LEGACY_LATENCY = ("true_e2e_ms", "e2e_latency_ms", "stt_latency_ms", "latency_a_ms")
SUMMARY_COUNTERS = ("chunks_attempted", "chunks_completed", "chunks_empty_stt", "chunks_hallucination", "chunks_dedup")


def _root(metrics_dir: Path, data_dir: Path) -> Path:
    root = metrics_dir.resolve().parent
    if metrics_dir.resolve() != root / "metrics" or data_dir.resolve() != root / "stark_data":
        raise ValueError("The lifecycle and review readers need <root>/metrics and <root>/stark_data")
    return root


def _latency(records: list[dict], field: str) -> dict:
    values = [v for r in records if (v := _safe_float(r.get(field))) is not None]
    if not values:
        return {"n": 0, "p50": None, "p95": None}
    return {"n": len(values), "p50": round(_percentile(values, 50), 1), "p95": round(_percentile(values, 95), 1)}


def _flags(record: dict) -> list[str]:
    flags = ["hallucination"] if record.get("is_hallucination") else []
    flags += [
        f"{k}:{len(record[k])}" for k in ("low_confidence_words", "homophone_flags", "near_miss_flags") if record.get(k)
    ]
    return flags + (["bad_split"] if record.get("bad_split") else [])


def _training_blockers(status: dict, approved: list[dict], registry: list[dict], session: str) -> list[str]:
    # The same refusals as a train-split ReviewStore.export (tools/review_data.py, export()).
    reasons = [] if status["exportable"] else [status["reason"]]
    if not approved:
        reasons.append("No chunk is human-approved")
    if any(not r["source_lang"] for r in approved):
        reasons.append("An approved chunk has no source language")
    if any(
        r["session_kind"] != "live"
        or r.get("dataset_split", r.get("split")) in ("eval", "test", "holdout")
        or r.get("is_eval")
        for r in approved
    ):
        reasons.append("Training exports need live-session provenance; replay or unknown sessions are evaluation only")
    if any(row.get("session") == session and row.get("split") != "train" for row in registry):
        reasons.append("This session is already assigned to the eval split")
    return reasons


def build_session_report(session_id: str, metrics_dir: Path, data_dir: Path, *, top: int = 20) -> dict:
    """Read the session files and return the report. This function writes nothing."""
    root = _root(Path(metrics_dir), Path(data_dir))
    missing: dict[str, str] = {}
    lifecycle, status = _read(root, session_id), session_status(root, session_id)
    if not lifecycle:
        missing["lifecycle"] = "session_lifecycle file is missing or unreadable"
    store = ReviewStore(root)
    diagnostics = store.diagnostics_path(session_id)
    try:
        records = store.records(session_id)
    except FileNotFoundError:
        records, missing["diagnostics"] = [], "diagnostics file is missing"
    # read_jsonl skips incomplete or unreadable lines silently; count them here.
    rows = read_jsonl(diagnostics, complete_only=True)
    skipped = len(diagnostics.read_bytes().splitlines()) - len(rows) if diagnostics.is_file() else 0
    events = [r for r in rows if "event" in r]
    summary = next((r for r in reversed(events) if r["event"] == "session_summary"), None)
    if summary is None:
        missing["session_summary"] = "no session_summary event; the pipeline did not reach its summary step"
    try:
        metadata = json.loads(_path(root, session_id, "session_metadata", "json").read_text())
    except (OSError, ValueError):
        metadata, missing["session_metadata"] = {}, "session_metadata file is missing or unreadable"
    try:
        duration = datetime.fromisoformat(lifecycle["ended_at"]) - datetime.fromisoformat(lifecycle["started_at"])
        duration_s: float | None = round(duration.total_seconds(), 1)
    except (KeyError, TypeError, ValueError):
        duration_s, missing["duration_s"] = None, "lifecycle has no started_at or ended_at"
    memory, persistence = lifecycle.get("memory") or {}, lifecycle.get("persistence")
    git_sha = lifecycle.get("git_sha") or metadata.get("git_sha")
    for key, value in (*((k, memory.get(k)) for k in ("peak_rss_bytes", "peak_metal_bytes")), ("git_sha", git_sha)):
        if value is None:
            missing[key] = "not recorded (session not finished, or the counter was not available)"
    if persistence is None:
        missing["write_failures"] = "session_lifecycle has no persistence ledger"

    def audio_exists(record: dict) -> bool:
        try:
            store.audio_path(session_id, record["chunk_id"], record)
            return True
        except (FileNotFoundError, ValueError):
            return False

    pending = [r for r in records if not r["excluded"] and not (r["transcript_approved"] and r["translation_approved"])]
    pending.sort(key=lambda r: (-(_safe_float(r.get("review_priority")) or 0), r["chunk_id"]))
    approved = [r for r in records if not r["excluded"] and (r["transcript_approved"] or r["translation_approved"])]
    registry = read_jsonl(store.corrections / "export_registry.jsonl")
    blockers = _training_blockers(status, approved, registry, session_id)
    return {
        "schema_version": SCHEMA_VERSION,
        "session_id": session_id,
        "inputs": {"diagnostics": str(diagnostics.relative_to(root)), "diagnostics_skipped_lines": skipped},
        "run": {
            "status": status["status"],
            "recorded_status": lifecycle.get("status"),
            "reason": status["reason"],
            "exit_code": lifecycle.get("exit_code"),
            "started_at": lifecycle.get("started_at"),
            "ended_at": lifecycle.get("ended_at"),
            "duration_s": duration_s,
            "peak_rss_bytes": memory.get("peak_rss_bytes"),
            "peak_metal_bytes": memory.get("peak_metal_bytes"),
            "git_sha": git_sha,
            "models": {
                role: {k: item.get(k) for k in ("requested_id", "manifest_revision", "resolved_revision")}
                for role, item in (lifecycle.get("models") or {}).items()
            },
            **{k: metadata.get(k) for k in ("audio_source", "session_kind", "source_lang", "target_lang")},
        },
        "counts": {
            "finals": len(records),
            "finals_by_route": dict(Counter(r.get("final_translation_route") or "unrecorded" for r in records)),
        },
        "latency_ms": {
            "schema2": {f: _latency(records, f) for f in SCHEMA2_LATENCY},
            "legacy_processing": {f: _latency(records, f) for f in LEGACY_LATENCY},
        },
        "problems": {
            "write_failures": {k: persistence.get(k) for k in ("failed", "failures")} if persistence else None,
            "hallucination_flags": sum(bool(r.get("is_hallucination")) for r in records),
            "chunks_with_low_confidence_words": sum(bool(r.get("low_confidence_words")) for r in records),
            "empty_stt_drop_events": dict(Counter(e.get("stage") for e in events if e["event"] == "empty_stt_drop")),
            "session_summary": None
            if summary is None
            else {
                **{k: summary.get(k) for k in SUMMARY_COUNTERS},
                "source_coverage_outcomes": dict(
                    Counter(o.get("state") for o in (summary.get("source_coverage") or {}).get("outcomes", []))
                ),
            },
        },
        "review": {
            "pending": len(pending),
            "top": [
                {
                    "chunk_id": r["chunk_id"],
                    "review_priority": r.get("review_priority"),
                    "flags": _flags(r),
                    "source_text": r["source_text"],
                    "translation_text": r["translation_text"],
                    "audio_path": r.get("audio_path"),
                    "audio_exists": audio_exists(r),
                }
                for r in pending[:top]
            ],
        },
        "integrity": {
            "diagnostics_sha256": (lifecycle.get("diagnostics") or {}).get("sha256"),
            "audio_not_retained": sum(not r.get("audio_path") for r in records),
            "audio_missing": sum(bool(r.get("audio_path")) and not audio_exists(r) for r in records),
        },
        "training": {
            "reviewed": sum(r["revision"] > 0 for r in records),
            "transcript_approved": sum(r["transcript_approved"] for r in records),
            "translation_approved": sum(r["translation_approved"] for r in records),
            "approved": len(approved),
            "excluded": sum(r["excluded"] for r in records),
            "ready_for_training": not blockers,
            "reasons": blockers,
        },
        "missing": missing,
    }


def _cell(value: object, width: int = 60) -> str:
    text = " ".join(str(value if value is not None else "").split()).replace("|", "/")
    return text if len(text) <= width else text[: width - 3] + "..."


def _pairs(values: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in values.items() if not isinstance(v, dict)) or "none"


def _shown(value: object, unit: str = "") -> str:
    return "not recorded" if value is None else f"{value}{unit}"


def render_markdown(report: dict) -> str:
    run, problems, training = report["run"], report["problems"], report["training"]
    gib = {
        k: f"{run[k] / 2**30:.2f} GiB" if run[k] is not None else "not recorded"
        for k in ("peak_rss_bytes", "peak_metal_bytes")
    }
    latency = [
        f"{field} {stats['p50']} / {stats['p95']} (n={stats['n']})"
        for group in report["latency_ms"].values()
        for field, stats in group.items()
        if stats["n"]
    ]
    summary, failures = problems["session_summary"], problems["write_failures"]
    lines = [
        f"# Session report {report['session_id']}",
        "",
        f"- Status: {run['status']} (recorded: {run['recorded_status'] or 'none'}). {run['reason']}".rstrip(),
        f"- Duration: {_shown(run['duration_s'], ' s')}. Finals: {report['counts']['finals']} "
        f"({_pairs(report['counts']['finals_by_route'])}).",
        f"- Peak memory: RSS {gib['peak_rss_bytes']}, Metal {gib['peak_metal_bytes']}. Git: {_shown(run['git_sha'])}.",
        "- Models: "
        + (
            ", ".join(
                f"{k} {_shown(v['resolved_revision'] or v['manifest_revision'])}" for k, v in run["models"].items()
            )
            or "not recorded"
        ),
        "- Latency p50 / p95 ms: " + ("; ".join(latency) or "no timed finals"),
        f"- Write failures: {failures['failed'] if failures else 'not recorded'}. "
        f"Hallucination flags: {problems['hallucination_flags']}. "
        f"Low-confidence chunks: {problems['chunks_with_low_confidence_words']}. "
        f"Empty STT drops by stage: {_pairs(problems['empty_stt_drop_events'])}.",
        "- Summary counters: "
        + (_pairs({**summary, **summary["source_coverage_outcomes"]}) if summary else "not recorded"),
        f"- Audio: {report['integrity']['audio_missing']} missing, "
        f"{report['integrity']['audio_not_retained']} not retained. "
        f"Skipped diagnostics lines: {report['inputs']['diagnostics_skipped_lines']}.",
        f"- Training: {'READY' if training['ready_for_training'] else 'NOT READY'}. Approved {training['approved']}, "
        f"reviewed {training['reviewed']}, excluded {training['excluded']}. {' '.join(training['reasons'])}".rstrip(),
        *(f"- Missing {key}: {reason}." for key, reason in report["missing"].items()),
        "",
        f"## Review first ({len(report['review']['top'])} of {report['review']['pending']} pending)",
        "",
        "| Chunk | Priority | Flags | Source | Translation | Audio |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for item in report["review"]["top"]:
        audio = (
            "missing"
            if item["audio_path"] and not item["audio_exists"]
            else _cell(item["audio_path"] or "not retained")
        )
        lines.append(
            f"| {item['chunk_id']} | {item['review_priority']} | {_cell(', '.join(item['flags']))} | "
            f"{_cell(item['source_text'])} | {_cell(item['translation_text'])} | {audio} |"
        )
    return "\n".join(lines) + "\n"


def _write_text(path: Path, text: str) -> None:
    # session_lifecycle._write writes JSON only; Markdown uses the same temp file, fsync and replace steps.
    fd, name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--session")
    target.add_argument("--latest", action="store_true", help="Use the newest session_lifecycle file")
    parser.add_argument("--metrics-dir", type=Path, default=Path("metrics"))
    parser.add_argument("--data-dir", type=Path, default=Path("stark_data"))
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--stdout", action="store_true", help="Print the JSON report and write no file")
    args = parser.parse_args(argv)
    session = args.session
    if args.latest:
        markers = sorted(args.metrics_dir.glob("session_lifecycle_*.json"), key=lambda p: p.stat().st_mtime)
        if not markers:
            parser.error(f"No session_lifecycle file in {args.metrics_dir}")
        session = markers[-1].stem.removeprefix("session_lifecycle_")
    report = build_session_report(session, args.metrics_dir, args.data_dir, top=args.top)
    report["generated_at"] = datetime.now(UTC).isoformat()
    if args.stdout:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0
    json_path = _path(_root(args.metrics_dir, args.data_dir), session, "session_report", "json")
    _write(json_path, report)
    _write_text(json_path.with_suffix(".md"), render_markdown(report))
    print(json_path)
    print(json_path.with_suffix(".md"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
