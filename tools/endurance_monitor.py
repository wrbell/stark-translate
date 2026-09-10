"""Read-only session endurance evidence, using stdlib and psutil (no model imports).

Example (launch after the session lifecycle marker exists)::

    python -m tools.endurance_monitor --root /data/project --session SESSION \
        --pid PIPELINE_PID --output /reports/SESSION --duration-seconds 3900

The output directory must be new. samples.jsonl is flushed each interval; report.json
is an atomic snapshot at attachment and monitor exit. SIGINT, SIGTERM, --stop-file,
and the deadline stop this monitor only. No process is signalled or waited/reaped.
Exit codes: 0 = verified completed recording and observed process cleanup,
2 = partial/unverified, 1 = monitor error. Sampling failures are recorded separately.
RSS sums are sampled process RSS, not unique physical RAM or a lifetime tree peak.
Source interval gaps are unclassified unless an optional JSON list of external
{start_s, end_s, label} annotations is supplied; no silence/music inference runs.
Reports omit caption text, process environments and raw logs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import signal
import socket
import statistics
import threading
import time
import wave
from collections import Counter, defaultdict
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path

from tools.session_lifecycle import _path


class OwnershipError(ValueError):
    """The marker and live process do not establish this session's ownership."""


def _json(path):
    try:
        if path.stat().st_size > 4 * 1024 * 1024:
            return {}
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError, UnicodeError):
        return {}


def _number(value):
    try:
        result = float(value) if value is not None and not isinstance(value, bool) else None
        return result if result is not None and math.isfinite(result) else None
    except (ValueError, TypeError):
        return None


def _stats(values):
    values = sorted(value for value in values if value is not None and math.isfinite(value))
    return {
        "n": len(values),
        "p50": statistics.median(values) if values else None,
        "p95": values[math.ceil(len(values) * 0.95) - 1] if values else None,
        "max": max(values) if values else None,
    }


def _marker(root, session):
    data = _json(_path(root, session, "session_lifecycle", "json"))
    if data.get("session_id") != session or data.get("schema_version") != 1:
        raise OwnershipError("Missing or invalid session lifecycle marker")
    return data


def attach(root, session, pid, ps):
    marker = _marker(root, session)
    if (
        marker.get("status") != "running"
        or marker.get("pid") != pid
        or marker.get("hostname") != socket.gethostname()
        or not marker.get("run_id")
    ):
        raise OwnershipError("Attachment requires this host's running lifecycle marker and exact pipeline PID")
    process = ps.Process(pid)
    created = process.create_time()
    try:
        started = datetime.fromisoformat(marker["started_at"]).timestamp()
    except (KeyError, ValueError, TypeError) as exc:
        raise OwnershipError("Lifecycle start time is missing or invalid") from exc
    if created > started + 1:
        raise OwnershipError("PID was created after the lifecycle marker (possible PID reuse)")
    args = process.cmdline()
    session_arg = None
    for index, arg in enumerate(args):
        if arg.startswith("--session-id="):
            session_arg = arg.split("=", 1)[1]
        elif arg == "--session-id" and index + 1 < len(args):
            session_arg = args[index + 1]
    if not any(Path(arg).name == "dry_run_ab.py" for arg in args) or session_arg != session:
        raise OwnershipError("Pipeline command does not identify this exact session")
    return {
        "session_id": session,
        "run_id": marker["run_id"],
        "pid": pid,
        "create_time": created,
        "hostname": marker["hostname"],
        "started_at": marker["started_at"],
    }


def verify_marker(marker, identity):
    if any(marker.get(key) != identity[key] for key in ("session_id", "run_id", "pid", "hostname")):
        raise OwnershipError("Session lifecycle ownership changed during monitoring")


class ProcessTreeSampler:
    """Track descendants only after observing ancestry; retain identities after reparenting."""

    def __init__(self, identity, ps):
        self.identity, self.ps = identity, ps
        self.known = {(identity["pid"], identity["create_time"]): "pipeline"}
        self.previous_cpu = {}
        self.previous_present = set()
        self.last_mono = None

    def sample(self, mono):
        errors, events, processes, discovered = [], [], {}, {}
        # Previously verified descendants can outlive the parent and spawn children.
        for key in list(self.known):
            pid, created = key
            try:
                process = self.ps.Process(pid)
                if process.create_time() != created:
                    events.append({"event": "pid_reused", "pid": pid, "create_time": created})
                    continue
                processes[key] = process
                for child in process.children(recursive=True):
                    child_key = (child.pid, child.create_time())
                    discovered[child_key] = child
            except self.ps.NoSuchProcess:
                continue
            except (self.ps.Error, OSError) as exc:
                errors.append({"pid": pid, "operation": "discover", "error": type(exc).__name__})
        for key, child in discovered.items():
            if key not in self.known:
                self.known[key] = "descendant"
                events.append({"event": "child_observed", "pid": key[0], "create_time": key[1]})
            processes[key] = child
        rows, present = [], set()
        dt = mono - self.last_mono if self.last_mono is not None else None
        for key, process in processes.items():
            try:
                # Recheck identity immediately before reading counters.
                if process.create_time() != key[1]:
                    continue
                status = process.status()
                if status in {self.ps.STATUS_ZOMBIE, self.ps.STATUS_DEAD}:
                    events.append({"event": "exit_observed", "pid": key[0], "status": status, "exit_code": None})
                    continue
                present.add(key)
                rss = process.memory_info().rss
                cpu = process.cpu_times()
                cpu_s = cpu.user + cpu.system
                previous = self.previous_cpu.get(key)
                percent = (
                    100 * max(0, cpu_s - previous[0]) / dt
                    if previous is not None and previous[1] == self.last_mono and dt and dt > 0
                    else None
                )
                self.previous_cpu[key] = (cpu_s, mono)
                rows.append(
                    {
                        "pid": key[0],
                        "create_time": key[1],
                        "role": self.known[key],
                        "rss_bytes": rss,
                        "cpu_seconds": cpu_s,
                        "cpu_percent": percent,
                        "status": status,
                    }
                )
            except self.ps.NoSuchProcess:
                present.discard(key)
            except (self.ps.Error, OSError) as exc:
                errors.append({"pid": key[0], "operation": "sample", "error": type(exc).__name__})
        for key in self.previous_present - present:
            events.append(
                {"event": "process_disappeared_or_unreadable", "pid": key[0], "create_time": key[1], "exit_code": None}
            )
        self.previous_present, self.last_mono = present, mono
        # Unknown/inaccessible counters never become an apparent zero-sized process tree.
        complete = not errors and len(rows) == len(present)
        cpu_complete = complete and bool(rows) and all(row["cpu_percent"] is not None for row in rows)
        return {
            "processes": rows,
            "process_count": len(present),
            "observed_process_count": len(rows),
            "pipeline_alive": ((self.identity["pid"], self.identity["create_time"]) in present)
            if not any(e["pid"] == self.identity["pid"] for e in errors)
            else None,
            "rss_sum_bytes": sum(row["rss_bytes"] for row in rows) if complete else None,
            "cpu_percent_sum": sum(row["cpu_percent"] for row in rows) if cpu_complete else None,
            "counters_complete": complete,
            "errors": errors,
            "events": events,
        }


def health_snapshot(root, session, wall):
    data = _json(_path(root, session, "health", "json"))
    if data.get("session_id") != session or data.get("schema_version") != 1:
        return {"available": False, "stale": True}
    result = {
        key: data[key]
        for key in (
            "updated_at",
            "phase",
            "input_seen",
            "input_age_s",
            "caption_age_s",
            "input_level",
            "errors",
            "error_count",
            "recording",
            "persistence",
            "queues",
            "clients",
            "publish_failures",
            "storage",
            "control_sequence",
            "capture",
            "operational_logging",
            "replay_client_wait",
        )
        if key in data
    }
    updated = _number(data.get("updated_at"))
    result.update(
        available=True,
        age_s=max(0, wall - updated) if updated is not None else None,
        stale=updated is None or wall - updated > 3,
    )
    result["caption_inventory"] = [
        {key: row.get(key) for key in ("chunk_id", "stage", "source_lang", "target_lang")}
        for row in data.get("captions", [])
        if isinstance(row, dict)
    ]
    return result


def _snapshot(path, issues, evidence):
    try:
        before = path.stat()
        if before.st_size > 64 * 1024 * 1024:
            raise ValueError("Artifact exceeds 64 MiB monitor limit")
        with path.open("rb") as stream:
            data = stream.read(64 * 1024 * 1024 + 1)
        if len(data) > 64 * 1024 * 1024:
            raise ValueError("Artifact exceeds 64 MiB monitor limit")
        after = path.stat()
        evidence[path.name] = {
            "sha256": hashlib.sha256(data).hexdigest(),
            "size_bytes": len(data),
            "stable_during_read": (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
        }
        return data.decode("utf-8")
    except FileNotFoundError:
        issues["missing_files"].append(path.name)
    except (OSError, ValueError, UnicodeError) as exc:
        issues["read_errors"].append({"file": path.name, "error": type(exc).__name__})
    return ""


def _records(path, session, issues, evidence):
    rows = []
    for line in _snapshot(path, issues, evidence).splitlines():
        try:
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError
            if row.get("session_id", row.get("session")) != session:
                issues["wrong_session_rows"] += 1
                continue
            rows.append(row)
        except ValueError:
            issues["malformed_rows"] += 1
    return rows


def _union(intervals):
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def source_coverage(rows, duration=None, annotations=()):
    intervals, invalid, unknown = [], 0, 0
    for row in rows:
        start, end, rate = (_number(row.get(key)) for key in ("sample_start", "sample_end", "sample_rate"))
        if None in (start, end, rate):
            unknown += 1
        elif rate <= 0 or start < 0 or end < start or (duration is not None and end / rate > duration + 0.05):
            invalid += 1
        elif end > start:
            intervals.append((start / rate, min(end / rate, duration) if duration is not None else end / rate))
    merged = _union(intervals)
    gaps, cursor = [], 0.0
    for start, end in merged:
        if start > cursor:
            gaps.append([cursor, start])
        cursor = max(cursor, end)
    if duration is not None and duration > cursor:
        gaps.append([cursor, duration])
    covered = sum(end - start for start, end in merged)
    return {
        "basis": "union of recorded half-open source sample intervals; excludes virtual padding",
        "scope": "caption windows, not all captured audio or all speech",
        "trailing_gap_known": duration is not None,
        "covered_seconds": covered,
        "source_duration_seconds": duration,
        "fraction": covered / duration if duration else None,
        "intervals_seconds": merged,
        "unknown_bounds_rows": unknown,
        "invalid_bounds_rows": invalid,
        "gaps": [
            {
                "start_s": start,
                "end_s": end,
                "duration_s": end - start,
                "classification": "unknown",
                "external_annotations": [a for a in annotations if a["start_s"] < end and a["end_s"] > start],
            }
            for start, end in gaps
        ],
        "interpretation": "Gaps may be silence, hymn/music, pause, filtering, missed speech, or unfinished replay; source listening/annotations are required.",
    }


def artifact_summary(root, session, identity, annotations=()):
    """Read artifacts once at monitor exit; group incompatible clocks/schema/sources separately."""
    issues = {"missing_files": [], "read_errors": [], "malformed_rows": 0, "wrong_session_rows": 0}
    diagnostics_path = _path(root, session, "diagnostics", "jsonl")
    partials_path = _path(root, session, "partials", "jsonl")
    evidence = {}
    diagnostics = _records(diagnostics_path, session, issues, evidence)
    partials = _records(partials_path, session, issues, evidence)
    metadata = _json(_path(root, session, "session_metadata", "json"))
    if metadata.get("session_id") != session:
        metadata = {}
    marker = _marker(root, session)
    verify_marker(marker, identity)
    source = {
        key: metadata.get(key) for key in ("input_audio_path", "input_audio_sha256", "session_kind", "audio_source")
    }
    duration = None
    if source["input_audio_path"]:
        try:
            with wave.open(str(source["input_audio_path"]), "rb") as wav:
                duration = wav.getnframes() / wav.getframerate()
                source["wav_header"] = {
                    "frames": wav.getnframes(),
                    "sample_rate": wav.getframerate(),
                    "channels": wav.getnchannels(),
                }
        except (OSError, ValueError, wave.Error, EOFError) as exc:
            source["header_error"] = type(exc).__name__
    source["audio_hash_basis"] = "recorded pipeline provenance; audio is not rehashed by monitor"
    diag_evidence = evidence.get(diagnostics_path.name, {})
    diagnostics_match = bool(diag_evidence.get("stable_during_read")) and marker.get("diagnostics") == {
        key: diag_evidence.get(key) for key in ("sha256", "size_bytes")
    }
    persistence = marker.get("persistence")
    persistence_ok = (
        isinstance(persistence, dict)
        and persistence.get("ok") is True
        and persistence.get("pending") == 0
        and persistence.get("failed") == 0
    )
    completed = (
        marker.get("status") == "completed" and marker.get("exit_code") == 0 and diagnostics_match and persistence_ok
    )
    finals, groups = [], defaultdict(lambda: {"finals": [], "partials": []})
    duplicate_ids, seen = [], set()
    for row in diagnostics:
        if row.get("chunk_id") is not None and not row.get("event"):
            if row["chunk_id"] in seen:
                duplicate_ids.append(row["chunk_id"])
                continue
            seen.add(row["chunk_id"])
            finals.append(row)
    for kind, rows in (("finals", finals), ("partials", partials)):
        for row in rows:
            cohort = tuple(
                str(row.get(key, "unknown"))
                for key in (
                    "timing_schema_version",
                    "timing_source",
                    "sample_rate",
                    "source_lang",
                    "target_lang",
                    "input_audio_sha256",
                )
            )
            groups[cohort][kind].append(row)
    cohorts = []
    for key, group in sorted(groups.items()):
        final_times = [_number(row.get("timing_stages_ms", {}).get("final_ready")) for row in group["finals"]]
        partial_times = [_number(row.get("emitted_at_ms")) for row in group["partials"]]
        times = sorted(t for t in final_times + partial_times if t is not None)
        # First partial per utterance, not every update's speech-start latency.
        first = {}
        for row in group["partials"]:
            utt, at = row.get("utterance_id"), _number(row.get("emitted_at_ms"))
            if utt is not None and at is not None and (utt not in first or at < first[utt][0]):
                first[utt] = (at, _number(row.get("speech_start_to_partial_ms")))
        source_matches = key[5] != "unknown" and key[5] == source.get("input_audio_sha256")
        cohorts.append(
            {
                "identity": dict(
                    zip(
                        (
                            "timing_schema_version",
                            "timing_source",
                            "sample_rate",
                            "source_lang",
                            "target_lang",
                            "input_audio_sha256",
                        ),
                        key,
                        strict=True,
                    )
                ),
                "final_count": len(group["finals"]),
                "final_metric": "speech_end_to_final_ms (estimated last VAD-positive frame end to server payload readiness)",
                "partial_count": len(group["partials"]),
                "endpoint_counts": dict(Counter((row.get("endpoint_reason") or "unknown") for row in group["finals"])),
                "final_metrics_by_endpoint": {
                    reason: _stats(
                        [
                            _number(r.get("speech_end_to_final_ms"))
                            for r in group["finals"]
                            if (r.get("endpoint_reason") or "unknown") == reason
                        ]
                    )
                    for reason in sorted({(r.get("endpoint_reason") or "unknown") for r in group["finals"]})
                },
                "first_partial_speech_start_ms": _stats([value[1] for value in first.values()]),
                "server_update_gap_ms": _stats([b - a for a, b in pairwise(times)]),
                "server_update_times_ms": times,
                "source_matches_session_metadata": source_matches,
                "final_source_coverage": source_coverage(
                    group["finals"], duration if source_matches else None, annotations if source_matches else ()
                ),
                "any_caption_source_coverage": source_coverage(
                    group["finals"] + group["partials"],
                    duration if source_matches else None,
                    annotations if source_matches else (),
                ),
            }
        )
    csv_path = _path(root, session, "ab_metrics", "csv")
    csv_ids = []
    try:
        csv_ids = [row.get("chunk_id") for row in csv.DictReader(io.StringIO(_snapshot(csv_path, issues, evidence)))]
    except csv.Error as exc:
        issues["read_errors"].append({"file": csv_path.name, "error": type(exc).__name__})
    consistency = (
        not any(issues.values())
        and not duplicate_ids
        and set(csv_ids) == {str(row["chunk_id"]) for row in finals}
        and len(csv_ids) == len(finals)
    )
    completed = completed and consistency
    summary_rows = [row for row in diagnostics if row.get("event") == "session_summary"]
    summary = summary_rows[-1] if summary_rows else {}
    return {
        "completion": {
            "verified": completed,
            "lifecycle_status": marker.get("status"),
            "exit_code": marker.get("exit_code"),
            "diagnostics_hash_matches": diagnostics_match,
            "artifact_consistency_verified": consistency,
            "persistence_verified": persistence_ok,
            "persistence": persistence,
            "ended_at": marker.get("ended_at"),
        },
        "lifecycle_provenance": {
            key: marker.get(key)
            for key in ("run_id", "git_sha", "pipeline_sha256", "source", "models", "managed_llama")
        },
        "pipeline_lifetime_memory": marker.get("memory"),
        "source": source,
        "artifact_scope": "full session files as read at monitor exit; may include time before attachment",
        "artifact_evidence": evidence,
        "read_issues": issues,
        "final_count": len(finals),
        "partial_count": len(partials),
        "duplicate_final_ids": duplicate_ids,
        "csv_final_ids_match_diagnostics": set(csv_ids) == {str(row["chunk_id"]) for row in finals}
        and len(csv_ids) == len(finals),
        "cohorts": cohorts,
        "latest_session_summary": {
            key: summary[key]
            for key in (
                "chunks_attempted",
                "chunks_completed",
                "chunks_empty_stt",
                "chunks_hallucination",
                "chunks_dedup",
                "ws_total_connections",
                "ws_total_disconnections",
                "ws_send_failures",
                "latency_experiment_counters",
            )
            if key in summary
        },
        "notes": [
            "Server update gaps include natural silence/music and loading; they are not all stalls.",
            "No browser render ACK latency or speech/music ground truth is inferred.",
            "Capture total/sample-loss coverage is unknown unless recorded separately in health.",
            "Pipeline lifetime memory does not include native children; see sampled process-tree RSS separately.",
        ],
    }


def summarize_samples(samples):
    queues = defaultdict(list)
    for sample in samples:
        for key, value in sample["health"].get("queues", {}).items():
            queues[key].append(_number(value))
    return {
        "sample_count": len(samples),
        "observed_seconds": samples[-1]["elapsed_s"] - samples[0]["elapsed_s"] if len(samples) > 1 else 0,
        "rss_sum_bytes": _stats([s["tree"]["rss_sum_bytes"] for s in samples]),
        "cpu_percent_sum": _stats([s["tree"]["cpu_percent_sum"] for s in samples]),
        "process_count": _stats([s["tree"]["process_count"] for s in samples]),
        "sampling_gaps_s": _stats([b["monotonic_s"] - a["monotonic_s"] for a, b in pairwise(samples)]),
        "incomplete_tree_samples": sum(not s["tree"]["counters_complete"] for s in samples),
        "memory_sampling_complete": bool(samples) and all(s["tree"]["counters_complete"] for s in samples),
        "process_cleanup_observed": bool(samples)
        and samples[-1]["tree"]["counters_complete"]
        and samples[-1]["tree"]["process_count"] == 0,
        "last_observed_processes": samples[-1]["tree"]["processes"] if samples else [],
        "health_missing_samples": sum(not s["health"]["available"] for s in samples),
        "health_stale_samples": sum(s["health"]["stale"] for s in samples),
        "health_phases": dict(Counter(s["health"].get("phase", "unknown") for s in samples)),
        "queue_depth_samples": {key: _stats(values) for key, values in queues.items()},
        "input_age_s": _stats([_number(s["health"].get("input_age_s")) for s in samples]),
        "caption_age_s": _stats([_number(s["health"].get("caption_age_s")) for s in samples]),
        "events": [{"elapsed_s": s["elapsed_s"], **event} for s in samples for event in s["tree"]["events"]],
        "errors": [{"elapsed_s": s["elapsed_s"], **event} for s in samples for event in s["tree"]["errors"]],
        "last_health": samples[-1]["health"] if samples else None,
        "memory_basis": "maximum observed simultaneous sum of pipeline and verified descendant RSS; sampled, not lifetime; shared pages may be counted multiple times",
        "cpu_basis": "per-process user+system delta / monotonic sample interval; 100% equals one core; first observation is null",
        "unobserved_children": "Children born and exited between samples are not measured; disappearance has unknown exit code.",
    }


def _publish(path, report):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def monitor(
    root,
    session,
    pid,
    output,
    *,
    duration=3900.0,
    interval=5.0,
    exit_grace=10.0,
    stop_file=None,
    stop_event=None,
    annotations=(),
    ps=None,
):
    if not all(math.isfinite(v) for v in (duration, interval, exit_grace)) or not (
        0.1 <= interval <= 60 and 0 < duration <= 86400 and 0 <= exit_grace <= 60 and duration / interval <= 20000
    ):
        raise ValueError("Require interval 0.1-60s, duration (0, 86400]s, and exit grace 0-60s; at most 20000 samples")
    if ps is None:
        import psutil as ps
    root, output = Path(root).resolve(), Path(output).resolve()
    if any(output.is_relative_to(root / name) for name in ("metrics", "stark_data", "adapters", "models")):
        raise ValueError("Monitor output must be separate from session data/models (use a new reports/cache directory)")
    identity = attach(root, session, pid, ps)
    output.mkdir(parents=True, exist_ok=False)
    stop_event = stop_event or threading.Event()
    sampler, samples = ProcessTreeSampler(identity, ps), []
    started, reason, terminal_since = time.monotonic(), "deadline", None
    report = {
        "schema_version": 1,
        "session_id": session,
        "root": str(root),
        "ownership": identity,
        "monitor_status": "running",
        "started_at": datetime.now(UTC).isoformat(),
        "requested_duration_s": duration,
        "interval_s": interval,
        "exit_grace_s": exit_grace,
        "monitor_only": True,
        "annotations": list(annotations),
    }
    _publish(output / "report.json", report)
    try:
        with (output / "samples.jsonl").open("x", encoding="utf-8") as stream:
            while True:
                now, wall = time.monotonic(), time.time()
                marker = _marker(root, session)
                verify_marker(marker, identity)
                tree = sampler.sample(now)
                sample = {
                    "monotonic_s": now,
                    "wall_time_unix": wall,
                    "wall_time_utc": datetime.fromtimestamp(wall, UTC).isoformat(),
                    "elapsed_s": now - started,
                    "tree": tree,
                    "health": health_snapshot(root, session, wall),
                    "lifecycle_status": marker.get("status"),
                }
                samples.append(sample)
                stream.write(json.dumps(sample, allow_nan=False) + "\n")
                stream.flush()
                terminal = marker.get("status") != "running" or tree["pipeline_alive"] is False
                if terminal and terminal_since is None:
                    terminal_since = now
                if terminal_since is not None and (
                    (tree["process_count"] == 0 and tree["counters_complete"]) or now - terminal_since >= exit_grace
                ):
                    reason = "session_terminal_or_pipeline_disappeared"
                    break
                if stop_event.is_set() or (stop_file is not None and Path(stop_file).exists()):
                    reason = "monitor_stop_requested"
                    break
                remaining = duration - (time.monotonic() - started)
                if remaining <= 0:
                    break
                stop_event.wait(min(interval, remaining))
    except Exception as exc:
        report["monitor_error"] = {"type": type(exc).__name__, "message": str(exc)}
        reason = "monitor_error"
    report.update(stop_reason=reason, ended_at=datetime.now(UTC).isoformat(), measurements=summarize_samples(samples))
    try:
        report["artifacts"] = artifact_summary(root, session, identity, annotations)
    except Exception as exc:
        report["artifact_error"] = {"type": type(exc).__name__, "message": str(exc)}
    completed = (
        report.get("artifacts", {}).get("completion", {}).get("verified", False)
        and report["measurements"]["process_cleanup_observed"]
    )
    report["monitor_status"] = (
        "error" if "monitor_error" in report or "artifact_error" in report else "completed" if completed else "partial"
    )
    _publish(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--session", required=True)
    parser.add_argument("--pid", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--duration-seconds", type=float, default=3900)
    parser.add_argument("--interval", type=float, default=5)
    parser.add_argument("--exit-grace-seconds", type=float, default=10)
    parser.add_argument("--stop-file", type=Path)
    parser.add_argument(
        "--annotations", type=Path, help="External source interval labels; no automatic silence/music classification"
    )
    args = parser.parse_args()
    annotations = []
    if args.annotations:
        annotations = json.loads(args.annotations.read_text())
        if not isinstance(annotations, list) or any(
            not isinstance(a, dict)
            or _number(a.get("start_s")) is None
            or _number(a.get("end_s")) is None
            or a["start_s"] < 0
            or a["end_s"] <= a["start_s"]
            or not isinstance(a.get("label"), str)
            for a in annotations
        ):
            parser.error("Annotations must be a list of {start_s, end_s, label} intervals")
    stop = threading.Event()
    old = {sig: signal.signal(sig, lambda *_: stop.set()) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        result = monitor(
            args.root,
            args.session,
            args.pid,
            args.output,
            duration=args.duration_seconds,
            interval=args.interval,
            exit_grace=args.exit_grace_seconds,
            stop_file=args.stop_file,
            stop_event=stop,
            annotations=annotations,
        )
        print(json.dumps({"report": str(args.output / "report.json"), "status": result["monitor_status"]}))
        return {"completed": 0, "partial": 2, "error": 1}[result["monitor_status"]]
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Monitor could not attach/write: {exc}\n")
    finally:
        for sig, previous in old.items():
            signal.signal(sig, previous)


if __name__ == "__main__":
    raise SystemExit(main())
