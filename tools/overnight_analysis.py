"""Read-only supplemental analysis of frozen overnight benchmark evidence.

No inference imports. Selection means a candidate merits confirmation; it never
means production promotion or reference-quality approval. See the source contract
in docs/evaluation/overnight_analysis_contract.md.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from itertools import pairwise
from pathlib import Path
from typing import Any

FINAL_METRICS = (
    "speech_end_to_final_ms",
    "vad_wait_ms",
    "stt_queue_wait_ms",
    "translation_queue_wait_ms",
    "stt_latency_ms",
    "latency_a_ms",
    "prefill_ms_a",
    "ttft_ms_a",
    "decode_ms_a",
    "generation_lock_wait_ms_a",
    "prompt_cache_prepare_ms_a",
    "finalization_overhead_ms",
    "broadcast_ms",
)


def number(value, *, signed=False):
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
        return result if math.isfinite(result) and (signed or result >= 0) else None
    except (TypeError, ValueError):
        return None


def distribution(values, *, signed=False):
    values = sorted(n for value in values if (n := number(value, signed=signed)) is not None)
    return {
        "n": len(values),
        "p50": statistics.median(values) if values else None,
        "p95": values[math.ceil(len(values) * 0.95) - 1] if values else None,
        "max": values[-1] if values else None,
    }


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def interval(row):
    start, end, rate = (number(row.get(key)) for key in ("sample_start", "sample_end", "sample_rate"))
    return (start / rate, end / rate) if start is not None and end is not None and rate and end > start else None


def bounds(row):
    values = [number(row.get(key)) for key in ("sample_start", "sample_end", "sample_rate", "speech_end_sample")]
    if any(value is None for value in values) or not interval(row):
        return None
    if not values[0] <= values[3] <= values[1]:
        return None
    return (*values, endpoint_classification(row), row.get("timing_source"))


def unique(rows, key):
    groups = defaultdict(list)
    for row in rows:
        groups[key(row)].append(row)
    return {name: items[0] for name, items in groups.items() if name is not None and len(items) == 1}, sum(
        len(items) for name, items in groups.items() if name is None or len(items) != 1
    )


def source_ranges(rows):
    merged: list[list[float]] = []
    for start, end in sorted(span for row in rows if (span := interval(row)) is not None):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(end, merged[-1][1])
        else:
            merged.append([start, end])
    return merged


def missing_seconds(reference, observed):
    return sum(
        max(0, end - start - sum(max(0, min(end, b) - max(start, a)) for a, b in observed)) for start, end in reference
    )


def endpoint_classification(row):
    """Separate EOF-padding-assisted finals without changing their raw reason."""
    reason = row.get("endpoint_reason") or "unknown"
    if str(row.get("timing_source", "")).startswith("replay_") and (number(row.get("padding_samples")) or 0) > 0:
        return f"{reason}_replay_tail"
    return reason


def endpoint(row):
    return f"{endpoint_classification(row)}|{row.get('timing_source', 'unknown')}"


def recorded_endpoint(row):
    reason = endpoint_classification(row)
    return reason != "eof" and not reason.endswith("_replay_tail")


def eligible_gain_target(metric):
    if metric.startswith(("final:", "final_visible:")):
        reason = metric.split(":", 1)[1].split("|", 1)[0]
        return reason != "eof" and not reason.endswith("_replay_tail")
    return metric.startswith("first_visible_preview:")


def lexical_change(before, after):
    """Literal whitespace-token suffix replacement; never an accuracy score."""
    left, right = str(before or "").split(), str(after or "").split()
    prefix = 0
    for a, b in zip(left, right, strict=False):
        if a != b:
            break
        prefix += 1
    return {
        "previous_tokens": len(left),
        "next_tokens": len(right),
        "stable_prefix_tokens": prefix,
        "retracted_suffix_tokens": len(left) - prefix,
        "changed": left != right,
    }


def preview_analysis(partials, finals, diagnostics):
    groups = defaultdict(list)
    source_lang = next((row.get("source_lang") for row in finals if row.get("source_lang")), "en")
    target_lang = "es" if source_lang == "en" else "en"
    emitted, ambiguous = unique(partials, lambda row: row.get("event_id"))
    emitted = {
        eid: row
        for eid, row in emitted.items()
        if str(row.get(f"text_{target_lang}") or "").strip() and row.get("utterance_id") is not None
    }
    for row in emitted.values():
        if str(row.get(f"text_{target_lang}") or "").strip() and row.get("utterance_id") is not None:
            groups[str(row["utterance_id"])].append(row)
    final_by_uid, _ = unique(
        finals, lambda row: str(row.get("utterance_id")) if row.get("utterance_id") is not None else None
    )
    diag_by_chunk, _ = unique(
        [row for row in diagnostics if row.get("chunk_id") is not None], lambda row: str(row["chunk_id"])
    )
    per_utterance, first, gaps, last_gaps = {}, {}, [], []
    for uid, rows in groups.items():
        rows.sort(
            key=lambda row: (
                number(row.get("emitted_at_ms")) if number(row.get("emitted_at_ms")) is not None else math.inf
            )
        )
        times = [t for row in rows if (t := number(row.get("emitted_at_ms"))) is not None]
        utterance_gaps = [b - a for a, b in pairwise(times)]
        gaps.extend(utterance_gaps)
        first_delay = number(rows[0].get("speech_start_to_partial_ms"))
        if first_delay is not None and uid in final_by_uid:
            first[uid] = first_delay
        final = final_by_uid.get(uid, {})
        ready = number(diag_by_chunk.get(str(final.get("chunk_id")), {}).get("timing_stages_ms", {}).get("final_ready"))
        last_gap = ready - times[-1] if times and ready is not None and ready >= times[-1] else None
        if last_gap is not None:
            last_gaps.append(last_gap)
        revisions = []
        for before, after in pairwise(rows):
            revisions.append(
                {
                    "before_event": before["event_id"],
                    "after_event": after["event_id"],
                    "source": lexical_change(before.get(f"text_{source_lang}"), after.get(f"text_{source_lang}")),
                    "translation": lexical_change(before.get(f"text_{target_lang}"), after.get(f"text_{target_lang}")),
                }
            )
        per_utterance[uid] = {
            "events": [row["event_id"] for row in rows],
            "first_text": {
                "source": rows[0].get(f"text_{source_lang}"),
                "translation": rows[0].get(f"text_{target_lang}"),
            },
            "last_text": {
                "source": rows[-1].get(f"text_{source_lang}"),
                "translation": rows[-1].get(f"text_{target_lang}"),
            },
            "first_preview_ms": first_delay,
            "update_gap_ms": distribution(utterance_gaps),
            "last_preview_to_final_ready_ms": last_gap,
            "preview_after_final_ready": bool(times and ready is not None and times[-1] > ready),
            "lexical_revisions": revisions,
            "last_preview_to_final_rewrite": {
                "source": lexical_change(rows[-1].get(f"text_{source_lang}"), final.get("english")),
                "translation": lexical_change(rows[-1].get(f"text_{target_lang}"), final.get("spanish_a")),
            }
            if final
            else None,
        }
    return {
        "first_by_utterance_ms": first,
        "first_preview_ms": distribution(first.values()),
        "missing_final_utterances": sorted(set(final_by_uid) - first.keys()),
        "unfinalized_preview_utterances": sorted(set(groups) - final_by_uid.keys()),
        "update_gap_ms": distribution(gaps),
        "last_preview_to_final_ready_ms": distribution(last_gaps),
        "utterances": per_utterance,
        "ambiguous_event_rows": ambiguous,
        "emitted_preview_processed_audio_seconds_proxy": sum(
            number(row.get("processed_audio_s")) or 0 for row in emitted.values()
        ),
        "total_decoded_audio_seconds": None,
        "work_note": "Proxy covers emitted previews only; discarded/suppressed work, ordinary final STT and speculative STT are not all sample-stamped in the trace.",
    }, emitted


def browser_analysis(acks, emitted, finals, diagnostics, session):
    final_by_chunk, _ = unique(
        finals, lambda row: str(row.get("chunk_id")) if row.get("chunk_id") is not None else None
    )
    diag_by_chunk, _ = unique(
        [row for row in diagnostics if row.get("chunk_id") is not None], lambda row: str(row["chunk_id"])
    )
    expected_uids = {str(row["utterance_id"]) for row in finals if row.get("utterance_id") not in (None, "")}
    final_by_uid, _ = unique(
        finals, lambda row: str(row["utterance_id"]) if row.get("utterance_id") not in (None, "") else None
    )
    clients: dict[str, dict[str, Any]] = defaultdict(lambda: {"finals": {}, "previews": {}, "unmatched": 0})
    seen = set()
    for ack in acks:
        if (
            ack.get("session_id") != session
            or ack.get("event") != "caption_rendered"
            or ack.get("visible") is not True
            or ack.get("timing_schema_version") != 2
        ):
            continue
        client, eid = ack.get("client_id"), ack.get("event_id")
        if not isinstance(client, str) or not client or not isinstance(eid, str) or not eid:
            continue
        if (client, eid) in seen:
            continue
        seen.add((client, eid))
        data = clients[client]
        if ack.get("stage") == "partial":
            row = emitted.get(eid)
            if row is None or any(
                ack.get(key) != row.get(key)
                for key in ("preview_revision", "utterance_id", "timing_source", "caption_delivery_mode")
            ):
                data["unmatched"] += 1
            else:
                data["previews"][eid] = ack
        elif ack.get("stage") == "complete":
            cid = str(ack.get("chunk_id"))
            row, diag = final_by_chunk.get(cid), diag_by_chunk.get(cid)
            if (
                row is None
                or diag is None
                or eid != diag.get("event_id")
                or any(
                    str(ack.get(key)) != str(row.get(key))
                    for key in ("sample_start", "sample_end", "sample_rate", "speech_end_sample", "timing_source")
                )
            ):
                data["unmatched"] += 1
            else:
                data["finals"].setdefault(cid, ack)
    result = {}
    for client, data in clients.items():
        first: dict[str, float] = {}
        times: dict[str, list[float]] = defaultdict(list)
        end_samples: dict[str, list[float]] = defaultdict(list)
        for eid, ack in data["previews"].items():
            uid = str(ack.get("utterance_id"))
            delay, start = (
                number(ack.get("speech_start_to_preview_ack_upper_bound_ms")),
                number(ack.get("captured_start_at_ms")),
            )
            if delay is not None and uid in expected_uids and ack.get("timing_source") == "replay_realtime":
                first[uid] = min(first.get(uid, math.inf), delay)
                if start is not None:
                    times[uid].append(start + delay)
        gaps = [b - a for values in times.values() for a, b in zip(sorted(values), sorted(values)[1:], strict=False)]
        final_times = {}
        for cid, ack in data["finals"].items():
            delay = number(ack.get("speech_end_to_ack_upper_bound_ms"))
            if delay is not None and ack.get("timing_source") == "replay_realtime":
                end_samples[endpoint(final_by_chunk[cid])].append(delay)
                final_times[cid] = delay
        translated = {
            eid
            for eid, row in emitted.items()
            if str(row.get("text_en" if row.get("target_lang") == "en" else "text_es") or "").strip()
        }
        # Readiness is earlier than sending/rendering a final. Distinguish it
        # from a preview ACK observed after that client's matching final ACK.
        # Reconstruct both receipt times on the server clock; browser absolute
        # timestamps are neither available nor required. These are observation
        # order, not a claim about the pixels remaining on screen.
        post_final, unassessable = [], []
        for eid, ack in data["previews"].items():
            uid = str(ack.get("utterance_id"))
            final = final_by_uid.get(uid, {})
            cid = str(final.get("chunk_id"))
            final_ack = data["finals"].get(cid, {})
            start = number(ack.get("captured_start_at_ms"))
            partial_delay = number(ack.get("speech_start_to_preview_ack_upper_bound_ms"))
            end = number(diag_by_chunk.get(cid, {}).get("timing_stages_ms", {}).get("speech_end"))
            final_delay = number(final_ack.get("speech_end_to_ack_upper_bound_ms"))
            if (
                None in (start, partial_delay, end, final_delay)
                or ack.get("timing_source") != "replay_realtime"
                or final_ack.get("timing_source") != "replay_realtime"
            ):
                unassessable.append(eid)
                continue
            delta = start + partial_delay - (end + final_delay)
            if delta > 0.2:  # two rounded 0.1-ms ACK durations, plus 0.001-ms stages
                post_final.append(
                    {"event_id": eid, "utterance_id": uid, "chunk_id": cid, "ack_after_final_ms": round(delta, 3)}
                )
        result[client] = {
            "final_acks": len(data["finals"]),
            "expected_finals": len(final_by_chunk),
            "timed_final_acks": sum(map(len, end_samples.values())),
            "final_by_chunk_ms": final_times,
            "final_coverage": len(data["finals"]) / len(final_by_chunk) if final_by_chunk else None,
            "missing_final_chunks": sorted(final_by_chunk.keys() - data["finals"].keys()),
            "final_ack_by_endpoint_ms": {key: distribution(values) for key, values in end_samples.items()},
            "first_by_utterance_ms": first,
            "first_preview_ack_ms": distribution(first.values()),
            "first_preview_coverage": len(first) / len(expected_uids) if expected_uids else None,
            "missing_first_preview_utterances": sorted(expected_uids - first.keys()),
            "missing_preview_events": sorted(translated - data["previews"].keys()),
            "visible_preview_events": len(data["previews"]),
            "unmatched_acks": data["unmatched"],
            "preview_acks_after_final_ack": post_final,
            "preview_ack_order_unassessable_events": unassessable,
            "ack_observation_gap_ms": distribution(gaps),
            "receive_to_render_ms": distribution(
                ack.get("receive_to_render_ms") for ack in [*data["previews"].values(), *data["finals"].values()]
            ),
        }
    return result


def inspect_result(path, metrics):
    run = json.loads(path.read_text())
    session = run["session_id"]
    errors, files = [], []
    raw: dict[str, list[dict]] = {}
    if not run.get("completion_validation", {}).get("valid") or run.get("error"):
        errors.append(run.get("error") or "Harness did not validate completion")
    lifecycle = run.get("session_lifecycle", {})
    if lifecycle.get("status") != "completed" or lifecycle.get("exit_code") != 0 or run.get("returncode") != 0:
        errors.append("Subprocess/lifecycle is not successfully completed")
    for kind, prefix, suffix in (
        ("finals", "ab_metrics", "csv"),
        ("partials", "partials", "jsonl"),
        ("diagnostics", "diagnostics", "jsonl"),
        ("acks", "display_metrics", "jsonl"),
    ):
        artifact = metrics / f"{prefix}_{session}.{suffix}"
        if not artifact.exists():
            if kind != "acks":
                errors.append(f"Missing raw {kind}")
            raw[kind] = []
            continue
        try:
            with artifact.open(newline="") as stream:
                raw[kind] = list(csv.DictReader(stream)) if suffix == "csv" else read_jsonl(artifact)
            stamp = {"path": str(artifact.resolve()), "sha256": sha(artifact), "size_bytes": artifact.stat().st_size}
            files.append(stamp)
            if kind == "diagnostics" and {k: stamp[k] for k in ("sha256", "size_bytes")} != lifecycle.get(
                "diagnostics"
            ):
                errors.append("Diagnostic hash differs from completed lifecycle")
            if kind in ("finals", "partials") and raw[kind] != run.get("observed", {}).get(kind):
                errors.append(f"Raw {kind} differ from harness-observed records")
        except (OSError, ValueError, TypeError) as exc:
            errors.append(f"Unreadable {kind}: {exc}")
            raw[kind] = []
    summary = next((row for row in reversed(raw["diagnostics"]) if row.get("event") == "session_summary"), {})
    if summary != run.get("observed", {}).get("session_summary"):
        errors.append("Final summary differs from harness observation")
    finals, malformed = unique(raw["finals"], bounds)
    if malformed or not finals:
        errors.append("Final source bounds missing, duplicated or empty")
    if any(
        str(row.get("timing_schema_version")) != "2" or row.get("timing_source") != "replay_realtime"
        for row in raw["finals"]
    ):
        errors.append("Finals are not uniformly real-time schema2")
    if any(number(row.get("speech_end_to_final_ms")) is None for row in raw["finals"]):
        errors.append("Final speech-end latency is missing or nonfinite")
    _, invalid_uids = unique(
        raw["finals"], lambda row: str(row.get("utterance_id")) if row.get("utterance_id") not in (None, "") else None
    )
    if invalid_uids:
        errors.append("Final utterance identities are missing or duplicated")
    metadata = run.get("session_metadata", {})
    if metadata.get("input_audio_sha256") != run.get("clip", {}).get("sha256"):
        errors.append("Input audio identity mismatch")
    config = metadata.get("latency_experiment_configuration", {})
    if config != summary.get("latency_experiment_configuration") or config != run.get(
        "requested_experiment_configuration"
    ):
        errors.append("Resolved experiment settings differ")
    if not run.get("source_cohort_sha256") or not lifecycle.get("models"):
        errors.append("Missing source/model provenance")
    if metadata.get("session_id") != session or metadata.get("source_lang") != run.get("language"):
        errors.append("Session/language provenance mismatch")
    if any(
        row.get("session_id") != session
        or row.get("timing_schema_version") != 2
        or row.get("timing_source") != "replay_realtime"
        for row in raw["partials"]
    ):
        errors.append("Preview identity/clock metadata is incompatible")
    if any(row.get("utterance_id") in (None, "") for row in raw["finals"]):
        errors.append("Final utterance identity is missing")
    previews, emitted = preview_analysis(raw["partials"], raw["finals"], raw["diagnostics"])
    browser = browser_analysis(raw["acks"], emitted, raw["finals"], raw["diagnostics"], session)
    grouped = defaultdict(list)
    for row in raw["finals"]:
        grouped[endpoint(row)].append(row)
    trace = summary.get("latency_trace", {})
    trace_events = trace.get("events", [])
    traces = {}
    for event, fields in {
        "audio_dequeued": ("capture_age_ms", "queue_depth"),
        "vad_complete": ("elapsed_ms",),
        "stt_worker_started": ("wait_ms",),
        "final_stt_resumed": ("worker_finish_to_resume_ms",),
        "caption_send_started": ("queue_ms",),
        "speculation_completed": ("model_ms",),
    }.items():
        traces[event] = {
            field: distribution(row.get(field) for row in trace_events if row.get("event") == event) for field in fields
        }
    stages = defaultdict(list)
    for row in raw["diagnostics"]:
        stamps = row.get("timing_stages_ms", {})
        for name, start, end in (
            ("stt_worker_wall_ms", "stt_started", "stt_finished"),
            ("translation_stage_wall_ms", "translation_started", "translation_finished"),
            ("post_stt_dispatch_ms", "stt_finished", "translation_requested"),
        ):
            a, b = number(stamps.get(start)), number(stamps.get(end))
            if a is not None and b is not None and b >= a:
                stages[name].append(b - a)
    cohort = identity(
        {
            "source": run.get("source_cohort_sha256"),
            "audio": metadata.get("input_audio_sha256"),
            "models": lifecycle.get("models"),
            "stt_backend": metadata.get("stt_backend"),
            "language": run.get("language"),
            "profile": metadata.get("profile", {}).get("name"),
        }
    )
    result = {
        "session_id": session,
        "experiment": run["experiment"],
        "size": run["size"],
        "repeat": run["repeat"],
        "clip_id": run["clip_id"],
        "language": run.get("language"),
        "cohort": cohort,
        "valid": not errors,
        "validation_errors": errors,
        "final_count": len(raw["finals"]),
        "source_ranges_s": source_ranges(raw["finals"]),
        "source_cohort_sha256": run.get("source_cohort_sha256"),
        "configuration": config,
        "replay_client_wait": metadata.get("replay_client_wait"),
        "endpoints": {
            name: {key: distribution(row.get(key) for row in rows) for key in FINAL_METRICS}
            for name, rows in grouped.items()
        },
        "endpoint_classifications": [
            {
                "chunk_id": row.get("chunk_id"),
                "raw_endpoint_reason": row.get("endpoint_reason"),
                "analytical_endpoint": endpoint_classification(row),
                "timing_source": row.get("timing_source"),
                "padding_samples": number(row.get("padding_samples")),
                "eligible_final_gain_target": recorded_endpoint(row),
            }
            for row in raw["finals"]
        ],
        "previews": previews,
        "browsers": browser,
        "counters": summary.get("latency_experiment_counters", {}),
        "trace": {
            "enabled": trace.get("enabled"),
            "total_events": trace.get("total_events"),
            "discarded_old_events": trace.get("discarded_old_events"),
            "event_counts_in_retained_trace": dict(Counter(row.get("event") for row in trace_events)),
            "metrics": traces,
        },
        "stage_wall_times_ms": {name: distribution(values) for name, values in stages.items()},
        "memory": lifecycle.get("memory"),
        "persistence": lifecycle.get("persistence"),
        "resources_at_finalization": run.get("observed", {}).get("resource_snapshots", []),
        "raw_files": files,
        "result_file": {"path": str(path.resolve()), "sha256": sha(path), "size_bytes": path.stat().st_size},
        "models": lifecycle.get("models"),
    }
    return result, raw["finals"]


def bind_browser(row, *, client_id=None, client_map=None):
    """Bind one connection per run under the declared controlled-display protocol.

    Runtime IDs are process-local ``id(websocket)`` values, not browser IDs.
    Never intersect them across subprocesses or pool two clients in one run.
    """
    clients = {
        key: data for key, data in row["browsers"].items() if data["final_acks"] or data["visible_preview_events"]
    }
    if client_map is not None:
        chosen, mode = client_map.get(row["session_id"]), "explicit_session_map"
    elif client_id is not None:
        chosen, mode = client_id, "explicit_literal_client_id"
    else:
        chosen, mode = next(iter(clients)) if len(clients) == 1 else None, "sole_visible_client_per_session"
    return {
        "client_id": chosen if chosen in clients else None,
        "mode": mode,
        "status": "bound" if chosen in clients else "unassessable",
        "reason": None
        if chosen in clients
        else "No unambiguous visible connection selected; supply a session-to-client map for multiple clients",
        "stable_physical_identity_verified": False,
    }


def paired(before, after):
    if not before or not after:
        return {"status": "unassessable", "reason": "Missing eligible compatible comparison", "metrics": {}}
    left, right = before[0], after[0]
    if left["cohort"] != right["cohort"]:
        return {"status": "unassessable", "reason": "Different source/model/audio cohorts", "metrics": {}}
    a, ambiguous_a = unique(before[1], bounds)
    b, ambiguous_b = unique(after[1], bounds)
    keys = sorted(a.keys() & b.keys())
    metrics: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"before": [], "after": [], "delta_ms": []})
    changes = []
    left_binding, right_binding = left.get("browser_binding", {}), right.get("browser_binding", {})
    paired_browser = left_binding.get("status") == right_binding.get("status") == "bound"

    def add(name, x, y):
        x, y = number(x), number(y)
        if x is not None and y is not None:
            metrics[name]["before"].append(x)
            metrics[name]["after"].append(y)
            metrics[name]["delta_ms"].append(y - x)

    for key in keys:
        x, y = a[key], b[key]
        uid_a, uid_b = str(x["utterance_id"]), str(y["utterance_id"])
        add("final:" + endpoint(x), x.get("speech_end_to_final_ms"), y.get("speech_end_to_final_ms"))
        add(
            "first_server_preview",
            left["previews"]["first_by_utterance_ms"].get(uid_a),
            right["previews"]["first_by_utterance_ms"].get(uid_b),
        )
        if paired_browser:
            client_a, client_b = left_binding["client_id"], right_binding["client_id"]
            add(
                "final_visible:" + endpoint(x) + ":declared_display",
                left["browsers"][client_a]["final_by_chunk_ms"].get(str(x["chunk_id"])),
                right["browsers"][client_b]["final_by_chunk_ms"].get(str(y["chunk_id"])),
            )
            add(
                "first_visible_preview:declared_display",
                left["browsers"][client_a]["first_by_utterance_ms"].get(uid_a),
                right["browsers"][client_b]["first_by_utterance_ms"].get(uid_b),
            )
        if (x.get("english"), x.get("spanish_a")) != (y.get("english"), y.get("spanish_a")):
            changes.append(
                {
                    "sample_bounds": key,
                    "before": {k: x.get(k) for k in ("chunk_id", "english", "spanish_a")},
                    "after": {k: y.get(k) for k in ("chunk_id", "english", "spanish_a")},
                }
            )
    return {
        "status": "assessable",
        "before_session": left["session_id"],
        "after_session": right["session_id"],
        "browser_bindings": {
            "before": {"session_id": left["session_id"], **left_binding},
            "after": {"session_id": right["session_id"], **right_binding},
        },
        "matched_final_bounds": len(keys),
        "before_unmatched": len(a.keys() - b.keys()),
        "after_unmatched": len(b.keys() - a.keys()),
        "ambiguous_rows": ambiguous_a + ambiguous_b,
        "missing_source_s": missing_seconds(left["source_ranges_s"], right["source_ranges_s"]),
        "added_source_s": missing_seconds(right["source_ranges_s"], left["source_ranges_s"]),
        "changed_matched_outputs": changes,
        "metrics": {
            name: {
                "before": distribution(value["before"]),
                "after": distribution(value["after"]),
                "paired_delta_ms": distribution(value["delta_ms"], signed=True),
                "samples": value,
            }
            for name, value in metrics.items()
        },
    }


def experiment_execution(row, finals):
    config, counters, missing = row["configuration"], row["counters"], []
    checks = {
        "latest_partial": counters.get("stt_worker_started", 0) > 0,
        "async_captions": counters.get("caption_sent", 0) > 0,
        "marian_memo": counters.get("marian_memo_hit", 0) > 0,
        "pause_preview_ms": counters.get("preview_pause", 0) > 0,
        "speculate_pause_ms": counters.get("speculation_reused", 0) > 0,
        "clause_preview_s": counters.get("preview_clause", 0) > 0,
        "gemma_prefix_cache": any(str(item.get("prompt_cache_hit_a")).lower() == "true" for item in finals),
    }
    for flag, exercised in checks.items():
        if config.get(flag) and not exercised:
            missing.append(flag)
    # First-preview and allocator flags have resolved settings but no independent
    # low-level activation stamp. VAD records execution without worker identity.
    notes = []
    if config.get("first_preview_s"):
        notes.append(
            "Early-first setting recorded; inspect first emitted source bounds/text because admission can be suppressed"
        )
    if config.get("vad_worker"):
        notes.append(
            "VAD worker setting recorded; retained vad_complete timings include dispatch, without a thread identity proof"
        )
    if config.get("mlx_cache_mb", 256) != 256:
        notes.append(
            "Allocator limit is a resolved setting; peak memory/latency are observed, cache occupancy is not traced"
        )
    if (
        config.get("incremental_stt") == "stream"
        and row["trace"]["event_counts_in_retained_trace"].get("stream_preview_stt", 0) == 0
    ):
        missing.append("incremental_stt_stream")
    if (
        config.get("incremental_stt") == "rolling"
        and counters.get("rolling_preview_joined", 0) + counters.get("rolling_preview_unaligned", 0) == 0
    ):
        missing.append("incremental_stt_rolling")
    return {"session_id": row["session_id"], "unexercised_flags": missing, "notes": notes}


def gain(before, after):
    if before is None or after is None:
        return False
    return before - after >= 150 or (before > 0 and (before - after) / before >= 0.15)


def p95_guard(before, after):
    if before is None or after is None:
        return False
    return after - before <= max(100, before * 0.05)


def expected_sessions(provenance):
    spec, repeats, tag = provenance["spec"], provenance["repeats"], provenance["tag"]
    configs = [*spec["experiments"], {"name": "baseline_anchor"}]
    return {
        f"{tag}_{config['name']}_{size}_r{repeat}_{clip['id']}_{clip['lang']}"
        for repeat in range(repeats)
        for config in configs
        for size in spec.get("sizes", ["e4b", "e2b"])
        for clip in spec["clips"]
    }


def analyze(input_dir, metrics_dir, *, client_id=None, client_map=None):
    provenance_path = input_dir / "provenance.json"
    provenance = json.loads(provenance_path.read_text())
    expected = expected_sessions(provenance)
    if client_map is not None:
        if client_id is not None:
            raise ValueError("Choose a session-to-client map or a literal client ID, not both")
        if not isinstance(client_map, dict) or any(
            not isinstance(key, str) or not isinstance(value, str) or not value for key, value in client_map.items()
        ):
            raise ValueError("Client map must be a JSON object of session IDs to nonempty connection ID strings")
        if client_map.keys() - expected:
            raise ValueError("Client map contains sessions outside the declared matrix")
    records, failures, seen = [], [], set()
    for path in sorted(input_dir.glob("*.json")):
        if path.name in {"provenance.json", "comparison.json", "analysis.json"}:
            continue
        try:
            candidate = json.loads(path.read_text())
            if "observed" not in candidate:
                continue
            if candidate.get("session_id") not in expected:
                failures.append({"path": str(path), "reason": "Session is outside declared matrix"})
                continue
            if candidate["session_id"] in seen:
                failures.append({"path": str(path), "reason": "Duplicate session result"})
                continue
            seen.add(candidate["session_id"])
            records.append(inspect_result(path, metrics_dir))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            failures.append({"path": str(path), "reason": str(exc)})
    missing = sorted(expected - seen)
    complete = not missing and not failures and len(records) == len(expected)
    indexed = defaultdict(list)
    for record in records:
        row = record[0]
        row["browser_binding"] = bind_browser(row, client_id=client_id, client_map=client_map)
        indexed[(row["experiment"], row["size"], row["clip_id"], row["repeat"], row["cohort"])].append(record)

    def lookup(name, row):
        values = indexed[(name, row["size"], row["clip_id"], row["repeat"], row["cohort"])]
        return values[0] if len(values) == 1 and values[0][0]["valid"] else None

    groups = defaultdict(list)
    for record in records:
        row = record[0]
        if row["experiment"] in {"baseline", "baseline_anchor"}:
            continue
        groups[(row["experiment"], row["size"], row["clip_id"], row["cohort"])].append(record)
    arms = []
    for (name, size, clip, cohort), observations in sorted(groups.items()):
        reasons, comparisons = [], []
        passing: dict[str, int] = defaultdict(int)
        for record in sorted(observations, key=lambda value: value[0]["repeat"]):
            row = record[0]
            opening, closing = lookup("baseline", row), lookup("baseline_anchor", row)
            before, after = (
                paired(opening, record if row["valid"] else None),
                paired(closing, record if row["valid"] else None),
            )
            drift = paired(opening, closing)
            comparisons.append(
                {
                    "repeat": row["repeat"],
                    "session_id": row["session_id"],
                    "versus_opening": before,
                    "versus_closing": after,
                    "opening_to_closing_drift": drift,
                }
            )
            if not row["valid"]:
                reasons.append(f"{row['session_id']}: failed/invalid evidence")
                continue
            if any(pair["status"] != "assessable" for pair in (before, after)):
                reasons.append(f"repeat{row['repeat']}: missing eligible compatible opening/closing baseline")
                continue
            if any(
                pair["before_unmatched"]
                or pair["after_unmatched"]
                or pair["ambiguous_rows"]
                or pair["missing_source_s"] > 1e-6
                for pair in (before, after)
            ):
                reasons.append(f"repeat{row['repeat']}: source segmentation or coverage differs")
                continue
            triple = [opening[0], row, closing[0]]
            if any(item["browser_binding"]["status"] != "bound" for item in triple):
                reasons.append(f"repeat{row['repeat']}: no unambiguous per-session visible browser binding")
                continue
            chosen = row["browser_binding"]["client_id"]
            if any(
                (item["browsers"][item["browser_binding"]["client_id"]]["final_coverage"] or 0) < 0.95
                or (item["browsers"][item["browser_binding"]["client_id"]]["first_preview_coverage"] or 0) < 0.95
                or item["browsers"][item["browser_binding"]["client_id"]]["timed_final_acks"]
                < 0.95 * item["final_count"]
                for item in triple
            ):
                reasons.append(f"repeat{row['repeat']}: final or first-preview timed ACK coverage below95%")
                continue
            if row["previews"]["missing_final_utterances"]:
                reasons.append(f"repeat{row['repeat']}: missing server first preview")
                continue
            if row["browsers"][chosen]["preview_acks_after_final_ack"]:
                reasons.append(f"repeat{row['repeat']}: preview ACK after final ACK; browser-order review required")
                continue
            if any(item["preview_after_final_ready"] for item in row["previews"]["utterances"].values()):
                reasons.append(
                    f"repeat{row['repeat']}: preview after server final readiness; browser-order review required"
                )
                continue
            guard_failures = []
            common_metrics = before["metrics"].keys() & after["metrics"].keys()
            for metric in common_metrics:
                if any(
                    not p95_guard(pair["metrics"][metric]["before"]["p95"], pair["metrics"][metric]["after"]["p95"])
                    for pair in (before, after)
                ):
                    guard_failures.append(metric)
            for control in (opening[0], closing[0]):
                for metric in ("update_gap_ms", "last_preview_to_final_ready_ms"):
                    old, new = control["previews"][metric], row["previews"][metric]
                    if old["n"] and (not new["n"] or not p95_guard(old["p95"], new["p95"])):
                        guard_failures.append(metric)
                base_memory, new_memory = control.get("memory") or {}, row.get("memory") or {}
                for metric in ("peak_rss_bytes", "peak_metal_bytes"):
                    old, new = number(base_memory.get(metric)), number(new_memory.get(metric))
                    if old is None or new is None or new > old + 1024**3:
                        guard_failures.append(metric)
            if guard_failures:
                reasons.append(
                    f"repeat{row['repeat']}: tail/memory guard fails: {', '.join(sorted(set(guard_failures)))}"
                )
                continue
            for metric in common_metrics:
                if not eligible_gain_target(metric):
                    continue
                if all(
                    gain(pair["metrics"][metric]["before"]["p50"], pair["metrics"][metric]["after"]["p50"])
                    for pair in (before, after)
                ):
                    passing[metric] += 1
        repeats = {record[0]["repeat"] for record in observations}
        if len(repeats) != provenance["repeats"]:
            reasons.append("Not all declared repeats are recorded")
        if provenance["repeats"] < 3:
            reasons.append("Fewer than three paired repetitions")
        if not complete:
            reasons.append("Whole matrix is incomplete or has unreadable/duplicate/unexpected records")
        winning = sorted(
            name for name, count in passing.items() if count >= max(2, math.ceil(provenance["repeats"] * 2 / 3))
        )
        if not winning:
            reasons.append("No eligible targeted median gain versus both controls in at least two thirds of repeats")
        pooled = {}
        for control in ("versus_opening", "versus_closing", "opening_to_closing_drift"):
            merged: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"before": [], "after": [], "delta_ms": []})
            for comparison in comparisons:
                for metric, detail in comparison[control]["metrics"].items():
                    for key, values in detail["samples"].items():
                        merged[metric][key].extend(values)
            pooled[control] = {
                metric: {
                    "before": distribution(values["before"]),
                    "after": distribution(values["after"]),
                    "paired_delta_ms": distribution(values["delta_ms"], signed=True),
                }
                for metric, values in merged.items()
            }
        pooled_winners = [
            metric
            for metric in winning
            if all(
                gain(
                    pooled[control].get(metric, {}).get("before", {}).get("p50"),
                    pooled[control].get(metric, {}).get("after", {}).get("p50"),
                )
                for control in ("versus_opening", "versus_closing")
            )
        ]
        if winning and not pooled_winners:
            reasons.append("Pooled matched median does not meet the gain threshold against both controls")
        winning = pooled_winners
        execution = [experiment_execution(record[0], record[1]) for record in observations]
        if any(item["unexercised_flags"] for item in execution):
            reasons.append("Requested optimization has no observed execution evidence in one or more repeats")
        arms.append(
            {
                "experiment": name,
                "size": size,
                "clip_id": clip,
                "cohort": cohort,
                "status": "worth_confirming" if not reasons else "not_selected",
                "reasons": reasons,
                "passing_repeat_counts": dict(passing),
                "target_metrics": winning,
                "comparisons": comparisons,
                "pooled_matched_metrics": pooled,
                "execution_evidence": execution,
                "changed_output_review": "pending; lexical disagreement is not a quality score",
                "production_promotion": False,
                "p95_promotion_evidence": False,
            }
        )
    return {
        "schema_version": 1,
        "endpoint_classification_version": 1,
        "input": str(input_dir.resolve()),
        "metrics": str(metrics_dir.resolve()),
        "matrix_status": "complete" if complete else "incomplete",
        "expected_runs": len(expected),
        "recorded_runs": len(records),
        "valid_runs": sum(record[0]["valid"] for record in records),
        "missing_sessions": missing,
        "input_failures": failures,
        "declared_client_id": client_id,
        "declared_client_map": client_map,
        "provenance_sha256": sha(provenance_path),
        "sessions": [record[0] for record in records],
        "arms": arms,
        "production_promotion": False,
        "definitions": {
            "percentiles": "Raw observations: median and nearest-rank p95; paired deltas retain negative values. Small-sample p95 is often maximum, not confidence.",
            "matching": "Unique exact sample_start/end/rate, speech_end_sample, analytical endpoint and timing source within model/clip/repeat/source/model-artifact cohort.",
            "endpoint_classification": "Replay finals with positive padding_samples receive a _replay_tail suffix on their unchanged raw endpoint reason. For example, silence_replay_tail is VAD finalization assisted by synthetic EOF padding, not natural recorded silence. EOF and replay-tail final metrics remain separately reported and guarded but cannot supply a selection gain target. Real first-preview events remain eligible even when their utterance later ends at EOF. Older raw/harness summaries retain emitted reasons and may mix padded finals with silence; use this corrected supplemental classification for gates without rewriting those artifacts.",
            "selection": "Whole matrix completed; every repeat passes validity/coverage/tail/memory guards; same target median improves15% OR150ms versus both controls in at least2/3repeats. Followup selection only.",
            "browser": "Per client only. Timed first-preview coverage and final ACK coverage>=95%. Connection barrier alone never certifies visibility; missing intermediate ACKs can reflect coalescing.",
            "browser_binding": "The controlled audience-tab protocol binds exactly one visible connection per session, or uses an explicit session-to-client map. Runtime socket IDs are process-local and cannot prove stable physical browser identity. Paired declared_display metrics use only the selected connection in each run; original IDs and per-client distributions remain separate. Ambiguity is rejected.",
            "preview_order": "Server preview emission after final payload readiness requires review, not proof of stale repaint. Per-client preview_acks_after_final_ack compares reconstructed server receipt times for matched events; missing clocks/ACKs are unassessable. ACK opportunity order does not prove which pixels remained visible. The automatic selector retains its conservative readiness guard.",
            "lexical": "Literal whitespace-token prefix/suffix revisions are display-change proxies, not semantic quality or WER. Smart-cut previews can include carryover beyond final bounds.",
            "components": "Generation metrics can belong to earlier speculation; do not sum latency_a_ms/prefill/decode into post-end latency. Diagnostic stage wall times describe the final path.",
            "trace": "Trace distributions cover retained events; truncation is explicit. Counters use one final summary per session. Total decoded audio remains unknown.",
            "coverage": "Final sample interval union versus controls is not acoustic speech recall or reference completeness.",
        },
    }


def markdown(report):
    lines = [
        "# Supplemental overnight screen analysis",
        "",
        f"Matrix: **{report['matrix_status']}**; {report['recorded_runs']}/{report['expected_runs']} recorded, {report['valid_runs']} valid.",
        "",
        "Selections justify confirmation only. Defaults and human quality gates remain unchanged.",
        "",
        "| Experiment | Model | Clip | Decision | Target |",
        "|---|---|---|---|---|",
    ]
    for arm in report["arms"]:
        lines.append(
            f"| {arm['experiment']} | {arm['size']} | {arm['clip_id']} | {arm['status']} | {', '.join(arm['target_metrics']).replace(chr(124), ' / ') or 'None'} |"
        )
    lines.extend(["", "## Selection reasons", ""])
    for arm in report["arms"]:
        lines.append(
            f"- {arm['experiment']} / {arm['size']} / {arm['clip_id']}: "
            + (
                "; ".join(arm["reasons"])
                or "Meets conservative screening checks; inspect changed outputs before historical confirmation."
            )
        )
    lines.extend(
        [
            "",
            "## Matched comparisons and baseline drift",
            "",
            "| Experiment / model / clip | Metric | Control | Before p50 / p95 | After p50 / p95 | Paired delta p50 / p95 |",
            "|---|---|---|---|---|---|",
        ]
    )

    def quantiles(stat):
        return " / ".join("—" if stat.get(key) is None else str(round(stat[key], 1)) for key in ("p50", "p95"))

    for arm in report["arms"]:
        for control, metrics in arm["pooled_matched_metrics"].items():
            for metric, values in metrics.items():
                lines.append(
                    f"| {arm['experiment']} / {arm['size']} / {arm['clip_id']} | {metric.replace(chr(124), ' / ')} | {control} | {quantiles(values['before'])} | {quantiles(values['after'])} | {quantiles(values['paired_delta_ms'])} |"
                )
    lines.extend(
        [
            "",
            "## Per-session endpoints and preview coverage",
            "",
            "| Session | Endpoint | Final n / p50 / p95 ms | First server preview n / p50 / p95 ms | Visible browser coverage (final / first preview) |",
            "|---|---|---|---|---|",
        ]
    )

    def fmt(stat):
        return " / ".join("—" if stat.get(key) is None else str(round(stat[key], 1)) for key in ("n", "p50", "p95"))

    for row in report["sessions"]:
        coverage = (
            "; ".join(
                f"{client}: {data['final_acks']}/{data['expected_finals']} / {len(data['first_by_utterance_ms'])}/{row['final_count']}"
                for client, data in row["browsers"].items()
            )
            or "Missing"
        )
        for name, values in row["endpoints"].items():
            lines.append(
                f"| {row['session_id']} | {name.replace(chr(124), ' / ')} | {fmt(values['speech_end_to_final_ms'])} | {fmt(row['previews']['first_preview_ms'])} | {coverage} |"
            )
        if row["validation_errors"]:
            lines.append(f"\nInvalid evidence for {row['session_id']}: {'; '.join(row['validation_errors'])}\n")
        late = [uid for uid, values in row["previews"]["utterances"].items() if values["preview_after_final_ready"]]
        if late:
            observed = (
                "; ".join(
                    f"{client}: {len(data['preview_acks_after_final_ack'])} preview ACKs after final ACK, "
                    f"{len(data['preview_ack_order_unassessable_events'])} ACK order unassessable"
                    for client, data in row["browsers"].items()
                )
                or "No visible browser ACK evidence"
            )
            lines.append(
                f"\n{row['session_id']}: preview emission after final readiness for utterances {', '.join(late)}. "
                f"This requires browser-order review; it does not establish stale repaint. {observed}.\n"
            )
    lines.extend(["", "## Interpretation", ""])
    lines.extend(f"- **{name}:** {definition}" for name, definition in report["definitions"].items())
    lines.extend(
        [
            "",
            "Raw-file hashes, signed per-bound deltas versus both controls, opening-to-closing drift, changed examples, per-client timing, lexical revisions, counters and resource records are retained in `analysis.json`. Missing/failed records are explicit; no absent value becomes zero latency.",
            "",
        ]
    )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--metrics", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    browser = parser.add_mutually_exclusive_group()
    browser.add_argument("--client-id", help="Use this literal connection ID in every run (usually IDs change)")
    browser.add_argument(
        "--client-map", type=Path, help="JSON session→connection IDs; default binds the sole visible client per session"
    )
    args = parser.parse_args()
    if any(
        args.output.resolve().is_relative_to(path.resolve()) for path in (args.input, args.metrics or args.input.parent)
    ):
        parser.error("Use a separate report output directory; evidence inputs are read-only")
    report = analyze(
        args.input,
        args.metrics or args.input.parent,
        client_id=args.client_id,
        client_map=json.loads(args.client_map.read_text()) if args.client_map else None,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "analysis.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    (args.output / "README.md").write_text(markdown(report))
    print(f"{report['matrix_status']}: {report['valid_runs']}/{report['expected_runs']} valid; report {args.output}")


if __name__ == "__main__":
    main()
