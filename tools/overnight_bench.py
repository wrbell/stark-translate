"""Serial, reproducible latency experiments with honest coverage/quality gates.

Run from a frozen checkout. One child owns inference at a time; stable display
ports let a visible audience browser reconnect across paired runs. Missing ACKs
remain missing and never become fabricated browser latency measurements.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.latency_experiments import LatencyExperiments
from tools.mac_evaluation import _report_browser_acknowledgments, environment, stats, write_json
from tools.replay_bench import argument_value, run_replay


def jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def number(value):
    if isinstance(value, bool):
        return None
    try:
        value = float(value)
        return value if 0 <= value < float("inf") else None
    except (TypeError, ValueError):
        return None


def ranges(rows: list[dict]) -> list[tuple[float, float]]:
    intervals = []
    for row in rows:
        start, end, rate = (number(row.get(key)) for key in ("sample_start", "sample_end", "sample_rate"))
        if start is not None and end is not None and rate and end > start:
            intervals.append((start / rate, end / rate))
    return merge_ranges(intervals)


def merge_ranges(intervals):
    merged: list[tuple[float, float]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = merged[-1][0], max(end, merged[-1][1])
        else:
            merged.append((start, end))
    return merged


def coverage_missing(reference, candidate) -> float:
    """Seconds covered by baseline finals but absent from candidate finals."""
    missing = 0.0
    for start, end in reference:
        overlap = sum(max(0.0, min(end, b) - max(start, a)) for a, b in candidate)
        missing += max(0.0, end - start - overlap)
    return missing


def segment_key(row):
    return tuple(
        str(row.get(key, ""))
        for key in (
            "sample_start",
            "sample_end",
            "sample_rate",
            "speech_end_sample",
            "endpoint_reason",
            "timing_source",
        )
    )


def translated_preview(row):
    return str(row.get("text_en" if row.get("target_lang") == "en" else "text_es") or "").strip()


def source_snapshot():
    source = environment()
    for name in ("torch", "silero-vad", "onnxruntime", "numpy", "scipy", "websockets", "sounddevice"):
        try:
            source["versions"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            source["versions"][name] = None
    tracked = (
        subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True).stdout.decode().split("\0")
    )
    source["all_code_sha256"] = {
        name: fingerprint(ROOT / name)
        for name in tracked
        if name.endswith((".py", ".js", ".html", ".toml")) and (ROOT / name).is_file()
    }
    # Include this executing harness even during its first, untracked smoke.
    source["all_code_sha256"]["tools/overnight_bench.py"] = fingerprint(Path(__file__))
    source["execution_environment"] = {
        key: value
        for key, value in os.environ.items()
        if key
        in {
            "STARK_MODELS_DIR",
            "HF_HOME",
            "HF_HUB_CACHE",
            "HUGGINGFACE_HUB_CACHE",
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
        }
        or key.startswith("MLX_")
    }
    if (ROOT / ".env").exists():
        raise ValueError("Frozen benchmark checkout must not contain .env; use explicit spec settings")
    return source


def source_identity(source):
    return {
        key: source.get(key) for key in ("all_code_sha256", "versions", "platform", "python", "execution_environment")
    }


def stable_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def configuration(spec, config, clip, size):
    # Ambient operator/lite/session settings must not silently change a probe.
    env = {key: value for key, value in os.environ.items() if not key.startswith("STARK_") or key == "STARK_MODELS_DIR"}
    declared = {**spec.get("baseline_env", {}), **config.get("env", {})}
    known = {"STARK_EXPERIMENT_" + name.upper() for name in LatencyExperiments().as_dict()}
    for key, value in declared.items():
        if not isinstance(value, str) or (key.startswith("STARK_EXPERIMENT_") and key not in known):
            raise ValueError(f"Invalid or unknown experiment environment field: {key}")
        if key.startswith("STARK_") and key not in known and not key.startswith(("STARK_VAD_", "STARK_TRANSLATE_")):
            raise ValueError(f"Use dedicated benchmark controls rather than overriding {key}")
    env.update(declared)
    env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", STARK_REPLAY_SPEED="1")
    env["STARK_SESSION_KIND"] = "synthetic" if clip.get("provenance", "").startswith("synthetic") else "replay"
    extras = [*spec.get("baseline_arguments", []), *config.get("arguments", [])]
    allowed = {
        "--silence-trigger",
        "--partial-interval",
        "--vad-backend",
        "--idle-warmup-only",
        "--no-idle-warmup-only",
        "--final-aware-partials",
        "--no-final-aware-partials",
        "--routing-policy",
        "--terminology-prompt",
        "--gain",
        "--vad-threshold",
        "--profile",
        "--no-tts",
    }
    if any(arg.startswith("--") and arg.split("=", 1)[0] not in allowed for arg in extras):
        raise ValueError("Experiment arguments override protected runtime controls or are unsupported")
    if argument_value(extras, "--profile", "standard") != "standard":
        raise ValueError("Mac latency screens require --profile standard")
    arguments = [
        *extras,
        "--profile",
        "standard",
        "--no-tts",
        "--model-family",
        "gemma4",
        "--gemma4-size",
        size,
        "--no-mts",
        "--stt-backend",
        "parakeet-mlx" if clip["lang"] == "en" else "mlx",
        "--replay-speed",
        "1",
        "--replay-wait-client-seconds",
        "15",
    ]
    return arguments, env, LatencyExperiments.from_env(env).as_dict()


def completion_errors(result, observed, metrics, expected, clip, source):
    errors = []
    session = result["session_id"]
    lifecycle = result.get("session_lifecycle", {})
    diagnostics = metrics / f"diagnostics_{session}.jsonl"
    if (
        lifecycle.get("status") != "completed"
        or lifecycle.get("session_id") != session
        or lifecycle.get("schema_version") != 1
        or lifecycle.get("exit_code") != 0
        or not diagnostics.is_file()
        or lifecycle.get("diagnostics")
        != {"sha256": fingerprint(diagnostics), "size_bytes": diagnostics.stat().st_size}
    ):
        errors.append("Completion marker does not prove unchanged, fully drained diagnostics")
    if result.get("returncode") != 0 or result.get("timed_out"):
        errors.append("Subprocess did not complete successfully")
    metadata = result.get("session_metadata", {})
    if metadata.get("latency_experiment_configuration") != expected:
        errors.append("Resolved startup experiment configuration differs from requested values")
    summary = observed["session_summary"]
    barrier = metadata.get("replay_client_wait", {})
    if (
        barrier.get("requested_seconds") != 15
        or barrier.get("status") != "connected"
        or not isinstance(barrier.get("actual_wait_seconds"), (int, float))
        or not math.isfinite(barrier["actual_wait_seconds"])
        or barrier["actual_wait_seconds"] < 0
        or barrier.get("connected_clients", 0) < 1
        or barrier.get("visibility_confirmed") is not False
        or summary.get("replay_client_wait") != barrier
    ):
        errors.append("Replay client barrier did not confirm a connected client before capture")
    if summary.get("latency_experiment_configuration") != expected:
        errors.append("Resolved summary experiment configuration differs from requested values")
    if not observed["final_count"] or summary.get("chunks_completed") != observed["final_count"]:
        errors.append("Final CSV coverage disagrees with completed chunk count or is empty")
    if any(
        str(row.get("timing_schema_version")) != "2" or row.get("timing_source") != "replay_realtime"
        for row in observed["finals"]
    ):
        errors.append("Final measurements do not all use the real-time schema-2 capture clock")
    if metadata.get("profile", {}).get("name") != "standard" or metadata.get("backend") != "mlx":
        errors.append("Resolved runtime is not standard MLX")
    kind = "synthetic" if clip.get("provenance", "").startswith("synthetic") else "replay"
    if (
        metadata.get("session_id") != session
        or metadata.get("session_kind") != kind
        or metadata.get("input_audio_sha256") != clip["sha256"]
        or metadata.get("source_lang") != clip["lang"]
        or metadata.get("replay_speed") != 1
    ):
        errors.append("Resolved source provenance or real-time replay identity differs from the spec")
    if lifecycle.get("pipeline_sha256") != source["all_code_sha256"].get("dry_run_ab.py"):
        errors.append("Started pipeline bytes differ from the frozen source snapshot")
    return errors


def preview_browsers(partials, acknowledgments, session):
    """Join exact preview IDs; keep every visible browser in its own cohort."""
    grouped = defaultdict(list)
    for row in partials:
        if row.get("event_id") and translated_preview(row) and row.get("session_id") == session:
            grouped[row["event_id"]].append(row)
    emitted = {key: values[0] for key, values in grouped.items() if len(values) == 1}
    clients: set[str] = set()
    events: dict[tuple, dict] = defaultdict(dict)
    cohort_keys = {
        (row.get("timing_source", "unknown"), row.get("caption_delivery_mode", "unknown")) for row in emitted.values()
    }
    for ack in acknowledgments:
        if (
            ack.get("event") != "caption_rendered"
            or ack.get("session_id") != session
            or ack.get("timing_schema_version") != 2
            or ack.get("visible") is not True
            or not isinstance(ack.get("client_id"), str)
            or not ack["client_id"]
        ):
            continue
        if ack.get("stage") in {"partial", "complete"}:
            clients.add(ack["client_id"])
        if not isinstance(ack.get("event_id"), str):
            continue
        row = emitted.get(ack.get("event_id"))
        if ack.get("stage") != "partial" or row is None:
            continue
        if row.get("timing_schema_version") != 2 or any(
            ack.get(key) != row.get(key) for key in ("preview_revision", "utterance_id", "timing_source")
        ):
            continue
        key = (ack["client_id"], row.get("timing_source", "unknown"), row.get("caption_delivery_mode", "unknown"))
        events[key].setdefault(ack["event_id"], ack)
    rows = []
    for client_id in sorted(clients):
        for timing_source, delivery in sorted(cohort_keys):
            first: dict = {}
            render = []
            matching = {
                key: row
                for key, row in emitted.items()
                if row.get("timing_source", "unknown") == timing_source
                and row.get("caption_delivery_mode", "unknown") == delivery
            }
            received = events[(client_id, timing_source, delivery)]
            for event_id, ack in received.items():
                uid = emitted[event_id].get("utterance_id")
                delay = number(ack.get("speech_start_to_preview_ack_upper_bound_ms"))
                if uid is not None and delay is not None and timing_source == "replay_realtime":
                    first[uid] = min(first.get(uid, delay), delay)
                if (overhead := number(ack.get("receive_to_render_ms"))) is not None:
                    render.append(overhead)
            rows.append(
                {
                    "client_id": client_id,
                    "timing_source": timing_source,
                    "caption_delivery_mode": delivery,
                    "visible_preview_events": len(received),
                    "emitted_translated_preview_events": len(matching),
                    "missing_preview_event_ids": sorted(matching.keys() - received.keys()),
                    "first_preview_ack_upper_bound_ms": stats(list(first.values())),
                    "receive_to_render_ms": stats(render),
                }
            )
    return {
        "cohorts": rows,
        "client_count": len(clients),
        "emitted_translated_preview_events": len(emitted),
        "available": any(row["visible_preview_events"] for row in rows),
    }


def inspect_run(report: dict, metrics: Path) -> dict:
    session = report["session_id"]
    csv_path = metrics / f"ab_metrics_{session}.csv"
    with csv_path.open(newline="") if csv_path.exists() else open(os.devnull) as source:
        rows = list(csv.DictReader(source))
    partials = jsonl(metrics / f"partials_{session}.jsonl")
    acknowledgments = jsonl(metrics / f"display_metrics_{session}.jsonl")
    first: dict = {}
    previous: dict = {}
    gaps = []
    for row in sorted(
        partials,
        key=lambda row: (
            number(row.get("emitted_at_ms")) if number(row.get("emitted_at_ms")) is not None else float("inf")
        ),
    ):
        # A source-only/non-visible/empty translation is not a translated preview.
        if not translated_preview(row):
            continue
        uid = row.get("utterance_id")
        if uid is None:
            continue
        delay = number(row.get("speech_start_to_partial_ms"))
        if uid not in first and delay is not None:
            first[uid] = delay
        ready = number(row.get("emitted_at_ms"))
        if ready is not None:
            if uid in previous and ready >= previous[uid]:
                gaps.append(ready - previous[uid])
            previous[uid] = ready
    endpoint_samples = defaultdict(list)
    for row in rows:
        latency = number(row.get("speech_end_to_final_ms"))
        if latency is not None and str(row.get("timing_schema_version")) == "2":
            key = "|".join(str(row.get(k) or "unknown") for k in ("endpoint_reason", "timing_source"))
            endpoint_samples[key].append(latency)
    covered = ranges(rows)
    diagnostic_records = jsonl(metrics / f"diagnostics_{session}.jsonl")
    hardware_path = metrics / f"hardware_{session}.json"
    hardware = json.loads(hardware_path.read_text()) if hardware_path.exists() else {}
    summary = next((record for record in reversed(diagnostic_records) if record.get("event") == "session_summary"), {})
    final_utterances = {str(row["utterance_id"]) for row in rows if row.get("utterance_id") not in (None, "")}
    return {
        "session_id": session,
        "finals": rows,
        "partials": partials,
        "final_count": len(rows),
        "partial_count": len(partials),
        "coverage_intervals_s": covered,
        "covered_source_s": sum(b - a for a, b in covered),
        "endpoint_samples_ms": dict(endpoint_samples),
        "first_preview_samples_ms": list(first.values()),
        "first_preview_by_utterance_ms": {str(uid): delay for uid, delay in first.items()},
        "final_utterances_without_preview": sorted(final_utterances - {str(uid) for uid in first}),
        "final_utterances_with_known_identity": len(final_utterances),
        "within_utterance_gap_samples_ms": gaps,
        "processed_partial_audio_s": sum(
            number(row.get("processed_audio_s", row.get("buffer_s"))) or 0 for row in partials
        ),
        "browser": _report_browser_acknowledgments(rows, acknowledgments, session),
        "preview_browser": preview_browsers(partials, acknowledgments, session),
        "hardware_profile": hardware,
        "resource_snapshots": [record["resources"] for record in diagnostic_records if record.get("resources")],
        "session_summary": summary,
    }


def compare_outputs(baseline: dict, candidate: dict) -> dict:
    def unique(rows):
        groups = defaultdict(list)
        for row in rows:
            groups[segment_key(row)].append(row)
        return {key: items[0] for key, items in groups.items() if len(items) == 1}, sum(
            len(items) - 1 for items in groups.values()
        )

    left, left_duplicates = unique(baseline["finals"])
    right, right_duplicates = unique(candidate["finals"])
    shared = left.keys() & right.keys()
    changed = []
    matched_latency: dict[tuple | str, dict] = defaultdict(
        lambda: {"baseline": [], "candidate": [], "paired_delta_ms": []}
    )
    for key in sorted(shared):
        a, b = left[key], right[key]
        before, after = number(a.get("speech_end_to_final_ms")), number(b.get("speech_end_to_final_ms"))
        if (
            before is not None
            and after is not None
            and str(a.get("timing_schema_version")) == str(b.get("timing_schema_version")) == "2"
        ):
            endpoint = "|".join(str(a.get(name) or "unknown") for name in ("endpoint_reason", "timing_source"))
            matched_latency[endpoint]["baseline"].append(before)
            matched_latency[endpoint]["candidate"].append(after)
            matched_latency[endpoint]["paired_delta_ms"].append(after - before)
        if (a.get("english"), a.get("spanish_a")) != (b.get("english"), b.get("spanish_a")):
            changed.append(
                {
                    "segment": key,
                    "before": {k: a.get(k) for k in ("english", "spanish_a")},
                    "after": {k: b.get(k) for k in ("english", "spanish_a")},
                }
            )
    return {
        "matched_final_bounds": len(shared),
        "baseline_finals": len(baseline["finals"]),
        "candidate_finals": len(candidate["finals"]),
        "baseline_duplicate_bounds": left_duplicates,
        "candidate_duplicate_bounds": right_duplicates,
        "baseline_unmatched": len(left.keys() - right.keys()),
        "candidate_unmatched": len(right.keys() - left.keys()),
        "changed_matched_outputs": changed,
        "matched_endpoint_latency_ms": {
            key: {
                "baseline": stats(values["baseline"]),
                "candidate": stats(values["candidate"]),
                "paired_delta_ms": values["paired_delta_ms"],
            }
            for key, values in matched_latency.items()
        },
        "missing_source_s": coverage_missing(baseline["coverage_intervals_s"], candidate["coverage_intervals_s"]),
        "quality_review": "pending; output agreement is not reference quality",
    }


def schedule(spec: dict, repeats: int):
    """Alternate size and configuration order, with baseline anchors each repeat."""
    configurations = spec["experiments"]
    names = [item["name"] for item in configurations]
    if not names or names[0] != "baseline" or len(set(names)) != len(names):
        raise ValueError("Experiments need one first baseline and unique names")
    if any(not re.fullmatch(r"[a-z0-9_]+", name) for name in names):
        raise ValueError("Invalid experiment name")
    if "baseline_anchor" in names:
        raise ValueError("baseline_anchor is reserved for the closing baseline")
    if repeats < 1:
        raise ValueError("At least one repeat is required")
    for repeat in range(repeats):
        candidates = configurations[1:]
        if repeat % 2:
            candidates = list(reversed(candidates))
        for config in [configurations[0], *candidates, {**configurations[0], "name": "baseline_anchor"}]:
            for clip in spec["clips"]:
                sizes = spec.get("sizes", ["e4b", "e2b"])
                for size in sizes if repeat % 2 == 0 else reversed(sizes):
                    yield repeat, config, clip, size


def run(args):
    spec = json.loads(args.spec.read_text())
    clips, sizes = spec.get("clips", []), spec.get("sizes", ["e4b", "e2b"])
    if not clips or len({clip["id"] for clip in clips}) != len(clips):
        raise ValueError("At least one clip with unique IDs is required")
    if not sizes or len(set(sizes)) != len(sizes) or set(sizes) - {"e4b", "e2b"}:
        raise ValueError("Sizes must be unique e4b/e2b entries")
    if any(not re.fullmatch(r"[A-Za-z0-9_-]+", clip["id"]) or clip.get("lang") not in {"en", "es"} for clip in clips):
        raise ValueError("Invalid clip ID or source language")
    planned = list(schedule(spec, args.repeats))
    for _, config, clip, size in planned:
        configuration(spec, config, clip, size)  # reject typos before any inference starts
    source = source_snapshot()
    args.output.mkdir(parents=True, exist_ok=True)
    provenance = {
        "spec": spec,
        "spec_sha256": fingerprint(args.spec),
        "environment": source,
        "repeats": args.repeats,
        "tag": args.tag,
    }
    lock = args.output / "provenance.json"
    if lock.exists():
        old = json.loads(lock.read_text())
        for key in ("spec", "repeats", "tag"):
            if old[key] != provenance[key]:
                raise ValueError("Run configuration changed; use a new output directory")
        if source_identity(old["environment"]) != source_identity(source):
            raise ValueError("Source changed; use a new frozen cohort")
    else:
        write_json(lock, provenance, exclusive=True)
    for index, (repeat, config, clip, size) in enumerate(planned):
        session = f"{args.tag}_{config['name']}_{size}_r{repeat}_{clip['id']}_{clip['lang']}"
        destination = args.output / f"{session}.json"
        if destination.exists():
            existing = json.loads(destination.read_text())
            if existing.get("error") and not getattr(args, "continue_on_error", False):
                raise RuntimeError(f"Previous failed run requires investigation: {session}")
            print(f"Recorded already (including failures): {session}", flush=True)
            continue
        wav = (ROOT / clip["path"]).resolve()
        if fingerprint(wav) != clip["sha256"]:
            raise ValueError(f"Audio identity changed: {wav}")
        arguments, env, expected = configuration(spec, config, clip, size)
        if source_identity(source_snapshot()) != source_identity(source):
            raise ValueError("Source or execution environment changed before the next frozen run")
        print(f"Starting {session}", flush=True)
        try:
            result = run_replay(
                clip,
                wav,
                session,
                arguments,
                index,
                ROOT / "metrics",
                env=env,
                ports=(args.ws_port, args.http_port),
                timeout_s=args.timeout,
            )
        except RuntimeError as exc:
            partial = ROOT / "metrics" / f"replay_{session}.json"
            result = json.loads(partial.read_text()) if partial.exists() else {"session_id": session}
            result["error"] = str(exc)
        observed = inspect_run(result, ROOT / "metrics")
        errors = completion_errors(result, observed, ROOT / "metrics", expected, clip, source)
        if source_identity(source_snapshot()) != source_identity(source):
            errors.append("Source or execution environment changed during inference")
        if errors:
            result["error"] = "; ".join([*([result["error"]] if result.get("error") else []), *errors])
        result.update(
            experiment=config["name"],
            repeat=repeat,
            size=size,
            clip_id=clip["id"],
            language=clip["lang"],
            source_cohort_sha256=stable_hash(source_identity(source)),
            requested_experiment_configuration=expected,
            resolved_experiment_configuration=result.get("session_metadata", {}).get(
                "latency_experiment_configuration"
            ),
            completion_validation={"valid": not errors and not result.get("error"), "errors": errors},
            tts_requested=False,
            profile_requested="standard",
            replay_wait_client_seconds_requested=15,
            experiment_env={
                key: value
                for key, value in env.items()
                if key.startswith("STARK_") or key in {"HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"}
            },
            observed=observed,
        )
        write_json(destination, result)
        print(f"Finished {session}: {result.get('error', 'OK')}", flush=True)
        if result.get("error") and not getattr(args, "continue_on_error", False):
            raise RuntimeError(f"Stopping after failed probe; inspect {destination}")


def report(args):
    runs = [
        json.loads(path.read_text())
        for path in sorted(args.input.glob("*.json"))
        if path.name not in {"provenance.json", "comparison.json"}
    ]
    runs = [run for run in runs if "observed" in run]

    def eligible(item):
        return not item.get("error") and item.get("completion_validation", {}).get("valid") is True

    def cohort(item):
        metadata = item.get("session_metadata", {})
        return stable_hash(
            {
                "source": item.get("source_cohort_sha256", "unknown"),
                "clip": item.get("clip", {}).get("sha256"),
                "models": item.get("session_lifecycle", {}).get("models"),
                "stt_backend": metadata.get("stt_backend"),
            }
        )

    baselines = {
        (run["repeat"], run["size"], run["clip_id"], cohort(run)): run
        for run in runs
        if run["experiment"] == "baseline" and eligible(run)
    }
    groups = defaultdict(list)
    for run in runs:
        groups[(run["experiment"], run["size"], run["clip_id"], cohort(run))].append(run)
    result = {
        "schema_version": 1,
        "recorded_runs": len(runs),
        "eligible_runs": sum(eligible(run) for run in runs),
        "excluded_sessions": [
            {"session_id": run["session_id"], "error": run.get("error", "completion not verified")}
            for run in runs
            if not eligible(run)
        ],
        "metric_definitions": {
            "final": "schema 2 estimated last speech capture frame to final payload readiness, separated by endpoint and timing source",
            "first_preview": "first nonempty translated preview readiness minus first captured speech sample of each utterance",
            "update_gap": "consecutive translated preview readiness times within the same utterance only",
            "browser": "speech-end-to-ACK is an upper bound including return network time; only visible real-time ACKs qualify",
            "coverage": "union of final source sample intervals; not VAD recall or reference WER",
        },
        "groups": [],
    }
    provenance_path = args.input / "provenance.json"
    if provenance_path.exists():
        provenance = json.loads(provenance_path.read_text())
        expected_sessions = {
            f"{provenance['tag']}_{config['name']}_{size}_r{repeat}_{clip['id']}_{clip['lang']}"
            for repeat, config, clip, size in schedule(provenance["spec"], provenance["repeats"])
        }
        result["expected_runs"] = len(expected_sessions)
        result["missing_sessions"] = sorted(expected_sessions - {run["session_id"] for run in runs})
    markdown = [
        "# Overnight latency experiment results",
        "",
        "All configurations remain opt-in. Original predictions and changed examples are retained in JSON. Natural bilingual reference quality and human meaning review remain separate pending gates.",
        "",
        "| Experiment | Model | Runs / failures | Silence final p50 / p95 ms | First preview p50 / p95 ms | Missing source seconds | Changed matched finals |",
        "|---|---|---|---|---|---|---|",
    ]
    for (name, size, clip_id, cohort_id), observations in sorted(groups.items()):
        endpoints = defaultdict(list)
        first, gaps, comparisons = [], [], []
        for item in observations:
            if not eligible(item):
                continue
            observed = item["observed"]
            for key, values in observed["endpoint_samples_ms"].items():
                endpoints[key].extend(values)
            first.extend(observed["first_preview_samples_ms"])
            gaps.extend(observed["within_utterance_gap_samples_ms"])
            baseline = baselines.get((item["repeat"], size, item["clip_id"], cohort_id))
            if baseline:
                comparisons.append(
                    {"session_id": item["session_id"], **compare_outputs(baseline["observed"], observed)}
                )
        failures = sum(not eligible(item) for item in observations)
        entry = {
            "experiment": name,
            "size": size,
            "clip_id": clip_id,
            "cohort_sha256": cohort_id,
            "runs": len(observations),
            "failures": failures,
            "eligible_runs": len(observations) - failures,
            "endpoints": {key: stats(values) for key, values in endpoints.items()},
            "first_preview_ms": stats(first),
            "update_gap_ms": stats(gaps),
            "output_comparisons": comparisons,
            "browser_sessions": [
                {
                    "session_id": item["session_id"],
                    "final": item["observed"]["browser"],
                    "preview": item["observed"]["preview_browser"],
                }
                for item in observations
                if eligible(item)
            ],
            "first_preview_missing_coverage": [
                {
                    "session_id": item["session_id"],
                    "final_utterances_without_preview": item["observed"].get("final_utterances_without_preview", []),
                }
                for item in observations
                if eligible(item)
            ],
            "reference_wer": None,
            "reference_translation_score": None,
            "human_review": "pending",
            "promotion": "not evaluated; requires matched coverage, quality, sufficient endpoint samples and browser coverage",
        }
        result["groups"].append(entry)
        silence = entry["endpoints"].get("silence|replay_realtime", {})
        missing = sum(pair["missing_source_s"] for pair in comparisons)
        changes = sum(len(pair["changed_matched_outputs"]) for pair in comparisons)

        def pair(stat):
            return f"{stat.get('p50', '—')} / {stat.get('p95', '—')}"

        markdown.append(
            f"| {name} ({clip_id}) | {size} | {len(observations)} / {failures} | {pair(silence)} | {pair(entry['first_preview_ms'])} | {missing:.3f} | {changes} |"
        )
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "comparison.json", result)
    (args.output / "README.md").write_text("\n".join(markdown) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    execute = commands.add_parser("run")
    execute.add_argument("--spec", type=Path, required=True)
    execute.add_argument("--output", type=Path, required=True)
    execute.add_argument("--tag", required=True)
    execute.add_argument("--repeats", type=int, default=3)
    execute.add_argument("--ws-port", type=int, default=8765)
    execute.add_argument("--http-port", type=int, default=8080)
    execute.add_argument("--timeout", type=float, default=240)
    execute.add_argument(
        "--continue-on-error", action="store_true", help="Record failures and continue; failed samples remain excluded"
    )
    summarize = commands.add_parser("report")
    summarize.add_argument("--input", type=Path, required=True)
    summarize.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "run":
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", args.tag):
            parser.error("Invalid session tag")
        run(args)
    else:
        report(args)


if __name__ == "__main__":
    main()
