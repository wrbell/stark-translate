"""Frozen serial server-latency screens; browser/device certification stays separate.

Opening-control source spans are frozen before candidate scoring. Every candidate
must beat both opening and closing controls, without hiding lost source audio.
This engineering screen never promotes a model or substitutes for bilingual review.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path

from tools.fixed_span_delivery import fixed_span_delivery
from tools.latency_experiments import LatencyExperiments
from tools.mac_evaluation import stats, write_json
from tools.overnight_bench import (
    ROOT,
    coverage_missing,
    fingerprint,
    inspect_run,
    jsonl,
    ranges,
    schedule,
    source_identity,
    source_snapshot,
    stable_hash,
    translated_preview,
)
from tools.replay_bench import run_replay
from tools.replay_integrity import audit_replay_integrity
from tools.source_coverage import merged


def configuration(spec, config, clip, size):
    profile = spec.get("profile", "standard")
    if profile not in {"standard", "lite-cpu"}:
        raise ValueError("Screen supports standard or lite-cpu only")
    env = {k: v for k, v in os.environ.items() if not k.startswith("STARK_") or k == "STARK_MODELS_DIR"}
    declared = {**spec.get("baseline_env", {}), **config.get("env", {})}
    known = {"STARK_EXPERIMENT_" + name.upper() for name in LatencyExperiments().as_dict()}
    if any(k not in known or not isinstance(v, str) for k, v in declared.items()):
        raise ValueError("Only explicit experiment environment fields are supported")
    env.update(declared)
    env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", STARK_SESSION_KIND="replay", STARK_REPLAY_SPEED="1")
    cadence = config.get("partial_interval", spec.get("partial_interval", 0.6))
    if cadence not in {0.6, 0.9, 1.2}:
        raise ValueError("Unsupported frozen partial cadence")
    stt = config.get("stt_backend", "parakeet-mlx" if clip["lang"] == "en" else "mlx")
    if stt not in {"parakeet-mlx", "mlx"}:
        raise ValueError("Unsupported Standard STT")
    arguments = [
        "--profile",
        profile,
        "--no-tts",
        "--model-family",
        "gemma4",
        "--gemma4-size",
        size,
        "--no-mts",
        "--stt-backend",
        stt if profile == "standard" else "faster-whisper",
        "--replay-speed",
        "1",
        "--silence-trigger",
        "0.5",
        "--partial-interval",
        str(cadence),
        "--gain",
        "1",
        "--routing-policy",
        "legacy",
        "--terminology-prompt",
        "none",
        "--no-idle-warmup-only",
        "--no-final-aware-partials",
    ]
    return arguments, env, LatencyExperiments.from_env(env).as_dict()


def validate_completion(result, observed, diagnostics, expected, clip, profile, source):
    errors = []
    lifecycle, metadata = result.get("session_lifecycle", {}), result.get("session_metadata", {})
    if result.get("returncode") != 0 or result.get("timed_out") or lifecycle.get("status") != "completed":
        errors.append("Replay did not finish successfully")
    if not diagnostics.is_file() or lifecycle.get("diagnostics") != {
        "sha256": fingerprint(diagnostics),
        "size_bytes": diagnostics.stat().st_size,
    }:
        errors.append("Persisted diagnostics identity mismatch")
    if metadata.get("input_audio_sha256") != clip["sha256"] or metadata.get("replay_speed") != 1:
        errors.append("Source or replay speed mismatch")
    if metadata.get("source_lang") != clip["lang"] or metadata.get("profile", {}).get("name") != profile:
        errors.append("Language or profile mismatch")
    summary = observed["session_summary"]
    if any(r.get("latency_experiment_configuration") != expected for r in (metadata, summary)):
        errors.append("Resolved experiment settings mismatch")
    if not summary.get("source_coverage", {}).get("complete"):
        errors.append("Source accounting incomplete")
    if any(r.get("state", "").endswith("_error") for r in summary.get("source_coverage", {}).get("outcomes", [])):
        errors.append("Inference error in source ledger")
    if any(r.get("failed") for r in summary.get("latency_trace", {}).get("events", [])):
        errors.append("Physical worker failed")
    if metadata.get("backend") != ("mlx" if profile == "standard" else "cpu"):
        errors.append("Resolved backend mismatch")
    if not observed["final_count"] or summary.get("chunks_completed") != observed["final_count"]:
        errors.append("Completed final count mismatch")
    if lifecycle.get("pipeline_sha256") != source["all_code_sha256"].get("dry_run_ab.py"):
        errors.append("Pipeline source mismatch")
    return errors


def run(args):
    spec = json.loads(args.spec.read_text())
    if not re.fullmatch(r"[a-zA-Z0-9_]+", args.tag):
        raise ValueError("Invalid tag")
    planned = list(schedule(spec, args.repeats))
    if len({c["id"] for c in spec["clips"]}) != len(spec["clips"]):
        raise ValueError("Duplicate clip identifiers")
    sizes = spec.get("sizes", ["e4b", "e2b"])
    if not sizes or len(set(sizes)) != len(sizes) or set(sizes) - {"e4b", "e2b"}:
        raise ValueError("Invalid or duplicate model sizes")
    for _, config, clip, size in planned:
        configuration(spec, config, clip, size)
        if clip.get("lang") not in {"en", "es"} or not re.fullmatch(r"[a-z0-9_]+", clip["id"]):
            raise ValueError("Invalid clip")
    source = source_snapshot()
    identity = stable_hash(source_identity(source))
    args.output.mkdir(parents=True, exist_ok=True)
    provenance = {
        "spec": spec,
        "spec_sha256": fingerprint(args.spec),
        "source": source,
        "repeats": args.repeats,
        "tag": args.tag,
        "source_identity": identity,
    }
    lock = args.output / "provenance.json"
    if lock.exists():
        if json.loads(lock.read_text()) != provenance:
            raise ValueError("Frozen cohort changed; choose a new output")
    else:
        write_json(lock, provenance, exclusive=True)
    for index, (repeat, config, clip, size) in enumerate(planned):
        session = f"{args.tag}_{config['name']}_{size}_r{repeat}_{clip['id']}_{clip['lang']}"
        destination = args.output / f"{session}.json"
        if destination.exists():
            continue  # failed runs remain failed; never overwrite favorable replacements
        wav = (ROOT / clip["path"]).resolve()
        if fingerprint(wav) != clip["sha256"] or stable_hash(source_identity(source_snapshot())) != identity:
            raise ValueError("Frozen source/audio changed")
        arguments, env, expected = configuration(spec, config, clip, size)
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
                backend="mlx" if spec.get("profile", "standard") == "standard" else "cpu",
                ports=(args.ws_port, args.http_port),
                timeout_s=args.timeout,
            )
        except RuntimeError as exc:
            path = ROOT / "metrics" / f"replay_{session}.json"
            result = json.loads(path.read_text()) if path.exists() else {"session_id": session}
            result["error"] = str(exc)
        observed = inspect_run(result, ROOT / "metrics")
        diagnostics = ROOT / "metrics" / f"diagnostics_{session}.jsonl"
        errors = validate_completion(
            result, observed, diagnostics, expected, clip, spec.get("profile", "standard"), source
        )
        if stable_hash(source_identity(source_snapshot())) != identity:
            errors.append("Source changed during replay")
        records = jsonl(diagnostics)
        finals = [r for r in records if r.get("timing_schema_version") == 2]
        integrity = audit_replay_integrity(
            data_root=ROOT,
            session_id=session,
            metadata=result.get("session_metadata", {}),
            diagnostic_finals=finals,
            partials=observed["partials"],
            trace=observed["session_summary"].get("latency_trace", {}),
        )
        if integrity["status"] != "passed":
            errors.append("Replay PCM/identity integrity failed: " + "; ".join(integrity["errors"]))
        result.update(
            experiment=config["name"],
            repeat=repeat,
            size=size,
            clip_id=clip["id"],
            source_identity=identity,
            observed=observed,
            diagnostic_finals=finals,
            replay_integrity=integrity,
            completion_errors=errors,
            requested_settings=expected,
            measurement_scope="server real-time replay; no physical visibility certification",
        )
        write_json(destination, result, exclusive=True)
        print(f"Finished {session}: {errors or result.get('error') or 'OK'}", flush=True)


def speech_intervals(run, *, finalized_only=False):
    observed = run["observed"]["session_summary"]["source_coverage"]["observed"]
    if not observed or any(type(row.get("vad_positive")) is not bool for row in observed):
        raise ValueError("Missing frozen per-frame VAD classification")
    positive = [(row["start"], row["end"]) for row in observed if row["vad_positive"]]
    if finalized_only:
        positive = [
            (max(a, row["sample_start"]), min(b, row["speech_end_sample"]))
            for a, b in positive
            for row in run["diagnostic_finals"]
            if max(a, row["sample_start"]) < min(b, row["speech_end_sample"])
        ]
    return merged(positive)


def score_pair(opening, candidate, closing):
    runs = (opening, candidate, closing)
    if len({r["source_identity"] for r in runs}) != 1 or len({r["clip"]["sha256"] for r in runs}) != 1:
        raise ValueError("Cannot pool different runtime/audio identities")
    anchors = [
        {
            "id": str(i),
            **{
                k: row[k]
                for k in ("sample_start", "sample_end", "speech_end_sample", "endpoint_reason", "padding_samples")
            },
        }
        for i, row in enumerate(opening["diagnostic_finals"])
    ]
    rate = opening["diagnostic_finals"][0]["sample_rate"]
    buffered_metrics = [fixed_span_delivery(anchors, r["diagnostic_finals"], sample_rate=rate) for r in runs]
    positive = speech_intervals(opening)
    for anchor in anchors:
        anchor["required_intervals"] = [
            [max(a, anchor["sample_start"]), min(b, anchor["speech_end_sample"])]
            for a, b in positive
            if max(a, anchor["sample_start"]) < min(b, anchor["speech_end_sample"])
        ]
    metrics = [fixed_span_delivery(anchors, r["diagnostic_finals"], sample_rate=rate) for r in runs]
    before, current, after = metrics
    reasons = []
    if any(r.get("error") or r.get("completion_errors") for r in runs):
        reasons.append("failed_or_incomplete_run")
    if any(m["status"] != "complete" for m in metrics):
        reasons.append("missing_fixed_source_span")
    for control in (before, after):
        if current.get("p50_ms") is None or control.get("p50_ms") is None:
            reasons.append("missing_median")
            continue
        gain = control["p50_ms"] - current["p50_ms"]
        if not (gain >= 150 or gain >= control["p50_ms"] * 0.15):
            reasons.append("median_gain_below_gate")
        if current["p95_ms"] > control["p95_ms"] + max(100, control["p95_ms"] * 0.05):
            reasons.append("tail_regression")
    preview_loss = []
    memory_comparisons = []
    queue_comparisons = []
    for control in (opening, closing):
        reference = ranges([r for r in control["observed"]["partials"] if translated_preview(r)])
        actual = ranges([r for r in candidate["observed"]["partials"] if translated_preview(r)])
        total = sum(b - a for a, b in reference)
        loss = coverage_missing(reference, actual) / total if total else 0
        preview_loss.append(loss)
        if loss > 0.02:
            reasons.append("preview_source_coverage_loss")
        delivered = merged([(r["sample_start"], r["sample_end"]) for r in candidate["diagnostic_finals"]])
        if coverage_missing(speech_intervals(control, finalized_only=True), delivered) > 0:
            reasons.append("final_source_coverage_loss")
        for field in ("peak_rss_bytes", "peak_metal_bytes"):
            old = control.get("session_lifecycle", {}).get("memory", {}).get(field)
            new = candidate.get("session_lifecycle", {}).get("memory", {}).get(field)
            if field == "peak_metal_bytes" and control.get("session_metadata", {}).get("backend") == "cpu":
                continue
            if any(type(value) is not int or value < 0 for value in (old, new)):
                reasons.append("missing_memory_evidence")
                continue
            allowed = max(old * 0.1, 256 * 1024 * 1024)
            memory_comparisons.append({"field": field, "control": old, "candidate": new, "allowed_increase": allowed})
            if new > old + allowed:
                reasons.append("memory_regression")
        old_pressure = control["observed"]["session_summary"].get("final_queue_pressure", {})
        new_pressure = candidate["observed"]["session_summary"].get("final_queue_pressure", {})
        if any(
            not p
            or p.get("bookkeeping_truncated")
            or p.get("terminal_outstanding") != 0
            or p.get("put_failed")
            or p.get("unmatched_dequeues")
            for p in (old_pressure, new_pressure)
        ):
            reasons.append("missing_or_failed_queue_accounting")
        else:
            if new_pressure["max_pending"] > old_pressure["max_pending"] + 1:
                reasons.append("final_queue_high_water_regression")
            if new_pressure["max_wait_ms"] > old_pressure["max_wait_ms"] + max(100, old_pressure["max_wait_ms"] * 0.05):
                reasons.append("final_queue_wait_regression")
            trends = []
            for pressure in (old_pressure, new_pressure):
                count = min(16, pressure.get("dequeued", 0) // 2)
                first = pressure.get("first_window_wait_ms", [])[:count]
                last = pressure.get("last_window_wait_ms", [])[-count:] if count else []
                if count < 2 or len(first) != count or len(last) != count:
                    reasons.append("insufficient_queue_trend_samples")
                    break
                if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in first + last):
                    reasons.append("invalid_queue_trend_samples")
                    break
                trends.append(sum(last) / count - sum(first) / count)
            if len(trends) == 2:
                queue_comparisons.append(
                    {"field": "disjoint_queue_wait_growth_ms", "control": trends[0], "candidate": trends[1]}
                )
                if trends[1] > max(0, trends[0]) + max(100, old_pressure["max_wait_ms"] * 0.05):
                    reasons.append("final_queue_sustained_growth")
        for field in ("stt_queue_wait_ms", "translation_queue_wait_ms", "generation_lock_wait_ms_a"):
            if field == "generation_lock_wait_ms_a":
                samples, route_counts = [], []
                valid = True
                for item in (control, candidate):
                    values = []
                    counts = {"gemma": 0, "marian": 0, "unknown": 0}
                    for row in item["diagnostic_finals"]:
                        route, value = row.get("final_translation_route"), row.get(field)
                        if route not in ("gemma", "marian"):
                            counts["unknown"] += 1
                            reasons.append("missing_or_invalid_final_translation_route")
                            valid = False
                        elif route == "marian":
                            counts[route] += 1
                            if value is not None:
                                reasons.append("invalid_marian_generation_lock_evidence")
                                valid = False
                        else:
                            counts[route] += 1
                            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                                reasons.append("missing_stage_queue_evidence")
                                valid = False
                            else:
                                values.append(value)
                    samples.append(values)
                    route_counts.append(counts)
                old_values, new_values = samples
                old_tail, new_tail = stats(old_values)["p95"], stats(new_values)["p95"]
                comparable = valid and bool(old_values) and bool(new_values)
                queue_comparisons.append(
                    {
                        "field": field,
                        "control_p95_ms": old_tail,
                        "candidate_p95_ms": new_tail,
                        "control_sample_count": len(old_values),
                        "candidate_sample_count": len(new_values),
                        "control_route_counts": route_counts[0],
                        "candidate_route_counts": route_counts[1],
                        "status": (
                            "invalid_evidence"
                            if not valid
                            else "compared_gemma_calls"
                            if comparable
                            else "not_applicable_no_gemma_in_one_or_both_runs"
                        ),
                    }
                )
                if comparable and new_tail > old_tail + max(100, old_tail * 0.05):
                    reasons.append("stage_queue_tail_regression")
                continue
            old_values = [r.get(field) for r in control["diagnostic_finals"]]
            new_values = [r.get(field) for r in candidate["diagnostic_finals"]]
            if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in old_values + new_values):
                reasons.append("missing_stage_queue_evidence")
                continue
            old_tail, new_tail = stats(old_values)["p95"], stats(new_values)["p95"]
            queue_comparisons.append({"field": field, "control_p95_ms": old_tail, "candidate_p95_ms": new_tail})
            if new_tail > old_tail + max(100, old_tail * 0.05):
                reasons.append("stage_queue_tail_regression")
    return {
        "status": "latency_candidate" if not reasons else "rejected",
        "reasons": sorted(set(reasons)),
        "opening": before,
        "candidate": current,
        "closing": after,
        "preview_missing_fractions": preview_loss,
        "memory_comparisons": memory_comparisons,
        "queue_comparisons": queue_comparisons,
        "anchors_sha256": stable_hash(anchors),
        "buffered_span_diagnostic": buffered_metrics,
        "p95_claim_eligible": all(m["n"] >= 100 for m in metrics),
        "promotion": "Requires >=2/3 repeats plus independent quality, queue/memory and untouched confirmation gates",
    }


def report(args):
    provenance = json.loads((args.input / "provenance.json").read_text())
    runs = [json.loads(p.read_text()) for p in args.input.glob("*.json") if p.name != "provenance.json"]
    runs = [r for r in runs if "experiment" in r]
    indexed = defaultdict(list)
    for item in runs:
        indexed[tuple(item.get(k) for k in ("repeat", "experiment", "size", "clip_id"))].append(item)
    planned = list(schedule(provenance["spec"], provenance["repeats"]))
    expected = {(repeat, config["name"], size, clip["id"]): clip for repeat, config, clip, size in planned}
    inventory_errors = []
    for key in set(expected) | set(indexed):
        if key not in expected:
            inventory_errors.append(f"Unexpected run key: {key}")
        elif len(indexed[key]) != 1:
            inventory_errors.append(f"Expected exactly one run: {key}; found {len(indexed[key])}")
        else:
            item = indexed[key][0]
            repeat, name, size, clip_id = key
            session = f"{provenance['tag']}_{name}_{size}_r{repeat}_{clip_id}_{expected[key]['lang']}"
            if (
                item.get("session_id") != session
                or item.get("source_identity") != provenance["source_identity"]
                or item.get("clip", {}).get("sha256") != expected[key].get("sha256")
            ):
                inventory_errors.append(f"Run identity mismatch: {key}")
    pairs = []
    for repeat, name, size, clip_id in expected:
        if name in {"baseline", "baseline_anchor"}:
            continue
        record = dict(experiment=name, repeat=repeat, size=size, clip_id=clip_id)
        try:
            keys = [(repeat, n, size, clip_id) for n in ("baseline", name, "baseline_anchor")]
            if any(len(indexed[key]) != 1 for key in keys):
                raise ValueError("Missing or duplicate candidate/control record")
            opening, candidate, closing = [indexed[key][0] for key in keys]
            record.update(session_id=candidate["session_id"], **score_pair(opening, candidate, closing))
            if inventory_errors:
                record["status"] = "invalid_inventory"
        except (ValueError, KeyError, IndexError) as exc:
            record.update(status="invalid_comparison", error=str(exc))
        pairs.append(record)
    write_json(
        args.output,
        {
            "schema_version": 1,
            "pairs": pairs,
            "run_count": len(runs),
            "expected_runs": len(expected),
            "inventory_complete": not inventory_errors,
            "inventory_errors": inventory_errors,
            "observed_final_counts": stats([r["observed"]["final_count"] for r in runs]),
            "defaults_changed": False,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    execute = sub.add_parser("run")
    execute.add_argument("--spec", type=Path, required=True)
    execute.add_argument("--output", type=Path, required=True)
    execute.add_argument("--tag", required=True)
    execute.add_argument("--repeats", type=int, default=3)
    execute.add_argument("--timeout", type=float, default=300)
    execute.add_argument("--ws-port", type=int, default=8865)
    execute.add_argument("--http-port", type=int, default=8866)
    summarize = sub.add_parser("report")
    summarize.add_argument("--input", type=Path, required=True)
    summarize.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    (run if args.command == "run" else report)(args)


if __name__ == "__main__":
    main()
