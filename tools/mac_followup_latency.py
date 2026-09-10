"""Frozen serial server-latency screens; browser/device certification stays separate.

Opening-control source spans are frozen before candidate scoring. Every candidate
must beat both opening and closing controls, without hiding lost source audio.
This engineering screen never promotes a model or substitutes for bilingual review.
"""

from __future__ import annotations

import argparse
import json
import os
import re
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
        result.update(
            experiment=config["name"],
            repeat=repeat,
            size=size,
            clip_id=clip["id"],
            source_identity=identity,
            observed=observed,
            diagnostic_finals=[r for r in records if r.get("timing_schema_version") == 2],
            completion_errors=errors,
            requested_settings=expected,
            measurement_scope="server real-time replay; no physical visibility certification",
        )
        write_json(destination, result, exclusive=True)
        print(f"Finished {session}: {errors or result.get('error') or 'OK'}", flush=True)


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
    for control in (opening, closing):
        reference = ranges([r for r in control["observed"]["partials"] if translated_preview(r)])
        actual = ranges([r for r in candidate["observed"]["partials"] if translated_preview(r)])
        total = sum(b - a for a, b in reference)
        loss = coverage_missing(reference, actual) / total if total else 0
        preview_loss.append(loss)
        if loss > 0.02:
            reasons.append("preview_source_coverage_loss")
        if (
            coverage_missing(control["observed"]["coverage_intervals_s"], candidate["observed"]["coverage_intervals_s"])
            > 0
        ):
            reasons.append("final_source_coverage_loss")
    return {
        "status": "latency_candidate" if not reasons else "rejected",
        "reasons": sorted(set(reasons)),
        "opening": before,
        "candidate": current,
        "closing": after,
        "preview_missing_fractions": preview_loss,
        "promotion": "Requires >=2/3 repeats plus independent quality, queue/memory and untouched confirmation gates",
    }


def report(args):
    runs = [json.loads(p.read_text()) for p in args.input.glob("*.json") if p.name != "provenance.json"]
    runs = [r for r in runs if "experiment" in r]
    pairs = []
    for candidate in runs:
        if candidate["experiment"] in {"baseline", "baseline_anchor"}:
            continue
        controls = {
            r["experiment"]: r
            for r in runs
            if all(r[k] == candidate[k] for k in ("repeat", "size", "clip_id"))
            and r["experiment"] in {"baseline", "baseline_anchor"}
        }
        record = {k: candidate[k] for k in ("session_id", "experiment", "repeat", "size", "clip_id")}
        try:
            record.update(score_pair(controls["baseline"], candidate, controls["baseline_anchor"]))
        except (ValueError, KeyError, IndexError) as exc:
            record.update(status="invalid_comparison", error=str(exc))
        pairs.append(record)
    write_json(
        args.output,
        {
            "schema_version": 1,
            "pairs": pairs,
            "run_count": len(runs),
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
