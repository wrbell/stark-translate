"""Serial, reproducible latency experiments with honest coverage/quality gates.

Run from a frozen checkout. One child owns inference at a time; stable display
ports let a visible audience browser reconnect across paired runs. Missing ACKs
remain missing and never become fabricated browser latency measurements.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
from collections import defaultdict
from pathlib import Path

from tools.mac_evaluation import _report_browser_acknowledgments, environment, stats, write_json
from tools.replay_bench import ROOT, run_replay


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
    return tuple(str(row.get(key, "")) for key in ("sample_start", "sample_end", "sample_rate", "endpoint_reason"))


def inspect_run(report: dict, metrics: Path) -> dict:
    session = report["session_id"]
    csv_path = metrics / f"ab_metrics_{session}.csv"
    with csv_path.open(newline="") if csv_path.exists() else open(os.devnull) as source:
        rows = list(csv.DictReader(source))
    partials = jsonl(metrics / f"partials_{session}.jsonl")
    acknowledgments = jsonl(metrics / f"display_metrics_{session}.jsonl")
    first, previous = {}, {}
    gaps = []
    for row in partials:
        # A source-only/non-visible/empty translation is not a translated preview.
        if not str(row.get("text_es") or "").strip():
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
        "within_utterance_gap_samples_ms": gaps,
        "processed_partial_audio_s": sum(
            number(row.get("processed_audio_s", row.get("buffer_s"))) or 0 for row in partials
        ),
        "browser": _report_browser_acknowledgments(rows, acknowledgments, session),
        "hardware_profile": hardware,
        "resource_snapshots": [record["resources"] for record in diagnostic_records if record.get("resources")],
        "session_summary": summary,
    }


def compare_outputs(baseline: dict, candidate: dict) -> dict:
    left = {segment_key(row): row for row in baseline["finals"]}
    right = {segment_key(row): row for row in candidate["finals"]}
    shared = left.keys() & right.keys()
    changed = []
    for key in sorted(shared):
        a, b = left[key], right[key]
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
        "baseline_finals": len(left),
        "candidate_finals": len(right),
        "baseline_duplicate_bounds": len(baseline["finals"]) - len(left),
        "candidate_duplicate_bounds": len(candidate["finals"]) - len(right),
        "baseline_unmatched": len(left.keys() - right.keys()),
        "candidate_unmatched": len(right.keys() - left.keys()),
        "changed_matched_outputs": changed,
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
    source = environment()
    # Capture every tracked source file, including newly shared runtime modules.
    import subprocess

    tracked = (
        subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True).stdout.decode().split("\0")
    )
    source["all_code_sha256"] = {
        name: fingerprint(ROOT / name)
        for name in tracked
        if name.endswith((".py", ".js", ".html", ".toml")) and (ROOT / name).is_file()
    }
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
        if old["environment"]["all_code_sha256"] != source["all_code_sha256"]:
            raise ValueError("Source changed; use a new frozen cohort")
    else:
        write_json(lock, provenance)
    for index, (repeat, config, clip, size) in enumerate(schedule(spec, args.repeats)):
        session = f"{args.tag}_{config['name']}_{size}_r{repeat}_{clip['id']}_{clip['lang']}"
        destination = args.output / f"{session}.json"
        if destination.exists():
            print(f"Recorded already (including failures): {session}", flush=True)
            continue
        wav = (ROOT / clip["path"]).resolve()
        if fingerprint(wav) != clip["sha256"]:
            raise ValueError(f"Audio identity changed: {wav}")
        env = dict(os.environ)
        for key in list(env):
            if key.startswith("STARK_EXPERIMENT_"):
                del env[key]
        env.update(spec.get("baseline_env", {}))
        env.update(config.get("env", {}))
        env["STARK_REPLAY_SPEED"] = "1"
        arguments = [
            "--model-family",
            "gemma4",
            "--gemma4-size",
            size,
            "--no-mts",
            "--stt-backend",
            "parakeet-mlx" if clip["lang"] == "en" else "mlx",
            *spec.get("baseline_arguments", []),
            *config.get("arguments", []),
            "--replay-speed",
            "1",
        ]
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
        result.update(
            experiment=config["name"],
            repeat=repeat,
            size=size,
            clip_id=clip["id"],
            language=clip["lang"],
            experiment_env={
                key: value
                for key, value in env.items()
                if key.startswith(("STARK_EXPERIMENT_", "STARK_VAD_", "STARK_TRANSLATE_"))
            },
            observed=inspect_run(result, ROOT / "metrics"),
        )
        if result.get("session_lifecycle", {}).get("status") != "completed":
            result["error"] = result.get("error", "No confirmed completed session lifecycle")
        write_json(destination, result)
        print(f"Finished {session}: {result.get('error', 'OK')}", flush=True)


def report(args):
    runs = [
        json.loads(path.read_text())
        for path in sorted(args.input.glob("*.json"))
        if path.name not in {"provenance.json", "comparison.json"}
    ]
    runs = [run for run in runs if "observed" in run]
    baselines = {(run["repeat"], run["size"], run["clip_id"]): run for run in runs if run["experiment"] == "baseline"}
    groups = defaultdict(list)
    for run in runs:
        groups[(run["experiment"], run["size"])].append(run)
    result = {
        "schema_version": 1,
        "metric_definitions": {
            "final": "schema 2 estimated last speech capture frame to final payload readiness, separated by endpoint and timing source",
            "first_preview": "first nonempty translated preview readiness minus first captured speech sample of each utterance",
            "update_gap": "consecutive translated preview readiness times within the same utterance only",
            "browser": "speech-end-to-ACK is an upper bound including return network time; only visible real-time ACKs qualify",
            "coverage": "union of final source sample intervals; not VAD recall or reference WER",
        },
        "groups": [],
    }
    markdown = [
        "# Overnight latency experiment results",
        "",
        "All configurations remain opt-in. Original predictions and changed examples are retained in JSON. Natural bilingual reference quality and human meaning review remain separate pending gates.",
        "",
        "| Experiment | Model | Runs / failures | Silence final p50 / p95 ms | First preview p50 / p95 ms | Missing source seconds | Changed matched finals |",
        "|---|---|---|---|---|---|---|",
    ]
    for (name, size), observations in sorted(groups.items()):
        endpoints = defaultdict(list)
        first, gaps, comparisons = [], [], []
        for item in observations:
            observed = item["observed"]
            for key, values in observed["endpoint_samples_ms"].items():
                endpoints[key].extend(values)
            first.extend(observed["first_preview_samples_ms"])
            gaps.extend(observed["within_utterance_gap_samples_ms"])
            baseline = baselines.get((item["repeat"], size, item["clip_id"]))
            if baseline:
                comparisons.append(
                    {"session_id": item["session_id"], **compare_outputs(baseline["observed"], observed)}
                )
        failures = sum(bool(item.get("error")) for item in observations)
        entry = {
            "experiment": name,
            "size": size,
            "runs": len(observations),
            "failures": failures,
            "endpoints": {key: stats(values) for key, values in endpoints.items()},
            "first_preview_ms": stats(first),
            "update_gap_ms": stats(gaps),
            "output_comparisons": comparisons,
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
            f"| {name} | {size} | {len(observations)} / {failures} | {pair(silence)} | {pair(entry['first_preview_ms'])} | {missing:.3f} | {changes} |"
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
