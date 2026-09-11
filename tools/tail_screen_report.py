"""Offline tail-screen gates from recorded artifacts; stdlib only, no inference.

Pools never cross clip identities. PASS means all G1..G7 checks on every declared
clip; sample-size eligibility is reported separately and is not a gate. A reduced
one-clip fixture only assesses that declared clip, not a two-clip deployment claim.
Missing gate observations fail closed. Route gates use explicit routes only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import statistics
import sys
from itertools import zip_longest
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.silence_final_stages import STAGES, _delta, _escape, _format, _number, _stats

G1_RATIO = 0.85
G1_REDUCTION_MS = 300
REGRESSION_RATIO = 0.05
G2_ALLOWANCE_MS = 100
G3_ALLOWANCE_MS = 50
IDENTITY_SHARE = 0.95
PREVIEW_RATIO = 0.98
MEMORY_RATIO = 0.10
MEMORY_ALLOWANCE_BYTES = 256 * 1024 * 1024
CHUNK_RATIO = 0.10
MIN_GEMMA_SILENCE_PER_CLIP = 50
MIN_GEMMA_SILENCE_TOTAL = 100
WORST_DECILE_SHARE = 0.10
SLOW_STAGE_MS = 800
ROUTES = ("gemma", "marian", "unknown")
STAGE_NAMES = ("stt_call", "translation_call", "translation_lock_wait")
SCALAR_NAMES = ("generation_lock_wait_ms_a", "decode_ms_a", "gen_tokens_a")
FILE_KEYS = ("replay_json", "diagnostics_jsonl", "csv", "lifecycle_json", "session_metadata_json")
THRESHOLDS = {
    name: value
    for name, value in globals().copy().items()
    if name.startswith(("G1_", "G2_", "G3_", "MIN_"))
    or name
    in {
        "REGRESSION_RATIO",
        "IDENTITY_SHARE",
        "PREVIEW_RATIO",
        "MEMORY_RATIO",
        "MEMORY_ALLOWANCE_BYTES",
        "CHUNK_RATIO",
        "WORST_DECILE_SHARE",
        "SLOW_STAGE_MS",
    }
}


def _route(row: dict) -> str:
    route = row.get("final_translation_route")
    return route if route in ROUTES[:2] else "unknown"


def _distribution(rows: list[dict]) -> dict:
    return _stats([_number(row.get("speech_end_to_final_ms")) for row in rows])


def _cohorts(rows: list[dict]) -> dict:
    return {
        endpoint: {
            "all": _distribution(selected),
            **{route: _distribution([r for r in selected if _route(r) == route]) for route in ROUTES},
        }
        for endpoint in sorted({str(r.get("endpoint_reason") or "unknown") for r in rows} | {"silence"})
        for selected in [[r for r in rows if str(r.get("endpoint_reason") or "unknown") == endpoint]]
    }


def _secondary(rows: list[dict]) -> dict:
    stages = {}
    for route in ROUTES:
        selected = [r for r in rows if _route(r) == route]
        stages[route] = {
            **{name: _stats([_delta(r, *STAGES[name]) for r in selected]) for name in STAGE_NAMES},
            **{name: _stats([_number(r.get(name)) for r in selected]) for name in SCALAR_NAMES},
        }
    silence = [r for r in rows if r.get("endpoint_reason") == "silence"]
    worst = sorted(silence, key=lambda r: r["speech_end_to_final_ms"], reverse=True)[
        : math.ceil(len(silence) * WORST_DECILE_SHARE)
    ]
    stt = [_delta(r, *STAGES["stt_call"]) for r in worst]
    translation = [_delta(r, *STAGES["translation_call"]) for r in worst]
    slow_stt = [v is not None and v > SLOW_STAGE_MS for v in stt]
    slow_translation = [v is not None and v > SLOW_STAGE_MS for v in translation]
    return {
        "stages_by_route": stages,
        "worst_decile": {
            "n": len(worst),
            "stt_call_gt_800_share": sum(slow_stt) / len(worst) if worst else None,
            "translation_call_gt_800_share": sum(slow_translation) / len(worst) if worst else None,
            "both_gt_800_share": (
                sum(a and b for a, b in zip(slow_stt, slow_translation, strict=True)) / len(worst) if worst else None
            ),
            "stt_missing": stt.count(None),
            "translation_missing": translation.count(None),
        },
        "draft_accept_a": _stats([_number(r.get("draft_accept_a")) for r in rows]),
    }


def _counters(metadata: dict, replay: dict, diagnostics: list[dict]) -> tuple[dict, str]:
    # The supplied sample has counters only in the diagnostics session_summary.
    sources = [
        (metadata, "session_metadata_json"),
        (replay.get("session_metadata", {}), "replay_json.session_metadata"),
    ]
    sources.extend(
        (r, "diagnostics_jsonl.session_summary") for r in reversed(diagnostics) if r.get("event") == "session_summary"
    )
    for data, source in sources:
        counters = data.get("latency_experiment_counters")
        if isinstance(counters, dict):
            return {
                k: v
                for k, v in counters.items()
                if k.startswith("partial_suppressed_") or k == "final_stt_waited_for_translation"
            }, source
    return {}, "missing"


def _identity(candidate: dict, control: dict) -> dict:
    left, right = candidate["csv_rows"], control["csv_rows"]
    aligned = min(len(left), len(right))
    shares = {
        field: sum(a[field].encode("utf-8") == b[field].encode("utf-8") for a, b in zip(left, right)) / aligned
        if aligned
        else None
        for field in ("spanish_a", "english")
    }
    differences = []
    for index, (a, b) in enumerate(zip_longest(left, right)):
        fields = [f for f in ("spanish_a", "english") if a is None or b is None or a[f] != b[f]]
        if fields:
            differences.append(
                {
                    "row_index": index,
                    "fields": fields,
                    "candidate": {k: a[k] for k in ("chunk_id", "spanish_a", "english")} if a else None,
                    "control": {k: b[k] for k in ("chunk_id", "spanish_a", "english")} if b else None,
                }
            )
    return {
        "candidate_tag": candidate["tag"],
        "control_tag": control["tag"],
        "aligned_length": aligned,
        "candidate_rows": len(left),
        "control_rows": len(right),
        "chunk_count_difference": len(left) - len(right),
        "identical_share": shares["spanish_a"],
        "english_identical_share": shares["english"],
        "differences": differences,
        "pass": shares["spanish_a"] is not None and shares["spanish_a"] >= IDENTITY_SHARE,
    }


def _bound(value: object, limit: float | None) -> bool:
    number = _number(value)
    return number is not None and limit is not None and number <= limit


def _mean(runs: list[dict], key: str) -> float | None:
    values = [_number(r["replay"].get(key)) for r in runs]
    return statistics.mean(values) if values and None not in values else None


def _gates(candidates: list[dict], controls: list[dict], arm: dict, rows: list[dict], ctl_rows: list[dict]) -> dict:
    def measure(data, endpoints, route=None, percentile="p95"):
        return _distribution(
            [r for r in data if r.get("endpoint_reason") in endpoints and (route is None or _route(r) == route)]
        )[percentile]

    gates = {}
    for gate, endpoints, route, percentile, allowance in (
        ("G1", ("silence",), "gemma", "p95", None),
        ("G2_silence", ("silence",), None, "p95", G2_ALLOWANCE_MS),
        ("G2_cuts", ("smart_cut", "hard_cut"), None, "p95", G2_ALLOWANCE_MS),
        ("G3", ("silence",), None, "p50", G3_ALLOWANCE_MS),
    ):
        value, baseline = measure(rows, endpoints, route, percentile), measure(ctl_rows, endpoints, route, percentile)
        limit = (
            None
            if baseline is None
            else (
                max(G1_RATIO * baseline, baseline - G1_REDUCTION_MS)
                if allowance is None
                else baseline + max(REGRESSION_RATIO * baseline, allowance)
            )
        )
        gates[gate] = {"pass": _bound(value, limit), "candidate": value, "control": baseline, "limit": limit}
    gates["G2"] = {
        "pass": gates["G2_silence"]["pass"] and gates["G2_cuts"]["pass"],
        "silence": gates.pop("G2_silence"),
        "cuts": gates.pop("G2_cuts"),
    }
    identities = []
    for run in candidates:
        matches = [c for c in controls if c["repeat"] == run["repeat"]]
        identities.extend(_identity(run, c) for c in matches)
        if not matches:
            identities.append({"candidate_tag": run["tag"], "pass": False, "reason": "missing same-repeat control"})
    gates["G4"] = {"pass": bool(identities) and all(i["pass"] for i in identities), "comparisons": identities}
    preview_mean = _mean(controls, "partial_count")
    preview_limit = preview_mean * PREVIEW_RATIO if preview_mean is not None else None
    preview_checks = [
        {
            "tag": r["tag"],
            "partial_count": r["replay"].get("partial_count"),
            "pass": (v := _number(r["replay"].get("partial_count"))) is not None
            and preview_limit is not None
            and v >= preview_limit,
        }
        for r in candidates
    ]
    gates["G5"] = {
        "pass": bool(preview_checks) and all(c["pass"] for c in preview_checks),
        "control_mean": preview_mean,
        "minimum": preview_limit,
        "runs": preview_checks,
    }
    memory_limits = {}
    for key in ("peak_rss_bytes", "peak_metal_bytes"):
        values = [_number(r["memory"].get(key)) for r in controls]
        maximum = max(values) if values and None not in values else None
        memory_limits[key] = (
            maximum + max(MEMORY_RATIO * maximum, MEMORY_ALLOWANCE_BYTES) if maximum is not None else None
        )
    memory_checks = []
    for run in candidates:
        if arm["memory_rule"] == "absolute":
            limits = {"peak_metal_bytes": arm["metal_budget_bytes"]}
            pageouts = run.get("pageouts")
            pageouts_ok = pageouts is None or (
                _number(pageouts.get("before")) is not None and pageouts.get("after") == pageouts["before"]
            )
        else:
            limits, pageouts_ok = memory_limits, True
        checks = {key: _bound(run["memory"].get(key), limit) for key, limit in limits.items()}
        memory_checks.append(
            {
                "tag": run["tag"],
                "memory": run["memory"],
                "limits": limits,
                "checks": checks,
                "pageouts": run.get("pageouts"),
                "pageouts_pass": pageouts_ok,
                "pass": all(checks.values()) and pageouts_ok,
            }
        )
    gates["G6"] = {"pass": bool(memory_checks) and all(c["pass"] for c in memory_checks), "runs": memory_checks}
    chunk_mean = _mean(controls, "chunk_count")
    health_checks = []
    for run in candidates:
        replay = run["replay"]
        count = _number(replay.get("chunk_count"))
        status = run["lifecycle"].get("status")
        checks = {
            "returncode": type(replay.get("returncode")) is int and replay["returncode"] == 0,
            "timed_out": replay.get("timed_out") is False,
            "lifecycle_status_recorded": isinstance(status, str) and bool(status.strip()),
            "chunk_count": count is not None
            and chunk_mean is not None
            and abs(count - chunk_mean) <= CHUNK_RATIO * chunk_mean,
        }
        health_checks.append(
            {
                "tag": run["tag"],
                "returncode": replay.get("returncode"),
                "timed_out": replay.get("timed_out"),
                "lifecycle_status": status,
                "chunk_count": count,
                "checks": checks,
                "pass": all(checks.values()),
            }
        )
    gates["G7"] = {
        "pass": bool(health_checks) and all(c["pass"] for c in health_checks),
        "control_chunk_count_mean": chunk_mean,
        "runs": health_checks,
    }
    return {name: gates[name] for name in sorted(gates)}


def analyze(runs_path: Path) -> dict:
    """Hash consumed artifacts and evaluate all declared clips without writing files."""
    inputs = {}

    def read(path):
        path = Path(path)
        path = path if path.is_absolute() else runs_path.resolve().parent / path
        raw = path.read_bytes()
        inputs[str(path.resolve())] = {"sha256": hashlib.sha256(raw).hexdigest(), "size_bytes": len(raw)}
        return raw

    spec = json.loads(read(runs_path.resolve()))
    if not spec.get("clips") or not spec.get("arms"):
        raise ValueError("Screen must declare clips and arms")
    for clip in spec["clips"].values():
        path = Path(clip["path"])
        path = path if path.is_absolute() else runs_path.resolve().parent / path
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != clip["sha256"]:
            raise ValueError(f"Clip SHA256 mismatch: {path}")
        inputs[str(path.resolve())] = {"sha256": digest, "size_bytes": path.stat().st_size}
    for arm in spec["arms"].values():
        if arm.get("kind") not in ("control", "candidate"):
            raise ValueError("Arm kind must be control or candidate")
        if arm["kind"] == "candidate":
            if arm.get("memory_rule") not in ("absolute", "relative"):
                raise ValueError("Candidate memory_rule must be absolute or relative")
            if arm["memory_rule"] == "absolute" and (
                (budget := _number(arm.get("metal_budget_bytes"))) is None or budget < 0
            ):
                raise ValueError("Absolute memory rule requires nonnegative metal_budget_bytes")
    runs = []
    seen = set()
    for entry in spec["runs"]:
        if entry["clip"] not in spec["clips"] or entry["arm"] not in spec["arms"]:
            raise ValueError("Run references undeclared clip or arm")
        key = (entry["clip"], entry["arm"], entry["repeat"])
        if key in seen or any(r["tag"] == entry["tag"] for r in runs):
            raise ValueError("Duplicate run tag or clip/arm/repeat")
        seen.add(key)
        files = {key: read(entry[key]) for key in FILE_KEYS}
        replay = json.loads(files["replay_json"])
        lifecycle = json.loads(files["lifecycle_json"])
        metadata = json.loads(files["session_metadata_json"])
        diagnostics = [json.loads(line) for line in files["diagnostics_jsonl"].splitlines() if line.strip()]
        if any(not isinstance(r, dict) for r in diagnostics):
            raise ValueError("Diagnostics rows must be objects")
        rows = [
            r
            for r in diagnostics
            if "event" not in r
            and type(r.get("timing_schema_version")) is int
            and r["timing_schema_version"] == 2
            and _number(r.get("speech_end_to_final_ms")) is not None
        ]
        reader = csv.DictReader(io.StringIO(files["csv"].decode("utf-8-sig"), newline=""))
        if not {"chunk_id", "english", "spanish_a"} <= set(reader.fieldnames or []):
            raise ValueError("CSV requires chunk_id, english, spanish_a")
        csv_rows = list(reader)
        if any(any(r[k] is None for k in ("chunk_id", "english", "spanish_a")) for r in csv_rows):
            raise ValueError("CSV has incomplete final rows")
        counters, source = _counters(metadata, replay, diagnostics)
        runs.append(
            {
                **entry,
                "rows": rows,
                "csv_rows": csv_rows,
                "replay": replay,
                "lifecycle": lifecycle,
                "observations": {
                    **{key: replay.get(key) for key in ("chunk_count", "partial_count", "returncode", "timed_out")},
                    "lifecycle_status": lifecycle.get("status"),
                    "eligible_finals": len(rows),
                    "excluded_diagnostics_rows": len(diagnostics) - len(rows),
                },
                "memory": replay.get("session_lifecycle", {}).get("memory", {}),
                "memory_source": "replay_json.session_lifecycle.memory",
                "cohorts": _cohorts(rows),
                "unknown_route_count": sum(_route(r) == "unknown" for r in rows),
                "experiment_counters": counters,
                "experiment_counters_source": source,
            }
        )
    clips = {}
    for clip in spec["clips"]:
        controls = [r for r in runs if r["clip"] == clip and spec["arms"][r["arm"]]["kind"] == "control"]
        ctl_rows = [row for r in controls for row in r["rows"]]
        arms = {}
        for name, arm in spec["arms"].items():
            selected = [r for r in runs if r["clip"] == clip and r["arm"] == name]
            rows = [row for r in selected for row in r["rows"]]
            arms[name] = {
                "cohorts": _cohorts(rows),
                "n_by_route": {route: sum(_route(r) == route for r in rows) for route in ROUTES},
                "secondary": _secondary(rows),
            }
            if arm["kind"] == "candidate":
                arms[name]["gates"] = _gates(selected, controls, arm, rows, ctl_rows)
        clips[clip] = {
            "arms": arms,
            "control_pool": {"tags": [r["tag"] for r in controls], "cohorts": _cohorts(ctl_rows)},
        }
    outcomes = {}
    for name, arm in spec["arms"].items():
        counts = {clip: data["arms"][name]["cohorts"]["silence"]["gemma"]["n"] for clip, data in clips.items()}
        eligible = (
            all(n >= MIN_GEMMA_SILENCE_PER_CLIP for n in counts.values())
            and sum(counts.values()) >= MIN_GEMMA_SILENCE_TOTAL
        )
        outcome = {
            "gemma_silence_n_by_clip": counts,
            "gemma_silence_n_total": sum(counts.values()),
            "p95_claim_eligible": eligible,
            "p95_claim_reason": None if eligible else "screen without p95 claim",
        }
        if arm["kind"] == "candidate":
            failures = [
                f"{clip}:{gate}"
                for clip, data in clips.items()
                for gate, result in data["arms"][name]["gates"].items()
                if not result["pass"]
            ]
            outcome.update(
                outcome="REJECTED" if failures else "PASS",
                failing_gates=failures,
                gates_on_all_clips={
                    gate: all(data["arms"][name]["gates"][gate]["pass"] for data in clips.values())
                    for gate in (f"G{i}" for i in range(1, 8))
                },
            )
        outcomes[name] = outcome
    return {
        "screen": spec["screen"],
        "thresholds": THRESHOLDS,
        "inputs": inputs,
        "specification": spec,
        "percentile_method": "nearest-rank: ceil(p / 100 * n) - 1",
        "clip_count": len(clips),
        "scope": "Gates cover every declared clip; p95 claim eligibility is separate.",
        "runs": [{k: v for k, v in r.items() if k not in ("rows", "csv_rows", "replay", "lifecycle")} for r in runs],
        "clips": clips,
        "outcomes": outcomes,
    }


def render_markdown(report: dict) -> str:
    lines = [
        f"# {_escape(report['screen'])}",
        "",
        report["scope"],
        "",
        "Nearest-rank percentiles; latency in ms. Unknown routes are counted in all-route gates only.",
        "",
    ]
    for clip, data in report["clips"].items():
        lines.extend(
            [
                f"## Clip {_escape(clip)}",
                "",
                "n by route covers all eligible finals; p50/p95 are all-route silence finals.",
                "",
                "| Arm | n Gemma / Marian / unknown | Gemma silence n | Silence p50 | Silence p95 | Gemma silence p95 | G1 | G2 | G3 | G4 | G5 | G6 | G7 |",
                "| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for name, arm in data["arms"].items():
            silence = arm["cohorts"]["silence"]
            gates = arm.get("gates", {})
            cells = [
                _escape(name),
                " / ".join(str(arm["n_by_route"][route]) for route in ROUTES),
                str(silence["gemma"]["n"]),
                _format(silence["all"]["p50"]),
                _format(silence["all"]["p95"]),
                _format(silence["gemma"]["p95"]),
            ]
            cells.extend(
                ("PASS" if gates[g]["pass"] else "FAIL") if g in gates else "—" for g in (f"G{i}" for i in range(1, 8))
            )
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
        for name, arm in data["arms"].items():
            for comparison in arm.get("gates", {}).get("G4", {}).get("comparisons", []):
                if "reason" in comparison:
                    lines.extend([f"Identity {_escape(comparison['candidate_tag'])}: {comparison['reason']}", ""])
                    continue
                lines.extend(
                    [
                        f"### Identity: {_escape(comparison['candidate_tag'])} vs {_escape(comparison['control_tag'])}",
                        "",
                        f"Translation share: {comparison['identical_share']}; English share: {comparison['english_identical_share']}; "
                        f"aligned rows: {comparison['aligned_length']}; chunk-count difference: {comparison['chunk_count_difference']}.",
                        "",
                    ]
                )
                for diff in comparison["differences"]:
                    # JSON preserves exact text escapes while keeping multiline strings reviewable.
                    lines.extend([f"Row {diff['row_index']}: " + _escape(json.dumps(diff, ensure_ascii=False)), ""])
    lines.extend(["## Experiment counters", "", "| Run | Source | Counters |", "| --- | --- | --- |"])
    for run in report["runs"]:
        lines.append(
            f"| {_escape(run['tag'])} | {run['experiment_counters_source']} | {_escape(json.dumps(run['experiment_counters'], sort_keys=True))} |"
        )
    lines.extend(["", "## Outcomes", ""])
    for name, outcome in report["outcomes"].items():
        if "outcome" in outcome:
            lines.extend(
                [
                    f"{_escape(name)}: {outcome['outcome']}",
                    "",
                    "Failing gates: " + (", ".join(outcome["failing_gates"]) or "none") + ".",
                    "",
                ]
            )
        lines.extend(
            [
                f"{_escape(name)} p95_claim_eligible: {str(outcome['p95_claim_eligible']).lower()}"
                + (f" — {outcome['p95_claim_reason']}" if outcome["p95_claim_reason"] else "")
                + ".",
                "",
            ]
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = analyze(args.runs)
        outputs = [args.output, args.markdown]
        for index, output in enumerate(outputs):
            for other in [Path(p) for p in report["inputs"]] + outputs[:index]:
                if output.resolve() == other.resolve() or (
                    output.exists() and other.exists() and output.samefile(other)
                ):
                    raise ValueError("Output paths must be distinct from each other and from inputs")
        output_json = json.dumps(report, indent=2, allow_nan=False) + "\n"
        markdown = render_markdown(report)
        args.output.write_text(output_json, encoding="utf-8")
        args.markdown.write_text(markdown, encoding="utf-8")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
