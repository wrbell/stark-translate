"""Tiny synthetic recorded screens; no real replay metrics or inference."""

import csv
import hashlib
import json

import pytest

from tools.tail_screen_report import MEMORY_ALLOWANCE_BYTES, analyze, main, render_markdown


def _json(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")
    return str(path)


def screen(tmp_path, passing=False, memory_rule="absolute"):
    clip = tmp_path / "clip.wav"
    clip.write_bytes(b"synthetic clip identity; never decoded")
    spec = {
        "screen": "synthetic_tail_screen",
        "clips": {"A": {"path": str(clip), "sha256": hashlib.sha256(clip.read_bytes()).hexdigest()}},
        "arms": {
            "ctl": {"kind": "control"},
            "candidate": {"kind": "candidate", "memory_rule": memory_rule, "metal_budget_bytes": 500},
        },
        "runs": [],
    }
    for arm in spec["arms"]:
        for repeat in range(2):
            tag = f"{arm}_{repeat}"
            candidate = arm == "candidate"
            bad = candidate and not passing
            latencies = (
                [1600, 2100, 2400, 2200, 900, 1000]
                if bad
                else ([800, 1600, 900, 1000, 500, 600] if candidate else [1000, 2000, 900, 1200, 500, 600])
            )
            rows = []
            for index, (latency, route, endpoint) in enumerate(
                zip(
                    latencies,
                    ["gemma", "gemma", "marian", "unrecorded", "gemma", "marian"],
                    ["silence"] * 4 + ["smart_cut", "hard_cut"],
                    strict=True,
                )
            ):
                rows.append(
                    {
                        "timing_schema_version": 2,
                        "speech_end_to_final_ms": latency,
                        "final_translation_route": route,
                        "endpoint_reason": endpoint,
                        "chunk_id": index,
                        "utterance_id": index,
                        "english": f"English {index}",
                        "spanish_a": f"Español {index}",
                        "timing_stages_ms": {
                            "stt_started": 100,
                            "stt_finished": 1000,
                            "translation_requested": 1000,
                            "translation_lock_acquired": 1020,
                            "translation_started": 1020,
                            "translation_finished": 2020,
                        },
                        "generation_lock_wait_ms_a": 10 + index,
                        "decode_ms_a": 100 + index,
                        "gen_tokens_a": 20 + index,
                        "draft_accept_a": 0.5 if index % 2 else None,
                    }
                )
            if bad and repeat == 0:
                rows[1]["spanish_a"] = "Distinto"
                rows[1]["english"] = "Different"
            memory = {"peak_rss_bytes": 1000, "peak_metal_bytes": 600 if bad else 400}
            if bad and memory_rule == "relative":
                memory = dict.fromkeys(memory, MEMORY_ALLOWANCE_BYTES + 2000)
            lifecycle = {"status": "" if bad else "completed", "memory": memory}
            counters = {
                "latency_experiment_counters": {
                    "partial_suppressed_in_flight": 2,
                    "final_stt_waited_for_translation": 3 if candidate else 0,
                    "warmup_requested": 99,
                }
            }
            replay = {
                "chunk_count": 8 if bad else 6,
                "partial_count": 95 if bad else (110 if candidate else 100 + 20 * repeat),
                "returncode": 1 if bad else 0,
                "timed_out": bad,
                "session_lifecycle": lifecycle,
                "session_metadata": counters if repeat == 1 else {},
            }
            metadata = counters if repeat == 0 else {}
            diagnostics = tmp_path / f"{tag}.jsonl"
            diagnostics.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
            csv_path = tmp_path / f"{tag}.csv"
            with csv_path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["chunk_id", "english", "spanish_a"], extrasaction="ignore")
                writer.writeheader()
                writer.writerows(rows)
            spec["runs"].append(
                {
                    "tag": tag,
                    "clip": "A",
                    "arm": arm,
                    "repeat": repeat,
                    "order": len(spec["runs"]),
                    "replay_json": _json(tmp_path / f"{tag}_replay.json", replay),
                    "diagnostics_jsonl": str(diagnostics),
                    "csv": str(csv_path),
                    "lifecycle_json": _json(tmp_path / f"{tag}_lifecycle.json", lifecycle),
                    "session_metadata_json": _json(tmp_path / f"{tag}_metadata.json", metadata),
                    "pageouts": {"before": 123, "after": 124 if bad else 123},
                }
            )
    path = tmp_path / "runs.json"
    _json(path, spec)
    return path, spec


def test_optional_first_visible_metrics_report_nearest_rank_without_changing_gates(tmp_path):
    path, spec = screen(tmp_path)
    original = analyze(path)
    assert all("first_visible" not in run for run in original["runs"])
    assert "First visible" not in render_markdown(original)
    for repeat, latencies in enumerate(([1100, 1200], [1300, 1400])):
        entry = spec["runs"][repeat]
        ack = {
            "event": "caption_rendered",
            "session_id": entry["tag"],
            "client_id": "audience",
            "event_id": f"{entry['tag']}:stream:1",
            "chunk_id": 1,
            "stage": "first_stream",
            "visible": True,
            "speech_end_to_ack_upper_bound_ms": latencies[0],
        }
        metrics = tmp_path / f"{entry['tag']}_display.jsonl"
        metrics.write_text(
            "\n".join(
                json.dumps(row)
                for row in [
                    ack,
                    ack,
                    {
                        **ack,
                        "chunk_id": 2,
                        "event_id": f"{entry['tag']}:stream:2",
                        "speech_end_to_ack_upper_bound_ms": latencies[1],
                    },
                    {**ack, "chunk_id": 3, "stage": "complete", "speech_end_to_ack_upper_bound_ms": 9000},
                    {**ack, "chunk_id": 4, "visible": False},
                    {**ack, "chunk_id": 5, "speech_end_to_ack_upper_bound_ms": None},
                ]
            )
            + "\n\n"
        )
        entry["display_metrics_jsonl"] = metrics.name  # relative paths share the diagnostics reader base
    _json(path, spec)
    report = analyze(path)
    assert report["runs"][0]["first_visible"] == {"n": 2, "missing": 1, "mean": 1150, "p50": 1100, "p95": 1200}
    arms = report["clips"]["A"]["arms"]
    assert arms["ctl"]["first_visible"] == {"n": 4, "missing": 2, "mean": 1250, "p50": 1200, "p95": 1400}
    assert "first_visible" not in arms["candidate"]
    assert report["outcomes"] == original["outcomes"]
    assert arms["candidate"]["gates"] == original["clips"]["A"]["arms"]["candidate"]["gates"]
    assert len(report["inputs"]) == len(original["inputs"]) + 2
    markdown = render_markdown(report)
    assert "reported, not gated" in markdown
    assert "| ctl | 4 | 1200.0 | 1400.0 |" in markdown
    assert "| ctl_0 | 2 | 1100.0 | 1200.0 |" in markdown


def test_optional_display_metrics_rejects_non_object_rows(tmp_path):
    path, spec = screen(tmp_path)
    metrics = tmp_path / "display.jsonl"
    metrics.write_text("[]\n")
    spec["runs"][0]["display_metrics_jsonl"] = str(metrics)
    _json(path, spec)
    with pytest.raises(ValueError, match="Display metrics rows must be objects"):
        analyze(path)


def test_rejected_screen_percentiles_gates_identity_and_secondary(tmp_path):
    path, spec = screen(tmp_path)
    report = analyze(path)
    arms = report["clips"]["A"]["arms"]
    assert arms["ctl"]["cohorts"]["silence"]["all"] == {"n": 8, "missing": 0, "mean": 1275, "p50": 1000, "p95": 2000}
    candidate = arms["candidate"]
    assert candidate["cohorts"]["silence"]["all"]["p50"] == 2100
    assert candidate["cohorts"]["silence"]["gemma"]["p95"] == 2100
    assert candidate["n_by_route"] == {"gemma": 6, "marian": 4, "unknown": 2}
    gates = candidate["gates"]
    assert {key: g["pass"] for key, g in gates.items()} == {
        **dict.fromkeys([f"G{i}" for i in range(1, 8)], False),
        "G4": True,
    }
    assert gates["G1"]["limit"] == 1700
    assert gates["G2"]["silence"]["limit"] == 2100
    assert gates["G2"]["cuts"]["limit"] == 700
    assert gates["G3"]["limit"] == 1050
    assert gates["G5"]["minimum"] == pytest.approx(107.8)
    identity = gates["G4"]["comparisons"][0]
    assert identity["identical_share"] == 1
    assert identity["english_identical_share"] == 5 / 6
    assert identity["aligned_length"] == 5
    assert identity["chunk_count_difference"] == 0
    assert identity["differences"] == [
        {
            "row_index": 1,
            "fields": ["spanish_a", "english"],
            "candidate": {"chunk_id": "1", "spanish_a": "Distinto", "english": "Different"},
            "control": None,
        },
        {
            "row_index": 1,
            "fields": ["spanish_a", "english"],
            "candidate": None,
            "control": {"chunk_id": "1", "spanish_a": "Español 1", "english": "English 1"},
        },
    ]
    assert gates["G4"]["comparisons"][1]["pass"]
    assert not gates["G6"]["runs"][0]["pageouts_pass"]
    assert not any(gates["G7"]["runs"][0]["checks"].values())
    secondary = candidate["secondary"]
    assert secondary["worst_decile"] == {
        "n": 1,
        "stt_call_gt_800_share": 1,
        "translation_call_gt_800_share": 1,
        "both_gt_800_share": 1,
        "stt_missing": 0,
        "translation_missing": 0,
    }
    assert secondary["draft_accept_a"]["mean"] == 0.5
    assert secondary["stages_by_route"]["gemma"]["stt_call"]["p95"] == 900
    assert secondary["stages_by_route"]["gemma"]["translation_lock_wait"]["p95"] == 20
    assert secondary["stages_by_route"]["gemma"]["gen_tokens_a"]["p95"] == 24
    outcome = report["outcomes"]["candidate"]
    assert outcome["outcome"] == "REJECTED"
    assert outcome["failing_gates"] == [f"A:G{i}" for i in range(1, 8) if i != 4]
    assert outcome["p95_claim_eligible"] is False
    assert outcome["p95_claim_reason"] == "screen without p95 claim"
    markdown = render_markdown(report)
    assert "candidate: REJECTED" in markdown
    assert "Distinto" in markdown and "Español 1" in markdown
    assert report["runs"][0]["experiment_counters_source"] == "session_metadata_json"
    assert report["runs"][1]["experiment_counters_source"] == "replay_json.session_metadata"
    assert report["runs"][0]["experiment_counters"] == {
        "partial_suppressed_in_flight": 2,
        "final_stt_waited_for_translation": 0,
    }
    assert len(report["inputs"]) == 22  # manifest, clip, five artifacts per run
    for run in spec["runs"]:
        from pathlib import Path

        assert report["inputs"][run["csv"]]["sha256"] == hashlib.sha256(Path(run["csv"]).read_bytes()).hexdigest()


@pytest.mark.parametrize("memory_rule", ["absolute", "relative"])
def test_passing_screen_and_cli(tmp_path, memory_rule):
    path, _ = screen(tmp_path, passing=True, memory_rule=memory_rule)
    output, markdown = tmp_path / "report.json", tmp_path / "report.md"
    assert main(["--runs", str(path), "--output", str(output), "--markdown", str(markdown)]) == 0
    report = json.loads(output.read_text())
    outcome = report["outcomes"]["candidate"]
    assert outcome["outcome"] == "PASS"
    assert outcome["gates_on_all_clips"] == dict.fromkeys([f"G{i}" for i in range(1, 8)], True)
    assert outcome["failing_gates"] == []
    assert not outcome["p95_claim_eligible"]
    assert "candidate: PASS" in markdown.read_text()


def test_relative_memory_boundary_and_missing_observations_fail_closed(tmp_path):
    path, spec = screen(tmp_path, passing=True, memory_rule="relative")
    from pathlib import Path

    replay_path = Path(spec["runs"][2]["replay_json"])
    replay = json.loads(replay_path.read_text())
    replay["session_lifecycle"]["memory"]["peak_rss_bytes"] = 1000 + MEMORY_ALLOWANCE_BYTES
    _json(replay_path, replay)
    assert analyze(path)["clips"]["A"]["arms"]["candidate"]["gates"]["G6"]["pass"]
    replay["session_lifecycle"]["memory"]["peak_rss_bytes"] += 1
    del replay["partial_count"]
    _json(replay_path, replay)
    gates = analyze(path)["clips"]["A"]["arms"]["candidate"]["gates"]
    assert not gates["G6"]["pass"] and not gates["G5"]["pass"]


def test_filters_counters_fallback_and_unaligned_csv_rows(tmp_path):
    from pathlib import Path

    path, spec = screen(tmp_path, passing=True)
    run = spec["runs"][2]
    _json(Path(run["session_metadata_json"]), {})
    diagnostic = Path(run["diagnostics_jsonl"])
    rows = [json.loads(line) for line in diagnostic.read_text().splitlines()]
    rows.extend(
        [
            {**rows[0], "timing_schema_version": 1},
            {**rows[0], "speech_end_to_final_ms": "500"},
            {**rows[0], "speech_end_to_final_ms": True},
            {**rows[0], "event": "ignored"},
            {"event": "session_summary", "latency_experiment_counters": {"partial_suppressed_backlog": 7}},
        ]
    )
    diagnostic.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    csv_path = Path(run["csv"])
    with csv_path.open("a", encoding="utf-8", newline="") as stream:
        csv.writer(stream).writerow([7, "Extra", "Adicional"])
    report = analyze(path)
    observed = report["runs"][2]
    assert observed["cohorts"]["silence"]["all"]["n"] == 4
    assert observed["experiment_counters_source"] == "diagnostics_jsonl.session_summary"
    assert observed["experiment_counters"] == {"partial_suppressed_backlog": 7}
    identity = report["clips"]["A"]["arms"]["candidate"]["gates"]["G4"]["comparisons"][0]
    assert identity["english_identical_share"] == 6 / 7
    assert identity["chunk_count_difference"] == 1
    assert identity["aligned_length"] == 6
    assert identity["identical_share"] == 1
    assert identity["differences"][0]["control"] is None


@pytest.mark.parametrize("target", ["manifest", "clip", "csv", "same_outputs", "hardlink"])
def test_cli_never_overwrites_inputs_or_output_aliases(tmp_path, target):
    from pathlib import Path

    path, spec = screen(tmp_path, passing=True)
    output = tmp_path / "output.json"
    markdown = tmp_path / "output.md"
    if target == "manifest":
        output = path
    elif target == "clip":
        output = Path(spec["clips"]["A"]["path"])
    elif target == "csv":
        output = Path(spec["runs"][0]["csv"])
    elif target == "same_outputs":
        markdown = output
    else:
        output.hardlink_to(path)
    before = output.read_bytes() if output.exists() else None
    with pytest.raises(SystemExit, match="2"):
        main(["--runs", str(path), "--output", str(output), "--markdown", str(markdown)])
    assert (output.read_bytes() if output.exists() else None) == before


def test_second_clip_failure_rejects_arm(tmp_path):
    path, spec = screen(tmp_path, passing=True)
    spec["clips"]["B"] = spec["clips"]["A"]
    # Missing candidate observations on a declared clip cannot vacuously pass.
    _json(path, spec)
    outcome = analyze(path)["outcomes"]["candidate"]
    assert outcome["outcome"] == "REJECTED"
    assert outcome["failing_gates"] == [f"B:G{i}" for i in range(1, 8)]


def test_absolute_memory_checks_pageouts_independently(tmp_path):
    path, spec = screen(tmp_path, passing=True)
    spec["runs"][2]["pageouts"]["after"] += 1
    _json(path, spec)
    gates = analyze(path)["clips"]["A"]["arms"]["candidate"]["gates"]
    assert not gates["G6"]["pass"]
    assert gates["G6"]["runs"][0]["checks"]["peak_metal_bytes"]
    assert all(g["pass"] for key, g in gates.items() if key != "G6")


def test_g1_absolute_reduction_alternative_and_unknown_route_exclusion(tmp_path):
    from pathlib import Path

    path, spec = screen(tmp_path, passing=True)
    for run in spec["runs"]:
        file = Path(run["diagnostics_jsonl"])
        rows = [json.loads(line) for line in file.read_text().splitlines()]
        for row in rows:
            if row["endpoint_reason"] == "silence" and row["final_translation_route"] == "gemma":
                row["speech_end_to_final_ms"] = 3600 if run["arm"] == "candidate" else 4000
        # An unknown route does affect all-route silence but cannot affect G1.
        if run["arm"] == "candidate":
            rows[3]["speech_end_to_final_ms"] = 9000
        file.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    gates = analyze(path)["clips"]["A"]["arms"]["candidate"]["gates"]
    assert gates["G1"] == {"pass": True, "candidate": 3600, "control": 4000, "limit": 3700}
    assert not gates["G2"]["pass"]
    assert gates["G2"]["silence"]["candidate"] == 9000


def test_identity_survives_one_extra_early_chunk():
    from tools.tail_screen_report import _identity

    rows = [{"chunk_id": str(i), "english": f"English {i}", "spanish_a": f"Español {i}"} for i in range(6)]
    candidate = [rows[0], {"chunk_id": "extra", "english": "Extra", "spanish_a": "Adicional"}, *rows[1:]]
    result = _identity({"tag": "candidate", "csv_rows": candidate}, {"tag": "control", "csv_rows": rows})
    assert result["identical_share"] == 1.0
    assert result["english_identical_share"] == 6 / 7
    assert result["aligned_length"] == 6 and result["pass"]
    assert len(result["differences"]) == 1
    assert result["differences"][0]["row_index"] == 1
    assert result["differences"][0]["control"] is None
    candidate[3] = {**candidate[3], "spanish_a": "Changed translation"}
    result = _identity({"tag": "candidate", "csv_rows": candidate}, {"tag": "control", "csv_rows": rows})
    assert result["identical_share"] == 5 / 6
    assert len(result["differences"]) == 2
    assert result["differences"][0]["fields"] == ["spanish_a"]
