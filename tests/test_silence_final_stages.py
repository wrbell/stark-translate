"""Small, hand-calculated fixtures for offline stage attribution."""

import hashlib
import json
from datetime import datetime

import pytest

from tools.silence_final_stages import SCALARS, STAGES, analyze, main, render_markdown


def caption(scale=1, **overrides):
    # Consecutive stage deltas: 100, 2, 3, 5, 20, 7, 11, 13, 200, 17, 19.
    # Final total = 378; broadcast is outside that total.
    stamps = dict(
        zip(
            (
                "speech_end",
                "vad_finalized",
                "submitted",
                "dequeued",
                "stt_started",
                "stt_finished",
                "translation_requested",
                "translation_lock_acquired",
                "translation_started",
                "translation_finished",
                "final_ready",
                "broadcast_finished",
            ),
            [1000 + scale * offset for offset in (0, 100, 102, 105, 110, 130, 137, 148, 161, 361, 378, 397)],
            strict=True,
        )
    )
    return {
        "timing_schema_version": 2,
        "timing_source": "replay_realtime",
        "endpoint_reason": "silence",
        "timing_stages_ms": stamps,
        "speech_end_to_final_ms": 378 * scale,
        **dict.fromkeys(SCALARS, 10 * scale),
        "spanish_gemma": "Traducción",
        "spanish_marian": "Vista previa",
        **overrides,
    }


def write_rows(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def diagnostics(tmp_path):
    missing = caption(3)
    missing["timing_stages_ms"]["translation_lock_acquired"] = None
    return write_rows(
        tmp_path / "diagnostics.jsonl",
        [
            caption(),
            caption(2),
            caption(4, tps_a=0, spanish_gemma=""),
            caption(9, endpoint_reason="smart_cut"),
            caption(99, event="session_summary"),
            missing,
            caption(99, timing_schema_version=1),
            caption(99, event=None),
            caption(99, timing_schema_version="2"),
        ],
    )


def test_stage_statistics_filtering_routes_and_checksum(diagnostics):
    report = analyze(diagnostics, "fixture")
    assert report["jsonl_rows"] == 4
    assert report["filters"]["endpoint_reason"] == "silence"
    assert "endpoint_cohorts" not in report
    gemma, marian = (report["cohorts"][route] for route in ("gemma", "marian"))
    assert gemma["jsonl_rows"] == 3
    assert marian["jsonl_rows"] == 1
    expected = {
        "silence_wait_and_vad_decision": 100,
        "submit": 2,
        "queue": 3,
        "stt_dispatch": 5,
        "stt_call": 20,
        "to_translation": 7,
        "translation_lock_wait": 11,
        "translation_prepare": 13,
        "translation_call": 200,
        "finalize": 17,
        "broadcast": 19,
        "total_speech_end_to_final": 378,
    }
    assert set(gemma["stages"]) == set(STAGES) | set(SCALARS)
    for stage, delta in expected.items():
        stats = gemma["stages"][stage]
        incomplete = stage in ("translation_lock_wait", "translation_prepare")
        assert stats["n"] == (2 if incomplete else 3)
        assert stats["missing"] == (1 if incomplete else 0)
        assert stats["p50"] == delta * (1 if incomplete else 2)
        assert stats["p95"] == delta * (2 if incomplete else 3)
        assert stats["mean"] == delta * (1.5 if incomplete else 2)
        assert stats["share_of_total_p50"] == stats["p50"] / 756
        assert marian["stages"][stage]["p50"] == 4 * delta
    assert gemma["stages"]["stt_latency_ms"]["p50"] == 20
    assert gemma["stages"]["stt_latency_ms"]["p95"] == 30
    assert gemma["stages"]["utterance_dur"]["unit"] == "s"
    assert gemma["stages"]["utterance_dur"]["share_of_total_p50"] is None
    assert report["checksum"]["gemma"]["total_speech_end_to_final_p50"] == 756
    assert report["checksum"]["gemma"]["speech_end_to_final_ms_p50"] == 756
    assert report["checksum"]["gemma"]["checksum_ok"] is True
    assert report["checksum"]["marian"]["checksum_ok"] is True
    assert report["ranked"][0]["stage"] == "total_speech_end_to_final"
    assert report["ranked"][1]["stage"] == "translation_call"
    assert [row["p50"] for row in report["ranked"]] == sorted([row["p50"] for row in report["ranked"]], reverse=True)


def test_cli_csv_all_endpoints_markdown_and_provenance(diagnostics, tmp_path):
    csv_path = tmp_path / "metrics.csv"
    csv_path.write_text("endpoint_reason\nsilence\nsmart_cut\nsilence\n", encoding="utf-8")
    output, markdown = tmp_path / "report.json", tmp_path / "report.md"
    before = diagnostics.read_bytes()
    assert (
        main(
            [
                "--diagnostics",
                str(diagnostics),
                "--csv",
                str(csv_path),
                "--label",
                "Test report",
                "--output",
                str(output),
                "--markdown",
                str(markdown),
                "--all-endpoints",
            ]
        )
        == 0
    )
    report = json.loads(output.read_text())
    assert report["label"] == "Test report"
    assert report["source_file"] == str(diagnostics)
    assert report["sha256"] == hashlib.sha256(before).hexdigest()
    assert datetime.fromisoformat(report["generated_at_utc"]).utcoffset().total_seconds() == 0
    assert report["csv_rows"] == 2
    assert report["jsonl_rows"] == report["jsonl_silence_rows"] == 4
    assert set(report["endpoint_cohorts"]) == {"silence", "smart_cut"}
    smart = report["endpoint_cohorts"]["smart_cut"]
    assert smart["jsonl_rows"] == 1
    assert smart["cohorts"]["gemma"]["stages"]["stt_call"]["p50"] == 180
    assert smart["checksum"]["marian"]["checksum_ok"] is None
    text = markdown.read_text()
    assert "### Ranked Gemma stages (p50 descending)" in text
    assert "| stage | p50 (ms) |" in text
    assert "| translation_call | 400.0 |" in text
    assert "| translation_lock_wait | 2.0 | 1.0 | 16.5 | 11.0 | 22.0 | 0.0 |" in text
    assert "### Gemma" in text and "### Marian" in text
    assert "approximation" in text and "medians do not sum" in text
    assert "## Endpoint: smart_cut" in text
    assert diagnostics.read_bytes() == before


def test_custom_endpoint_and_empty_cohorts(diagnostics):
    assert analyze(diagnostics, "cut", endpoint="smart_cut")["jsonl_rows"] == 1
    empty = analyze(diagnostics, "empty", endpoint="eof")
    assert empty["jsonl_rows"] == 0
    for cohort in empty["cohorts"].values():
        assert cohort["checksum"]["checksum_ok"] is None
        for stats in cohort["stages"].values():
            assert stats["n"] == stats["missing"] == 0
            assert stats["p50"] is stats["p95"] is stats["mean"] is None
    assert "| stt_call | 0.0 | 0.0 | — | — | — | — |" in render_markdown(empty)


@pytest.mark.parametrize(("difference", "ok"), [(0.5, True), (0.501, False)])
def test_checksum_tolerance(tmp_path, difference, ok):
    path = write_rows(tmp_path / "input.jsonl", [caption(speech_end_to_final_ms=378 + difference)])
    assert analyze(path, "checksum")["checksum"]["gemma"]["checksum_ok"] is ok


def test_missing_stamps_scalars_and_zero_total(tmp_path):
    rows = [caption(0), caption(timing_stages_ms=None, speech_end_to_final_ms=None)]
    rows[0]["stt_latency_ms"] = None
    del rows[1]["decode_ms_a"]
    report = analyze(write_rows(tmp_path / "missing.jsonl", rows), "missing")
    stages = report["cohorts"]["gemma"]["stages"]
    assert stages["stt_call"]["n"] == stages["stt_call"]["missing"] == 1
    assert stages["stt_call"]["p50"] == 0
    assert stages["stt_call"]["share_of_total_p50"] is None
    assert stages["stt_latency_ms"]["missing"] == stages["decode_ms_a"]["missing"] == 1
    checksum = report["checksum"]["gemma"]
    assert checksum["stamp_missing"] == checksum["scalar_missing"] == 1
    assert checksum["checksum_ok"] is True


@pytest.mark.parametrize(
    ("overrides", "route"),
    [
        ({"tps_a": None, "spanish_gemma": None}, "marian"),
        ({"tps_a": 0, "spanish_gemma": ""}, "marian"),
        ({"tps_a": 0}, "gemma"),
        ({"tps_a": 0, "spanish_gemma": "", "spanish_marian": ""}, "gemma"),
        ({"tps_a": 12, "spanish_gemma": ""}, "gemma"),
    ],
)
def test_routing_conditions(tmp_path, overrides, route):
    report = analyze(write_rows(tmp_path / "route.jsonl", [caption(**overrides)]), "route")
    assert report["cohorts"][route]["jsonl_rows"] == 1
    assert report["cohorts"][{"gemma": "marian", "marian": "gemma"}[route]]["jsonl_rows"] == 0


def test_input_collision_and_bad_json_do_not_write(diagnostics, tmp_path):
    before = diagnostics.read_bytes()
    with pytest.raises(SystemExit, match="2"):
        main(["--diagnostics", str(diagnostics), "--label", "collision", "--output", str(diagnostics)])
    assert diagnostics.read_bytes() == before
    bad = tmp_path / "bad.jsonl"
    bad.write_text("{}\nnot json\n", encoding="utf-8")
    output = tmp_path / "output.json"
    with pytest.raises(ValueError, match=r"bad\.jsonl:2: invalid JSON"):
        analyze(bad, "bad")
    with pytest.raises(SystemExit, match="2"):
        main(["--diagnostics", str(bad), "--label", "bad", "--output", str(output)])
    assert not output.exists()
