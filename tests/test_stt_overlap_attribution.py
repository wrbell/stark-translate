"""Synthetic shared-clock attribution; no inference or external artifacts."""

import json

import pytest

from tools.stt_overlap_attribution import analyze, main, overlap_ms


def diagnostics(tmp_path):
    events = []

    def stt(kind, cid, start, end, **fields):
        identity = {"chunk_id": cid} if kind == "final" else {"utterance_id": cid}
        events.extend(
            [
                {"event": "physical_stt_started", "kind": kind, "at_ms": start, **identity, **fields},
                {"event": "physical_stt_finished", "kind": kind, "at_ms": end, "cpu_ms": 2, **identity, **fields},
            ]
        )

    stt("final", 1, 0, 100, translation_active=False, concurrent_partial=0, concurrent_final=1)
    stt("final", 2, 800, 1800, translation_active=True, concurrent_partial=1, concurrent_final=1)
    stt("final", 3, 3000, 3200, translation_active=False, concurrent_partial=0, concurrent_final=1)
    stt("partial", 9, 700, 1000, translation_active=True, request_sequence=1)
    stt("partial", 9, 900, 1100, translation_active=True, request_sequence=2)
    # Repeated utterance identity without sequence exercises FIFO fallback.
    stt("partial", 10, 4000, 4050, translation_active=False)
    stt("partial", 10, 4100, 4150, translation_active=False)
    events.append({"event": "physical_translation_finished", "chunk_id": 1, "route": "gemma", "cpu_ms": 7})
    rows = []
    for cid, interval, endpoint, delivery in [
        (1, (100, 1600), "silence", 1700),
        (2, (1800, 2000), "smart_cut", 300),
        (3, (3200, 3400), "hard_cut", 400),
    ]:
        rows.append(
            {
                "chunk_id": cid,
                "utterance_id": cid,
                "timing_schema_version": 2,
                "endpoint_reason": endpoint,
                "timing_stages_ms": {"translation_started": interval[0], "translation_finished": interval[1]},
                "final_translation_route": "gemma",
                "speech_end_to_final_ms": delivery,
                "gen_tokens_a": 20,
                "decode_ms_a": 500,
            }
        )
    rows.append({**rows[0], "timing_schema_version": 1})
    rows.append(
        {
            "event": "session_summary",
            "latency_trace": {
                "schema_version": 2,
                "enabled": True,
                "events": sorted(events, key=lambda e: e.get("at_ms", 5000)),
            },
        }
    )
    path = tmp_path / "diagnostics.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows))
    return path


def test_three_chunk_overlap_buckets_and_endpoint_mix(tmp_path):
    path = diagnostics(tmp_path)
    report = analyze([path])
    file = report["files"][0]
    assert file["finals"][1]["stt_elapsed_ms"] == 1000
    assert file["finals"][1]["gemma_overlap_ms"] == 800
    assert file["finals"][1]["partial_stt_overlap_ms"] == 300
    assert file["finals"][1]["translation_active"] is True
    assert file["finals"][1]["cpu_ms"] == 2
    assert file["finals"][2]["gemma_overlap_ms"] == file["finals"][2]["partial_stt_overlap_ms"] == 0
    assert file["gemma_finals"][0]["decode_elapsed_ms"] == 1500
    assert file["gemma_finals"][0]["partial_stt_overlap_ms"] == 400
    assert file["gemma_finals"][0]["next_final_stt_overlap_ms"] == 800
    assert file["gemma_finals"][0]["tokens_per_second"] == 40
    assert file["gemma_finals"][0]["cpu_ms"] == 7
    assert file["gemma_finals"][1]["cpu_ms"] is None
    aggregate = report["pooled"]
    assert aggregate == file["aggregates"]
    assert aggregate["counts"]["partial_stt_calls"] == 4
    assert aggregate["counts"]["partial_stt_calls_translation_active"] == 2
    assert aggregate["stt_buckets"]["slow"] == {
        "count": 1,
        "median_overlap_shares": {"gemma_overlap_share": 0.8, "partial_stt_overlap_share": 0.3},
    }
    assert aggregate["stt_buckets"]["normal"]["count"] == 2
    assert aggregate["gemma_buckets"]["slow"]["count"] == 1
    assert aggregate["gemma_buckets"]["normal"]["count"] == 2
    assert aggregate["gemma_decode_partial_overlap_share"] == pytest.approx(400 / 1900)
    assert aggregate["endpoint_mix"]["silence"] == {
        "count": 1,
        "speech_end_to_final_ms": {"p50": 1700, "p95": 1700},
        "gen_tokens_a_p95": 20,
    }
    output, markdown = tmp_path / "report.json", tmp_path / "report.md"
    main(["--diagnostics", str(path), "--output", str(output), "--markdown", str(markdown)])
    assert json.loads(output.read_text()) == report
    assert "| stt | slow | 1 |" in markdown.read_text()
    assert "Chunk detail" in markdown.read_text()


def test_union_and_exact_slow_threshold_missing_trace(tmp_path):
    assert overlap_ms((0, 100), [(10, 40), (20, 60), (60, 80), (100, 200)]) == 70
    path = diagnostics(tmp_path)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[-1]["latency_trace"]["events"] = []
    path.write_text("\n".join(map(json.dumps, rows)))
    result = analyze([path])["files"][0]
    assert result["finals"][0]["stt_elapsed_ms"] is None
    assert result["aggregates"]["counts"]["final_stt_intervals"] == 0
    from tools.stt_overlap_attribution import _buckets, _pct

    assert _buckets([{"ms": 800}], "ms", ())["normal"]["count"] == 1
    assert _pct([1, 2, 3, 4], 50) == 2


@pytest.mark.parametrize("alias", ["input", "same_outputs", "hardlink"])
def test_cli_rejects_output_aliases_without_overwriting(tmp_path, alias):
    path = diagnostics(tmp_path)
    before = path.read_bytes()
    output, markdown = tmp_path / "out.json", tmp_path / "out.md"
    if alias == "input":
        output = path
    elif alias == "same_outputs":
        markdown = output
    else:
        output.hardlink_to(path)
    with pytest.raises(SystemExit):
        main(["--diagnostics", str(path), "--output", str(output), "--markdown", str(markdown)])
    assert path.read_bytes() == before


def test_pooled_sessions_use_separate_clocks(tmp_path):
    first = diagnostics(tmp_path)
    second = tmp_path / "second.jsonl"
    second.write_bytes(first.read_bytes())
    report = analyze([first, second])
    assert report["pooled"]["counts"]["finals"] == 6
    assert report["pooled"]["counts"]["partial_stt_calls"] == 8
    assert report["pooled"]["gemma_decode_partial_overlap_share"] == pytest.approx(400 / 1900)
