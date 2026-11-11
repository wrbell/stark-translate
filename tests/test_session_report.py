"""Session report contracts: fixtures come from the real lifecycle, review and diagnostics writers."""

import json
import statistics
import subprocess
import sys
from pathlib import Path

import pytest

from tools.review_data import ReviewStore, atomic_jsonl
from tools.session_lifecycle import finish_session, start_session
from tools.session_report import build_session_report, main, render_markdown

SID = "20261108_101502_en"
LATENCY = [900.0, 1200.0, 1500.0, 4000.0, 1100.0]
SUMMARY = {
    "event": "session_summary",
    "chunks_attempted": 7,
    "chunks_completed": 5,
    "chunks_empty_stt": 1,
    "chunks_hallucination": 0,
    "chunks_dedup": 1,
    "source_coverage": {"outcomes": [{"state": "final_ready"}, {"state": "rejected_low_energy"}]},
}


def _row(root: Path, cid: int, *, kind: str = "live", audio: bool = True) -> dict:
    path = root / "stark_data" / "live_sessions" / SID / f"chunk_{cid:04d}.wav"
    if audio:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"RIFFtest")
    return {
        "chunk_id": cid,
        "session": SID,
        "english": f"Line {cid}.",
        "spanish_gemma": f"Linea {cid}.",
        "source_lang": "en",
        "session_kind": kind,
        "audio_path": str(path.relative_to(root)),
        "review_priority": cid,
        "is_hallucination": cid == 4,
        "low_confidence_words": [{"word": "x"}] if cid in (2, 3) else [],
        "final_translation_route": "marian" if cid == 5 else "gemma",
        "speech_end_to_final_ms": LATENCY[cid - 1],
        "true_e2e_ms": LATENCY[cid - 1] + 500,
    }


def _session(root: Path, *, finish: dict | None = None, kind: str = "live", approve: bool = True) -> ReviewStore:
    run = start_session(root, SID, git_sha="abc123")
    rows = [_row(root, cid, kind=kind, audio=cid != 5) for cid in range(1, 6)]
    events = [{"event": "empty_stt_drop", "stage": "final", "id": 9}, SUMMARY]
    store = ReviewStore(root)
    atomic_jsonl(store.diagnostics_path(SID), [*rows, *events])
    if finish is not None:
        finish_session(root, SID, run_id=run["run_id"], **finish)
    if approve:
        update = {"expected_revision": 0, "source_lang": "en", "transcript_approved": True}
        store.save(SID, 1, {**update, "translation_approved": True})
    return store


def _report(root: Path) -> dict:
    return build_session_report(SID, root / "metrics", root / "stark_data")


def test_completed_session_with_approval_is_ready(tmp_path):
    _session(tmp_path, finish={})
    report = _report(tmp_path)
    lifecycle = json.loads((tmp_path / "metrics" / f"session_lifecycle_{SID}.json").read_text())
    run, problems = report["run"], report["problems"]
    assert report["schema_version"] == 1 and report["session_id"] == SID
    assert run["status"] == run["recorded_status"] == "completed" and run["git_sha"] == "abc123"
    assert run["peak_rss_bytes"] == lifecycle["memory"]["peak_rss_bytes"]
    assert report["counts"] == {"finals": 5, "finals_by_route": {"gemma": 4, "marian": 1}}
    assert problems["hallucination_flags"] == 1 and problems["chunks_with_low_confidence_words"] == 2
    assert problems["empty_stt_drop_events"] == {"final": 1}
    assert problems["session_summary"]["chunks_dedup"] == 1
    assert problems["session_summary"]["source_coverage_outcomes"] == {"final_ready": 1, "rejected_low_energy": 1}
    for field, values in (("speech_end_to_final_ms", LATENCY), ("true_e2e_ms", [v + 500 for v in LATENCY])):
        cuts = statistics.quantiles(values, n=100, method="inclusive")
        group = "schema2" if field == "speech_end_to_final_ms" else "legacy_processing"
        assert report["latency_ms"][group][field] == {"n": 5, "p50": round(cuts[49], 1), "p95": round(cuts[94], 1)}
    assert report["latency_ms"]["schema2"]["broadcast_ms"]["p50"] is None
    top = report["review"]["top"]
    assert [item["chunk_id"] for item in top] == [5, 4, 3, 2]  # both approvals: not pending (ReviewStore.sessions)
    assert top[0]["audio_exists"] is False and top[1]["audio_exists"] is True and "hallucination" in top[1]["flags"]
    assert report["integrity"]["audio_missing"] == 1
    training = report["training"]
    assert training["approved"] == training["reviewed"] == 1 and training["excluded"] == 0
    assert training["ready_for_training"] is True and training["reasons"] == []


def test_completed_session_without_approval_is_not_ready(tmp_path):
    _session(tmp_path, finish={}, approve=False)
    training = _report(tmp_path)["training"]
    assert training["ready_for_training"] is False
    assert training["reasons"] == ["No chunk is human-approved"]


def test_replay_session_with_approval_is_not_ready(tmp_path):
    _session(tmp_path, finish={}, kind="replay")
    assert any("live" in reason for reason in _report(tmp_path)["training"]["reasons"])


@pytest.mark.parametrize(
    ("finish", "status"),
    [
        ({"status": "interrupted", "exit_code": 1}, "interrupted"),
        ({"persistence": {"ok": False, "pending": 0, "failed": 1, "failures": [{"code": "OSError"}]}}, "failed"),
    ],
)
def test_abnormal_session_reports_reason_and_is_not_ready(tmp_path, finish, status):
    _session(tmp_path, finish=finish)
    report = _report(tmp_path)
    assert report["run"]["status"] == status and "abnormally" in report["run"]["reason"]
    assert report["training"]["ready_for_training"] is False
    if status == "failed":
        assert report["problems"]["write_failures"] == {"failed": 1, "failures": [{"code": "OSError"}]}


def test_diagnostics_changed_after_completion_is_not_ready(tmp_path):
    store = _session(tmp_path, finish={})
    with store.diagnostics_path(SID).open("a") as stream:
        stream.write('{"chunk_id": 6, "english": "late"}\n')
    report = _report(tmp_path)
    assert report["run"]["recorded_status"] == "completed" and report["run"]["status"] == "unknown"
    assert report["training"]["ready_for_training"] is False and "changed" in report["training"]["reasons"][0]


def test_crash_with_truncated_last_line_is_counted_not_fatal(tmp_path, monkeypatch):
    store = _session(tmp_path, approve=False)
    with store.diagnostics_path(SID).open("a") as stream:
        stream.write('{"chunk_id": 6, "english": "cut off')
    monkeypatch.setattr("tools.session_lifecycle._alive", lambda pid: False)
    report = _report(tmp_path)
    assert report["run"]["recorded_status"] == "running" and report["run"]["status"] == "failed"
    assert report["run"]["duration_s"] is None and "ended_at" in report["missing"]["duration_s"]
    assert report["inputs"]["diagnostics_skipped_lines"] == 1 and report["counts"]["finals"] == 5
    assert report["training"]["ready_for_training"] is False
    assert "- Duration: not recorded." in render_markdown(report)


def test_excluded_chunk_and_top_limit(tmp_path):
    store = _session(tmp_path, finish={})
    store.save(SID, 5, {"expected_revision": 0, "source_lang": "en", "transcript_approved": True, "excluded": True})
    report = build_session_report(SID, tmp_path / "metrics", tmp_path / "stark_data", top=2)
    assert (report["training"]["approved"], report["training"]["excluded"]) == (1, 1)
    assert [item["chunk_id"] for item in report["review"]["top"]] == [4, 3] and report["review"]["pending"] == 3


def test_missing_inputs_give_nulls_with_reasons(tmp_path):
    report = _report(tmp_path)
    assert report["run"]["status"] == "unknown" and report["counts"]["finals"] == 0
    assert report["problems"]["session_summary"] is None and "session_summary" in report["missing"]
    assert report["run"]["peak_rss_bytes"] is None and report["training"]["ready_for_training"] is False
    assert "Git: not recorded." in render_markdown(report)


def test_markdown_is_short_and_lists_review_rows(tmp_path):
    _session(tmp_path, finish={})
    text = render_markdown(_report(tmp_path))
    lines = text.splitlines()
    assert lines[0] == f"# Session report {SID}"
    assert "- Training: READY." in text and "| 5 | 5 |" in text and "missing" in text
    assert len(lines) <= 20 + 2 + 4  # summary lines, table header, four review rows


def test_cli_writes_both_files_for_latest_session(tmp_path, capsys):
    _session(tmp_path, finish={})
    assert (
        main(["--latest", "--metrics-dir", str(tmp_path / "metrics"), "--data-dir", str(tmp_path / "stark_data")]) == 0
    )
    written = capsys.readouterr().out.split()
    assert [Path(p).name for p in written] == [f"session_report_{SID}.json", f"session_report_{SID}.md"]
    assert json.loads(Path(written[0]).read_text())["training"]["ready_for_training"] is True


def test_real_diagnostics_writer_fields_feed_the_report(tmp_path, monkeypatch):
    import dry_run_ab as d

    path = tmp_path / "metrics" / f"diagnostics_{SID}.jsonl"
    monkeypatch.setattr(d, "DIAG_PATH", str(path))
    monkeypatch.setattr(d, "SESSION_ID", SID)
    for name in ("homophones", "near_misses", "bad_splits", "marian_diverge", "durations", "stt_corrections"):
        monkeypatch.setattr(d, f"diag_{name}", [])
    data = {"chunk_id": 1, "timestamp": "t", "english": "Amen.", "spanish_a": "Amen."}
    d.write_diag_jsonl({**data, "final_translation_route": "gemma", "speech_end_to_final_ms": 800.0}, None)
    report = _report(tmp_path)
    assert report["counts"]["finals_by_route"] == {"gemma": 1}
    assert report["latency_ms"]["schema2"]["speech_end_to_final_ms"]["p50"] == 800.0
    assert report["review"]["top"][0]["audio_path"] is None and report["integrity"]["audio_not_retained"] == 1


def test_import_loads_no_model_or_pipeline_module():
    code = "import sys, tools.session_report; print(sorted({'torch', 'mlx', 'dry_run_ab'} & set(sys.modules)))"
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, cwd=Path(__file__).parents[1]
    )
    assert result.stdout.strip() == "[]"
