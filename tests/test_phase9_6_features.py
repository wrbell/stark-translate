"""Phase 9.6 — feature integration tests (verses + summary)."""

from __future__ import annotations

import csv
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def _reset():
    from operator_app import audio, features, metrics, pipeline_manager

    metrics.reset_collector_for_tests()
    pipeline_manager.reset_runner_for_tests()
    audio.reset_watcher_for_tests()
    features.reset_features_for_tests()
    yield
    metrics.reset_collector_for_tests()
    pipeline_manager.reset_runner_for_tests()
    audio.reset_watcher_for_tests()
    features.reset_features_for_tests()


@pytest.fixture
def tmp_csv(tmp_path: Path) -> Path:
    p = tmp_path / "ab_metrics_test.csv"
    with p.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["chunk_id", "english", "spanish", "stt_ms", "translate_ms", "latency_ms", "confidence"])
        w.writerow([1, "Turn to Romans 8:28 with me.", "Vuelvan a Romanos 8:28 conmigo.", 100, 200, 300, 0.9])
        w.writerow([2, "And John 3:16 says God so loved the world.", "Y Juan 3:16 dice...", 110, 210, 320, 0.92])
    return p


# -- VerseHighlightWatcher ---------------------------------------------------


class TestVerseHighlightWatcher:
    def test_force_scan_extracts_references(self, tmp_csv):
        from operator_app.features import VerseHighlightWatcher

        # Inject a stub VerseExtractor so we don't depend on the real
        # regex tables (which are tested elsewhere).
        watcher = VerseHighlightWatcher(csv_path=tmp_csv)
        stub = MagicMock()
        stub.references = []

        def extract(text, timestamp="", speaker=None):
            if "Romans" in text:
                stub.references.append({"reference": "Romans 8:28", "context": text})
            elif "John" in text:
                stub.references.append({"reference": "John 3:16", "context": text})

        stub.extract_from_text = extract
        watcher._extractor = stub

        highlights = watcher.force_scan()
        refs = [h["reference"] for h in highlights]
        assert "Romans 8:28" in refs
        assert "John 3:16" in refs
        # Each highlight has chunk_id + reference + context
        assert all("chunk_id" in h and "reference" in h for h in highlights)

    def test_only_processes_new_chunks(self, tmp_csv):
        """Subsequent scans don't re-emit references from already-seen chunks."""
        from operator_app.features import VerseHighlightWatcher

        watcher = VerseHighlightWatcher(csv_path=tmp_csv)
        stub = MagicMock()
        call_log = []
        stub.references = []

        def extract(text, timestamp="", speaker=None):
            call_log.append(text)
            if "Romans" in text:
                stub.references.append({"reference": "Romans 8:28", "context": text})

        stub.extract_from_text = extract
        watcher._extractor = stub

        watcher.force_scan()
        first = len(call_log)
        watcher.force_scan()
        # No new rows in CSV → no new extract calls.
        assert len(call_log) == first

    def test_since_chunk_filters_results(self, tmp_csv):
        from operator_app.features import VerseHighlight, VerseHighlightWatcher

        watcher = VerseHighlightWatcher(csv_path=tmp_csv)
        watcher._highlights = [
            VerseHighlight(chunk_id=1, reference="A", context=""),
            VerseHighlight(chunk_id=5, reference="B", context=""),
            VerseHighlight(chunk_id=9, reference="C", context=""),
        ]
        snap = watcher.snapshot(since_chunk=5)
        refs = [h["reference"] for h in snap]
        assert refs == ["C"]


# -- SummaryTaskRunner -------------------------------------------------------


class TestSummaryTaskRunner:
    def test_submit_runs_subprocess_and_records_result(self, tmp_path):
        from operator_app.features import SummaryTaskRunner

        # Stand up a fake features/summarize_sermon.py that writes a JSON
        # to the requested --output path and exits 0.
        feats = tmp_path / "features"
        feats.mkdir()
        stub = feats / "summarize_sermon.py"
        stub.write_text(
            "import argparse, json, sys\n"
            "p = argparse.ArgumentParser()\n"
            "p.add_argument('--input', required=True)\n"
            "p.add_argument('--output', required=True)\n"
            "args = p.parse_args()\n"
            "with open(args.output, 'w') as f: json.dump({'summary_en': 'hi', 'summary_es': 'hola'}, f)\n"
            "sys.exit(0)\n"
        )
        # Provide a dummy input CSV
        csv_p = tmp_path / "fake.csv"
        csv_p.write_text("chunk_id,english\n1,Hello\n")

        runner = SummaryTaskRunner(project_root=tmp_path)
        task = runner.submit(csv_path=str(csv_p))
        # Wait for the background thread to finish (subprocess is tiny)
        for _ in range(40):
            t = runner.get(task.task_id)
            if t and t.state in ("done", "error"):
                break
            time.sleep(0.1)
        t = runner.get(task.task_id)
        assert t is not None
        assert t.state == "done", f"state={t.state}, error={t.error}"
        assert t.return_code == 0
        assert t.result == {"summary_en": "hi", "summary_es": "hola"}

    def test_subprocess_failure_records_error(self, tmp_path):
        from operator_app.features import SummaryTaskRunner

        feats = tmp_path / "features"
        feats.mkdir()
        stub = feats / "summarize_sermon.py"
        stub.write_text("import sys; sys.stderr.write('boom\\n'); sys.exit(2)\n")
        csv_p = tmp_path / "fake.csv"
        csv_p.write_text("x\n")

        runner = SummaryTaskRunner(project_root=tmp_path)
        task = runner.submit(csv_path=str(csv_p))
        for _ in range(40):
            t = runner.get(task.task_id)
            if t and t.state in ("done", "error"):
                break
            time.sleep(0.1)
        t = runner.get(task.task_id)
        assert t is not None
        assert t.state == "error"
        assert t.return_code == 2
        assert "boom" in (t.error or "")


# -- /api/features/* endpoints -----------------------------------------------


@pytest.fixture
def client_and_session(tmp_path):
    """TestClient where the runner singleton points at a tmp project + CSV."""
    from fastapi.testclient import TestClient

    from operator_app import features, pipeline_manager
    from operator_app.main import app

    pipeline_manager.reset_runner_for_tests()
    features.reset_features_for_tests()

    runner = pipeline_manager.PipelineRunner(project_root=tmp_path)
    pipeline_manager._runner = runner
    return TestClient(app), tmp_path, runner


class TestFeatureEndpoints:
    def test_verses_returns_empty_when_no_session(self, client_and_session):
        client, _, _ = client_and_session
        resp = client.get("/api/features/verses")
        assert resp.status_code == 200
        body = resp.json()
        assert body["highlights"] == []

    def test_summary_400_when_no_csv(self, client_and_session):
        client, _, _ = client_and_session
        resp = client.post("/api/features/summary", json={})
        assert resp.status_code == 400

    def test_summary_404_when_csv_missing(self, client_and_session):
        client, root, _ = client_and_session
        resp = client.post("/api/features/summary", json={"csv_path": str(root / "missing.csv")})
        assert resp.status_code == 404

    def test_summary_status_404_for_unknown_task(self, client_and_session):
        client, _, _ = client_and_session
        resp = client.get("/api/features/summary/notarealid")
        assert resp.status_code == 404
