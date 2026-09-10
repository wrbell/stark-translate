"""Real producer/consumer contract checks, without loading inference models."""

import time
from pathlib import Path

from operator_app.features import SummaryTaskRunner, get_verse_watcher, reset_features_for_tests
from operator_app.metrics import MetricsCollector
from operator_app.pipeline_manager import parse_metrics_row


def test_current_and_legacy_csv_timing_do_not_mix():
    collector = MetricsCollector()
    collector.record_segment(**parse_metrics_row({"chunk_id": "1", "e2e_latency_ms": "9000"}))
    collector.record_segment(
        **parse_metrics_row(
            {
                "chunk_id": "2",
                "timing_schema_version": "2",
                "stt_latency_ms": "150",
                "latency_a_ms": "300",
                "speech_end_to_final_ms": "720",
                "stt_confidence": "0.9",
            }
        )
    )
    snap = collector.snapshot()["latency"]
    assert snap["n"] == 1 and snap["total_ms_p50"] == 720
    assert snap["stt_ms_p50"] == 150 and snap["translate_ms_p50"] == 300
    assert snap["basis"] == "speech_end_to_final_ms"
    assert snap["timing_schema_version"] == "2"


def test_missing_values_stay_null_even_with_legacy_latency_present():
    values = parse_metrics_row(
        {
            "chunk_id": "1",
            "timing_schema_version": "2",
            "e2e_latency_ms": "850",
            "stt_confidence": "",
            "stt_latency_ms": "NaN",
        }
    )
    assert values["total_ms"] is None and values["confidence"] is None and values["stt_ms"] is None
    collector = MetricsCollector()
    collector.record_segment(**values)
    snap = collector.snapshot()["latency"]
    assert snap["total_ms_p50"] is None and snap["confidence_mean"] is None
    collector.reset_session("next_en")
    assert collector.snapshot()["latency"] == {"n": 0}


def test_verse_poll_reuses_real_extractor_and_context(tmp_path):
    csv = tmp_path / "session.csv"
    csv.write_text('chunk_id,english\n1,"Turn to Romans 8:28."\n')
    reset_features_for_tests()
    try:
        first = get_verse_watcher(csv)
        assert first.force_scan()
        second = get_verse_watcher(csv)
        assert second is first
        assert second.force_scan() == first.snapshot()
        assert len(second.snapshot()) == 1
    finally:
        reset_features_for_tests()


def test_summary_uses_real_cli_positional_input(tmp_path):
    empty = tmp_path / "empty.csv"
    empty.write_text("chunk_id,english\n")
    runner = SummaryTaskRunner(Path(__file__).resolve().parents[1])
    task = runner.submit(str(empty), str(tmp_path / "summary.json"))
    for _ in range(60):
        if task.state in ("done", "error"):
            break
        time.sleep(0.05)
    assert task.return_code == 1
    assert "No transcript content found" in task.error
    assert "unrecognized arguments" not in task.error


def test_startup_buttons_allow_stop_but_not_pause():
    import shutil
    import subprocess

    import pytest

    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is needed to execute the operator UI")
    script = r"""
const {createHarness, preflightPayload} = require(process.cwd() + '/tests/frontend/operator_harness.js');
const assert = require('assert');
const h = createHarness();
h.app.renderChecks(preflightPayload());
h.app.renderStatus({state: 'starting'});
assert.strictEqual(h.el('start-btn').disabled, true);
assert.strictEqual(h.el('stop-btn').disabled, false);
assert.strictEqual(h.el('pause-btn').disabled, true);
assert.strictEqual(h.el('flip-btn').disabled, true);
h.app.renderStatus({state: 'running'});
assert.strictEqual(h.el('pause-btn').disabled, false);
h.app.renderStatus({state: 'stopping'});
assert.strictEqual(h.el('stop-btn').disabled, true);
h.app.renderStatus({state: 'error', error: 'pipeline did not stop within 10s'});
assert.strictEqual(h.el('stop-btn').disabled, false); // recovery from an error keeps Stop available
assert.strictEqual(h.el('pause-btn').disabled, true);
"""
    subprocess.run([node, "-e", script], cwd=Path(__file__).resolve().parents[1], check=True, timeout=10)
