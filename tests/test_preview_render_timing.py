"""Join actual preview records to visible acknowledgments on the server clock."""

import asyncio
import csv
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from unittest.mock import Mock

import numpy as np
import pytest

from tools.latency_experiments import LatencyExperiments
from tools.pipeline_timing import RenderTracker


@pytest.mark.parametrize("queued", [False, True])
def test_actual_pause_preview_log_payload_and_ack_have_one_identity(monkeypatch, queued):
    import dry_run_ab as pipeline

    records, messages = [], []

    class Client:
        async def send(self, payload):
            messages.append(json.loads(payload))

        async def close(self):
            pass

    client = Client()
    tracker = RenderTracker()
    monkeypatch.setattr(pipeline, "_latency", replace(LatencyExperiments(), async_captions=queued))
    monkeypatch.setattr(pipeline, "SESSION_ID", "preview-test")
    monkeypatch.setattr(pipeline, "_SESSION_CLOCK_ORIGIN", 10)
    monkeypatch.setattr(pipeline.time, "perf_counter", lambda: 20)
    monkeypatch.setattr(pipeline, "ws_clients", {client})
    monkeypatch.setattr(pipeline, "_render_tracker", tracker)
    monkeypatch.setattr(pipeline, "_caption_delivery", None)
    monkeypatch.setattr(pipeline, "_partial_sequence", 0)
    monkeypatch.setattr(pipeline, "_partial_emitted_sequence", {})
    monkeypatch.setattr(pipeline, "_partial_source_text", {})
    monkeypatch.setattr(pipeline, "_closed_utterances", set())
    monkeypatch.setattr(pipeline, "_stt_scheduler", None)
    monkeypatch.setattr(pipeline, "MULTIPROCESS", True)
    monkeypatch.setattr(pipeline, "_final_pending", threading.Event())
    monkeypatch.setattr(pipeline, "_pipeline_chunk_queue", None)
    monkeypatch.setattr(pipeline, "_active_partial_future", None)
    monkeypatch.setattr(pipeline.settings.translation, "final_aware_partials", False)
    monkeypatch.setattr(pipeline, "partial_translations", {})
    monkeypatch.setattr(pipeline, "partial_latencies", {})
    monkeypatch.setattr(pipeline, "_run_partial_stt_via_worker", lambda audio: ("God loves every person", 100))
    monkeypatch.setattr(pipeline, "translate_marian", lambda text: ("Dios ama a cada persona", 40))
    monkeypatch.setattr(pipeline, "_is_garbage_text", lambda text: False)
    monkeypatch.setattr(pipeline, "_should_suppress", lambda *a, **k: None)
    io = Mock()
    io.submit.side_effect = lambda fn, record: records.append(record)
    monkeypatch.setattr(pipeline, "_io_pool", io)

    async def run():
        # Repeated snapshot bounds and text must still produce unique event IDs.
        for _ in range(2):
            await pipeline.process_partial(
                np.ones(16000, np.float32) * 0.1,
                7,
                captured_start=18,
                captured_end=19.5,
                speech_end=19.25,
                timing_source="replay_realtime",
                sample_bounds={"sample_start": 32000, "sample_end": 48000, "sample_rate": 16000},
                preview_kind="pause",
            )
        if pipeline._caption_delivery is not None:
            await pipeline._caption_delivery.close()

    with ThreadPoolExecutor(max_workers=2) as pool:
        monkeypatch.setattr(pipeline, "_stt_comm_pool", pool)
        monkeypatch.setattr(pipeline, "_pytorch_pool", pool)
        asyncio.run(run())
    assert len(records) == 2 and messages
    by_id = {row["event_id"]: row for row in records}
    assert set(by_id) == {"preview-test:partial:1", "preview-test:partial:2"}
    for payload in messages:
        row = by_id[payload["event_id"]]
        assert row["captured_end_to_partial_ms"] == 500
        assert row["speech_end_to_partial_ms"] == 750  # includes the 250 ms pause
        assert row["speech_start_to_partial_ms"] == 2000
        assert row["preview_revision"] == payload["preview_revision"]
        ack = tracker.acknowledge(
            client,
            {
                "event_id": payload["event_id"],
                "receive_to_render_ms": 32,
                "visible": True,
                "captured_start_at_ms": -999999,  # client cannot override producer references
                "preview_revision": -1,
            },
            20.1,
        )
        assert ack["preview_revision"] == row["preview_revision"]
        assert ack["preview_kind"] == "pause"
        assert ack["speech_start_to_preview_ack_upper_bound_ms"] == 2100
        assert ack["captured_end_to_preview_ack_upper_bound_ms"] == 600
        assert ack["speech_end_to_preview_ack_upper_bound_ms"] == 850
        assert ack["speech_end_to_ack_upper_bound_ms"] is None  # final metric retains its meaning
        assert ack["caption_delivery_mode"] == ("queued" if queued else "awaited")


@pytest.mark.parametrize("hidden,nonrealtime", [(True, False), (False, True)])
def test_unobservable_previews_do_not_claim_visible_latency(monkeypatch, hidden, nonrealtime):
    import dry_run_ab as pipeline

    tracker, client = RenderTracker(), object()
    monkeypatch.setattr(pipeline, "_render_tracker", tracker)
    monkeypatch.setattr(pipeline, "_SESSION_CLOCK_ORIGIN", 10)
    pipeline._caption_before_send(
        client,
        {
            "type": "translation",
            "stage": "partial",
            "event_id": "s:partial:1",
            "captured_start_at_ms": 1000,
            "captured_end_at_ms": 2000,
            "speech_end_at_ms": 1500,
            "timing_source": "replay_nonrealtime" if nonrealtime else "portaudio_adc",
        },
        12,
        50,
    )
    ack = tracker.acknowledge(
        client, {"event_id": "s:partial:1", "receive_to_render_ms": 32, "visible": not hidden}, 12.1
    )
    assert all(value is None for key, value in ack.items() if key.endswith("_upper_bound_ms"))
    assert ack["caption_queue_wait_ms"] == 50


def test_caption_mode_appends_to_csv_without_shifting_existing_columns(tmp_path, monkeypatch):
    import dry_run_ab as pipeline

    path = tmp_path / "metrics.csv"
    monkeypatch.setattr(pipeline, "CSV_PATH", str(path))
    pipeline.init_csv()
    columns = next(csv.reader(path.open()))
    assert columns[-1] == "caption_delivery_mode"
    assert columns[-2] == "prompt_cache_hit_a"
    assert columns[5] == "stt_latency_ms" and columns[8] == "e2e_latency_ms"
