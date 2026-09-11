"""Blank Marian results must not replace a usable translated preview."""

import asyncio
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from unittest.mock import Mock

import numpy as np
import pytest

from tools.latency_experiments import LatencyExperiments
from tools.latency_trace import LatencyTrace
from tools.pipeline_timing import RenderTracker


@pytest.mark.parametrize("language", ["en", "es"])
@pytest.mark.parametrize("empty_target", ["", " \t\n"])
def test_empty_translation_preserves_previous_preview_then_accepts_next(monkeypatch, tmp_path, language, empty_target):
    import dry_run_ab as pipeline

    source = "God loves every person" if language == "en" else "Dios ama a cada persona"
    targets = iter(["First valid target", empty_target, "Next valid target"])
    sent = []
    trace = LatencyTrace(True, origin=0)

    class Client:
        async def send(self, payload):
            sent.append(json.loads(payload))

        async def close(self):
            pass

    monkeypatch.setattr(pipeline, "SOURCE_LANG", language)
    monkeypatch.setattr(pipeline, "SESSION_ID", "empty-preview-test")
    monkeypatch.setattr(pipeline, "_SESSION_CLOCK_ORIGIN", 0)
    monkeypatch.setattr(pipeline, "_latency", LatencyExperiments())
    monkeypatch.setattr(pipeline, "_experiment_counters", {})
    monkeypatch.setattr(pipeline, "_latency_trace", trace)
    monkeypatch.setattr(pipeline, "_render_tracker", RenderTracker())
    monkeypatch.setattr(pipeline, "ws_clients", {Client()})
    monkeypatch.setattr(pipeline, "MULTIPROCESS", True)
    monkeypatch.setattr(pipeline, "_partial_sequence", 0)
    monkeypatch.setattr(pipeline, "_published_final_session", None)
    monkeypatch.setattr(pipeline, "_published_final_floor", 0)
    monkeypatch.setattr(pipeline, "_discarded_utterance_id", 0)
    for name in ("_published_final_utterances", "_closed_utterances"):
        monkeypatch.setattr(pipeline, name, set())
    for name in ("_partial_emitted_sequence", "_partial_source_text", "partial_translations", "partial_latencies"):
        monkeypatch.setattr(pipeline, name, {})
    for name in ("_health", "_caption_delivery", "_stt_scheduler", "_active_partial_future", "_pipeline_chunk_queue"):
        monkeypatch.setattr(pipeline, name, None)
    monkeypatch.setattr(pipeline, "_final_pending", threading.Event())
    monkeypatch.setattr(pipeline, "_active_stt_workers", {"partial": 0, "final": 0})
    monkeypatch.setattr(pipeline.settings.translation, "final_aware_partials", False)
    monkeypatch.setattr(pipeline, "_run_partial_stt_via_worker", lambda audio: (source, 100))
    monkeypatch.setattr(pipeline, "translate_marian", lambda text: (next(targets), 40))
    monkeypatch.setattr(pipeline, "_is_garbage_text", lambda text: False)
    monkeypatch.setattr(pipeline, "_should_suppress", lambda *args, **kwargs: None)
    path = tmp_path / "metrics/partials.jsonl"
    monkeypatch.setattr(pipeline, "PARTIALS_PATH", str(path))
    io = Mock()
    io.submit.side_effect = lambda function, *args: function(*args)
    monkeypatch.setattr(pipeline, "_io_pool", io)
    bounds = {"sample_start": 32000, "sample_end": 48000, "sample_rate": 16000, "padding_samples": 0}

    def records():
        return [json.loads(line) for line in path.read_text().splitlines()]

    async def preview():
        await pipeline.process_partial(
            np.full(16000, 0.1, dtype=np.float32),
            7,
            sample_bounds=bounds,
            preview_kind="periodic",
            timing_source="replay_realtime",
        )

    async def run():
        await preview()
        assert len(sent) == len(records()) == io.submit.call_count == 1
        before = deepcopy(
            {
                name: getattr(pipeline, name)
                for name in (
                    "_partial_emitted_sequence",
                    "_partial_source_text",
                    "partial_translations",
                    "partial_latencies",
                )
            }
        )
        await preview()
        assert len(sent) == len(records()) == io.submit.call_count == 1
        assert {name: getattr(pipeline, name) for name in before} == before
        assert pipeline._experiment_snapshot()["partial_emitted"] == 1
        assert pipeline._experiment_snapshot()["partial_suppressed_empty_translation"] == 1
        suppressed = [
            {key: value for key, value in event.items() if key not in ("event", "at_ms", "thread_id", "thread_name")}
            for event in trace.snapshot()["events"]
            if event["event"] == "partial_suppressed_empty_translation"
        ]
        assert len(suppressed) == 1
        assert suppressed[0] == {
            "session_id": "empty-preview-test",
            "utterance_id": 7,
            "request_sequence": 2,
            "preview_kind": "periodic",
            "source_lang": language,
            "target_lang": "es" if language == "en" else "en",
            "stt_ms": 100,
            "marian_ms": 40,
            **bounds,
        }
        await preview()

    with ThreadPoolExecutor(max_workers=2) as pool:
        monkeypatch.setattr(pipeline, "_stt_comm_pool", pool)
        monkeypatch.setattr(pipeline, "_pytorch_pool", pool)
        asyncio.run(run())

    emitted = records()
    assert len(sent) == len(emitted) == io.submit.call_count == 2
    assert [record["preview_revision"] for record in emitted] == [1, 3]
    assert [message["preview_revision"] for message in sent] == [1, 3]
    assert all(record["text_en"].strip() and record["text_es"].strip() for record in emitted)
    assert [message["spanish_a"] for message in sent] == ["First valid target", "Next valid target"]
    target_key = "text_es" if language == "en" else "text_en"
    assert [record[target_key] for record in emitted] == ["First valid target", "Next valid target"]
    assert pipeline.partial_translations == {7: "Next valid target"}
    assert pipeline._partial_emitted_sequence == {7: 3}
    counters = pipeline._experiment_snapshot()
    assert counters["partial_emitted"] == counters["preview_periodic"] == 2
    assert counters["partial_suppressed_empty_translation"] == 1
    assert counters["partial_stt_started"] == counters["partial_stt_finished"] == 3
