"""Physical scheduling, exact reuse and bounded delivery without loading models."""

import asyncio
import json
import threading
from dataclasses import replace

import pytest

from tools.caption_delivery import CaptionDelivery
from tools.latency_experiments import LatencyExperiments
from tools.latency_scheduler import ExactTextMemo, LatestSTTWorker, PartialRuntimePredictor
from tools.latency_trace import LatencyTrace


def test_default_experiments_preserve_deployment_behavior():
    assert LatencyExperiments.from_env({}) == LatencyExperiments()
    assert LatencyExperiments.from_env({"STARK_EXPERIMENT_LATEST_PARTIAL": "true"}).latest_partial
    for key, value in (("LATEST_PARTIAL", "yes"), ("FIRST_PREVIEW_S", "nan"), ("MLX_CACHE_MB", "4096")):
        with pytest.raises(ValueError, match="STARK_EXPERIMENT"):
            LatencyExperiments.from_env({"STARK_EXPERIMENT_" + key: value})


@pytest.mark.parametrize("value", ["1024", "131072", "262144"])
def test_trace_capacity_is_explicit_bounded_and_snapshot_visible(value):
    config = LatencyExperiments.from_env({"STARK_EXPERIMENT_TRACE_CAPACITY": value})
    assert config.trace is False  # More capacity does not enable research tracing.
    assert config.trace_capacity == int(value)
    assert config.as_dict()["trace_capacity"] == int(value)
    trace = LatencyTrace(config.trace, capacity=config.trace_capacity)
    trace.record("ignored")
    snapshot = trace.snapshot()
    assert snapshot["capacity"] == int(value)
    assert snapshot["total_events"] == 0 and snapshot["events"] == []
    assert LatencyExperiments.from_env({}).trace_capacity == LatencyTrace().snapshot()["capacity"] == 8192


@pytest.mark.parametrize("value", ["0", "1023", "262145", "131072.0", "NaN", "true", "-1"])
def test_trace_capacity_rejects_invalid_environment_values(value):
    with pytest.raises(ValueError, match="STARK_EXPERIMENT_TRACE_CAPACITY"):
        LatencyExperiments.from_env({"STARK_EXPERIMENT_TRACE_CAPACITY": value})


def test_deadline_predictor_cold_start_duration_bins_and_expiry():
    predictor = PartialRuntimePredictor(history=3)
    assert predictor.admit(3, now=10, deadline=10.1, margin_ms=100) == (True, None)
    for value in (200, 300, 400):
        predictor.observe(3, value)
    assert predictor.admit(3, now=10, deadline=10.49, margin_ms=100) == (False, 400)
    assert predictor.admit(3, now=10, deadline=10.5, margin_ms=100) == (True, 400)
    assert predictor.admit(5, now=10, deadline=10.1, margin_ms=100) == (True, None)
    for _ in range(3):
        predictor.observe(3, 100)
    assert predictor.predict_ms(3) == 100  # old slow work ages out
    assert predictor.admit(3, now=10, deadline=None, margin_ms=250) == (True, 100)


def test_early_clause_options_require_both_bounds_and_preserve_minimum():
    for values in (
        {"EARLY_CLAUSE_S": "2"},
        {"EARLY_CLAUSE_S": "0.5", "EARLY_CLAUSE_PAUSE_MS": "160"},
        {"PARTIAL_DEADLINE_MARGIN_MS": "nan"},
    ):
        with pytest.raises(ValueError):
            LatencyExperiments.from_env({"STARK_EXPERIMENT_" + k: v for k, v in values.items()})


def test_latest_pending_partial_replaced_but_running_model_finishes_before_final():
    started, release = threading.Event(), threading.Event()
    calls = []
    worker = LatestSTTWorker()

    def current():
        started.set()
        assert release.wait(2)
        calls.append("current")

    try:
        running = worker.submit("partial", current, key=1)
        assert started.wait(1)
        assert not running.cancel()  # physical work cannot be canceled
        old = worker.submit("partial", calls.append, "obsolete", key=2)
        latest = worker.submit("partial", calls.append, "latest", key=2)
        final = worker.submit("final", calls.append, "final", key=1)
        assert old.cancelled() and worker.busy
        release.set()
        final.result(1)
        latest.result(1)
        assert calls == ["current", "final", "latest"]
    finally:
        release.set()
        worker.shutdown()


def test_worker_failure_propagates_without_losing_later_final():
    worker = LatestSTTWorker()
    try:

        def fail():
            raise ValueError("actual inference failed")

        with pytest.raises(ValueError, match="actual inference"):
            worker.submit("final", fail).result(1)
        assert worker.submit("final", lambda: "next").result(1) == "next"
    finally:
        worker.shutdown()


def test_exact_memo_does_not_merge_case_direction_or_model_and_evicts_lru():
    memo = ExactTextMemo(2)
    first = ("model1", "en", "es", "Jesus")
    memo.put(first, "Jesús")
    assert memo.get(("model1", "en", "es", "jesus")) is None
    assert memo.get(("model1", "es", "en", "Jesus")) is None
    memo.put(("model2", "en", "es", "Jesus"), "other")
    assert memo.get(first) == "Jesús"
    memo.put(("model1", "en", "es", "Hello"), "Hola")
    assert memo.get(("model2", "en", "es", "Jesus")) is None


class Client:
    def __init__(self, blocked=False):
        self.messages = []
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.closed = False
        if not blocked:
            self.release.set()

    async def send(self, message):
        self.entered.set()
        await self.release.wait()
        self.messages.append(json.loads(message))

    async def close(self):
        self.closed = True


def caption(text, stage="partial", chunk=1):
    return {"type": "translation", "stage": stage, "chunk_id": chunk, "session_id": "s", "text": text}


def test_slow_caption_client_does_not_block_fast_client_and_only_latest_pending_partial_survives():
    async def run():
        slow, fast = Client(True), Client()
        sent = []
        delivery = CaptionDelivery(before_send=lambda client, message, *_: sent.append((client, message["text"])))
        delivery.publish(slow, caption("running"))
        await slow.entered.wait()
        delivery.publish(slow, caption("old"))
        delivery.publish(slow, caption("new"))
        delivery.publish(slow, caption("final", "complete"))
        delivery.publish(fast, caption("independent", "complete"))
        await fast.entered.wait()
        assert not slow.messages
        slow.release.set()
        await delivery.close()
        assert [m["text"] for m in slow.messages] == ["running", "new", "final"]
        assert [m["text"] for m in fast.messages] == ["independent"]
        assert (slow, "old") not in sent  # no render timing for unsent superseded event

    asyncio.run(run())


def test_reliable_queue_overflow_disconnects_instead_of_silently_dropping_final():
    async def run():
        failures = []
        client = Client()
        delivery = CaptionDelivery(capacity=1, on_failure=lambda *args: failures.append(args))
        delivery.publish(client, caption("one", "complete"))
        delivery.publish(client, caption("two", "complete", 2))
        await delivery.close()
        assert len(failures) == 1 and client.closed
        assert "capacity" in str(failures[0][1])

    asyncio.run(run())


def test_caption_send_timeout_isolated_and_reported():
    async def run():
        failures = []
        client = Client(True)
        delivery = CaptionDelivery(send_timeout=0.01, on_failure=lambda *args: failures.append(args))
        delivery.publish(client, caption("one", "complete"))
        await delivery.close()
        assert len(failures) == 1 and client.closed

    asyncio.run(run())


def test_trace_reports_discarded_events_instead_of_unbounded_memory():
    trace = LatencyTrace(True, capacity=2)
    for i in range(3):
        trace.record("event", index=i)
    snapshot = trace.snapshot()
    assert snapshot["capacity"] == 2
    assert snapshot["total_events"] == 3
    assert snapshot["discarded_old_events"] == 1
    assert [r["index"] for r in snapshot["events"]] == [1, 2]


def test_trace_uses_pipeline_origin_and_records_failed_physical_work(monkeypatch):
    import tools.latency_trace as module

    monkeypatch.setattr(module.time, "perf_counter", lambda: 103.5)
    trace = LatencyTrace(True, origin=100.0)
    with pytest.raises(ValueError, match="worker failure"), trace.span("stt", kind="partial"):
        raise ValueError("worker failure")
    snapshot = trace.snapshot()
    assert snapshot["schema_version"] == 2
    assert snapshot["origin_perf_counter_s"] == 100.0
    assert all(row["at_ms"] == 3500 for row in snapshot["events"])
    assert snapshot["events"][-1]["failed"] is True
    assert snapshot["events"][-1]["thread_id"] == threading.get_ident()
    with pytest.raises(ValueError, match="capacity"):
        LatencyTrace(capacity=0)


def test_live_marian_facade_exact_hit_preserves_zero_and_request_identity(monkeypatch):
    import dry_run_ab as pipeline

    calls = []

    class Marian:
        model_id = "fixture"

        def translate(self, text, **kwargs):
            from engines.base import TranslationResult

            calls.append((text, kwargs))
            return TranslationResult("Hola", 123)

    monkeypatch.setattr(pipeline, "_marian_engine", Marian())
    monkeypatch.setattr(pipeline, "_latency", replace(LatencyExperiments(), marian_memo=2))
    monkeypatch.setattr(pipeline, "_marian_memo", ExactTextMemo(2))
    monkeypatch.setattr(pipeline, "SOURCE_LANG", "en")
    assert pipeline.translate_marian("Hello") == ("Hola", 123)
    assert pipeline.translate_marian("Hello")[0] == "Hola"
    assert len(calls) == 1
    monkeypatch.setattr(pipeline, "SOURCE_LANG", "es")
    pipeline.translate_marian("Hello")
    assert len(calls) == 2


@pytest.mark.parametrize("tokens", [1, 2, 3, 4])
def test_full_model_draft_environment_is_opt_in(tokens):
    model = "mlx-community/gemma-4-e2b-it-OptiQ-4bit"
    env = {"STARK_EXPERIMENT_DRAFT_MODEL_ID": model, "STARK_EXPERIMENT_DRAFT_TOKENS": str(tokens)}
    result = LatencyExperiments.from_env(env)
    assert result.draft_model_id == model
    assert result.draft_tokens == tokens
    assert result.as_dict()["draft_model_id"] == model
    assert result.as_dict()["draft_tokens"] == tokens
    assert LatencyExperiments.from_env({}).draft_model_id == ""
    assert LatencyExperiments.from_env({}).draft_tokens == 0
    assert (
        LatencyExperiments.from_env(
            {
                "STARK_EXPERIMENT_DRAFT_MODEL_ID": "",
                "STARK_EXPERIMENT_DRAFT_TOKENS": "0",
            }
        )
        == LatencyExperiments()
    )


@pytest.mark.parametrize(
    "env",
    [
        {"DRAFT_MODEL_ID": "model"},
        {"DRAFT_TOKENS": "2"},
        {"DRAFT_MODEL_ID": "model", "DRAFT_TOKENS": "0"},
        {"DRAFT_MODEL_ID": "", "DRAFT_TOKENS": "2"},
        *[{"DRAFT_MODEL_ID": "model", "DRAFT_TOKENS": v} for v in ("-1", "5", "nan", "1.5")],
    ],
)
def test_full_model_draft_rejects_incomplete_or_unbounded_configuration(env):
    with pytest.raises(ValueError, match="STARK_EXPERIMENT_DRAFT"):
        LatencyExperiments.from_env({"STARK_EXPERIMENT_" + k: v for k, v in env.items()})
