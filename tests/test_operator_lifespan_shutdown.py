"""Actual app lifespan owns lazy polling workers, including rapid shutdown."""

from __future__ import annotations

import asyncio
import threading

import pytest
from fastapi.testclient import TestClient

from operator_app import audio, features, main, metrics, pipeline_manager


@pytest.fixture(autouse=True)
def isolated_workers(monkeypatch):
    def close():
        pipeline_manager.shutdown_runner()
        features.shutdown_features()
        metrics.shutdown_collector()
        audio.shutdown_watcher()

    close()
    monkeypatch.setattr(audio, "list_devices", lambda: audio.DeviceListing())
    monkeypatch.setattr(metrics, "_read_vram_mib", lambda: None)
    monkeypatch.setattr(metrics, "_read_cpu_percent", lambda: None)
    yield
    close()


def client():
    return TestClient(main.app, base_url="http://127.0.0.1", client=("127.0.0.1", 50000))


def assert_quiet(workers):
    assert all(worker._stop_event.is_set() for worker in workers)
    assert all(not worker._thread.is_alive() for worker in workers)
    assert audio._watcher is None
    assert metrics._collector is None
    assert features._verse_watcher is None
    assert features._diarize_watcher is None
    assert features._summary_runner is None
    assert pipeline_manager._runner is None


def test_health_then_immediate_lifespan_exit_cancels_delayed_device_import(monkeypatch):
    entered = threading.Event()
    native_calls = []

    class HeldInitialWait(threading.Event):
        """Hold first device polling wait until stop, independently of CI speed."""

        def wait(self, timeout=None):
            entered.set()
            return super().wait(2)

    original_init = audio.DeviceWatcher.__init__

    def delayed_init(self):
        original_init(self)
        self._stop_event = HeldInitialWait()

    monkeypatch.setattr(audio.DeviceWatcher, "__init__", delayed_init)
    monkeypatch.setattr(audio, "list_devices", lambda: native_calls.append(True) or audio.DeviceListing())
    with client() as session:
        assert session.get("/healthz").status_code == 200
        watcher, collector = audio._watcher, metrics._collector
        assert entered.wait(1)
        assert watcher._thread.is_alive()
    assert_quiet([watcher, collector])
    assert native_calls == [], "Shutdown must wake the initial wait before a late native import"


def test_repeated_real_lifespans_recreate_all_pollers(tmp_path):
    prior = []
    for iteration in range(2):
        with client() as session:
            assert session.get("/healthz").status_code == 200
            verse = features.get_verse_watcher(tmp_path / f"{iteration}.csv", project_root=tmp_path)
            diarize = features.get_diarize_watcher(tmp_path / f"{iteration}.jsonl")
            summary = features.get_summary_runner(project_root=tmp_path)
            runner = pipeline_manager.get_runner()
            current = [audio._watcher, metrics._collector, verse, diarize]
            assert all(worker._thread.is_alive() for worker in current)
            assert all(worker not in prior for worker in current)
        assert_quiet(current)
        assert features._summary_runner is not summary
        assert pipeline_manager._runner is not runner
        prior.extend(current)


def test_empty_lifespan_does_not_construct_any_singletons(monkeypatch):
    def absent(*args, **kwargs):
        pytest.fail("Shutdown constructed an absent worker")

    monkeypatch.setattr(audio, "DeviceWatcher", absent)
    monkeypatch.setattr(metrics, "MetricsCollector", absent)
    monkeypatch.setattr(features, "VerseHighlightWatcher", absent)
    monkeypatch.setattr(features, "LiveDiarizationWatcher", absent)
    monkeypatch.setattr(features, "SummaryTaskRunner", absent)
    monkeypatch.setattr(pipeline_manager, "PipelineRunner", absent)
    with client():
        pass
    assert_quiet([])


def test_exception_unwinding_runs_all_cleanup_even_if_pipeline_shutdown_fails(monkeypatch):
    watcher = audio.get_watcher()
    collector = metrics.get_collector()

    def failure():
        raise RuntimeError("controlled pipeline shutdown failure")

    monkeypatch.setattr(main, "shutdown_runner", failure)

    async def lifespan():
        async with main.app.router.lifespan_context(main.app):
            raise ValueError("original handler failure")

    with pytest.raises(ValueError, match="original handler failure"):
        asyncio.run(lifespan())
    assert_quiet([watcher, collector])


@pytest.mark.parametrize(
    "worker_type",
    [audio.DeviceWatcher, metrics.MetricsCollector, features.VerseHighlightWatcher, features.LiveDiarizationWatcher],
)
def test_polling_stop_reports_a_worker_that_did_not_join(worker_type, tmp_path):
    class StuckThread:
        def join(self, timeout):
            assert timeout == 2

        def is_alive(self):
            return True

    worker = (
        worker_type(tmp_path / "input")
        if worker_type in (features.VerseHighlightWatcher, features.LiveDiarizationWatcher)
        else worker_type()
    )
    worker._thread = StuckThread()
    with pytest.raises(RuntimeError, match="did not stop within 2 seconds"):
        worker.stop()
    assert worker._stop_event.is_set()
