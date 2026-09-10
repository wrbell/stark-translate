"""Real WAV replay tests; only scipy is unmocked, never devices or models."""

import asyncio
import importlib
import sys
import threading
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from tools.audio_bridge_client import FileAudioStream, open_audio_stream


@pytest.fixture
def real_scipy(monkeypatch):
    # conftest mocks scipy as well as ML libraries. Restore only scipy for
    # these DSP tests, and restore the exact prior module state afterwards.
    with monkeypatch.context() as patch:
        # scipy's array API checks torch.Tensor with issubclass even for
        # numpy inputs; keep torch mocked but give that check a real type.
        patch.setattr(sys.modules["torch"], "Tensor", type("MockTensor", (), {}))
        for name in list(sys.modules):
            if name == "scipy" or name.startswith("scipy."):
                patch.delitem(sys.modules, name)
        try:
            yield importlib.import_module("scipy.io.wavfile")
        finally:
            for name in list(sys.modules):
                if name == "scipy" or name.startswith("scipy."):
                    del sys.modules[name]


@pytest.fixture
def wav_path(tmp_path, real_scipy):
    path = tmp_path / "tone.wav"
    tone = (np.sin(2 * np.pi * 440 * np.arange(32000) / 16000) * 16384).astype(np.int16)
    real_scipy.write(path, 16000, tone)
    return path


@pytest.mark.parametrize("rate,channels", [(16000, 1), (48000, 2), (22050, 1)])
def test_replay_blocks_tail_and_completion(wav_path, rate, channels):
    blocks = []
    notified = threading.Event()

    def callback(data, frames, time_info, status):
        assert frames == 512
        from tools.pipeline_timing import CaptureStamp

        assert isinstance(time_info, CaptureStamp) and status is None
        assert time_info.end > time_info.start
        blocks.append(data)

    stream = FileAudioStream(
        wav_path,
        callback=callback,
        samplerate=rate,
        channels=channels,
        blocksize=512,
        speed=50,
        on_finished=notified.set,
    )
    with stream:
        assert stream._thread.daemon
        assert stream.finished.wait(3)
        assert notified.wait(1)
    assert stream.error is None
    assert all(block.shape == (512, channels) and block.dtype == np.float32 for block in blocks)
    audio = np.concatenate(blocks)
    assert 4 * rate <= len(audio) < 4 * rate + 512
    assert np.max(np.abs(audio)) <= 1
    assert np.max(np.abs(audio[: 2 * rate])) > 0.4
    assert np.all(audio[2 * rate :] == 0)
    if channels == 2:
        np.testing.assert_array_equal(audio[:, 0], audio[:, 1])


def test_env_dispatch(wav_path, monkeypatch):
    monkeypatch.setenv("STARK_AUDIO_SOURCE", "file")
    monkeypatch.setenv("STARK_AUDIO_FILE", str(wav_path))
    monkeypatch.setenv("STARK_REPLAY_SPEED", "50")
    stream = open_audio_stream(
        lambda *args: None, samplerate=16000, channels=1, dtype="float32", blocksize=512, device=None
    )
    assert isinstance(stream, FileAudioStream)
    assert stream.speed == 50
    with stream:
        assert stream.finished.wait(3)


@pytest.mark.parametrize("source_rate,target_rate", [(16000, 16000), (8000, 16000)])
def test_replay_sample_bounds_exclude_virtual_tail_and_last_block_padding(
    tmp_path, real_scipy, source_rate, target_rate
):
    path = tmp_path / "short.wav"
    real_scipy.write(path, source_rate, np.ones(5, np.int16) * 1000)
    stamps = []
    with FileAudioStream(
        path,
        samplerate=target_rate,
        blocksize=4,
        tail_silence_s=8 / target_rate,
        speed=0,
        callback=lambda data, frames, stamp, status: stamps.append(stamp),
    ) as stream:
        assert stream.finished.wait(1)
    audio_count = 5 * target_rate // source_rate
    assert stream._audio_sample_count == audio_count
    assert sum(stamp.sample_end - stamp.sample_start for stamp in stamps) == audio_count
    assert all(stamp.sample_rate == target_rate for stamp in stamps)
    assert all(0 <= stamp.sample_start <= stamp.sample_end <= audio_count for stamp in stamps)
    assert all(stamp.sample_end - stamp.sample_start + stamp.padding_samples == 4 for stamp in stamps)
    assert [stamp.sample_start for stamp in stamps] == sorted(stamp.sample_start for stamp in stamps)
    assert any(0 < stamp.padding_samples < 4 for stamp in stamps)
    assert stamps[-1].sample_start == stamps[-1].sample_end == audio_count
    assert stamps[-1].padding_samples == 4


@pytest.mark.parametrize("speed", [0, -1])
def test_unpaced_and_unsigned_stereo(tmp_path, real_scipy, speed):
    path = tmp_path / "unsigned.wav"
    real_scipy.write(path, 16000, np.array([[0, 128], [255, 128], [128, 128]], np.uint8))
    blocks = []
    with FileAudioStream(
        path, callback=lambda data, *args: blocks.append(data), speed=speed, blocksize=4, tail_silence_s=0
    ) as stream:
        assert stream.finished.wait(1)
    np.testing.assert_allclose(blocks[0][:, 0], [-0.5, 127 / 256, 0, 0])


def test_stop_interrupts_pacing_and_can_restart(wav_path):
    called = threading.Event()
    stream = FileAudioStream(wav_path, callback=lambda *args: called.set(), speed=0.001)
    stream.start()
    # First block becomes available after pacing; stop interrupts that wait.
    assert not called.wait(0.05)
    stream.stop()
    assert not stream._thread.is_alive()
    assert not stream.finished.is_set()
    stream.speed = 0
    stream.start()
    assert stream.finished.wait(1)
    stream.close()


def test_callback_failure_finishes(wav_path):
    def fail(*args):
        raise ValueError("callback failed")

    with FileAudioStream(wav_path, callback=fail, speed=0) as stream:
        assert stream.finished.wait(1)
    assert isinstance(stream.error, ValueError)


def test_soundfile_fallback(wav_path, real_scipy, monkeypatch):
    monkeypatch.setattr(real_scipy, "read", MagicMock(side_effect=ValueError("unsupported WAV")))
    fake_sf = MagicMock()
    fake_sf.read.return_value = (np.array([-2, 0.25, 2], np.float32), 16000)
    monkeypatch.setitem(sys.modules, "soundfile", fake_sf)
    blocks = []
    with FileAudioStream(
        wav_path, callback=lambda data, *args: blocks.append(data), blocksize=4, tail_silence_s=0, speed=0
    ) as stream:
        assert stream.finished.wait(1)
    np.testing.assert_array_equal(blocks[0][:, 0], [-1, 0.25, 1, 0])


@pytest.mark.parametrize("speech_frames", [0, 3, 25])
def test_audio_loop_drains_eof_and_waits_for_partial(monkeypatch, speech_frames):
    import dry_run_ab as d
    from tools import audio_bridge_client

    async def exercise():
        queue = asyncio.Queue()
        for _ in range(speech_frames):
            queue.put_nowait(np.ones(512, np.float32) * 0.1)
        stream = MagicMock()
        stream.finished = threading.Event()
        stream.finished.set()
        stream.error = None
        monkeypatch.setattr(audio_bridge_client, "open_audio_stream", lambda **kwargs: stream)
        monkeypatch.setattr(d, "audio_queue", queue)
        monkeypatch.setattr(d, "_pipeline_chunk_queue", asyncio.Queue())
        monkeypatch.setattr(d, "EXIT_AFTER_REPLAY", True)
        monkeypatch.setattr(d, "is_speech", lambda *args: True)
        monkeypatch.setattr(d, "process_final", AsyncMock())
        monkeypatch.setattr(d, "process_partial", AsyncMock())
        monkeypatch.setattr(d, "_active_partial_future", None)
        task = asyncio.create_task(asyncio.sleep(0.15))
        monkeypatch.setattr(d, "_partial_tasks", {task})
        await asyncio.wait_for(d.audio_loop(), 2)
        assert queue.empty()
        assert task.done()
        assert d.process_final.await_count == (1 if speech_frames == 25 else 0)

    asyncio.run(exercise())


@pytest.mark.parametrize("lang", ["en", "es"])
def test_every_partial_is_logged(tmp_path, monkeypatch, lang):
    from concurrent.futures import ThreadPoolExecutor

    import dry_run_ab as d

    path = tmp_path / "partials.jsonl"
    monkeypatch.setattr(d, "PARTIALS_PATH", str(path))
    monkeypatch.setattr(d, "SOURCE_LANG", lang)
    monkeypatch.setattr(d, "MULTIPROCESS", True)
    monkeypatch.setattr(d, "_final_pending", threading.Event())
    monkeypatch.setattr(d, "_pipeline_chunk_queue", None)
    monkeypatch.setattr(d, "_active_partial_future", None)
    monkeypatch.setattr(d, "partial_translations", {})
    monkeypatch.setattr(d, "partial_latencies", {})
    monkeypatch.setattr(d, "_run_partial_stt_via_worker", lambda audio: ("source", 100))
    monkeypatch.setattr(d, "translate_marian", lambda text: ("target", 40))
    monkeypatch.setattr(d, "_is_garbage_text", lambda text: False)
    monkeypatch.setattr(d, "_should_suppress", lambda *args, **kwargs: None)
    monkeypatch.setattr(d, "broadcast", AsyncMock())
    with ThreadPoolExecutor(max_workers=2) as pool:
        for name in ("_io_pool", "_stt_comm_pool", "_pytorch_pool"):
            monkeypatch.setattr(d, name, pool)

        async def exercise():
            await d.process_partial(np.ones(16000, np.float32) * 0.1, 7)
            await d.process_partial(np.ones(32000, np.float32) * 0.1, 7)

        asyncio.run(exercise())
    import json

    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(records) == 2
    assert {r["buffer_s"] for r in records} == {1, 2}
    for record in records:
        assert record["utterance_id"] == 7
        assert record["stt_ms"] + record["marian_ms"] == 140
        assert record["ts"]
        assert record["text_en"] == ("source" if lang == "en" else "target")
        assert record["text_es"] == ("target" if lang == "en" else "source")


def test_no_exit_after_replay_keeps_loop_open(monkeypatch):
    import dry_run_ab as d
    from tools import audio_bridge_client

    monkeypatch.setattr(d.sd, "PortAudioError", type("PortAudioError", (Exception,), {}))

    async def exercise():
        stream = MagicMock()
        stream.finished = threading.Event()
        stream.finished.set()
        stream.error = None
        monkeypatch.setattr(audio_bridge_client, "open_audio_stream", lambda **kwargs: stream)
        monkeypatch.setattr(d, "audio_queue", asyncio.Queue())
        monkeypatch.setattr(d, "EXIT_AFTER_REPLAY", False)
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(d.audio_loop(), 0.25)

    asyncio.run(exercise())


def test_file_callback_wakes_async_queue_from_reader_thread(monkeypatch):
    import dry_run_ab as d
    from tools import audio_bridge_client

    monkeypatch.setenv("STARK_AUDIO_SOURCE", "file")
    monkeypatch.setattr(d, "EXIT_AFTER_REPLAY", True)
    monkeypatch.setattr(d, "_partial_tasks", set())
    monkeypatch.setattr(d, "_active_partial_future", None)
    monkeypatch.setattr(d, "is_speech", lambda *args: False)
    monkeypatch.setattr(d, "_warmup_pending", False)
    monkeypatch.setattr(d, "_last_warmup_time", float("inf"))

    async def exercise():
        loop = asyncio.get_running_loop()
        queue = asyncio.Queue()
        monkeypatch.setattr(d, "audio_queue", queue)
        monkeypatch.setattr(d, "_pipeline_chunk_queue", asyncio.Queue())
        delivered = []

        def receive(data, frames, time_info, status):
            assert asyncio.get_running_loop() is loop
            delivered.append(frames)
            queue.put_nowait(data[:, 0])

        monkeypatch.setattr(d, "audio_callback", receive)

        class ThreadStream:
            finished = threading.Event()

            def __init__(self, callback, **kwargs):
                self.callback = callback

            def __enter__(self):
                def emit():
                    self.callback(np.zeros((512, 1), np.float32), 512, None, None)
                    self.finished.set()

                self.thread = threading.Thread(target=emit)
                self.thread.start()
                return self

            def __exit__(self, *args):
                self.thread.join()

        monkeypatch.setattr(audio_bridge_client, "open_audio_stream", ThreadStream)
        await asyncio.wait_for(d.audio_loop(), 2)
        assert delivered == [512]
        assert queue.empty()

    asyncio.run(exercise(), debug=True)
