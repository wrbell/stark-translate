"""Actual isolated-reader/parent offset mapping, without native capture or models."""

import io
import json
import struct
from types import SimpleNamespace

from tools.isolated_audio import IsolatedInputStream
from tools.pipeline_timing import CaptureSampleClock


def test_restarted_isolated_reader_keeps_parent_sample_positions_across_pause():
    clock = CaptureSampleClock()
    captured = []

    def read_capture_context(received_at):
        wire = bytearray()
        # Every fresh native child starts at source position zero.
        for index in range(2):
            header = json.dumps(
                dict(
                    frames=1536,
                    channels=1,
                    inputBufferAdcTime=90 + index * 0.032,
                    currentTime=90.032 + index * 0.032,
                    received=received_at + index * 0.032,
                    sample_start=index * 1536,
                    dropped=0,
                    status="",
                )
            ).encode()
            wire.extend(struct.pack("!I", len(header)) + header + bytes(1536 * 4))
        seen = []

        def callback(samples, frames, stamp, status):
            parent_stamp = clock.capture(frames, 48000, stamp)
            captured.append(parent_stamp)
            seen.append(parent_stamp)
            if len(seen) == 2:
                stream._stop.set()

        stream = IsolatedInputStream(
            callback=callback, samplerate=48000, channels=1, dtype="float32", blocksize=1536, device=None
        )
        stream.sample_offset = clock.next_sample  # actual audio_loop assignment before each __enter__
        stream._proc = SimpleNamespace(stdout=io.BytesIO(wire))
        stream._read()  # actual framed-pipe parser, no Popen/device/reader thread
        assert stream.error is None

    for received_at in (100.0, 3700.0):
        read_capture_context(received_at)
    assert [(s.sample_start, s.sample_end) for s in captured] == [(0, 1536), (1536, 3072), (3072, 4608), (4608, 6144)]
    assert clock.next_sample == 6144
    assert captured[2].start - captured[1].end > 3599
    assert all(s.source == "portaudio_adc" and s.sample_rate == 48000 for s in captured)
