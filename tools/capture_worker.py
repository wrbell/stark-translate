"""Disposable PortAudio worker. No models; native device opens stay out of the parent."""

from __future__ import annotations

import json
import queue
import signal
import struct
import sys
import time

from tools.capture_protocol import capture_status, write_terminal_receipt


class StopCapture(BaseException):
    """POSIX TERM requests bounded context-manager cleanup."""


def request_capture_stop(_signal, _frame):
    raise StopCapture


def write_frame(stream, metadata, payload):
    header = json.dumps(metadata).encode()
    stream.write(struct.pack("!I", len(header)))
    stream.write(header)
    stream.write(payload)
    stream.flush()


def main():
    import numpy as np
    import sounddevice as sd

    options = json.loads(sys.argv[1])
    mode = options.pop("mode", "capture")
    duration = options.pop("duration_s", 2)
    device_name = options.pop("device_name", None)
    device_host_api = options.pop("device_host_api", None)
    terminal_path = options.pop("terminal_path", None)
    capture_token = options.pop("capture_token", None)
    identity = None
    if mode != "output":
        from tools.input_devices import resolve_input_device

        identity = resolve_input_device(options.get("device"), name=device_name, host_api=device_host_api, sd=sd)
        if identity is not None:
            options["device"] = identity["index"]
    if mode == "output":
        rate = 48000
        # Short quiet test tone with smooth fade; never persist any audio.
        t = np.arange(int(rate * duration)) / rate
        audio = (0.08 * np.sin(2 * np.pi * 440 * t) * np.sin(np.pi * np.arange(len(t)) / len(t)) ** 2).astype("float32")
        sd.play(audio, rate, device=options.get("device"), blocking=True)
        print(json.dumps({"ok": True, "device": options.get("device"), "duration_s": duration}))
        return
    if mode == "probe":
        rate = 48000
        samples = sd.rec(
            int(duration * rate), samplerate=rate, channels=1, dtype="float32", device=options.get("device")
        )
        sd.wait()
        print(
            json.dumps(
                {
                    "ok": True,
                    "device": options.get("device"),
                    "input_device": identity,
                    "duration_s": duration,
                    "rms": float(np.sqrt(np.mean(samples * samples))),
                    "peak": float(np.max(np.abs(samples))),
                    "samples": len(samples),
                    "recorded": False,
                }
            )
        )
        return

    frames_queue = queue.Queue(maxsize=32)
    sample_position = dropped = 0
    callback_count = dropped_callbacks = input_overflows = 0
    pipe_written_samples = 0

    def callback(indata, frames, time_info, status):
        nonlocal sample_position, dropped, callback_count, dropped_callbacks, input_overflows
        captured = time.perf_counter()
        input_overflow = bool(capture_status(status).input_overflow_callbacks)
        callback_count += 1
        input_overflows += int(input_overflow)
        metadata = {
            "capture_schema_version": 2,
            "frames": frames,
            "channels": options["channels"],
            "inputBufferAdcTime": time_info.inputBufferAdcTime,
            "currentTime": time_info.currentTime,
            "received": captured,
            "sample_start": sample_position,
            "dropped": dropped,
            "worker_fifo_dropped_samples": dropped,
            "portaudio_input_overflow": input_overflow,
            "portaudio_input_overflow_callbacks": input_overflows,
            "status": str(status) if status else "",
            "input_device": identity,
        }
        sample_position += frames
        try:
            frames_queue.put_nowait((metadata, indata.tobytes()))
        except queue.Full:
            dropped += frames
            dropped_callbacks += 1

    stream = None
    stop_reason = "capture_error"
    old_handler = None
    if terminal_path is not None and sys.platform != "win32":
        old_handler = signal.signal(signal.SIGTERM, request_capture_stop)
    try:
        stream = sd.InputStream(callback=callback, **options)
        with stream:
            while True:
                metadata, payload = frames_queue.get(timeout=10)
                write_frame(sys.stdout.buffer, metadata, payload)
                pipe_written_samples += metadata["frames"]
    except StopCapture:
        stop_reason = "requested_stop"
    finally:
        if old_handler is not None:
            signal.signal(signal.SIGTERM, old_handler)
        if terminal_path is not None:
            with frames_queue.mutex:
                pending_samples = sum(item[0]["frames"] for item in frames_queue.queue)
            write_terminal_receipt(
                terminal_path,
                {
                    "schema_version": 1,
                    "capture_token": capture_token,
                    "stream_closed": bool(stream is not None and getattr(stream, "closed", False)),
                    "stop_reason": stop_reason,
                    "callback_samples": sample_position,
                    "callback_count": callback_count,
                    "worker_fifo_dropped_samples": dropped,
                    "worker_fifo_dropped_callbacks": dropped_callbacks,
                    "worker_fifo_admitted_samples": sample_position - dropped,
                    "portaudio_input_overflow_callbacks": input_overflows,
                    "worker_fifo_pending_samples": pending_samples,
                    "pipe_written_samples": pipe_written_samples,
                    # Includes a partially written or not-yet-bookkept frame.
                    # A flush acknowledges the pipe, never parent consumption.
                    "writer_unfinished_samples": sample_position - dropped - pending_samples - pipe_written_samples,
                },
            )


if __name__ == "__main__":
    main()
