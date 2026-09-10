"""Disposable PortAudio worker. No models; native device opens stay out of the parent."""

from __future__ import annotations

import json
import queue
import struct
import sys
import time


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

    def callback(indata, frames, time_info, status):
        nonlocal sample_position, dropped
        captured = time.perf_counter()
        metadata = {
            "frames": frames,
            "channels": options["channels"],
            "inputBufferAdcTime": time_info.inputBufferAdcTime,
            "currentTime": time_info.currentTime,
            "received": captured,
            "sample_start": sample_position,
            "dropped": dropped,
            "status": str(status) if status else "",
            "input_device": identity,
        }
        sample_position += frames
        try:
            frames_queue.put_nowait((metadata, indata.tobytes()))
        except queue.Full:
            dropped += frames

    with sd.InputStream(callback=callback, **options):
        while True:
            metadata, payload = frames_queue.get(timeout=10)
            write_frame(sys.stdout.buffer, metadata, payload)


if __name__ == "__main__":
    main()
