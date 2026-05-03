"""vram_sampler.py — continuous nvidia-smi polling.

Extracted from ``bench_translate_t1_t4.py`` so multiple benchmarks (translation,
STT) can share one source of truth.

Why not ``torch.cuda.max_memory_allocated()``? It misses bnb scratch and the
bf16 PLE embeddings on Gemma 4, and is useless for out-of-process processes
(llama-server). End-of-run nvidia-smi misses transient KV-cache peaks that
shrink back. Continuous sampling at a fixed interval catches both.
"""

from __future__ import annotations

import statistics
import subprocess
import threading


def nvidia_smi_used_mib(gpu_index: int = 0) -> int:
    """Total VRAM used on a single GPU in MiB. Returns 0 on any error."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
                "-i",
                str(gpu_index),
            ],
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        return int(out.decode().strip().splitlines()[0])
    except Exception:
        return 0


class VramSampler:
    """Background thread polling nvidia-smi at a fixed interval, tracking running max.

    Use as a context manager *or* explicit start/stop pair:

        with VramSampler() as sampler:
            ...
        result = sampler.last_result   # populated by __exit__

    or:

        sampler = VramSampler(interval_s=0.5)
        sampler.start()
        ...
        result = sampler.stop()

    Result dict keys: ``max_mib``, ``max_gb``, ``n_samples``, ``median_mib``.
    """

    def __init__(self, interval_s: float = 1.0, gpu_index: int = 0):
        self._interval = float(interval_s)
        self._gpu_index = gpu_index
        self._max_mib = 0
        self._samples: list[int] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.last_result: dict | None = None

    def start(self) -> None:
        def _loop() -> None:
            while not self._stop.is_set():
                m = nvidia_smi_used_mib(self._gpu_index)
                if m > 0:
                    self._samples.append(m)
                    if m > self._max_mib:
                        self._max_mib = m
                self._stop.wait(self._interval)

        self._thread = threading.Thread(target=_loop, daemon=True)
        self._thread.start()

    def stop(self) -> dict:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        result = {
            "max_mib": self._max_mib,
            "max_gb": round(self._max_mib / 1024, 2),
            "n_samples": len(self._samples),
            "median_mib": int(statistics.median(self._samples)) if self._samples else 0,
        }
        self.last_result = result
        return result

    def __enter__(self) -> VramSampler:
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()
