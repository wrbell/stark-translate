"""Rolling speech window for live diarization (Phase 9.6.1).

Keeps the last ~25 s of *speech* (concatenated final-chunk audio) on disk as
``rolling.wav`` plus a ``chunks.jsonl`` index and ``rolling_meta.json`` so the
diarization daemon can map relative pyannote times back to wall-clock.
"""

from __future__ import annotations

import json
import threading
import wave
from collections import deque
from pathlib import Path

import numpy as np

DEFAULT_WINDOW_S = 25.0
SAMPLE_RATE = 16000


def write_wav_mono(path: str | Path, audio, sample_rate: int = SAMPLE_RATE) -> Path:
    """Write float32 mono audio as 16-bit PCM WAV (stdlib ``wave``, no scipy)."""
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(audio, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        arr = np.zeros(1, dtype=np.float32)
    pcm = np.clip(np.round(arr * 32767.0), -32768, 32767).astype(np.int16)
    with wave.open(str(dest), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm.tobytes())
    return dest


class RollingSpeechWindow:
    """Ring of recent speech chunks; refreshes ``rolling.wav`` after each append."""

    def __init__(
        self,
        session_dir: str | Path,
        max_seconds: float = DEFAULT_WINDOW_S,
        sample_rate: int = SAMPLE_RATE,
    ) -> None:
        self.session_dir = Path(session_dir)
        self.max_seconds = float(max_seconds)
        self.sample_rate = int(sample_rate)
        self.rolling_wav = self.session_dir / "rolling.wav"
        self.meta_path = self.session_dir / "rolling_meta.json"
        self.chunks_jsonl = self.session_dir / "chunks.jsonl"
        self._chunks: deque[dict] = deque()
        self._lock = threading.Lock()
        self.session_dir.mkdir(parents=True, exist_ok=True)

    def append(self, audio, chunk_id: int, start_ts: float, end_ts: float) -> Path:
        """Append one final's audio, trim to ``max_seconds``, rewrite rolling WAV."""
        arr = np.asarray(audio, dtype=np.float32).reshape(-1)
        duration = float(arr.size) / float(self.sample_rate) if self.sample_rate else 0.0
        record = {
            "chunk_id": int(chunk_id),
            "start_ts": float(start_ts),
            "end_ts": float(end_ts),
            "duration_s": duration,
            "audio": arr,
        }
        with self._lock:
            self._chunks.append(record)
            self._trim()
            self._write_rolling_wav()
            self._write_meta()
            self._append_chunk_index(record)
            return self.rolling_wav

    def _trim(self) -> None:
        total = sum(c["duration_s"] for c in self._chunks)
        while self._chunks and total > self.max_seconds and len(self._chunks) > 1:
            dropped = self._chunks.popleft()
            total -= dropped["duration_s"]

    def _write_rolling_wav(self) -> None:
        if not self._chunks:
            return
        combined = np.concatenate([c["audio"] for c in self._chunks])
        write_wav_mono(self.rolling_wav, combined, sample_rate=self.sample_rate)

    def _write_meta(self) -> None:
        if not self._chunks:
            return
        offset = 0.0
        chunks_meta = []
        for c in self._chunks:
            chunks_meta.append(
                {
                    "chunk_id": c["chunk_id"],
                    "start_ts": c["start_ts"],
                    "end_ts": c["end_ts"],
                    "duration_s": round(c["duration_s"], 3),
                    "offset_s": round(offset, 3),
                }
            )
            offset += c["duration_s"]
        payload = {
            "sample_rate": self.sample_rate,
            "duration_s": round(offset, 3),
            "window_start_ts": self._chunks[0]["start_ts"],
            "window_end_ts": self._chunks[-1]["end_ts"],
            "chunks": chunks_meta,
        }
        self.meta_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    def _append_chunk_index(self, record: dict) -> None:
        wav_name = f"chunk_{int(record['chunk_id']):04d}.wav"
        line = {
            "chunk_id": record["chunk_id"],
            "wav": wav_name,
            "start_ts": record["start_ts"],
            "end_ts": record["end_ts"],
            "duration_s": round(record["duration_s"], 3),
        }
        with self.chunks_jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(line) + "\n")
