"""Speaker-label helpers for live diarization (Phase 9.6.1).

Pure functions: JSONL parsing, timestamp overlap assignment, and online
cosine clustering. No model loads — safe for unit tests and the live
pipeline's asyncio thread (JSONL lookup is a short file read).
"""

from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

# Display labels. pyannote emits SPEAKER_00; clustering emits these directly.
_LETTER_LABELS = tuple(f"Speaker {chr(ord('A') + i)}" for i in range(26))

DEFAULT_CLUSTER_THRESHOLD = 0.65
DEFAULT_MAX_SPEAKERS = 4


@dataclass
class SpeakerSegment:
    """One labeled interval from the live diarization JSONL."""

    speaker: str
    start_ts: float | None = None
    end_ts: float | None = None
    confidence: float = 0.0
    chunk_id: int = 0
    timestamp: float | None = None  # wall-clock write time (`ts` in JSONL)

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "speaker": self.speaker,
            "confidence": self.confidence,
            "ts": self.timestamp,
            "start_ts": self.start_ts,
            "end_ts": self.end_ts,
        }


def normalize_speaker_label(raw: str | None) -> str:
    """Map pyannote ``SPEAKER_00`` / already-pretty labels to ``Speaker A``."""
    if not raw:
        return "Speaker A"
    text = str(raw).strip()
    if not text:
        return "Speaker A"
    if text.lower().startswith("speaker "):
        rest = text.split(None, 1)[-1]
        if rest:
            return f"Speaker {rest.upper()[:1]}" if rest[:1].isalpha() else text
        return "Speaker A"
    digits = ""
    for ch in text:
        if ch.isdigit():
            digits += ch
    if digits:
        idx = int(digits)
        if 0 <= idx < len(_LETTER_LABELS):
            return _LETTER_LABELS[idx]
    return text


def next_speaker_label(index: int) -> str:
    if 0 <= index < len(_LETTER_LABELS):
        return _LETTER_LABELS[index]
    return f"Speaker {index + 1}"


def overlap_seconds(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    """Inclusive overlap in seconds. Zero-width (point) intervals count as 1.0 if inside."""
    if a_end < a_start:
        a_start, a_end = a_end, a_start
    if b_end < b_start:
        b_start, b_end = b_end, b_start
    if a_start == a_end:
        return 1.0 if b_start <= a_start <= b_end else 0.0
    if b_start == b_end:
        return 1.0 if a_start <= b_start <= a_end else 0.0
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def _as_float(value) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_jsonl_record(rec: dict) -> SpeakerSegment | None:
    """Parse one daemon JSONL object, including legacy records without timestamps.

    Legacy (PR #70 stub)::

        {"chunk_id": 1, "speaker": "Speaker A", "confidence": 0.9, "ts": 1.0}

    Current::

        {"chunk_id": 1, "speaker": "Speaker A", "confidence": 0.9,
         "ts": 1.0, "start_ts": 0.2, "end_ts": 1.0}
    """
    if not isinstance(rec, dict):
        return None
    if rec.get("_metadata"):
        return None
    raw_speaker = rec.get("speaker")
    if raw_speaker is None or raw_speaker == "":
        return None
    ts = _as_float(rec.get("ts") if rec.get("ts") is not None else rec.get("timestamp"))
    start_ts = _as_float(rec.get("start_ts"))
    end_ts = _as_float(rec.get("end_ts"))
    # Legacy: no interval — treat `ts` as a point so overlap still works.
    if start_ts is None and end_ts is None and ts is not None:
        start_ts = ts
        end_ts = ts
    try:
        chunk_id = int(rec.get("chunk_id", 0) or 0)
    except (TypeError, ValueError):
        chunk_id = 0
    confidence = _as_float(rec.get("confidence")) or 0.0
    return SpeakerSegment(
        speaker=normalize_speaker_label(str(raw_speaker)),
        start_ts=start_ts,
        end_ts=end_ts,
        confidence=confidence,
        chunk_id=chunk_id,
        timestamp=ts,
    )


def load_segments(path: str | Path) -> list[SpeakerSegment]:
    """Load speaker segments from a JSONL file. Skips malformed lines."""
    jsonl_path = Path(path)
    if not jsonl_path.exists():
        return []
    segments: list[SpeakerSegment] = []
    try:
        text = jsonl_path.read_text(encoding="utf-8")
    except OSError:
        return []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except (json.JSONDecodeError, TypeError):
            continue
        seg = parse_jsonl_record(rec)
        if seg is not None:
            segments.append(seg)
    return segments


def assign_speaker(
    utterance_start: float,
    utterance_end: float,
    segments: list[SpeakerSegment],
    *,
    carry_forward: bool = True,
) -> str | None:
    """Pick the speaker whose labeled interval overlaps ``[start, end]`` most.

    When the daemon lags (labels land after the final is broadcast) there is
    often no overlap with the *current* utterance. ``carry_forward`` then
    returns the latest segment that started at or before ``utterance_end`` —
    the Sunday long-turn case (1–3 speakers, long stretches of one voice).
    """
    if utterance_end < utterance_start:
        utterance_start, utterance_end = utterance_end, utterance_start
    best_speaker: str | None = None
    best_overlap = 0.0
    best_conf = -1.0
    for seg in segments:
        if seg.start_ts is None or seg.end_ts is None:
            continue
        ov = overlap_seconds(utterance_start, utterance_end, seg.start_ts, seg.end_ts)
        if ov > best_overlap or (ov == best_overlap and ov > 0 and seg.confidence > best_conf):
            best_overlap = ov
            best_speaker = seg.speaker
            best_conf = seg.confidence
    if best_speaker:
        return best_speaker
    if not carry_forward or not segments:
        return None
    eligible = [s for s in segments if s.start_ts is not None and s.start_ts <= utterance_end]
    if not eligible:
        eligible = [s for s in segments if s.timestamp is not None and s.timestamp <= utterance_end]
        if not eligible:
            return None
        return max(eligible, key=lambda s: s.timestamp or 0.0).speaker
    return max(eligible, key=lambda s: s.start_ts or 0.0).speaker


def assign_speaker_from_jsonl(
    jsonl_path: str | Path | None,
    utterance_start: float,
    utterance_end: float,
    *,
    carry_forward: bool = True,
) -> str | None:
    """Load ``jsonl_path`` and assign a speaker. Missing file → None."""
    if not jsonl_path:
        return None
    return assign_speaker(
        utterance_start,
        utterance_end,
        load_segments(jsonl_path),
        carry_forward=carry_forward,
    )


def utterance_wallclock(
    utterance_start_perf: float | None,
    duration_s: float,
    *,
    now_perf: float | None = None,
    now_wall: float | None = None,
) -> tuple[float, float]:
    """Convert a perf_counter utterance start + duration to wall-clock ``[start, end]``.

    ``end`` is the end of *speech* (start + duration), not "now" — translation
    latency must not stretch the labeled interval.
    """
    now_perf = time.perf_counter() if now_perf is None else now_perf
    now_wall = time.time() if now_wall is None else now_wall
    duration_s = max(0.0, float(duration_s))
    if utterance_start_perf is not None:
        start_wall = now_wall - (now_perf - float(utterance_start_perf))
    else:
        start_wall = now_wall - duration_s
    return start_wall, start_wall + duration_s


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    dot = 0.0
    na = 0.0
    nb = 0.0
    for i in range(n):
        dot += a[i] * b[i]
        na += a[i] * a[i]
        nb += b[i] * b[i]
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


@dataclass
class OnlineSpeakerCluster:
    """Online cosine clustering of speaker embeddings → Speaker A/B/C/…"""

    threshold: float = DEFAULT_CLUSTER_THRESHOLD
    max_speakers: int = DEFAULT_MAX_SPEAKERS
    _centroids: list[tuple[str, list[float]]] = field(default_factory=list)
    _counts: dict[str, int] = field(default_factory=dict)

    def assign(self, embedding: list[float]) -> tuple[str, float]:
        """Return ``(display_label, confidence)``. Confidence is cosine to centroid."""
        vec = [float(x) for x in embedding]
        if not vec:
            label = next_speaker_label(0)
            if not self._centroids:
                self._centroids.append((label, [1.0]))
                self._counts[label] = 1
            return label, 0.0

        best_label: str | None = None
        best_sim = -1.0
        for label, centroid in self._centroids:
            sim = cosine_similarity(vec, centroid)
            if sim > best_sim:
                best_sim = sim
                best_label = label

        at_cap = len(self._centroids) >= self.max_speakers
        if best_label is not None and (best_sim >= self.threshold or at_cap):
            self._update_centroid(best_label, vec)
            return best_label, max(0.0, min(1.0, float(best_sim)))

        label = next_speaker_label(len(self._centroids))
        self._centroids.append((label, list(vec)))
        self._counts[label] = 1
        return label, 1.0

    def _update_centroid(self, label: str, embedding: list[float]) -> None:
        for i, (lab, centroid) in enumerate(self._centroids):
            if lab != label:
                continue
            n = self._counts.get(label, 1)
            updated = []
            m = min(len(centroid), len(embedding))
            for j in range(m):
                updated.append((centroid[j] * n + embedding[j]) / (n + 1))
            if len(embedding) > m:
                updated.extend(embedding[m:])
            self._centroids[i] = (lab, updated)
            self._counts[label] = n + 1
            return


def load_recent_captions(csv_path: str | Path | None, limit: int = 8) -> list[dict]:
    """Trailing session-CSV rows for the operator caption view.

    Legacy CSVs without a ``speaker`` column still return english/spanish.
    """
    if not csv_path:
        return []
    path = Path(csv_path)
    if not path.exists():
        return []
    try:
        with path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    except OSError:
        return []
    captions: list[dict] = []
    for row in rows[-max(1, int(limit)) :]:
        english = str(row.get("english") or "").strip()
        if not english:
            continue
        speaker = str(row.get("speaker") or "").strip() or None
        captions.append(
            {
                "chunk_id": row.get("chunk_id"),
                "speaker": speaker,
                "english": english,
                "spanish": str(row.get("spanish_a") or "").strip() or None,
            }
        )
    return captions


def speaker_field_for_result(enabled: bool, speaker: str | None) -> dict:
    """``result_data`` gets ``speaker`` only when diarization is enabled."""
    if not enabled:
        return {}
    return {"speaker": speaker}
