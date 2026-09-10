"""Bounded source accounting; classifications are machine decisions, not speech truth."""

from __future__ import annotations

import threading


def merged(intervals):
    result = []
    for start, end in sorted(intervals):
        if result and start <= result[-1][1]:
            result[-1][1] = max(end, result[-1][1])
        else:
            result.append([start, end])
    return result


def missing(intervals, expected):
    cursor, gaps = 0, []
    for start, end in merged(intervals):
        if start > cursor:
            gaps.append([cursor, min(start, expected)])
        cursor = max(cursor, end)
    if cursor < expected:
        gaps.append([cursor, expected])
    return [pair for pair in gaps if pair[1] > pair[0]]


class SourceCoverage:
    def __init__(self, capacity=131072):
        if capacity < 1:
            raise ValueError("Coverage capacity must be positive")
        self.capacity = capacity
        self._lock = threading.RLock()
        self._observed = []
        self._outcomes = {}
        self._dropped = self._unknown = self._overlap = 0
        self._last_end = 0
        self._rate = None
        self._eof = None

    def _bounds(self, metadata):
        start, end, rate = (metadata.get(k) for k in ("sample_start", "sample_end", "sample_rate"))
        if any(isinstance(x, bool) or not isinstance(x, int) for x in (start, end, rate)):
            self._unknown += 1
            return None
        if start < 0 or end < start or rate <= 0 or self._rate not in (None, rate):
            self._unknown += 1
            return None
        self._rate = rate
        return start, end

    def observe(self, metadata, state, *, speech=None):
        with self._lock:
            bounds = self._bounds(metadata)
            if bounds is None or bounds[0] == bounds[1]:
                return
            start, end = bounds
            if start < self._last_end:
                self._overlap += min(end, self._last_end) - start
            self._last_end = max(end, self._last_end)
            if (
                self._observed
                and self._observed[-1]["state"] == state
                and self._observed[-1]["end"] == start
                and self._observed[-1].get("vad_positive") == speech
            ):
                self._observed[-1]["end"] = end
            elif len(self._observed) < self.capacity:
                self._observed.append({"start": start, "end": end, "state": state, "vad_positive": speech})
            else:
                self._dropped += 1

    def outcome(self, metadata, state, utterance_id=None):
        with self._lock:
            bounds = self._bounds(metadata)
            if bounds is None or bounds[0] == bounds[1]:
                return
            key = (*bounds, utterance_id)
            if key not in self._outcomes and len(self._outcomes) >= self.capacity:
                self._dropped += 1
                return
            self._outcomes[key] = {"start": bounds[0], "end": bounds[1], "utterance_id": utterance_id, "state": state}

    def eof(self, source_samples, sample_rate):
        with self._lock:
            if self._bounds({"sample_start": 0, "sample_end": source_samples, "sample_rate": sample_rate}) is not None:
                self._eof = {"source_samples": source_samples, "reader_eof": True, "capture_queues_drained": True}

    def snapshot(self):
        with self._lock:
            expected = self._eof["source_samples"] if self._eof else self._last_end
            observed = [(r["start"], r["end"]) for r in self._observed]
            classified = [(r["start"], r["end"]) for r in self._observed if r["state"] != "buffered"]
            classified += [(r["start"], r["end"]) for r in self._outcomes.values() if r["state"] != "submitted"]
            capture_gaps = missing(observed, expected)
            disposition_gaps = missing(classified, expected)
            outside = sum(max(0, end - max(start, expected)) for start, end in observed)
            return {
                "schema_version": 1,
                "classification_scope": "machine VAD/filter/caption decisions; not acoustic truth or approved speech recall",
                "sample_rate": self._rate,
                "expected_source_samples": expected,
                "eof": self._eof,
                "dropped_records": self._dropped,
                "unknown_bounds": self._unknown,
                "duplicate_observed_samples": self._overlap,
                "observed_samples_beyond_eof": outside,
                "capture_gaps": capture_gaps,
                "unclassified_intervals": disposition_gaps,
                "complete": bool(self._eof)
                and not (
                    capture_gaps or disposition_gaps or self._dropped or self._unknown or self._overlap or outside
                ),
                "observed": [dict(r) for r in self._observed],
                "outcomes": [dict(r) for r in self._outcomes.values()],
            }
