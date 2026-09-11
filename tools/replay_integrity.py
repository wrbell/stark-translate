"""Model-free structural integrity guard for terminal real-time replay artifacts.

Source coordinates use the callback rate (normally 48 kHz); persisted/processed
PCM normally uses 16 kHz. Span duration includes explicitly recorded virtual
padding. Hash/header/trace consistency is not waveform equality with the source
and is not proof that VAD classified all speech correctly.
"""

from __future__ import annotations

import csv
import hashlib
import math
import re
import struct
from pathlib import Path

BOUNDS = ("sample_start", "sample_end", "sample_rate", "padding_samples")
IDENTITY_NUMBERS = ("chunk_id", "utterance_id", "timing_schema_version", *BOUNDS, "speech_end_sample")
IDENTITY_TEXT = (
    "timestamp",
    "english",
    "endpoint_reason",
    "timing_source",
    "source_lang",
    "target_lang",
    "session_kind",
)


def integer(value, name, minimum=0):
    if isinstance(value, bool):
        raise ValueError(f"{name}: boolean is not a sample or identity")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name}: missing/non-numeric value") from exc
    if not math.isfinite(number) or not number.is_integer() or number < minimum:
        raise ValueError(f"{name}: invalid integer {value!r}")
    return int(number)


def number(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name}: boolean is not a duration")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name}: missing/non-numeric duration") from exc
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name}: invalid duration")
    return result


def span(row):
    start, end, rate, padding = [integer(row.get(k), k, 1 if k == "sample_rate" else 0) for k in BOUNDS]
    if end <= start:
        raise ValueError("Source interval is empty or reversed")
    return start, end, rate, padding


def duration_check(row, duration, *, tolerance=1 / 16000):
    start, end, rate, padding = span(row)
    expected = (end - start + padding) / rate
    actual = number(duration, "processed PCM duration")
    if abs(actual - expected) > tolerance + 1e-9:
        raise ValueError(f"PCM/source span mismatch: actual={actual:.9f}s expected={expected:.9f}s")
    return {"source_span_plus_padding_s": expected, "actual_pcm_s": actual, "delta_s": actual - expected}


def wav_identity(path: Path):
    """Inspect PCM/FLOAT RIFF without soundfile/scipy/model imports; hash all bytes."""
    before = path.stat()
    size = before.st_size
    if size > 128 * 1024 * 1024:
        raise ValueError("Stored chunk exceeds bounded 128 MiB size")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        head = stream.read(12)
        if len(head) != 12 or head[:4] != b"RIFF" or head[8:] != b"WAVE":
            raise ValueError("Expected RIFF WAVE chunk")
        if struct.unpack_from("<I", head, 4)[0] + 8 != size:
            raise ValueError("Truncated or trailing RIFF bytes")
        fmt = None
        data_size = None
        while stream.tell() < size:
            header = stream.read(8)
            if len(header) != 8:
                raise ValueError("Truncated RIFF section")
            tag, length = struct.unpack("<4sI", header)
            if stream.tell() + length + (length % 2) > size:
                raise ValueError("RIFF section exceeds file")
            if tag == b"fmt ":
                if fmt is not None or length < 16:
                    raise ValueError("Missing/duplicate WAVE format")
                fmt = struct.unpack("<HHIIHH", stream.read(16))
                stream.seek(length - 16, 1)
            elif tag == b"data":
                if data_size is not None:
                    raise ValueError("Duplicate WAVE data")
                data_size = length
                stream.seek(length, 1)
            else:
                stream.seek(length, 1)
            stream.seek(length % 2, 1)
        if fmt is None or data_size is None:
            raise ValueError("WAVE format/data missing")
        encoding, channels, rate, byte_rate, align, bits = fmt
        if (encoding, channels, rate, bits) != (3, 1, 16000, 32) or align != 4 or byte_rate != rate * align:
            raise ValueError("Expected production mono 16 kHz FLOAT32 stored chunk")
        if not data_size or data_size % align:
            raise ValueError("Empty or fractional WAVE frame")
        stream.seek(0)
        while block := stream.read(1024 * 1024):
            digest.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("Stored chunk changed during integrity read")
    return {
        "path": str(path),
        "sha256": digest.hexdigest(),
        "size_bytes": size,
        "frames": data_size // align,
        "sample_rate": rate,
        "channels": channels,
        "bits": bits,
        "encoding": "FLOAT",
    }


def _index(rows, field):
    result = {}
    for row in rows:
        key = integer(row.get(field), field, 1)
        if key in result:
            raise ValueError(f"Duplicate {field} {key}")
        result[key] = row
    return result


def _same_numbers(first, second, fields):
    for field in fields:
        if integer(first.get(field), field) != integer(second.get(field), field):
            raise ValueError(f"Identity mismatch: {field}")


def _provenance(row, metadata, session_id):
    if row.get("session_id", row.get("session")) != session_id:
        raise ValueError("Session identity mismatch")
    if row.get("timing_schema_version") != 2 or row.get("timing_source") != "replay_realtime":
        raise ValueError("Unsupported source clock/schema")
    for field in ("input_audio_sha256", "session_kind", "source_lang", "target_lang"):
        if row.get(field) != metadata.get(field) or row.get(field) is None:
            raise ValueError(f"Provenance mismatch: {field}")
    integer(row.get("utterance_id"), "utterance_id", 1)


def _trace_key(event):
    kind = event.get("kind")
    if kind not in {"partial", "final"}:
        raise ValueError(f"Unsupported physical STT kind {kind!r}")
    uid = integer(event.get("utterance_id"), "trace utterance_id", 1)
    identity = integer(event.get("request_sequence" if kind == "partial" else "chunk_id"), "trace request identity", 1)
    return kind, uid, identity


def audit_replay_integrity(
    *,
    data_root: Path,
    session_id: str,
    metadata: dict,
    diagnostic_finals: list,
    partials: list,
    trace: dict,
    require_trace: bool = True,
) -> dict:
    """Return a fail-closed structural report; callers retain errors as run failures.

    Paths resolve relative to the pipeline's writable data root. No source audio
    is opened. Trace disabled/missing may be explicitly tolerated for historical
    inspection, but then status is ``unassessable``, never ``passed``.
    """
    errors: list[str] = []
    checks: list[dict] = []
    trace_checks: list[dict] = []
    root = Path(data_root).resolve()
    result = {
        "schema_version": 1,
        "session_id": session_id,
        "errors": errors,
        "finals": checks,
        "physical_stt": trace_checks,
        "waveform_equality": "not_checked",
        "limitations": [
            "Headers, hashes and sample counts cannot prove source waveform equality or semantic speech coverage.",
            "Stored final FLOAT32 WAVs are checked; emitted preview audio is not persisted.",
            "Trace covers physical calls, including suppressed previews; truncated traces are unavailable evidence.",
        ],
    }
    try:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,159}", session_id) or ".." in session_id:
            raise ValueError("Invalid session identity")
        if (
            metadata.get("session_id") != session_id
            or metadata.get("audio_source") != "file"
            or metadata.get("replay_speed") != 1
        ):
            raise ValueError("Expected identified real-time file replay")
        if metadata.get("timing_schema_version") != 2 or metadata.get("session_kind") not in {"replay", "synthetic"}:
            raise ValueError("Unsupported replay schema/kind")
        csv_path = root / "metrics" / f"ab_metrics_{session_id}.csv"
        with csv_path.open(newline="") as handle:
            csv_rows = list(csv.DictReader(handle, strict=True))
        csv_index = _index(csv_rows, "chunk_id")
        finals = _index(diagnostic_finals, "chunk_id")
        if set(csv_index) != set(finals):
            raise ValueError("CSV/diagnostic final identity set mismatch")
        preview_index = _index(partials, "preview_revision")
    except (ValueError, OSError, csv.Error) as exc:
        errors.append(f"artifact_inventory: {exc}")
        return {**result, "status": "failed"}
    paths = set()
    for cid, row in finals.items():
        try:
            _provenance(row, metadata, session_id)
            joined = csv_index[cid]
            first, last, _, _ = span(row)
            speech_end = integer(row.get("speech_end_sample"), "speech_end_sample")
            if not first <= speech_end <= last:
                raise ValueError("Speech endpoint outside source bounds")
            _same_numbers(row, joined, IDENTITY_NUMBERS)
            for field in IDENTITY_TEXT:
                if str(row.get(field)) != joined.get(field):
                    raise ValueError(f"CSV/diagnostic mismatch: {field}")
            if row.get("spanish_gemma") != joined.get("spanish_a"):
                raise ValueError("CSV/diagnostic translation mismatch")
            raw_path = row.get("audio_path")
            if not isinstance(raw_path, str) or not raw_path:
                raise ValueError("Final audio recording unavailable")
            path = (root / raw_path).resolve()
            if not path.is_relative_to(root / "stark_data" / "live_sessions" / session_id):
                raise ValueError("Stored WAV escapes owning session audio directory")
            if path in paths:
                raise ValueError("Stored WAV reused by different final identities")
            paths.add(path)
            audio = wav_identity(path)
            if audio["sha256"] != row.get("audio_sha256"):
                raise ValueError("Stored WAV SHA256 mismatch")
            duration = duration_check(row, audio["frames"] / audio["sample_rate"])
            # Both diagnostics and CSV intentionally round duration to two decimals.
            if (
                abs(number(row.get("utterance_dur"), "diagnostic utterance_dur") - duration["actual_pcm_s"])
                > 0.005000001
            ):
                raise ValueError("Diagnostic rounded duration disagrees with WAV")
            if abs(number(joined.get("utterance_dur"), "CSV utterance_dur") - duration["actual_pcm_s"]) > 0.005000001:
                raise ValueError("CSV rounded duration disagrees with WAV")
            checks.append({"chunk_id": cid, "utterance_id": row["utterance_id"], **duration, "audio": audio})
        except (ValueError, OSError, struct.error) as exc:
            errors.append(f"final[{cid}]: {exc}")
    for revision, row in preview_index.items():
        try:
            _provenance(row, metadata, session_id)
            duration_check(row, row.get("buffer_s"))
            rate = integer(row.get("processed_window_sample_rate"), "processed sample rate", 1)
            offset = integer(row.get("processed_window_offset_samples"), "processed window offset") / rate
            processed = number(row.get("processed_audio_s"), "processed_audio_s")
            if processed + offset > number(row.get("buffer_s"), "buffer_s") + 1 / rate:
                raise ValueError("Processed preview exceeds buffered interval")
        except ValueError as exc:
            errors.append(f"preview[{revision}]: {exc}")
    trace_available = (
        bool(trace.get("enabled"))
        and trace.get("schema_version") == 2
        and trace.get("clock") == "session_monotonic"
        and not trace.get("discarded_old_events")
    )
    if not trace_available:
        if require_trace:
            errors.append("physical_trace: missing, disabled, unsupported or truncated")
    else:
        starts: dict[tuple, dict] = {}
        finishes: dict[tuple, dict] = {}
        processed_results: dict[tuple, dict] = {}
        for event in trace.get("events", []):
            if event.get("event") == "partial_physical_result":
                try:
                    key = _trace_key({**event, "kind": "partial"})
                    if key in processed_results:
                        raise ValueError("Duplicate partial physical result")
                    processed_results[key] = event
                    _same_numbers(event, event, BOUNDS)
                    mode = event.get("incremental_mode")
                    if mode not in {"off", "rolling", "stream"}:
                        raise ValueError("Unknown incremental mode")
                    rate = integer(event.get("audio_sample_rate"), "processed rate", 1)
                    count = integer(event.get("processed_audio_samples"), "processed samples", 1)
                    offset = integer(event.get("rolling_offset_samples"), "rolling offset")
                    if mode == "off" and offset:
                        raise ValueError("Unexpected full-buffer offset")
                    if mode == "stream":
                        raise ValueError("Incremental stream processed source positions unavailable")
                    duration_check(event, (count + offset) / rate)
                except ValueError as exc:
                    errors.append(f"partial_physical_result: {exc}")
        for event in trace.get("events", []):
            if event.get("event") not in {"physical_stt_started", "physical_stt_finished"}:
                continue
            try:
                key = _trace_key(event)
                target = starts if event["event"].endswith("_started") else finishes
                if key in target:
                    raise ValueError(f"Duplicate physical event identity {key}")
                target[key] = event
                rate = integer(event.get("audio_sample_rate"), "physical audio rate", 1)
                count = integer(event.get("audio_samples"), "physical audio samples", 1)
                # Rolling/incremental variants need their own explicit processed interval.
                bounds = event.get("processed_source_bounds", event)
                offset = 0
                if key[0] == "partial":
                    processed = processed_results.get(key)
                    if processed is None:
                        raise ValueError("Missing keyed partial physical result")
                    offset = integer(processed.get("rolling_offset_samples"), "rolling offset") / rate
                    _same_numbers(event, processed, BOUNDS)
                detail = duration_check(bounds, count / rate + offset)
                if event.get("failed"):
                    raise ValueError("Physical STT failed")
                if target is finishes:
                    trace_checks.append({"kind": key[0], "utterance_id": key[1], "request_identity": key[2], **detail})
            except ValueError as exc:
                errors.append(f"physical_trace: {exc}")
        if set(processed_results) != {key for key in finishes if key[0] == "partial"}:
            errors.append("physical_trace: unmatched partial processed-result identities")
        if set(starts) != set(finishes):
            errors.append("physical_trace: unmatched started/finished identities")
        for key in starts.keys() & finishes.keys():
            try:
                _same_numbers(starts[key], finishes[key], (*BOUNDS, "audio_samples", "audio_sample_rate"))
                if number(finishes[key].get("at_ms"), "finish time") < number(starts[key].get("at_ms"), "start time"):
                    raise ValueError("Physical finish precedes start")
            except ValueError as exc:
                errors.append(f"physical_trace[{key}]: {exc}")
        for kind, rows, field in (("final", finals, "chunk_id"), ("partial", preview_index, "preview_revision")):
            for identity, row in rows.items():
                try:
                    key = kind, integer(row.get("utterance_id"), "utterance_id", 1), identity
                    if key not in finishes:
                        raise ValueError("No matching completed physical STT call")
                    _same_numbers(row, finishes[key], BOUNDS)
                    if kind == "partial":
                        event = processed_results.get(key)
                        if event is None:
                            raise ValueError("Missing keyed partial physical result")
                        physical = integer(event["processed_audio_samples"], "audio samples", 1) / integer(
                            event["audio_sample_rate"], "audio rate", 1
                        )
                        if integer(row.get("processed_window_offset_samples"), "preview offset") != integer(
                            event.get("rolling_offset_samples"), "trace offset"
                        ):
                            raise ValueError("Preview processed offset differs from physical call")
                        if abs(physical - number(row.get("processed_audio_s"), "processed_audio_s")) > 1 / 16000 + 1e-9:
                            raise ValueError("Emitted preview processed duration differs from physical call")
                except ValueError as exc:
                    errors.append(f"{kind}[{row.get(field)}] physical_join: {exc}")
    result.update(
        status="failed" if errors else "passed" if trace_available else "unassessable",
        final_count=len(finals),
        preview_count=len(preview_index),
        trace_available=trace_available,
        final_audio_verified=len(checks),
        completed_physical_calls=len(trace_checks),
    )
    return result
