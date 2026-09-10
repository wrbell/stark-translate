"""Real RIFF bytes and production record shapes; no audio/model libraries."""

import copy
import csv
import hashlib
import struct

import pytest

from tools import replay_integrity as ri


def write_float_wav(path, frames):
    path.parent.mkdir(parents=True, exist_ok=True)
    fmt = struct.pack("<HHIIHH", 3, 1, 16000, 64000, 4, 32)
    data = struct.pack("<f", 0.125) * frames
    chunks = b"fmt " + struct.pack("<I", len(fmt)) + fmt + b"data" + struct.pack("<I", len(data)) + data
    path.write_bytes(b"RIFF" + struct.pack("<I", len(chunks) + 4) + b"WAVE" + chunks)
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def cohort(tmp_path):
    sid = "integrity_en"
    root = tmp_path
    metadata = dict(
        session_id=sid,
        audio_source="file",
        replay_speed=1,
        session_kind="replay",
        timing_schema_version=2,
        input_audio_sha256="a" * 64,
        source_lang="en",
        target_lang="es",
    )
    audio = root / "stark_data" / "live_sessions" / sid / "chunk_0002.wav"
    bounds = dict(sample_start=48000, sample_end=96000, sample_rate=48000, padding_samples=0, speech_end_sample=90000)
    final = dict(
        metadata,
        **bounds,
        chunk_id=2,
        utterance_id=7,
        timing_source="replay_realtime",
        timestamp="now",
        english="Hello",
        spanish_gemma="Hola",
        endpoint_reason="silence",
        audio_path=str(audio.relative_to(root)),
        audio_sha256=write_float_wav(audio, 16000),
        utterance_dur=1.0,
    )
    preview = dict(
        metadata,
        **bounds,
        utterance_id=7,
        preview_revision=3,
        timing_source="replay_realtime",
        buffer_s=1.0,
        processed_audio_s=1.0,
        processed_window_offset_samples=0,
        processed_window_sample_rate=16000,
    )
    base = dict(bounds, utterance_id=7, audio_samples=16000, audio_sample_rate=16000)
    events = []
    for kind, identity in [("final", dict(chunk_id=2)), ("partial", dict(request_sequence=3))]:
        for suffix, at in [("started", 1000), ("finished", 1100)]:
            events.append(dict(base, **identity, kind=kind, event="physical_stt_" + suffix, at_ms=at, failed=False))
    events.append(
        dict(
            bounds,
            event="partial_physical_result",
            utterance_id=7,
            request_sequence=3,
            processed_audio_samples=16000,
            audio_sample_rate=16000,
            rolling_offset_samples=0,
            incremental_mode="off",
        )
    )
    trace = dict(schema_version=2, clock="session_monotonic", enabled=True, discarded_old_events=0, events=events)
    args = dict(
        data_root=root, session_id=sid, metadata=metadata, diagnostic_finals=[final], partials=[preview], trace=trace
    )
    save_csv(args)
    return args


def save_csv(args):
    path = args["data_root"] / "metrics" / ("ab_metrics_" + args["session_id"] + ".csv")
    path.parent.mkdir(exist_ok=True)
    rows = [{**r, "spanish_a": r["spanish_gemma"]} for r in args["diagnostic_finals"]]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_production_shapes_join_distinct_chunk_and_utterance(cohort):
    report = ri.audit_replay_integrity(**cohort)
    assert report["errors"] == []
    assert report["status"] == "passed"
    assert report["final_audio_verified"] == 1
    assert report["completed_physical_calls"] == 2
    assert report["waveform_equality"] == "not_checked"
    assert report["finals"][0]["utterance_id"] == 7


def test_virtual_padding_is_pcm_but_not_real_source(cohort):
    for row in [*cohort["diagnostic_finals"], *cohort["partials"], *cohort["trace"]["events"]]:
        row["sample_end"] = 94500
        row["padding_samples"] = 1500
    save_csv(cohort)
    assert ri.audit_replay_integrity(**cohort)["status"] == "passed"


def test_retained_timeline_short_gap_fails_even_matching_csv(cohort):
    # Historical bug: skipped 32 ms kept in source span while stored PCM is short.
    for row in cohort["diagnostic_finals"]:
        row["sample_end"] += 1536
    save_csv(cohort)
    report = ri.audit_replay_integrity(**cohort)
    assert report["status"] == "failed"
    assert any("PCM/source span mismatch" in e for e in report["errors"])


def test_duration_rounding_not_false_failure(cohort):
    row = cohort["diagnostic_finals"][0]
    row["sample_end"] = 96003
    row["audio_sha256"] = write_float_wav(cohort["data_root"] / row["audio_path"], 16001)
    for e in cohort["trace"]["events"]:
        if e.get("kind") == "final":
            e["sample_end"] = 96003
            e["audio_samples"] = 16001
    save_csv(cohort)
    assert ri.audit_replay_integrity(**cohort)["status"] == "passed"


@pytest.mark.parametrize("field,value", [("utterance_id", 8), ("english", "Wrong"), ("sample_end", 96003)])
def test_csv_diagnostic_identity_mismatch(cohort, field, value):
    cohort["diagnostic_finals"][0][field] = value
    assert any("mismatch" in e.lower() for e in ri.audit_replay_integrity(**cohort)["errors"])


def test_duplicate_or_missing_final_id_fails(cohort):
    cohort["diagnostic_finals"].append(copy.deepcopy(cohort["diagnostic_finals"][0]))
    assert ri.audit_replay_integrity(**cohort)["status"] == "failed"


def test_hash_mismatch_and_truncated_wav_detected(cohort):
    row = cohort["diagnostic_finals"][0]
    path = cohort["data_root"] / row["audio_path"]
    data = bytearray(path.read_bytes())
    data[-1] ^= 1
    path.write_bytes(data)
    assert any("SHA256 mismatch" in e for e in ri.audit_replay_integrity(**cohort)["errors"])
    path.write_bytes(data[:-1])
    assert any("Truncated" in e for e in ri.audit_replay_integrity(**cohort)["errors"])


def test_audio_path_escape_rejected(cohort, tmp_path):
    cohort["diagnostic_finals"][0]["audio_path"] = str(tmp_path / "outside.wav")
    assert any("escapes" in e for e in ri.audit_replay_integrity(**cohort)["errors"])


def test_preview_buffer_and_physical_trace_disagreement(cohort):
    cohort["partials"][0]["buffer_s"] = 0.968
    assert any("PCM/source span mismatch" in e for e in ri.audit_replay_integrity(**cohort)["errors"])
    cohort["partials"][0]["buffer_s"] = 1.0
    cohort["trace"]["events"][-1]["processed_audio_samples"] = 15488
    assert ri.audit_replay_integrity(**cohort)["status"] == "failed"


def test_lost_physical_finish_cannot_be_hidden_by_emitted_output(cohort):
    cohort["trace"]["events"] = [
        e for e in cohort["trace"]["events"] if not (e.get("kind") == "partial" and e["event"].endswith("finished"))
    ]
    report = ri.audit_replay_integrity(**cohort)
    assert report["status"] == "failed"
    assert any("No matching completed physical" in e for e in report["errors"])


def test_trace_overflow_missing_identity_and_unavailable_never_pass(cohort):
    cohort["trace"]["discarded_old_events"] = 1
    assert ri.audit_replay_integrity(**cohort)["status"] == "failed"
    assert ri.audit_replay_integrity(**cohort, require_trace=False)["status"] == "unassessable"
    cohort["trace"]["discarded_old_events"] = 0
    del cohort["trace"]["events"][0]["utterance_id"]
    assert ri.audit_replay_integrity(**cohort)["status"] == "failed"


def test_rolling_offset_uses_processed_rate_and_stream_missing_positions_fails(cohort):
    cohort["partials"][0].update(processed_audio_s=0.5, processed_window_offset_samples=8000)
    for row in cohort["trace"]["events"]:
        if row.get("kind") == "partial":
            row["audio_samples"] = 8000
        elif row["event"] == "partial_physical_result":
            row.update(processed_audio_samples=8000, rolling_offset_samples=8000, incremental_mode="rolling")
    assert ri.audit_replay_integrity(**cohort)["status"] == "passed"
    cohort["trace"]["events"][-1]["incremental_mode"] = "stream"
    assert any("source positions unavailable" in e for e in ri.audit_replay_integrity(**cohort)["errors"])
