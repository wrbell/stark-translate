"""Real metadata/worker/shutdown code with stdlib fakes; no native audio or models."""

import ast
import io
import json
import struct
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import uuid
from collections import deque
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tools import capture_worker
from tools.capture_handoff import CaptureTransportSummary
from tools.capture_protocol import (
    CaptureStatus,
    capture_status,
    count,
    read_terminal_receipt,
    status_from_metadata,
    write_terminal_receipt,
)
from tools.persistence import PersistenceExecutor
from tools.pipeline_timing import capture_stamp

ROOT = Path(__file__).resolve().parents[1]
TOKEN = "a" * 32


def source_definitions(path, names, namespace, *, callback_status_only=False):
    """Execute actual definitions without the application's native import graph."""
    module = ast.parse((ROOT / path).read_text())
    nodes = [node for node in module.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    assert {node.name for node in nodes} == set(names)
    if callback_status_only:
        # Stop at the first resampling statement; exercise the real status branch.
        fn = nodes[0]
        end = next(i for i, node in enumerate(fn.body) if isinstance(node, ast.Assign))
        fn.body = fn.body[:end]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(ROOT / path), "exec"), namespace)
    return namespace


def isolated_class():
    # Only numpy and process-cleanup dependencies are absent. Neither is used by
    # the tested terminal path; cleanup is replaced before any fake process exit.
    namespace = dict(
        globals(),
        cleanup_children=lambda *args, **kwargs: None,
        struct=struct,
        threading=threading,
        time=time,
        uuid=uuid,
        deque=deque,
        replace=replace,
        count=count,
        capture_stamp=capture_stamp,
    )
    return source_definitions(
        "tools/isolated_audio.py", {"IsolatedInputStream", "worker_argv", "AudioCaptureError"}, namespace
    )["IsolatedInputStream"]


def receipt(**overrides):
    result = dict(
        schema_version=1,
        capture_token=TOKEN,
        stream_closed=True,
        stop_reason="requested_stop",
        callback_samples=64,
        callback_count=4,
        worker_fifo_dropped_samples=0,
        worker_fifo_dropped_callbacks=0,
        worker_fifo_admitted_samples=64,
        portaudio_input_overflow_callbacks=0,
        pipe_written_samples=48,
        worker_fifo_pending_samples=16,
        writer_unfinished_samples=0,
    )
    result.update(overrides)
    if "worker_fifo_admitted_samples" not in overrides:
        result["worker_fifo_admitted_samples"] = result["callback_samples"] - result["worker_fifo_dropped_samples"]
    return result


def stream_for_receipt(path, value):
    stream = isolated_class()(
        callback=lambda *args: None, samplerate=48000, channels=1, dtype="float32", blocksize=1536, device=None
    )
    stream._terminal_expected = True  # POSIX accounting contract, independently of CI platform.
    stream._terminal_status = "pending"
    stream._terminal_path, stream._capture_token = path, TOKEN
    if value is not None:
        write_terminal_receipt(path, value)
    return stream


class CaptureProtocolTests(unittest.TestCase):
    def test_fifo_and_native_loss_remain_independent(self):
        fifo, overflow, status = status_from_metadata(
            dict(
                dropped=1536,
                worker_fifo_dropped_samples=1536,
                status="input overflow",
                portaudio_input_overflow=True,
                portaudio_input_overflow_callbacks=2,
            ),
            0,
            0,
        )
        self.assertEqual((fifo, overflow, str(status)), (1536, 2, "capture_overflow:1536"))
        self.assertEqual(capture_status(status).input_overflow_callbacks, 2)
        self.assertEqual(status.portaudio_status, "input overflow")

    def test_native_flag_on_dropped_callback_reaches_next_admitted_frame(self):
        _, overflow, status = status_from_metadata(
            dict(dropped=16, status="", portaudio_input_overflow=False, portaudio_input_overflow_callbacks=1), 0, 0
        )
        self.assertEqual(overflow, 1)
        self.assertEqual(status.fifo_dropped_samples, 16)
        self.assertEqual(status.input_overflow_callbacks, 1)

    def test_legacy_and_direct_flags(self):
        self.assertEqual(str(capture_status("capture_overflow:16")), "capture_overflow:16")
        self.assertEqual(capture_status("input overflow, output underflow").input_overflow_callbacks, 1)
        self.assertEqual(capture_status(SimpleNamespace(input_overflow=True)).input_overflow_callbacks, 1)
        self.assertEqual(capture_status("output underflow").input_overflow_callbacks, 0)
        self.assertFalse(capture_status(None))
        self.assertEqual(status_from_metadata(dict(dropped=0, status=""), 0, 0)[:2], (0, 0))

    def test_invalid_wire_counters_fail_closed(self):
        cases = [
            dict(dropped=-1),
            dict(dropped=True),
            dict(dropped=0, worker_fifo_dropped_samples=16),
            dict(dropped=0, worker_fifo_dropped_samples=False),
            dict(dropped=0, status="capture_overflow:5"),
            dict(dropped=0, status=[]),
            dict(dropped=0, portaudio_input_overflow=1),
            dict(
                dropped=0, status="input overflow", portaudio_input_overflow=False, portaudio_input_overflow_callbacks=0
            ),
            dict(dropped=0, portaudio_input_overflow=True, portaudio_input_overflow_callbacks=0),
        ]
        for metadata in cases:
            with self.subTest(metadata=metadata), self.assertRaises(ValueError):
                status_from_metadata(metadata, 0, 0)
        with self.assertRaises(ValueError):
            status_from_metadata(dict(dropped=0), 1, 0)
        with self.assertRaises(ValueError):
            status_from_metadata(dict(dropped=0, portaudio_input_overflow_callbacks=0), 0, 1)

    def test_terminal_identity_counts_and_conservation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "terminal.json"
            write_terminal_receipt(path, receipt())
            self.assertEqual(
                read_terminal_receipt(path, TOKEN, observed_fifo=0, observed_overflows=0, observed_end=48), receipt()
            )
            cases = [
                dict(capture_token="b" * 32),
                dict(schema_version=True),
                dict(stream_closed=False),
                dict(worker_fifo_dropped_samples=16),
                dict(callback_count=False),
                dict(callback_samples=63),
                dict(worker_fifo_admitted_samples=48),
                dict(portaudio_input_overflow_callbacks=5),
                dict(worker_fifo_dropped_callbacks=1),
                dict(
                    callback_samples=100,
                    callback_count=100,
                    worker_fifo_dropped_samples=1,
                    worker_fifo_dropped_callbacks=99,
                    pipe_written_samples=99,
                    worker_fifo_pending_samples=0,
                ),
                dict(
                    callback_count=4,
                    worker_fifo_dropped_callbacks=4,
                    worker_fifo_dropped_samples=48,
                    pipe_written_samples=16,
                    worker_fifo_pending_samples=0,
                ),
            ]
            for fields in cases:
                write_terminal_receipt(path, receipt(**fields))
                with self.subTest(fields=fields), self.assertRaises(ValueError):
                    read_terminal_receipt(path, TOKEN, observed_fifo=0, observed_overflows=0, observed_end=48)
            write_terminal_receipt(path, receipt())
            with self.assertRaises(ValueError):
                read_terminal_receipt(path, TOKEN, observed_fifo=16, observed_overflows=0, observed_end=48)
            self.assertEqual([p.name for p in Path(directory).iterdir()], ["terminal.json"])

    def test_already_observed_samples_cannot_still_be_in_worker_fifo(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "terminal.json"
            write_terminal_receipt(path, receipt(pipe_written_samples=0, worker_fifo_pending_samples=64))
            with self.assertRaises(ValueError):
                read_terminal_receipt(path, TOKEN, observed_fifo=0, observed_overflows=0, observed_end=48)
            # A complete flush can reach the reader before writer bookkeeping.
            write_terminal_receipt(path, receipt(pipe_written_samples=32, writer_unfinished_samples=16))
            final = read_terminal_receipt(path, TOKEN, observed_fifo=0, observed_overflows=0, observed_end=48)
            self.assertEqual(final["writer_unfinished_samples"], 16)


class CaptureWorkerTests(unittest.TestCase):
    def run_worker(self, path, *, callbacks, native_overflow=False, complete_write=False):
        recorded = []

        class NativeStream:
            closed = False

            def __init__(self, callback, **options):
                self.callback = callback
                assert "terminal_path" not in options and "capture_token" not in options

            def __enter__(self):
                for index in range(callbacks):
                    self.callback(
                        SimpleNamespace(tobytes=lambda: bytes(64)),
                        16,
                        SimpleNamespace(inputBufferAdcTime=1, currentTime=1.032),
                        "input overflow" if native_overflow and index == callbacks - 1 else "",
                    )
                return self

            def __exit__(self, *args):
                self.closed = True

        def write(stream, metadata, payload):
            recorded.append(metadata)
            if not complete_write or len(recorded) == 2:
                capture_worker.request_capture_stop(None, None)

        options = dict(device=None, channels=1, terminal_path=str(path), capture_token=TOKEN)
        with (
            patch.dict(
                sys.modules, {"numpy": SimpleNamespace(), "sounddevice": SimpleNamespace(InputStream=NativeStream)}
            ),
            patch.object(sys, "argv", ["capture_worker", json.dumps(options)]),
            patch.object(sys, "stdout", SimpleNamespace(buffer=io.BytesIO())),
            patch.object(capture_worker, "write_frame", write),
            patch.object(capture_worker.signal, "signal", return_value=0),
        ):
            capture_worker.main()
        return recorded, read_terminal_receipt(path, TOKEN, observed_fifo=0, observed_overflows=0, observed_end=0)

    def test_terminal_catches_fifo_and_native_loss_without_a_later_admitted_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            recorded, final = self.run_worker(Path(directory) / "terminal.json", callbacks=33, native_overflow=True)
        self.assertEqual(recorded[0]["dropped"], 0)  # parent cannot yet observe the later FIFO loss
        self.assertEqual(recorded[0]["portaudio_input_overflow_callbacks"], 0)
        self.assertEqual(final["worker_fifo_dropped_samples"], 16)
        self.assertEqual(final["worker_fifo_dropped_callbacks"], 1)
        self.assertEqual(final["portaudio_input_overflow_callbacks"], 1)
        self.assertEqual(final["worker_fifo_pending_samples"], 31 * 16)
        self.assertEqual(final["writer_unfinished_samples"], 16)
        self.assertTrue(final["stream_closed"])

    def test_normal_stop_retains_pending_and_completed_pipe_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            _, final = self.run_worker(Path(directory) / "terminal.json", callbacks=32, complete_write=True)
        self.assertEqual(final["worker_fifo_dropped_samples"], 0)
        self.assertEqual(final["pipe_written_samples"], 16)
        self.assertEqual(final["worker_fifo_pending_samples"], 30 * 16)
        self.assertEqual(final["writer_unfinished_samples"], 16)


class CaptureTerminalTests(unittest.TestCase):
    def test_real_framed_reader_keeps_both_flags_and_original_source_coordinates(self):
        wire, seen = io.BytesIO(), []
        for index, dropped, overflow in ((0, 0, 0), (2, 16, 1)):
            capture_worker.write_frame(
                wire,
                dict(
                    frames=16,
                    channels=1,
                    sample_start=index * 16,
                    dropped=dropped,
                    worker_fifo_dropped_samples=dropped,
                    portaudio_input_overflow=False,
                    portaudio_input_overflow_callbacks=overflow,
                    status="",
                    received=time.perf_counter(),
                    inputBufferAdcTime=1,
                    currentTime=1.032,
                ),
                bytes(64),
            )
        wire.seek(0)
        cls = isolated_class()
        cls._read.__globals__["np"] = SimpleNamespace(
            frombuffer=lambda *args, **kwargs: SimpleNamespace(reshape=lambda *shape: None)
        )

        def callback(samples, frames, stamp, status):
            seen.append((stamp, status))
            if len(seen) == 2:
                stream._stop.set()

        stream = cls(
            callback=callback, samplerate=48000, channels=1, dtype="float32", blocksize=16, device=None, argv=["fake"]
        )
        stream.sample_offset = 1000  # actual parent pause/resume offset applied to fresh child coordinates
        stream._proc = SimpleNamespace(stdout=wire)
        stream._read()
        self.assertIsNone(stream.error)
        self.assertEqual([(stamp.sample_start, stamp.sample_end) for stamp, _ in seen], [(1000, 1016), (1032, 1048)])
        self.assertIsNone(seen[0][1])
        self.assertEqual(str(seen[1][1]), "capture_overflow:16")
        self.assertEqual(seen[1][1].input_overflow_callbacks, 1)
        self.assertEqual(stream._observed_end, 48)  # receipt remains in this child's local coordinates
        self.assertEqual(stream.capture_snapshot()["upstream_dropped_samples"], 16)
        self.assertEqual(
            stream.capture_snapshot()["source_gaps"], [dict(sample_start=1016, sample_end=1032, sample_rate=48000)]
        )

    def test_verified_pending_stop_is_not_a_loss_and_finish_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            stream = stream_for_receipt(Path(directory) / "terminal.json", receipt())
            stream._finish_terminal_accounting()
            stream._finish_terminal_accounting()
            snap = stream.capture_snapshot()
        self.assertEqual(snap["terminal_accounting"]["status"], "verified")
        self.assertEqual(snap["capture_completeness_failures"], [])
        self.assertEqual(snap["upstream_dropped_samples"], 0)
        self.assertIsNone(snap["portaudio_input_overflow_lost_samples"])

    def test_tail_loss_preserves_legacy_observed_counter_and_marks_required_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            stream = stream_for_receipt(
                Path(directory) / "terminal.json",
                receipt(
                    worker_fifo_dropped_samples=16,
                    worker_fifo_dropped_callbacks=1,
                    worker_fifo_pending_samples=0,
                    portaudio_input_overflow_callbacks=1,
                ),
            )
            stream._finish_terminal_accounting()
            snap = stream.capture_snapshot()
        self.assertEqual(snap["upstream_dropped_samples"], 0)
        self.assertEqual(snap["terminal_accounting"]["receipt"]["worker_fifo_dropped_samples"], 16)
        self.assertEqual(
            snap["capture_completeness_failures"],
            ["worker_fifo_samples_dropped_at_close", "portaudio_input_overflow_at_close"],
        )

    def test_expected_missing_receipt_is_unverified_not_a_hardware_diagnosis(self):
        with tempfile.TemporaryDirectory() as directory:
            stream = stream_for_receipt(Path(directory) / "missing.json", None)
            stream._finish_terminal_accounting()
        self.assertEqual(stream._terminal_status, "unverified")
        self.assertIsNone(stream._terminal_receipt)
        self.assertEqual(stream._accounting_failures, ["capture_terminal_accounting_unverified"])

    def test_windows_and_fake_workers_are_explicitly_unavailable(self):
        cls = isolated_class()
        kwargs = dict(
            callback=lambda *args: None, samplerate=48000, channels=1, dtype="float32", blocksize=1536, device=None
        )
        with patch.object(sys, "platform", "win32"):
            stream = cls(**kwargs)
        stream._finish_terminal_accounting()
        self.assertEqual(stream._terminal_status, "unsupported_platform")
        self.assertEqual(stream._accounting_failures, [])
        stream = cls(**kwargs, argv=["fake"])
        stream._finish_terminal_accounting()
        self.assertEqual(stream._terminal_status, "unavailable_legacy_worker")
        self.assertEqual(stream.argv, ["fake"])
        self.assertEqual(stream._accounting_failures, [])

    def test_cleanup_records_unverified_even_if_process_wait_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            stream = stream_for_receipt(Path(directory) / "terminal.json", receipt())
            cleaned = []
            stream._terminal_directory = SimpleNamespace(cleanup=lambda: cleaned.append(True))
            stream._proc = SimpleNamespace(
                poll=lambda: None,
                terminate=lambda: None,
                kill=lambda: None,
                wait=lambda **kw: (_ for _ in ()).throw(subprocess.TimeoutExpired("fake", 2)),
                stdout=io.BytesIO(),
                pid=123,
            )
            with self.assertRaises(subprocess.TimeoutExpired):
                stream.__exit__()
        self.assertEqual(cleaned, [True])
        self.assertEqual(stream._terminal_status, "unverified")

    def test_active_reader_prevents_false_verified_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            stream = stream_for_receipt(Path(directory) / "terminal.json", receipt())
            stream._reader = SimpleNamespace(is_alive=lambda: True)
            stream._finish_terminal_accounting()
        self.assertEqual(stream._terminal_status, "unverified")


class CapturePersistenceTests(unittest.TestCase):
    def test_parsed_loss_status_never_consumed_still_fails_at_close(self):
        wire, pending = io.BytesIO(), []
        capture_worker.write_frame(
            wire,
            dict(
                frames=16,
                channels=1,
                sample_start=16,
                dropped=16,
                status="input overflow",
                portaudio_input_overflow=True,
                portaudio_input_overflow_callbacks=1,
                received=time.perf_counter(),
                inputBufferAdcTime=1,
                currentTime=1.032,
            ),
            bytes(64),
        )
        wire.seek(0)
        cls = isolated_class()
        cls._read.__globals__["np"] = SimpleNamespace(
            frombuffer=lambda *args, **kwargs: SimpleNamespace(reshape=lambda *shape: None)
        )

        def handoff(*frame):
            pending.append(frame)  # deliberately never passed to audio_callback
            stream._stop.set()

        stream = cls(
            callback=handoff, samplerate=48000, channels=1, dtype="float32", blocksize=16, device=None, argv=["fake"]
        )
        stream._proc = SimpleNamespace(stdout=wire)
        stream._read()
        self.assertIsNone(stream.error)
        self.assertEqual(len(pending), 1)
        with PersistenceExecutor() as ledger:
            namespace = dict(_io_pool=ledger, _health=None, _capture_transport=CaptureTransportSummary())
            hook = source_definitions("dry_run_ab.py", {"_record_capture_transport"}, namespace)[
                "_record_capture_transport"
            ]
            self.assertTrue(ledger.snapshot()["ok"])
            hook(stream, SimpleNamespace(snapshot=lambda: dict(dropped_frames=0)))
            failures = ledger.snapshot()["failures"]
        self.assertEqual(
            {item["code"] for item in failures},
            {"worker_fifo_samples_dropped_observed_at_close", "portaudio_input_overflow_observed_at_close"},
        )

    def test_verified_zero_loss_pending_tail_does_not_fail_required_ledger(self):
        with tempfile.TemporaryDirectory() as directory, PersistenceExecutor() as ledger:
            stream = stream_for_receipt(Path(directory) / "terminal.json", receipt())
            stream._finish_terminal_accounting()
            namespace = dict(_io_pool=ledger, _health=None, _capture_transport=CaptureTransportSummary())
            hook = source_definitions("dry_run_ab.py", {"_record_capture_transport"}, namespace)[
                "_record_capture_transport"
            ]
            hook(stream, SimpleNamespace(snapshot=lambda: dict(dropped_frames=0)))
            self.assertTrue(ledger.snapshot()["ok"])

    def test_observed_native_overflow_and_fifo_each_fail_real_required_ledger(self):
        with PersistenceExecutor() as ledger:
            namespace = dict(_io_pool=ledger, _health=None, sys=sys)
            callback = source_definitions("dry_run_ab.py", {"audio_callback"}, namespace, callback_status_only=True)[
                "audio_callback"
            ]
            with patch.object(sys, "stderr", io.StringIO()):
                callback(
                    None,
                    None,
                    None,
                    CaptureStatus("input overflow", fifo_dropped_samples=16, input_overflow_callbacks=1),
                )
            result = ledger.snapshot()
        self.assertFalse(result["ok"])
        self.assertEqual(
            {failure["code"] for failure in result["failures"]}, {"samples_dropped", "portaudio_input_overflow"}
        )

    def test_terminal_hook_marks_missing_accounting_in_required_ledger(self):
        with tempfile.TemporaryDirectory() as directory, PersistenceExecutor() as ledger:
            stream = stream_for_receipt(Path(directory) / "missing.json", None)
            stream._finish_terminal_accounting()
            summary = CaptureTransportSummary()
            namespace = dict(_io_pool=ledger, _health=None, _capture_transport=summary)
            hook = source_definitions("dry_run_ab.py", {"_record_capture_transport"}, namespace)[
                "_record_capture_transport"
            ]
            hook(stream, SimpleNamespace(snapshot=lambda: dict(dropped_frames=0)))
            self.assertFalse(ledger.snapshot()["ok"])
            self.assertEqual(ledger.snapshot()["failures"][0]["code"], "capture_terminal_accounting_unverified")
            self.assertEqual(summary.snapshot()["terminal_accounting_unverified_segments"], 1)

    def test_pause_resume_totals_and_loss_flags_survive_bounded_details(self):
        summary = CaptureTransportSummary()
        handoff = SimpleNamespace(snapshot=lambda: dict(dropped_frames=0))
        for index in range(20):
            terminal = receipt(
                worker_fifo_dropped_samples=16 if index == 0 else 0,
                portaudio_input_overflow_callbacks=1 if index == 0 else 0,
            )
            stream = SimpleNamespace(
                capture_snapshot=lambda terminal=terminal: dict(
                    upstream_dropped_samples=0,
                    portaudio_input_overflow_callbacks_observed=0,
                    terminal_accounting=dict(status="verified", receipt=terminal),
                )
            )
            summary.record(stream, handoff)
        snap = summary.snapshot()
        self.assertEqual(snap["worker_fifo_dropped_samples_terminal"], 16)
        self.assertEqual(snap["portaudio_input_overflow_callbacks_terminal"], 1)
        self.assertEqual(snap["upstream_dropped_samples"], 0)
        self.assertTrue(snap["segments_truncated"])
        self.assertEqual(len(snap["segments"]), 16)
        summary.record(SimpleNamespace(), handoff)
        snap = summary.snapshot()
        self.assertIsNone(snap["worker_fifo_dropped_samples_terminal"])
        self.assertIsNone(snap["portaudio_input_overflow_callbacks_terminal"])
        self.assertIsNone(snap["portaudio_input_overflow_callbacks_observed"])
        self.assertTrue(snap["worker_fifo_loss_detected"])
        self.assertTrue(snap["portaudio_input_overflow_detected"])

    def test_file_and_legacy_streams_do_not_claim_terminal_zero(self):
        summary = CaptureTransportSummary()
        handoff = SimpleNamespace(snapshot=lambda: dict(dropped_frames=0))
        summary.record(type("FileAudioStream", (), {})(), handoff)
        summary.record(SimpleNamespace(capture_snapshot=lambda: dict(upstream_dropped_samples=16)), handoff)
        snap = summary.snapshot()
        self.assertEqual(snap["terminal_accounting_not_applicable_segments"], 1)
        self.assertEqual(snap["terminal_accounting_unavailable_segments"], 1)
        self.assertEqual(snap["upstream_dropped_samples"], 16)
        self.assertIsNone(snap["worker_fifo_dropped_samples_terminal"])


if __name__ == "__main__":
    unittest.main()
