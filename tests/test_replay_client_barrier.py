"""Replay connection waits must not consume audio or suppress Stop cancellation."""

import argparse
import ast
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tools.replay_client_barrier import ReplayClientBarrier, validate_replay_client_wait

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "seconds,source",
    [(15, "mic"), (15, "ws"), (-1, "file"), (61, "file"), (float("nan"), "file"), (float("inf"), "file")],
)
def test_invalid_or_nonfile_wait_rejected(seconds, source):
    with pytest.raises(ValueError, match="replay-wait-client-seconds"):
        validate_replay_client_wait(seconds, source)


def test_default_cli_disabled_and_file_only_validation_before_model_loading():
    tree = ast.parse((ROOT / "dry_run_ab.py").read_text())
    declaration = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "--replay-wait-client-seconds"
    )
    parser = argparse.ArgumentParser()
    eval(compile(ast.Expression(declaration), "runtime", "eval"), {"parser": parser})
    assert parser.parse_args([]).replay_wait_client_seconds == 0
    assert parser.parse_args(["--replay-wait-client-seconds", "15"]).replay_wait_client_seconds == 15
    validate_replay_client_wait(0, "mic")
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    calls = {
        node.func.id: node.lineno
        for node in ast.walk(main)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert calls["validate_replay_client_wait"] < calls["start_session"]


def test_disabled_is_zero_wait_and_does_not_inspect_live_clients():
    async def scenario():
        barrier = ReplayClientBarrier()
        clients = Mock(side_effect=AssertionError("default must not inspect clients"))
        await barrier.wait(clients)
        assert barrier.snapshot()["actual_wait_seconds"] == 0
        assert barrier.snapshot()["status"] == "disabled"

    asyncio.run(scenario())


def integration_scope(tmp_path, barrier, clients, starts):
    """Execute the actual runtime's pre-capture try body, without its model setup."""
    tree = ast.parse((ROOT / "dry_run_ab.py").read_text())
    main = next(node for node in tree.body if isinstance(node, ast.AsyncFunctionDef) and node.name == "main_async")
    block = next(
        node
        for node in main.body
        if isinstance(node, ast.Try)
        and any(
            isinstance(item, ast.Expr)
            and isinstance(item.value, ast.Await)
            and isinstance(item.value.value, ast.Call)
            and isinstance(item.value.value.func, ast.Name)
            and item.value.value.func.id == "audio_loop"
            for item in node.body
        )
    )
    function = ast.AsyncFunctionDef(
        name="run",
        args=ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]),
        body=block.body,
        decorator_list=[],
    )
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))

    async def audio_loop():
        starts.append(barrier._clock())
        # Metadata must be final before the file reader can establish its clock.
        saved = json.loads((tmp_path / "session_metadata_s.json").read_text())
        assert saved["replay_client_wait"]["status"] == "connected"

    scope = {
        "_replay_client_wait": barrier,
        "ws_clients": clients,
        "_health": Mock(),
        "_latency_trace": Mock(),
        "metadata": {},
        "Path": Path,
        "os": __import__("os"),
        "json": json,
        "DIAG_PATH": str(tmp_path / "diagnostics_s.jsonl"),
        "SESSION_ID": "s",
        "logger": Mock(),
        "audio_loop": audio_loop,
    }
    exec(compile(module, "actual_runtime_barrier", "exec"), scope)
    return SimpleNamespace(**scope)


def test_late_client_starts_capture_only_after_wait_and_records_metadata(tmp_path):
    async def scenario():
        now = [100.0]
        clients, starts = set(), []

        async def tick(seconds):
            assert not starts
            now[0] += seconds
            if now[0] >= 106:
                clients.add("socket")

        barrier = ReplayClientBarrier(15, clock=lambda: now[0], sleep=tick)
        scope = integration_scope(tmp_path, barrier, clients, starts)
        await scope.run()
        assert starts == [now[0]] and now[0] >= 106
        result = scope.metadata["replay_client_wait"]
        assert result["actual_wait_seconds"] == pytest.approx(now[0] - 100)
        assert result["connected_clients"] == 1 and result["visibility_confirmed"] is False
        assert scope._health.phase.call_args_list[0].args == ("waiting_for_display",)
        assert scope._health.phase.call_args.args == ("listening",)
        assert scope._latency_trace.record.call_args_list[-1].args == ("replay_client_wait_finished",)

    asyncio.run(scenario())


def test_timeout_fails_without_starting_capture_and_persists_outcome(tmp_path):
    async def scenario():
        now = [10.0]

        async def tick(seconds):
            now[0] += seconds

        barrier = ReplayClientBarrier(0.1, clock=lambda: now[0], sleep=tick)
        starts = []
        scope = integration_scope(tmp_path, barrier, set(), starts)
        with pytest.raises(TimeoutError, match="audio was not started"):
            await scope.run()
        assert not starts
        assert scope.metadata["replay_client_wait"]["status"] == "timed_out"
        assert scope.metadata["replay_client_wait"]["actual_wait_seconds"] == pytest.approx(0.1)

    asyncio.run(scenario())


def test_stop_cancellation_is_responsive_and_never_starts_capture(tmp_path):
    async def scenario():
        sleeping = asyncio.Event()

        async def tick(seconds):
            sleeping.set()
            await asyncio.Future()

        barrier = ReplayClientBarrier(15, sleep=tick)
        starts = []
        scope = integration_scope(tmp_path, barrier, set(), starts)
        task = asyncio.create_task(scope.run())
        await sleeping.wait()
        assert barrier.snapshot()["status"] == "waiting"
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not starts
        assert scope.metadata["replay_client_wait"]["status"] == "cancelled"

    asyncio.run(scenario())
