"""Replay parsing and orchestration without invoking models or the pipeline."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import replay_bench as bench

FIXTURES = Path(__file__).parent / "fixtures" / "replay"


def test_report():
    report = bench.analyze_run(FIXTURES / "chunks.csv", FIXTURES / "partials.jsonl", FIXTURES / "pipeline.log")
    assert report["chunk_count"] == 4
    assert report["partial_count"] == 3
    assert report["metrics"]["true_e2e_ms"] == {"n": 3, "p50": 2000, "p95": 3000, "mean": 2000}
    assert report["metrics"]["partial_total_ms"] == {"n": 2, "p50": 140, "p95": 260, "mean": 200}
    assert report["metrics"]["gen_tokens_a"]["mean"] == pytest.approx(40 / 3)
    assert report["metrics"]["stt_latency_ms"]["n"] == 3
    assert report["metrics"]["latency_a_ms"]["n"] == 3
    assert report["marian_only_share"] == pytest.approx(1 / 3)
    assert report["marian_only_observations"] == 3
    assert report["special_token_outputs"] == 3
    assert report["special_token_counts"] == {"<turn|>": 2, "<|channel>": 2}
    assert report["overlap_pct"] == 67


def test_missing_optional_metrics_and_empty_values(tmp_path):
    csv_path = tmp_path / "minimal.csv"
    csv_path.write_text("chunk_id,stt_latency_ms\n1,\n")
    report = bench.analyze_run(csv_path, tmp_path / "missing.jsonl")
    assert report["metrics"]["stt_latency_ms"] == {"n": 0, "p50": None, "p95": None, "mean": None}
    assert "ttft_ms_a" not in report["metrics"]
    assert report["marian_only_share"] is None
    assert report["overlap_pct"] is None
    assert report["partial_count"] == 0
    assert report["special_token_outputs"] == 0


def test_percentiles_and_delta():
    assert bench._pct([], 95) == 0
    assert bench._pct([4, 1, 2, 3, 5], 50) == 3
    assert bench._pct([4, 1, 2, 3, 5], 95) == 5
    baseline = {"metrics": {"stt_latency_ms": bench.summarize([100])}, "chunk_count": 0}
    current = {"metrics": {"stt_latency_ms": bench.summarize([80])}, "chunk_count": 1}
    table = bench.delta_table(current, baseline)
    assert "| stt_latency_ms.p50 | 100.00 | 80.00 | -20.00 | -20.00 |" in table
    assert "| chunk_count | 0.00 | 1.00 | 1.00 | — |" in table


def test_configs_inline_and_json(tmp_path):
    path = tmp_path / "configs.json"
    path.write_text(json.dumps({"baseline": [], "mts": "--mts", "custom": ["--mlx-model", "local model"]}))
    assert bench.load_configs([]) == {"baseline": []}
    assert bench.load_configs([str(path)])["custom"] == ["--mlx-model", "local model"]
    assert bench.load_configs(["mts=--no-mts", "small=--gemma4-size e2b"], path)["mts"] == ["--no-mts"]
    with pytest.raises(ValueError, match="replay-managed"):
        bench.load_configs(["bad=--session-id other"])


@pytest.mark.parametrize("returncode", [0, 1])
def test_launch_and_report_use_real_flags_without_launching_pipeline(tmp_path, monkeypatch, returncode):
    seen = []

    def fake_run(command, **kwargs):
        seen.append(command)
        assert kwargs["cwd"] == bench.ROOT
        kwargs["stdout"].write((FIXTURES / "pipeline.log").read_text())
        (tmp_path / "ab_metrics_test.csv").write_text((FIXTURES / "chunks.csv").read_text())
        (tmp_path / "partials_test.jsonl").write_text((FIXTURES / "partials.jsonl").read_text())
        return SimpleNamespace(returncode=returncode)

    monkeypatch.setattr(bench.subprocess, "run", fake_run)
    clip = {"lang": "en", "path": "clip.wav"}
    if returncode:
        with pytest.raises(RuntimeError, match="Replay failed"):
            bench.run_replay(clip, tmp_path / "clip.wav", "test", ["--mts"], 2, tmp_path)
    else:
        report = bench.run_replay(clip, tmp_path / "clip.wav", "test", ["--mts"], 2, tmp_path)
        assert report["chunk_count"] == 4
    command = seen[0]
    assert command[:2] == [sys.executable, str(bench.ROOT / "dry_run_ab.py")]
    assert command[command.index("--ws-port") + 1] == "8769"
    assert command[command.index("--http-port") + 1] == "8770"
    assert command[command.index("--session-id") + 1] == "test"
    assert command[-1] == "--mts"
    assert json.loads((tmp_path / "replay_test.json").read_text())["returncode"] == returncode
    with pytest.raises(FileExistsError):
        bench.run_replay(clip, tmp_path / "clip.wav", "test", [], 2, tmp_path)
