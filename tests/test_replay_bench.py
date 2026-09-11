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
    assert report["metrics"]["partial_total_ms"] == {"n": 2, "p50": 200, "p95": 260, "mean": 200}
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


def _comparable_report(**values):
    return {
        "returncode": 0,
        "replay_speed": 1,
        "percentile_method": "median_p50_nearest_rank_p95",
        "clip": {"sha256": "a" * 64, "lang": "en"},
        "timing_schema_versions": ["2"],
        "timing_sources": ["replay_paced"],
        "session_lifecycle": {"pipeline_sha256": "c" * 64},
        **values,
    }


def test_percentiles_and_delta():
    assert bench._pct([], 95) == 0
    assert bench._pct([4, 1, 2, 3, 5], 50) == 3
    assert bench._pct([4, 1, 2, 3, 5], 95) == 5
    baseline = _comparable_report(metrics={"stt_latency_ms": bench.summarize([100])}, chunk_count=0)
    current = _comparable_report(metrics={"stt_latency_ms": bench.summarize([80])}, chunk_count=1)
    table = bench.delta_table(current, baseline)
    assert "| stt_latency_ms.p50 | 100.00 | 80.00 | -20.00 | -20.00 |" in table
    assert "| chunk_count | 0.00 | 1.00 | 1.00 | — |" in table


@pytest.mark.parametrize(
    "values,p50,p95",
    [
        ([2044], 2044, 2044),
        ([2204, 2044], 2124, 2204),
        ([2204, 1800, 2044], 2044, 2204),
        (list(range(20, 0, -1)), 10.5, 19),
    ],
)
def test_summary_percentiles_match_operator_and_comparison_reports(values, p50, p95):
    result = bench.summarize(values)
    assert result["p50"] == p50 and result["p95"] == p95


@pytest.mark.parametrize(
    "changes,reason",
    [
        ({"clip": {"sha256": "b" * 64, "lang": "en"}}, "audio sha256"),
        ({"clip": {"sha256": "a" * 64, "lang": "es"}}, "audio lang"),
        ({"clip": {"lang": "en"}}, "audio hash"),
        ({"timing_schema_versions": ["legacy"]}, "incompatible timing_schema"),
        ({"timing_sources": ["callback_receipt_estimate"]}, "incompatible timing_sources"),
        ({"replay_speed": 2}, "real-time"),
        ({"returncode": 1}, "failed"),
        ({"percentile_method": None}, "rebuild percentile"),
        ({"session_metadata": {"input_audio_sha256": "b" * 64}}, "contradicts"),
        ({"metric_definitions": {"e2e_latency_ms": "different meaning"}}, "metric_definitions"),
        ({"session_lifecycle": {}}, "unknown pipeline source"),
    ],
)
def test_delta_rejects_incompatible_or_unknown_baselines(changes, reason):
    with pytest.raises(ValueError, match=reason):
        bench.delta_table(_comparable_report(), _comparable_report(**changes))


def test_delta_separates_endpoints_and_identifies_intentional_configuration_changes():
    baseline = _comparable_report(
        metrics={"speech_end_to_final_ms": bench.summarize([50, 600])},
        speech_end_by_endpoint={"silence": bench.summarize([600]), "hard_cut": bench.summarize([50])},
        session_metadata={"model_a": "e4b"},
        session_lifecycle={"pipeline_sha256": "c" * 64},
    )
    current = _comparable_report(
        metrics={"speech_end_to_final_ms": bench.summarize([800])},
        speech_end_by_endpoint={"silence": bench.summarize([800])},
        session_metadata={"model_a": "e2b"},
        session_lifecycle={"pipeline_sha256": "c" * 64},
    )
    table = bench.delta_table(current, baseline)
    assert "Configuration fields changed: model_a." in table
    assert "speech_end_to_final_ms.p50" not in table
    assert "speech_end_to_final_ms[silence].p50 | 600.00 | 800.00 | 200.00 | 33.33" in table
    assert "speech_end_to_final_ms[hard_cut].p50 | 50.00 | — | — | —" in table
    current["session_lifecycle"]["pipeline_sha256"] = "d" * 64
    with pytest.raises(ValueError, match="different pipeline source"):
        bench.delta_table(current, baseline)


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

    monkeypatch.setattr(bench, "run_child", fake_run)
    monkeypatch.setattr(bench, "_free_port", lambda start: start)
    clip = {"lang": "en", "path": "clip.wav"}
    if returncode:
        with pytest.raises(RuntimeError, match="Replay failed"):
            bench.run_replay(clip, tmp_path / "clip.wav", "test", ["--mts"], 2, tmp_path)
    else:
        report = bench.run_replay(clip, tmp_path / "clip.wav", "test", ["--mts"], 2, tmp_path)
        assert report["chunk_count"] == 4
    command = seen[0]
    assert command[:2] == [sys.executable, str(bench.ROOT / "dry_run_ab.py")]
    ws_port = int(command[command.index("--ws-port") + 1])
    http_port = int(command[command.index("--http-port") + 1])
    assert ws_port >= 8865  # dynamically chosen free ports (8769 is squatted by another app on the dev Mac)
    assert http_port > ws_port
    assert command[command.index("--session-id") + 1] == "test"
    assert command[-1] == "--mts"
    assert json.loads((tmp_path / "replay_test.json").read_text())["returncode"] == returncode
    with pytest.raises(FileExistsError):
        bench.run_replay(clip, tmp_path / "clip.wav", "test", [], 2, tmp_path)


def test_replay_reports_effective_speed_and_retains_timeout_failure(tmp_path, monkeypatch):
    import subprocess

    monkeypatch.setattr(bench, "_free_port", lambda start: start)

    def timed_out(command, **kwargs):
        assert kwargs["timeout"] == 0.01
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(bench, "run_child", timed_out)
    with pytest.raises(RuntimeError, match="Replay failed"):
        bench.run_replay(
            {"lang": "en"}, tmp_path / "clip.wav", "timeout", ["--replay-speed=2"], 0, tmp_path, timeout_s=0.01
        )
    report = json.loads((tmp_path / "replay_timeout.json").read_text())
    assert report["replay_speed"] == 2 and not report["realtime_latency_eligible"]
    assert report["timed_out"] and "timeout" in report["error"]
    assert report["command"][-1] == "--replay-speed=2"


def test_bounded_worker_terminates_hanging_child(tmp_path):
    import subprocess
    import time

    started = time.monotonic()
    with (tmp_path / "worker.log").open("w") as log, pytest.raises(subprocess.TimeoutExpired):
        bench.run_child([sys.executable, "-c", "import time; time.sleep(60)"], cwd=tmp_path, stdout=log, timeout=0.05)
    assert time.monotonic() - started < 3


@pytest.fixture
def real_wavfile(monkeypatch):
    # conftest mocks scipy as well as ML dependencies. Temporarily import the
    # actual installed reader/writer; never substitute a fake PCM implementation.
    saved = {name: module for name, module in sys.modules.items() if name == "scipy" or name.startswith("scipy.")}
    for name in saved:
        del sys.modules[name]
    try:
        yield pytest.importorskip("scipy.io.wavfile", reason="clip preparation requires installed scipy.io.wavfile")
    finally:
        for name in list(sys.modules):
            if name == "scipy" or name.startswith("scipy."):
                del sys.modules[name]
        sys.modules.update(saved)


def _raw_wavs(tmp_path):
    import struct
    import wave

    raw = tmp_path / "raw"
    raw.mkdir()
    for name in ("Gospel_Message_one.wav", "Gospel_Message_two.wav", "spanish_test_2cor1.wav"):
        with wave.open(str(raw / name), "wb") as wav:
            wav.setnchannels(2)
            wav.setsampwidth(2)
            wav.setframerate(100)
            wav.writeframes(struct.pack("<400h", *range(400)))
    return raw


@pytest.mark.parametrize(
    "source,lang,offset", [("Gospel_Message_two.wav", "en", 0.5), ("spanish_test_2cor1.wav", "es", 0.0)]
)
def test_prepare_single_source_preserves_pcm_and_spanish_offset(tmp_path, source, lang, offset, real_wavfile):
    import wave

    raw = _raw_wavs(tmp_path)
    replay = tmp_path / "replay"
    manifest = bench.prepare_clips(raw, replay, 1, 0.5, source=source, clip_key="clip-1_A")
    assert manifest.name == "manifest_clip-1_A.json"
    entries = json.loads(manifest.read_text())["clips"]
    assert len(entries) == 1
    assert entries[0]["lang"] == lang and entries[0]["offset_s"] == offset
    assert entries[0]["duration"] == 1
    assert sorted(p.name for p in replay.iterdir()) == sorted([source, manifest.name])
    with wave.open(str(raw / source), "rb") as original, wave.open(str(replay / source), "rb") as cut:
        assert (cut.getnchannels(), cut.getsampwidth(), cut.getframerate()) == (2, 2, 100)
        original.setpos(int(offset * 100))
        assert cut.readframes(100) == original.readframes(100)


def test_prepare_default_manifest_and_selection_validation(tmp_path, real_wavfile):
    raw = _raw_wavs(tmp_path)
    replay = tmp_path / "replay"
    manifest = bench.prepare_clips(raw, replay, 1, 0.5)
    expected = []
    for name in ("Gospel_Message_one.wav", "Gospel_Message_two.wav", "spanish_test_2cor1.wav"):
        import hashlib

        expected.append(
            {
                "path": name,
                "sha256": hashlib.sha256((replay / name).read_bytes()).hexdigest(),
                "duration": 1.0,
                "lang": "es" if name.startswith("spanish") else "en",
                "offset_s": 0.0 if name.startswith("spanish") else 0.5,
            }
        )
    assert manifest.name == "manifest.json"
    assert manifest.read_bytes() == (json.dumps({"clips": expected}, indent=2) + "\n").encode()
    with pytest.raises(FileNotFoundError, match="missing"):
        bench.prepare_clips(raw, replay, source="missing.wav")
    for key in ("", "../bad", "has space"):
        with pytest.raises(ValueError, match="clip_key"):
            bench.prepare_clips(raw, replay, source="Gospel_Message_one.wav", clip_key=key)
    with pytest.raises(ValueError, match="single source"):
        bench.prepare_clips(raw, replay, clip_key="single")
    with pytest.raises(ValueError, match="file name"):
        bench.prepare_clips(raw, replay, source="../outside.wav")


def test_prepare_cli_passes_source_and_clip_key(tmp_path, monkeypatch, capsys):
    from unittest.mock import Mock

    prepare = Mock(return_value=tmp_path / "manifest_slice.json")
    monkeypatch.setattr(bench, "prepare_clips", prepare)
    bench.main(
        ["--prepare", "--source", "spanish_test_2cor1.wav", "--clip-key", "slice", "--seconds", "1", "--offset", "2"]
    )
    assert prepare.call_args.args[2:] == (1, 2)
    assert prepare.call_args.kwargs == {"source": "spanish_test_2cor1.wav", "clip_key": "slice"}
    assert "manifest_slice.json" in capsys.readouterr().out
