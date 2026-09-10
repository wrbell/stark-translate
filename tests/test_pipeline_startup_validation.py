"""Invalid replay arguments cannot leave a phantom running session."""

import sys
from pathlib import Path
from unittest.mock import Mock

import pytest


@pytest.mark.parametrize(
    "selection,failure",
    [(selection, failure) for selection in ("cli", "environment") for failure in ("missing", "directory", "unreadable")]
    + [("cli", "invalid_path")],
)
def test_replay_read_failure_precedes_session_start(tmp_path, monkeypatch, capsys, selection, failure):
    import dry_run_ab as d
    from tools import session_lifecycle

    audio = tmp_path / "input.wav"
    if failure == "directory":
        audio.mkdir()
    elif failure == "unreadable":
        audio.write_bytes(b"audio")
        original_read = Path.read_bytes

        def read_bytes(path):
            if path == audio:
                raise PermissionError("test input is unreadable")
            return original_read(path)

        monkeypatch.setattr(Path, "read_bytes", read_bytes)
    selected_path = str(audio) if failure != "invalid_path" else "invalid\0path.wav"
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("STARK_AUDIO_SOURCE", raising=False)
    monkeypatch.delenv("STARK_AUDIO_FILE", raising=False)
    argv = ["dry_run_ab.py", "--session-id", "invalid_input_en"]
    if selection == "cli":
        argv += ["--audio-file", selected_path]
    else:
        monkeypatch.setenv("STARK_AUDIO_SOURCE", "file")
        monkeypatch.setenv("STARK_AUDIO_FILE", selected_path)
    monkeypatch.setattr(sys, "argv", argv)
    start = Mock(side_effect=AssertionError("Invalid input must not start a session"))
    monkeypatch.setattr(session_lifecycle, "start_session", start)
    monkeypatch.setattr(d, "PipelineHealth", Mock(side_effect=AssertionError("No health thread expected")))
    monkeypatch.setattr(d, "ThreadPoolExecutor", Mock(side_effect=AssertionError("No inference pool expected")))
    with pytest.raises(SystemExit) as error:
        d.main()
    assert error.value.code == 2
    assert "Cannot read audio input" in capsys.readouterr().err
    start.assert_not_called()
    assert not (tmp_path / "metrics").exists()


@pytest.mark.parametrize("flag", ["--silence-trigger", "--partial-interval"])
def test_invalid_timing_argument_precedes_session_start(tmp_path, monkeypatch, flag):
    import dry_run_ab as d
    from tools import session_lifecycle

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("STARK_AUDIO_SOURCE", raising=False)
    monkeypatch.setattr(sys, "argv", ["dry_run_ab.py", flag, "-1"])
    start = Mock(side_effect=AssertionError("Invalid arguments must not start a session"))
    monkeypatch.setattr(session_lifecycle, "start_session", start)
    with pytest.raises(SystemExit) as error:
        d.main()
    assert error.value.code == 2
    start.assert_not_called()
    assert not (tmp_path / "metrics").exists()
