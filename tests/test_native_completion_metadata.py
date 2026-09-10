"""Native operational counters persist without weakening required recording gates."""

import json

import pytest

from tools.session_lifecycle import SessionNotComplete, finish_session, session_status, start_session


@pytest.mark.parametrize("recording_ok", [False, True])
def test_native_final_log_counters_persist_under_recording_guards(tmp_path, recording_ok):
    session = "native_quality_en"
    run = start_session(tmp_path, session)
    diagnostics = tmp_path / "metrics" / f"diagnostics_{session}.jsonl"
    diagnostics.write_text('{"chunk_id": 1}\n')
    native = {
        "backend": "cpu",
        "model_sha256": "a" * 64,
        "native_revision": "b" * 40,
        "logging": {"queued": 0, "dropped": 7, "write_failures": 1, "read_failures": 0, "drain_incomplete": False},
    }
    finished = finish_session(
        tmp_path,
        session,
        run_id=run["run_id"],
        persistence={"ok": recording_ok, "pending": 0, "failed": 0 if recording_ok else 1},
        native_server=native,
    )
    persisted = json.loads((tmp_path / "metrics" / f"session_lifecycle_{session}.json").read_text())
    assert persisted["managed_llama"] == native == finished["managed_llama"]
    # Optional log drops do not fail successful required recording; failed
    # required recording still fails even when native model identity is present.
    assert session_status(tmp_path, session)["exportable"] is recording_ok
    assert persisted["status"] == ("completed" if recording_ok else "failed")
    assert persisted["exit_code"] == (0 if recording_ok else 1)
    before = (tmp_path / "metrics" / f"session_lifecycle_{session}.json").read_bytes()
    with pytest.raises(SessionNotComplete, match="ownership"):
        finish_session(tmp_path, session, run_id="foreign", native_server={"logging": {}})
    assert (tmp_path / "metrics" / f"session_lifecycle_{session}.json").read_bytes() == before
