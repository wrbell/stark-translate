"""Exact-byte archival and compression determinism, without model dependencies."""

import hashlib
import json
import subprocess
import sys
import tarfile
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "docs/evaluation/mac_followup_20260910/normalized-integrity-pilot/archive_cohort.py.txt"
)


def test_compressed_archive_preserves_failed_evidence_and_is_deterministic(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    provenance = b'{"source": "frozen", "whitespace": true}\n'
    (source / "provenance.json").write_bytes(provenance)
    run = {
        "experiment": "negative",
        "session_id": "negative_en",
        "repeat": 0,
        "size": "e4b",
        "clip_id": "public_en",
        "source_identity": "revision",
        "clip": {"sha256": "audio"},
        "session_metadata": {"source_lang": "en"},
        "session_lifecycle": {"status": "failed"},
        "replay_integrity": {"status": "failed"},
        "observed": {
            "session_summary": {},
            "final_count": 0,
            "partial_count": 0,
            "endpoint_samples_ms": {},
        },
        "caption_quality": {"status": "unavailable"},
        "requested_runtime": {},
        "returncode": 1,
        "error": "retained failure",
        "completion_errors": ["failed"],
    }
    raw = json.dumps(run, indent=1).encode() + b"\n\n"
    (source / "negative.json").write_bytes(raw)
    log = tmp_path / "runner.log"
    log.write_bytes(b"negative failed\n")
    hashes = []
    for name in ("first", "second"):
        output = tmp_path / name
        subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--input",
                str(source),
                "--output",
                str(output),
                "--runner-log",
                str(log),
                "--compress",
            ],
            check=True,
        )
        assert not (output / "raw").exists()
        index = json.loads((output / "evidence-index.json").read_text())
        archived = next(f for f in index["files"] if f["path"] == "raw-evidence.tar.gz")
        payload = (output / archived["path"]).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == archived["sha256"]
        hashes.append(archived["sha256"])
        with tarfile.open(output / archived["path"], "r:gz") as tar:
            assert tar.extractfile("raw/negative.json").read() == raw
            assert tar.extractfile("raw/provenance.json").read() == provenance
            assert {m.name for m in tar.getmembers()} == {m["path"] for m in archived["members"]}
            for member in archived["members"]:
                data = tar.extractfile(member["path"]).read()
                assert hashlib.sha256(data).hexdigest() == member["sha256"]
                assert len(data) == member["size_bytes"]
        summary = json.loads((output / "run-summary.json").read_text())
        assert summary["runs"][0]["error"] == "retained failure"
        assert (output / "provenance.json").read_bytes() == provenance
    assert hashes[0] == hashes[1]


def test_archive_keeps_minimal_failures_long_names_and_production_csv(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "provenance.json").write_text("{}")
    session = "failed_" + "long_" * 25 + "en"
    (source / (session + ".json")).write_text(
        json.dumps(
            {
                "experiment": "baseline",
                "session_id": session,
                "repeat": 0,
                "size": "e4b",
                "clip_id": "en",
                "returncode": 1,
                "error": "startup failed before metadata",
            }
        )
    )
    metrics = tmp_path / "metrics"
    metrics.mkdir()
    csv = metrics / ("ab_metrics_" + session + ".csv")
    csv.write_bytes(b"chunk_id,stt_ms\n")
    log = tmp_path / "runner.log"
    log.write_text("failed\n")
    output = tmp_path / "archive"
    command = [
        sys.executable,
        str(SCRIPT),
        "--input",
        str(source),
        "--output",
        str(output),
        "--runner-log",
        str(log),
        "--metrics",
        str(metrics),
        "--compress",
    ]
    subprocess.run(command, check=True)
    with tarfile.open(output / "raw-evidence.tar.gz", "r:gz") as tar:
        assert tar.extractfile("raw/" + csv.name).read() == csv.read_bytes()
    summary = json.loads((output / "run-summary.json").read_text())["runs"][0]
    assert summary["error"] == "startup failed before metadata"
    assert summary["final_count"] is None and summary["source_lang"] is None
    original_index = (output / "evidence-index.json").read_bytes()
    assert subprocess.run(command, capture_output=True).returncode != 0
    assert (output / "evidence-index.json").read_bytes() == original_index
