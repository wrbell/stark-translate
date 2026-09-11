"""Review small archive metadata without reading archives, audio, or model weights."""

import argparse
import datetime
import hashlib
import json
from pathlib import Path, PurePosixPath

SMALL_FILE_LIMIT = 262144


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def json_file(path):
    require(path.stat().st_size <= SMALL_FILE_LIMIT, f"Not small metadata: {path}")
    return json.loads(path.read_bytes())


def safe_path(name):
    path = PurePosixPath(name)
    require(not path.is_absolute() and ".." not in path.parts, f"Unsafe indexed path: {name}")
    return path


def review(root):
    index_path = root / "archive-index.json"
    index = json_file(index_path)
    interpretation = json_file(root / "interpretation.json")
    inputs = json_file(root / "input-inventory.json")
    entries = {row["path"]: row for row in index["files"]}
    require(len(entries) == len(index["files"]), "Duplicate archive index entry")
    checks, deferred = [], []
    for name, row in entries.items():
        path = root / safe_path(name)
        require(path.resolve().is_relative_to(root.resolve()), "Indexed path escapes archive root")
        require(path.is_file() and path.stat().st_size == row["size_bytes"], f"Indexed size mismatch: {name}")
        require(len(row["sha256"]) == 64, f"Invalid recorded hash: {name}")
        if name.endswith(".tar.gz") or path.stat().st_size > SMALL_FILE_LIMIT:
            deferred.append(
                {
                    "path": name,
                    "size_bytes": row["size_bytes"],
                    "recorded_sha256": row["sha256"],
                    "reason": "Compressed archive or larger raw payload: stat only; no read/hash/decompression during active model queue",
                }
            )
        else:
            require(sha(path.read_bytes()) == row["sha256"], f"Indexed small-file hash mismatch: {name}")
            checks.append(name)
    cohorts = []
    for name, short_name in (("standard-screen-normalized-v1", "v1"), ("measurement-pilot-v4", "v4")):
        folder = root / name
        runs = json_file(folder / "run-summary.json")["runs"]
        evidence = json_file(folder / "evidence-index.json")
        provenance = json_file(folder / "provenance.json")
        protocol = json_file(folder / "protocol.json")
        expected = interpretation[short_name]
        inventory = next(row for row in inputs["cohorts"] if row["cohort"] == name)
        source = interpretation["source_verification"][name]
        require(
            len(runs) == expected["persisted_run_results"] == inventory["persisted_run_json_count"],
            "Cohort counts disagree",
        )
        require(
            source["code_sha"] == provenance["source"]["code_sha"] == inventory["recorded_source_revision"],
            "Recorded source revisions disagree",
        )
        require(
            source["recorded_code_files_verified"] == len(provenance["source"]["all_code_sha256"]),
            "Recorded source-file count disagrees",
        )
        require(provenance["source"]["tracked_diff_sha256"] == sha(b""), "Unexpected dirty-source provenance")
        require(provenance["source_identity"] == inventory["source_identity"], "Source identity disagrees")
        require(
            protocol == provenance["spec"]
            and sha((folder / "protocol.json").read_bytes()) == provenance["spec_sha256"],
            "Protocol differs from recorded frozen bytes",
        )
        require(
            all(row["source_identity"] == provenance["source_identity"] for row in runs),
            "Mixed source identity in summary",
        )
        require(all(row["p95_claim_eligible"] is False for row in runs), "Unexpected p95 claim in diagnostic archive")
        require(
            evidence["audio_included"] is False and evidence["raw_payloads_unmodified"] is True,
            "Unexpected producer evidence policy",
        )
        for row in evidence["files"]:
            top = entries[str(PurePosixPath(name) / safe_path(row["path"]))]
            require(all(row[k] == top[k] for k in ("size_bytes", "sha256")), "Nested/top-level index mismatch")
        raw = next(row for row in evidence["files"] if row["path"] == "raw-evidence.tar.gz")
        members = {row["path"]: row for row in raw["members"]}
        require(len(members) == len(raw["members"]), "Duplicate recorded raw archive member")
        for path in members:
            require(safe_path(path).parts[0] == "raw", "Recorded raw member is outside raw/")
        expected_members = {"raw/provenance.json"}
        for row in runs:
            sid = row["session_id"]
            expected_members.update({f"raw/{sid}.json", f"raw/ab_metrics_{sid}.csv", f"raw/session_{sid}.log"})
        require(set(members) == expected_members, "Summary and recorded raw member inventory disagree")
        require(
            all(
                members["raw/provenance.json"][k] == entries[name + "/provenance.json"][k]
                for k in ("size_bytes", "sha256")
            ),
            "Copied/raw provenance identity differs",
        )
        for row in inventory["files"]:
            member = members["raw/" + Path(row["path"]).name]
            require(row["size_bytes"] == member["size_bytes"], "Input inventory and raw member sizes differ")
        run_ids = {row["session_id"] for row in runs}
        require(len(run_ids) == len(runs), "Duplicate summarized run")
        lines = (folder / "runner.log").read_text().splitlines()
        started = [line.removeprefix("Starting ") for line in lines if line.startswith("Starting ")]
        finished = [line.removeprefix("Finished ").split(": ", 1)[0] for line in lines if line.startswith("Finished ")]
        require(
            set(finished) == run_ids and len(finished) == len(runs), "Runner finished inventory differs from summary"
        )
        failures = [
            row
            for row in runs
            if row["returncode"] != 0 or row["completion_errors"] or row["integrity_status"] != "passed"
        ]
        planned = (
            len(protocol["clips"]) * len(protocol["sizes"]) * (len(protocol["experiments"]) + 1) * provenance["repeats"]
        )
        require(len(started) == len(runs) + (1 if short_name == "v1" else 0), "Unexpected runner start count")
        if short_name == "v1":
            require(len(runs) == 13 and planned == 96 and len(failures) == 1, "Unexpected aborted v1 classification")
            require(
                started[-1] == expected["interrupted_runner_session"] and started[-1] not in run_ids,
                "Interrupted v1 result was synthesized",
            )
        else:
            require(
                len(runs) == planned == 3 and not failures and provenance["repeats"] == 1,
                "Unexpected v4 regression pilot classification",
            )
            require(
                {row["source_lang"] for row in runs} == {"es"} and {row["size"] for row in runs} == {"e4b"},
                "Unexpected v4 language/model scope",
            )
        cohorts.append(
            {
                "cohort": name,
                "recorded_source_revision": source["code_sha"],
                "persisted_results": len(runs),
                "planned_runs": planned,
                "recorded_integrity_passes": len(runs) - len(failures),
                "failed_runs": [
                    {key: row[key] for key in ("session_id", "returncode", "completion_errors", "integrity_status")}
                    for row in failures
                ],
                "recorded_raw_member_count": len(members),
                "raw_member_payloads_rechecked": False,
                "summary_final_count_range": [
                    min(row["final_count"] for row in runs),
                    max(row["final_count"] for row in runs),
                ],
                "promotion_eligible": False,
            }
        )
    interrupted = interpretation["v1"]["interrupted_runner_session"]
    lifecycle = json_file(root / "interrupted-v1-run" / ("session_lifecycle_" + interrupted + ".json"))
    metadata = json_file(root / "interrupted-v1-run" / ("session_metadata_" + interrupted + ".json"))
    require(
        lifecycle["status"] == interpretation["v1"]["raw_pipeline_lifecycle_status"]
        and lifecycle["exit_code"] == interpretation["v1"]["raw_pipeline_exit_code"],
        "Interrupted lifecycle interpretation differs",
    )
    require(
        lifecycle["git_sha"] == metadata["git_sha"] == cohorts[0]["recorded_source_revision"],
        "Interrupted source identity differs",
    )
    expected_interrupted = {"interrupted-v1-run/" + Path(path).name for path in inputs["interrupted_raw_files"]}
    require(
        expected_interrupted == {path for path in entries if path.startswith("interrupted-v1-run/")},
        "Interrupted artifact inventory differs",
    )
    require(
        interpretation["defaults_changed"] is False and interpretation["audio_included"] is False,
        "Unexpected archive scope",
    )
    return {
        "schema_version": 1,
        "reviewed_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "status": "metadata_checks_passed_payload_reverification_deferred",
        "archive_index_sha256": sha(index_path.read_bytes()),
        "indexed_files": len(entries),
        "small_files_rehashed": checks,
        "stat_only_deferred_payloads": deferred,
        "cohorts": cohorts,
        "interrupted_artifact_count": len(expected_interrupted),
        "limits": [
            "No archive content was read, hashed, decompressed or recompressed during this review.",
            "Recorded archive/member/source-code hashes and producer readback checks are retained; their payload bytes were not independently reverified here.",
            "Run summaries were cross-checked against small metadata and recorded member inventory, not recomputed from large raw result JSON.",
            "No model, audio, device, Git, build or install operation was performed.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = review(args.root)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(
        json.dumps(
            {
                "status": result["status"],
                "small_files_rehashed": len(result["small_files_rehashed"]),
                "payloads_deferred": len(result["stat_only_deferred_payloads"]),
            }
        )
    )


if __name__ == "__main__":
    main()
