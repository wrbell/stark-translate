#!/usr/bin/env python3
"""Archive explicitly selected, terminal installed sessions; never package live input.

--plan inspects only terminal sessions; --build creates a fresh sharded archive.
No audio/model files, environment files, correction sidecars or directory globs.
Raw caption text/logs are intentionally included; this is not a public-safe bundle.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import socket
import sys
import tempfile
from datetime import datetime
from pathlib import Path

sys.dont_write_bytecode = True

from evidence_common import (
    Inventory, PART_BYTES, digest, gzip_bytes, json_bytes, read_exact,
    safe_name, shard, verify_archive, write_archive,
)

METRICS = {
    "ab_metrics": "csv", "diagnostics": "jsonl", "partials": "jsonl",
    "display_metrics": "jsonl", "session_metadata": "json",
    "session_lifecycle": "json", "health": "json", "session": "log",
}
FORBIDDEN = {".env", "models", "adapters", "corrections", "review_notes", "live_sessions", "raw"}
EXTRA_SUFFIXES = {
    "source_provenance": {".json", ".jsonl"},
    "reader_output": {".json", ".jsonl", ".csv", ".md", ".txt"},
    "ui_ax": {".txt"},
    "ui_screenshot": {".png"},
}
HASH = re.compile(r"[0-9a-f]{64}")
COMMIT = re.compile(r"[0-9a-f]{40}")


def explicit_path(value):
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("Evidence inputs must be explicit absolute paths")
    if any(p.lower() in FORBIDDEN or p.lower().startswith(".env") for p in path.parts):
        raise ValueError("Disallowed evidence source category")
    return path


def dead(pid):
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        raise ValueError("Missing valid owner PID; cannot establish stopped state")
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return
    except PermissionError as exc:
        raise ValueError("Cannot establish that session owner stopped") from exc
    raise ValueError("Session or observed owned child still exists; wait for cleanup")


def terminal(lifecycle, sid):
    if lifecycle.get("schema_version") != 1 or lifecycle.get("session_id") != sid:
        raise ValueError("Lifecycle session/schema mismatch")
    if lifecycle.get("status") not in {"completed", "failed", "interrupted"}:
        raise ValueError("Lifecycle is not explicitly terminal; never infer completion")
    if lifecycle.get("hostname") != socket.gethostname():
        raise ValueError("Only local-host evidence can establish stopped owner state")
    for key in ("started_at", "ended_at"):
        value = datetime.fromisoformat(lifecycle[key])
        if value.tzinfo is None:
            raise ValueError("Lifecycle timestamps require timezone")
    if datetime.fromisoformat(lifecycle["ended_at"]) < datetime.fromisoformat(lifecycle["started_at"]):
        raise ValueError("Lifecycle end precedes start")
    if not lifecycle.get("run_id") or not isinstance(lifecycle.get("exit_code"), int):
        raise ValueError("Terminal lifecycle lacks ownership/exit evidence")
    if lifecycle["status"] == "completed" and lifecycle["exit_code"] != 0:
        raise ValueError("Completed lifecycle contradicts exit status")
    dead(lifecycle["pid"])


def pointer(document, value):
    if not isinstance(value, str) or not value.startswith("/"):
        raise ValueError("Proof requires an explicit JSON pointer")
    for part in value[1:].split("/"):
        part = part.replace("~1", "/").replace("~0", "~")
        document = document[int(part)] if isinstance(document, list) else document[part]
    return document


def proof(inventory, session, role, spec):
    if not HASH.fullmatch(spec.get("sha256", "")):
        raise ValueError("Proof input requires its expected SHA-256")
    path = explicit_path(spec["path"])
    if path.suffix != ".json":
        raise ValueError("Runtime proof must be a JSON receipt")
    data = inventory.add(
        f"sessions/{session}/binding/{role}.json", path,
        expected={"sha256": spec["sha256"]}, category="runtime_binding",
    )
    return pointer(json.loads(data), spec["pointer"])


def prepare(spec_path):
    spec_path = explicit_path(str(spec_path))
    inventory = Inventory()
    specification = json.loads(inventory.add("provenance/input-spec.json", spec_path, category="selection"))
    if specification.get("schema_version") != 1 or not specification.get("sessions"):
        raise ValueError("Requires schema 1 and explicit session selections")
    ids, summaries, owner_records = set(), [], []
    for session in specification["sessions"]:
        sid = session["session_id"]
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,159}", sid) or ".." in sid or sid in ids:
            raise ValueError("Invalid or duplicate session ID")
        ids.add(sid)
        metrics = explicit_path(session["metrics_dir"])
        life_path = metrics / f"session_lifecycle_{sid}.json"
        # Gate before reading any large/continuously-written metrics.
        life = json.loads(read_exact(life_path))
        terminal(life, sid)
        missing = session.get("missing_metrics", {})
        if set(missing) - (set(METRICS) - {"session_lifecycle", "session_metadata"}):
            raise ValueError("Invalid missing-metric declaration")
        raw = {}
        for prefix, ext in METRICS.items():
            name = f"{prefix}_{sid}.{ext}"
            path = metrics / name
            if not path.exists() and prefix in missing and str(missing[prefix]).strip():
                continue
            if prefix in missing:
                raise ValueError("Missing declaration contradicts existing evidence")
            raw[prefix] = inventory.add(f"sessions/{sid}/metrics/{name}", path, category="raw_metric")
        if json.loads(raw["session_lifecycle"]) != life:
            raise ValueError("Lifecycle changed during selection")
        metadata = json.loads(raw["session_metadata"])
        if metadata.get("session_id") != sid:
            raise ValueError("Metadata session mismatch")
        if "health" in raw and json.loads(raw["health"]).get("session_id") != sid:
            raise ValueError("Health session mismatch")
        if "diagnostics" in raw and life.get("diagnostics"):
            expected = life["diagnostics"]
            if (digest(raw["diagnostics"]), len(raw["diagnostics"])) != (expected["sha256"], expected["size_bytes"]):
                raise ValueError("Terminal diagnostic integrity mismatch; preserve and investigate original")
        binding = session["runtime_binding"]
        if len({(binding[k]["path"], binding[k]["sha256"]) for k in ("wheel", "source", "pipeline")}) != 1:
            raise ValueError("Wheel/source/pipeline must reference the same immutable build receipt")
        wheel = proof(inventory, sid, "wheel", binding["wheel"])
        source = proof(inventory, sid, "source", binding["source"])
        pipeline = proof(inventory, sid, "pipeline", binding["pipeline"])
        if not isinstance(wheel, str) or not HASH.fullmatch(wheel):
            raise ValueError("Wheel proof does not resolve to a SHA-256")
        if not isinstance(source, str) or not COMMIT.fullmatch(source):
            raise ValueError("Source proof must resolve to an immutable full commit")
        if not isinstance(pipeline, str) or not HASH.fullmatch(pipeline):
            raise ValueError("Pipeline proof does not resolve to a SHA-256")
        provenance = life.get("source") or {}
        if life.get("pipeline_sha256") != pipeline or metadata.get("pipeline_sha256") != pipeline:
            raise ValueError("Executed pipeline differs from declared artifact")
        if provenance.get("package_artifact_hashes", {}).get("sha256") != wheel:
            raise ValueError("Lifecycle does not bind the actual installed wheel")
        if metadata.get("package_artifact_hashes", {}).get("sha256") != wheel:
            raise ValueError("Startup metadata does not bind the same installed wheel")
        if life.get("git_sha") not in (None, source) or metadata.get("git_sha") not in (None, source):
            raise ValueError("Recorded source commit contradicts build receipt")
        monitor_spec = session["monitor"]
        monitor = json.loads(inventory.add(
            f"sessions/{sid}/monitor/report.json", explicit_path(monitor_spec["report"]), category="monitor"))
        if (monitor.get("session_id") != sid or monitor.get("monitor_status") not in {"completed", "partial", "error"}
                or not monitor.get("ended_at") or monitor.get("ownership", {}).get("run_id") != life["run_id"]
                or monitor.get("ownership", {}).get("pid") != life["pid"]):
            raise ValueError("Monitor is live or its session/ownership differs")
        for process in monitor.get("measurements", {}).get("last_observed_processes", []):
            dead(process["pid"])
        samples = inventory.add(f"sessions/{sid}/monitor/samples.jsonl",
                                explicit_path(monitor_spec["samples"]), category="monitor")
        for line in samples.splitlines():
            sample = json.loads(line)
            for process in sample.get("tree", {}).get("processes", []):
                if process.get("role") == "pipeline" and (
                    process.get("pid") != life["pid"]
                    or process.get("create_time") != monitor["ownership"].get("create_time")
                ):
                    raise ValueError("Monitor samples contain a different pipeline identity")
        owner_records.append(life)
        summaries.append({
            "session_id": sid, "status": life["status"], "exit_code": life["exit_code"],
            "wheel_sha256": wheel, "source_revision": source, "pipeline_sha256": pipeline,
            "missing_metrics": missing, "monitor_status": monitor["monitor_status"],
            "known_limitations": session.get("known_limitations", []),
        })
    for extra in specification.get("extras", []):
        category = extra["category"]
        if category not in EXTRA_SUFFIXES:
            raise ValueError("Extra category is not allowlisted")
        path = explicit_path(extra["path"])
        if path.suffix.lower() not in EXTRA_SUFFIXES[category]:
            raise ValueError("Extra suffix is not allowlisted")
        if not HASH.fullmatch(extra.get("sha256", "")):
            raise ValueError("Explicit extra requires its expected SHA-256")
        name = safe_name(extra["name"])
        if "/" in name:
            raise ValueError("Extra archive name must be a simple filename")
        if category.startswith("ui_") and extra.get("reviewed_no_private_notes") is not True:
            raise ValueError("UI evidence needs explicit visual/text privacy review")
        payload = inventory.add(
            f"extras/{category}/{name}", path, expected={"sha256": extra["sha256"]},
            category=category, scan=category != "ui_screenshot",
        )
        if category == "ui_screenshot" and not payload.startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValueError("Screenshot is not a PNG")
    inventory.add("provenance/session-bindings.json", data=json_bytes({
        "schema_version": 1, "sessions": summaries,
        "scope": "Retained observations; terminal recording status is not timing/quality or physical-device certification.",
        "privacy": "Caption text and selected logs/UI are included. No audio, models, environment files or corrections.",
    }), category="bindings")
    return inventory, owner_records


def unchanged(inventory, owners):
    for name in inventory.files:
        inventory.payload(name)
    for life in owners:
        terminal(life, life["session_id"])


def build(spec_path, output):
    output = Path(output).absolute()
    if output.exists():
        raise ValueError("Choose a new output directory; never overwrite retained evidence")
    inventory, owners = prepare(spec_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".endurance-stage-", dir=output.parent))
    try:
        records = b"".join(json_bytes(row).replace(b"\n", b"") + b"\n" for row in inventory.records())
        inventory.add("inventory.jsonl", data=records, category="inventory")
        write_archive(stage / "endurance-evidence.tar.gz", inventory)
        verify_archive(stage / "endurance-evidence.tar.gz", inventory)
        unchanged(inventory, owners)
        artifact = shard(stage / "endurance-evidence.tar.gz")
        inventory_zip = gzip_bytes(records)
        if len(inventory_zip) > PART_BYTES:
            raise ValueError("Inventory exceeds shard size; split selection into smaller packs")
        (stage / "inventory.jsonl.gz").write_bytes(inventory_zip)
        tool_records = []
        for name in ("package_endurance.py", "verify_endurance.py", "evidence_common.py"):
            payload = read_exact(Path(__file__).with_name(name))
            (stage / name).write_bytes(payload)
            tool_records.append({"path": name, "size_bytes": len(payload), "sha256": digest(payload)})
        manifest = {
            "schema_version": 1, "kind": "terminal-session-evidence", "part_bytes": PART_BYTES,
            "artifact": artifact,
            "inventory": {"path": "inventory.jsonl.gz", "size_bytes": len(inventory_zip),
                          "sha256": digest(inventory_zip)},
            "source_member_count": len(inventory.files) - 1,
            "tools": tool_records,
        }
        (stage / "artifact-manifest.json").write_bytes(json_bytes(manifest))
        checksums = "".join(f"{digest(p.read_bytes())}  {p.name}\n" for p in sorted(stage.iterdir()))
        (stage / "SHA256SUMS").write_text(checksums)
        if any(p.stat().st_size > PART_BYTES for p in stage.iterdir()):
            raise ValueError("Published sidecar exceeds 480 KiB")
        unchanged(inventory, owners)
        if output.exists():
            raise ValueError("Output appeared during packaging")
        # Claim a new directory with mkdir rather than replacing an empty one
        # that might have appeared between exists() and rename().
        output.mkdir()
        try:
            for path in stage.iterdir():
                path.rename(output / path.name)
        except BaseException:
            shutil.rmtree(output)
            raise
        return {"output": str(output), "manifest_sha256": digest((output / "artifact-manifest.json").read_bytes()),
                "source_members": len(inventory.files) - 1}
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--build", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.plan:
        inventory, owners = prepare(args.spec)
        unchanged(inventory, owners)
        print(json.dumps({"terminal_sessions": len(owners), "members": inventory.records()}, indent=2))
    else:
        if args.output is None:
            parser.error("--build requires --output NEW_DIRECTORY")
        print(json.dumps(build(args.spec, args.output), indent=2))


if __name__ == "__main__":
    main()
