#!/usr/bin/env python3
"""Verify and reconstruct terminal-session evidence in a NEW directory.

No extraction, original-file reads, models, network or third-party dependencies.
The expected manifest hash should come from the separately retained build result.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import shutil
import sys
import tarfile
from pathlib import Path

sys.dont_write_bytecode = True

from evidence_common import PART_BYTES, digest, read_exact, safe_name

MAX_MEMBER_BYTES = 128 * 1024 * 1024
MAX_TOTAL_BYTES = 512 * 1024 * 1024


def simple(value):
    safe_name(value)
    if "/" in value:
        raise ValueError("Artifact must be a simple filename")
    return value


def checked(root, item):
    if item["size_bytes"] > PART_BYTES or item["size_bytes"] < 0:
        raise ValueError("Published file exceeds 480 KiB")
    return read_exact(root / simple(item["path"]), item)


def verify(root, output, expected_manifest=None):
    root, output = Path(root), Path(output)
    if output.exists():
        raise ValueError("Choose a new verification output directory")
    manifest_bytes = read_exact(root / "artifact-manifest.json")
    if len(manifest_bytes) > PART_BYTES or (expected_manifest and digest(manifest_bytes) != expected_manifest):
        raise ValueError("Manifest size/hash mismatch")
    manifest = json.loads(manifest_bytes)
    if (manifest.get("schema_version") != 1 or manifest.get("kind") != "terminal-session-evidence"
            or manifest.get("part_bytes") != PART_BYTES):
        raise ValueError("Unsupported manifest")
    tools = manifest.get("tools", [])
    if len(tools) != 3 or {item["path"] for item in tools} != {
        "package_endurance.py", "verify_endurance.py", "evidence_common.py"
    }:
        raise ValueError("Missing or duplicate tool provenance")
    for item in tools:
        checked(root, item)
    # This also binds the copied tools and every sidecar; SHA256SUMS excludes itself.
    sums = read_exact(root / "SHA256SUMS")
    expected_names = {"SHA256SUMS"}
    for line in sums.decode().splitlines():
        hash_value, name = line.split("  ", 1)
        simple(name)
        if name in expected_names:
            raise ValueError("Duplicate checksum entry")
        expected_names.add(name)
        payload = read_exact(root / name)
        if len(payload) > PART_BYTES or digest(payload) != hash_value:
            raise ValueError("Published sidecar/shard checksum mismatch")
    if {p.name for p in root.iterdir()} != expected_names:
        raise ValueError("Unindexed or missing published file")
    inventory_gzip = checked(root, manifest["inventory"])
    with gzip.GzipFile(fileobj=io.BytesIO(inventory_gzip)) as stream:
        inventory_data = stream.read(MAX_MEMBER_BYTES + 1)
    if len(inventory_data) > MAX_MEMBER_BYTES:
        raise ValueError("Inventory exceeds bounded size")
    records = [json.loads(line) for line in inventory_data.splitlines()]
    indexed = {safe_name(row["archive_path"]): row for row in records}
    if len(indexed) != len(records) or "inventory.jsonl" in indexed:
        raise ValueError("Duplicate/recursive inventory")
    if len(records) != manifest["source_member_count"] or len(records) > 4096:
        raise ValueError("Inventory count mismatch")
    artifact = manifest["artifact"]
    if artifact.get("filename") != "endurance-evidence.tar.gz" or len(artifact["parts"]) > 2048:
        raise ValueError("Unexpected archive/shard count")
    output.mkdir(parents=True)
    try:
        total, whole, seen = 0, hashlib.sha256(), set()
        target = output / artifact["filename"]
        with target.open("xb") as stream:
            for item in artifact["parts"]:
                name = simple(item["path"])
                if name in seen:
                    raise ValueError("Duplicate shard")
                seen.add(name)
                data = checked(root, item)
                total += len(data)
                if total > MAX_TOTAL_BYTES:
                    raise ValueError("Archive exceeds bounded total size")
                whole.update(data)
                stream.write(data)
        if total != artifact["size_bytes"] or whole.hexdigest() != artifact["sha256"]:
            raise ValueError("Whole-stream digest mismatch")
        names, unpacked = set(), 0
        with tarfile.open(target, "r|gz") as archive:
            for member in archive:
                name = safe_name(member.name)
                if not member.isfile() or name in names or member.size > MAX_MEMBER_BYTES:
                    raise ValueError("Unsafe/duplicate/oversized archive member")
                names.add(name)
                unpacked += member.size
                if unpacked > MAX_TOTAL_BYTES:
                    raise ValueError("Archive expansion exceeds bounded total size")
                payload = archive.extractfile(member).read()
                if name == "inventory.jsonl":
                    if payload != inventory_data:
                        raise ValueError("Internal/external inventory mismatch")
                else:
                    item = indexed.get(name)
                    if not item or len(payload) != item["size_bytes"] or digest(payload) != item["sha256"]:
                        raise ValueError("Archive member differs from indexed bytes")
        if names != set(indexed) | {"inventory.jsonl"}:
            raise ValueError("Missing archive members")
        return {"verified": True, "members": len(names), "manifest_sha256": digest(manifest_bytes),
                "archive": str(target), "archive_sha256": whole.hexdigest()}
    except BaseException:
        shutil.rmtree(output)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest-sha256")
    args = parser.parse_args()
    print(json.dumps(verify(args.root, args.output, args.manifest_sha256), indent=2))


if __name__ == "__main__":
    main()
