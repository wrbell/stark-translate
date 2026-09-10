#!/usr/bin/env python3
"""Verify evidence shards and reconstruct gzip streams in a NEW directory.

python verify_evidence.py --root EVIDENCE_DIR --output NEW_REASSEMBLED_DIR

Checks part/whole SHA-256, exact archive member inventory and supplemental JSON.
No tar extraction, models, network, source mutation or third-party dependency.
"""

import argparse
import gzip
import hashlib
import io
import json
import shutil
import tarfile
from pathlib import Path, PurePosixPath

MAX_PART_BYTES = 480 * 1024
MAX_MEMBER_BYTES = 128 * 1024 * 1024


def sha(data):
    return hashlib.sha256(data).hexdigest()


def simple_path(value):
    if not isinstance(value, str) or Path(value).name != value or value in ("", ".", "..") or "\\" in value:
        raise ValueError("Unsafe artifact path")
    return value


def checked_read(root, record):
    path = root / simple_path(record["path"])
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_PART_BYTES:
        raise ValueError("Missing, linked or oversized published file")
    data = path.read_bytes()
    if len(data) != record["size_bytes"] or sha(data) != record["sha256"]:
        raise ValueError(f"Published file hash mismatch: {path.name}")
    return data


def decompress_bounded(data, limit=MAX_MEMBER_BYTES):
    with gzip.GzipFile(fileobj=io.BytesIO(data), mode="rb") as stream:
        value = stream.read(limit + 1)
    if len(value) > limit:
        raise ValueError("Decompressed artifact exceeds the declared safety bound")
    return value


def verify(root, output):
    if output.exists():
        raise ValueError("Choose a new output directory; originals will not be overwritten")
    manifest = json.loads((root / "artifact-manifest.json").read_text())
    if manifest.get("schema_version") != 1 or manifest.get("part_bytes") != MAX_PART_BYTES:
        raise ValueError("Unsupported evidence manifest")
    artifacts = manifest["artifacts"]
    if {a["filename"] for a in artifacts} != {"screen-evidence.tar.gz", "supplemental-analysis.json.gz"} or len(
        artifacts
    ) != 2:
        raise ValueError("Unexpected compressed artifacts")
    inventory_data = decompress_bounded(checked_read(root, manifest["inventory"]))
    records = [json.loads(line) for line in inventory_data.splitlines()]
    indexed = {row["archive_path"]: row for row in records}
    if len(records) != len(indexed) or "inventory.jsonl" in indexed:
        raise ValueError("Duplicate or recursive inventory")
    output.mkdir(parents=True)
    try:
        used = set()
        for artifact in artifacts:
            filename = simple_path(artifact["filename"])
            total, whole = 0, hashlib.sha256()
            with (output / filename).open("xb") as stream:
                for part in artifact["parts"]:
                    if part["path"] in used:
                        raise ValueError("Duplicate shard identity")
                    used.add(part["path"])
                    data = checked_read(root, part)
                    stream.write(data)
                    whole.update(data)
                    total += len(data)
            if total != artifact["size_bytes"] or whole.hexdigest() != artifact["sha256"]:
                raise ValueError("Reassembled artifact differs from whole-stream digest")
        actual_names = set()
        archived_analysis = None
        with tarfile.open(output / "screen-evidence.tar.gz", "r|gz") as archive:
            for member in archive:
                name = PurePosixPath(member.name)
                if (
                    not member.isfile()
                    or name.is_absolute()
                    or ".." in name.parts
                    or "\\" in member.name
                    or member.name in actual_names
                    or member.size > MAX_MEMBER_BYTES
                ):
                    raise ValueError("Unexpected, duplicate or unsafe archive member")
                actual_names.add(member.name)
                data = archive.extractfile(member).read()
                if member.name == "inventory.jsonl":
                    if data != inventory_data:
                        raise ValueError("Internal/external inventory mismatch")
                    continue
                record = indexed.get(member.name)
                if not record or sha(data) != record["sha256"] or len(data) != record["size_bytes"]:
                    raise ValueError(f"Archive-member integrity failure: {member.name}")
                if member.name == "supplemental/analysis.json":
                    archived_analysis = data
        if actual_names != set(indexed) | {"inventory.jsonl"}:
            raise ValueError("Missing archive evidence")
        analysis = decompress_bounded((output / "supplemental-analysis.json.gz").read_bytes())
        if analysis != archived_analysis:
            raise ValueError("Supplemental JSON differs from archived original")
        (output / "analysis.json").write_bytes(analysis)
        return {"verified": True, "members": len(actual_names), "analysis_sha256": sha(analysis)}
    except BaseException:
        shutil.rmtree(output)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.root, args.output), indent=2))


if __name__ == "__main__":
    main()
