"""Selected model-free primitives from the tested screen evidence packager.
No matrix assumptions; originals remain unchanged. Kept with the endurance tool.
"""
import gzip
import hashlib
import io
import json
import os
import re
import stat
import tarfile
from pathlib import Path, PurePosixPath

MAX_FILE_BYTES = 128 * 1024 * 1024


PART_BYTES = 480 * 1024


SECRET_RULES = {
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "credential_token": re.compile(r"\b(?:gh[pousr]_|github_pat_|hf_|sk-(?:proj-)?)\w{20,}\b"),
    "aws_access_key": re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    "authorization_value": re.compile(r"(?i)\bAuthorization[\"']?\s*[:=]\s*[\"']?(?:Bearer|Basic)\s+\S+"),
    "secret_assignment": re.compile(
        r"(?i)\b(?:password|passwd|api[_-]?key|client[_-]?secret|access[_-]?token|refresh[_-]?token|"
        r"HF_TOKEN|GITHUB_TOKEN|AWS_SECRET_ACCESS_KEY)[\"']?\s*[:=]\s*[\"']?[^\s\"',}]{6,}"
    ),
    "full_environment_dump": re.compile(r"(?m)^\s*(?:HOME|PATH|SHELL|TMPDIR|SSH_AUTH_SOCK|CONDA_PREFIX)=\S+"),
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode()


def safe_name(value):
    path = PurePosixPath(value)
    if not value or value == "." or path.is_absolute() or ".." in path.parts or "\\" in value or str(path) != value:
        raise ValueError("Unsafe archive member path")
    return value


def read_exact(path, expected=None):
    path = Path(path)
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError(f"Refusing symbolic-link evidence: {path.name}")
    before = path.stat()
    if not stat.S_ISREG(before.st_mode) or before.st_size > MAX_FILE_BYTES:
        raise ValueError(f"Evidence is nonregular or exceeds bounded file size: {path.name}")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(descriptor, "rb") as stream:
        opened = os.fstat(stream.fileno())
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise ValueError(f"Evidence replaced before opening: {path.name}")
        data = stream.read(MAX_FILE_BYTES + 1)
        end = os.fstat(stream.fileno())
    after = path.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ) or (end.st_size, end.st_mtime_ns) != (before.st_size, before.st_mtime_ns) or len(data) != before.st_size:
        raise ValueError(f"Evidence changed while reading: {path.name}")
    if expected and (digest(data) != expected["sha256"] or len(data) != expected.get("size_bytes", len(data))):
        raise ValueError(f"Evidence hash no longer matches analysis/provenance: {path.name}")
    return data


def sensitive_findings(data, name):
    """Conservative local scan; report rule/count only, never the secret value."""
    text = data.decode("utf-8", errors="strict")
    return [
        {"archive_path": name, "rule": rule, "count": len(matches)}
        for rule, pattern in SECRET_RULES.items()
        if (matches := pattern.findall(text))
    ]


class Inventory:
    def __init__(self):
        self.files = {}

    def add(self, name, path=None, *, data=None, expected=None, category="raw", scan=True):
        safe_name(name)
        if name in self.files:
            raise ValueError(f"Duplicate archive member: {name}")
        payload = read_exact(path, expected) if path is not None else data
        if payload is None:
            raise ValueError("Missing evidence payload")
        findings = sensitive_findings(payload, name) if scan else []
        if findings:
            raise ValueError("Sensitive-data review required (originals unchanged): " + json.dumps(findings))
        self.files[name] = {
            "archive_path": name,
            "source_path": str(Path(path).absolute()) if path else None,
            "size_bytes": len(payload),
            "sha256": digest(payload),
            "category": category,
            "payload": payload if path is None else None,
        }
        return payload

    def records(self):
        return [{k: v for k, v in self.files[name].items() if k != "payload"} for name in sorted(self.files)]

    def payload(self, name):
        record = self.files[name]
        return read_exact(record["source_path"], record) if record["source_path"] else record["payload"]


def gzip_bytes(data):
    out = io.BytesIO()
    with gzip.GzipFile(fileobj=out, mode="wb", filename="", mtime=0, compresslevel=6) as stream:
        stream.write(data)
    return out.getvalue()


def write_archive(path, inventory):
    with path.open("xb") as raw:
        with (
            gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0, compresslevel=6) as zipped,
            tarfile.open(fileobj=zipped, mode="w|", format=tarfile.PAX_FORMAT) as archive,
        ):
            for name in sorted(inventory.files):
                data = inventory.payload(name)
                info = tarfile.TarInfo(name)
                info.size, info.mode, info.mtime = len(data), 0o444, 0
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                archive.addfile(info, io.BytesIO(data))
        raw.flush()
        os.fsync(raw.fileno())


def verify_archive(path, inventory):
    with tarfile.open(path, "r:gz") as archive:
        members = archive.getmembers()
        if [m.name for m in members] != sorted(inventory.files) or any(not m.isfile() for m in members):
            raise ValueError("Archive has unexpected members")
        for member in members:
            data = archive.extractfile(member).read()
            record = inventory.files[member.name]
            if digest(data) != record["sha256"] or len(data) != record["size_bytes"]:
                raise ValueError(f"Archived bytes differ: {member.name}")


def shard(path):
    parts, total, whole = [], 0, hashlib.sha256()
    with path.open("rb") as stream:
        while data := stream.read(PART_BYTES):
            name = f"{path.name}.part{len(parts) + 1:04d}"
            with path.with_name(name).open("xb") as output:
                output.write(data)
            parts.append({"path": name, "size_bytes": len(data), "sha256": digest(data)})
            whole.update(data)
            total += len(data)
    path.unlink()  # Only the newly generated temporary stream, never original evidence.
    return {"filename": path.name, "size_bytes": total, "sha256": whole.hexdigest(), "parts": parts}
