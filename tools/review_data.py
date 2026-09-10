"""Language-aware session review and portable exports; no inference dependencies.

Corrections are revisioned sidecars. Predictions/audio are never rewritten.
The same normalizer is used by the operator and the existing export CLI.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import threading
import zipfile
import zlib
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from tools.session_lifecycle import require_completed, session_status

_LOCK = threading.RLock()
_SESSION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,159}$")


class ReviewConflict(ValueError):
    """Another review revision or an incompatible dataset split already exists."""


@contextmanager
def _file_lock(directory: Path):
    """Serialize revisions across multiple operator worker processes as well as threads."""
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".review.lock").open("a+b") as stream:
        if os.name == "nt":
            import msvcrt

            if stream.tell() == 0:
                stream.write(b"\0")
                stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def session_key(value: str) -> str:
    if not _SESSION.fullmatch(value) or ".." in value:
        raise ValueError("Invalid session ID")
    return value


def read_jsonl(path: Path, *, complete_only: bool = False) -> list[dict]:
    if not path.exists():
        return []
    lines = path.read_bytes().splitlines(keepends=True)
    result = []
    for line in lines:
        if complete_only and not line.endswith(b"\n"):
            continue
        try:
            value = json.loads(line)
            if isinstance(value, dict):
                result.append(value)
        except (ValueError, UnicodeDecodeError):
            continue
    return result


def atomic_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as out:
            for row in rows:
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            os.fsync(out.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def normalize_record(record: dict, correction: dict | None = None) -> dict:
    """Legacy english/spanish_* slots mean source/translation, even in ES sessions."""
    result = dict(record)
    correction = correction or {}
    session = str(record.get("session") or "")
    source_lang = correction.get("source_lang") or record.get("source_lang")
    if source_lang not in ("en", "es"):
        source_lang = session.rsplit("_", 1)[-1]
    if source_lang not in ("en", "es"):
        source_lang = None
    original = str(record.get("source_text") or record.get("english") or "")
    translated = str(record.get("translation_text") or record.get("spanish_gemma") or "")
    result.update(
        source_lang=source_lang,
        target_lang=("es" if source_lang == "en" else "en") if source_lang else None,
        source_text=original,
        translation_text=translated,
        corrected_source_text=correction.get("corrected_source_text", record.get("corrected_english") or original),
        corrected_translation_text=correction.get(
            "corrected_translation_text", record.get("corrected_spanish") or translated
        ),
        transcript_approved=bool(correction.get("transcript_approved", False)),
        translation_approved=bool(correction.get("translation_approved", False)),
        excluded=bool(correction.get("excluded", False)),
        revision=int(correction.get("revision", 0)),
        review_note=str(correction.get("review_note") or ""),
        session_kind=record.get("session_kind") or "unknown",
    )
    if correction:
        result["review_timestamp"] = correction.get("review_timestamp")
        result["correction_source"] = "human"
    return result


def latest_corrections(path: Path) -> dict[int, dict]:
    result = {}
    for row in read_jsonl(path):
        try:
            result[int(row["chunk_id"])] = row
        except (KeyError, TypeError, ValueError):
            continue
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _bundle_files(directory: Path) -> list[Path]:
    if directory.is_symlink():
        raise ReviewConflict("Existing export directory must not be a symlink")
    paths = sorted(directory.rglob("*"))
    if any(path.is_symlink() for path in paths):
        raise ReviewConflict("Existing export bundle contains a symlink; inspect it before exporting")
    return [path for path in paths if path.is_file()]


def _archive_matches(archive: Path, directory: Path, files: list[Path]) -> bool:
    """Check every member and its content, including ZIP CRC/truncation errors."""
    if archive.is_symlink():
        raise ReviewConflict("Existing export archive must not be a symlink")
    if not archive.is_file():
        return False
    expected = {path.relative_to(directory).as_posix(): path for path in files}
    try:
        with zipfile.ZipFile(archive) as bundle:
            members = bundle.infolist()
            if len(members) != len(expected) or {item.filename for item in members} != set(expected):
                return False
            for item in members:
                path = expected[item.filename]
                if item.file_size != path.stat().st_size:
                    return False
                digest = hashlib.sha256()
                with bundle.open(item) as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(block)
                if digest.hexdigest() != sha256_file(path):
                    return False
        return True
    except (OSError, ValueError, EOFError, RuntimeError, zipfile.BadZipFile, zlib.error):
        return False


def _validate_bundle_samples(directory: Path, manifest: dict, records: list[dict]) -> None:
    """A damaged copied WAV or metadata must not become a newly valid archive."""
    samples = {row["sample_id"]: row for row in manifest["samples"]}
    reviewed = {f"{manifest['session']}__{row['chunk_id']}": row for row in records}
    split = manifest["split"]
    expected_stt = {
        lang: {
            key
            for key, row in reviewed.items()
            if row["source_lang"] == lang and row["transcript_approved"] and samples[key]["audio_sha256"]
        }
        for lang in ("en", "es")
    }
    expected_pairs = {
        key for key, row in reviewed.items() if row["transcript_approved"] and row["translation_approved"]
    }

    def metadata_rows(path):
        if not path.is_file():
            raise ReviewConflict("Existing export metadata is missing; inspect the bundle before retrying")
        rows = read_jsonl(path)
        if len(rows) != sum(bool(line.strip()) for line in path.read_bytes().splitlines()):
            raise ReviewConflict("Existing export metadata is corrupt; inspect the bundle before retrying")
        return rows

    def validate_rows(rows, expected):
        if len(rows) != len(expected) or {row.get("sample_id") for row in rows} != expected:
            raise ReviewConflict("Existing export metadata conflicts with the approved samples")
        for row in rows:
            sample = samples[row["sample_id"]]
            if (
                row.get("session") != manifest["session"]
                or row.get("split") != split
                or any(row.get(key) != sample[key] for key in ("revision", "audio_sha256"))
            ):
                raise ReviewConflict("Existing export metadata conflicts with sample provenance")

    if manifest.get("stt_samples") != {lang: len(ids) for lang, ids in expected_stt.items()} or manifest.get(
        "translation_pairs"
    ) != len(expected_pairs):
        raise ReviewConflict("Existing export manifest conflicts with approved sample counts")
    for lang, expected in expected_stt.items():
        audio_dir = directory / "whisper" / split / lang
        rows = metadata_rows(audio_dir / "metadata.jsonl")
        validate_rows(rows, expected)
        for row in rows:
            name = f"{row['sample_id']}.wav"
            audio = audio_dir / name
            original = reviewed[row["sample_id"]]
            if (
                row.get("file_name") != name
                or row.get("source_lang") != lang
                or row.get("transcription") != original["corrected_source_text"].strip()
                or not audio.is_file()
                or sha256_file(audio) != row["audio_sha256"]
            ):
                raise ReviewConflict(
                    "Existing export audio or transcript is damaged; inspect the bundle before retrying"
                )
    pairs = metadata_rows(directory / "translation" / f"{split}.jsonl")
    validate_rows(pairs, expected_pairs)
    for pair in pairs:
        original = reviewed[pair["sample_id"]]
        source, target = original["source_lang"], original["target_lang"]
        if (
            pair.get(source) != original["corrected_source_text"].strip()
            or pair.get(target) != original["corrected_translation_text"].strip()
        ):
            raise ReviewConflict("Existing export translation is damaged; inspect the bundle before retrying")


def _publish_bundle_archive(archive: Path, directory: Path, files: list[Path]) -> None:
    """Never expose a partly written ZIP under the downloadable final name."""
    fd, name = tempfile.mkstemp(prefix=".export-zip-", suffix=".tmp", dir=archive.parent)
    os.close(fd)
    try:
        with zipfile.ZipFile(name, "w", compression=zipfile.ZIP_DEFLATED) as out:
            for path in files:
                out.write(path, path.relative_to(directory))
        with open(name, "rb") as stream:
            os.fsync(stream.fileno())
        os.replace(name, archive)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def _finish_bundle_export(directory, archive, bundle_id, manifest, registry_path, registry, records):
    files = _bundle_files(directory)
    _validate_bundle_samples(directory, manifest, records)
    if not _archive_matches(archive, directory, files):
        _publish_bundle_archive(archive, directory, files)
    entry = {"session": manifest["session"], "split": manifest["split"], "bundle_id": bundle_id}
    if not any(all(row.get(key) == value for key, value in entry.items()) for row in registry):
        atomic_jsonl(registry_path, [*registry, entry])
    return {"bundle_id": bundle_id, "archive": str(archive), **manifest}


class ReviewStore:
    def __init__(self, project_root: Path):
        self.root = project_root.resolve()
        self.metrics = self.root / "metrics"
        self.corrections = self.root / "stark_data" / "corrections"

    def diagnostics_path(self, session: str) -> Path:
        path = self.metrics / f"diagnostics_{session_key(session)}.jsonl"
        if path.resolve().parent != self.metrics.resolve():
            raise ValueError("Diagnostics must stay inside metrics")
        return path

    def sidecar_path(self, session: str) -> Path:
        path = self.corrections / f"{session_key(session)}.jsonl"
        if path.resolve().parent != self.corrections.resolve():
            raise ValueError("Corrections must stay inside their directory")
        return path

    def sessions(self, active_session: str | None = None) -> list[dict]:
        sessions = []
        for path in sorted(self.metrics.glob("diagnostics_*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True):
            session = path.stem.removeprefix("diagnostics_")
            try:
                records = self.records(session)
            except ValueError:
                continue
            if not records:
                continue
            lifecycle = session_status(self.root, session)
            sessions.append(
                {
                    "session": session,
                    **lifecycle,
                    "active": session == active_session or lifecycle["active"],
                    "exportable": session != active_session and lifecycle["exportable"],
                    "segments": len(records),
                    "pending": sum(
                        not r["excluded"] and not (r["transcript_approved"] and r["translation_approved"])
                        for r in records
                    ),
                    "source_lang": records[0]["source_lang"],
                    "session_kind": records[0]["session_kind"],
                }
            )
        return sessions

    def records(self, session: str) -> list[dict]:
        path = self.diagnostics_path(session)
        if not path.exists():
            raise FileNotFoundError("Session diagnostics not found")
        edits = latest_corrections(self.sidecar_path(session))
        confirmations = {
            row.get("session"): row
            for row in read_jsonl(self.corrections / "session_provenance.jsonl")
            if row.get("session_kind") == "live"
        }
        records = {}
        for row in read_jsonl(path, complete_only=True):
            if "event" in row or "chunk_id" not in row or row.get("is_final") is False:
                continue
            try:
                chunk = int(row["chunk_id"])
            except (ValueError, TypeError):
                continue
            row["session"] = session
            row["chunk_id"] = chunk
            if (not row.get("session_kind") or row["session_kind"] == "unknown") and session in confirmations:
                row["session_kind"] = "live"
                row["provenance_confirmation"] = confirmations[session]
            records[chunk] = normalize_record(row, edits.get(chunk))
        return [records[k] for k in sorted(records)]

    def record(self, session: str, chunk: int) -> dict:
        for record in self.records(session):
            if record["chunk_id"] == chunk:
                return record
        raise FileNotFoundError("Finalized chunk not found")

    def audio_path(self, session: str, chunk: int, record: dict | None = None) -> Path:
        record = record or self.record(session, chunk)
        raw = record.get("audio_path")
        if not raw:
            raise FileNotFoundError("Audio was not retained for this chunk")
        path = Path(raw)
        if not path.is_absolute():
            path = self.root / path
        path = path.resolve()
        sessions_root = (self.root / "stark_data" / "live_sessions").resolve()
        allowed = (sessions_root / session_key(session)).resolve()
        if (
            not allowed.is_relative_to(sessions_root)
            or not path.is_relative_to(allowed)
            or path.suffix.lower() != ".wav"
        ):
            raise ValueError("Audio must belong to the selected session")
        if not path.is_file():
            raise FileNotFoundError("Audio is missing; transcript review is still available")
        return path

    def public_record(self, record: dict) -> dict:
        result = dict(record)
        result.pop("audio_path", None)
        result.pop("input_audio_path", None)
        try:
            self.audio_path(record["session"], record["chunk_id"], record)
            result["audio_available"] = True
        except (FileNotFoundError, ValueError):
            result["audio_available"] = False
        return result

    def save(self, session: str, chunk: int, update: dict) -> dict:
        with _LOCK, _file_lock(self.corrections):
            current = self.record(session, chunk)
            if update["expected_revision"] != current["revision"]:
                raise ReviewConflict("This segment was edited elsewhere. Reload before saving.")
            edit = {
                key: update[key]
                for key in (
                    "corrected_source_text",
                    "corrected_translation_text",
                    "transcript_approved",
                    "translation_approved",
                    "excluded",
                    "review_note",
                    "source_lang",
                )
                if key in update
            }
            merged = normalize_record(current, {**current, **edit})
            if (merged["transcript_approved"] or merged["translation_approved"]) and not merged["source_lang"]:
                raise ValueError("Choose the source language before approving a segment")
            if merged["transcript_approved"] and not merged["corrected_source_text"].strip():
                raise ValueError("Approved transcript cannot be empty")
            if merged["translation_approved"] and not merged["corrected_translation_text"].strip():
                raise ValueError("Approved translation cannot be empty")
            edit = {
                key: merged[key]
                for key in (
                    "corrected_source_text",
                    "corrected_translation_text",
                    "transcript_approved",
                    "translation_approved",
                    "excluded",
                    "review_note",
                    "source_lang",
                )
            }
            edit.update(
                session=session,
                chunk_id=chunk,
                revision=current["revision"] + 1,
                review_timestamp=datetime.now(UTC).isoformat(),
            )
            path = self.sidecar_path(session)
            atomic_jsonl(path, [*read_jsonl(path), edit])
            return self.public_record(normalize_record(current, edit))

    def export(self, session: str, *, split: str = "train") -> dict:
        if split not in ("train", "eval"):
            raise ValueError("Export split must be train or eval")
        with _LOCK, _file_lock(self.corrections):
            require_completed(self.root, session)
            records = [
                r
                for r in self.records(session)
                if not r["excluded"] and (r["transcript_approved"] or r["translation_approved"])
            ]
            if not records:
                raise ValueError("Approve at least one transcript or translation before exporting")
            if any(not r["source_lang"] for r in records):
                raise ValueError("Resolve the language of every approved segment before exporting")
            if split == "train" and any(
                r["session_kind"] != "live"
                or r.get("dataset_split", r.get("split")) in ("eval", "test", "holdout")
                or r.get("is_eval")
                for r in records
            ):
                raise ValueError(
                    "Training exports require live-session provenance; replay/unknown sessions are evaluation only"
                )
            registry_path = self.corrections / "export_registry.jsonl"
            registry = read_jsonl(registry_path)
            if any(r.get("session") == session and r.get("split") != split for r in registry):
                raise ReviewConflict("This session is already assigned to a different dataset split")
            exports = self.corrections / "exports"
            other_split = "eval" if split == "train" else "train"
            opposite = re.compile(rf"{re.escape(session)}-{other_split}-[0-9a-f]{{16}}(?:\.zip)?")
            if exports.exists() and any(opposite.fullmatch(path.name) for path in exports.iterdir()):
                raise ReviewConflict("This session has an existing export assigned to a different dataset split")
            # Hash revisions and audio so repeated exports are deterministic and portable.
            payload = []
            for rec in records:
                try:
                    audio = self.audio_path(session, rec["chunk_id"], rec)
                except (FileNotFoundError, ValueError):
                    audio = None
                payload.append(
                    {
                        "sample_id": f"{session}__{rec['chunk_id']}",
                        "revision": rec["revision"],
                        "audio_sha256": sha256_file(audio) if audio else None,
                    }
                )
            key = hashlib.sha256(json.dumps([split, payload], sort_keys=True).encode()).hexdigest()[:16]
            bundle_id = f"{session}-{split}-{key}"
            destination = exports / bundle_id
            archive = exports / f"{bundle_id}.zip"
            if destination.exists():
                _bundle_files(destination)
                try:
                    manifest = json.loads((destination / "manifest.json").read_text())
                except (OSError, ValueError) as exc:
                    raise ReviewConflict(
                        "Existing export manifest is unreadable; inspect the bundle before retrying"
                    ) from exc
                expected = {"schema_version": 1, "session": session, "split": split, "samples": payload}
                if not isinstance(manifest, dict) or any(manifest.get(k) != v for k, v in expected.items()):
                    raise ReviewConflict("Existing export manifest conflicts with the requested session revisions")
                return _finish_bundle_export(
                    destination, archive, bundle_id, manifest, registry_path, registry, records
                )
            exports.mkdir(parents=True, exist_ok=True)
            temp = Path(tempfile.mkdtemp(prefix=".export-", dir=exports))
            try:
                transcripts = {"en": [], "es": []}
                pairs = []
                for rec, provenance in zip(records, payload, strict=True):
                    common = {
                        **provenance,
                        "session": session,
                        "chunk_id": rec["chunk_id"],
                        "split": split,
                        "source_lang": rec["source_lang"],
                        "target_lang": rec["target_lang"],
                        "session_kind": rec["session_kind"],
                        "provenance_confirmation": rec.get("provenance_confirmation"),
                    }
                    if rec["transcript_approved"] and provenance["audio_sha256"]:
                        language = rec["source_lang"]
                        name = f"{provenance['sample_id']}.wav"
                        audio_dir = temp / "whisper" / split / language
                        audio_dir.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(self.audio_path(session, rec["chunk_id"], rec), audio_dir / name)
                        transcripts[language].append(
                            {**common, "file_name": name, "transcription": rec["corrected_source_text"].strip()}
                        )
                    if rec["transcript_approved"] and rec["translation_approved"]:
                        source = rec["corrected_source_text"].strip()
                        target = rec["corrected_translation_text"].strip()
                        pairs.append(
                            {
                                **common,
                                "en": source if rec["source_lang"] == "en" else target,
                                "es": target if rec["source_lang"] == "en" else source,
                            }
                        )
                if not any(transcripts.values()) and not pairs:
                    raise ValueError(
                        "No exportable samples: approve a transcript with audio, or both texts for translation"
                    )
                for language, rows in transcripts.items():
                    atomic_jsonl(temp / "whisper" / split / language / "metadata.jsonl", rows)
                atomic_jsonl(temp / "translation" / f"{split}.jsonl", pairs)
                atomic_jsonl(temp / "corrections.jsonl", list(latest_corrections(self.sidecar_path(session)).values()))
                manifest = {
                    "schema_version": 1,
                    "session": session,
                    "split": split,
                    "stt_samples": {lang: len(rows) for lang, rows in transcripts.items()},
                    "translation_pairs": len(pairs),
                    "samples": payload,
                }
                (temp / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
                os.replace(temp, destination)
                return _finish_bundle_export(
                    destination, archive, bundle_id, manifest, registry_path, registry, records
                )
            finally:
                if temp.exists():
                    shutil.rmtree(temp)
