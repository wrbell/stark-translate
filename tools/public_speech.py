"""Freeze a bounded public EN/ES speech evaluation set without inference imports.

FLEURS references are upstream dataset annotations, never local human approval.
Development and confirmation use the original dev/test partitions respectively.
No training records or model predictions enter this manifest.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import shutil
import tarfile
from collections import defaultdict
from pathlib import Path

REVISION = "70bb2e84b976b7e960aa89f1c648e09c59f894dd"
LANGUAGES = {"en": "en_us", "es": "es_419"}
PARTITIONS = {"development": "dev", "confirmation": "test"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_tsv(text: str) -> list[dict]:
    rows = []
    for fields in csv.reader(io.StringIO(text), delimiter="\t", quoting=csv.QUOTE_NONE):
        if len(fields) != 7:
            raise ValueError("FLEURS TSV must contain seven fields")
        sentence, filename, raw, normalized, _phonemes, samples, gender = fields
        if not sentence.isdecimal() or not re.fullmatch(r"[0-9]+\.wav", filename):
            raise ValueError("Invalid sentence or audio identity")
        if not raw.strip() or not normalized.strip() or int(samples) <= 0:
            raise ValueError("Empty reference or invalid sample count")
        rows.append(
            {
                "sentence_id": int(sentence),
                "filename": filename,
                "reference": raw,
                "upstream_normalized_reference": normalized,
                "num_samples": int(samples),
                "gender": gender,
            }
        )
    if len({r["filename"] for r in rows}) != len(rows):
        raise ValueError("Duplicate audio filename")
    return rows


def select(rows: list[dict], count: int) -> list[dict]:
    """Round-robin duration/gender strata with SHA ordering; one recording per sentence."""
    groups: dict[tuple, list] = defaultdict(list)
    for row in rows:
        seconds = row["num_samples"] / 16000
        bucket = "short" if seconds < 5 else "medium" if seconds < 10 else "long"
        groups[(bucket, row["gender"])].append(row)
    for values in groups.values():
        values.sort(key=lambda row: hashlib.sha256(row["filename"].encode()).hexdigest())
    selected, seen = [], set()
    while len(selected) < count:
        progressed = False
        for key in sorted(groups):
            while groups[key]:
                row = groups[key].pop(0)
                if row["sentence_id"] not in seen:
                    selected.append(row)
                    seen.add(row["sentence_id"])
                    progressed = True
                    break
            if len(selected) == count:
                break
        if not progressed:
            raise ValueError(f"Need {count} distinct sentences; found {len(selected)}")
    return selected


def download(revision: str, relative: str, cache: Path) -> Path:
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Dataset revision must be an immutable 40-hex commit")
    if not re.fullmatch(r"data/(?:en_us|es_419)/(?:dev\.tsv|test\.tsv|audio/(?:dev|test)\.tar\.gz)", relative):
        raise ValueError("Only frozen EN/ES FLEURS source paths are supported")
    path = cache / revision / relative
    receipt = path.with_name(path.name + ".sha256")
    if path.exists():
        if not receipt.exists() or receipt.read_text().strip() != sha256(path):
            raise ValueError(f"Cached source failed integrity check: {path}")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    print(f"Downloading {relative}", flush=True)
    try:
        from huggingface_hub import hf_hub_download

        source = hf_hub_download(
            repo_id="google/fleurs",
            repo_type="dataset",
            filename=relative,
            revision=revision,
            cache_dir=cache / ".hf-cache",
        )
        with Path(source).open("rb") as response, temporary.open("wb") as target:
            shutil.copyfileobj(response, target, length=1024 * 1024)
        digest = sha256(temporary)
        os.replace(temporary, path)
        receipt.write_text(digest + "\n")
    finally:
        temporary.unlink(missing_ok=True)
    return path


def copy_selected_audio(archive: Path, selected: list[dict], target: Path) -> dict[str, dict]:
    wanted = {row["filename"]: row for row in selected}
    copied = {}
    target.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r|gz") as source:
        for member in source:
            name = Path(member.name).name
            if name not in wanted:
                continue
            if not member.isfile() or name in copied or member.size > 16_000_000:
                raise ValueError(f"Invalid or duplicate selected archive member: {member.name}")
            stream = source.extractfile(member)
            if stream is None:
                raise ValueError(f"Missing archive audio: {name}")
            data = stream.read()
            # FLEURS source WAVs can be float format; soundfile is data-preparation only.
            import soundfile as sf

            samples, rate = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
            if rate != 16000 or samples.shape != (wanted[name]["num_samples"], 1):
                raise ValueError(f"Unexpected audio shape/rate: {name}")
            path = target / name
            # Keep original audio bytes; replay already handles source conversion.
            with path.open("xb") as output:
                output.write(data)
            copied[name] = {"path": str(path), "sha256": hashlib.sha256(data).hexdigest(), "sample_rate": rate}
    if copied.keys() != wanted.keys():
        raise ValueError(f"Missing selected recordings: {sorted(wanted.keys() - copied.keys())}")
    return copied


def prepare(output: Path, cache: Path, revision: str = REVISION, count: int = 50) -> dict:
    if output.exists():
        raise FileExistsError("Use a new output directory; frozen evaluation data is never overwritten")
    if count < 1:
        raise ValueError("Count must be positive")
    tables, selections, sources = {}, {}, []
    for lang, config in LANGUAGES.items():
        for split, upstream in PARTITIONS.items():
            relative = f"data/{config}/{upstream}.tsv"
            path = download(revision, relative, cache)
            tables[lang, split] = parse_tsv(path.read_text())
            selections[lang, split] = select(tables[lang, split], count)
            sources.append({"file": relative, "sha256": sha256(path)})
    development_ids = {
        r["sentence_id"] for (lang, split), rows in tables.items() if split == "development" for r in rows
    }
    confirmation_ids = {
        r["sentence_id"] for (lang, split), rows in tables.items() if split == "confirmation" for r in rows
    }
    if development_ids & confirmation_ids:
        raise ValueError("Source sentence overlap between development and confirmation partitions")
    output.mkdir(parents=True)
    records = []
    for (lang, split), selected in selections.items():
        other = "es" if lang == "en" else "en"
        references: dict[int, set[str]] = defaultdict(set)
        for row in tables[other, split]:
            references[row["sentence_id"]].add(row["reference"])
        relative = f"data/{LANGUAGES[lang]}/audio/{PARTITIONS[split]}.tar.gz"
        archive = download(revision, relative, cache)
        sources.append({"file": relative, "sha256": sha256(archive)})
        audio = copy_selected_audio(archive, selected, output / "audio" / lang / split)
        for row in selected:
            paired = references[row["sentence_id"]]
            records.append(
                {
                    **row,
                    **audio[row["filename"]],
                    "path": str(Path(audio[row["filename"]]["path"]).relative_to(output)),
                    "id": f"fleurs-{lang}-{split}-{row['sentence_id']}-{row['filename'][:-4]}",
                    "source_lang": lang,
                    "target_lang": other,
                    "partition": split,
                    "translation_reference": next(iter(paired)) if len(paired) == 1 else None,
                    "translation_reference_status": "shared_sentence_id_unique"
                    if len(paired) == 1
                    else "unavailable_or_ambiguous",
                    "provenance": "public_natural_read_speech",
                    "reference_provenance": "upstream_fleurs_annotation",
                    "human_approved_locally": False,
                    "training_eligible": False,
                    "speaker_id": None,
                }
            )
    manifest = {
        "schema_version": 1,
        "dataset": "google/fleurs",
        "revision": revision,
        "license": "CC-BY-4.0",
        "attribution": "FLEURS: Few-shot Learning Evaluation of Universal Representations of Speech (Google, 2022)",
        "selection": "SHA256 filename order; round-robin <5s/5-10s/>=10s and published gender strata; one audio per sentence",
        "split_policy": "upstream dev=development, test=confirmation; sentence IDs disjoint across languages",
        "normalization": "Raw and upstream normalized references retained unchanged; scorer must record its own normalization",
        "sources": sources,
        "records": records,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--per-partition", type=int, default=50)
    args = parser.parse_args()
    manifest = prepare(args.output, args.cache, args.revision, args.per_partition)
    print(json.dumps({"records": len(manifest["records"]), "revision": manifest["revision"]}))


if __name__ == "__main__":
    main()
