"""Generate evaluation-only blinded EN/ES packets from completed translation runs.

No inference, source rewriting, reference invention or review approval occurs.
Keep the private mapping key away from reviewers until their ratings are sealed.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import mac_followup_quality as quality

ENGINES = ("e4b", "e2b")
LANGUAGES = ("en", "es")
PREFERENCE = [None, "A", "B", "tie", "neither", "not_assessable"]
SEVERITY = [None, "none", "minor", "major", "critical", "not_assessable"]


def text_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def fresh_review():
    return {
        "meaning_error": {"A": None, "B": None},
        "meaning_error_severity": {"A": None, "B": None},
        "meaning_error_notes": {"A": "", "B": ""},
        "terminology_preference": None,
        "overall_preference": None,
        "preference_reason": "",
        "reviewer_id": None,
        "reviewed_at": None,
        "approval": False,
    }


def review_schema():
    """Schema for one editable annotation, identified by immutable case/input hashes."""

    def pair(spec):
        return {
            "type": "object",
            "properties": {"A": spec, "B": spec},
            "required": ["A", "B"],
            "additionalProperties": False,
        }

    fields = {
        "meaning_error": pair({"type": ["boolean", "null"]}),
        "meaning_error_severity": pair({"enum": SEVERITY}),
        "meaning_error_notes": pair({"type": "string"}),
        "terminology_preference": {"enum": PREFERENCE},
        "overall_preference": {"enum": PREFERENCE},
        "preference_reason": {"type": "string"},
        "reviewer_id": {"type": ["string", "null"]},
        "reviewed_at": {"type": ["string", "null"]},
        "approval": {"type": "boolean"},
    }
    properties = {
        "schema_version": {"const": 1},
        "case_id": {"type": "string"},
        "immutable_case_sha256": {"type": "string", "pattern": "^[a-f0-9]{64}$"},
        "human_approved_locally": {"const": False},
        "training_eligible": {"const": False},
        "review": {"type": "object", "properties": fields, "required": list(fields), "additionalProperties": False},
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Unapproved bilingual translation review annotation",
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def _sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def load_cohort(directory, manifest_path, partition, repeats):
    if type(repeats) is not int or repeats < 1 or partition not in quality.PARTITIONS:
        raise ValueError("Explicit positive repeats and development/confirmation partition required")
    manifest = quality.read_json(manifest_path)
    quality.validate_portable(manifest)
    canaries = manifest.get("canaries", [])
    if (
        len(canaries) != 18
        or len({r.get("id") for r in canaries}) != 18
        or any(
            r.get("reference") is not None
            or r.get("source_lang") != "en"
            or r.get("target_lang") != "es"
            or r.get("human_approved_locally") is not False
            or r.get("training_eligible") is not False
            for r in canaries
        )
    ):
        raise ValueError("Require all 18 distinct original EN→ES lexical canaries without full references or approval")
    index_path = directory / "index.json"
    index = quality.read_json(index_path)
    manifest_hash = quality.digest(manifest_path)
    expected = {(name, lang, repeat) for name in ENGINES for lang in LANGUAGES for repeat in range(repeats)}
    if (
        index.get("manifest_sha256") != manifest_hash
        or index.get("task") != "translate"
        or index.get("partition") != partition
    ):
        raise ValueError("Index manifest/task/partition mismatch")
    if (
        index.get("completed") is not True
        or index.get("planned_runs") != len(expected)
        or len(index.get("runs", [])) != len(expected)
    ):
        raise ValueError("Incomplete/missing translation arms; no packet generated")
    runs = {}
    entries = {}
    model_hashes: dict[str, set[str]] = {name: set() for name in ENGINES}
    common_cohorts = set()
    files = set()
    for entry in index["runs"]:
        key = entry.get("engine"), entry.get("source_lang"), entry.get("repeat")
        if type(key[2]) is not int or key not in expected or key in runs or entry.get("file") in files:
            raise ValueError("Duplicate/unexpected run identity or file")
        files.add(entry["file"])
        path = quality.relative_file(directory, entry["file"])
        if not _sha(entry.get("sha256")) or quality.digest(path) != entry["sha256"]:
            raise ValueError("Run SHA256 differs from index")
        run = quality.read_json(path)
        if type(run.get("repeat")) is not int or any(
            run.get(k) != entry.get(k) for k in ("engine", "source_lang", "repeat", "returncode")
        ):
            raise ValueError("Worker/index identity mismatch")
        if (
            entry.get("completed") is not True
            or entry.get("returncode") != 0
            or run.get("status") != "completed"
            or run.get("error")
            or run.get("cleanup_error")
        ):
            raise ValueError("Failed/incomplete worker arm; no packet generated")
        if (
            run.get("manifest_sha256") != manifest_hash
            or run.get("partition") != partition
            or run.get("task") != "translate"
        ):
            raise ValueError("Worker manifest/task/partition mismatch")
        if run.get("human_approved_locally") is not False or run.get("training_eligible") is not False:
            raise ValueError("Expected unapproved evaluation-only raw runs")
        items = quality.selected_items(manifest, "translate", partition, key[1], True)
        if any(
            item.get("source_lang") != key[1] or item.get("target_lang") != ("es" if key[1] == "en" else "en")
            for item in items
        ):
            raise ValueError("Expected EN↔ES item direction")
        if run.get("expected_ids") != [item["id"] for item in items]:
            raise ValueError("Missing input/canary or changed frozen item order")
        summary = quality.summarize_run(run, manifest)
        if not summary["eligible_for_comparison"]:
            raise ValueError("Invalid raw translation rows or loaded model identity")
        if run.get("config") != quality.engine_config(key[0], "translate"):
            raise ValueError("Unexpected translation model/configuration")
        inventory = run.get("model", {})
        if not isinstance(inventory.get("resolved_path"), str) or not inventory["resolved_path"]:
            raise ValueError("Missing resolved primary model path")
        file_rows = inventory.get("files", [])
        if not file_rows or quality.json_hash(file_rows) != inventory.get("files_sha256"):
            raise ValueError("Model file inventory hash mismatch")
        for item in file_rows:
            name = Path(item.get("path", ""))
            if (
                name.is_absolute()
                or ".." in name.parts
                or not name.parts
                or not _sha(item.get("sha256"))
                or type(item.get("size_bytes")) is not int
                or item["size_bytes"] < 0
            ):
                raise ValueError("Invalid portable model file inventory")
        if run["model_identity"].get("requested_model_id") != inventory.get("resolved_path"):
            raise ValueError("Loaded model request differs from resolved primary")
        environment = run.get("environment", {})
        code = environment.get("quality_source_sha256")
        versions = environment.get("versions")
        if (
            not code
            or not all(_sha(v) for v in code.values())
            or not isinstance(versions, dict)
            or not versions.get("mlx")
        ):
            raise ValueError("Missing exact runtime source/version provenance")
        common_cohorts.add(quality.json_hash({"source": code, "versions": versions}))
        model_hashes[key[0]].add(inventory["files_sha256"])
        runs[key], entries[key] = run, entry
    if set(runs) != expected or len(common_cohorts) != 1 or any(len(values) != 1 for values in model_hashes.values()):
        raise ValueError("Missing arm or mixed source/runtime/model-file cohorts")
    return (
        manifest,
        index,
        runs,
        entries,
        {
            "manifest_sha256": manifest_hash,
            "index_sha256": quality.digest(index_path),
            "runtime_cohort_sha256": next(iter(common_cohorts)),
        },
    )


def blind_labels(seed, item_id, manifest_hash):
    order = (
        ENGINES
        if hmac.digest(seed.encode(), f"{manifest_hash}:{item_id}".encode(), "sha256")[0] % 2 == 0
        else ENGINES[::-1]
    )
    return dict(zip(("A", "B"), order, strict=True))


def make_packet(directory, manifest_path, *, partition, repeats, seed):
    if not isinstance(seed, str) or len(seed) < 16:
        raise ValueError("Private deterministic blinding seed must have at least 16 characters")
    manifest, index, runs, entries, provenance = load_cohort(directory, manifest_path, partition, repeats)
    packet_id = hmac.new(seed.encode(), quality.json_hash(provenance).encode(), "sha256").hexdigest()[:24]
    cases, mapping = [], []
    source_records = {r["id"]: r for r in manifest["records"]}
    run_rows = {key: {row["id"]: row for row in run["rows"]} for key, run in runs.items()}
    for lang in LANGUAGES:
        for item in quality.selected_items(manifest, "translate", partition, lang, True):
            labels = blind_labels(seed, item["id"], provenance["manifest_sha256"])
            drift = {
                label: len({run_rows[(engine, lang, r)][item["id"]]["hypothesis"] for r in range(repeats)}) > 1
                for label, engine in labels.items()
            }
            original = source_records.get(item["id"])
            for repeat in range(repeats):
                case_id = hmac.new(seed.encode(), f"{packet_id}:{item['id']}:{repeat}".encode(), "sha256").hexdigest()[
                    :24
                ]
                hypotheses = {
                    label: run_rows[(engine, lang, repeat)][item["id"]]["hypothesis"]
                    for label, engine in labels.items()
                }
                case = {
                    "schema_version": 1,
                    "case_id": case_id,
                    "source_id": item["id"],
                    "repeat": repeat,
                    "source_lang": item["source_lang"],
                    "target_lang": item["target_lang"],
                    "partition": item["partition"],
                    "category": item["domain"],
                    "source": item["source"],
                    "reference": item["reference"],
                    "reference_provenance": item["reference_provenance"],
                    "required_substrings": item.get("required_substrings", []),
                    "source_text_sha256": text_hash(item["source"]),
                    "reference_text_sha256": text_hash(item["reference"]) if item["reference"] is not None else None,
                    "source_audio_sha256": original["sha256"] if original else None,
                    "sentence_id": item.get("sentence_id"),
                    "input_sha256": quality.json_hash(item),
                    "hypotheses": hypotheses,
                    "hypothesis_sha256": {label: text_hash(text) for label, text in hypotheses.items()},
                    "outputs_differ": hypotheses["A"] != hypotheses["B"],
                    "repeat_output_drift": drift,
                    "human_approved_locally": False,
                    "training_eligible": False,
                }
                case["immutable_case_sha256"] = quality.json_hash(case)
                case["review"] = fresh_review()
                cases.append(case)
                mapping.append(
                    {
                        "case_id": case_id,
                        "source_id": item["id"],
                        "repeat": repeat,
                        "labels": {
                            label: {
                                "engine": engine,
                                "model_id": runs[(engine, lang, repeat)]["config"]["requested_model_id"],
                                "model_files_sha256": runs[(engine, lang, repeat)]["model"]["files_sha256"],
                                "run_file": entries[(engine, lang, repeat)]["file"],
                                "run_sha256": entries[(engine, lang, repeat)]["sha256"],
                                "raw_row_sha256": quality.json_hash(run_rows[(engine, lang, repeat)][item["id"]]),
                            }
                            for label, engine in labels.items()
                        },
                    }
                )
    packet = {
        "schema_version": 1,
        "packet_id": packet_id,
        "status": "unreviewed",
        "completed_input_inventory": True,
        "manifest_sha256": provenance["manifest_sha256"],
        "partition": partition,
        "repeats": repeats,
        "cases": cases,
        "human_approved_locally": False,
        "training_eligible": False,
        "blinding": "A/B mapping is deterministic and fixed across repeats for a source; it may change for another source. Keep the separate private key sealed. Text/style can reveal identity; anonymity is procedural, not guaranteed.",
        "attribution": {
            "dataset": "Google FLEURS (2022), Google and contributors",
            "revision": manifest["revision"],
            "license": "CC BY 4.0",
            "dataset_url": "https://huggingface.co/datasets/google/fleurs",
            "license_url": "https://creativecommons.org/licenses/by/4.0/",
            "changes": "Selected source/reference annotations preserved; model hypotheses and unfilled review fields added; no audio included.",
        },
        "review_limits": [
            "Meaning and terminology judgments are human tasks; automatic reference overlap is not approval.",
            "Public references may admit valid alternatives; assess the source before literal reference matching.",
            "All 18 reused EN→ES canaries are separate and have lexical targets only, no full reference translations.",
            "Repeats are repeated model outputs, not independent source/reference examples.",
        ],
        "counts": {
            "cases": len(cases),
            "unique_sources": len({r["source_id"] for r in cases}),
            "changed_cases": sum(r["outputs_differ"] for r in cases),
            "sources_with_repeat_drift": len({r["source_id"] for r in cases if any(r["repeat_output_drift"].values())}),
        },
    }
    key = {
        "schema_version": 1,
        "packet_id": packet_id,
        "private_do_not_give_to_reviewers": True,
        "blind_seed": seed,
        **provenance,
        "generator_sha256": quality.digest(Path(__file__)),
        "models": list(ENGINES),
        "cases": mapping,
        "input_index": index,
        "human_approved_locally": False,
        "training_eligible": False,
    }
    return packet, key


def literal_block(value):
    """Display arbitrary actual text literally, including Markdown/HTML/code fences."""
    if value is None:
        return "No full reference translation exists for this lexical canary."
    longest = max((len(part) for part in re.findall(r"`+", value)), default=0)
    fence = "`" * max(3, longest + 1)
    return f"{fence}text\n{value}\n{fence}"


def markdown(packet):
    lines = [
        "# Blinded bilingual translation review",
        "",
        f"Packet `{packet['packet_id']}` — **unreviewed**.",
        "",
        "A/B labels are fixed across repeats of one source and may change for another source. Keep the private mapping key separate. No model scores or timing information is shown.",
        "",
        "Record meaning errors separately from terminology preference. All ratings start unset; approval is false. Use `annotations.jsonl` with `review.schema.json`; preserve case IDs and immutable hashes. No training eligibility is granted.",
        "",
        f"{packet['counts']['cases']} cases; {packet['counts']['unique_sources']} unique sources; {packet['counts']['changed_cases']} cases with different A/B text; {packet['counts']['sources_with_repeat_drift']} sources with repeated-output drift.",
        "",
        "Repeated outputs are retained, not independent evidence. Public reference wording can have valid alternatives. Canaries below have only existing lexical targets and no full reference translations.",
        "",
        "Attribution: Google FLEURS (2022), Google and contributors, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Source/reference text is preserved; no audio is included.",
    ]
    for category, title in [
        ("fleurs_parallel", "Public parallel references"),
        ("theological_canary", "Separate reused EN→ES lexical canaries"),
    ]:
        selected = [row for row in packet["cases"] if row["category"] == category]
        lines += ["", f"## {title}", ""]
        changed = [row["case_id"] for row in selected if row["outputs_differ"]]
        lines += ["Different A/B output case IDs: " + (", ".join(f"`{value}`" for value in changed) or "None."), ""]
        for row in selected:
            lines += [
                f"### {row['case_id']} — {row['source_lang']} → {row['target_lang']}, repeat {row['repeat']}",
                "",
                f"Source ID: `{row['source_id']}`. Partition: `{row['partition']}`.",
                "",
                f"A/B text differs: {row['outputs_differ']}. Repeated-output drift: A={row['repeat_output_drift']['A']}, B={row['repeat_output_drift']['B']}.",
                "",
                "**Source**",
                "",
                literal_block(row["source"]),
                "",
                "**Reference**",
                "",
                literal_block(row["reference"]),
            ]
            if row["required_substrings"]:
                lines += [
                    "",
                    "**Existing lexical targets (not a full reference)**",
                    "",
                    literal_block(json.dumps(row["required_substrings"], ensure_ascii=False)),
                ]
            for label, text in row["hypotheses"].items():
                lines += ["", f"**{label}**", "", literal_block(text)]
            lines += [
                "",
                "Meaning error A / B: unset. Terminology preference: unset. Overall preference: unset. Approval: false.",
                "",
            ]
    return "\n".join(lines) + "\n"


def generate(directory, manifest_path, output, key_output, *, partition, repeats, seed):
    output, key_output = Path(output).resolve(), Path(key_output).resolve()
    if output.exists() or key_output.exists():
        raise FileExistsError("Preserve existing review/ratings/key; choose new output paths")
    if key_output.is_relative_to(output) or output.is_relative_to(key_output):
        raise ValueError("Private mapping key must be outside the reviewer packet directory")
    packet, key = make_packet(Path(directory), Path(manifest_path), partition=partition, repeats=repeats, seed=seed)
    output.mkdir(parents=True, exist_ok=False)
    key_output.parent.mkdir(parents=True, exist_ok=True)
    quality.save(output / "packet.json", packet, exclusive=True)
    with (output / "annotations.jsonl").open("x", encoding="utf-8") as handle:
        for row in packet["cases"]:
            annotation = {
                k: row[k]
                for k in (
                    "schema_version",
                    "case_id",
                    "immutable_case_sha256",
                    "human_approved_locally",
                    "training_eligible",
                    "review",
                )
            }
            handle.write(json.dumps(annotation, ensure_ascii=False) + "\n")
    with (output / "packet.md").open("x", encoding="utf-8") as handle:
        handle.write(markdown(packet))
    quality.save(output / "review.schema.json", review_schema(), exclusive=True)
    inventory = [
        {"file": p.name, "sha256": quality.digest(p), "size_bytes": p.stat().st_size} for p in sorted(output.iterdir())
    ]
    key["reviewer_artifacts"] = inventory
    encoded_key = json.dumps(key, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    fd = os.open(key_output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(encoded_key)
    quality.save(
        output / "packet-index.json",
        {
            "schema_version": 1,
            "completed": True,
            "packet_id": packet["packet_id"],
            "files": inventory,
            "private_key_sha256": quality.digest(key_output),
            "status": "unreviewed",
            "human_approved_locally": False,
            "training_eligible": False,
        },
        exclusive=True,
    )
    return packet


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("input", "manifest", "output", "key-output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--partition", choices=list(quality.PARTITIONS), default="development")
    parser.add_argument("--repeats", type=int, required=True)
    parser.add_argument(
        "--blind-seed",
        required=True,
        help="Private stable seed, at least 16 characters; stored only with the separate key",
    )
    args = parser.parse_args()
    result = generate(
        args.input,
        args.manifest,
        args.output,
        args.key_output,
        partition=args.partition,
        repeats=args.repeats,
        seed=args.blind_seed,
    )
    print(json.dumps({"packet_id": result["packet_id"], **result["counts"], "status": "unreviewed"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
