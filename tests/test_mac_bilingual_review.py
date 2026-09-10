"""Blinded review uses real manifest shapes and synthetic, unapproved worker fixtures."""

import copy
import json
import subprocess
import sys

import pytest

from tools import mac_bilingual_review as review
from tools import mac_followup_quality as quality


@pytest.fixture
def cohort(tmp_path):
    manifest = quality.read_json(quality.ROOT / "docs/evaluation/mac_followup_20260910/public_data/manifest.json")
    manifest["records"] = [
        next(r for r in manifest["records"] if r["source_lang"] == lang and r["partition"] == part)
        for lang in ("en", "es")
        for part in ("development", "confirmation")
    ]
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    directory = tmp_path / "runs"
    directory.mkdir()
    index = dict(
        schema_version=1,
        completed=True,
        task="translate",
        partition="development",
        serial=True,
        planned_runs=8,
        manifest_sha256=quality.digest(manifest_path),
        runs=[],
    )
    for repeat in range(2):
        for lang in ("en", "es"):
            for engine in ("e4b", "e2b"):
                items = quality.selected_items(manifest, "translate", "development", lang, True)
                rows = []
                for item in items:
                    hypothesis = item["reference"] or " ".join(item["required_substrings"])
                    if engine == "e2b":
                        hypothesis += "!"
                    if repeat == 1 and engine == "e4b" and item["id"] == items[0]["id"] and lang == "en":
                        hypothesis += " Again."
                    terms = item.get("required_substrings", [])
                    rows.append(
                        {
                            **item,
                            "hypothesis": hypothesis,
                            "status": "ok",
                            "call_wall_ms": 100,
                            "engine_latency_ms": 90,
                            "chrf_counts": quality.chrf_counts(item["reference"], hypothesis)
                            if item["reference"]
                            else None,
                            "canary_pass": all(t.casefold() in hypothesis.casefold() for t in terms) if terms else None,
                        }
                    )
                model_files = [
                    dict(path="weights.safetensors", sha256=("a" if engine == "e4b" else "b") * 64, size_bytes=200)
                ]
                resolved = f"/fixture/model/{engine}"
                run = dict(
                    schema_version=1,
                    completed=True,
                    status="completed",
                    task="translate",
                    engine=engine,
                    source_lang=lang,
                    partition="development",
                    repeat=repeat,
                    returncode=0,
                    manifest_sha256=index["manifest_sha256"],
                    items_sha256=quality.json_hash(items),
                    expected_ids=[r["id"] for r in items],
                    rows=rows,
                    normalization=quality.NORMALIZATION,
                    chrf_contract=quality.CHRF_CONTRACT,
                    config=quality.engine_config(engine, "translate"),
                    model=dict(resolved_path=resolved, files=model_files, files_sha256=quality.json_hash(model_files)),
                    model_identity=dict(
                        requested_model_id=resolved, actual_model_id=resolved, primary_identity_verified=True
                    ),
                    environment=dict(
                        quality_source_sha256={"engines/mlx_engine.py": "c" * 64}, versions={"mlx": "0.32.2"}
                    ),
                    human_approved_locally=False,
                    training_eligible=False,
                )
                path = directory / f"translate_{engine}_{lang}_development_r{repeat}.json"
                path.write_text(json.dumps(run))
                index["runs"].append(
                    dict(
                        file=path.name,
                        sha256=quality.digest(path),
                        engine=engine,
                        source_lang=lang,
                        repeat=repeat,
                        returncode=0,
                        completed=True,
                    )
                )
    (directory / "index.json").write_text(json.dumps(index))
    return directory, manifest_path, tmp_path


def mutate_run(cohort, fn):
    directory, _, _ = cohort
    index = quality.read_json(directory / "index.json")
    entry = index["runs"][0]
    path = directory / entry["file"]
    run = quality.read_json(path)
    fn(run)
    path.write_text(json.dumps(run))
    entry["sha256"] = quality.digest(path)
    (directory / "index.json").write_text(json.dumps(index))


def make(cohort):
    return review.make_packet(cohort[0], cohort[1], partition="development", repeats=2, seed="private-unit-test-seed")


def test_exact_hypotheses_blind_mapping_canaries_and_repeat_drift(cohort):
    packet, key = make(cohort)
    assert packet["counts"] == dict(cases=40, unique_sources=20, changed_cases=40, sources_with_repeat_drift=1)
    assert packet["status"] == "unreviewed" and not packet["human_approved_locally"] and not packet["training_eligible"]
    source_labels = {}
    by_key = {r["case_id"]: r for r in key["cases"]}
    for case in packet["cases"]:
        private = by_key[case["case_id"]]
        labels = {label: item["engine"] for label, item in private["labels"].items()}
        prior = source_labels.setdefault(case["source_id"], labels)
        assert labels == prior
        for label, mapping in private["labels"].items():
            run = quality.read_json(cohort[0] / mapping["run_file"])
            raw = next(r for r in run["rows"] if r["id"] == case["source_id"])
            assert case["source"] == raw["source"] and case["reference"] == raw["reference"]
            assert case["hypotheses"][label] == raw["hypothesis"]
            assert mapping["raw_row_sha256"] == quality.json_hash(raw)
        assert case["review"] == review.fresh_review()
        if case["category"] == "theological_canary":
            assert case["reference"] is None and case["reference_text_sha256"] is None
            assert case["partition"] == "reused_canary" and case["source_lang"] == "en" and case["target_lang"] == "es"
    assert len([c for c in packet["cases"] if c["category"] == "theological_canary"]) == 36
    assert len({tuple(sorted(labels.items())) for labels in source_labels.values()}) == 2
    public = json.dumps(packet)
    assert "private-unit-test-seed" not in public
    assert "mlx-community/gemma-" not in public and "run_file" not in public and "call_wall_ms" not in public
    assert "e4b" not in json.dumps(packet["cases"][0]["review"])


def test_seed_reproduces_mapping_but_changes_anonymous_identities(cohort):
    first, key = make(cohort)
    assert make(cohort) == (first, key)
    other, other_key = review.make_packet(
        cohort[0], cohort[1], partition="development", repeats=2, seed="a-different-private-seed"
    )
    assert other["packet_id"] != first["packet_id"]
    assert other_key["cases"] != key["cases"]
    assert [r["source"] for r in first["cases"]] == [r["source"] for r in other["cases"]]


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda r: r.update(completed=False, status="failed"), "Failed/incomplete"),
        (lambda r: r.update(source_lang="es"), "identity mismatch"),
        (lambda r: r.update(partition="confirmation"), "partition mismatch"),
        (lambda r: r["config"].update(requested_model_id="wrong-model"), "model/configuration"),
        (lambda r: r["model_identity"].update(actual_model_id="fallback"), "model identity"),
        (lambda r: r["environment"]["quality_source_sha256"].update(other="d" * 64), "mixed"),
        (lambda r: r["model"].update(files_sha256="d" * 64), "inventory hash"),
        (lambda r: r["rows"][0].update(reference="invented"), "Invalid raw"),
        (lambda r: r.update(expected_ids=r["expected_ids"][:-1]), "Missing input/canary"),
        (lambda r: r.update(human_approved_locally=True), "unapproved"),
    ],
)
def test_invalid_failed_or_mixed_cohorts_fail_before_output(cohort, mutation, match):
    mutate_run(cohort, mutation)
    output = cohort[2] / "packet"
    with pytest.raises(ValueError, match=match):
        review.generate(
            cohort[0],
            cohort[1],
            output,
            cohort[2] / "key.json",
            partition="development",
            repeats=2,
            seed="private-unit-test-seed",
        )
    assert not output.exists()


@pytest.mark.parametrize("change", ["missing", "duplicate", "incomplete", "hash"])
def test_missing_duplicate_and_unbound_run_inventory_rejected(cohort, change):
    path = cohort[0] / "index.json"
    index = quality.read_json(path)
    if change == "missing":
        index["runs"].pop()
    elif change == "duplicate":
        index["runs"][-1] = copy.deepcopy(index["runs"][0])
    elif change == "incomplete":
        index["completed"] = False
    else:
        index["runs"][0]["sha256"] = "0" * 64
    path.write_text(json.dumps(index))
    with pytest.raises(ValueError):
        make(cohort)


def test_missing_canary_is_not_invented_or_silently_skipped(cohort):
    path = cohort[1]
    manifest = quality.read_json(path)
    manifest["canaries"].pop()
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="all 18"):
        make(cohort)


def test_packet_generation_typed_annotations_private_key_and_immutability(cohort):
    directory, manifest, tmp = cohort
    before = {p.name: quality.digest(p) for p in directory.iterdir()}
    output, key = tmp / "reviewer", tmp / "private" / "mapping.json"
    packet = review.generate(
        directory, manifest, output, key, partition="development", repeats=2, seed="private-unit-test-seed"
    )
    annotations = [json.loads(line) for line in (output / "annotations.jsonl").read_text().splitlines()]
    assert len(annotations) == 40
    for row in annotations:
        assert row["review"] == review.fresh_review() and row["training_eligible"] is False
        assert set(row) == set(review.review_schema()["required"])
    index = quality.read_json(output / "packet-index.json")
    assert index["completed"] and index["private_key_sha256"] == quality.digest(key)
    for item in index["files"]:
        assert item["sha256"] == quality.digest(output / item["file"])
    assert (output / "packet.md").read_text().count("### ") == 40
    assert {p.name: quality.digest(p) for p in directory.iterdir()} == before
    (output / "annotations.jsonl").write_text("existing human draft")
    with pytest.raises(FileExistsError):
        review.generate(
            directory, manifest, output, key, partition="development", repeats=2, seed="private-unit-test-seed"
        )
    assert (output / "annotations.jsonl").read_text() == "existing human draft"
    with pytest.raises(ValueError, match="outside"):
        review.generate(
            directory,
            manifest,
            tmp / "new",
            tmp / "new" / "key.json",
            partition="development",
            repeats=2,
            seed="private-unit-test-seed",
        )
    assert packet["completed_input_inventory"]


def test_markdown_actual_text_is_literal_without_hidden_rewriting():
    text = "```\n<script>bad</script>\n| table |\n`````\n"
    block = review.literal_block(text)
    assert block.startswith("``````text\n")
    assert text in block
    assert review.literal_block(None).startswith("No full reference")


def test_import_is_model_free():
    script = "import sys; import tools.mac_bilingual_review; assert not any(k in sys.modules for k in ['mlx','torch','numpy','soundfile','parakeet_mlx'])"
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
