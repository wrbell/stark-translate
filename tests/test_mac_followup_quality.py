"""Public-data and child-runner contracts; no models or inference dependencies."""

import copy
import hashlib
import io
import json
import subprocess
import sys
import tarfile
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import mac_followup_quality as q


def wav(value, samples=1600):
    output = io.BytesIO()
    with wave.open(output, "wb") as stream:
        stream.setparams((1, 2, 16000, samples, "NONE", "not compressed"))
        stream.writeframes(value.to_bytes(2, "little", signed=True) * samples)
    return output.getvalue()


@pytest.fixture
def dataset(tmp_path):
    root, downloads = tmp_path / "audio-root", tmp_path / "downloads"
    manifest = {"dataset": "google/fleurs", "revision": q.REVISION, "records": [], "sources": []}
    refs = {"en": {1: "Grace is good.", 2: "He is the Lord."}, "es": {1: "La gracia es buena.", 2: "Él es el Señor."}}
    number = 0
    for lang, config in q.LANGUAGES.items():
        other = "es" if lang == "en" else "en"
        for partition, split in q.PARTITIONS.items():
            number += 1
            sentence = 1 if partition == "development" else 2
            filename = f"{number}.wav"
            reference = refs[lang][sentence]
            source = f"data/{config}/{split}.tsv"
            path = downloads / q.REVISION / source
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"{sentence}\t{filename}\t{reference}\t{q.normalize(reference)}\tx\t1600\tFEMALE\n")
            manifest["sources"].append({"file": source, "sha256": q.digest(path)})
            audio = wav(number)
            relative = f"audio/{lang}/{partition}/{filename}"
            local = root / relative
            local.parent.mkdir(parents=True, exist_ok=True)
            local.write_bytes(audio)
            source = f"data/{config}/audio/{split}.tar.gz"
            archive = downloads / q.REVISION / source
            archive.parent.mkdir(parents=True, exist_ok=True)
            with tarfile.open(archive, "w:gz") as tar:
                member = tarfile.TarInfo(f"{split}/{filename}")
                member.size = len(audio)
                tar.addfile(member, io.BytesIO(audio))
            manifest["sources"].append({"file": source, "sha256": q.digest(archive)})
            manifest["records"].append(
                {
                    "sentence_id": sentence,
                    "filename": filename,
                    "reference": reference,
                    "upstream_normalized_reference": q.normalize(reference),
                    "num_samples": 1600,
                    "gender": "FEMALE",
                    "path": relative,
                    "sha256": hashlib.sha256(audio).hexdigest(),
                    "sample_rate": 16000,
                    "id": f"fleurs-{lang}-{partition}-{sentence}-{number}",
                    "source_lang": lang,
                    "target_lang": other,
                    "partition": partition,
                    "translation_reference": refs[other][sentence],
                    "human_approved_locally": False,
                    "training_eligible": False,
                }
            )
    path = tmp_path / "source-manifest.json"
    path.write_text(json.dumps(manifest))
    return path, downloads, root


def test_audit_verifies_original_tsv_parallel_and_tar_bytes(dataset):
    manifest, audit = q.audit_manifest(*dataset)
    assert audit["completed"] and audit["records"] == 4
    assert audit["all_selected_audio_matches_original_archive"]
    assert audit["cross_partition_sentence_id_overlap"] == []
    assert len(manifest["canaries"]) == 18
    assert all(not r["training_eligible"] and not r["human_approved_locally"] for r in manifest["records"])
    english = manifest["records"][0]
    assert english["parallel_reference_evidence"]["matching_audio_filenames"] == ["3.wav"]
    assert "grace" in english["terms"]
    assert not Path(english["path"]).is_absolute()
    assert q.selected_items(manifest, "stt", "development", "en", False) == [english]
    q.validate_portable(manifest)


@pytest.mark.parametrize(
    "change,match",
    [
        (lambda m: m["records"][0].update(reference="Invented reference"), "annotation changed"),
        (lambda m: m["records"][0].update(translation_reference="Wrong translation"), "parallel reference"),
        (lambda m: m["records"][0].update(training_eligible=True), "eligibility"),
        (lambda m: m["records"][0].update(path="../escape.wav"), "safe relative"),
    ],
)
def test_audit_rejects_annotation_and_provenance_changes(dataset, change, match):
    path, downloads, audio_root = dataset
    m = q.read_json(path)
    change(m)
    path.write_text(json.dumps(m))
    with pytest.raises(ValueError, match=match):
        q.audit_manifest(path, downloads, audio_root)


def test_audit_checks_all_upstream_sentence_ids_not_only_selected(dataset):
    path, downloads, audio_root = dataset
    m = q.read_json(path)
    source = "data/es_419/test.tsv"
    tsv = downloads / q.REVISION / source
    tsv.write_text(tsv.read_text() + "1\t999.wav\tExtra\textra\tx\t1600\tMALE\n")
    next(s for s in m["sources"] if s["file"] == source)["sha256"] = q.digest(tsv)
    path.write_text(json.dumps(m))
    with pytest.raises(ValueError, match="Upstream sentence IDs"):
        q.audit_manifest(path, downloads, audio_root)


def test_original_archive_differs_even_if_local_hash_manifest_is_changed(dataset):
    path, downloads, audio_root = dataset
    m = q.read_json(path)
    row = m["records"][0]
    local = audio_root / row["path"]
    local.write_bytes(wav(22))
    row["sha256"] = q.digest(local)
    path.write_text(json.dumps(m))
    with pytest.raises(ValueError, match="Archive/local"):
        q.audit_manifest(path, downloads, audio_root)


def test_float_header_and_truncated_audio():
    fmt = __import__("struct").pack("<HHIIHH", 3, 1, 16000, 64000, 4, 32)
    body = b"WAVEfmt " + (16).to_bytes(4, "little") + fmt + b"data" + (8).to_bytes(4, "little") + b"\0" * 8
    raw = b"RIFF" + len(body).to_bytes(4, "little") + body
    assert q.wav_header(raw)["num_samples"] == 2
    with pytest.raises(ValueError, match="Truncated"):
        q.wav_header(raw[:-1])


def test_normalization_keeps_accents_numbers_and_word_boundaries():
    assert q.normalize(" ¡SÍ, Señor! \uff12\uff10—one ") == "sí señor 20 one"
    assert q.word_errors("one two three", "one four five") == {
        "errors": 2,
        "reference_words": 3,
        "substitutions": 2,
        "deletions": 0,
        "insertions": 0,
        "wer": 2 / 3,
    }
    assert q.word_errors("sí señor", "si") == {
        "errors": 2,
        "reference_words": 2,
        "substitutions": 1,
        "deletions": 1,
        "insertions": 0,
        "wer": 1.0,
    }
    assert q.word_errors("", "extra")["wer"] is None
    assert q.word_errors("one", "one extra")["insertions"] == 1
    assert not q.contains_term("faithfulness", "faith")
    assert q.contains_term("The HOLY SPIRIT!", "Holy Spirit")


def test_chrf_explicit_case_whitespace_and_corpus_counts():
    assert q.chrf_score(q.chrf_counts("Sí, Señor", "Sí, Señor")) == 100
    assert q.chrf_score(q.chrf_counts("a b c", "abc")) == 100
    assert q.chrf_score(q.chrf_counts("a", "A")) == 0
    assert q.chrf_score(q.chrf_counts("abc", "")) == 0
    # One-order corpus precision 1, recall .5 => F2 = 5*.5/(4+.5).
    counts = [[2, 1, 1], *[[0, 0, 0] for _ in range(5)]]
    assert q.chrf_score(counts) == pytest.approx(500 / 9)
    assert q.distribution([2204, 2044]) == {"n": 2, "p50": 2124, "p95": 2204}


def test_confirmation_is_explicit_and_canaries_are_separate(dataset):
    manifest, _ = q.audit_manifest(*dataset)
    args = q.parser().parse_args(["stt", "--manifest", "m", "--audio-root", "a", "--output", "o"])
    assert args.partition == "development"
    items = q.selected_items(manifest, "translate", "confirmation", "en", True)
    assert len(items) == 19
    assert items[0]["partition"] == "confirmation"
    assert all(i["partition"] == "reused_canary" and i["reference"] is None for i in items[1:])
    assert len(q.selected_items(manifest, "translate", "confirmation", "es", True)) == 1
    bad = copy.deepcopy(manifest)
    bad["records"][1]["sentence_id"] = bad["records"][0]["sentence_id"]
    with pytest.raises(ValueError, match="partition overlap"):
        q.validate_portable(bad)


@pytest.fixture
def worker_case(dataset, tmp_path, monkeypatch):
    manifest, _ = q.audit_manifest(*dataset)
    m = tmp_path / "portable.json"
    m.write_text(json.dumps(manifest))
    args = SimpleNamespace(
        manifest=m,
        audio_root=dataset[2],
        task="stt",
        partition="development",
        lang="es",
        include_canaries=False,
        repeat=0,
        engine="whisper-mlx",
        model_override=None,
        output=tmp_path / "worker.json",
    )
    monkeypatch.setattr(q, "source_identity", lambda: {"quality_source_sha256": {"test.py": "1"}})
    monkeypatch.setattr(q, "model_inventory", lambda config: {"resolved_path": "/local/model", "files_sha256": "2"})
    monkeypatch.setattr(q, "memory_snapshot", lambda: {"peak_rss_bytes": 100, "peak_metal_bytes": None})
    monkeypatch.setattr(q, "read_audio", lambda path, item: [0.1])
    return args, manifest


def test_worker_actual_flow_forces_language_and_records_timing(worker_case, monkeypatch):
    from engines.base import STTResult

    args, manifest = worker_case
    calls = []

    class Engine:
        model_id = "/local/model"

        def load(self):
            calls.append("load")

        def unload(self):
            calls.append("unload")

        def transcribe(self, audio, **kwargs):
            calls.append(kwargs)
            return STTResult("La gracia es buena.", 12, confidence=0.2)

    monkeypatch.setattr(q, "make_engine", lambda *a: Engine())
    assert q.run_worker(args)
    run = q.read_json(args.output)
    assert calls[1] == {"language": "es", "initial_prompt": None, "word_timestamps": False, "beam_size": 5}
    assert calls[-1] == "unload"
    assert run["rows"][0]["wer_counts"]["wer"] == 0
    assert run["rows"][0]["engine_latency_ms"] == 12
    assert run["rows"][0]["call_wall_ms"] >= 0
    assert run["model_identity"]["primary_identity_verified"]
    run["returncode"] = 0
    assert q.summarize_run(run, manifest)["eligible_for_comparison"]


@pytest.mark.parametrize("failure", ["startup_fallback", "per_call_fallback", "item_error"])
def test_worker_failures_never_count_as_success_and_release_engine(worker_case, monkeypatch, failure):
    from engines.base import STTResult

    args, manifest = worker_case
    calls = []

    class Engine:
        model_id = "/local/model"

        def load(self):
            if failure == "startup_fallback":
                self.model_id = "english-only-fallback"

        def unload(self):
            calls.append("unload")

        def transcribe(self, *a, **kwargs):
            if failure == "item_error":
                raise RuntimeError("broken decoder")
            return STTResult("bad", 5, used_fallback=True)

    monkeypatch.setattr(q, "make_engine", lambda *a: Engine())
    assert not q.run_worker(args)
    run = q.read_json(args.output)
    assert calls
    assert not run["completed"]
    run["returncode"] = 1
    assert not q.summarize_run(run, manifest)["eligible_for_comparison"]
    if failure == "startup_fallback":
        assert run["rows"] == []
        assert "identity mismatch" in run["error"]
    else:
        assert run["rows"][0]["status"] == "failed"


def test_serial_orchestrator_explicit_order_environment_and_failure_retention(dataset, tmp_path, monkeypatch):
    from tools import mac_evaluation

    manifest, _ = q.audit_manifest(*dataset)
    m = tmp_path / "portable.json"
    m.write_text(json.dumps(manifest))
    args = q.parser().parse_args(
        [
            "stt",
            "--manifest",
            str(m),
            "--audio-root",
            str(dataset[2]),
            "--output",
            str(tmp_path / "runs"),
            "--engines",
            "ct2-small",
            "ct2-base",
            "--repeats",
            "2",
        ]
    )
    monkeypatch.setattr(q, "source_identity", lambda: {})
    monkeypatch.setenv("STARK_EXPERIMENT_FIXED_PREFIX", "1")
    monkeypatch.setenv("STARK_PROFILE", "lite-cpu")
    seen = []

    def child(command, destination, supplied_manifest, *, env, timeout):
        assert "STARK_PROFILE" not in env and "STARK_EXPERIMENT_FIXED_PREFIX" not in env
        assert env["HF_HUB_OFFLINE"] == "1"
        assert timeout == 3600
        seen.append((command[command.index("--engine") + 1], command[command.index("--lang") + 1]))
        failed = len(seen) == 1
        q.save(destination, {"completed": not failed})
        return SimpleNamespace(returncode=1 if failed else 0)

    monkeypatch.setattr(mac_evaluation, "_run_evaluation_worker", child)
    result = q.run_serial(args)
    assert seen == [
        ("ct2-small", "en"),
        ("ct2-base", "en"),
        ("ct2-small", "es"),
        ("ct2-base", "es"),
        ("ct2-base", "en"),
        ("ct2-small", "en"),
        ("ct2-base", "es"),
        ("ct2-small", "es"),
    ]
    assert result["planned_runs"] == 8 and len(result["runs"]) == 8
    assert not result["completed"] and result["runs"][0]["returncode"] == 1
    with pytest.raises(FileExistsError):
        q.run_serial(args)


def test_import_and_help_are_model_free():
    code = "import sys; import tools.mac_followup_quality; assert not any(n in sys.modules for n in ('mlx','torch','numpy','soundfile','transformers','engines.mlx_engine'))"
    result = subprocess.run([sys.executable, "-c", code], cwd=q.ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_engine_construction_is_explicit_without_loading(monkeypatch):
    import engines.cuda_engine as cuda
    import engines.mlx_engine as mlx
    import engines.parakeet_mlx_engine as para

    made = []

    def fake(**kwargs):
        made.append(kwargs)
        return kwargs

    monkeypatch.setattr(cuda, "FasterWhisperEngine", fake)
    monkeypatch.setattr(mlx, "MLXWhisperEngine", fake)
    monkeypatch.setattr(mlx, "MLXGemmaEngine", fake)
    monkeypatch.setattr(para, "ParakeetMLXEngine", fake)
    for name in q.STT_MODELS:
        q.make_engine(q.engine_config(name, "stt"), "/frozen/model", "es")
    assert made[0] == {"model_id": "/frozen/model"}
    assert made[1]["session_language"] == "es" and made[1]["fallback_on_low_conf"] is False
    for config in made[2:]:
        assert config["device"] == "cpu" and config["local_files_only"] and not config["fallback_on_low_conf"]
    q.make_engine(q.engine_config("e4b", "translate"), "/frozen/gemma", "en")
    assert made[-1]["model_family"] == "gemma4" and not made[-1]["use_prompt_cache"]


def test_report_rejects_tampered_row_scores_and_preserves_failures(worker_case, monkeypatch, tmp_path):
    from engines.base import STTResult

    args, manifest = worker_case

    class Engine:
        model_id = "/local/model"

        def load(self):
            pass

        def unload(self):
            pass

        def transcribe(self, *a, **kwargs):
            return STTResult("La gracia es buena.", 12)

    monkeypatch.setattr(q, "make_engine", lambda *a: Engine())
    assert q.run_worker(args)
    run = q.read_json(args.output)
    run["returncode"] = 0
    assert q.summarize_run(run, manifest)["eligible_for_comparison"]
    run["rows"][0]["audio_sha256"] = "bad"
    assert not q.summarize_run(run, manifest)["eligible_for_comparison"]
    run["rows"][0]["audio_sha256"] = next(r["sha256"] for r in manifest["records"] if r["id"] == run["rows"][0]["id"])
    run["rows"][0]["wer_counts"]["wer"] = 0.6
    assert not q.summarize_run(run, manifest)["eligible_for_comparison"]
    q.save(args.output, run)
    index = {
        "manifest_sha256": q.digest(args.manifest),
        "planned_runs": 1,
        "completed": True,
        "runs": [
            {
                "file": args.output.name,
                "sha256": q.digest(args.output),
                "engine": args.engine,
                "source_lang": args.lang,
                "repeat": 0,
                "returncode": 0,
            }
        ],
    }
    q.save(tmp_path / "index.json", index)
    report = q.report(tmp_path, args.manifest, tmp_path / "report.json")
    assert not report["completed"] and not report["runs"][0]["eligible_for_comparison"]
    with pytest.raises(FileExistsError):
        q.report(tmp_path, args.manifest, tmp_path / "report.json")


def test_translation_worker_uses_fixed_upstream_text_and_canary_contract(worker_case, monkeypatch):
    from engines.base import TranslationResult

    args, manifest = worker_case
    args.task, args.engine, args.lang, args.include_canaries = "translate", "e4b", "en", True
    calls = []

    class Engine:
        model_id = "/local/model"

        def load(self):
            pass

        def unload(self):
            pass

        def translate(self, text, **kwargs):
            calls.append((text, kwargs))
            return TranslationResult("La gracia es buena.", 13, generated_tokens=7)

    monkeypatch.setattr(q, "make_engine", lambda *a: Engine())
    assert q.run_worker(args)
    run = q.read_json(args.output)
    run["returncode"] = 0
    assert len(calls) == 19
    assert calls[0] == ("Grace is good.", {"source_lang": "en", "target_lang": "es"})
    assert run["rows"][0]["chrf_counts"] == q.chrf_counts("La gracia es buena.", "La gracia es buena.")
    summary = q.summarize_run(run, manifest)
    assert summary["reference_items"] == 1 and summary["chrf"] == 100
    assert summary["canaries_total"] == 18 and summary["eligible_for_comparison"]
    assert all(r["chrf_counts"] is None for r in run["rows"][1:])


def test_missing_local_model_does_not_download(monkeypatch):
    import engines.model_paths as paths

    calls = []

    def missing(model, *, local_only):
        calls.append(local_only)
        return None

    monkeypatch.setattr(paths, "resolve_model_path", missing)
    with pytest.raises(FileNotFoundError, match="not installed locally"):
        q.model_inventory(q.engine_config("ct2-base", "stt"))
    assert calls == [True]
