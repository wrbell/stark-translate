"""Portable correction imports must preserve provenance and newer human edits."""

import hashlib
from pathlib import Path

import pytest

from tools.merge_corrections import merge_translation, merge_whisper
from tools.review_data import atomic_jsonl, read_jsonl


def pair(**updates):
    return {
        "en": "Grace",
        "es": "Gracia",
        "sample_id": "live_en__1",
        "session_kind": "live",
        "source_lang": "en",
        "revision": 2,
        **updates,
    }


def whisper_export(root: Path, **updates) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "clip.wav").write_bytes(b"RIFFsame-audio")
    atomic_jsonl(
        root / "metadata.jsonl",
        [
            {
                "file_name": "clip.wav",
                "transcription": "Grace",
                "sample_id": "live_en__1",
                "session_kind": "live",
                "source_lang": "en",
                "revision": 2,
                **updates,
            }
        ],
    )
    return root


@pytest.mark.parametrize(
    "tag", [{"split": "eval"}, {"dataset_split": "test"}, {"is_eval": True}, {"session_kind": "replay"}]
)
def test_existing_evaluation_corpus_cannot_be_carried_into_new_output(tmp_path, tag):
    corrections = tmp_path / "corrections.jsonl"
    existing = tmp_path / "existing.jsonl"
    output = tmp_path / "new_train.jsonl"
    atomic_jsonl(corrections, [pair()])
    atomic_jsonl(existing, [pair(sample_id="existing", **tag)])
    with pytest.raises(ValueError):
        merge_translation(corrections, existing, output=output)
    assert not output.exists()
    audio = whisper_export(tmp_path / "audio")
    corpus = tmp_path / "corpus"
    atomic_jsonl(corpus / "metadata.jsonl", [{"transcription": "Old", **tag}])
    before = (corpus / "metadata.jsonl").read_bytes()
    with pytest.raises(ValueError):
        merge_whisper(audio, corpus)
    assert (corpus / "metadata.jsonl").read_bytes() == before
    assert not list(corpus.glob("*.wav"))


def test_existing_evaluation_path_cannot_be_renamed_to_training(tmp_path):
    corrections = tmp_path / "corrections.jsonl"
    atomic_jsonl(corrections, [pair()])
    existing = tmp_path / "eval" / "metadata.jsonl"
    atomic_jsonl(existing, [{"en": "Held out", "es": "Reservado"}])
    with pytest.raises(ValueError):
        merge_translation(corrections, existing, output=tmp_path / "new_train.jsonl")


@pytest.mark.parametrize("kind", [None, "unknown", "synthetic"])
def test_unknown_correction_provenance_never_enters_training(tmp_path, kind):
    path = tmp_path / "corrections.jsonl"
    atomic_jsonl(path, [pair(session_kind=kind)])
    with pytest.raises(ValueError):
        merge_translation(path, tmp_path / "train.jsonl")
    with pytest.raises(ValueError):
        merge_whisper(whisper_export(tmp_path / "audio", session_kind=kind), tmp_path / "train")


def test_older_revision_cannot_overwrite_newer_correction(tmp_path):
    path, target = tmp_path / "corrections.jsonl", tmp_path / "train.jsonl"
    atomic_jsonl(path, [pair()])
    merge_translation(path, target)
    atomic_jsonl(path, [pair(revision=1, en="Stale")])
    result = merge_translation(path, target)
    assert result["stale_skipped"] == 1 and result["updated"] == 0
    assert read_jsonl(target)[0]["en"] == "Grace"
    audio, corpus = whisper_export(tmp_path / "audio"), tmp_path / "train"
    merge_whisper(audio, corpus)
    before = (corpus / "metadata.jsonl").read_bytes()
    whisper_export(audio, revision=1, transcription="Stale")
    result = merge_whisper(audio, corpus)
    assert result["stale_skipped"] == 1 and result["updated"] == 0
    assert (corpus / "metadata.jsonl").read_bytes() == before


def test_equal_revision_conflicts_do_not_modify_any_audio_or_metadata(tmp_path):
    audio, corpus = whisper_export(tmp_path / "audio"), tmp_path / "train"
    merge_whisper(audio, corpus)
    before = {p.name: p.read_bytes() for p in corpus.iterdir()}
    atomic_jsonl(
        audio / "metadata.jsonl",
        [
            {**read_jsonl(audio / "metadata.jsonl")[0], "sample_id": "new"},
            {**read_jsonl(audio / "metadata.jsonl")[0], "transcription": "Conflicting text"},
        ],
    )
    with pytest.raises(ValueError, match="Conflicting content"):
        merge_whisper(audio, corpus)
    assert {p.name: p.read_bytes() for p in corpus.iterdir()} == before


def test_same_sample_id_cannot_silently_replace_audio(tmp_path):
    audio, corpus = whisper_export(tmp_path / "audio"), tmp_path / "train"
    merge_whisper(audio, corpus)
    (audio / "clip.wav").write_bytes(b"different recording")
    with pytest.raises(ValueError, match="audio identity"):
        merge_whisper(audio, corpus)


def test_translation_dedupe_tracks_pairs_after_revision_update(tmp_path):
    path, target = tmp_path / "corrections.jsonl", tmp_path / "train.jsonl"
    atomic_jsonl(path, [pair()])
    merge_translation(path, target)
    atomic_jsonl(
        path,
        [pair(revision=3, en="Mercy", es="Misericordia"), pair(sample_id="another", en="Mercy", es="Misericordia")],
    )
    result = merge_translation(path, target)
    assert result["updated"] == 1 and result["added"] == 0
    assert len(read_jsonl(target)) == 1


def test_whisper_separate_output_preserves_original_training_examples(tmp_path):
    corrections = whisper_export(tmp_path / "corrections")
    original = tmp_path / "original_train"
    original.mkdir()
    (original / "old.wav").write_bytes(b"existing-wave")
    atomic_jsonl(original / "metadata.jsonl", [{"file_name": "old.wav", "transcription": "Existing"}])
    output = tmp_path / "new_train"
    result = merge_whisper(corrections, original, output_dir=output)
    assert result["before"] == 1 and result["after"] == 2
    assert (output / "old.wav").read_bytes() == b"existing-wave"
    assert len(read_jsonl(original / "metadata.jsonl")) == 1
    assert merge_whisper(corrections, original, output_dir=output)["after"] == 2


def test_whisper_destination_symlink_cannot_write_outside_corpus(tmp_path):
    corrections = whisper_export(tmp_path / "corrections")
    target = tmp_path / "train"
    target.mkdir()
    outside = tmp_path / "unrelated.wav"
    outside.write_bytes(b"keep me")
    name = "al_" + hashlib.sha256(b"live_en__1").hexdigest()[:24] + ".wav"
    (target / name).symlink_to(outside)
    with pytest.raises(ValueError, match="destination audio"):
        merge_whisper(corrections, target)
    assert outside.read_bytes() == b"keep me"
    assert not (target / "metadata.jsonl").exists()


@pytest.mark.parametrize("revision", [-1, 1.5, "invalid", True])
def test_malformed_first_revision_cannot_enter_either_corpus(tmp_path, revision):
    path = tmp_path / "corrections.jsonl"
    atomic_jsonl(path, [pair(revision=revision)])
    with pytest.raises(ValueError, match="revision"):
        merge_translation(path, tmp_path / "train.jsonl")
    with pytest.raises(ValueError, match="revision"):
        merge_whisper(whisper_export(tmp_path / "audio", revision=revision), tmp_path / "train")
    assert not (tmp_path / "train.jsonl").exists()
    assert not (tmp_path / "train").exists()


@pytest.mark.parametrize(
    "flags",
    [
        {"transcript_approved": False},
        {"transcript_approved": "false"},
        {"transcript_approved": 1},
        {"approved_for_training": False},
        {"approved_for_training": "False"},
        {"training_eligible": False},
        {"training_eligible": "false"},
        {"training_eligible": "unknown"},
        {"review_status": "unapproved"},
        {"excluded": True},
        {"excluded": "true"},
    ],
)
def test_explicit_unapproved_or_excluded_rows_never_enter_either_corpus(tmp_path, flags):
    correction = tmp_path / "corrections.jsonl"
    atomic_jsonl(correction, [pair(**flags)])
    with pytest.raises(ValueError, match=r"Unapproved|Excluded"):
        merge_translation(correction, tmp_path / "train.jsonl")
    with pytest.raises(ValueError, match=r"Unapproved|Excluded"):
        merge_whisper(whisper_export(tmp_path / "audio", **flags), tmp_path / "train")
    assert not (tmp_path / "train.jsonl").exists()
    assert not (tmp_path / "train").exists()


def test_whisper_accepts_approved_transcript_without_bilingual_approval(tmp_path):
    flags = {"transcript_approved": True, "translation_approved": False}
    correction = tmp_path / "corrections.jsonl"
    atomic_jsonl(correction, [pair(**flags)])
    with pytest.raises(ValueError, match="translation_approved"):
        merge_translation(correction, tmp_path / "train.jsonl")
    audio = whisper_export(tmp_path / "audio", **flags)
    assert merge_whisper(audio, tmp_path / "train")["added"] == 1
    assert merge_whisper(audio, tmp_path / "train")["added"] == 0


def test_evaluation_usage_cannot_be_overridden_by_live_provenance(tmp_path):
    correction = tmp_path / "corrections.jsonl"
    atomic_jsonl(correction, [pair(usage="evaluation_only")])
    with pytest.raises(ValueError, match="Evaluation"):
        merge_translation(correction, tmp_path / "train.jsonl")
    with pytest.raises(ValueError, match="Evaluation"):
        merge_whisper(whisper_export(tmp_path / "audio", usage="evaluation_only"), tmp_path / "train")
    assert not (tmp_path / "train.jsonl").exists()
    assert not (tmp_path / "train").exists()


def test_csv_literal_boolean_approvals_preserve_historical_import(tmp_path):
    audio = whisper_export(tmp_path / "audio")
    (audio / "metadata.jsonl").unlink()
    (audio / "metadata.csv").write_text(
        "file_name,transcription,session_kind,source_lang,transcript_approved,excluded\n"
        "clip.wav,Grace,live,en,True,False\n"
    )
    assert merge_whisper(audio, tmp_path / "train")["added"] == 1


def test_existing_unapproved_records_cannot_be_carried_to_new_output(tmp_path):
    correction, original, output = (tmp_path / name for name in ["corrections.jsonl", "old.jsonl", "new.jsonl"])
    atomic_jsonl(correction, [pair()])
    atomic_jsonl(original, [pair(sample_id="old", translation_approved=False)])
    before = original.read_bytes()
    with pytest.raises(ValueError, match="translation_approved"):
        merge_translation(correction, original, output=output)
    assert original.read_bytes() == before and not output.exists()


@pytest.mark.parametrize("missing", [True, False])
def test_missing_or_empty_correction_input_cannot_report_success(tmp_path, missing):
    correction, target = tmp_path / "corrections.jsonl", tmp_path / "train.jsonl"
    if not missing:
        correction.write_text("\n")
    with pytest.raises((FileNotFoundError, ValueError)):
        merge_translation(correction, target)
    assert not target.exists()
    audio = tmp_path / "audio"
    audio.mkdir()
    if not missing:
        (audio / "metadata.jsonl").write_text("\n")
    with pytest.raises((FileNotFoundError, ValueError)):
        merge_whisper(audio, tmp_path / "train")
    assert not (tmp_path / "train").exists()


@pytest.mark.parametrize("alias", [False, True])
@pytest.mark.parametrize("as_output", [False, True])
def test_registered_v2_holdout_cannot_be_read_as_base_or_written_via_alias(tmp_path, monkeypatch, alias, as_output):
    from tools import merge_corrections as merger

    monkeypatch.setattr(merger, "PROJECT_ROOT", tmp_path)
    holdout = tmp_path / "bible_data/aligned/verse_pairs_test_v2.jsonl"
    atomic_jsonl(holdout, [{"en": "Held out text", "es": "Texto reservado"}])
    target = holdout
    if alias:
        target = tmp_path / "ordinary_name.jsonl"
        target.symlink_to(holdout)
    correction = tmp_path / "correction.jsonl"
    atomic_jsonl(correction, [pair()])
    before = holdout.read_bytes()
    with pytest.raises(ValueError, match="Evaluation"):
        merger.merge_translation(
            correction,
            tmp_path / "train.jsonl" if as_output else target,
            output=target if as_output else tmp_path / "scratch_train.jsonl",
        )
    assert holdout.read_bytes() == before
    assert not (tmp_path / "scratch_train.jsonl").exists()
