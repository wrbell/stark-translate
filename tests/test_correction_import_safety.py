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
