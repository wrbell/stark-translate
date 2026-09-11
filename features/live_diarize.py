#!/usr/bin/env python3
"""Live diarization daemon (Phase 9.6.1).

Runs as a **separate process** from ``dry_run_ab.py`` so pyannote / ECAPA
never touch the MLX GPU pool. Labels are appended to a JSONL file that
``operator_app.features.LiveDiarizationWatcher`` tails.

Two modes
---------
* ``embed`` (default, Sunday path) — speaker embedding per final chunk +
  online cosine clustering. No rolling window required.
* ``pyannote`` — full speaker-diarization-3.1 pipeline on ``rolling.wav``.

``dry_run_ab.py --diarize`` starts this daemon and kills it on exit.

Usage::

    python features/live_diarize.py \\
        --rolling-wav stark_data/live_sessions/<sid>/rolling.wav \\
        --chunks-jsonl stark_data/live_sessions/<sid>/chunks.jsonl \\
        --output metrics/diarization_<sid>.jsonl \\
        --mode embed \\
        --interval-s 2
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import time
import wave
from pathlib import Path

# Allow ``python features/live_diarize.py`` (sys.path[0] is features/).
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from features.speaker_labels import (
    DEFAULT_CLUSTER_THRESHOLD,
    DEFAULT_MAX_SPEAKERS,
    OnlineSpeakerCluster,
    normalize_speaker_label,
)

logger = logging.getLogger("live_diarize")


def hf_token() -> str | None:
    """Return a Hugging Face token if the environment has one. Never logs the value."""
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        value = os.environ.get(key)
        if value:
            return value
    return None


def _load_pyannote():
    """Load the optional CPU pipeline with pinned local nested checkpoints."""
    token = hf_token()
    if not token:
        logger.warning("HF_TOKEN not set — pyannote needs a HuggingFace token; live diarization disabled")
        return None
    try:
        import yaml
        from pyannote.audio import Model
        from pyannote.audio.pipelines import SpeakerDiarization

        from engines.model_paths import pinned_hf_entry, resolve_model_for_loading

        source = "pyannote/speaker-diarization-3.1"
        entry = pinned_hf_entry(source)
        root = Path(resolve_model_for_loading(source, token=token))
        config = yaml.safe_load((root / "config.yaml").read_text())
        if config["pipeline"]["name"] != "pyannote.audio.pipelines.SpeakerDiarization":
            raise ValueError("Unsupported pinned pyannote pipeline class; review the model manifest")
        params = dict(config["pipeline"]["params"])
        # The upstream YAML names moving repositories. Resolve every nested
        # source ourselves and give the pipeline Model objects, never HF IDs.
        for name in ("segmentation", "embedding"):
            expected = entry["dependencies"][name]
            if params[name] != expected:
                raise ValueError(f"Unregistered pyannote {name} source; review the pinned pipeline config")
            checkpoint = Path(resolve_model_for_loading(expected, token=token)) / "pytorch_model.bin"
            if not checkpoint.is_file():
                raise ValueError(f"Missing local pyannote {name} checkpoint: {checkpoint}")
            params[name] = Model.from_pretrained(str(checkpoint), map_location="cpu", strict=False)
            if params[name] is None:
                raise ValueError(f"Pinned pyannote {name} checkpoint could not be loaded")
        pipeline = SpeakerDiarization(**params)
        pipeline.instantiate(config["params"])
        return pipeline
    except Exception as exc:
        logger.warning("pyannote unavailable: %s — live diarization disabled", exc)
        return None


def _emit(out_path: Path, record: dict) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def _emit_fake_label(out_path: Path, chunk_id: int, now: float | None = None, duration: float = 2.0) -> None:
    """Synthetic A/B labels for scaffolding tests (``--fake-labels`` only)."""
    now = time.time() if now is None else now
    speaker = "Speaker A" if chunk_id % 2 == 0 else "Speaker B"
    _emit(
        out_path,
        {
            "chunk_id": chunk_id,
            "speaker": speaker,
            "confidence": 0.5,
            "ts": now,
            "start_ts": now - duration,
            "end_ts": now,
        },
    )


def _read_rolling_meta(wav_path: Path) -> dict:
    meta_path = wav_path.with_name("rolling_meta.json")
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _wav_duration_s(wav_path: Path) -> float:
    try:
        with wave.open(str(wav_path), "rb") as wf:
            rate = wf.getframerate() or 1
            return wf.getnframes() / float(rate)
    except Exception:
        return 0.0


def _as_rel_time(segment, attr: str, default: float) -> float:
    val = getattr(segment, attr, default)
    try:
        return float(val)
    except (TypeError, ValueError):
        return float(default)


def _emit_real_label(out_path: Path, chunk_id: int, run_diarization, wav_path: Path) -> None:
    """Compat wrapper used by coverage tests; emits pyannote segments to JSONL."""
    emit_pyannote_segments(out_path, chunk_id, run_diarization, wav_path)


def emit_pyannote_segments(
    out_path: Path,
    chunk_id: int,
    run_diarization,
    wav_path: Path,
    *,
    min_speakers: int = 1,
    max_speakers: int = 4,
) -> int:
    """Run pyannote on the rolling WAV and emit one JSONL row per segment."""
    try:
        diarization = run_diarization(str(wav_path), min_speakers=min_speakers, max_speakers=max_speakers)
    except Exception as exc:
        logger.warning("diarization failed for chunk %s: %s", chunk_id, exc)
        return 0

    meta = _read_rolling_meta(wav_path)
    window_start = meta.get("window_start_ts")
    if window_start is None:
        duration = float(meta.get("duration_s") or _wav_duration_s(wav_path))
        window_start = time.time() - duration

    emitted = 0
    now = time.time()
    try:
        for segment, _, label in diarization.itertracks(yield_label=True):
            start_rel = _as_rel_time(segment, "start", 0.0)
            end_rel = _as_rel_time(segment, "end", start_rel)
            record = {
                "chunk_id": chunk_id,
                "segment_i": emitted,
                "speaker": normalize_speaker_label(str(label)),
                "confidence": 1.0,
                "ts": now,
                "start_ts": float(window_start) + start_rel,
                "end_ts": float(window_start) + end_rel,
            }
            _emit(out_path, record)
            emitted += 1
    except Exception as exc:
        logger.warning("diarization parse failed for chunk %s: %s", chunk_id, exc)
        return emitted
    return emitted


class FakeEmbedder:
    """Deterministic 2-d embedding from PCM energy. Tests only (``--embedder fake``)."""

    def embed(self, wav_path: str) -> list[float]:
        path = Path(wav_path)
        try:
            with wave.open(str(path), "rb") as wf:
                nframes = wf.getnframes()
                raw = wf.readframes(nframes)
                width = wf.getsampwidth()
        except Exception:
            data = path.read_bytes() if path.exists() else b""
            if not data:
                return [1.0, 0.0]
            # Distinct files hash to distinct poles without loading models.
            total = sum(data) / (len(data) * 255.0)
            total = max(0.0, min(1.0, total))
            return [1.0 - total, total]
        if width != 2 or not raw:
            return [1.0, 0.0]
        n = len(raw) // 2
        acc = 0
        for i in range(0, len(raw), 2):
            acc += abs(int.from_bytes(raw[i : i + 2], "little", signed=True))
        mean_abs = acc / (n * 32768.0) if n else 0.0
        mean_abs = max(0.0, min(1.0, mean_abs))
        return [1.0 - mean_abs, mean_abs]


def _load_speechbrain_embedder():
    """Public ECAPA (Apache-2.0, not gated). Returns None if import/load fails."""
    try:
        from speechbrain.inference.speaker import EncoderClassifier
    except Exception:
        try:
            from speechbrain.pretrained import EncoderClassifier  # type: ignore[attr-defined]
        except Exception as exc:
            logger.info("SpeechBrain ECAPA unavailable: %s", exc)
            return None
    try:
        import inspect

        from engines.model_paths import pinned_hf_entry, resolve_model_for_loading

        source = "speechbrain/spkrec-ecapa-voxceleb"
        revision = pinned_hf_entry(source)["revision"]
        local_source = resolve_model_for_loading(source)
        # SpeechBrain 1.0.x takes revision directly. 1.1.x instead forwards a
        # FetchConfig; sending revision via **kwargs there reaches the model
        # constructor rather than the downloader. Both use local YAML/weights.
        fetch_kwargs = {}
        if "revision" in inspect.signature(EncoderClassifier.from_hparams).parameters:
            fetch_kwargs["revision"] = revision
        else:
            from speechbrain.utils.fetching import FetchConfig

            fetch_kwargs["fetch_config"] = FetchConfig(revision=revision, allow_network=False)
        classifier = EncoderClassifier.from_hparams(
            source=local_source,
            # The pinned YAML otherwise sends pretrainer.collect_files back
            # to the bare remote repo independently of the source argument.
            overrides={"pretrained_path": local_source},
            run_opts={"device": "cpu"},
            **fetch_kwargs,
        )
    except Exception as exc:
        logger.warning("SpeechBrain ECAPA failed to load: %s", exc)
        return None

    class _SB:
        def embed(self, wav_path: str) -> list[float]:
            import numpy as np
            import torchaudio

            signal, _fs = torchaudio.load(wav_path)
            emb = classifier.encode_batch(signal)
            arr = emb.squeeze().detach().cpu().numpy()
            return [float(x) for x in np.asarray(arr).reshape(-1)]

    return _SB()


def _load_pyannote_embedder():
    token = hf_token()
    if not token:
        return None
    try:
        from pyannote.audio import Inference
    except Exception as exc:
        logger.info("pyannote embedding unavailable: %s", exc)
        return None
    try:
        from engines.model_paths import resolve_model_for_loading

        checkpoint = Path(resolve_model_for_loading("pyannote/embedding", token=token)) / "pytorch_model.bin"
        if not checkpoint.is_file():
            raise ValueError(f"Missing pinned pyannote embedding checkpoint: {checkpoint}")
        # Inference accepts a local checkpoint and passes it to Model's local
        # file branch. No unsupported revision kwarg or bare HF ID is passed.
        inference = Inference(str(checkpoint), window="whole")
    except Exception as exc:
        logger.warning("pyannote/embedding failed to load: %s", exc)
        return None

    class _PA:
        def embed(self, wav_path: str) -> list[float]:
            import numpy as np

            emb = inference(wav_path)
            arr = np.asarray(emb).reshape(-1)
            return [float(x) for x in arr]

    return _PA()


def load_embedder(kind: str = "auto"):
    """Return an object with ``embed(wav_path) -> list[float]``, or None."""
    if kind == "fake":
        return FakeEmbedder()
    # SpeechBrain ECAPA is not gated; try it before pyannote/embedding.
    embedder = _load_speechbrain_embedder()
    if embedder is not None:
        logger.info("embedder: speechbrain/spkrec-ecapa-voxceleb (CPU)")
        return embedder
    embedder = _load_pyannote_embedder()
    if embedder is not None:
        logger.info("embedder: pyannote/embedding (CPU)")
        return embedder
    logger.warning("No speaker embedder available (SpeechBrain/pyannote) — live diarization disabled")
    return None


def _resolve_wav(wav_field: str, session_dir: Path | None, rolling_wav: Path) -> Path:
    path = Path(wav_field)
    if path.is_absolute() and path.exists():
        return path
    bases = []
    if session_dir is not None:
        bases.append(session_dir)
    bases.append(rolling_wav.parent)
    for base in bases:
        candidate = base / wav_field
        if candidate.exists():
            return candidate
    return (session_dir or rolling_wav.parent) / wav_field


def read_new_jsonl_records(path: Path, byte_offset: int) -> tuple[list[dict], int]:
    """Tail a JSONL file from ``byte_offset``. Returns (records, new_offset)."""
    if not path.exists():
        return [], byte_offset
    try:
        with path.open("rb") as f:
            f.seek(byte_offset)
            chunk = f.read()
            new_offset = f.tell()
    except OSError:
        return [], byte_offset
    if not chunk:
        return [], byte_offset
    records: list[dict] = []
    for line in chunk.decode("utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(rec, dict):
            records.append(rec)
    return records, new_offset


def process_embed_chunks(
    records: list[dict],
    embedder,
    cluster: OnlineSpeakerCluster,
    out_path: Path,
    *,
    session_dir: Path | None,
    rolling_wav: Path,
    label_id_start: int = 0,
) -> int:
    """Embed each chunk record, cluster, emit JSONL. Returns number emitted."""
    emitted = 0
    now = time.time()
    for rec in records:
        wav_field = rec.get("wav") or rec.get("audio_path") or rec.get("path")
        if not wav_field:
            continue
        wav_path = _resolve_wav(str(wav_field), session_dir, rolling_wav)
        if not wav_path.exists():
            logger.debug("chunk wav not yet present: %s", wav_path)
            continue
        try:
            embedding = embedder.embed(str(wav_path))
        except Exception as exc:
            logger.warning("embed failed for %s: %s", wav_path, exc)
            continue
        speaker, confidence = cluster.assign(list(embedding))
        start_ts = rec.get("start_ts")
        end_ts = rec.get("end_ts")
        try:
            start_ts_f = float(start_ts) if start_ts is not None else now
            end_ts_f = float(end_ts) if end_ts is not None else start_ts_f
        except (TypeError, ValueError):
            start_ts_f = now
            end_ts_f = now
        try:
            src_chunk = int(rec.get("chunk_id", label_id_start + emitted) or 0)
        except (TypeError, ValueError):
            src_chunk = label_id_start + emitted
        _emit(
            out_path,
            {
                "chunk_id": src_chunk,
                "speaker": speaker,
                "confidence": round(float(confidence), 3),
                "ts": now,
                "start_ts": start_ts_f,
                "end_ts": end_ts_f,
            },
        )
        emitted += 1
    return emitted


def build_daemon_command(
    *,
    python: str,
    script: str,
    rolling_wav: str,
    output: str,
    mode: str = "embed",
    interval_s: float = 2.0,
    session_dir: str | None = None,
    chunks_jsonl: str | None = None,
    embedder: str = "auto",
    fake_labels: bool = False,
    extra: list[str] | None = None,
) -> list[str]:
    """Argv used by ``dry_run_ab`` to spawn this daemon."""
    cmd = [
        python,
        "-u",
        script,
        "--rolling-wav",
        rolling_wav,
        "--output",
        output,
        "--mode",
        mode,
        "--interval-s",
        str(interval_s),
        "--embedder",
        embedder,
    ]
    if session_dir:
        cmd += ["--session-dir", session_dir]
    if chunks_jsonl:
        cmd += ["--chunks-jsonl", chunks_jsonl]
    if fake_labels:
        cmd.append("--fake-labels")
    if extra:
        cmd.extend(extra)
    return cmd


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rolling-wav", required=True, help="Rolling WAV file written by dry_run_ab")
    parser.add_argument("--output", required=True, help="JSONL output path (LiveDiarizationWatcher tails this)")
    parser.add_argument(
        "--mode",
        choices=["embed", "pyannote"],
        default="embed",
        help="embed (default, per-chunk clustering) or pyannote (full pipeline on rolling WAV)",
    )
    parser.add_argument("--chunks-jsonl", default=None, help="Index of per-final chunk WAVs (embed mode)")
    parser.add_argument("--session-dir", default=None, help="Session directory containing chunk_*.wav")
    parser.add_argument("--interval-s", type=float, default=2.0, help="Polling interval (default: 2)")
    parser.add_argument(
        "--max-iters",
        type=int,
        default=0,
        help="Max polling iterations before exit (0 = run forever; useful for tests)",
    )
    parser.add_argument(
        "--embedder",
        choices=["auto", "fake"],
        default="auto",
        help="auto = SpeechBrain ECAPA then pyannote/embedding; fake = PCM-energy stub",
    )
    parser.add_argument(
        "--cluster-threshold",
        type=float,
        default=DEFAULT_CLUSTER_THRESHOLD,
        help=f"Cosine threshold for a new speaker (default: {DEFAULT_CLUSTER_THRESHOLD})",
    )
    parser.add_argument(
        "--max-speakers",
        type=int,
        default=DEFAULT_MAX_SPEAKERS,
        help=f"Cap on online clusters (default: {DEFAULT_MAX_SPEAKERS})",
    )
    parser.add_argument(
        "--fake-labels",
        action="store_true",
        default=False,
        help="Emit alternating Speaker A/B without models (tests only)",
    )
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s [%(levelname)s] %(message)s")

    rolling_wav = Path(args.rolling_wav)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    session_dir = Path(args.session_dir) if args.session_dir else rolling_wav.parent
    chunks_jsonl = Path(args.chunks_jsonl) if args.chunks_jsonl else session_dir / "chunks.jsonl"

    stopped = {"flag": False}

    def _stop(*_):
        stopped["flag"] = True

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    run_diarization = None
    pyannote_attempted = False
    embedder = None
    embedder_failed = False
    cluster = OnlineSpeakerCluster(threshold=args.cluster_threshold, max_speakers=args.max_speakers)

    if args.fake_labels:
        logger.warning("--fake-labels set: emitting synthetic Speaker A/B (tests only)")

    chunk_id = 0
    iters = 0
    chunks_offset = 0
    logger.info(
        "live_diarize started: mode=%s rolling_wav=%s output=%s interval=%ss",
        args.mode,
        rolling_wav,
        out_path,
        args.interval_s,
    )

    while not stopped["flag"]:
        iters += 1
        if args.fake_labels:
            if rolling_wav.exists() or chunks_jsonl.exists():
                _emit_fake_label(out_path, chunk_id)
                chunk_id += 1
        elif args.mode == "embed":
            new_records, chunks_offset = read_new_jsonl_records(chunks_jsonl, chunks_offset)
            standalone_rolling = rolling_wav.exists() and not chunks_jsonl.exists()
            has_work = bool(new_records) or standalone_rolling
            if has_work and embedder is None and not embedder_failed:
                embedder = load_embedder(args.embedder)
                if embedder is None:
                    embedder_failed = True
                    logger.warning("embed mode disabled — daemon idle")
            if has_work and embedder is not None:
                to_process = new_records
                if not to_process and standalone_rolling:
                    to_process = [
                        {
                            "chunk_id": chunk_id,
                            "wav": str(rolling_wav),
                            "start_ts": time.time() - _wav_duration_s(rolling_wav),
                            "end_ts": time.time(),
                        }
                    ]
                emitted = process_embed_chunks(
                    to_process,
                    embedder,
                    cluster,
                    out_path,
                    session_dir=session_dir,
                    rolling_wav=rolling_wav,
                    label_id_start=chunk_id,
                )
                chunk_id += emitted
        elif args.mode == "pyannote":
            if rolling_wav.is_file() and _wav_duration_s(rolling_wav) > 0:
                # The pinned helper loads models, so keep it behind actual
                # file input readiness. Missing/partial WAVs must not trigger
                # downloads, and unavailable optional models are tried once.
                if not pyannote_attempted:
                    pyannote_attempted = True
                    run_diarization = _load_pyannote()
                    if run_diarization is None:
                        logger.warning("pyannote mode unavailable — daemon idle; restart after resolving model access")
                if run_diarization is not None:
                    emitted = emit_pyannote_segments(out_path, chunk_id, run_diarization, rolling_wav)
                    if emitted:
                        chunk_id += 1
            else:
                logger.debug("rolling WAV not yet ready at %s, waiting…", rolling_wav)
        elif not rolling_wav.exists():
            logger.debug("rolling WAV not yet present at %s, waiting…", rolling_wav)

        if args.max_iters and iters >= args.max_iters:
            break
        time.sleep(args.interval_s)

    logger.info("live_diarize stopped after %d iterations", iters)
    return 0


if __name__ == "__main__":
    sys.exit(main())
