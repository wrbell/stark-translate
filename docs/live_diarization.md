# Live diarization on a rolling buffer (Phase 9.6.1 / #133)

Opt-in speaker labels on live finals. **Off by default** (`--diarize`). The
STT/translation path is unchanged; diarization runs in a separate process and
must not move final p95 by more than **+50 ms**.

## Architecture

```
mic → VAD → STT ∥ translation          GPU / _pipeline_pool (untouched)
                 │
                 ▼
            result_data  ── JSONL lookup (cheap) ──► speaker field
                 │
                 ▼  _io_pool only
         chunk_XXXX.wav
         rolling.wav  (last 20–30 s of *speech*)
         chunks.jsonl (chunk_id, wav, start_ts, end_ts)
                 │
                 ▼  subprocess, killed on exit
     features/live_diarize.py --mode embed|pyannote
                 │
                 ▼
     metrics/diarization_<SESSION>.jsonl
       {chunk_id, speaker, confidence, ts, start_ts, end_ts}
                 │
                 ├── dry_run_ab overlap-assigns the next final
                 └── LiveDiarizationWatcher tails it for the operator UI
```

`dry_run_ab.py --diarize` does three isolated things:

1. After `result_data` is built, look up `speaker` by timestamp overlap
   (`features/speaker_labels.py`) and put it on the WebSocket payload, CSV
   (trailing column), and diagnostics JSONL. **No models on this path.**
2. On `_io_pool` (after the per-chunk WAV already saved for fine-tuning),
   refresh `stark_data/live_sessions/<SESSION>/rolling.wav` — concatenation of
   the last ~25 s of speech — plus `chunks.jsonl` and `rolling_meta.json`.
3. Spawn `features/live_diarize.py` as a subprocess (`start_new_session=True`)
   and SIGTERM the process group on exit. A daemon crash is logged; the
   pipeline keeps captioning unlabeled.

The operator control plane passes `--diarize` when the **Live diarization**
checkbox is on (`SessionConfig.diarize`). The watcher binds to
`metrics/diarization_<session_id>.jsonl` at session start.

Wall-clock intervals for a final use `_utterance_start_times` (perf_counter of
the first speech frame) plus audio duration — **not** "now", so translation
latency does not stretch the labeled span.

Daemon JSONL (current)::

    {"chunk_id": 12, "speaker": "Speaker A", "confidence": 0.91,
     "ts": 1714060800.5, "start_ts": 1714060798.1, "end_ts": 1714060800.4}

Legacy PR #70 records (`ts` only, no `start_ts`/`end_ts`) still parse: `ts` is
treated as a point interval so overlap assignment keeps working.

Because the daemon labels a chunk *after* that final is broadcast, the current
utterance often has no overlap yet. Assignment then **carry-forwards** the
latest segment that started at or before `utterance_end` — the Sunday pattern
(long turns, 1–3 voices). The first labeled chunk of a session may still be
unlabeled on the projector; the operator pill updates from the JSONL tail
independently.

## Two labelling modes

### `pyannote` — full pipeline on the rolling window

Reuses `features.diarize.run_diarization` (`pyannote/speaker-diarization-3.1`,
CNRS / pyannote.ai). Each poll runs segmentation + embedding + clustering on
the last 20–30 s of speech and emits **every** segment with wall-clock times
mapped through `rolling_meta.json` (`window_start_ts + segment.start`).

| | |
|---|---|
| Needs | Gated HF models; `HF_TOKEN` with the 3.1 terms accepted |
| Cost on M3 Pro CPU | **Several seconds per 25 s window** (estimate 3–8 s, RTF ~0.12–0.3). pyannote 3.1 is slower than 3.0 on CPU; the embeddings stage dominates ([pyannote-audio#1621](https://github.com/pyannote/pyannote-audio/issues/1621), [#1626](https://github.com/pyannote/pyannote-audio/issues/1626)). |
| Label lag | One poll interval (default 2 s) **plus** the multi-second run |
| Strength | Overlap / rapid turn-taking (Q&A), speaker count |

Must stay off the Metal GPU used by MLX Whisper/Gemma. CPU-only in the daemon.

### `embed` — per-chunk embedding + online cosine clustering (default)

Each final chunk is embedded independently. An online cluster
(`OnlineSpeakerCluster`, cosine threshold **0.65**, cap **4** speakers)
assigns `Speaker A` / `Speaker B` / …. No rolling window required; the daemon
tails `chunks.jsonl`. Embedder preference:

1. SpeechBrain ECAPA (`speechbrain/spkrec-ecapa-voxceleb`, Apache-2.0, **not
   gated**) on CPU
2. `pyannote/embedding` if `HF_TOKEN` is set
3. Otherwise the daemon stays idle (clear log line)

Same-speaker cosine ~0.6 is the usual verification cutoff
([SpeechBrain ECAPA card](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb);
common endpoint docs use 0.6).

| | |
|---|---|
| Needs | SpeechBrain (preferred) or gated pyannote/embedding |
| Cost on M3 Pro CPU | **~100–300 ms per 2 s chunk** (conservative). diart measures the ECAPA forward at ~41 ms CPU and pyannote/embedding at ~26 ms CPU on 5 s chunks ([diart README](https://github.com/juanmc2005/diart)); church chunks plus wav I/O plus clustering sit above that, still far from the full 3.1 pipeline. |
| Label lag | One poll (~2 s) after the chunk WAV lands; no multi-second clustering |
| Strength | 1–3 speakers, long turns, low CPU, no GPU contention |

`--embedder fake` (PCM-energy stub) and `--fake-labels` exist for CI only.

### Sunday recommendation: **`embed`**

Stark Road sermons are 1–3 talkers with **long turns**, not rapid overlap.
`--diarize-mode embed` is the default because:

- It fits the +50 ms p95 budget: work is in another process, per-chunk cost
  is hundreds of milliseconds on CPU, never on `_pipeline_pool`.
- Long turns make cosine centroids stable; carry-forward covers the
  one-chunk daemon lag.
- SpeechBrain ECAPA is public (no gated token for the happy path).
- pyannote-on-the-window is better for overlapping speech but costs several
  CPU seconds and a fan-up on the M3 Pro during the service.

Keep `--diarize-mode pyannote` for quality A/B and for the existing offline
`diarize.py` path.

## Failure modes

| Condition | Behaviour |
|---|---|
| `--diarize` omitted (default) | No daemon, no `speaker` key on `result_data`, CSV `speaker` column empty |
| `HF_TOKEN` missing + `--mode pyannote` | Daemon **not** started from `dry_run_ab`; log: `HF_TOKEN not set — pyannote diarization daemon not started`. `--fake-labels` (tests only) still emits A/B. |
| No embedder (SpeechBrain + pyannote both missing) | Daemon idles; log: `No speaker embedder available … live diarization disabled`. Pipeline unlabeled. |
| Daemon crash / kill | Pipeline continues. `stop_diarize_daemon()` is best-effort (SIGTERM process group, then SIGKILL). |
| JSONL missing / malformed lines | Lookup returns `None`; watcher skips the line (unchanged). |
| Gated model download failure | Caught in the daemon; no `sys.exit` on the live process (`diarize.py` still exits 1 as the offline CLI). |

Tokens are read from `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` and never logged.

## Gate

A two-speaker replay (or live dry-run) **passes** when:

1. Finals show **distinct** `Speaker A` / `Speaker B` (audience prefix
   `Speaker A:` and operator caption view / current-speaker pill).
2. `tools/replay_bench.py` `e2e_latency_ms` **p95** with `--diarize` is within
   **+50 ms** of the same clip without `--diarize`.

This gate has **not** been run in this change (mocked tests only; no model
loads, no `HF_TOKEN` in the shell).

## Operator / displays

- Audience (`displays/audience_display.html`): finals prefix `Speaker A:`
  when `data.speaker` is set (`.spk` is CSS-light).
- Operator SPA: **Live diarization** checkbox → `--diarize`; current-speaker
  pill + caption list from the existing watcher snapshot (`captions` from the
  session CSV when bound, else recent JSONL labels).
- Daemon start: **pipeline-owned** (`dry_run_ab` subprocess). Do not start
  `live_diarize.py` by hand unless debugging.

## CLI

```bash
python dry_run_ab.py --diarize                          # embed mode (Sunday default)
python dry_run_ab.py --diarize --diarize-mode pyannote  # rolling-window 3.1
python dry_run_ab.py --diarize --diarize-interval-s 2
```

Manual daemon (only if the pipeline is not spawning it)::

    python features/live_diarize.py \
        --rolling-wav stark_data/live_sessions/<sid>/rolling.wav \
        --chunks-jsonl stark_data/live_sessions/<sid>/chunks.jsonl \
        --output metrics/diarization_<sid>.jsonl \
        --mode embed

## What remains before the gate can be run

1. Accept terms for `pyannote/speaker-diarization-3.1` (and `pyannote/embedding`
   if not using SpeechBrain) and export `HF_TOKEN` on the church Mac — **or**
   `pip install speechbrain` in `stt_env` for the embed path (no gated token).
2. A two-speaker clip (pastor + second voice, or a short Q&A) long enough for
   at least two turns per speaker.
3. Baseline: `python tools/replay_bench.py --audio-file <clip>` (no `--diarize`).
4. Same command with the live pipeline `--diarize --diarize-mode embed`; compare
   `e2e_latency_ms` p95 (must be ≤ baseline + 50 ms) and confirm distinct
   labels on finals / the projector.
5. Optional: repeat with `--diarize-mode pyannote` for quality, not latency.

No Chinese-origin models. pyannote (CNRS/pyannote.ai) and SpeechBrain ECAPA
are the only embedding/diarization backends on this path.
