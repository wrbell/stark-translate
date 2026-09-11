# L4 — live diarization (#133) latency half-gate on a synthetic two-voice clip

**Lane state:** DONE as a *screening* result. Latency half of the #133 gate: **passes on the natural church clip (3/3 pairs), 2/3 on the synthetic clip**. Label half: distinct, deterministic speaker labels on finals, but on an *accented synthetic* clip — natural two-speaker labels (`natural-two-speaker`) remain pending. `--diarize` stays off by default.

## Setup

- Source: branch `codex/overnight-20260911-diarize` at `afb71da` (= PR #201 content merged to `main` as `50f81c6`), clean checkout `../SRTranslate-wt-diarize`; run serially, HF offline.
- Pipeline interpreter: working `stt_env`. Diarization daemon interpreter (`--diarize-python`): the audited Torch 2.13 venv from lane L6 (`.cache/mac-en-es-closeout/torch213-main-20260911/venv`, speechbrain 1.1.1), which is the only environment here with SpeechBrain. Embedder: `speechbrain/spkrec-ecapa-voxceleb` at the pinned revision `0f99f2d0…` (prefetched; loaded offline), `--diarize-mode embed`, interval 2 s. Prerequisite fix in the same PR: the daemon reads WAVs with `soundfile` (TorchAudio 2.11 delegates `load` to TorchCodec, which is not installed).
- Clips (`manifest.json`): the 150 s natural English church clip `Gospel_Message_(12_14_25)` (SHA256 `db11fb4c…`) and `synthetic_two_voice_en.wav` (61.44 s, 16 kHz mono PCM16, SHA256 `99927543…`, `synthetic_two_voice_en.recipe.json`): twelve alternating turns, voice a = Piper `en_US-lessac-high`, voice b = Piper `es_MX-claude-high` reading English (Spanish voice, strong accent — deliberate caveat), 0.5 s lead, 1.2 s gaps.
- Arms: identical pipeline arguments (`--profile standard --stt-backend parakeet-mlx --gemma4-size e4b --vad-backend torch --no-mts --gain 1 --replay-speed 1`) with and without `--diarize --diarize-mode embed --diarize-python <venv>`; order off/on, on/off, off/on; each config ran both clips → 12 runs, all exit 0, none timed out. Raw: `replays/`, daemon output: `diarization/`.

## Latency (silence-ended finals, schema 2, nearest-rank)

| clip | pair | off p50 / p95 ms | on p50 / p95 ms | Δp95 | within +50 ms? |
|---|---|---:|---:|---:|---|
| church (n=8 per run) | r0 | 2008.7 / 3117.5 | 1843.7 / 2352.8 | −764.7 | yes |
| church | r1 | 1837.4 / 2411.2 | 2168.9 / 2369.2 | −42.0 | yes |
| church | r2 | 2113.8 / 2572.1 | 2119.8 / 2526.2 | −45.9 | yes |
| synthetic (n=10 per run) | r0 | 1910.7 / 3023.3 | 1752.9 / 1968.9 | −1054.4 | yes |
| synthetic | r1 | 1797.4 / 2265.3 | 2019.1 / 2943.1 | **+677.8** | **no** |
| synthetic | r2 | 1752.4 / 2489.1 | 1833.0 / 2207.6 | −281.5 | yes |

With 8–10 silence finals per run, p95 is the largest or second-largest value; the control-to-control spread (church off p95 2411 → 3118) is as large as any on/off difference. The daemon runs on CPU in a separate process and never blocks inference; nothing in these runs indicates a systematic cost, but the synthetic r1 pair is retained as a failed pair rather than explained away. No p95 claim (n < 100).

Final captions were byte-identical between arms in all six pairs (26/26 church, 10/10 synthetic, English and Spanish), so diarization changed no text.

## Labels (synthetic clip, `--diarize` on; identical in r0, r1, r2)

| final | source s | truth voice | label |
|---|---|---|---|
| 0 | 0.5–4.4 | a | — (daemon warm-up; first final unlabeled) |
| 1 | 5.3–10.7 | b | Speaker A |
| 2 | 11.6–16.3 | a | Speaker B |
| 3 | 23.6–27.9 | a | Speaker A ← the one inconsistent label |
| 4 | 28.8–33.3 | b | Speaker A |
| 5 | 34.1–38.1 | a | Speaker B |
| 6 | 38.9–43.0 | b | Speaker A |
| 7 | 43.8–46.8 | a | Speaker B |
| 8 | 47.7–51.5 | b | Speaker A |
| 9 | 52.3–55.8 | a | Speaker B |

- Two distinct labels appear; under the stable mapping Speaker A ↔ voice b, Speaker B ↔ voice a, **8 of 9 labeled finals are consistent**; `speaker` is filled on 9/10 finals (≥ 80 %). On the church clip 25/26 finals carry a label (single speaker, as expected).
- Turns 4 (17–22 s) and 12 (the last) — both the accented voice b — produced **no final** in either arm; the accented voice was also transcribed poorly by Parakeet (e.g. "sank gyoubroter, letus open obro bibles"). That is a property of this synthetic input, not of diarization, and is the main reason this clip cannot stand in for natural two-speaker audio.

## What this establishes and what it does not

- Establishes: the `embed` daemon runs end-to-end from a separate interpreter with pinned, offline ECAPA weights; it labels finals deterministically and without changing captions; on the natural church clip it stays inside the +50 ms p95 budget in every pair.
- Does not establish: label accuracy on natural speech (needs the pending natural two-speaker recording with human transition labels), a p95 guarantee, or any default change. Backlog `issue-133-diarize-gate` stays `implemented` / certification pending with this screening evidence attached.
