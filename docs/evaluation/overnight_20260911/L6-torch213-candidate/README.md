# L6 — audited Torch 2.13 / TorchAudio 2.11 application candidate built from current main

**Lane state:** DONE — candidate is functionally equivalent to the working `stt_env` on this Mac; **go/no-go is Willem's call** (recommendation below). The working `stt_env` and the launcher are untouched.

## What was built

- Scratch tree: `git archive 9983f68` → `.cache/mac-en-es-closeout/torch213-main-20260911/source`, with the two dependency edits from the earlier feasibility note applied to the *scratch* `pyproject.toml` only (`torch>=2.10,<2.11` → `torch==2.13.0` in `mlx`; `torchaudio>=2.10,<2.11` → `torchaudio==2.11.0` in `diarization`). Production `pyproject.toml` is unchanged.
- Wheel `stark_translate-2026.14.0.0-py3-none-any.whl` built with `.cache/package-smoke/bin/python -m build --no-isolation`; metadata check passed (`wheel-metadata.json`).
- Fresh venv from `/Users/willem/anaconda3/bin/python3.11`, resolved against the audited `constraints.txt` (hash-pinned `resolved-requirements.txt`, 124 third-party distributions), installed with `--require-hashes`; `pip check` clean; installed-metadata consistency passed; native imports + bundled Silero VAD smoke passed (`native-imports-vad` receipt in `command-receipts.json`).
- `pip-audit` over the installed site-packages: **0 known vulnerabilities across 124 distributions**; the unpublished first-party package is the only skip (`installed-audit.json`).

Installed: torch 2.13.0, torchaudio 2.11.0, mlx 0.32.2, mlx-lm/mlx-whisper/parakeet-mlx as in `stt_env`, speechbrain 1.1.1.

## Functional replay checks (same script and validator as the earlier feasibility note)

Normalized FLEURS EN and ES replays (`protocol/standard-screen.json` clips, real-time, Parakeet EN / Whisper ES, Gemma 4 E4B, Marian CT2, Torch VAD, HF offline): both **`passed_basic_functionality`**, and `validate_normalized_replays.py` reported **all checks passed for both languages** (`normalized-r1-summary.json`, receipts `replay-receipt-normalized_r1_{en,es}.json`).

## Paired latency comparison, control (`stt_env`) vs candidate venv

Same checkout `dry_run_ab.py`, same 150 s natural English church clip (`Gospel_Message_(12_14_25)`, SHA256 `db11fb4c…`), identical arguments (`--profile standard --stt-backend parakeet-mlx --gemma4-size e4b --vad-backend torch --no-mts --gain 1 --replay-speed 1`), HF offline, one process at a time, order ctl/cand, cand/ctl, ctl/cand. Raw results: `replays/replay_on0911_t213_*.json`.

| pair | arm | finals | previews | silence finals | speech_end→final p50 ms | p95 ms | peak RSS MiB | peak Metal MiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| r0 | control | 26 | 219 | 8 | 1854.9 | 2487.9 | 3407 | 8782 |
| r0 | candidate | 26 | 221 | 8 | 1829.3 | 2426.6 | 4111 | 8699 |
| r1 | candidate | 26 | 214 | 8 | 1899.0 | 2526.0 | 4748 | 8802 |
| r1 | control | 26 | 220 | 8 | 1898.0 | 2400.8 | 3915 | 8844 |
| r2 | control | 26 | 220 | 8 | 1892.0 | 2390.7 | 4238 | 8734 |
| r2 | candidate | 26 | 219 | 8 | 1871.6 | 2368.5 | 3825 | 8704 |

- All six runs completed (exit 0, lifecycle `completed`, no timeout).
- **Final captions byte-identical between arms in all three pairs (26/26 English and 26/26 Spanish each)**; the two control runs are also identical to each other, so the pipeline is deterministic on this clip.
- Candidate silence-final p50 is within the max(5 %, 100 ms) equivalence band in 3/3 pairs (−25.6, +1.0, −20.4 ms). With only 8 silence finals per run this is an equivalence screen, not a speed claim; no p95 claim is made (n < 100).
- Preview counts differ by at most 6 (−2.7 %) in one pair and ≤ 2 in the others.
- Peak Metal memory is flat (8.7–8.8 GiB both arms). Peak RSS is noisier (3.4–4.7 GiB) and overlaps between arms; the candidate is not consistently higher.

## Recommendation (for Willem)

**Go for promotion, in a separate attended step.** The audited Torch 2.13 candidate built from current main passes every functional gate the earlier feasibility note defined, removes the two open Torch advisories, produces identical captions, and shows no latency or Metal-memory regression on the paired replays. Promotion still means: rebuild the final release source into a fresh environment with these constraints, switch the launcher with a rollback path, and re-run the installed smokes. Not done tonight: no launcher switch, no `stt_env` change, no human-quality or live-microphone check.

## Note found on the way (fixed in PR #201)

TorchAudio 2.11 delegates `torchaudio.load` to TorchCodec, which the `diarization` extra does not install, so the live-diarization SpeechBrain path could not read its rolling WAV in this environment. `features/live_diarize.py` now reads WAVs with `soundfile` (already a declared dependency) and falls back to torchaudio only when soundfile is missing.
