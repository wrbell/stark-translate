# P1-H — hymn #193: second one-control diagnostic (promoted runtime) and search slices (2026-09-11)

**Result: `completed_technical_diagnostic`, errors [].** Session `hymn193_20260911_183511_488ce36e_en`, 2026-09-11T18:35:11Z → 2026-09-11T18:41:18Z,
source `main` @ `dc983c6`, interpreter `venv/bin/python` (Torch 2.13.0 / TorchAudio 2.11.0), the prepared wrapper
`.cache/mac-en-es-closeout/hymn-capture-preparation/run.py` unchanged (SHA256 `32cbf991…`, packet `d7e59d19…`),
600 s owned bound, TTS and recording off, gain 1, unchanged E4B / Parakeet / Marian, RMS threshold 0.15, nominal
holdoff 5 s, silence 0.5 s, cadence 0.6 s. Input: the lossless 350 s crop `hymn-transition-400-750.wav`
(SHA256 `7937eb4f…`, parent offset 6,400,000 samples) whose parent (`8bec0f10…`, 58,240,848 samples) was re-hashed
before and after the run. This is the **second control** on the same crop; the first ran on `c13f51f` with
`stt_env` on 2026-09-10 (`docs/evaluation/mac_followup_20260910/final-c13f51f/hymn-capture.md`).

| | first control (c13f51f, stt_env) | this control (dc983c6, venv Torch 2.13) |
|---|---|---|
| finals | 18 (10 silence, 4 smart cuts, 4 hard cuts) | 18 (4 hard_cut, 10 silence, 4 smart_cut) |
| final routes | 13 Gemma / 5 Marian | 5/13 (Gemma/Marian) |
| partials emitted | 63 | 89 |
| music hold | never activated | never activated (max VAD-false streak 48 frames of the 156 needed; VAD-positive frames 2012 / 11000) |
| physical STT spans | — | partial 118, final 22 (all finished) |
| silence-final speech_end→final p50 / p95 | — | 1159 / 1413 ms (n=10) |
| frame audit | passed | passed (11000 frames dequeued = VAD frames) |

The endpoint mix and routing are identical to the first control; the promoted runtime changes nothing about how
this crop is segmented. The hold heuristic again never entered (the longest no-speech streak above the RMS
threshold was 48 × 32 ms, far below the 4.992 s entry requirement), so this run — like the first —
says nothing about singing detection. Raw terminal artifacts (replay summary, metadata, lifecycle, health, CSV)
are under [`raw/`](raw/); the full `diagnostic.json` is beside this file.

## Search slices for the human labeller

Cut losslessly from the parent with [`slices/cut_slices.py`](slices/cut_slices.py) (stdlib `wave`, no resampling or
normalization); receipts in [`slices/`](slices/). The WAVs stay local (`.cache/series3-20260912/P1H/slices/`).

| slice | parent-native samples | duration | SHA256 | bytes |
|---|---|---|---|---|
| `hymn-search-400-470.wav` | [6,400,000, 7,520,000) | 70.0 s | `ed850359bda133cecfb712dd780e88dcbb0ce6fc8bde528b67073f3a2b30ebf1` | 2240044 |
| `hymn-search-700-780.wav` | [11,200,000, 12,480,000) | 80.0 s | `10c0ed641c8bb894383eaad079619ddd75a2fc93c24a62b71e2d01b3c0fcc2ab` | 2560044 |

No acoustic labels exist; no threshold, VAD, cadence or model setting was tuned; this run is a control for the
future labelled comparison, not evidence about singing detection, caption quality, audience delivery or
microphone capture. Issue #193 stays open pending independent labels.
