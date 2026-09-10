# Public EN↔ES speech evaluation inputs

This is a portable, **evaluation-only** selection of 200 original FLEURS recordings and upstream text annotations. The audio stays in the local `.cache/mac-en-es-closeout/fleurs-v1/audio/` directory. No model evaluation has run as part of preparing this evidence.

[manifest.json](manifest.json) preserves the raw and upstream-normalized source text, a uniquely joined opposite-language reference, relative audio path, original WAV SHA-256, and all 18 repository theological canaries. [audit.json](audit.json) records the independent checks against the pinned original TSVs and tar archives.

## Attribution and provenance

FLEURS: Few-shot Learning Evaluation of Universal Representations of Speech, Google and contributors (2022). Dataset: [google/fleurs](https://huggingface.co/datasets/google/fleurs); immutable revision [`70bb2e84b976b7e960aa89f1c648e09c59f894dd`](https://huggingface.co/datasets/google/fleurs/tree/70bb2e84b976b7e960aa89f1c648e09c59f894dd). FLEURS is distributed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Its public speech is based on the parallel FLoRes text benchmark; see the [authors’ FLEURS paper](https://arxiv.org/abs/2205.12446).

Changes here: a deterministic subset was selected, metadata was reorganized, and glossary/canary evaluation fields were added. Selected WAV bytes, raw source annotations and opposite-language reference text remain unchanged. Attribution and upstream file URLs/hashes travel with the manifest and audit. No audio is redistributed in this directory.

The original selection was prepared by `tools/public_speech.py`. `tools/mac_followup_quality.py audit` independently re-parsed the four original TSVs, recomputed their hashes and the four archive hashes, reconstructed deterministic selection, and compared every selected local WAV against its original tar member. It verified WAV format, 16 kHz mono sample counts, references, shared sentence joins, and evaluation-only provenance flags.

| Language | Development (upstream dev) | Confirmation (upstream test) |
|---|---:|---:|
| English, en_us | 50 | 50 |
| Spanish, es_419 | 50 | 50 |

The full TSV unions contain 150 development sentence IDs and 350 confirmation IDs across the two languages, with **zero cross-partition sentence-ID overlap** and zero normalized-reference-text overlap. All 200 selected audio hashes are unique. All 200 translation references have exactly one distinct opposite-language raw text for the same upstream partition and sentence ID. Different recordings of the same sentence may occur across languages **within** a partition; those are paired linguistic content, not independent sentence evidence.

Speaker IDs are unavailable, so speaker-disjointness is unverified. The set is natural **read speech**, not spontaneous church speech, live room audio, singing, overlap, or microphone/device evidence. Prior exposure of the models to this public benchmark is unknown. Upstream annotations are usable reference text, but `human_approved_locally=false` and `training_eligible=false` remain explicit. This is not a claim of local bilingual review, consent for training, or an untouched model-pretraining holdout.

## Scoring contract

- STT uses NFKC, casefold, Unicode punctuation replaced by spaces and collapsed whitespace. Accents and digit forms remain; no language-specific number expansion, source-dependent cleanup, glossary correction, or predicted reference is used. Corpus WER is summed word-edit distance divided by summed reference words; individual counts and raw hypotheses are preserved. A zero-reference denominator is unavailable, not zero error.
- Glossary recall uses normalized whole-word/phrase presence for unique glossary term types actually present in each reference. The frozen set has only **three** such opportunities (EN development 0, EN confirmation 2, ES development 1, ES confirmation 0). This broad glossary overlap cannot establish theological-term accuracy. Zero opportunities yield null recall.
- Fixed-text translation always receives the verified **upstream source transcript**, never STT output. Character chrF uses character orders 1–6, beta 2, case-sensitive text with whitespace removed, clipped corpus n-gram counts and effective orders present on both sides. Its full implementation contract is stored in output JSON. It is a model-free local chrF implementation, **not chrF++**, COMET, or a semantic/human quality score.
- The 18 existing EN→ES canaries have expected substrings but no full reference translations. Their existing case-insensitive substring pass/fail contract is reported separately from public-reference chrF. Canaries are reused development checks; appending them to a confirmation run does not turn them into independent confirmation items. There is no invented reverse-direction canary reference.
- Both engine-reported latency and caller wall time are retained. Whole-recording STT calls exclude WAV loading, model load/built-in warmup, VAD, audio capture, queueing and browser rendering. They are not live end-to-end latency. RSS is the process high-water mark including loading and decoding; Metal allocation is separate and must not be added to RSS.

## Commands for the assigned serial inference slot

Run from the repository root. These commands prepare no downloads, install nothing, and refuse to overwrite existing output directories. `stt_env` remains unchanged. The parent scheduler must reserve the Mac so no other model job runs concurrently.

```bash
# Re-audit into a NEW directory if the input set is intentionally revised.
stt_env/bin/python tools/mac_followup_quality.py audit \
  --manifest .cache/mac-en-es-closeout/fleurs-v1/manifest.json \
  --downloads .cache/mac-en-es-closeout/downloads \
  --audio-root .cache/mac-en-es-closeout/fleurs-v1 \
  --output .cache/mac-en-es-closeout/fleurs-audit-recheck

# Development: Parakeet v3 versus MLX Whisper Turbo, both EN and ES.
stt_env/bin/python tools/mac_followup_quality.py stt \
  --manifest docs/evaluation/mac_followup_20260910/public_data/manifest.json \
  --audio-root .cache/mac-en-es-closeout/fleurs-v1 \
  --engines parakeet-mlx whisper-mlx --languages en es \
  --partition development --repeats 3 --timeout-seconds 3600 \
  --output .cache/mac-en-es-closeout/quality/stt-mlx-development

# CPU INT8 small versus base; all models must already be local.
stt_env/bin/python tools/mac_followup_quality.py stt \
  --manifest docs/evaluation/mac_followup_20260910/public_data/manifest.json \
  --audio-root .cache/mac-en-es-closeout/fleurs-v1 \
  --engines ct2-small ct2-base --languages en es \
  --partition development --repeats 3 --timeout-seconds 7200 \
  --output .cache/mac-en-es-closeout/quality/stt-cpu-development

# E4B/E2B verified fixed-text EN↔ES translation and all 18 EN→ES canaries.
stt_env/bin/python tools/mac_followup_quality.py translate \
  --manifest docs/evaluation/mac_followup_20260910/public_data/manifest.json \
  --audio-root .cache/mac-en-es-closeout/fleurs-v1 \
  --engines e4b e2b --languages en es --include-canaries \
  --partition development --repeats 3 --timeout-seconds 3600 \
  --output .cache/mac-en-es-closeout/quality/translation-development

# Model-free report; use one input directory at a time and new output paths.
stt_env/bin/python tools/mac_followup_quality.py report \
  --manifest docs/evaluation/mac_followup_20260910/public_data/manifest.json \
  --input .cache/mac-en-es-closeout/quality/stt-mlx-development \
  --output .cache/mac-en-es-closeout/quality/stt-mlx-development-report.json
```

After development decisions are frozen, the same commands with explicit `--partition confirmation` and a new output directory evaluate the separate test partition. Default commands never select confirmation. Preserve negative arms and the original development results. Do not tune on confirmation outputs and then call them untouched confirmation.

A preinstalled model can be selected explicitly with, for example, `--model-override ct2-base=/absolute/path/to/pinned/base`. The runner hashes all non-hidden model files and records resolved path/revision, requested identity, loaded identity, model source hashes and dependency versions. It refuses a missing model; it never downloads one. A failed primary load that selects another model fails the arm, and per-call fallback results are rejected. Whisper confidence fallback is disabled; Parakeet language selection remains automatic because its API cannot force EN/ES. CPU CT2 uses INT8, four CPU threads, one worker and forced language; Whisper beam size is five and Parakeet remains greedy.

Each repeat/language/engine gets a fresh child process. Engine order reverses on alternate repeats, children are strictly serial, and the existing bounded process-group cleanup handles timeouts. Parent STARK experiment/profile overrides are removed; model-cache directory selection is retained. Workers preserve per-item exceptions, load errors, cleanup errors, return codes and partial results. Reports validate expected identities and recompute score lineage, retain failed arms as ineligible, and do not pool engines, languages, partitions or repeats. No selector or default-promotion policy is implemented here.

## Validation status

The focused model-free suite exercises original TSV/tar mismatch rejection, cross-language partition isolation, reference/approval invariants, float WAV headers, scorer edge cases, exact runtime engine construction, real worker orchestration with mocked engines, language propagation, model fallback failures, serial ordering, and report integrity. No STT/Gemma model was loaded to prepare this directory. Actual inference results belong in separately hash-bound run reports.
