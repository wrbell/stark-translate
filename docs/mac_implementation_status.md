# Mac implementation status — v2026.14 candidate

This work follows v2026.13 (`09e4679`, PRs #180–191 already merged). Changes are on
`codex/mac-reliability-roadmap` in separate commits. E4B remains the default;
the operator has no new model-selection control. MTP remains disabled.

## Implemented

- Operator-owned session identity, production CSV parsing, unavailable values,
  persistent verse watcher, positional summary CLI and surfaced subprocess errors.
- Capture/sample timing through partials, silence, smart-cut remainders, forced
  endings and EOF. Additive schema 2 fields retain the old CSV column ordering.
- Visible caption render acknowledgments and delayed, chunk-specific speaker
  updates. Caption timing excludes TTS; synthesis and playback calls have separate metrics.
- Shared MLX generation/prompt/stop handling, and opt-in idle warmups,
  final-aware partial scheduling, silence thresholds, conservative Marian routing,
  terminology examples and ONNX VAD.
- Mac-aware model resolution, cached model setup, environment selection, generated
  launchd installation, complete wheel/source ZIP contents and release identity checks.
- Live and post-session Review, independent transcript/translation approvals,
  local drafts, revision checks, portable audio bundles, direction-aware exports,
  idempotent imports and evaluation isolation.

## Measurement and data integrity

The [evaluation README](evaluation/README.md) records the manifest, commands,
quality report and reference repair. Only schema 2 `speech_end_to_final_ms` measures
estimated speech end to final payload readiness. A visible browser's
`speech_end_to_ack_upper_bound_ms` includes its render and return-network time.
Neither is interchangeable with archived `e2e_latency_ms` processing times.

The first measured replay exceeds one second; the caption-delivery goal is **not
yet achieved**. Full paired baselines and bounded experiments are being collected.
Do not promote experimental settings from an isolated translation measurement.

An existing training export test overwrote the local holdout with two fixtures.
The export now writes its holdout beside the requested dataset, and the original
local holdout was restored from the aligned test corpus. Separately, that older
parallel corpus contains known row-ID alignment errors. Evaluation manifest v2
rebinds references using exact source text and book/chapter/verse; ambiguous items
have no reference score. Predictions and measured timings were preserved.

## Validated so far

Focused operator, timing, review/export, setup and engine regression suites pass.
The isolated base wheel starts outside the checkout and serves both `/healthz`
and `/operator/`. The complete MLX/diarization/evaluation extras resolve to the
verified runtime minor lines without modifying `stt_env`. Mac setup reuses five
cached default artifacts. Full-suite and final GPU/browser results will be recorded
after sequential benchmarking finishes.

## Explicit pending gates

- At least 50 human-reviewed natural utterances per language. There are 50 English
  and 11 Spanish candidates, all still unapproved; the user does not currently
  have a natural Spanish recording location. Synthetic Spanish stays separate.
- Natural two-speaker audio, human speaker-transition labels and the ≤50 ms
  additional final-p95 diarization gate.
- Bilingual blinded review of meaning errors and terminology preferences.
- Physical second-output selection, unplug/replug and acoustic playback validation.
- Full rehearsal with hymns, live pauses, continuous speech and EN↔ES restart.
- PyPI trusted publisher account setup: the browser is signed out and the user
  explicitly chose to leave publishing pending. Required mapping is owner
  `wrbell`, repository `stark-translate`, workflow `pypi.yml`, environment `pypi`.

The v2026.13 MSI SHA-256 matches the release digest
`e772992f5ad925cac7d78984574615846125e9837e497c081f315c632a1cc7a2`;
its embedded ProductVersion is `2026.13.0`. The obsolete v2026.12 MSI asset was
removed from that release. Windows installation execution remains untested here.
Published tags have not been moved. A future release must use a new version/tag.

WSL preprocessing, training, CUDA experiments and adapter conversion remain
separate execution work. Portable reviewed data is the handoff boundary.
