# Mac implementation status — v2026.14 candidate

This work follows v2026.13 (`09e4679`, PRs #180–191 already merged). Changes are on
`codex/mac-reliability-roadmap` in separate commits. E4B remains the default;
the operator has no new model-selection control. English STT remains Parakeet;
Spanish STT remains Whisper. MTP remains disabled. Local implementation and
artifact validation are complete; the external gates below remain open.
v2026.14 publication is pending by the user's choice.

## Implemented

- Operator-owned session identity, production CSV parsing, unavailable values,
  persistent verse watcher, positional summary CLI and surfaced subprocess errors.
- Capture/sample timing through partials, silence, smart-cut remainders, forced
  endings and EOF. Additive schema 2 fields retain the old CSV column ordering.
- Visible caption render acknowledgments and delayed, chunk-specific speaker
  updates. New sessions clear caption history; same-session reconnects preserve it.
  Caption timing excludes TTS; synthesis and playback calls have separate metrics.
- Shared MLX generation/prompt/stop handling, and opt-in idle warmups,
  final-aware partial scheduling, silence thresholds, conservative Marian routing,
  terminology examples and ONNX VAD.
- Mac-aware model resolution, cached model setup, environment selection, generated
  launchd installation, complete wheel/source ZIP contents and release identity checks.
  Managed cache markers must match the pinned manifest; explicit user paths remain
  supported. The default Piper voices match the English/Spanish setup profile.
  Mac VAD loads bundled Silero 6.2.1 weights without Torch Hub. Setup reuses
  existing complete Marian CT2 adapters or converts both pinned HF directions
  into an isolated managed cache using the selected interpreter. Setup,
  preflight and inference share that CT2 resolver; atomic publication preserves
  working artifacts even if interrupted after the active pointer changes.
- Live and post-session Review, independent transcript/translation approvals,
  local drafts, revision checks, portable audio bundles, direction-aware exports,
  idempotent imports and evaluation isolation.
- Explicit session completion after worker/diagnostics draining for both SIGINT
  and SIGTERM. Running, failed and unknown legacy sessions cannot export training
  data. Completion records process peak RSS/Metal memory and local model provenance,
  including actual Marian CT2 weight hashes where available.

## Measurement and data integrity

The [evaluation README](evaluation/README.md) records the manifest, commands,
quality report and reference repair. Only schema 2 `speech_end_to_final_ms` measures
estimated speech end to final payload readiness. A visible browser's
`speech_end_to_ack_upper_bound_ms` includes its render and return-network time.
Neither is interchangeable with archived `e2e_latency_ms` processing times.

All 18 historical real-time baseline replays have completed: E4B/E2B, three repeats,
two English sermon clips and one separate synthetic Spanish clip. The measured
caption-delivery goal is **not yet achieved**. Baseline results retain their source
and configuration cohorts; startup/shutdown and instrumentation revisions during
that collection must not be pooled into a single claim about the current runtime.

The [frozen English screening report](evaluation/mac_v2026_14_screening/README.md)
is complete: **48/48 runs exited zero**, covering eight configurations, both
models and three alternating pairs on the same 45-second input. It retains
348 finals, 3,184 partials and a hash index of 193 raw files. The result supports
**no additional combined configuration**: gains were model-specific, some tails
and first-partial delays worsened, and shorter silence changed the captions.
Matched later-caption analysis did not show a consistent gain across both models.
Keep E4B, 0.5-second silence and 0.6-second partial cadence as defaults; experiments
remain opt-in. No browser clients were observed (visible final ACK coverage
0/348), so this screen cannot establish the caption-delivery goal.

The conservative routing branch was unexercised on that sermon clip. The separate
[synthetic EN/ES routing report](evaluation/mac_v2026_14_routing/README.md) now
records **24/24 runs exited zero**, with 72 finals and 90 partials after the
packaged VAD and automatic CT2 setup changes. Across both models, languages and
three repeats, conservative routing used Marian for the two allowlisted phrases
and Gemma for the non-allowlisted sentence; legacy routing used Marian for all
three. All runs recorded the installed Silero 6.2.1 JIT artifact and its weight
hash. Visible final ACK coverage was 0/72. Keep conservative routing opt-in:
these synthetic path checks do not replace natural Spanish references or human
translation review. The 48-run screen retains its original source/loader cohort.

The [translation comparison](evaluation/mac_v2026_14_quality/comparison.md) uses
identical text inputs and three repeats. E2B's translation-only median is 37–43%
lower across the tested directions/prompts, with different outputs and fewer
English-to-Spanish canary passes: 11/18 versus E4B's 13/18 without terminology
examples, and 14/18 versus 15/18 with them. This is a bounded speed/quality tradeoff,
not evidence to change the default or a speech-end-to-display result. Bilingual
meaning and terminology review is still pending.

An existing training export test overwrote the local holdout with two fixtures.
The export now writes its holdout beside the requested dataset, and the original
local holdout was restored from the aligned test corpus. Separately, that older
parallel corpus contains known row-ID alignment errors. Evaluation manifest v2
rebinds references using exact source text and book/chapter/verse; ambiguous items
have no reference score. Predictions and measured timings were preserved.

## Completed validation

- Final CPU suite after the VAD/CT2 setup changes: **1,790 passed, 4 skipped**,
  59.07% coverage against the 50% gate. An earlier rerun exposed a stale packaging
  test that rejected the new derived-CT2 manifest type; its schema assertion was
  corrected and the complete suite rerun. The failure log was retained.
- Earlier cached MLX GPU regression suite: **3 passed**, covering E4B EOS/canary
  behavior and the worker's first forward pass. The later 24 routing runs cover
  actual post-setup pipeline execution separately.
- Ruff lint/format (236 files) and mypy (19 files) pass. Official HTML5 Tidy 5.8.0 reports zero warnings
  or errors across all five displays; it was built only in the repository cache.
- CI-configured Bandit passes with zero medium/high findings. The final expanded
  run retaining B615 reports **27 medium findings**: the original 26 call sites
  with their documented local/fallback limitations, plus the managed CT2 source
  download whose revision is guarded by a full 40-character commit check. That
  additional static-analysis report is not a new unpinned path; the earlier
  pinning debt remains outside the CI pass. Vulture reports three advisory findings.
- CI-filtered Mac, Windows and NVIDIA requirement audits report zero known
  vulnerabilities. This scope does not certify every optional package or model.
- VAD/setup regression subset: **119 passed**, including real JIT and ONNX CPU
  loads and a frame/reset with empty Torch Hub caches and network access blocked.
  Packaged weights and relevant loader files match the prior Hub cache byte for
  byte. The subsequent atomic-publication interruption fix passed all **8 managed
  CT2 setup tests**, including a replace-then-interrupt regression.
- Actual offline CT2 conversion in `.cache/package-smoke` produced both int8
  directions from pinned HF snapshots under an empty project root. Both CPU
  translations were nonempty, repeated setup reused the results, and the weight
  hashes matched the existing adapters. Existing adapters and `stt_env` were
  unchanged. The setup CLI separately reused all five Mac defaults without
  downloads (0 installed, 5 skipped, 0 failed).
- The final v2026.14 wheel, sdist and Mac ZIP are validated from source `977583b`;
  [installation evidence](evaluation/mac_v2026_14_installation.md) records hashes
  and the post-build evidence boundary. The unpacked ZIP launches directly and
  builds a byte-identical wheel. Both wheels installed outside the checkout and
  served `/healthz`, `/operator/` and the review script. The full
  `[mlx,eval,diarization]` extras were **installed**, all 18 runtime import checks
  passed, 106 installed/source hashes matched, and `pip check` was clean.
  Existing `stt_env` was unchanged. Real installed EN/ES STT→E4B→target-voice WAV
  sessions completed with exit zero on the preceding r3 wheel. The final r4 wheel
  changes only the Conda shell resolver and generated package inventory; all 113
  other members match the exercised artifact. Its launcher/install checks were
  executed separately. No natural-quality or physical-playback gate is inferred.
- Independent STT inference completed on 50 English and 11 Spanish saved clips,
  with no approved human references, so **no WER or natural-audio acceptance claim**
  is made. Offline Hindi generation completed on both models for 43 English
  inputs each; Hindi references and human quality review remain absent.

Local evidence is recorded in `.cache/mac-roadmap/validation.json`,
`.cache/mac-roadmap/full-tests.log`, `.cache/mac-roadmap/full-tests-final.log`,
`.cache/mac-roadmap/full-tests-delivery.log`, `.cache/mac-roadmap/coverage-delivery.json`,
`.cache/html5-validation/report.json`,
`.cache/security-audit/`, `.cache/package-artifacts-validation/`,
`.cache/mac-roadmap/vad-cache-proof.json` and
`.cache/mac-roadmap/ct2-setup-validation/report.json`. See the
[security scope audit](evaluation/mac_v2026_14_security.md) for the distinction
between pinned default setup paths and remaining optional/fallback download debt.

## Operator rehearsal completed

Three controlled mixed/synthetic sessions completed with exit code zero and
drained lifecycle markers. These runs used TTS and are explicitly excluded from
the frozen latency acceptance configurations.

| Session purpose | Final captions | Finals with visible completion acknowledgment | Browser receipt-to-render p50 |
|---|---:|---:|---:|
| English hymn, pause/resume and speech | 9 | 8 | 12.2 ms |
| Synthetic Spanish restart | 1 | 1 | 7.4 ms |
| English verse cue, speech, Review, Stop and summary | 8 | 8 | 6.9 ms |

The browser figures measure local render overhead only. They do not include
speech recognition or translation and cannot establish sub-second caption delivery.

Observed checks passed for startup readiness, pause/resume, EN→ES→EN session
identity, audience reconnect/history reset, visible caption acknowledgments,
John 3:16 from the production CSV remaining stable across polls, live finalized
Review, persisted draft notes and selection, audience/review separation, and
rejection of unapproved exports. No transcript or translation approval was fabricated.
Per-language TTS device choices persisted; normal Stop drained and marked the
session complete. The positional summary command worked. An initial summary
format failure was retained, and the corrected rerun produced one three-sentence
Spanish translation without alternatives or notes.

Piper synthesized both configured languages and host playback calls completed on
MacBook Pro Speakers. This validates the built-in output path, not acoustic
quality/onset or a second physical device. Rehearsal details and limitations are
in `.cache/mac-roadmap/rehearsal_report.json`; the retained summary failure is
`.cache/mac-roadmap/summary_format_failure_20260909_211201_787725_en.json`.

## Explicit pending gates

- At least 50 human-reviewed natural utterances per language. There are 50 English
  and 11 Spanish candidates, all still unapproved; the user does not currently
  have a natural Spanish recording location. Synthetic Spanish stays separate.
- Natural two-speaker audio, human speaker-transition labels and the ≤50 ms
  additional final-p95 diarization gate.
- Bilingual blinded review of meaning errors and terminology preferences.
- Physical second-output selection, unplug/replug and acoustic playback validation;
  only the built-in speaker path has been exercised.
- A service rehearsal with natural bilingual speech, real speaker transitions and
  church audio hardware. The controlled hymn/pause/restart/Review/Stop rehearsal
  above does not close those human and device gates.
- A frozen visible-browser timing run on the unlocked Mac. The screen and routing
  reports have no visible ACKs; the earlier controlled operator rehearsal remains
  separate evidence. Local artifact and installed-runtime checks are complete.
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
