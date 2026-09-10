# Mac operator rehearsal — 2026-09-09

Three actual operator sessions completed successfully through the production Mac pipeline. The rehearsal exercised startup, pause/resume, EN→ES→EN restarts, audience reconnection, verse detection, live review, TTS WAV output, normal stopping, and a clean summary rerun. These are workflow results; **none of these sessions qualifies for the latency or human-reviewed language-quality gates**.

The [machine-readable evidence report](mac_v2026_14_rehearsal_report.json) records session identities, source hashes, model revisions, artifact checksums, review state, and remaining gates. It combines the driver's saved browser observations with independent checks of the persisted files. This documentation audit used only light JSON/CSV, checksum, and WAV-header reads; it did not run models, playback, training, or tests during the frozen latency screen. Raw audio, diagnostics, sidecars, and logs remain local generated artifacts; this commit contains the report and its checksummed inventory.

The original [68-second input manifest](mac_v2026_14_rehearsal.json) remains unchanged. The final English session used the separate [verse-cue input manifest](mac_v2026_14_verse_rehearsal.json). The operator deliberately reused `.cache/mac-roadmap/operator_input.wav`, replacing its contents before each language restart. Each session's recorded hash is matched to its preserved source WAV; the mutable filename alone does not identify its historical input.

| Session | Input and direction | Recorded completion | Final captions | Visible final chunks acknowledged | TTS WAV files |
|---|---|---|---:|---:|---:|
| `20260909_210408_972943_en` | Historical English hymn/speech with inserted silence; EN→ES; paused/resumed | Completed, exit 0 | 9 | 8 / 9 | 9 |
| `20260909_210738_020946_es` | Synthetic Spanish; ES→EN | Completed, exit 0 | 1 | 1 / 1 | 1 |
| `20260909_211201_787725_en` | Synthetic John 3:16 cue plus historical English speech; EN→ES | Completed, exit 0 | 8 | 8 / 8 | 8 |

All three completion markers match the current diagnostics checksums. The finalized diagnostic IDs match the CSV IDs, all **18 retained source-chunk WAVs** exist, and all **18 TTS WAVs** are present. Model provenance identifies Parakeet v3 for both English sessions, Whisper large-v3-turbo for Spanish, Gemma 4 E4B finals, and the corresponding Marian CT2 direction. The recorded startup pipeline hash is the same across all three sessions; their Git revisions differ because other work continued. A startup source-file hash does not prove every imported engine's loaded bytecode.

The first session lacks a visible final ACK for **chunk 1**. The saved records do not establish why; this is missing evidence rather than a failed-display verdict. ACK coverage counts each final chunk once, whereas receipt-to-render measurements include all visible partial and final events. The driver's recorded medians were 12.2, 7.4, and 6.9 ms respectively, over 11, 8, and 71 visible ACKs. Those clocks start **after browser receipt** and do not measure speech-end-to-display latency. The paused session also has disrupted replay pacing and queue history. All three sessions used TTS and controlled mixed/synthetic inputs outside the frozen performance configurations, so these observations must not be pooled into the latency screen.

**Observed through the operator and audience browser**

The saved driver report records that startup stayed in `starting` until the production metrics header appeared; pause/resume worked; each language flip created a new session identity; and audience reconnection changed direction and cleared the previous session's history. The final English production CSV contains the John 3:16 cue, and repeated operator polls kept the verse result stable.

Live review showed persisted final segments, saved notes independently of audience captions, and retained a draft through polling and reload. Normal stopping drained work and produced completion markers. A completed-session export without approvals was rejected. The driver also recorded per-language TTS preference persistence. These observations are distinct from the automated safety and compatibility checks below.

The rehearsal exposed three operator presentation defects that were corrected: premature readiness, a small-sample p95 below p50, and stale language/TTS form values after a restart or reload. Startup now waits for a complete CSV header; p95 uses the nearest-rank definition; active configuration reflects confirmed session settings and is disabled for editing, while idle choices remain available for the next start. The p95 example `[2204, 2044]` now yields p50 **2124** and p95 **2204** in the automated collector test. These fixes are not a claim that the running operator process was hot-patched during the rehearsal.

**Correction workflow against the implementation plan**

| Requested behavior | Implementation and automated contract | Evidence from these sessions |
|---|---|---|
| Review during and after a session; finalized persisted chunks only | [Review API](../../operator_app/review.py), [review store](../../tools/review_data.py), [review tests](../../tests/test_operator_review.py) | Live finalized review observed; all three sessions have verified completion evidence. |
| Drafts, revision conflict recovery, separate transcript/translation approvals and exclusion | [Review UI](../../displays/operator/review.js), [browser race tests](../../tests/test_operator_review_browser.py), [revision/approval tests](../../tests/test_operator_review.py) | Two note revisions persisted; draft recovery observed. Both approval flags remain false on every saved rehearsal correction. |
| Safe session audio and EN/ES handling for legacy and current records | [Normalization and path checks](../../tools/review_data.py), [audio/language contract tests](../../tests/test_operator_review.py) | All retained final-chunk audio files verified; actual EN→ES→EN sessions remained separate. Traversal, symlink, ambiguous-language and collision cases are automated checks. |
| Completed-session portable exports with copied WAVs | [Exporter](../../tools/review_data.py), [bundle roundtrip test](../../tests/test_operator_review.py) | Completion and approval gates observed. An approved portable bundle was not manufactured from these rehearsal notes; its roundtrip is covered by automated fixtures. |
| Reuse existing CLI formats; idempotent, revision-safe merges; train/eval separation | [CLI preparation](../../tools/prepare_finetune_data.py), [merger](../../tools/merge_corrections.py), [CLI/provenance tests](../../tests/test_prepare_finetune_data.py) and [integration tests](../../tests/test_operator_review.py) | These sessions remain replay/evaluation data. No transcript approval, translation approval, training import, or retraining was performed. |

The saved notes are in `stark_data/corrections/20260909_210408_972943_en.jsonl` (chunk 4, revision 1) and `stark_data/corrections/20260909_211201_787725_en.jsonl` (chunk 2, revision 1). They describe the rehearsal and draft-recovery check. They are not human-approved reference text. Approval of one text does not approve the other; replay/unknown provenance cannot enter training exports. Legacy CLI use retains its explicit stop-before-handoff responsibility, while the web portable exporter requires completion evidence.

The previously recorded full automated suite reports **1,722 passed, 2 skipped, 57.83% coverage** in `.cache/mac-roadmap/full-tests.log`. The evidence JSON preserves that log's checksum and separates this result from browser observations. Representative tests also cover stale response ordering, numeric device IDs versus names, independent approvals, missing audio, portable bundle roundtrips, repeated imports, and evaluation leakage. No new test run was performed for this documentation audit.

**TTS and summary evidence**

All three operator sessions used `tts_output_mode=wav`. Their synthesis timing records have no playback-request boundary, so they do not prove audible onset or physical language routing. A separate two-language smoke synthesized and invoked playback on **MacBook Pro Speakers**:

| Language | Voice | WAV duration | Synthesis time | Selected physical output |
|---|---|---:|---:|---|
| EN | `en_US-lessac-high` | 1.765 s | 257.71 ms | MacBook Pro Speakers |
| ES | `es_MX-claude-high` | 2.101 s | 85.79 ms | MacBook Pro Speakers |

Both saved smoke WAVs have been verified. Host playback calls completed without a reported error; the smoke did not assess acoustic quality or speaker onset. The other enumerated endpoint was Microsoft Teams Audio, not a second physical output. No unplug/replug was performed.

The final English session's first summary had a formatting failure: its Spanish field contained alternate versions and explanatory text. That output remains preserved at `.cache/mac-roadmap/summary_format_failure_20260909_211201_787725_en.json`. The rerun in `metrics/summary_20260909_211201_787725_en.json` preserves the English summary and returns one clean three-sentence Spanish translation, with no alternate options or notes. The output records eight input segments, 147 words, an undiarized three-sentence format, and direct Gemma 4 E4B translation. This verifies output shape and the summary workflow; bilingual accuracy approval remains pending.

**Still pending**

- Natural Spanish and bilingual quality validation with approved human references.
- A real two-speaker diarization rehearsal; this summary has no speaker labels.
- A second physical TTS output, distinct language routing, and physical hotplug behavior.
- Acoustic TTS quality/onset verification.
- An approved training handoff from real live-session corrections. Automated export/import fixtures are not a substitute for this human workflow.
- Uninterrupted latency and visible-browser delivery gates on their frozen comparison inputs. These operator sessions are excluded.
