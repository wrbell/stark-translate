# Installed Mac delivery — 760e948 / V6

V6 build, isolated installation, dependency assessment, four operator checks and
four EN/ES file smokes have completed on source
`760e948ec9efb9266a5419cf68ce5f1cb5cfabab`. The separate Standard and CPU Lite
full-service terminal gates also passed. The seven-stage delivery coordinator
completed at 00:22:27 UTC on September 11. Archive assembly and standalone
verification are a separate gate. Local delivery is separate from publication
and human or device certification.

Standard uses English Parakeet, Spanish Whisper and the selected E4B final
engine with existing Marian final routing, plus Marian previews. This V6 CPU Lite cohort uses Whisper-small CT2, Marian CT2 finals and
ONNX VAD on the CPU. The optional E2B `lite-cpu-quality` smoke is separate
evidence; no E2B full-service result is inferred here.

The build verified 169 first-party runtime members. The canonical wheel and both
wheels rebuilt from the sdist and Mac ZIP are byte-identical, with all 174 wheel
members preserved. The build report has five artifact entries because the same
wheel appears in three build roles; there are three unique artifact contents:

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `stark_translate-2026.14.0.0-py3-none-any.whl` | 784,904 | `70c52b836a7855787a17d67d18dc7fc56e21e289332d883337d8a9a24f949c9b` |
| `stark_translate-2026.14.0.0.tar.gz` | 113,502,965 | `5ede81f29dcad1f8e8d3257bfafb8fda8770e07df627f8de328177a8f8228181` |
| `stark-translate-v2026.14.0.0-mac.zip` | 115,076,986 | `766a032247e805951828ddb5e5c681e2715b0a058b53ad178f0933d063c1d09f` |

Fresh Standard and Lite environments were installed under the V6 output.
Checks ran outside the checkout beneath
`/private/tmp/stark-final-delivery-l_pgypg2`; working `stt_env` and baseline
environment inventories remained unchanged. The installed monitor explicitly
uses the 128 MiB read limit. Its production default remains 64 MiB, with a
validated 256 MiB hard cap.

| Installed dependency assessment | Installed distributions | Audited third parties | Known findings |
|---|---:|---:|---:|
| Standard | 130 | 129 | Two in Torch 2.10.0 |
| Lite | 67 | 66 | Zero |

Standard's retained findings are `PYSEC-2026-139` / `CVE-2026-4538` and
`PYSEC-2025-194` / `CVE-2025-3000`. Its audit exited 1 with those findings; Lite's
audit exited 0. Both assessments completed without environment changes. The
first-party package was the explicitly skipped distribution; this is not a
whole-project security clearance. Evaluation dependencies were included. The
separate patched dependency candidate was not promoted into working `stt_env`.

All four direct-CLI/launcher operator checks completed: six HTTP routes returned
200 for each profile/entry point, for 24 responses. Shutdown verification retained
the actual `-15` exits and required matching process identity, intentional TERM,
completed application lifecycle, no error/forced cleanup and a disappeared owned
process group. These were read-only installed-operator checks, without live
capture, playback or browser interaction.

The smoke rows use completed receipts through 22:20:18 UTC on September 10;
Standard full-service completed at 23:21:27 UTC that day and Lite at 00:22:25 UTC
on September 11. Every pipeline and monitor listed below exited 0, with no forced
cleanup or cleanup signals.

| File replay | Duration | Finals | Previews | Required writes completed | Validator checks |
|---|---:|---:|---:|---:|---|
| Standard EN→ES | 46.94 s | 6 | 49 | 63/63 | 47/47 passed |
| Standard ES→EN | 47.60 s | 6 | 23 | 38/38 | 47/47 passed |
| Lite EN→ES | 46.94 s | 6 | 7 | 22/22 | 46/46 passed |
| Lite ES→EN | 47.60 s | 6 | 8 | 22/22 | 46/46 passed |
| Standard full English service | 3,640.053 s | 563 | 2,800 | 4,200/4,200 | 47/47 passed |
| Lite full English service | 3,640.053 s | 467 | 214 | 1,188/1,188 | 46/46 passed |

Both full services used the same original source SHA-256
`8bec0f104dd883fd001f2a4b12ff9b454217c344f233458868feda07a0e83f53`,
gain 1 and 1× replay. The source's leading 403.1448125 seconds of digital zeros
remain in place, as documented in the [earlier source-bound rehearsal](../final-c13f51f/installed-delivery.md).
That source time is not an inference stall. Input remained file-only with
explicit `--no-tts`; the session monitor observed metadata and process resources,
without native capture probes or output-device playback.

The smokes total 24 finals, 87 previews and 145 required writes. Their validators
cover recorded model/source identity, source spans and EOF, required persistence,
preview/final records and process cleanup. Both full-service physical traces were
truncated and remain explicitly unassessable; their complete-trace call counts
are unavailable, not zero.

Final timing below is estimated last VAD-positive frame end → server payload
ready. First-partial timing starts at estimated speech start. Intervals between
successive server updates can contain intervening silence/music and exclude
leading/trailing source spans. There were no browser ACKs; these are server-side
measurements. Latency p95 is withheld below 100 observations in each cohort.

Standard's complete 174,722,544-sample source ledger at 48 kHz reached drained
EOF. The empty-target guard suppressed one preview, and the nonempty-preview
check passed. Whole-session STT counters balanced at 3,318 partial and 599 final
calls; its truncated physical trace remains unassessable, with no complete-trace
call count inferred. Silence finals (n=401) had server p50/p95 of
1,458.7/3,283.2 ms; smart cuts (n=124) 3,372.8/7,220.7 ms; hard cuts (n=38)
had p50 2,247.1 ms with p95 withheld. These are separate observed endpoint
cohorts, not a causal comparison or proof of the sub-second final-delivery goal.

Lite also accounted for the complete 174,722,544-sample source through drained
EOF. STT counters balanced at 782 partial and 599 final calls; the latter produced
467 completed finals, 21 empty-STT results and 111 rejections by the hallucination
policy. Those rejections are machine classifications, not verified acoustic
labels. No empty-target partial was observed. Silence finals (n=310) had server
p50/p95 of 3,297.7/10,322.0 ms; smart cuts (n=121) 5,002.1/14,268.4 ms; hard cuts
(n=36) had p50 3,534.3 ms with p95 withheld. First-partial delay (n=149 utterances)
had median 6,096.0 ms. Sampled queued finals reached eight, with p95 two; the
terminal drain completed. Lite passed its technical file-replay gate, but sparse
previews and long tails do not support recommending it as a fast production
profile on this Mac. These sessions do not establish performance on an RTX 2070.

The two full services total 1,030 finals, 3,014 previews and 5,388 required writes.
Together with the smokes, all six individual terminal validators passed with
1,054 finals, 3,101 previews and 5,533 completed writes. This establishes the
recorded file-replay and persistence acceptance. It does not establish archive
verification, microphone behavior, human meaning quality or
visible-browser delivery. Per-session timing populations remain separate.

The [c13/V5 cohort](../final-c13f51f/installed-delivery.md) remains separate. Its
monitor failed the old read limit; a separately retained reconstruction passed
artifact parsing but the original validator then rejected one blank translated
preview. Neither failed result is relabeled as passed by the new guard or V6.

The final delivery summary also rechecked source/artifact/installed-byte bindings,
unchanged working and baseline environment inventories, and all four operator
receipts. Its status is passed. The concise receipt-bound report was generated
at 00:23:02 UTC from `final-delivery-report-preparation/result.json` (SHA-256
`a45538291e6c353ec11ce44645994ed87c61ca2d722be66251226b0256d5ce3e`).
It reports unavailable full-service trace counts as null and withholds latency
p95 for every endpoint/cohort with fewer than 100 observations.

Primary origins are `final-delivery-v6/build/report.json`, `install/report.json`,
`installed-audit/report.json`, the four operator receipts and
`sessions/<case>/{receipt,validation}.json`. Archive entry points are the
[manifest](raw/artifact-manifest.json) and [standalone verifier](raw/verify_endurance.py);
archive verification has its own receipt and is not inferred from these local
delivery results. File-only input, explicit `--no-tts`, unchanged
defaults and separate source identities apply throughout. These observations do
not establish a causal Standard/Lite speed comparison, bilingual quality,
visible-browser latency, sustained microphone capture, speaker playback, RTX
2070 performance or public release publication.

The separate [archive verification](archive-verify-receipt.json) passed.
[Session timing and counter data](sessions-summary.json) retains endpoint counts
and unavailable physical traces explicitly.
