# Installed c13 Mac delivery

The tested inference source is `c13f51f1346b1581079b3457e34cb4d5fd0c2565`.
Build, installation, read-only operator checks and four EN↔ES file smokes passed.
The Standard full-service pipeline completed; its original monitor hit a reader
limit. A separate corrected report recovered its artifacts, then the unchanged
terminal validator failed on one blank translated preview. The failed acceptance
is retained; final rehearsals against a repaired producer remain pending.

## Artifacts and installation

| Artifact | SHA-256 | Bytes |
|---|---|---:|
| `stark_translate-2026.14.0.0-py3-none-any.whl` | `7369382dca07e0d9f129850f3a3f1a639c24955035c8d7c31c5e8148be414c44` | 783,944 |
| `stark_translate-2026.14.0.0.tar.gz` | `73c65578e79e4af0600c2806cad11e469296c246d808bf5adb11ea93136ef213` | 113,465,626 |
| `stark-translate-v2026.14.0.0-mac.zip` | `ce43f77d05b693d87703d1f97fa5712a391ca3b360e0edc27d53d1530a1ddb43` | 115,027,649 |
| `stark_translate-2026.14.0.0-py3-none-any.whl` | `7369382dca07e0d9f129850f3a3f1a639c24955035c8d7c31c5e8148be414c44` | 783,944 |
| `stark_translate-2026.14.0.0-py3-none-any.whl` | `7369382dca07e0d9f129850f3a3f1a639c24955035c8d7c31c5e8148be414c44` | 783,944 |

Canonical, sdist-rebuilt and Mac-ZIP-rebuilt wheels are byte-identical. All 169
runtime members match frozen source; all 174 wheel members were checked. Mac ZIP
extraction restored the 11 committed executable modes. Fresh isolated Standard
`[mlx,diarization,eval]` and CPU Lite `[lite-cpu,eval]` installations passed metadata,
profile, launchd rendering, `pip check`, installed smoke and member checks outside
the checkout. The working `stt_env` and baseline environments were unchanged.

Four read-only operator launches exercised the installed CLI and installed shell
launcher for each profile. All six requested HTTP routes returned 200 in each
launch. Uvicorn completed ordered graceful shutdown on the requested SIGTERM;
raw exit −15 is retained. No microphone, speaker or live-device probe was used.

## Installed EN↔ES file smokes

| Profile / direction | Source seconds | Finals | Previews | Required writes | Physical STT calls | Validation checks |
|---|---:|---:|---:|---:|---:|---:|
| Standard EN→ES | 46.94 | 6 | 52 | 66 | 59 | 47 |
| Standard ES→EN | 47.60 | 6 | 30 | 44 | 49 | 47 |
| CPU Lite EN→ES | 46.94 | 6 | 6 | 21 | 23 | 46 |
| CPU Lite ES→EN | 47.60 | 6 | 7 | 21 | 23 | 46 |

Each pipeline and monitor exited zero, consumed the source, completed required
writes and cleaned up without signals or forced termination. Each smoke has a
complete physical trace and verified retained WAV headers, hashes and sample
spans. These engineering inputs have unapproved human quality; counts are not
pooled latency evidence. Standard uses Parakeet for English, Whisper for Spanish,
E4B finals and Marian previews. CPU Lite uses Whisper small and Marian CT2 finals,
without E2B, Torch or MLX.

## Full-service status

The original 3,640.053-second English source, including 403.1448125 seconds of
leading digital zero samples, is replayed at gain 1 and real-time speed with
explicit `--audio-file` and `--no-tts`. Its SHA-256 is
`8bec0f104dd883fd001f2a4b12ff9b454217c344f233458868feda07a0e83f53`.
These separate runs are observational functional checks, not a paired causal
Standard/Lite speed comparison or visible-browser delivery measurement.

Standard's pipeline exited zero with 563 finals and all 3,946 writes complete.
Its original monitor exited two: the valid 91,240,742-byte diagnostics exceeded
its 64 MiB reader limit. Independent retained-data review found no JSON errors;
the file's SHA-256 exactly matches the completed lifecycle marker. The original
failed monitor and coordinator receipts remain unchanged. The bounded reporter
repair successfully reconstructed terminal artifacts with an explicit 128 MiB
limit in a separate posthoc report. Original runtime samples, timestamps, process
ownership and cleanup observations were unchanged.

The unchanged terminal validator then passed 46 of 47 checks and failed
`nonempty_previews`: event `partial:2013`, utterance 271, had source text
`Просид о` and an empty Spanish target. This was one of 2,562 persisted previews.
The production producer awaited Marian but accepted its empty result, replacing
the previous translated preview. It was not an intentional source-only stage.
The original validator, corrected monitor report and failed reconciliation all
remain retained. The run is not reclassified as accepted by weakening the gate.

A narrow producer repair keeps the previous caption when Marian returns an empty
or whitespace-only target and records the rejected update explicitly. It changes
no model, language detector, confidence threshold or fallback policy. Fresh final
Standard/Lite rehearsals will use the repaired source and bounded monitor;
the old c13 service remains a negative result.

The full-service research trace retained 262,144 of Standard's 474,952 events;
212,808 earlier events were discarded. Whole-session source, persistence and
terminal validation are distinct from complete physical-call trace evidence,
which is unassessable for that truncated trace. No complete-trace claim is made.

## Dependency audit and retained failures

The full installed Standard audit covered 130 distributions, including 129 third
parties, and retained two known findings in Torch 2.10: PYSEC-2026-139 /
CVE-2026-4538 and PYSEC-2025-194 / CVE-2025-3000. CPU Lite covered 67 distributions,
including 66 third parties, with zero known findings. Each explicitly skipped
only the unpublished first-party package. Audit execution completed; Standard
is not security-cleared. The separate patched Torch/audio candidate's clean
audit does not transfer to these unchanged installed environments.

Earlier delivery-helper failures are preserved: v1 lost executable modes during
ZIP extraction; v2 rejected intentional Uvicorn −15; v3 observed a lingering
child without a passive shutdown grace; v4 required an optional shutdown log
prelude. Their narrowly repaired validators do not rewrite those outcomes. V5's
Standard monitor limit failure is likewise retained separately from any later
reconciliation. None is relabeled as a successful original delivery run.

Human acoustic/bilingual approval, sustained native capture, physical outputs,
two-speaker diarization, x86/RTX2070, public package publication and source merge
remain separate gates. Source checks are in [source validation](source-validation.md).
