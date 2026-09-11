# Mac source validation — 760e948

These results bind source `760e948ec9efb9266a5419cf68ce5f1cb5cfabab`.
The [installed-delivery record](installed-delivery.md) separately records all
seven completed V6 delivery stages and six passed session validators, including
both full-service replays. Source tests alone do not certify that service behavior.

| Check | Completed result | Scope |
|---|---|---|
| Local Python 3.11 | 2,861 passed, four skipped, 20 subtests; 66.76% coverage; 137.85 s | Full `tests/` suite with production parser, writer, event and lifecycle regressions |
| [GitHub Python 3.11](https://github.com/wrbell/stark-translate/actions/runs/34535847823/job/103067142375) | 2,859 passed, six skipped, 20 subtests; 66.63% coverage; 138.03 s | Exact source head; job succeeded |
| [GitHub Python 3.12](https://github.com/wrbell/stark-translate/actions/runs/34535847823/job/103067142615) | 2,859 passed, six skipped, 20 subtests; 66.63% coverage; 137.74 s | Same source head and workflow run; job succeeded |
| Prescribed static checks | All 11 commands passed | Ruff, formatting, mypy, configured Bandit, Tidy version and six HTML pages |
| Actual MLX regressions | Three passed, none skipped; 21.50 s | Text-only model loading/generation; no audio device |
| Isolated pre-commit | Seven hooks passed | Existing isolated tool/cache; no installation into working `stt_env` |

Subtests are reported separately and must not be added to the independent passing
test count. Bandit retained the prescribed B101/B603/B607/B615 exclusions; it does
not establish dependency security clearance or exhaustive download pinning.

The original local wrapper failed after pytest passed because it attempted to
create its mutable state file a second time with exclusive-write semantics. Its
failure and stale state remain preserved. A separately identified continuation
validated the original pytest receipt, log, JUnit and coverage, then ran only
static checks, GPU tests and pre-commit. The consolidated source-check summary
records this provenance; pytest was not repeated to hide the wrapper failure.

The coverage-comment correction was exercised on GitHub: Python 3.11 used
`pytest-json-coverage-path: coverage.json` and its comment step succeeded without
the old format annotation. The Python 3.12 comment step was intentionally skipped.
Coverage JSON/XML and the coverage threshold remain in place.

The tested source includes the [blank-preview guard](../final-c13f51f/empty-preview-repair.md)
and the bounded monitor-reader override. Their hashes are:

- `dry_run_ab.py`: `95e9b66bbd06ddbf913a5f3d9fccdf9617a0ff6b9c923e05ef1e2e3945ab50c1`.
- `tools/endurance_monitor.py`: `4a754c528681ee83552b5063f04c8e809676169f8168eaf301e5f62dd76698a8`.

The retained source summary is `final-fixed-source-760e948/summary.json`. Its
archive member is `validation-760e948/summary.json`, alongside CI logs/metadata
and the original CPU and continuation receipts. The [archive manifest](raw/artifact-manifest.json)
and [standalone verifier](raw/verify_endurance.py) provide the archive entry points;
their separate verification receipt determines archive status. No microphone,
speaker or human-quality certification is inferred from these source checks.
The assembled documentation refresh passed all 38 documentation contract tests,
backlog schema/render checks and local links in 20 canonical guides plus 12
final-packet documents. [Documentation receipts](documentation-checks/documentation.json)
retain actual command outcomes; these checks introduce no new inference claim.

The separate [archive verification](archive-verify-receipt.json) passed.
[Session timing and counter data](sessions-summary.json) retains endpoint counts
and unavailable physical traces explicitly.
