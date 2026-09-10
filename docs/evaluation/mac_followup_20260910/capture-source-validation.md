# Capture source and fixture validation — 2026-09-10

Configured CI passed for PR head
`f7b959d7b7f8de953fef2265c0a29f0ab264fd2f`, after the capture-accounting fixture
repair. Both jobs checked out GitHub's PR merge revision
`d137251171ea5f51053471f9d36a7244f70c2729`; the [JSON receipt](capture-source-validation.json)
records head and checkout identities separately, primary workflow/job metadata,
exact log hashes, summary lines and scope. The earlier
[`476e349` validation](current-source-validation.md) and its
[JSON receipt](current-source-validation.json) are preserved unchanged.

| Job | Passed | Failed | Skipped | Subtests passed | Warnings | Coverage | Duration |
|---|---:|---:|---:|---:|---:|---:|---:|
| [Python 3.11.16](https://github.com/wrbell/stark-translate/actions/runs/34514178514/job/102995386031) | 2,834 | 0 | 6 | 20 | 2 | 66.49% | 133.30 s |
| [Python 3.12.14](https://github.com/wrbell/stark-translate/actions/runs/34514178514/job/102995386364) | 2,834 | 0 | 6 | 20 | 2 | 66.49% | 149.79 s |

The Ubuntu 24.04.5 jobs installed `.[dev]` plus SciPy, jiwer and openpyxl and ran
`pytest tests/` with coverage over engines, tools and features and a 50% floor.
Both logs retain all 21 new capture-accounting cases and all six previously
failing cases as passed. The two warnings concern Starlette test-client/AnyIO
deprecations. These are configured Linux source tests, not native Mac capture
or output-device acceptance.

## Retained failure and narrow correction

The [preceding run](https://github.com/wrbell/stark-translate/actions/runs/34513515987)
on PR head `d7ed43d881c054a4a621dfbc354ceaebee52dc0e` remains failed evidence.
Its checkout merge revision was `711761e2af0f64d2ae4755cfeffad305e9219383`.

| Job | GitHub conclusion | Passed | Failed | Skipped | Subtests passed | Coverage | Duration |
|---|---|---:|---:|---:|---:|---:|---:|
| [Python 3.11.16](https://github.com/wrbell/stark-translate/actions/runs/34513515987/job/102993182051) | failure | 2,828 | 6 | 6 | 20 | 66.49% | 127.96 s |
| [Python 3.12.14](https://github.com/wrbell/stark-translate/actions/runs/34513515987/job/102993181638) | cancelled | 2,828 | 6 | 6 | 20 | 66.49% | 143.22 s |

The 3.12 job is labelled **cancelled** by GitHub, but its completed test log
contains the six-failure summary and its Run tests step is marked failure. It
must not be counted as a passing run or assigned an invented missing result.
Both failed logs identify the same six cases: two music/pause tests, capture
retry, and stop/pause/disconnect variants of the buffer-outcome test.

Unrestricted `MagicMock` streams invented a `capture_snapshot()` method and
truthy loss counters. The new close reconciliation correctly treated these as
reported capture failures, exposing inaccurate fixtures and shared transport
state. [Commit `aca2604`](https://github.com/wrbell/stark-translate/commit/aca2604ac14425650dc2a0d3320d8a07cf393f38)
restricts scripted mocks to their context-manager interface and supplies a real,
fresh `CaptureTransportSummary` for each test. It changes the two test files
and the accounting note; runtime behavior is unchanged. Subsequent `f7b959d`
adds archived experiment evidence/documentation. The original six failures are
retained in this receipt rather than relabelled successful.

## Lint and actual security scope

[Lint passed](https://github.com/wrbell/stark-translate/actions/runs/34514178583):
Ruff check, format check (350 files), mypy over engines/settings (22 source
files), the configured Bandit scan, advisory Vulture and HTML error validation.
Bandit explicitly excludes **B101, B603, B607 and B615**. Vulture runs with
`|| true`; its successful step is not a clean dead-code verdict.

The [Security workflow](https://github.com/wrbell/stark-translate/actions/runs/34514178486)
passed its wrapper and **ran `audit-lite` successfully**. Change detection
recorded `requirements=false`, `lite=true`, with `pyproject.toml` matching the
Lite filter. The legacy requirements `audit` job was **skipped**.

The [installed Lite job](https://github.com/wrbell/stark-translate/actions/runs/34514178486/job/102995472128)
resolved **`.[lite-cpu,tts]`**, passed `pip check`, and audited the resulting
site-packages. Its retained [artifact](https://github.com/wrbell/stark-translate/actions/runs/34514178486/artifacts/10167031069)
contains 61 entries: **60 dependencies audited, zero known vulnerabilities**, and
the unpublished local `stark-translate` project skipped explicitly. The 943-byte
artifact download matches GitHub's recorded SHA-256
`38b52d362d90d92f8cdd327d8c3301bbaf0d90a5daec67ed7d143333f52c592e`.

This audit does **not** install the `[eval]` extra or establish the installed
Mac GPU environment's security status. It does not replace the
[model-source inventory](live-hf-source-inventory.md), remove Bandit's exclusions,
or support an “all security clear” claim.

Preparing this receipt used read-only GitHub metadata/log downloads and a tiny
JSON audit artifact. No local tests, model execution, native imports, audio
capture/output, build or installation were performed. Final installed-artifact
checks, queued CPU/deadline experiments, human quality and device gates remain
separate. Microphone capture and output playback remain prohibited for this
session; defaults and publication status are unchanged.
