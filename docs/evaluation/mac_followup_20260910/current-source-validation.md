# Current source validation — 2026-09-10

Configured CI passed for `476e349d6887b9f4e2379f9e57646bdd2bb2c581`.
Read-only GitHub workflow/job metadata and both retained local logs were checked
independently. The [JSON receipt](current-source-validation.json) records exact
job identities, log hashes, summary lines and scope.

| Job | Passed | Skipped | Warnings | Coverage | Test duration |
|---|---:|---:|---:|---:|---:|
| [Python 3.11](https://github.com/wrbell/stark-translate/actions/runs/34506381777/job/102969484815) | 2,813 | 6 | 2 | 66.24% | 127.63 s |
| [Python 3.12](https://github.com/wrbell/stark-translate/actions/runs/34506381777/job/102969484404) | 2,813 | 6 | 2 | 66.24% | 134.24 s |

Both jobs passed the four new startup fallback cases and the repaired
documentation contract. [Lint](https://github.com/wrbell/stark-translate/actions/runs/34506381619)
also passed: Ruff check/format, scoped mypy and Bandit, advisory Vulture and HTML
validation. Bandit still explicitly skips B615; this does not clear the broader
[model-source inventory](live-hf-source-inventory.md).

The [Security workflow](https://github.com/wrbell/stark-translate/actions/runs/34506381639)
and its wrapper succeeded, but both `audit` and `audit-lite` were **skipped**.
The job log records `requirements = false` and `lite = false`: no dependency
changes matched the configured filters. This run supplies no new dependency audit.

The earlier [full suite on `fb44a0f`](https://github.com/wrbell/stark-translate/actions/runs/34505780125)
remains failed evidence: Python 3.11 recorded 2,812 passed, one failed and six
skipped. Its four startup cases passed; the sole failure was a documentation
assertion that still required PR #192 where the current draft was PR #196.
Commit `476e349` repaired that contract without changing runtime source.
The [pinning receipts](live-hf-pinning-receipts.json) retain their earlier
observation-time status; this separate receipt records the subsequent CI result.

No local test suite, native model, audio capture/output, build or installation was
run to prepare this receipt. Final installed-artifact EN/ES checks, queued model
experiments, human quality and device acceptance remain separate. Microphone
capture and output playback remain prohibited for this session. Defaults and
publication status are unchanged.
