# Final Mac source validation — c13f51f

This record binds validation to source
`c13f51f1346b1581079b3457e34cb4d5fd0c2565`. The installed-package and full-service
results belong in their separate delivery record. Later documentation or workflow
changes must retain this source identity rather than relabeling these tests.

| Check | Actual result | Scope |
|---|---|---|
| Local Python 3.11 suite | 2,837 passed, four skipped, 20 subtests; 66.62% coverage; 144.84 s | Real production parser, writer, event and lifecycle regressions |
| GitHub Python 3.11 | 2,835 passed, six skipped, 20 subtests; 66.49% coverage; 138.38 s | Successful job in run 34519299109, exact PR head c13f51f |
| GitHub Python 3.12 | 2,835 passed, six skipped, 20 subtests; 66.49% coverage; 152.00 s | Same successful run and PR head |
| Prescribed static checks | All 11 command receipts passed | Ruff, formatting, mypy, configured Bandit and six HTML5 Tidy pages |
| Real MLX regressions | Three passed, none skipped; 27.85 s | Text-only model loading/generation; no microphone or playback |
| Isolated pre-commit | All seven hooks passed | Fresh tool environment and cache; working `stt_env` unchanged |

The original sequence recorded pre-commit as unavailable. A separately retained
supplemental run executed it successfully against unchanged source; the original
skip was not rewritten. JUnit includes subtests as separate cases, so its totals
must not be reported as additional independent passing tests.

GitHub's coverage-comment step emitted a format annotation while concluding
success: the workflow sent JSON through the action's text-coverage input. That
reporting defect did not fail either test job. The subsequent one-line workflow correction selects the action's JSON input;
its later CI result is recorded separately from this c13 test run.
The workflow's narrow Lite dependency audit passed while its broader audit job
was skipped. That does not clear the final Standard environment or `[eval]`.

Bandit used the repository's documented exclusions, including B615. A separate
source review examined selected pinned Mac loading/fallback paths, input-device
identity, terminal capture accounting and shutdown. It found no actionable blocker
within that scope. Neither check establishes global download pinning or dependency
security clearance.

The Python 3.11 Stop regression reproduces the old cancellation loss in the actual
production loop and passes with the same-task timeout repair. Capture tests keep
counted FIFO loss separate from PortAudio overflow with an unknown sample count.
They do not retrospectively pass the retained failed native Spanish session.

The operator widgets are integrated, the current runbook links recorded browser
evidence, and all six Tidy checks pass. Health/readiness and stalled-capture failure
reporting have production regression coverage. These meet the corresponding
bounded implementation acceptances in the backlog. Sustained native capture,
physical output and visible-display delivery retain their separate gates.

No native microphone or speaker testing occurred in this validation after the
public-place restriction. The completed hymn control used file input with TTS
disabled; the boundary comparison used isolated text. Human acoustic annotations,
bilingual approvals, CUDA/RTX2070 execution and public package publication remain
outside these results.

The final packet index records raw archive provenance and subsequent reporting
changes separately from this frozen source validation.
