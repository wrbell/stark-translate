# Final integration validation — September 10, 2026

**Source `84832fb` passed 2,384 CPU tests, four skips and 63.80% coverage (91.56 s),
prescribed static checks and repeated installed-artifact checks.** This includes
the fix for a real operator shutdown crash discovered during the preceding package
check. PR #192 is still open and not merged at this snapshot. No new model or
endurance run was performed after the cleanup fix.

| Evidence | Result and retained receipt |
|---|---|
| CPU suite | 2,384 passed, four skipped; clean source before/after. [Receipt](overnight_final_validation_20260910/cpu/report.json), [pytest log](overnight_final_validation_20260910/cpu/pytest.log). |
| Prescribed static checks | Ruff/format, mypy engines/settings, configured Bandit with zero findings, and six HTML5 pages passed. [Receipt](overnight_final_validation_20260910/static/report.json). This is not a new local dependency audit. |
| Wheel, sdist and Mac ZIP | All 152 packaged runtime members match final source; wheels rebuilt from sdist/ZIP match canonical. Isolated Mac and Lite installs passed without dependency changes or model inference. [Receipt](overnight_final_validation_20260910/artifacts/report.json). |
| Native enumeration and process exit | Three checks per environment, six total: HTTP200, exit0, no surviving polling workers or model imports. **Zero input/output devices were visible; no audio stream was opened.** This does not certify microphones, speakers or hotplug. |
| Wheel continuity | **138/143 code/resource members identical; five operator cleanup modules changed.** Every pipeline/engine module is unchanged. [Full comparison](overnight_final_validation_20260910/continuity/README.md), [all member hashes](overnight_final_validation_20260910/continuity/report.json). |
| Exact-head remote CI | Eight checks succeeded; the legacy `audit` check was skipped. [Snapshot](overnight_final_validation_20260910/ci/pr192-84832fb.json). Later documentation commits are outside this snapshot. |

Final source: `84832fb98caf4504cf65b1f5969558b0be911f5c`.
Canonical wheel SHA-256:
`ac3d5216ce79f2ef004e2ed5be4c42d5eccb3f2ce8612661a2a7d138c7c2c41e`.
Exact receipts, hashes, command arguments and scope are in the
[machine-readable report](overnight_final_validation_20260910.json) and
[retained evidence inventory](overnight_final_validation_20260910/inventory.json).

## The shutdown defect and its fix

The earlier `60ad4db` package smoke printed successful HTTP and import assertions,
then the process exited with **SIGSEGV (-11)**. The background audio watcher could
still be importing native sounddevice code as the interpreter shut down. This was
a real lifecycle bug, not a successful check or a harness-only false failure.
The [failed receipt](overnight_final_validation_20260910/failed-60ad/report.json)
and [short process log](overnight_final_validation_20260910/failed-60ad/mac-installed-smoke.log)
are retained. The log itself records completed assertions; the receipt records the
subsequent signal exit. Private OS crash content is excluded.

The fix stops and joins existing application workers on lifespan exit, including
exceptional exit, without creating missing workers; failed bounded joins are
reported. The five changed wheel members are `operator_app/audio.py`,
`operator_app/features.py`, `operator_app/main.py`, `operator_app/metrics.py` and
`operator_app/pipeline_manager.py`. The repeated native checks deliberately imported
sounddevice for enumeration and then exited; the validator now distinguishes that
native device module from model-runtime imports. No capture or output stream opened.

## What remains bound to 752

Both completed Standard and Lite hours used source `752ab9a` and wheel
`7477574d25c91739b6a88ca142a35bf36258599a66671b8dbb32237d1fa852b5`.
Their [validation](overnight_validation_20260910.json),
[artifact receipt](overnight_artifact_validation_20260910.json),
[three-cohort endurance report](overnight_endurance_20260910/README.md) and raw
archive are unchanged. The 138 identical code/resource members establish exact
continuity outside the five operator modules; they do not make the new cleanup
code part of an earlier hour. Three Markdown resources and two wheel metadata
members also differ, as recorded separately in the comparison.

The three real GPU regressions passed on 752 in 21.16 seconds; none was rerun for
this cleanup-only validation. The separate [96-run screen](overnight_screen_20260910/README.md)
still selected 0/28 arms and did not meet the sub-second final goal. No latency or
quality promotion follows from the final mechanical checks.

This is a pre-merge source snapshot. #134/#176/#177 are still open in this record;
#193/#194 remain hymn/quality follow-ups. The last published release is v2026.13.
Source merge, release publication, human review, attended devices and x86/RTX2070
certification are separate actions and evidence.
