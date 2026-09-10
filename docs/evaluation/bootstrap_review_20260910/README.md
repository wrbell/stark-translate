# Bootstrap review corrections — September 10, 2026

The automated review of PR #192 found two installer control-flow defects after
the frozen `84832fb` runtime checks. Bootstrap rejected an explicit new VENV
before it could create it, and started services before model setup/preflight.
Both are fixed. An explicit missing or empty target can now be created; existing
environment precedence is preserved. The target is resolved to an absolute path,
with its systemd executable quoted. Setup and doctor must both succeed before
systemd or launchd is changed.

The [worktree-bound receipt](report.json) records the parent revision and exact
script/test hashes. The new tests executed copied production shell control flow
with inert installer/service commands and only the log destination isolated:
**eight failures and six passes before the fix; all 14 passed after it**.
Eleven existing environment/bootstrap checks also passed. Bash syntax and Ruff
checks passed. No real installations, model downloads, services or working
`stt_env` changes occurred in these tests.

`bootstrap.sh` is distributed in the source/Mac ZIP, not in the canonical wheel.
The [848 runtime validation](../overnight_final_validation_20260910.md) remains
bound to that source; these installer checks are separate and do not establish
new inference, hardware or latency results.
