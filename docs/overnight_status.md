# September 10 delivery status

The candidate branch `codex/mac-reliability-roadmap` is proposed in
[PR #192](https://github.com/wrbell/stark-translate/pull/192). Its operator, reliability,
Lite, latency and documentation work is integrated. Final validation, measured
experiment conclusions and the authorized merge remain in progress. PyPI, package
publication and release tags remain pending by user choice.

## Where to look

| Need | Current document |
|---|---|
| Implemented behavior, validation and unresolved gates | [Mac implementation status](mac_implementation_status.md) |
| Every remaining task with acceptance and ownership | [Machine-readable backlog](backlog.json), [rendered backlog](backlog.md) |
| Volunteer operation and recovery | [Operator runbook](operator_runbook.md) |
| Current runtime interfaces and defaults | [Architecture](current_architecture.md) |
| EN↔ES speed experiments and decision rules | [Experiment plan](evaluation/overnight_experiment_plan.md) |
| Actual browser/session evidence | [Operator rehearsal](evaluation/overnight_operator_rehearsal.md) |
| CPU and original RTX2070 installation/performance gates | [Lite profiles](lite_profiles.md) |
| Installed dependency findings and scope | [Security assessment](evaluation/overnight_security/README.md) |
| Original open-issue acceptance | [Issue closure audit](issue_closure_audit.md) |

## Documentation refresh

Root and directory README/CLAUDE/AGENTS guides now describe shared-code Lite,
actual Mac STT defaults, health-derived readiness, correction approvals and timing
schema boundaries. Training and Windows guides separate WSL execution from native
Windows inference. Deployment and Gemma tuning documents label planned interfaces
that never shipped instead of presenting them as available commands.

The old long Windows design guide, August Mac refresh and pre-overnight Mac status
are preserved as dated historical records. Current Mac and Windows packaging guides
replace obsolete installation claims. MSI file/hash inspection remains distinct
from an actual Windows first-launch test. The PyApp reference TOML is not consumed
by the workflow and cannot establish automatic profile/extras installation.

## Decisions and evidence boundaries

- EN↔ES is the active latency priority. E4B remains standard; E2B and scheduler
  experiments remain opt-in until matched speed/coverage/quality evidence warrants
  a change. Hindi is separate offline R&D and has no further overnight experiments.
- Real-time visible browser measurements use schema 2. Speech end → ACK is an upper
  bound including return-network time. Missing ACKs remain missing. File replays,
  synthetic audio and unreviewed transcripts are always labeled.
- The user deferred live microphone and physical-output tests until tomorrow.
  Natural Spanish, two-speaker audio, bilingual approval, x86/RTX2070 execution,
  WSL training and adapter transfer remain explicit external dependencies.
- Issue #134 permits a laptop stand-in with a full recorded hymn, spoken segment,
  setup-to-first-caption timing and written UX note. Do not add live-mic or church
  hardware requirements to that issue. #131 has its own live EN/ES microphone gate.
- No approved correction, retrain cycle, natural-language reference score or target
  hardware certification is inferred from synthetic/stub tests.
- Source push, justified issue updates and merge are authorized. Published tags must
  not move; release artifacts and PyPI remain pending.

## Validation commands

```bash
python tools/render_backlog.py validate
python tools/render_backlog.py render --check
python tools/render_backlog.py check-links
pytest tests/test_documentation.py -v
```

Refresh evidence and integration state after the final checks and merge; retain
historical measurements and failure logs with their original source boundaries.
