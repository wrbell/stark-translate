# September 10 delivery status

The v2026.14 source (`2026.14.0.0`) is tracked by
[PR #192](https://github.com/wrbell/stark-translate/pull/192); v2026.13 is the last published release recorded here.
The operator, reliability, Lite, latency and documentation work is integrated. The English matrix is complete:
96/96 valid runs, 672 finals and 0/28 selected arms. The sub-second final-delivery
goal was not met on this cohort; E4B remains unchanged. Endurance and the
source-integration evidence remain separate from that screen. PyPI, package
publication and release tags remain pending by user choice.

## Where to look

| Need | Current document |
|---|---|
| Implemented behavior, validation and unresolved gates | [Mac implementation status](mac_implementation_status.md) |
| Every remaining task with acceptance and ownership | [Machine-readable backlog](backlog.json), [rendered backlog](backlog.md) |
| Volunteer operation and recovery | [Operator runbook](operator_runbook.md) |
| Current runtime interfaces and defaults | [Architecture](current_architecture.md) |
| Completed English matrix and retained negative results | [Compact screen report](evaluation/overnight_screen_20260910/README.md) |
| EN↔ES speed experiments and decision rules | [Experiment plan](evaluation/overnight_experiment_plan.md) |
| Final local CPU/GPU check counts | [Validation record](evaluation/overnight_validation_20260910.json) |
| Actual browser/session evidence | [Operator rehearsal](evaluation/overnight_operator_rehearsal.md) |
| CPU and original RTX2070 installation/performance gates | [Lite profiles](lite_profiles.md) |
| Installed wheel, sdist and Mac ZIP evidence | [Runtime artifact validation](evaluation/overnight_artifact_validation_20260910.json) |
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

The earlier standard full-service attempt is retained as failed source-bound
validation evidence and does not satisfy endurance acceptance. The next validation
uses the complete source recording: a fresh full-length standard replay, followed
by a separate full-length CPU Lite replay. Results for both remain pending. The
prepared 350-second slice was unused and is not a validation cohort.

The recorded local CPU suite passed 2,213 tests
with four skips and 63.52% coverage; all three GPU regressions passed. Endurance
completion and remote CI must be recorded separately.

## Decisions and evidence boundaries

- EN↔ES is the active latency priority. E4B remains standard; E2B and scheduler
  experiments remain opt-in. No arm warrants ordinary confirmation or a combined
  configuration after the 96-run screen. New endpoint/scheduling hypotheses are
  separate work. Hindi is offline R&D with no further overnight experiments.
- Real-time visible browser measurements use schema 2. Speech end → ACK is an upper
  bound including return-network time. Missing ACKs remain missing. File replays,
  synthetic audio and unreviewed transcripts are always labeled. In the completed
  screen the native Mac was locked while the DOM reported visible; physical display
  visibility is unverified. The package-activity overlap remains annotated.
- Live microphone and physical-output tests are deferred to the next attended session.
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
