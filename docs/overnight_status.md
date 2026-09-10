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

The earlier Standard hour completed required writes and cleanup but failed
source-bound validation (23/549 final spans and 97/2,720 preview spans); it remains
a separate failed cohort. The fresh full-service Standard session
`20260910_043120_839144_en`, installed from frozen source `752ab9a`, completed at
09:32:38.589810 UTC with exit 0. All 563 final spans and retained WAV headers,
and all 2,814 preview spans, are consistent; all 7,594 writes completed without
failure and process cleanup was observed. One document-visible ACK connection
matched all 563 finals. These checks do not establish human quality, physical
screen visibility or complete source-speech coverage.

CPU Lite session `20260910_053518_894101_en`, using the same `752ab9a` wheel,
completed at 10:36:47.110148 UTC with exit 0. All 468 final spans/WAV headers and
271 preview spans agree; all 1,979 writes completed with zero pending/failed, and
cleanup was observed. One matched document-visible connection acknowledged all
468 finals and 271 translated previews, but only 174/468 finalized utterances had
a first translated preview. Lite is functionally exercised on this Mac; its sparse
previews and large observed latency tails do not support recommending it as a fast
production profile today. Hardware and human-quality certification remain pending.

[The retained endurance report](evaluation/overnight_endurance_20260910/README.md)
keeps the old failed Standard, repaired Standard and Lite cohorts separate. Their
concurrent lightweight activities are disclosed; observed Standard/Lite speed
is not a causal paired comparison. Selected waveform reconstruction matched three
windows in each repaired cohort, while reproducing the old chunk 141 deficit.
That is supporting regression evidence, not exact EOF or all-source coverage.
Lite's last caption was followed by 162.549 seconds of unclassified source; its
closing-hymn context must not be described as a quiet tail. The prepared 350-second
slice was unused.

Frozen source `752ab9a` passed 2,363 CPU-suite tests with four skips and 63.80%
coverage (89.97 s), plus three real GPU regressions (21.16 s), pre-commit and the
prescribed static checks. Its canonical wheel matches all 152 runtime members;
sdist/ZIP rebuilds are byte-identical, and both isolated installs passed their
mechanical checks without inference or dependency changes. Previous check/artifact
records remain linked separately. The Standard and Lite terminal results are recorded separately from those
mechanical checks. Final documentation/workflow-head CI and the authorized source
merge remain pending; package publication is separate.

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
