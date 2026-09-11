# Mac validation after the blank-preview and monitor-reader repairs

This packet records source `760e948ec9efb9266a5419cf68ce5f1cb5cfabab` and its V6
installed artifacts. Source checks and CI passed. All seven V6 delivery stages
completed on September 11 at 00:22:27 UTC, including four installed operator
checks, four EN/ES file smokes and separate Standard/CPU Lite full-service runs.
The six terminal validators passed with 1,054 finals, 3,101 previews and all
5,533 required writes completed. The evidence archive was built and independently
reconstructed and checksum-verified at 00:42:38 UTC on September 11.

- [Source validation](source-validation.md): local and CI suites, actual text-only
  GPU checks, prescribed static checks, pre-commit and retained wrapper failure.
- [Installed delivery](installed-delivery.md): three unique artifact contents,
  isolated environments, actual dependency findings, completed smokes and both
  3,640.053-second file-service gates.
- [Earlier c13 packet](../final-c13f51f/README.md): separate historical hymn/source
  evidence and the failed V5 monitor/blank-preview acceptance. Its measurements
  and failures retain their original identity.

Blank Marian previews now preserve the last usable caption and record their
suppression. The monitor supports an explicit bounded 128 MiB artifact read while
keeping the 64 MiB default. Neither repair changes model selection or inference
thresholds. EN↔ES latency and quality gates, human/device checks, CUDA/RTX2070 work
and publication retain their recorded dependencies.

Standard suppressed one empty translated preview in its fresh full service, and
all published previews passed the nonempty check. CPU Lite remained functional
but produced sparse previews and long latency tails. Neither session meets the
sub-second final-delivery goal. Their bounded full-service physical traces remain
unassessable; whole-session counters do not reconstruct discarded events.

Archive entry points are the [manifest](raw/artifact-manifest.json) and
[standalone verifier](raw/verify_endurance.py). The verifier checks the sharded
archive into a new output directory. Internal `validation-760e948/`, `delivery/`
and `historical-v5/` paths are archive member names, not individual public URLs;
extract the reconstructed archive to read them. Archive creation and verification
status belongs to the separately recorded archive receipt.

The [archive verification receipt](archive-verify-receipt.json) and [result](archive-verify.log)
record 993 evidence members plus the inventory, in 82 bounded shards
(40,259,903 compressed bytes). Manifest SHA-256:
`23692ba4a9958a1b3c7fca1b2f042fc4ab3f3f90616951b737cacaf7b3701f8f`.
The verifier reconstructs the archive and checks every member without extracting
files or opening original audio. [Redaction provenance](archive-redaction-manifest.json)
retains original/derivative identities for four already-masked CI header fields
and three embedded package-description fields. All dependency resolutions, test
results and original failed-run identities are preserved; originals stay local.

[Session summary JSON](sessions-summary.json) retains each cohort and endpoint
separately. The [generator source](sessions-summary-generator.py.txt) and
[source/schema review](sessions-summary-review.md) record how terminal receipts
were checked; it is a retained local report generator, not an installed CLI.

All 169 installed runtime source members remain byte-identical after this
[documentation refresh](runtime-documentation-continuity.json). The tested
artifacts retain their exact 760e948 identities; this is no new inference run.

[Actual integration closeout](../integration-closeout/README.md) records PR #196
merged and #132 closed, separately from these runtime measurements.
