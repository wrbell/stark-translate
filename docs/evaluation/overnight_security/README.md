# September 10 installed-dependency security assessment

This supplements the [earlier scope audit](../mac_v2026_14_security.md). A passing
CI gate does not imply every package in an existing developer environment is free
of known advisories.

The read-only refresh at **07:55 UTC** retained the same findings: the
[working environment](working-mac-20260910T0755.json) has 52 advisory entries
across 11 packages; the [isolated Mac runtime](isolated-mac-20260910T0755.json)
has two entries in Torch; the [Lite runtime](lite-20260910T0755.json) has none.
The inventories contain 144, 167 and 61 dependency entries respectively, including
explicit unpublished-project skips. [Command receipts](audit-20260910T0755-receipts.json)
record the actual paths, timestamps and audit exit codes. The earlier snapshots
below remain available; no dependencies were changed by either audit.

## Actual installations inspected

- **Preserved working `stt_env`:** the read-only [full installed audit](working_mac_audit.json)
  reports 52 advisory entries across 11 packages. Some source entries are duplicate
  advisory IDs, so this is not 52 distinct vulnerabilities. This environment also
  contains development/optional packages not required by the fresh Mac runtime.
  No installed package in it was changed; the frozen EN↔ES benchmark uses its
  recorded versions.
- **Separate Mac artifact environment:** compatible installer tools were upgraded,
  `pip check` passes, and its [full installed audit](isolated_mac_audit.json) reports
  two findings in Torch 2.10.0. The remaining audited packages have no known finding
  in this audit. The unpublished project itself is skipped explicitly by PyPI;
  it is subject to source review, not a fabricated dependency audit result.
- **Separate Lite CPU/TTS runtime:** [the recorded audit](../lite_installer_security_20260910.json)
  reports zero known vulnerabilities after updating only pip/setuptools. Actual
  Torch-free imports and `pip check` pass. New CI installs the real Lite extras in
  an independent venv, audits the installed inventory, rejects unexpected skips
  and runs whenever its dependency/packaging contract changes.

## Unresolved Mac Torch findings

The selected MLX/SpeechBrain installation remains pinned to the exercised Torch
2.10 line and matching torchaudio. The audit reports:

| Advisory | Reported affected operation | Available remediation in the audit |
|---|---|---|
| PYSEC-2026-139 / CVE-2026-4538 | Local `pt2` loading/deserialization | No fixed version recorded |
| PYSEC-2025-194 / CVE-2025-3000 | Local `torch.jit.script` memory corruption | Torch 2.13.0 |

A separate upgrade attempt resolved a Mac ARM64 Torch 2.13.0 wheel but found no
matching torchaudio 2.13.0 distribution for this interpreter/platform. It failed
before mutation. Available matching torchaudio versions stopped at 2.11.0. We did
not force an incompatible pair or change the working benchmark environment.

Default Mac VAD uses bundled Silero weights; setup uses pinned model identities.
The live application does not offer a `pt2` or model-checkpoint upload endpoint.
These constraints limit the application paths exposed to untrusted models, but
are not patches and do not turn the audit green. Keep this dependency migration
open: validate a patched Torch/matching audio package set or separately isolate
optional diarization, then repeat installed imports, VAD tests and real EN/ES
inference before changing the verified dependency line.

The older working environment also contains Accelerate with a checkpoint-index
path traversal advisory. It is absent from the fresh Mac artifact installation.
The upstream [advisory](https://github.com/advisories/GHSA-4j2p-28q2-5m79) lists no
patched release at inspection. Do not introduce it into inference merely to make
an existing development environment match the new installer.

## Source review

The source review found an audience HTTP root exposing non-display files and an
operator server binding all interfaces by default without browser-origin checks.
The audience now serves an explicit display-asset allowlist (`tools/display_server.py`);
39 actual HTTP regression cases cover traversal, aliases, symlinks and private paths.
The operator defaults to localhost and validates Host, browser Origin and control
WebSocket requests (`operator_app/security.py`). An integrated browser recheck
remained connected; local API access returned 200 and a foreign Origin returned 403.
Generation-checked controls also prevent a queued Stop from acting on a newly
started session. Review exports omit private drafts, notes and unapproved target
text; cached/downloaded bundles are validated against that export contract.
The final delivery note records the complete integrated checks.

No secret values, transcripts, audio or authentication material are included in
these dependency audit files. Publication of packages/tags remains pending.
