# Windows delivery and MSI status

Updated September 10, 2026. A v2026.13 MSI exists; its downloaded SHA-256 and embedded
ProductVersion were verified on the Mac. That is artifact inspection, not a Windows
installation test. [The recorded verification](../mac_implementation_status_20260909.md)
also documents removal of the obsolete v2026.12 asset. Published tags were not moved.
The v2026.14 source candidate has not been published as a package or release.

## Supported implementation and pending execution

The shared runtime has `lite-cpu`, `lite-cpu-quality` and `lite-cuda-8gb` profiles.
For native Windows/RTX2070 installation commands, model/native binary setup,
offline readiness and hardware gates, use [Lite profiles](../lite_profiles.md).
Those instructions still need execution on Windows. Mac CPU smokes do not certify
Windows, NVIDIA driver compatibility or RTX2070 memory/performance.

WSL training is a separate environment and task. See
[the Windows/WSL guide](../../CLAUDE-windows.md). Do not use a training environment
as the minimal volunteer inference installation.

## What the MSI workflow actually builds

[release-win.yml](../../.github/workflows/release-win.yml) reads the package version,
validates release identity, embeds project/interpreter/entry-point environment
variables while compiling PyApp, and wraps the executable with Briefcase 0.4.1.
A tag-triggered run attaches the MSI to that tag's GitHub release. Manual workflow
execution also builds an artifact; no workflow was dispatched for this turn.

The [asset directory](../../packaging/windows/README.md) contains the icon, WiX
fragment and a design-reference PyApp TOML. The workflow configures PyApp using
`PYAPP_*` environment variables; it does not read that TOML. Its `args`, update
strategy and extras field are therefore not proof of launcher behavior.

The WiX fragment proposes NVIDIA detection through `STARK_INSTALL_EXTRAS`, but
an end-to-end test must prove that the launcher actually consumes the selected
runtime extras, obtains a published matching wheel, starts the operator when
launched from the Start Menu and completes model setup. The current CLI entry
point expects a subcommand. Do not promise volunteers a verified one-click MSI
installation until this bootstrap chain is exercised and repaired as necessary.

## Remaining release gates

- Install the current MSI in a clean native Windows account, record signature,
  paths and first-launch behavior, and test offline relaunch/uninstall.
- Verify explicit CPU/RTX2070 profile selection, model download/resume, preflight,
  operator launch, EN/ES captions, microphone permission and physical output.
- Validate the original RTX2070's memory/latency targets using recorded evidence.
- Establish signing and updater delivery separately. No signing certificate,
  SmartScreen reputation threshold or functioning auto-updater is claimed here.
- Publish a new version/tag only after authorization. PyPI trusted-publisher
  configuration and package/release publication remain pending by user choice.

See [PyPI delivery](pypi.md), [model bootstrap](models.md),
[operator runbook](../operator_runbook.md) and [remaining tasks](../backlog.md).
