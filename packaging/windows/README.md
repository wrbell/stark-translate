# Windows MSI build assets

This directory holds the inputs to the v2026.7.2.0 Windows MSI build:

| File | Purpose |
|---|---|
| `pyapp-config.toml` | PyApp launcher config (Python version, target wheel, extras detection) |
| `wix-fragment.wxi` | WiX fragment Briefcase wraps PyApp's binary with (Start Menu shortcut, CUDA detection, ARP metadata) |
| `icon.ico` | App icon — **TODO**, not committed yet (placeholder before the icon source is finalized) |

The build is driven by `.github/workflows/release-win.yml` on `v*` tags.

See [`docs/packaging/windows.md`](../../docs/packaging/windows.md) for the full
plan including the unsigned-MSI / SmartScreen click-through UX and the
v2026.7.2.1 code-signing follow-up.

## Local build (for dev iteration)

```pwsh
# Install PyApp + Briefcase build deps
cargo install pyapp --root packaging\windows\pyapp-build
pip install briefcase==0.3.25

# Build the launcher
$env:PYAPP_PROJECT_NAME = "stark-translate"
$env:PYAPP_PROJECT_VERSION = "2026.7.2.0"
$env:PYAPP_PYTHON_VERSION = "3.12"
$env:PYAPP_EXEC_SPEC = "operator_app.cli:main"
$env:PYAPP_PIP_EXTERNAL = "true"
cd packaging\windows\pyapp-build
cargo build --release

# Wrap into MSI
cd ..\..\..
briefcase package windows --no-sign
```

The resulting MSI lands at `dist\stark-translate-2026.7.2.0.msi`.
