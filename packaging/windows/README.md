# Windows MSI build assets

This directory holds the inputs to the v2026.7.2.0 Windows MSI build:

| File | Purpose |
|---|---|
| `pyapp-config.toml` | PyApp launcher config (Python version, target wheel, extras detection) |
| `wix-fragment.wxi` | WiX fragment Briefcase wraps PyApp's binary with (Start Menu shortcut, CUDA detection, ARP metadata) |
| `icon.ico` | App icon — Stark Road Gospel Hall logo, multi-resolution ICO (16/24/32/48/64/128/256 px). Source: `wp-content/uploads/2015/12/SRGH_Logo.jpg` from starkroadgospelhall.com, padded to square + upscaled to a 512 px LANCZOS master before ICO export. |

The build is driven by `.github/workflows/release-win.yml` on `v*` tags.

See [`docs/packaging/windows.md`](../../docs/packaging/windows.md) for the full
plan including the unsigned-MSI / SmartScreen click-through UX and the
v2026.7.2.1 code-signing follow-up.

## Local build (for dev iteration)

```pwsh
# 1. Install Briefcase
pip install briefcase==0.3.25

# 2. Set PyApp env vars *before* cargo install — PyApp's build.rs embeds
#    these into the resulting binary at compile time.
$env:PYAPP_PROJECT_NAME = "stark-translate"
$env:PYAPP_PROJECT_VERSION = "2026.7.2.0"
$env:PYAPP_PYTHON_VERSION = "3.12"
$env:PYAPP_EXEC_SPEC = "operator_app.cli:main"
$env:PYAPP_PIP_EXTERNAL = "true"

# 3. cargo install IS the build — produces a customized pyapp.exe
cargo install pyapp --root packaging\windows\pyapp-build
Copy-Item packaging\windows\pyapp-build\bin\pyapp.exe `
          packaging\windows\stark-translate.exe -Force

# 4. Wrap into MSI
briefcase package windows --adhoc-sign
```

The resulting MSI lands at `dist\stark-translate-2026.7.2.0.msi`.
