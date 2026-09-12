# Windows MSI build assets

This directory holds Windows MSI build inputs. The unsigned MSI ships with the
v2026.14.0.0 GitHub Release (2026-09-11); native Windows installation and first-launch
behavior remain unverified (`windows-msi-bootstrap` in `docs/backlog.json`). Boundaries and
gaps: [Windows delivery status](../../docs/packaging/windows.md).

| File | Purpose |
|---|---|
| `pyapp-config.toml` | Design reference only and stale (still carries `2026.7.2.0` and an old exec spec); the workflow uses environment variables and does not read this file |
| `wix-fragment.wxi` | WiX fragment Briefcase wraps PyApp's binary with (Start Menu shortcut, CUDA detection, ARP metadata) |
| `icon.ico` | App icon — Stark Road Gospel Hall logo, multi-resolution ICO (16/24/32/48/64/128/256 px). Source: `wp-content/uploads/2015/12/SRGH_Logo.jpg` from starkroadgospelhall.com, padded to square + upscaled to a 512 px LANCZOS master before ICO export. |

The build is driven by `.github/workflows/release-win.yml` on `v*` tags.

See [`docs/packaging/windows.md`](../../docs/packaging/windows.md) for the full
implementation boundaries, bootstrap gaps and remaining signing/hardware gates.

## Local build (for dev iteration)

```pwsh
# 1. Install Briefcase
pip install briefcase==0.4.1

# 2. Set PyApp env vars *before* cargo install — PyApp's build.rs embeds
#    these into the resulting binary at compile time.
$env:PYAPP_PROJECT_NAME = "stark-translate"
$env:PYAPP_PROJECT_VERSION = python -c "import tomllib; print(tomllib.load(open('pyproject.toml', 'rb'))['project']['version'])"
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

The resulting MSI lands under `dist` with the package version. Building it does
not verify that its first-launch dependency/model setup works on Windows.
