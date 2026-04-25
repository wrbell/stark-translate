# Windows MSI — v2026.7.2 (planned, unsigned)

> **Status:** scaffold only. The Windows MSI build is the next track on
> the v2026.7 roadmap. Code-signing is a follow-up patch (v2026.7.2.1)
> blocked on Sectigo/DigiCert cert procurement (~$300/yr).

For non-technical Windows volunteers. The MSI installer drops a thin Rust
launcher (PyApp, ~5 MB) into Program Files, registers a Start Menu entry,
and on first run downloads python-build-standalone + the
`stark-translate[cuda|cpu]` wheel into `%LOCALAPPDATA%\stark-translate`.

---

## First-launch UX (unsigned)

On first install, Windows SmartScreen will show:

> *"Windows protected your PC — Microsoft Defender SmartScreen prevented
> an unrecognized app from starting."*

Click **More info → Run anyway**. The warning gradually disappears as the
binary accumulates download reputation (typically after ~30 unique installs
and a few weeks of dwell time). The operator runbook will document this
click path with screenshots so volunteers don't bounce off it.

After the click-through, a Start Menu entry "Stark Translate" launches the
operator UI in the default browser.

---

## What's planned

| Component | Tool | Why |
|---|---|---|
| Rust launcher | [PyApp](https://github.com/ofek/pyapp) | ~5 MB binary, vs PyInstaller --onefile's 800 MB-1.5 GB |
| MSI wrapper | [Briefcase 0.3.25](https://briefcase.readthedocs.io/) "external app packaging" | Wraps PyApp's binary into a standard WiX MSI |
| Auto-updater | [WinSparkle](https://winsparkle.org/) | EdDSA-signed appcast; works against unsigned MSIs |
| CUDA detection at install time | PowerShell `Get-WmiObject Win32_VideoController` | Sets `STARK_INSTALL_EXTRAS=[cuda]` or `[cpu]` |

```text
packaging/windows/
  icon.ico
  wix-fragment.wxi              MSI customization (Start Menu, mic permission opt-in)
  pyapp-config.toml             pinned Python version, target wheel, launch arg
.github/workflows/release-win.yml   Windows runner builds the MSI on v* tag
```

---

## Why PyApp instead of PyInstaller

|  | PyInstaller --onefile | PyApp |
|---|---|---|
| Binary size | 800 MB – 1.5 GB | ~5 MB |
| Cold-start delay | 10–30 s extract | <500 ms |
| Updates | Rebuild whole binary | Bump `pyproject`, PyApp re-pip-installs |
| Antivirus false positives | Frequent | Rare |
| ML wheel ecosystem fit | Painful | Designed for it |

PyApp downloads python-build-standalone on first run (~30 s) and
pip-installs the published wheel. After first launch it caches the venv in
`%LOCALAPPDATA%\stark-translate` and starts in <500 ms thereafter.

---

## Code-signing roadmap

- **v2026.7.2.0** — unsigned MSI ships. SmartScreen click-through documented.
- **v2026.7.2.1** — purchase Sectigo/DigiCert cert (~$300/yr, ~1 week
  procurement), attach via `signtool` in CI, ship signed MSI. WinSparkle
  EdDSA appcast signing is independent and will be enabled at the same time.

---

## See also

- [`pypi.md`](./pypi.md) — what PyApp pip-installs at first run
- [`models.md`](./models.md) — model bootstrap that ships in the same wheel
- [`../operator_runbook.md`](../operator_runbook.md) — operator UI walkthrough
