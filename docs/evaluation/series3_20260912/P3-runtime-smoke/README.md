# P3-S — installed smoke and runtime audit of the promoted Mac environment (2026-09-11)

**Both pass.** The wheel installed in `venv/` (stark-translate 2026.14.0.0, built from `337790a`) passes the
out-of-checkout installed smoke, and a full `pip-audit` of the environment's site-packages reports
**0 findings across 124 distributions** (first-party skipped: stark-translate).

## Installed smoke

`cd /private/tmp && /Users/willem/Code/vibes/SRTranslate/venv/bin/python -m tools.installed_smoke` (run from outside the
checkout so the provenance guard proves the installed wheel, not the checkout, was imported). Output
([`installed_smoke.json`](installed_smoke.json)): package `/Users/willem/Code/vibes/SRTranslate/venv/lib/python3.11/site-packages/operator_app/__init__.py`; routes {'/healthz': 200, '/operator/': 200, '/operator/review.js': 200, '/operator/widgets/qr.js': 200, '/api/capabilities': 200}; model manifest
entries 22; runtime files 45; verse parser passed. No model loads, no network.

## Runtime audit

`pip-audit --path venv/lib/python3.11/site-packages --format json (auditor .cache/package-smoke/bin/pip-audit)` → [`installed-audit-summary.json`](installed-audit-summary.json). Key versions:
ctranslate2 4.7.1, mlx 0.32.2, mlx-lm 0.31.3, mlx-whisper 0.4.3, parakeet-mlx 0.5.2, pip 26.2.1, setuptools 84.0.0, silero-vad 6.2.1, torch 2.13.0, torchaudio 2.11.0, transformers 5.12.1. CI's weekly audit strips `torch*`/`mlx*` and cannot
cover this environment; the P2-L launcher PR adds `scripts/audit_mac_runtime.sh` and a `--runtime mac` mode of
`tools/check_dependency_audit.py` so this audit is a one-line monthly command (no scheduler is installed on the
Mac; that remains Willem's choice). The check-mode result (PR #213 merged as `c914e14`) is appended below.

This certifies package provenance and known-vulnerability status only; it is not a functional, quality or device gate.

### `--runtime mac` check (tool from `main` @ `c914e14`)

```
{"audited": 123, "skipped_local_project": ["stark-translate"], "known_vulnerabilities": 0, "runtime": "mac"}
```
