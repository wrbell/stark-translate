# P — Torch 2.13 Mac runtime promotion (2026-09-11)

**Result: promoted.** The Mac launcher now selects `venv/` (Torch 2.13.0 / TorchAudio 2.11.0, the audited
constraint set) through the repository pointer `.stark-python`; the unmodified `stt_env` (Torch 2.10.0,
freeze SHA256 `a09be8422c195824…` before and after) remains the rollback environment, switched back with one
line. Source PR [#208](https://github.com/wrbell/stark-translate/pull/208) (merged `337790a`) carried the
pins, the constraints file, the preflight bounds and the pointer; PR
[#210](https://github.com/wrbell/stark-translate/pull/210) made preflight accept the rollback runtime again
after the drill below exposed the gap. Raw receipts are under [`raw/`](raw/).

This does not certify human translation quality, live-microphone capture, physical outputs or a p95/speed
claim; the Sunday attended microphone session on the promoted environment is a separate gate.

## P1 — source (PR #208, Codex lane, reviewed and merged 15:30Z)

`pyproject.toml` `mlx` extra pins `torch==2.13.0`; `diarization` pins `torch==2.13.0` + `torchaudio==2.11.0` on
darwin/arm64 and keeps `torchaudio>=2.10,<2.11` elsewhere. `constraints/macos-arm64-py311-runtime.txt` holds
the audited 122 pins. `operator_app/preflight.py` requires `torch>=2.13,<2.14` (and `torchaudio>=2.11,<2.12` with
`--diarize`). `scripts/runtime_env.sh::stark_apply_python_pointer` reads `.stark-python` in the caller's shell
and exports `STARK_PYTHON`; precedence is `STARK_PYTHON` > `VENV` > `.stark-python` > `VIRTUAL_ENV` >
`CONDA_PREFIX` > `stt_env` > `venv`, so the pointer beats the auto-activated conda base on this Mac.
`run_operator.sh` prints the selected interpreter. `venv/`, `.venv/` and `.stark-python` are git-ignored.
Full local suite on the branch: 2975 passed, 4 skipped.

## P2 — promoted environment build (off-GPU, 15:35–15:38Z)

- Source: `main` @ `337790a` (`git archive HEAD` → scratch `source/`; production `pyproject.toml` SHA256 `39d9cc3b7bcea6df…`, identical to the archived copy).
- Constraints: `constraints/macos-arm64-py311-runtime.txt` (SHA256 `8159db5f25bc0bb9…`; 122 pins, payload identical to the audited candidate's `25c6b79e…`).
- Wheel: `stark_translate-2026.14.0.0-py3-none-any.whl` SHA256 `b7499b7be6eafde5…`; Requires-Dist carries `torch==2.13.0` (mlx) and `torchaudio==2.11.0` (diarization, darwin/arm64).
- Environment: `venv/` at the repository root, created by `/Users/willem/anaconda3/bin/python3.11 -m venv`, pip 26.2.1 / setuptools 84.0.0, `--only-binary=:all:` resolution against the constraints; hash-pinned install (`--require-hashes`, 123 requirement lines). The resolved third-party set is **byte-identical** to the audited overnight candidate (`torch213-main-20260911`); only the first-party wheel differs.
- Receipts (`command-receipts.json`): venv-create rc=0, installer-bootstrap rc=0, full-application-resolution rc=0, full-application-install rc=0, pip-check rc=0, installed-inventory rc=0, installed-metadata-consistency rc=0, native-imports-vad rc=0, full-installed-audit rc=0.
- Installed metadata consistency: `passed` (124 distributions, extras mlx+diarization, 0 issues).
- Native smoke: `passed` (imports of torch/torchaudio/speechbrain/mlx/parakeet/operator/engines inside `venv`, bundled Silero VAD on CPU).
- Installed audit (`pip-audit --path site-packages`): **0 findings across 124 distributions** (first-party skipped).
- `venv/bin/python -m pip freeze | shasum -a 256` = `d8b3d68187e2f6d1…`; `torch 2.13.0 · torchaudio 2.11.0 · mlx 0.32.2`.
- `doctor --backend mlx` in `venv`: en → GPU=pass, Runtime dependencies=pass, Models=pass, Microphone=pass, Adapter manifest=warn; es → GPU=pass, Runtime dependencies=pass, Models=pass, Microphone=pass, Adapter manifest=warn; en `--diarize` → GPU=pass, Runtime dependencies=pass, Models=pass, Microphone=pass, Adapter manifest=warn. (The adapter-manifest warning is the same base-model notice `stt_env` shows.)

## P3 — GPU revalidation in the promoted environment (15:39–15:58Z)

1. **Normalized FLEURS replays from the installed wheel** (`replay_candidate.py --lang en|es --execute`,
   `venv/bin/python -m dry_run_ab` from site-packages, `STARK_MODELS_DIR=.cache/mac-roadmap/ct2-setup-validation/managed`):
   `torch213_promote_normalized_r1_en` and `_es` both `passed_functionality`, validator 19/19 checks each
   (`raw/normalized-r1-summary.json`, `raw/normalized_{en,es}_receipt.json`).
2. **Paired 150 s church-clip replays** (`stark_data/replay/Gospel_Message_(12_14_25)_5D2rOMvkwrk.wav`, offset
   1290 s, SHA256 `db11fb4c…`; `tools/replay_bench.py` from the checkout at `337790a`, `--profile standard
   --stt-backend parakeet-mlx --gemma4-size e4b --vad-backend torch --no-mts --gain 1 --replay-speed 1`; control
   `stt_env/bin/python`, candidate `venv/bin/python`; order ctl/cand, cand/ctl, ctl/cand; same `pipeline_sha256`
   `36799a551fdf…` in every run):

| pair | arm | finals | previews | silence n | silence p50 ms | silence p95 ms | peak RSS MiB | peak Metal MiB | pipeline sha |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| r0 | ctl | 26 | 220 | 8 | 1919.8999999999999 | 2395.2 | 4444 | 8783 | 36799a551fdf |
| r0 | cand | 26 | 220 | 8 | 1834.4 | 2364.6 | 4030 | 8650 | 36799a551fdf |
| r0 | identity | 26/26 spanish, 26/26 english identical | p50 delta -85.5 ms (within gate: True) | previews ok: True | metal ok: True | | | | |
| r1 | ctl | 26 | 221 | 8 | 1833.6 | 2356.4 | 4058 | 8776 | 36799a551fdf |
| r1 | cand | 26 | 222 | 8 | 1844.0 | 2594.6 | 3980 | 8660 | 36799a551fdf |
| r1 | identity | 26/26 spanish, 26/26 english identical | p50 delta 10.4 ms (within gate: True) | previews ok: True | metal ok: True | | | | |
| r2 | ctl | 26 | 222 | 8 | 1843.0 | 2367.2 | 4258 | 8687 | 36799a551fdf |
| r2 | cand | 26 | 220 | 8 | 1816.6 | 2317.4 | 4187 | 8632 | 36799a551fdf |
| r2 | identity | 26/26 spanish, 26/26 english identical | p50 delta -26.4 ms (within gate: True) | previews ok: True | metal ok: True | | | | |

   Gate: final captions byte-identical in **3/3 pairs (26/26 Spanish and 26/26 English each)**; silence-final
   p50 within max(5 %, 100 ms) in 3/3 pairs (−85.5, +10.4, −26.4 ms); preview counts within 2 pp; peak Metal
   within 1 GiB (8.6–8.8 GiB). Eight silence finals per run: an equivalence screen, not a speed or p95 claim.

## P4 — launcher switch and rollback drill (15:58Z, repeated 16:06Z)

`.stark-python` = `/Users/willem/Code/vibes/SRTranslate/venv/bin/python`. From a terminal with conda base
auto-activated (`CONDA_PREFIX=/Users/willem/anaconda3`), `./run_operator.sh` on port 9017, no session started:

```
pointer -> /Users/willem/Code/vibes/SRTranslate/venv/bin/python
[promoted] startup line: starting operator at http://127.0.0.1:9017 (logs: /Users/willem/Code/vibes/SRTranslate/metrics; python: /Users/willem/Code/vibes/SRTranslate/venv/bin/python Python 3.11.11)
[promoted] preflight backend=mlx&lang=en rc=0 checks 5 | non-pass: [('Adapter manifest', 'warn')]
[promoted] preflight backend=mlx&lang=es rc=0 checks 5 | non-pass: [('Adapter manifest', 'warn')]
[promoted] operator stopped; python line in log: python: /Users/willem/Code/vibes/SRTranslate/venv/bin/python Python 3.11.11
pointer -> /Users/willem/Code/vibes/SRTranslate/stt_env/bin/python  (rollback drill)
[rollback] startup line: starting operator at http://127.0.0.1:9017 (logs: /Users/willem/Code/vibes/SRTranslate/metrics; python: /Users/willem/Code/vibes/SRTranslate/stt_env/bin/python Python 3.11.11)
[rollback] preflight backend=mlx&lang=en rc=0 checks 5 | non-pass: [('Runtime dependencies', 'fail'), ('Adapter manifest', 'warn')]
[rollback] preflight backend=mlx&lang=es rc=0 checks 5 | non-pass: [('Runtime dependencies', 'fail'), ('Adapter manifest', 'warn')]
[rollback] operator stopped; python line in log: python: /Users/willem/Code/vibes/SRTranslate/stt_env/bin/python Python 3.11.11
pointer restored -> /Users/willem/Code/vibes/SRTranslate/venv/bin/python
```

**Finding.** The promoted pointer works end to end (operator healthy in 2 s; preflight `en`/`es` pass; the only
warning is the pre-existing adapter-manifest notice). The rollback pointer launched `stt_env`, but preflight
reported `Runtime dependencies: fail` because `#208` tightened the bound to `torch>=2.13,<2.14`, so Start would
have been blocked on the rollback environment. Fix: PR #210 accepts the retained rollback ranges
(`torch>=2.10,<2.11`, `torchaudio>=2.10,<2.11`) on the mlx backend with a `pass` that names both ranges. Repeat of
the drill against the fixed source (port 9018, from the fix worktree, which has no `adapters/` symlink — hence
its `Models: fail` on both arms; it is not an environment result):

```
pointer -> /Users/willem/Code/vibes/SRTranslate/venv/bin/python
[promoted] startup line: starting operator at http://127.0.0.1:9018 (logs: /Users/willem/Code/vibes/SRTranslate-wt-preflight/metrics; python: /Users/willem/Code/vibes/SRTranslate/venv/bin/python Python 3.11.11)
[promoted] preflight backend=mlx&lang=en rc=0 checks 5 | non-pass: [('Models', 'fail'), ('Adapter manifest', 'warn')]
[promoted] preflight backend=mlx&lang=es rc=0 checks 5 | non-pass: [('Models', 'fail'), ('Adapter manifest', 'warn')]
[promoted] operator stopped; python line in log: python: /Users/willem/Code/vibes/SRTranslate/venv/bin/python Python 3.11.11
pointer -> /Users/willem/Code/vibes/SRTranslate/stt_env/bin/python  (rollback drill)
[rollback] startup line: starting operator at http://127.0.0.1:9018 (logs: /Users/willem/Code/vibes/SRTranslate-wt-preflight/metrics; python: /Users/willem/Code/vibes/SRTranslate/stt_env/bin/python Python 3.11.11)
[rollback] preflight backend=mlx&lang=en rc=0 checks 5 | non-pass: [('Models', 'fail'), ('Adapter manifest', 'warn')]
[rollback] preflight backend=mlx&lang=es rc=0 checks 5 | non-pass: [('Models', 'fail'), ('Adapter manifest', 'warn')]
[rollback] operator stopped; python line in log: python: /Users/willem/Code/vibes/SRTranslate/stt_env/bin/python Python 3.11.11
pointer restored -> /Users/willem/Code/vibes/SRTranslate/venv/bin/python
```

Rollback runtime dependencies now pass on `stt_env`. The pointer was restored to `venv/bin/python`; `stt_env`'s
freeze SHA256 is unchanged.

**Final drill on `main` @ `77de207` (17:58Z, after #210 merged, main checkout, port 9017):**

```
pointer -> /Users/willem/Code/vibes/SRTranslate/venv/bin/python
[promoted] startup line: starting operator at http://127.0.0.1:9017 (logs: /Users/willem/Code/vibes/SRTranslate/metrics; python: /Users/willem/Code/vibes/SRTranslate/venv/bin/python Python 3.11.11)
[promoted] preflight backend=mlx&lang=en rc=0 checks 5 | non-pass: [('Adapter manifest', 'warn')]
[promoted] preflight backend=mlx&lang=es rc=0 checks 5 | non-pass: [('Adapter manifest', 'warn')]
[promoted] operator stopped; python line in log: python: /Users/willem/Code/vibes/SRTranslate/venv/bin/python Python 3.11.11
pointer -> /Users/willem/Code/vibes/SRTranslate/stt_env/bin/python  (rollback drill)
[rollback] startup line: starting operator at http://127.0.0.1:9017 (logs: /Users/willem/Code/vibes/SRTranslate/metrics; python: /Users/willem/Code/vibes/SRTranslate/stt_env/bin/python Python 3.11.11)
[rollback] preflight backend=mlx&lang=en rc=0 checks 5 | non-pass: [('Adapter manifest', 'warn')]
[rollback] preflight backend=mlx&lang=es rc=0 checks 5 | non-pass: [('Adapter manifest', 'warn')]
[rollback] operator stopped; python line in log: python: /Users/willem/Code/vibes/SRTranslate/stt_env/bin/python Python 3.11.11
pointer restored -> /Users/willem/Code/vibes/SRTranslate/venv/bin/python
stt_env freeze: a09be8422c195824
```

Both pointers pass readiness (the adapter-manifest warning is the pre-existing base-model notice). `.stark-python` is left at `venv/bin/python`.

**Rollback (one line, documented in `docs/packaging/macos.md` and `CLAUDE-macbook.md`):**

```bash
printf '%s\n' "$PWD/stt_env/bin/python" > .stark-python
```

Note: the wheel installed in `venv` was built from `337790a`; launches from the checkout (`./run_operator.sh`,
`python -m …`) run the checkout's code, so later merges on `main` do not require rebuilding the environment
unless a dependency pin changes.
