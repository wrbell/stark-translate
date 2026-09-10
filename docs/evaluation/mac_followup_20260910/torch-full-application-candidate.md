# Full application Torch migration candidate

The isolated full application candidate with Torch **2.13.0** and TorchAudio
**2.11.0** passes installation, dependency consistency, native imports and
bundled CPU VAD. After fixing six non-model dependency pins in a second fresh
venv, its full installed audit returns **zero known findings for 123 third-party
distributions**. The unpublished first-party `stark-translate` entry is explicitly
skipped. In a later reserved window, installed **EN and ES replay functionality
passed** on normalized public development clips. Production migration,
performance regression and service certification remain pending.

This extends the [minimal runtime feasibility](../security_feasibility_20260910/README.md).
The 144-package working `stt_env` inventory and production `pyproject.toml`
[remain unchanged after both installations](torch_candidate/working-environment-preserved-final.json). Existing
packaged runtimes were not installation targets. The installation checks loaded
no STT, Gemma or speaker model; only three silence frames ran through bundled
CPU VAD. The subsequent replays loaded the existing STT and Gemma models without
further installation.

## Frozen application and complete dependencies

The candidate is a complete wheel of the selected application extras
`stark-translate[mlx,diarization]`, plus `jiwer==4.0.0` for later evaluation.
Only the scratch wheel's dependency lines change from Torch/TorchAudio
`>=2.10,<2.11` to `torch==2.13.0` and `torchaudio==2.11.0`.
The [actual wheel METADATA](torch_candidate/remediated/wheel-metadata.json)
confirms those requirements, so `pip check` is meaningful. An additional
[installed metadata check](torch_candidate/remediated/installed-metadata-consistency.json)
explicitly evaluates both selected project extras; ordinary `pip check` alone
does not prove optional-extra consistency.

The [source manifest](torch_candidate/preparation.json) binds 163 build inputs
to base commit `c1a9041a48adc7c55e123eb39db02edb2a6fa6bd` plus the retained
[working-source patch](torch_candidate/frozen-source-changes.patch). Five inputs
were uncommitted at snapshot time: `dry_run_ab.py`, `tools/fixed_span_delivery.py`,
`tools/mac_followup_latency.py`, `tools/source_coverage.py` and
`tools/training_preflight.py`. The separate
[dependency patch](torch_candidate/pyproject.candidate.diff) changes only the
scratch project. This is a frozen source snapshot, not a claim that the base
commit alone equals the wheel.

Both candidate venvs use the identical local wheel:
`stark_translate-2026.14.0.0-py3-none-any.whl`, SHA-256
`48ae49c1214f8a0bcabcc018794b675ece271114da172b7cf81c0696f88c7786`.
Its local hash distinguishes it from production artifacts of the same version;
it is not a release artifact and must not be published.

| Runtime dependency | Candidate version |
|---|---:|
| Torch / TorchAudio | 2.13.0 / 2.11.0 |
| MLX / MLX Metal | 0.32.2 / 0.32.2 |
| mlx-lm / mlx-whisper | 0.31.3 / 0.4.3 |
| mlx-optiq / parakeet-mlx | 0.4.34 / 0.5.2 |
| Transformers / SpeechBrain | 5.12.1 / 1.1.1 |
| pip / setuptools | 26.2.1 / 84.0.0 |

The MLX/STT package versions match the working inventory. SpeechBrain is added
for the selected optional diarization extra. Torch 2.13 requires newer setuptools;
that build helper changes only inside the new environments. The initial offline
[METADATA closure](torch_candidate/metadata-closure.json) was preparation, and
the actual [pip resolution](torch_candidate/remediated/pip-resolution.json)
subsequently selected 123 distributions including the application. The installed
inventory contains 124 distributions including pip.

Every dependency is pinned in the final
[constraints](torch_candidate/remediated/constraints.txt). The
[hash-locked requirements](torch_candidate/remediated/resolved-requirements.txt)
retain exact wheel URLs and SHA-256 values. Installation used `--require-hashes`
with dependency resolution enabled; there was no `--no-deps` workaround.

## Preserved initial failure and completed remediation

The [first candidate audit](torch_candidate/initial/installed-audit.json) reported
35 findings in six baseline-pinned packages. That venv and its original inventory
are retained. The second venv changes exactly these six distributions, as verified
by the [inventory comparison](torch_candidate/remediated/result-summary.json):

| Dependency | Initial | Remediated | Initial findings |
|---|---:|---:|---:|
| Pillow | 12.1.1 | 12.3.0 | 26 |
| pydantic-settings | 2.13.1 | 2.14.2 | 1 |
| Pygments | 2.19.2 | 2.20.0 | 2 |
| python-dotenv | 1.2.1 | 1.2.2 | 1 |
| Requests | 2.32.5 | 2.33.0 | 1 |
| urllib3 | 2.6.3 | 2.7.0 | 4 |

These are the minimum versions satisfying all reported fixes for each package.
The [collected PyPI metadata](torch_candidate/remediated/pypi-fix-metadata.json)
retains exact version endpoints, requirements, available wheel hashes and
collection times. Official release pages are [Pillow](https://pypi.org/project/pillow/12.3.0/),
[pydantic-settings](https://pypi.org/project/pydantic-settings/2.14.2/),
[Pygments](https://pypi.org/project/Pygments/2.20.0/),
[python-dotenv](https://pypi.org/project/python-dotenv/1.2.2/),
[Requests](https://pypi.org/project/requests/2.33.0/) and
[urllib3](https://pypi.org/project/urllib3/2.7.0/).

| Remediated check | Actual result | UTC completion |
|---|---|---|
| Full pip resolution | Exit 0, 123 distributions | 13:51:05.110910 |
| Hash-locked full installation | Exit 0 | 13:51:52.888683 |
| `pip check` | Exit 0 | 13:51:53.188831 |
| Selected-extra metadata consistency | Exit 0 | 13:51:53.654339 |
| Full installed audit | Exit 0; zero findings, one explicit first-party skip | 13:53:02.001710 |
| Installed native imports and CPU VAD | Exit 0 | 13:53:34.506351 |

The [complete receipts](torch_candidate/remediated/command-receipts.json) preserve
argv, working directories, timestamps, exit codes and log hashes. The
[native smoke result](torch_candidate/remediated/native-imports-vad.json) proves
module origins inside the new venv for Torch, TorchAudio, SpeechBrain,
Transformers, MLX Whisper, Parakeet and application entry points. Marian and
Whisper model classes were imported without model construction. The VAD result
binds the packaged JIT weights by SHA-256. The
[full installed audit](torch_candidate/remediated/installed-audit.json) was run
using the existing auditor against the new site-packages directory, without
installing audit tooling into the candidate. Zero findings is this audit's
observation, not a proof of an upstream fix for every historical advisory or a
clearance of unchanged older environments.

The initial resolve succeeded; its first receipt postprocessor failed on the
canonical spelling of `ruamel.yaml`. Normalization was corrected and the existing
successful resolver report was locked without re-resolving. No dependency failure
was hidden. The wheel was built before an explicit build reservation; exact
process timestamps were not instrumented. The
[build-window receipt](torch_candidate/build-window-receipt.json) retains the
conservative 13:38:38.689530–13:38:49 UTC interval. The coordinator verified the
last quality worker ended at 13:38:20.896462 UTC, so there was no overlap.
All later installation/runtime checks ran in explicitly reserved windows.

## Ready interpreter and reproduction

The usable candidate interpreter is:

```text
.cache/mac-en-es-closeout/full-application-torch-candidate-remediated/venv/bin/python
```

The original non-green venv remains in the sibling directory without the
`-remediated` suffix. Both share only the task-owned download cache; neither uses
`stt_env` site-packages. Commands run outside the checkout source with
`PYTHONPATH`/`PYTHONHOME` removed and user-site loading disabled. HF/Transformers
are offline for imports and the prepared replay commands.

Exact executed sources are retained as
[stage runner](torch_candidate/remediated/execute.py.txt),
[metadata checker](torch_candidate/remediated/verify_install.py.txt) and
[native smoke](torch_candidate/remediated/native_smoke.py.txt). They are text
snapshots so documentation checks do not treat receipt scripts as application
modules. The actual executable copies are beside the ready venv.

For a separate fresh reproduction, run the pure-source
[restoration script](torch_candidate/restore_source.py.txt) from the repository
root with a new scratch name. It restores the base commit plus frozen patch,
verifies all 163 source hashes, applies only the candidate dependency lines and
copies the remediated constraints and executable recipes:

```bash
/Users/willem/anaconda3/bin/python3.11 -S \
  docs/evaluation/mac_followup_20260910/torch_candidate/restore_source.py.txt \
  --name torch-full-reproduction-1
```

Then, **in a newly reserved build/install window**, build using the existing
build tool environment and the restored source. Record start/end timestamps and
compare the resulting wheel hash to the retained identity before reusing any
replay comparison. A changed hash must be retained as a separate artifact.

```bash
TORCH_CANDIDATE_DIR="$PWD/.cache/mac-en-es-closeout/torch-full-reproduction-1"
.cache/package-smoke/bin/python -m build --wheel --no-isolation \
  --outdir "$TORCH_CANDIDATE_DIR/dist" "$TORCH_CANDIDATE_DIR/source"
/Users/willem/anaconda3/bin/python3.11 -S "$TORCH_CANDIDATE_DIR/execute.py" metadata
/Users/willem/anaconda3/bin/python3.11 -S "$TORCH_CANDIDATE_DIR/execute.py" resolve --reserved-window
/Users/willem/anaconda3/bin/python3.11 -S "$TORCH_CANDIDATE_DIR/execute.py" install --reserved-window
/Users/willem/anaconda3/bin/python3.11 -S "$TORCH_CANDIDATE_DIR/execute.py" smoke --reserved-window
/Users/willem/anaconda3/bin/python3.11 -S "$TORCH_CANDIDATE_DIR/execute.py" audit --reserved-window
```

The actual package-install invocation is retained in the receipts and uses
`python -m pip --isolated install --only-binary=:all: --require-hashes -r
resolved-requirements.txt`. Each stage refuses to overwrite an earlier receipt.
Restoration and future reproduction commands are prepared recipes; they are not
additional completed installations.

## Completed installed EN/ES functionality

The [validated receipts](torch_candidate/normalized_r1/normalized-r1-summary.json)
record sequential installed candidate replays from scratch working directories
outside the checkout source. Both used Standard, E4B OptiQ, Torch VAD, fixed
`--gain 1`, drafting disabled and offline caches. English used Parakeet; Spanish
used Whisper Turbo. Each normalized input contains five public **development**
recordings. Original clips, confirmation data, corpora and evaluation approvals
were preserved.

| Session | Actual process interval, UTC | Finals / previews | Required writes | Source accounting |
|---|---|---:|---:|---|
| `torch213_full_normalized_r1_en` | 14:40:50.480826–14:42:11.134022 | 6 / 45 | 59 / 59 | 2,253,120 samples at 48 kHz; complete EOF |
| `torch213_full_normalized_r1_es` | 14:42:21.790803–14:43:36.881419 | 6 / 10 | 24 / 24 | 2,284,800 samples at 48 kHz; complete EOF |

Both exited 0 with completed lifecycle, empty process groups, nonempty final
source/translation text, six retained audio chunks, no failed/pending writes,
reader EOF and drained capture queues. Source accounting has no capture gaps,
unclassified intervals, unknown bounds, duplicates or dropped records. These are
machine decisions, not proof that all acoustic speech was recognized correctly.

The [CPU-only receipt checker](torch_candidate/normalized_r1/validate_normalized_replays.py.txt)
passes 19 checks per language. It verifies the actual pinned STT/Gemma snapshot
revisions, installed Torch VAD artifact, wheel/pipeline hashes and CT2 partial
backend. The legacy Marian adapters took precedence over the supplied managed
cache. Their direction, source model and declared model.bin hashes match the
recorded bytes; their upstream source revisions remain unknown. No HF partial
translator fallback occurred.

The EN runner initially used the stale `STARK_TRANSLATE__MARIAN_BACKEND` name,
which was ignored: effective settings report `auto`, while actual diagnostics
report `ct2-int8`. ES uses the corrected `STARK_TRANSLATE_MARIAN_BACKEND=ct2` and
also reports `ct2-int8`. Both exact [EN](torch_candidate/normalized_r1/en/plan.json)
and [ES](torch_candidate/normalized_r1/es/plan.json) plans and runner source hashes
are retained. An initial checker required a repo-ID string for ES STT, whose
lifecycle instead records the local snapshot path; that
[failed check](torch_candidate/normalized_r1/normalized-r1-summary-initial-checker.json)
is preserved. The final checker verifies its exact repository namespace, pinned
revision and config hash without changing runtime output.

The earlier [English](torch_candidate/remediated/replay-plan-en.json) and
[Spanish](torch_candidate/remediated/replay-plan-es.json) plans remain unexecuted.
New print-only plans are also retained, separately from each actual session plan.
The protocol file changed between EN and ES; each executed plan retains its own
protocol hash and full clip/span definition. These are separate functionality
receipts, not pooled latency cohorts. The installed source remains the older
frozen artifact identified above.

The current [replay runner](torch_candidate/normalized_r1/replay_candidate.py.txt)
defaults to printing a plan; execution requires a new coordinator reservation,
`--execute --reserved-model-window`, and a unique label. It checks frozen input,
installed source, wheel and model-manifest hashes, enforces a 600-second timeout
per language and refuses output-directory reuse. Raw WAVs remain in the scratch
session directories; their hashes and sizes are in each archived receipt.

## Pending performance and service gates

A performance/no-regression claim additionally requires alternating controls
with the **same frozen application source, models, audio, thread limits and
protocol**, and enough observations for the stated tail statistic. The working
`stt_env` has a different installed first-party package version, and subsequent
source work may differ from this frozen wheel, so it is not automatically a
matched control. Successful replay alone will not certify attended microphone,
physical output, browser ACKs, speaker embeddings or human translation quality.

## Bounded source migration recommendation

These results justify preparing a **new Mac arm64 Python 3.11 runtime** with the
tested dependency closure. They do not justify upgrading `stt_env` in place or
publishing the candidate wheel. A source-only change can enable this closure
while leaving the existing interpreter and launch configuration intact. Rebuild
the final reviewed source in a reserved window before installing or switching a
production runtime; the candidate wheel predates subsequent application fixes.

The narrow proposed `pyproject.toml` edits are:

- In `[project.optional-dependencies].mlx`, replace `torch>=2.10,<2.11` with
  `torch==2.13.0`. Preserve every MLX/STT model and package constraint.
- In `diarization`, replace the single TorchAudio requirement with the three
  requirements below. This binds the tested Mac pair even when `diarization`
  is installed without `mlx`, and preserves the existing non-Apple-Silicon
  TorchAudio range. Leave the CUDA, CPU and training requirements unchanged.

```toml
"torch==2.13.0; sys_platform == 'darwin' and platform_machine == 'arm64'",
"torchaudio==2.11.0; sys_platform == 'darwin' and platform_machine == 'arm64'",
"torchaudio>=2.10,<2.11; sys_platform != 'darwin' or platform_machine != 'arm64'",
```

The unusual version pair is evidence-based: this installed TorchAudio wheel
declares **no Torch requirement**. `pip check` alone cannot certify its ABI;
native imports and real inference provide the additional local evidence. The
actual Torch wheel is `cp311-cp311-macosx_14_0_arm64`, and TorchAudio is
`cp311-cp311-macosx_11_0_arm64`. This closure is limited to the tested Python 3.11
Apple Silicon environment on macOS 14 or later. Python 3.12, Intel Mac,
Linux/CUDA and Windows installation/runtime compatibility are not established
by these receipts; keep the project's broader Python declaration unchanged
until separately testing each advertised channel.

For reproducible Mac installation, add a clearly named new constraint file
such as `constraints/macos-arm64-py311-runtime.txt` containing the exact retained
[remediated constraints](torch_candidate/remediated/constraints.txt), and make
the Mac packaging/install command consume it. The six audit fixes must survive:
Pillow 12.3.0, pydantic-settings 2.14.2, Pygments 2.20.0, python-dotenv 1.2.2,
Requests 2.33.0 and urllib3 2.7.0. A Torch-only metadata edit does not enforce
these transitive fixes. Keep the full dependency closure, not only those six
lines. Regenerate the hash-locked install requirements for the **new** first-party
wheel, replacing its path/hash; do not label the existing candidate's wheel URLs
as a portable multi-platform lock. `models.lock.json` requires no change.

`requirements-mac.txt` is explicitly a deprecated snapshot/cache input. It
contains Torch 2.10, TorchAudio 2.10 and TorchVision 0.25, along with a different
set of versions from the candidate. The safest option is to retain it as an
historical snapshot and point new runtime instructions to the named constraints
and wheel extras. Do not combine `-r requirements-mac.txt` with the new Mac
extras, or update only its Torch line: that leaves contradictory pins. Replacing
the whole file is a larger, separate migration requiring an archival copy and
review of CI/cache consumers. TorchVision was absent from the passing full
candidate and is not a runtime dependency in `pyproject.toml`; there is no
validated replacement TorchVision pin to invent.

The exact next acceptance sequence is a fresh final-source wheel/ZIP/sdist,
outside-checkout package smoke, installed `pip check` plus selected-extra
metadata validation, full installed audit, native imports/VAD and the same
bounded EN/ES functionality check. Optional speaker inference needs its own
candidate check before claiming diarization support. Matched-source performance
and bilingual quality review remain separate gates. Switching the launcher to
the fresh runtime and retaining the previous executable path supplies rollback
without touching `stt_env`. A Mac installed audit must cover all installed
distributions; the legacy Linux CI audit strips Torch packages and cannot
substitute for it.

All retained files and hashes are enumerated in the
[evidence index](torch_candidate/evidence-index.json). Production dependency
constraints remain unchanged pending candidate review and the remaining gates.
