# September 10 live model pinning and Torch migration feasibility

This is a separate follow-up to the [installed dependency assessment](../overnight_security/README.md)
and [B615 scope review](../mac_v2026_14_security.md). Earlier audits remain unchanged.
No working `stt_env` packages, model revisions, release artifacts or publication
settings were changed by this work. The later
[full-application candidate](../mac_followup_20260910/torch-full-application-candidate.md)
completed its installed audit and actual EN/ES replay checks; this page retains
the earlier minimal-environment scope and its original receipts.

## Live fallback model sources

Missing standard Piper voices now download the model and its configuration from
the same full commit and filenames in `models.lock.json`. Marian HF fallback
passes the registered full commit to both its tokenizer and model loader when
the shared resolver cannot find a local model. Unknown remote identities fail
before a network request with explicit pinned-setup guidance. Local Marian loads
use `local_files_only=True`.

Setup-managed models and explicit local paths remain supported. Piper also keeps
the existing custom `piper_voices/<language>/<name>.onnx` path and complete flat
user-managed cache files. These local overrides do not acquire invented revision
evidence. The legacy Piper HF cache now selects only the registered snapshot;
another cached revision is not silently accepted. An incomplete flat ONNX file
without its config no longer bypasses the pinned download.

For a different remote Piper voice, register its `voice`, `repo_id`, full
40-character `revision`, `subdir`, ONNX/config `required_files`, matching
`allow_patterns`, `weights_required: false` and `optional_group: "tts"` in the
manifest, then prepare it with `setup --include tts`. An explicitly configured
local ONNX/config pair remains available for custom voices without a remote
registration. No voice name or model commit was changed here. Lite still fails
offline on missing voices and asks for setup.

Validation: 95 focused mocked tests passed across
`test_pinned_live_model_sources.py`, `test_tts_engine.py`, `test_marian_engine.py`,
`test_managed_marian_setup.py` and `test_mac_setup_runtime.py`; Ruff check/format
and mypy passed on the edited engine files. No model loads or network downloads
occurred in these unit tests. Command details are retained in the
[validation receipt](raw/source-validation.json).

The [unskipped scoped Bandit run](raw/bandit-live-model-sources.json) still emits
four B615 findings for the two Marian and two Piper calls: its static rule cannot
follow the shared resolver and `**source_kwargs` through runtime commit
validation. The regression tests assert the actual loader arguments. No rule was
disabled, suppressed or made green by these edits; this does not close the wider
operator/training HF-download inventory or installed dependency advisories.

## Dependency feasibility

The official [TorchAudio installation guide](https://docs.pytorch.org/audio/stable/installation.html)
now documents that TorchAudio 2.11 uses PyTorch's stable ABI and supports PyTorch
2.11 and later. Therefore the earlier failure to find `torchaudio==2.13.0` does
not establish that Torch 2.13 is incompatible with available audio packages.
The [retained reproduction](raw/unavailable-numbered-pair-receipt.json) still
fails with that exact unavailable requirement; it is evidence of an overly
restrictive package selection, not the current compatibility boundary.

At this inspection, official PyPI metadata lists [Torch 2.14.0](https://pypi.org/project/torch/2.14.0/)
and [TorchAudio 2.11.0](https://pypi.org/project/torchaudio/2.11.0/) as current
releases. Torch 2.13.0 and 2.14.0 both publish macOS 14+ ARM64 CPython 3.11/3.12
wheels; TorchAudio 2.11.0 publishes matching interpreter/architecture wheels
requiring macOS 11+. The [metadata receipt](raw/metadata.json) retains source
URLs, collection times, wheel identities, hashes, versions and version-specific
advisory data, with the original small wheel metadata files beside it.

The preserved baseline and the new isolated feasibility interpreter both use
CPython 3.11.11 on native ARM64 macOS. A [fresh venv](raw/environment-creation.json)
and separate pip cache were created under `.cache/mac-en-es-closeout/`; `stt_env`
is only inventoried and used for mocked source checks.

The [actual isolated pip dry-run](raw/compatible-pair-receipt.json) at
13:14:09–13:15:07 UTC successfully resolved **Torch 2.13.0 + TorchAudio 2.11.0**
with dependencies, exit 0. The complete [pip report](raw/compatible-pair-pip-report.json)
and [log](raw/compatible-pair.log) retain wheel hashes and selected versions.
The 111.2 MB Torch wheel and smaller dependency wheels were downloaded into the
new cache during a coordinated idle window; nothing was installed. The isolated
environment still contains only pip 24.0 and setuptools 65.5.0. Its unchanged
inventory is recorded in the receipt; it is a resolver environment, not a
security-approved runtime. The working environment's 144-package inventories
[before](raw/stt-env-inventory-before.json) and [after](raw/stt-env-inventory-after.json)
also match.

## Actual isolated candidate

After the replay pilot stopped, a second new environment at
`.cache/mac-en-es-closeout/security-candidate-runtime` was created and exercised
at **13:18:25–13:19:33 UTC**, using the download cache above. This environment is
separate from both the dry-run venv and the existing Mac artifact installation.
Its [command receipts](raw/candidate-runtime-receipts.json) retain full argv,
timestamps, exit codes and log hashes.

| Check | Actual result |
|---|---|
| Fresh installation | Torch 2.13.0, TorchAudio 2.11.0, Silero 6.2.1, SpeechBrain 1.1.1, Transformers 5.12.1; exit 0 |
| Scratch installer tools | pip 26.2.1, setuptools 84.0.0 |
| Dependency consistency | `pip check`, exit 0 |
| Native offline imports | Torch, TorchAudio, SpeechBrain `EncoderClassifier`, Marian model/tokenizer; exit 0 |
| Application VAD | `tools.vad_runtime.load_packaged_vad("torch")`, three synthetic silence frames on CPU; finite probabilities within [0, 1], exit 0 |
| Full installed audit | 51 dependencies, zero known findings, zero skips; exit 0 |

The [smoke result](raw/candidate-smoke.json) binds the bundled VAD weights by
SHA-256. The [exact smoke source](raw/candidate-smoke-source.txt),
[installed inventory](raw/candidate-inventory.log), and
[full audit](raw/candidate-installed-audit.json) are retained. The audit ran from
the existing audit tool against the candidate's site-packages, without installing
audit tooling into the candidate. HF was offline during the CPU smoke; no speaker,
STT or translation model was loaded or downloaded. No CUDA or Metal inference was
run, and the work completed before the next replay slot.

## Remediation boundary

The current project `mlx` extra still requires Torch `>=2.10,<2.11`, and the
optional diarization extra requires TorchAudio `>=2.10,<2.11`. A candidate newer
pair requires an intentional dependency-line migration and full application
validation. The isolated imports/VAD smoke now passes for these packages, but it
does not exercise the complete MLX/Parakeet/Whisper dependency stack, speaker
embeddings, or real EN/ES translation. No project constraints were changed.

The old installed Torch 2.10 inventory still reports both advisories. The
[TorchScript advisory](https://github.com/advisories/GHSA-rrmf-rvhw-rf47)
records 2.13.0 as a fixed version. The earlier `pt2` advisory records no fixed
version. PyPI's version-specific metadata has no advisory entries for 2.13.0 or
2.14.0 at collection. The later full installed candidate audit also has zero
findings, but neither observation establishes an upstream `pt2` patch or clears
the unchanged Torch 2.10 environments. The next step is a separate full
application candidate install with explicitly revised experimental constraints,
then real EN/ES regression checks and a fresh installed audit. Preserve the
working benchmark environment throughout. Do not infer security clearance from
CI's dependency exclusions or promote this small CPU smoke to full service
certification.
