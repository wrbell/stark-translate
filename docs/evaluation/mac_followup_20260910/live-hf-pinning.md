# Mac live model download pinning — 2026-09-10

The live Mac loading boundaries now resolve complete local models or download a
registered full-commit HF snapshot before calling MLX wrappers. Setup and preflight
still use the pure, offline `resolve_model_path`; importing `engines.model_paths`
does not import a downloader. An unknown uncached remote ID fails with instructions
to register an immutable source or configure a local model. Existing complete
managed/HF caches and explicit local paths retain their prior lookup behavior.

This is source hardening with mocked loading validation. It does not certify
native models, diarization quality, microphone/speaker functionality, or clear
every B615 finding. The [repo-wide inventory](live-hf-source-inventory.md) separates
other live and manual training, export, evaluation and operator paths. The earlier
[security assessment](../mac_v2026_14_security.md) remains a historical snapshot.

## Changed boundaries

| Boundary | Behavior |
| --- | --- |
| `MLXWhisperEngine` primary, startup fallback and quality fallback | Complete local path passed to `mlx_whisper.transcribe`; unregistered primary fails before warmup rather than silently selecting another model. Existing language and confidence thresholds remain unchanged. |
| `dry_run_ab.load_whisper("mlx")`, `workers.stt_worker_main` | Same load-time resolver; CPU/CUDA branch is unchanged. |
| `ParakeetMLXEngine.load` | Local path passed to `from_pretrained`, whose installed API has no revision parameter. |
| `MLXGemmaEngine.load`, optional draft model | Complete local path passed to `mlx_lm.load`; existing family/EOS/warmup behavior remains unchanged. Uncached custom/draft models require a manifest pin or local path. Live MTS is still rejected by its existing guard. |
| Operator session summaries and optional TranslateGemma summary translation | Same pinned load-time resolver, retaining explicit local paths, family/config detection, EOS policy and existing load-thread warmup. |
| Optional SpeechBrain ECAPA | Pinned snapshot plus local `pretrained_path` YAML override prevents secondary CKPT retrieval from a bare repo. SpeechBrain 1.0 uses its explicit `revision`; 1.1 uses `FetchConfig(revision=…, allow_network=False)`. Both retain CPU loading and graceful failure. |
| Optional Pyannote embedding | `Inference` receives the pinned local `pytorch_model.bin`, selecting the wrapper's local-file branch. |
| Optional live full Pyannote pipeline | Verified pinned pipeline config, separately pinned segmentation and embedding checkpoints, local `Model` objects, then original pipeline/instantiation parameters. Unknown nested sources or unavailable gated files disable the optional loader gracefully. The standalone offline `features.diarize` path is unchanged. |

The full Pyannote daemon defers model loading until a real file has a valid WAV
header and at least one frame. Missing paths, directories and empty/partial files
do not trigger downloads or native imports. An unavailable optional pipeline is
attempted once per daemon run, then stays idle until restarted.

Setup now rejects any selected HF snapshot without a full lowercase 40-character
commit before creating cache directories, looking up cached entries, processing
earlier downloads or changing installation markers. Its direct snapshot wrapper
validates again before importing the downloader or creating a target directory.
Runtime local overrides retain their contract; setup requires a valid manifest
even when files from an old moving-ref entry already exist, and leaves those
files untouched on rejection. The current manifest already meets this rule.

The default English Whisper fallback retains the actual working cache revision
`31357b406d55cf380abe4101b6a10207b156c4a3` for
`wbell7/distil-whisper-large-v3.5-mlx`. Its model ID, English-only policy, and
thresholds were not changed. Six new optional manifest entries all have empty
`required_for`, so ordinary Mac setup does not start downloading optional models.
No package versions or working environments changed.

The actual setup parser accepts the optional manifest groups. These are future
operator commands, not commands executed during this source work:

```sh
stark-translate setup --backend mlx --profile standard --include whisper-fallback
stark-translate setup --backend mlx --profile standard --include diarization
stark-translate setup --backend mlx --profile standard --include diarization-pyannote
```

Each command includes the usual Standard Mac models plus the requested optional
group. The Pyannote group needs authorized access to gated repositories; the
segmentation access failure below remains unresolved. Optional native packages
are still required for inference. Lite profiles retain their existing
`--include tts`-only policy and reject these three optional groups.

## Source verification and access limits

[Machine-readable receipts](live-hf-pinning-receipts.json) include full revisions,
metadata URLs and response hashes, configuration hashes/access outcomes, API source
hashes, and observed implementation hashes. Public metadata and small YAML
configuration files were read; model weights were neither downloaded nor loaded.
The existing credential was passed through the HF client only for gated config
reads; credential values are absent from receipts.

The Pyannote 3.1 configuration at
`84fd25912480287da0247647c3d2b4853cb3ee5d` names:

- Segmentation: `pyannote/segmentation-3.0` at
  `e66f3d3b9eb0873085418a7b813d3b369bf160bb`.
- Embedding: `pyannote/wespeaker-voxceleb-resnet34-LM` at
  `837717ddb9ff5507820346191109dc79c958d614`.

The existing credential could read the pipeline and embedding configurations, but
the pinned segmentation configuration returned **HTTP 403 / GatedRepoError**.
Consequently full Pyannote remains unavailable here without existing authorized
access or a complete local copy. No login or terms acceptance was performed.
The loader catches this failure and returns `None`; it never retries `main`.
API compatibility was reviewed against Pyannote 3.3.2 source and SpeechBrain 1.0.3
source / installed 1.1.1 source. Native compatibility remains unexecuted, and
immutable source selection does not replace trust in checkpoint/YAML contents.

## Validation and remaining execution

Sixteen tiny stdlib source fixtures passed: offline behavior, exact pinned download
arguments, cache reuse, explicit local overrides, partial snapshot rejection,
SpeechBrain API/secondary-source handling, local Pyannote loading and rejection of
an unknown nested source, the real setup parser/Lite policy, rejection of malformed
setup pins without mutation, both summary loaders and lazy daemon input readiness.
Seventeen changed Python files passed AST parsing and
focused Ruff checks. All six new entries matched fetched metadata filenames and
full revisions. Existing MLX unit fixtures now mock the load-time boundary so
ordinary CI cannot accidentally download models.

Focused pytest tests are prepared in `tests/test_pinned_mlx_model_loading.py` and
the existing MLX/fallback suites; they exercise the actual resolver and YAML
parser with mocked download and native APIs. They were not executed during the
active model measurement queue. The coordinator can run these with the normal
test environment and `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 STARK_RUN_VAD_TESTS=0`, then
run the repository's full CI checks. This receipt must not be read as those tests
already passing.

Final installed-artifact EN/ES tests still require a reserved execution window,
explicit file input and `--no-tts`. Diarization remains off by default, and its
native compatibility, two-speaker reference/quality and latency gates remain
separate. No physical audio or human-review claims follow from this change.
