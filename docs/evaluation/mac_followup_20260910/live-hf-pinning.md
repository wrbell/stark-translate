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

The committed startup repair in `fb44a0f` puts `dry_run_ab.load_whisper` primary
resolution inside the same `try` as warmup. An English primary download/cache
failure can therefore use the configured pinned or cached fallback. Spanish
fails its English-only fallback guard before fallback resolution or inference.
Both languages re-raise `UnpinnedModelError` for an unregistered primary instead
of silently switching the declared model. The engine wrapper retains the same
policy; model IDs and confidence thresholds are unchanged.

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

The original sixteen tiny stdlib source fixtures passed: offline behavior, exact pinned download
arguments, cache reuse, explicit local overrides, partial snapshot rejection,
SpeechBrain API/secondary-source handling, local Pyannote loading and rejection of
an unknown nested source, the real setup parser/Lite policy, rejection of malformed
setup pins without mutation, both summary loaders and lazy daemon input readiness.
Seventeen changed Python files passed AST parsing and
focused Ruff checks. All six new entries matched fetched metadata filenames and
full revisions. Existing MLX unit fixtures now mock the load-time boundary so
ordinary CI cannot accidentally download models.

These checks and the earlier 17-file AST/Ruff results retain their original
scope. After the resolved-path adapter fixture correction `fad0558`,
[Python 3.11 CI](https://github.com/wrbell/stark-translate/actions/runs/34504851183/job/102964373192)
passed 2,809 tests, with six skipped, two warnings and 66.24% coverage in 127.39 s.
Python 3.12 and lint also passed that head. This precedes the startup repair.

Five additional stdlib checks exercised the actual AST-extracted startup loader,
real resolver/language guard, a tiny fake cache and fake downloader/native APIs:
English acquisition failure used cached fallback; Spanish acquisition failure
made no fallback inference; unregistered English and Spanish primary IDs each
raised the policy error; and no native/runtime modules remained after shim
restoration. The embedded receipt preserves the original result from
`.cache/mac-en-es-closeout/startup-fallback-review/source-checks.json`; its script
and four source hashes match the committed `fb44a0f` files.

The repair adds four pytest cases in three test functions, with EN/ES
parameterization for unregistered IDs. The source-check harness did not execute
pytest. Subsequent [CI for `fb44a0f`](https://github.com/wrbell/stark-translate/actions/runs/34505780125)
passed all four new startup cases. The full Python 3.11 suite recorded 2,812
passed, one failed, six skipped and two warnings, with 66.24% coverage in 126.60 s.
Its sole failure was a stale documentation assertion hardcoding PR #192; the
coordinator reported the same failure on Python 3.12. Commit `476e349` repairs
that assertion without changing runtime files; its full CI remains pending here.
The earlier failed full suite is not relabeled as passing. Mocked tests do not
establish native model or installed-artifact behavior.

The refreshed [inventory](live-hf-source-inventory.json) binds `476e349`, whose
211 source hashes are unchanged since `fb44a0f`; the receipt rechecks its 20
current source/test hashes and five existing
local upstream API-source hashes. Metadata, configuration/access observations
and model pins are unchanged, with no new remote requests. Exact pre-refresh
inventory `10c9b338347b9118946e770582e3030a8f5f1e0014b71f028f488cffb464c2df`
and receipt `bf5fd0822d28e64a93635616d8fdd2256eb2f82ed95e426d92bc64c67a917385`
bytes remain with both Markdown files in
`.cache/mac-en-es-closeout/live-pinning-preparation/before-startup-fallback-20260910T170347Z/`.

Final installed-artifact EN/ES tests still require a reserved execution window,
explicit file input and `--no-tts`. Diarization remains off by default, and its
native compatibility, two-speaker reference/quality and latency gates remain
separate. No physical audio or human-review claims follow from this change.
