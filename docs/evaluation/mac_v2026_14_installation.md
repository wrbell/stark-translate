# v2026.14 local installation evidence

The local wheel, source distribution and Mac ZIP are validated. They were built
from **`977583bdb970369b0979412ccb06ef0646c9d85f`**, with project and Briefcase
version `2026.14.0.0`. Publication, a new release tag and signing remain pending.
This document and its [machine-readable report](mac_v2026_14_installation.json)
were written **after the build**; the artifacts do not contain their own final
checksums or this subsequent evidence.

## Final local artifacts

Files are in `.cache/package-final-v14-r4/` beneath the checkout.

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `stark_translate-2026.14.0.0-py3-none-any.whl` | 521,238 | `f6c02f64624cd911d79887d6004ceea0499aaffd624ce25f2b0c088a0e851f65` |
| `stark_translate-2026.14.0.0.tar.gz` | 2,399,825 | `f1589cb9ead18d17ab95f6ae56b64d67c3cc0906cb60c7458de3083170cabc87` |
| `stark-translate-v2026.14.0.0-mac.zip` | 3,306,940 | `4e3bd7fed6beb06cc09a404adc9933acb985233b8e890f3f7571b72d32bddaa3` |

The build used `.cache/package-smoke/bin/python` and installed build tools,
with package/model network access disabled. Existing `stt_env` and working
adapters were preserved. The isolated environment already had the full
`[mlx,eval,diarization]` extras installed; artifact reinstalls used `--no-deps`.

## Checks executed

- `python -m build --no-isolation` built the source distribution, then its wheel.
  Version/tag consistency, required resources and `twine check` passed.
- The Mac ZIP was unpacked without Git metadata, restoring its archived Unix
  permissions. Direct `./bootstrap.sh --help` exited zero; direct
  `./run_operator.sh` served `/healthz` with HTTP 200. The owned server was stopped
  afterward. No model session was started by this launcher check.
- A wheel built offline from that unpacked ZIP is **byte-identical** to the
  canonical source-distribution-built wheel, including executable attributes.
- Both wheels were installed in turn and tested from fresh `/private/tmp`
  directories with isolated Python imports. `/healthz`, `/operator/` and
  `/operator/review.js` returned 200; the installed manifest had 14 entries.
  The canonical final wheel was reinstalled last.
- All 18 native dependency imports passed, `pip check` was clean, and 106 source,
  wheel and installed runtime-file hashes matched. Native imports required host
  Metal access; the earlier sandbox failure is preserved separately.

The final build's commands, log hashes, exact paths and source-file comparisons
are retained in `.cache/package-final-v14-r4/report.json` and copied into the
machine-readable report. The [installation guide](../packaging/macos.md)
contains normal checkout/ZIP setup commands.

## Actual installed inference and its artifact boundary

Real inference exercised the **r3** wheel built from
`f22e6e98f201d837c672eb3542468c95f6188235`, SHA-256
`1ca39d41c9f147c4daa69f7acd70f0ebffeb44caff0983b812322fd6c6598d58`.
The operator and children ran in fresh directories outside the checkout, using
the isolated environment and managed CT2 cache from the offline setup check.

| Source | Actual path | Observed source → translation | Result |
|---|---|---|---|
| Synthetic EN | Parakeet → E4B → Spanish Piper WAV | “The grace of God is sufficient.” → “La gracia de Dios es suficiente.” | Completed, exit 0 |
| Synthetic ES | Whisper → E4B → English Piper WAV | “La gracia de Dios es suficiente.” → “God's grace is sufficient.” | Completed, exit 0 |

Both sessions produced actual Gemma generation tokens and nonempty target-voice
WAVs, used the expected pinned STT/Gemma and managed Marian artifacts, recorded
packaged Silero 6.2.1 provenance, and drained to completed lifecycle markers with
matching diagnostic hashes. No browser, microphone or physical playback was used.

After those sessions, r4 changed the shell environment selector to honor active
Conda and corrected Mac bootstrap messages. An exact wheel-member comparison
finds only `scripts/runtime_env.sh` and the generated `.dist-info/RECORD` changed.
**All 113 other members match**, including every Python module, display resource
and model manifest. The per-file hash pairs are retained in the JSON report.
This binds the earlier inference to unchanged runtime contents; it does not
claim that r4 itself ran another GPU session. The r4 launcher, install and import
checks above were executed separately.

## Preserved failures and corrections

The first ZIP rebuild exposed ignored roundtrip audio/text being bundled.
Explicit wheel/sdist/ZIP exclusions now prevent that, including without Git
metadata. A separate verifier bug mutated global wheel requirements after
checking a ZIP; an order-independence regression covers the fix. Direct-launch
permission checks also found missing executable bits on the two launcher scripts;
the scripts and archive now preserve them.

The installed-inference harness had three separate assumptions corrected:
`features` is a namespace package; a closed socket in TIME_WAIT is not a listening
server; and Whisper may report its already-resolved pinned snapshot as its
requested ID. Attempt 01 started no operator/model. Attempt 02 completed English
before the port assertion stopped Spanish startup. Attempt 03 completed Spanish
before the model-ID assertion. Original reports and recordings remain unchanged.
The final retained-evidence validator checked both completed sessions, their
model/manifest identities, WAVs and all 12 recorded artifacts per attempt without
rerunning inference. Its source reports and hashes are in the JSON report.

Final CPU validation is **1,790 passed, 4 skipped**, 59.07% coverage; see the
[validation record](mac_v2026_14_validation.json). These synthetic functional
checks do not establish natural EN/ES WER, bilingual meaning quality, visible
caption latency, diarization, acoustic playback or physical-device readiness.
Those gates and PyPI publication remain explicitly pending.
