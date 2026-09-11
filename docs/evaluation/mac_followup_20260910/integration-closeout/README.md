# Mac EN↔ES integration closeout

[PR #196](https://github.com/wrbell/stark-translate/pull/196) merged into main at
`ea4af9a7efc87cd6fc8c86787d15c5610ab1ddd6` on **2026-09-11T01:00:13Z**, from reviewed head
`1b723bd87f4a2f4adecd081a281ded26cafbb0bf`. The four required checks passed on that exact head,
and Cursor's approval, Bugbot and security reviews also passed. The normal
squash merge used the expected-head guard; no branch-protection bypass was used.

[Issue #132](https://github.com/wrbell/stark-translate/issues/132) closed
**COMPLETED** at **2026-09-11T01:00:33Z** after merge. Its
[acceptance comment](https://github.com/wrbell/stark-translate/issues/132#issuecomment-5627758631)
uses the previously recorded independent EN/ES selected-output routing evidence.
It does not certify human audibility, virtual far-end reception, integrated
caption-triggered playback or physical unplug/replug. No new microphone or speaker
activity was used for closure or these integration checks.

Eight issues remain open with their original acceptance conditions:

| Issue | Remaining dependency |
|---|---|
| [#131](https://github.com/wrbell/stark-translate/issues/131) | Sustained EN/ES microphone rehearsal and original church CUDA-host preflight |
| [#133](https://github.com/wrbell/stark-translate/issues/133) | Natural two-speaker labels, lag/accuracy and added-final-p95 gate |
| [#135](https://github.com/wrbell/stark-translate/issues/135) | Actual WSL W16/v2-CPO artifacts and the specified Mac stock/adapted comparison |
| [#136](https://github.com/wrbell/stark-translate/issues/136) | Human terminology preferences, holdout-safe CPO and training |
| [#137](https://github.com/wrbell/stark-translate/issues/137) | Approved authentic correction through export/corpus and trainer smoke |
| [#138](https://github.com/wrbell/stark-translate/issues/138) | Deferred Hindi R&D/live-language decision and references |
| [#193](https://github.com/wrbell/stark-translate/issues/193) | Human acoustic hymn labels and capture acceptance |
| [#194](https://github.com/wrbell/stark-translate/issues/194) | Audio-derived title boundaries and bilingual meaning review |

The [43-item backlog](../../../backlog.md) preserves every original acceptance
condition: 13 items validated, six in progress, five implemented, 13 awaiting
inputs/hardware, five deferred and one experimental. These implementation states
do not imply that every roadmap gate is complete.

The [final installed-delivery packet](../final-760e948/README.md) records both
completed file-service replays, 5,533 required writes, verified raw evidence and
unchanged runtime bytes. Standard silence-final median was 1,458.7 ms; CPU Lite
was 3,297.7 ms with sparse previews. These are separate server-readiness cohorts;
the sub-second caption goal remains unmet. E4B, Spanish Whisper and 0.6-second
partial cadence remain unchanged. Standard's two known Torch findings are still
recorded; the isolated patched candidate was not promoted.

[Merge](merged-pr.json), [premerge checks](premerge-validation.json),
[issue closure](closed-issue132.json), [comment](closure-comment.json),
[remaining issues](open-issues.json) and [main identity](main-after-196.compact.json)
are actual API observations. This document records completed PR #196 events;
it does not claim its own later documentation PR has already merged. PyPI,
GHCR, release uploads and new tags remain pending by user choice. No published
tag was moved and no obsolete release asset was removed.
