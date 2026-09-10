# Keep the prior caption when a preview translation is empty

The c13 Standard service produced one blank Spanish target among 2,562 persisted
previews. It failed the original `nonempty_previews` terminal check; the other 46
checks passed after the separately recorded monitor reader repair. Neither the
original monitor failure nor this preview failure is rewritten as a passing run.

`process_partial` waited for Marian and then published its result without checking
for an empty target. That incremented the emitted-preview counter, persisted an
empty translated field, updated the preview cache and replaced the audience's
previous translation. This was a translated-preview event, not intentional
source-only final staging.

The repair rejects an empty or whitespace-only Marian result before changing
preview state, persistence, counters or broadcast. The previous usable caption
remains until a later translated preview or final arrives. The explicit
`partial_suppressed_empty_translation` counter, trace event and operational log
retain session/utterance/request identity, sample bounds and spent STT/Marian
work. Rejection is not counted as a faster emitted caption. Model selection,
language detection, confidence thresholds and fallback behavior are unchanged.

The model-free regression uses the actual production coroutine, JSONL writer and
serialized broadcast events. A controlled Marian result sequence supplies a valid
translation, then an empty/whitespace target, then another valid translation.
Both EN→ES and ES→EN preserve revision 1 and publish revision 3; revision 2 does not
replace the caption or enter the preview JSONL. Physical STT counts still include
all three attempts. Restoring the old producer reproduces the extra invalid event.
An earlier fixture-setup error is separately retained and is not the bug proof.

The focused producer, preview and final-identity checks passed 21 tests in 0.24 s;
Ruff and formatting passed. The repaired pipeline SHA-256 is
`95e9b66bbd06ddbf913a5f3d9fccdf9617a0ff6b9c923e05ef1e2e3945ab50c1`;
the new regression file SHA-256 is
`039ffdcc3b96448b5b0164822aa7c0732865c5ef878163044c6cf0cd9c3fc98d`.
This is a post-c13 source repair, not a relabeling of the c13 measurement.
Fresh whole-source validation and installed Standard/Lite rehearsals are required
before final delivery acceptance. No microphone, output device, live audio probe,
model inference or human approval was used in these regression tests.
