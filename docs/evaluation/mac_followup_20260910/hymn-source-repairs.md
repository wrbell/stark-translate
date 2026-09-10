# Hymn-related source repairs — September 10

These repairs address concrete source defects found while reviewing issues
[#193](https://github.com/wrbell/stark-translate/issues/193) and
[#194](https://github.com/wrbell/stark-translate/issues/194). Both issues remain
open. Natural transition labels and independent bilingual meaning review are
still required; the repairs do not establish classifier or translation quality.

## Speech onset after music hold

Commit `26f854c` retains the original stamped audio while the existing music-hold
recovery decision accumulates 15 speech frames. Previously the first 14 frames
of accepted speech were discarded: 448 ms at 512 processed samples / 16 kHz.
The accepted onset now joins the utterance with its original capture positions
and timestamps. The 15-frame threshold and 0.7-second minimum final are unchanged;
short bursts that never reach the decision remain suppressed by the old policy.

Tentative frames have an explicit pending disposition. They are recovered after
acceptance or suppressed on silence, pause, EOF, stop, capture failure or a source
gap. The source ledger requires a final, filter or discard disposition before
counting recovered audio as complete; staging it does not count as delivery.
Hold logs use a capture-frame counter that survives pause. Captured-frame duration
excludes pauses; a separate monotonic trace duration includes them.

## Existing text boundaries

Commit `7c06b27` preserves whitespace while applying existing context-aware STT
corrections. For example, supplied `Eternity\nTime will soon end.` retains its
line break. Previously `split()` followed by a space join erased it even when no
word was corrected. Case, punctuation, phrase and theological-context rules are
preserved. This cannot reconstruct a boundary absent from STT or undo earlier
CT2 segment flattening, and it is not a proven correction of the retained natural
hymn translation.

## Validation scope

Regression coverage uses the real capture loop, correction function, source ledger
and framed input reader with controlled inference/device boundaries. It checks
exact recovered PCM, 48 kHz source / 16 kHz processed timing, short-burst policy,
pause/EOF/stop/failure/gaps, bounded staging, and existing text delimiters.
The first CI run found a context-manager fixture signature mismatch, repaired in
`ffa34c5`. [CI on that revision](https://github.com/wrbell/stark-translate/actions/runs/34500508591)
passed the Python 3.11 and 3.12 suites; lint also passed. These are controlled
regression checks, not native capture or meaning-quality certification. No new
microphone or playback test accompanied these changes.

The c13 [350-second natural file control](final-c13f51f/hymn-capture.md) and
[102-call E4B/E2B boundary packet](final-c13f51f/hymn-boundary.md) have executed.
The control never entered music hold, so it cannot validate the recovery threshold
on a natural transition. The text packet passed supplied raw/newline/quoted
hypotheses through the actual correction stage; those delimiters are not approved
audio boundaries. Natural transition annotations and independent bilingual meaning
review remain required under #193/#194. No prompt, model default, short-word
blacklist or human approval was added. See the separate [current source checks](final-c13f51f/source-validation.md).
