# Full-service source and first-hymn sequence

**The evidence supports describing this as a full-length service recording used
for real-time replay, containing the first hymn followed by a spoken prayer.** It
was not the prepared 350-second excerpt. This is an input/workflow claim, not proof
that every input frame was captured or every lyric was transcribed correctly.
Exact singing boundaries and human lyric review are not new requirements for #134.

The local download catalog identifies `5D2rOMvkwrk` as “Gospel Message (12/14/25)”
from Stark Road Gospel Hall, duration 3,640.053 seconds, downloaded February 9.
The launcher and session metadata both bind the original full WAV, SHA-256
`8bec0f104dd883fd001f2a4b12ff9b454217c344f233458868feda07a0e83f53`, at replay speed 1.
There is no offset, duration crop or pause/language restart in the recorded launch
configuration/action ledger. The alternate 350-second excerpt was prepared but unused.

Chronological retained transcript evidence (recognition output, not an independently
transcribed reference):

| Position | Retained evidence | Source bounds as recorded |
|---|---|---|
| Chunk 2 | Announces the first hymn, number 36 | 410.464–414.592 s; span agrees with stored duration. |
| Chunks 3–7 | Hymn title/related words, then a short incomplete line | 414.592–436.832 s in successive windows. This does not distinguish reading from singing. |
| Chunk 8 | Transition before continuing | Raw 441.600–720.800 s is **invalid** for 2.91 s stored PCM. It must not be used as the hymn's exact singing interval. |
| Chunks 9–12, 14–15 | Prayer proceeds through its closing “Amen” | From 720.864 s through 770.112 s in recorded windows. These individual spans are consistent; chunk 13 has no completed transcript row. |
| Chunk 16 onward | Announces another hymn; later captions continue into the sermon | Starts 770.112 s. This gives chronology beyond the first hymn and prayer. |

The first-hymn announcement and following complete prayer sequence support a
service segment that passes through the hymn into spoken material. The missing
caption interval cannot independently identify precise singing boundaries, and
hymn wording in an STT row alone is not an acoustic classification. No `[MUSIC]`
hold event was recorded.

The pipeline ran from 06:49:41.024246 to 07:51:26.876949 UTC (longer than the source),
with input/status activity near the expected file end and a later explicit Stop.
The installed reader starts at offset zero, iterates the entire resampled source
and appends two seconds of virtual silence; the launch did not set early-exit or
crop arguments. **No separate persisted EOF event, final reader cursor, or total
source-sample counter was found.** Completed lifecycle/exit 0 proves clean shutdown,
not a byte-complete EOF audit. The terminal health input age is compatible with
input ending roughly 53 seconds before Stop, but it is not an exact EOF timestamp.

Recommended public wording:

> The full 60-minute-40-second service recording, including the first hymn and
> subsequent spoken prayer, was used for real-time replay. Retained captions run
> from the hymn introduction through the prayer and later sermon. The recording
> was not cropped. The original run completed, but its source-span/capture defects
> prevent a complete coverage pass; the fresh full-source replay is reported
> separately.

This supports the chosen full-hymn-plus-spoken workload without requiring a new
lyric-quality or exact-boundary gate. Whether #134's written rehearsal is accepted
is separate from the failed endurance/source-provenance check and the remaining
UX issue mapping. Fresh Standard/Lite results remain pending.
