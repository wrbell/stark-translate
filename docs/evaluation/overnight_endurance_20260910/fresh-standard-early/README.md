# Fresh Standard: first captions and hymn-to-prayer observations

**Provisional snapshot, 08:48:26 UTC, September 10. The hour was still running.**
The retained prefix has 33 matched final CSV/diagnostic rows and 156 preview records,
all consistent with their recorded source spans under the existing reader's 20 ms
rounding allowance. The last completed caption window ends at source 820.384 s.
This is later than the earlier 17-final/76-preview audit; it does not replace that
snapshot or certify the rest of the recording.

Session `20260910_043120_839144_en` uses installed source `752ab9a`, wheel
`7477574d…`, pipeline hash `48de5255…`, and the same uncropped 3,640.053-second
English service source (`8bec0f10…`) at speed 1. [Exact snapshot](snapshot.json)
contains the complete hashes, parser identities and captured prefix lengths/hashes.
The original old-hour failure remains separate.

## Start and first-caption timing

| Observation | Value |
|---|---:|
| Actual SPA Start button | 08:31:20.797–08:31:21.067 UTC |
| Lifecycle start | 08:31:24.962266 UTC |
| First translated preview ready on server | 08:38:21.558818 UTC |
| Start → first server preview, wall time | **420.492–420.762 s** |
| First final ready on server | 08:38:30.526435 UTC |
| Start → first server final, wall time | **429.459–429.729 s** |
| File's exact digital-zero lead-in | 403.1448125 s; first nonzero sample is not speech onset |
| First preview source window | 405.056–405.664 s |
| Captured start → first translated preview | **1,095.6 ms server; 1,140.5 ms browser ACK upper bound** |
| First smart-cut final, estimated speech end → ready/ACK | **4,655.9 ms server; 4,668.4 ms browser ACK upper bound** |

Raw caption wall timestamps use America/Detroit and are converted to UTC. No
session-relative trace origin is joined to the Start clock. Leading zeros, time
spent launching/loading and caption latency stay separate; subtracting the zeros
from wall wait would not measure model-loading time. This single first-caption
observation is not a median or an improvement claim against the old runtime.

One eligible audience connection acknowledged all 33 finals and 156 translated
preview events in this prefix. These are session-local, DOM-visible render ACKs,
including the return path. Native Mac lock/physical display visibility remains a
separate caveat. Prefix coverage does not pass the complete-hour browser gate.

## Hymn, short fragments and following prayer

The actual operator page showed hymn-related captions at **08:39:15.717 UTC** and
the following prayer at **08:44:04.834 UTC**. In the retained records, chunk 2
announces the first hymn; chunks 3–7 contain hymn-related words. The subsequent
prayer appears in chunks 18–21 and 23–24, with individually consistent windows
from 720.864 through **770.112 s**. Chunk 24 contains the closing “Amen”; chunk 25
announces the next hymn. The later stored closing is distinct from what the earlier
SPA snapshot already showed.

The uncropped input and this chronology support the full-hymn-plus-spoken
workload. They do not identify exact singing boundaries, prove every source frame
was captured, or certify lyric/transcript accuracy. Exact lyric review is not an
extra #134 requirement. The session remained uninterrupted; no Pause was inserted
to suppress negative evidence in this replay.

Short or incomplete finals occurred during the hymn context, including:

| Final chunk / utterance | Source interval | Recorded English → generated Spanish |
|---|---|---|
| 7 / 7 | 434.528–436.832 s | “It dies a” → “Muere a” |
| 11 / 12 | 445.600–446.592 s | “Uh” → “Uh” |
| 14 / 16 | 454.528–456.384 s | “Changing uh” → “Cambiando eh” |
| 15 / 17 | 456.416–457.856 s | “Okay.” → “Bien.” |

These are actual displayed/persisted fragments, not approved transcripts. Without
new listening, this note does not declare each fragment a hallucination or label
every caption gap as singing/silence. The full bounded list is in the JSON.

A separate final translation error is visible in chunk 3: recorded **“Time will
soon end”** becomes **“La eternidad pronto terminará”** (“eternity will soon end”).
The subject changes from time to eternity, relative to the recorded English. This
finding does not claim the English perfectly represents the audio. The automatic
QE score was 1.0; that score did not catch this meaning change and is not human
approval. Both the short-caption behavior and lyric-meaning problem need their
own follow-up; neither is hidden by consistent timing spans.

For an attended live service, the existing Pause/Resume controls remain the
practical operator response during congregational singing. File replay Pause
preserves its cursor, so it would not skip the hymn or demonstrate automatic music
suppression. No such action was taken in this uninterrupted endurance run.

## Evidence boundary

The raw evidence is retained locally as captured, newline-complete byte prefixes;
each has its byte length and SHA-256. Sources can grow after capture. Files were
read sequentially, not as a transaction, and these hashes do not bind the eventual
whole session files. CSV/diagnostic IDs and source identity agree in this prefix;
any later result needs its own terminal audit.

There were no model calls, playback, runtime changes, approvals or tests in this
read-only report preparation. At this snapshot, required-write completion, EOF/whole-source capture,
cleanup, final memory, the remaining hour, CPU Lite and long-summary UI were
pending. Public text excludes full corpus excerpts and private correction notes.


## Later operator observation — 08:51 UTC

A draft entered at 08:46:26.943 UTC remained intact while new records arrived.
The displayed list intentionally stayed at **33** during editing. After actual
**Save and next** at 08:51:00.704 UTC, it refreshed to **40** and selected chunk 2.
This does not mean the displayed count grew while editing. No operator reload was
attempted. Approval boxes were unchecked; saved-sidecar revision verification is
pending at this observation, and the private note is omitted.

The actual Live reference list at 08:51:18.761 UTC showed **Luke 23:32** and
**Luke 23**, with the earlier bad **Luke 20:3** absent from that list. This is a UI
observation, not a spoken-reference accuracy certificate. The
[later UI ledger](ui-update-0851.json) keeps these observations separate from the
08:48:26 telemetry prefix. The hour remained in progress at that observation.


## Later terminal verification

The [current main report](../README.md) records the completed Standard hour and
long-summary UI, separately from this unchanged early telemetry prefix. At
09:04:42 UTC, the [review audit](../fresh-standard-terminal/review-audit.json)
verified chunk1 revision1, both approvals false, and unchanged original CSV,
diagnostic, partial and display-metric prefixes captured before Save. These are
bounded prefix comparisons, not a before/after hash of the eventual whole files.
No private correction note or successful operator reload is claimed.
