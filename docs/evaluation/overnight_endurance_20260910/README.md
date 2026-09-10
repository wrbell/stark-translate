# Installed service rehearsal and endurance — September 10, 2026

**The repaired Standard hour completed with consistent bounds for all 563 final
captions and 2,814 previews, all 7,594 required writes finished, and process cleanup
verified. Its long-summary UI also finished and disclosed that the middle of the
transcript was omitted. CPU Lite also completed with consistent retained spans and
durable writes/cleanup, but sparse previews and large tails do not support
recommending it as a fast production profile today.**
The original Standard hour's source-span failure is preserved separately. No
whole-source speech-coverage, human-quality, issue-closure or release-publication
claim follows from these results.

This is a MacBook M3 Pro laptop stand-in for the
[#134 runbook rehearsal](https://github.com/wrbell/stark-translate/issues/134).
A laptop is explicitly allowed by that issue. All three runs selected the
uncropped natural English service at real time: 3,640.053 seconds, SHA-256
`8bec0f104dd883fd001f2a4b12ff9b454217c344f233458868feda07a0e83f53`.
The prepared 350-second transition slice was **not run**.

| Cohort | Installed identity | Observed result |
|---|---|---|
| Original Standard, `20260910_024936_300302_en` | Source `b65e6e0`; wheel `ea489f72…` | Completed and cleaned up, but 23/549 final spans and 97/2,720 preview spans were inconsistent. |
| Repaired Standard, `20260910_043120_839144_en` | Source `752ab9a`; wheel `7477574d…`; Parakeet EN, MLX E4B with existing Marian routing | Completed; 563/563 final spans, 2,814/2,814 preview spans and all 563 saved WAV-header durations agree. Three selected waveform windows also match exactly; this is not all-source coverage. |
| CPU Lite, `20260910_053518_894101_en` | Same `752ab9a` wheel; CPU Whisper small INT8, Marian finals, ONNX VAD | Completed at 10:36:47.110148 UTC; 468 final spans/WAV headers and 271 preview spans consistent, all 1,979 writes completed and cleanup verified. Only 174/468 final utterances had first translated previews. |

[Machine-readable results](report.json) retain exact identities and measurement
scopes. The [retained archive manifest](raw/artifact-manifest.json) and
[standalone verification receipt](archive-verification.json) bind **94 source
members plus the inventory, all 95 verified**, in nine shards of at most 480 KiB
(3,999,389 compressed bytes). The [standalone verifier](raw/verify_endurance.py)
checks shard, stream and member hashes without modifying the evidence. Manifest
SHA-256: `68fee0fc07b6a7d405e90fffea70131ad754406b096a42f287f443539a653a60`. Raw transcripts,
operational logs and local provenance paths are retained; audio, model files and
private correction notes are excluded. The
[earlier operator rehearsal](../overnight_operator_rehearsal.md), its
[machine evidence](../overnight_operator_rehearsal.json), and the
[96-run latency screen](../overnight_screen_20260910/README.md) are separate
workloads. Neither the old failed hour nor the screen is pooled with this hour.

## Setup and first captions

A setup helper prepared installed packages, offline model caches, loopback operator
port 9014, real-time file input and process monitoring. The hours used the
**installed package from a dedicated data directory**: no local project Python
modules, with `PYTHONPATH` cleared and installed wheel/pipeline hashes bound to a
receipt. That directory is physically within checkout ancestry. Separate mechanical
installation checks really did run outside the checkout under `/private/tmp`.
Initial installation/download time was not measured. Device enumeration is not a
microphone capture test.

| Repaired Standard observation | UTC / elapsed time |
|---|---|
| Installed operator service launched | 08:31:04.768730 |
| Actual SPA Start captions button | 08:31:20.797–08:31:21.067 |
| Pipeline lifecycle began | 08:31:24.962266 |
| First translated preview ready on server | 08:38:21.558818 |
| Prepared installed operator launch → first server preview | **436.790088 s**; operational setup boundary, excludes unmeasured one-time install/download |
| Start → first server preview, wall time | **420.492–420.762 s** |
| First final ready on server | 08:38:30.526435 |
| Start → first server final, wall time | **429.459–429.729 s** |
| First preview, captured-start → readiness | **1,095.6 ms server; 1,140.5 ms browser ACK upper bound** |
| First smart-cut final, estimated speech end → readiness | **4,655.9 ms server; 4,668.4 ms browser ACK upper bound** |

The file begins with **403.1448125 seconds of exact digital zeros**; the first
caption window starts at source 405.056 seconds. First nonzero PCM is not necessarily
speech onset. The seven-minute wall wait includes startup and source lead-in; it
is not caption latency. Subtracting the zeros would not isolate model-loading time.
The [early snapshot](fresh-standard-early/README.md) preserves these observations
and their original bounded-prefix hashes.

Lite’s [first-caption record](lite-early/first-caption.json), captured earlier while its
hour was running, reports Start→first translated preview **418.298–418.556 s** and
Start→first persisted final **427.182–427.440 s**. Its first preview belongs to
utterance 1 (6,090.4 ms captured-start→server; 6,104.5 ms ACK upper bound), while
its first final is chunk 2/utterance 2 (5,438.3 ms estimated-speech-end→server;
5,447.7 ms ACK upper bound). These are different utterances and independent first
events. The same digital-zero lead-in applies; no model-loading time or sustained
profile advantage is inferred.

For the original hour, Start was 06:49:36.244–36.526 UTC; first server preview/final
were 06:56:39.121432/06:56:46.859679 UTC. Start-to-preview was 422.595–422.877 s
and Start-to-final 430.334–430.616 s. Its first captured-start preview was 2,336.5 ms
and first smart-cut final 4,666.8 ms. These are distinct observations, **not a causal
before/after improvement claim**; the original source-span failure remains.

## What needed a helper, and what used the operator page

| Step | Actual observation | Assistance boundary |
|---|---|---|
| Install, prepare caches and choose profile | Installed environments, pinned artifacts and offline preflight | Setup helper; not a fresh-install speed test. |
| Select the full-service file and collect evidence | Controlled file environment, source hash, launcher and monitor | Setup helper; ordinary volunteers do not configure replay/audit tooling. |
| Start and inspect Live/audience | Actual SPA Start and visible caption text in browser state | Operator page once prepared; physical projector/phone visibility not certified. |
| Pause, Resume, Stop and change speaker language | Earlier three-session rehearsal exercised controls; switching language created a new session | Operator page; its Spanish material was synthetic Piper audio. Fresh Standard itself ran without a Pause through the hymn. |
| Review while finalized records arrived | Fresh Standard draft stayed intact with displayed list fixed at 33; after Save and next the list refreshed to 40 and selected chunk 2 | Operator page. Sidecar revision 1 and both approvals false verified; no operator reload attempted. |
| Create the long summary | Actual Create click, Finished state, English/Spanish output and omitted-middle warning | Operator page after Stop; no helper API substitute and no human fidelity approval. |
| Download support | Earlier rehearsal downloaded a metadata-only ZIP | Operator page; no support bundle was sent. |
| Verify spans, writes and process cleanup | Independent terminal audits and owned-process monitoring | Setup helper; a green page alone cannot establish these facts. |

The [review audit](fresh-standard-terminal/review-audit.json) verifies the separate
chunk 1 correction revision at 08:51:00.437741 UTC. Original CSV, diagnostics,
partial and render-metric byte prefixes captured before Save were unchanged at
09:04:42 UTC. Files grew meanwhile: this is a **bounded-prefix** comparison, not a
whole-file before/after hash claim. Neither transcript nor translation was approved;
no private draft note is included. The old Standard review used a fresh same-origin
page after a canceled reload, which is also not a successful-reload claim.

These are exercised steps, not an unassisted novice-user trial. See the
[operator runbook](../../operator_runbook.md) for ordinary workflow and escalation.

## Full hymn, spoken material and visible problems

The full recorded service containing the first hymn and subsequent prayer was
selected without cropping. In repaired Standard, chunk 2 announces the hymn;
chunks 3–7 contain hymn-related words. Actual SPA snapshots showed hymn-context
fragments at 08:39:15.717 UTC and the following prayer at 08:44:04.834 UTC. Retained
prayer rows 18–21 and 23–24 continue through closing “Amen”, with individually
consistent windows from 720.864 to 770.112 source seconds. These observations
support the full-hymn-plus-spoken **input workload**. Exact singing boundaries and
lyric review are not additional #134 requirements; neither all-source capture nor
transcript accuracy is inferred from this chronology.

The [early evidence](fresh-standard-early/README.md) retains short hymn-context
finals such as “It dies a” → “Muere a” and “Changing uh” → “Cambiando eh”. Without
new listening, not every fragment is labeled a hallucination. A separate recorded
translation changes **“Time will soon end”** into **“La eternidad pronto terminará”**
(“eternity will soon end”); the meaning changes relative to the recognized English,
while audio transcription accuracy remains unreviewed. Automatic QE 1.0 did not
catch that meaning change. Consistent timing does not erase these quality problems.

Lite’s [early actual output](lite-early/quality-observations.json), observed at
09:42:46 UTC while its hour was running, retains the time/eternity boundary:
“Eternity, Time will soon will end.” → “Eternidad, El tiempo pronto terminará.”
The next Lite final omits the remaining recognized clause after “¿Dónde pasarás la
eternidad? Es un día inmutable.” STT wording, segmentation, cadence and translator
all differ between profiles. These are unreviewed output differences, not a
controlled translator comparison or a Lite quality win.

No `[MUSIC]` detector event was recorded. For an attended service, Pause during
congregational singing and Resume before spoken prayer remains an operator option;
no such Pause was inserted into this replay. File replay Pause preserves the cursor
and would not demonstrate automatic singing suppression.

| Observed or reproduced issue | Fix or remaining evidence |
|---|---|
| Original source spans disagreed with stored audio | Original 23/549 failures retained; a missing 32 ms frame was independently found. Repaired 563/563 span/header checks pass; three selected Standard and three Lite waveform windows match exactly. The old chunk 141 deficit was reproduced separately. |
| Discarded short utterances could strand previews | Session/utterance-specific removal has production-handler tests. Repaired runtime recorded 32 discarded utterances and 8 suppressed late preview tasks; no complete human inspection of every removal is claimed. |
| Old impossible/misparsed Bible references | Conservative English number/context parsing and structural KJV bounds were added. Actual fresh UI showed Luke 23:32 / Luke 23 and, after Stop, Isaiah 53:11 instead of old invalid references. Valid bounds do not prove spoken-reference accuracy. |
| Audio-feed wording and technical details confused status | Current UI distinguishes feed updates from heard sound and discloses technical details separately. |
| Support preview implied an empty ZIP; log list crowded Sessions | Current preview separates metadata from optional attachment bytes; detailed log list is collapsed with selections retained. |
| Hymn fragments and lyric meaning error | Negative evidence retained in [#193 — hymn fragments](https://github.com/wrbell/stark-translate/issues/193) and [#194 — title boundaries and meaning](https://github.com/wrbell/stark-translate/issues/194); these are remaining hymn/quality follow-ups, separate from the fixed operator UX blockers. |

**Follow-up mapping is recorded:** the operator UX blockers above were fixed;
remaining hymn/quality work is tracked in #193 and #194. This report does not
claim #134 is closed.

## Completion, audience acknowledgments and resources

Repaired Standard completed at **09:32:38.589810 UTC, exit 0**. All **7,594 writes
completed**, with zero pending/failed; final diagnostics hash and CSV joins agree.
The monitor finished at 09:32:42.555986 and verified pipeline/descendant cleanup.
The actual SPA showed Ready at 09:32:51.219. Audio, capture-handoff and final queues
were zero; stream_tokens=1 is the installed shutdown `None` sentinel, checked
separately from required-write completion.

One matched document-visible audience connection acknowledged **563/563 finals and
2,813 nonempty translated previews**. One raw source-only preview with empty
translation is excluded from translated-preview timing. First translated previews
covered 538/563 final utterances. No matched preview ACK followed its final ACK;
64 preview events lacked assessable final ordering. ACKs include the return path
and do not prove that the native locked Mac screen was physically visible.

All emitted finals have zero virtual padding. Last caption window ended at source
3479.360 s, leaving **160.693 s unclassified**. No explicit exact reader EOF/cursor
record is retained. Input age, elapsed playback and clean Stop are not an exact
all-frame or all-speech coverage audit. The union of final caption windows is
2181.600 s, not a measure of all captured audio or semantic coverage.

Observational latency for this one Standard cohort follows. Values are p50/p95,
nearest-rank p95, milliseconds; schema 2, real-time replay, 48 kHz coordinates.
Analytical endpoint classification v1 keeps replay-tail assistance separate;
none occurred among this run's finals. These are **not quiet paired benchmarks**.

| Endpoint | n | Server final readiness | Browser final ACK upper bound | Endpoint decision | STT call | Translation call |
|---|---:|---:|---:|---:|---:|---:|
| In-file silence |401|1429.7 /3019.0|1442.2 /3033.9|491.1 /575.0|414.3 /1218.6|381.3 /1289.1|
| Smart cut |124|3504.9 /6451.3|3519.6 /6465.5|1259.0 /5031.2|218.5 /453.0|1578.5 /2739.6|
| Hard cut |38|2333.1 /3220.2|2346.1 /3231.5|8.9 /12.5|632.0 /1406.4|1654.0 /2074.7|

Stage medians must not be added to claim a median total. First server-preview
p50/p95 was 865.8 / 4113.9 ms (n=538); browser upper bounds 883.8 / 4125.4 ms. Update-gap
p50/p95 was 622.1 / 1221.2 ms (n=2246). Fast first previews sometimes occur, but these
observations do not satisfy a general subsecond final-caption target or justify a
default change.

Pipeline lifetime peaks were RSS 2,588,459,008 B and Metal 9,372,995,258 B. Maximum
sampled simultaneous process-tree RSS was 575,602,688 B across 729 observations.
These scopes must not be added or called a true combined lifetime peak; late
attachment and between-sample/short-lived allocations can be missed. This Mac does
not certify lower-RAM x86 or RTX 2070 hardware.

The original hour separately completed 7,372 writes and cleanup despite failed
source spans. Its [source-interval note](source-interval-note.md) and raw failure
remain unchanged. Repaired Standard had concurrent model-free helper preparation,
metadata/UI reads, light static checks and two synthetic tests lasting 1.483 / 1.471 s
with unrecorded exact UTC intervals. No other model inference ran during that hour.
The later header-only audit occurred during Lite; these conditions are retained in
the terminal assessment. Timing is descriptive rather than a causal comparison.

## Completed CPU Lite hour

Lite completed at **10:36:47.110148 UTC, exit 0**, with **1,979/1,979 required writes**
finished and no pending/failed writes. All **468 final spans/WAV headers and 271
preview spans** agree. Monitoring verified cleanup at 10:36:52.628263; the actual
SPA showed Ready at 10:37:09.160. The operator was terminated later, separately from
pipeline completion. Its CPU profile used no Gemma model.

One matched document-visible connection ACKed 468/468 finals and 271 translated
previews. Only **174/468 finalized utterances (37.18%)** had a first translated
preview; 294 had none. No matched preview ACK followed its final ACK; 7 events had
unassessable ordering. The first-any preview belonged to utterance 1 with no
persisted final; its direct 6,104.5 ms ACK is separate from the 174 final-linked
first-preview distribution. Missing final text for an utterance is not assigned
a reason without its recorded runtime event.

This CPU profile was observationally slower than Standard on this Mac. The runs
are separate functional rehearsals, not a controlled paired comparison, and have
different STT/translation outputs. **Lite cannot be recommended as a fast
production profile today.** Its own endpoint observations follow, p50/p95 in ms:

| Lite endpoint | n | Server final readiness | Browser final ACK upper bound | Endpoint decision | STT call | Translation call |
|---|---:|---:|---:|---:|---:|---:|
| In-file silence | 311 | 2854.6 / 11246.8 | 2865.4 / 11264.0 | 490.9 / 491.4 | 1294.1 / 1462.1 | 58.8 / 136.3 |
| Smart cut | 121 | 4328.3 / 8779.9 | 4343.4 / 8797.8 | 1195.8 / 4780.1 | 1406.9 / 1537.5 | 117.4 / 185.1 |
| Hard cut | 36 | 2752.2 / 10216.2 | 2764.8 / 10226.2 | 9.7 / 11.4 | 1489.9 / 1597.6 | 155.0 / 210.1 |

Final-linked first-preview server p50/p95 was **5503.2 / 7703.0 ms**
(n=174); browser upper bounds were 5512.1 / 7711.3 ms.
Within-utterance update gaps were 1972.9 / 2309.5 ms (n=93). Stage
medians must not be added to claim a median total. There were 599 final attempts:
**468 completed, 21 empty STT results and 110 hallucination-filter rejections**.
Those filter labels are runtime classifications, not human ground truth or a
measure of how much valid speech was lost. Natural transcript/translation quality
remains unreviewed; hymn fragments and omitted clauses remain visible follow-ups.

Last final window ended at 3477.504 source seconds, leaving **162.549 s unclassified**.
Late actual UI contains a closing-hymn announcement; the remainder is not described
as quiet. No explicit exact reader EOF/cursor is retained, all final padding is 0,
and the 2063.680 s union of final windows is not all captured audio or speech coverage.

Pipeline lifetime RSS peaked at 3,158,310,912 B; sampled simultaneous tree RSS at
2,027,012,096 B over 729 samples. Metal peak is unavailable on this CPU run. These
scopes must remain separate and do not certify an 8 GiB x86 machine. Light concurrent
UI/metadata/documentation work and the earlier Standard header check are disclosed.

The configuration already uses **beam size 1, three STT threads/one worker, one
Marian thread and 0.6 s partial cadence**. A next bounded CPU experiment should test
partial admission/cadence with preview-coverage, final-tail and meaning/omission
guards. Observed STT admission and roughly 1.3–1.5 s STT calls dominate; Marian silence
p50 is 58.8 ms. Optional E2B quality mode would not address that STT queue cost.
No new runs or defaults are proposed as completed here.

After both hours, [selected waveform checks](waveform-evidence.json) found exact
sequences for Standard chunks 18/19/157 and Lite 20/21/157. Old Standard chunks 9/10
also matched; old chunk 141 reproduced a 512-sample deficit at 16 kHz (**32 ms**). This
adds bounded regression evidence without certifying unselected audio, exact EOF,
whole-source capture or content quality.

## Actual long-summary result

The operator clicked Create summary at **09:32:58.554–58.825 UTC** after captions
stopped. Task `efc80421f1df` returned 0 in **19.097662 s** and the UI showed Finished
at 09:34:09.851 UTC. Its input had 563 segments and 6,746 words; output had three
English and three Spanish sentences. The model used the **beginning and end** of
the transcript and omitted the middle. The actual UI showed that warning and
identified the result as not reviewed by a person. This is successful generation,
not complete-transcript coverage or a summary-fidelity approval.

[Summary metadata and evidence](fresh-standard-terminal/summary-evidence.json) and
the [retained summary-only UI](fresh-standard-terminal/summary-ui.txt) bind the
observed result. A later selector lookup failed during observation; it did not
launch a second job. The owned operator service was deliberately terminated at
09:34:37.617548 after pipeline and summary completion. Launcher SIGTERM return −15
and driver shell exit 241 are separate from the pipeline's successful exit 0.

## Remaining acceptance and evidence

| #134 requirement | Evidence / remaining work |
|---|---|
| Runbook walk on church hardware or laptop | Laptop stand-in and actual operator page exercised; helper boundaries above. |
| Setup→first-caption timing and assistance | Both Standard Start brackets and first-caption events recorded; first-install/download duration unavailable. |
| Full hymn and spoken segment | Uncropped full service selected; hymn→prayer chronological/UI evidence retained. Repaired declared spans/header durations agree. Exact EOF/all-input sequence coverage is not claimed. |
| Written results and issues for blocking UX holes | The written note and verified archive record successes and failures; fixed operator UX blockers are listed above, with remaining hymn/quality follow-ups in #193 and #194. |

CPU Lite's terminal result and selected waveform reconstruction after both hours
are recorded above and retained in the verified archive. Native Mac lock/physical visibility,
microphone capture, independent TTS output routes, natural two-speaker diarization,
natural Spanish, bilingual human review and approved training remain separate
gates. Microphone and church hardware are **not added #134 requirements**. Audio,
models, private correction notes and full corpus excerpts are excluded from this
public note; no approvals, issue closures or release publication are fabricated.
