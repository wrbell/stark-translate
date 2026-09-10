# September 10 attended microphone and operator checks

The unlocked Mac and newly granted microphone permission allowed the real built-in
microphone startup retest. Both English and Spanish sessions received fresh audio
and reached `ready`; neither reproduced the previous capture stall. The room
produced no speech-classified frames, so this verifies capture/readiness and
controls, not spoken microphone transcription or translation quality.

The [source/runtime receipt](raw/source-runtime.json) records current checkout
source on `20bc410ed51b379d7c577ea3bf3aad4c96ef3f9b`, pipeline SHA-256
`c708e5557035dd480b6b1665e68aa91af0c6947354e28791f64b62895c583872`, and the
established `.cache/package-smoke` interpreter. Package version is 2026.14.0.0;
Torch/TorchAudio remain 2.10.0 here. This was current source using the existing
artifact environment's dependencies, not a newly installed wheel or the separate
[Torch 2.13 feasibility runtime](../security_feasibility_20260910/README.md).
The existing project Marian CT2 adapters were selected; their unknown source
revision was not replaced with an invented manifest pin. Exact model hashes are
in the retained lifecycle records.

## Actual microphone sessions

| Session | Actual result |
|---|---|
| `20260910_092405_457769_en` | `audio_source=mic`, English→Spanish; fresh nonzero input and `ready`; Pause acknowledged as `paused`, Resume acknowledged as `ready` with the same identity; completed exit 0 on the language switch. |
| `20260910_092544_206252_es` | New identity after **Switch to Spanish speaker**; `audio_source=mic`, Spanish→English; fresh nonzero input and `ready`; audience headers changed direction; **Stop captions** completed exit 0. |

The browser's two-second **Test microphone** selected **MacBook Pro Microphone**
(input device 1) and measured peak 77%, average 2% of full scale. The live pipeline
then repeatedly logged increasing input-frame counts and nonzero RMS. Every
logged `speech` count was zero, and both sessions persisted zero final captions.
Each lifecycle records all two required writes completed, zero failed/pending
writes, and no capture error. Readiness snapshots, frame/RMS samples, exact
session times and copies of the session artifacts are retained in the
[machine evidence](raw/session-evidence.json).

Both sessions had connected audience/operator browser clients, but zero caption
events means there are no microphone caption-render acknowledgements. A spoken
EN/ES microphone check remains necessary for the speech and translation portion
of #131. The attended run exercised the EN→ES live restart; the return to English
after stopping was a configuration restoration, not another live restart.

## Separate controlled replay and UI recovery

Session `20260910_092824_526799_en` used a separately identified retained file,
with four seconds of leading silence and three seconds of trailing silence.
The [source receipt](raw/replay-source.json) binds the original and derived file
hashes. Its original speaker/natural-speech provenance was not independently
established. It is explicitly `session_kind=replay`, `audio_source=file`, and
does not substitute for speech in the attended microphone sessions.

The replay produced two revisable partials and one final. The actual audience
showed “The grace of God brings salvation.” and “La gracia de Dios trae la
salvación.” All three events have `caption_rendered`, `visible=true` receipts
for client `21589014544`. The final measured **4,816.7 ms payload-ready** and
**4,837.8 ms to visible-browser ACK**, including 21.0 ms send-to-ACK. These are
one-session observations, not a sub-second or cohort promotion claim.
The [original ACK rows](raw/display_metrics_20260910_092824_526799_en.jsonl)
and [final CSV](raw/ab_metrics_20260910_092824_526799_en.csv) remain separate.

A native Chrome screenshot taken after foreground selection showed both final
caption lines in the actual unlocked window. A Browser Guard notice covered the
upper-right area but did not obscure either line. The screenshot was observed in
the CUA task output; the [UI observation record](raw/ui-observations.json) is an
explicit transcription of those observations, not a fabricated screenshot.
This is laptop visibility, not a physical projector test or continuous native
proof for every earlier partial.

In **Sessions**, an unapproved note was saved as revision 1. A different unsaved
note survived page reload and the UI reported **Restored unsaved draft**. The
recovery result was saved as revision 2. Both transcript and bilingual approval
flags remain false, and original predicted text is unchanged. The
[revisioned sidecar](raw/replay-unapproved-review.jsonl) records both saves.

**Create summary** completed as the short-session excerpt path: one segment,
six words, `model: null`, `translation_method: recorded prediction`, and
`human_reviewed: false`. The UI displayed the insufficient-text notice and both
recorded language lines. The [summary result](raw/summary_20260910_092824_526799_en.json)
does not claim a generated sermon summary. This check used no model inference.

## Device and cleanup boundaries

The [actual device inventory](raw/audio-device-inventory.json) includes MacBook
Pro Speakers (stereo, output 2) and Microsoft Teams Audio (mono input/output 3).
The latter is an available virtual-routing candidate, not a completed routing
test. No speaker test, audible TTS, physical second output or virtual audio
loopback was exercised.

All three pipelines and capture children were stopped before the model slot was
returned; the [cleanup receipt](raw/cleanup.json) confirms the owned process IDs
are gone. The separate file-replay operator on port 9001 was then shut down and
its test tabs closed. The primary operator remains idle on
`http://127.0.0.1:9000/operator/` (PID 6564), in **Prepare**, English→Spanish,
Standard profile, with the tested built-in microphone selected and TTS off.
No system audio settings, model defaults, human approvals, release publication or
GitHub state were changed by this rehearsal.
