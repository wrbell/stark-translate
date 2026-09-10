# Integrated operator rehearsal — 2026-09-10

Three actual operator/browser sessions completed using controlled local recordings.
This verifies the installed workflow's browser/API/pipeline integration; it does
not certify microphone permission, physical speakers or a full-hymn rehearsal.
Those device checks were explicitly deferred by the user.

| Session | Exercised behavior | Result |
|---|---|---|
| `20260910_003912_869693_en` | English→Spanish, actual partial/final captions, John 3:16 detection, Pause, Resume, Stop | 8 persisted finals; monotonic source positions; a `pause` final and a `stop` final retained buffered speech; completed lifecycle with no capture/persistence failures. |
| `20260910_004130_713413_es` | Spanish→English using a separately labeled synthetic Piper recording | “La gracia de Dios es suficiente.” → “God's grace is enough.”; completed on language restart. |
| `20260910_004221_108762_en` | Restart from Spanish to English through the Live button, new identity/direction, Stop | “The grace of God brings salvation.” → “La gracia de Dios trae la salvación.”; completed lifecycle. |

The live audience page showed the two language directions and new-session history
reset. It acknowledged 7/8 English finals and both short-session finals. The page
was reloaded during the first English caption, so that caption has no render
acknowledgement. **This rehearsal does not pass the 95% browser timing gate.**
The separate benchmark uses a pre-playback connection barrier to prevent this
startup loss; final/preview acknowledgements still determine measured coverage.

A note was saved on English session chunk 2, then recovered after reloading the
operator. Neither transcript nor translation was approved, and the original
predictions were retained. The audio player referenced the matching session/chunk
API route. This is a persistence test, not an approved active-learning example.

The short English session's actual summary action returned its recorded English
and Spanish text as a labeled excerpt, with the insufficient-text notice and
unreviewed status. It did not invent a longer sermon. Raw JSON is available in
Advanced while the ordinary page displays readable bilingual text.

The Help workflow built and downloaded a metadata-only support ZIP. Its only
member was a 2,090-byte `metadata.json`; the tested caption phrases, audio,
`/Users/` paths and environment-token names were absent. No bundle was sent to
anyone. The UI's preview byte count represents attachments, not generated metadata;
the follow-up UI now labels it “Metadata plus 0 B of optional attachments” for
this case. The existing support-preview regression assertion covers the corrected
wording. The Disk space card also keeps its detailed session checkbox list inside
a collapsed native disclosure, preserving selections and cleanup controls. These
are subsequent presentation fixes; the original captured rehearsal evidence is
unchanged.

Actual rehearsal caught and corrected stale status descriptions after startup,
Pause and Resume. The prominent live status now reflects input errors/stale health;
missing initial health is a startup wait, and confirmed controls hide obsolete
“waiting for acknowledgment” text. The raw status remains inspectable in Advanced.

[Machine evidence](overnight_operator_rehearsal.json) contains exact action times,
source/model/runtime metadata, final timing rows, completion records, render-ACK
coverage, summary output and artifact hashes. It distinguishes control-plane
success from browser latency certification. The [runbook](../operator_runbook.md)
now describes Prepare, Live, Sessions, Help and Advanced rather than the previous
single-page panel layout.
