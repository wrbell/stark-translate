# Hymn title-boundary experiment — September 10, 2026

All **102 text-only translation calls completed**, with nonempty outputs, normal
stop reasons and no exhausted token budgets. The 34 text/model cells each produce
the same text across their three repeats. The retained raw hymn announcement still
exposes a subject/boundary problem; investigator-supplied newlines or quotes change
both models' interpretation. **This is descriptive evidence, without an approved
audio boundary, transcription, translation or default promotion.**

The packet contains 17 EN→ES inputs: three title-bearing examples in raw, newline
and quoted forms, plus eight contrastive controls. E4B and E2B run in six serial
workers with alternating model order and rotated case order. Workers ran from
19:21:46 to 19:24:28 UTC on `c13f51f1346b1581079b3457e34cb4d5fd0c2565`, using the
shared production correction and Gemma 4 translation paths. Thinking, drafting,
prompt cache and terminology prompts are off. This stage uses no audio, STT, TTS,
device stream or browser.

## Observed boundary and meaning differences

The retained input begins “Hymn number thirty six Eternity Time will soon end…”.
It is model-recognized text from historical session `20260910_043120_839144_en`
on `752ab9a`, not an approved transcription. Its title boundary is ambiguous in
the retained string.

| Input layout | E4B output excerpt | E2B output excerpt |
|---|---|---|
| Raw | “La eternidad pronto terminará” | “El tiempo de la eternidad pronto terminará” |
| Newline around `Eternity` | Separate `Eternidad`, then “El tiempo pronto terminará” | Separate `Eternidad`, then “El tiempo pronto terminará” |
| Quoted title on separate lines | Separate `"Eternidad"`, then “El tiempo pronto terminará” | Separate `"Eternidad"`, then “El tiempo pronto terminará” |

E4B's raw translation exactly reproduces the historical Spanish output, with
eternity as the ending subject. E2B combines the adjacent concepts as “the time
of eternity”. The supplied delimiters make both outputs separate the title word
and use time as the ending subject. They do not demonstrate that the system can
infer that boundary from speech. E2B also uses a “want to spend” construction,
omits an explicit “say”, and uses future `pasarán` for the passing moments where
the English uses present tense. Its `querréis` becomes `querrás` with delimiters.
E4B retains `digan` and `pasan`, with a `digan`/`pasaréis` register mix. The source
itself combines plural “sinners” with singular “thou”; these observations do not
establish an approved number/register choice.

The synthetic “Eternity Time will not end” example keeps explicit negation in all
layouts, but E2B's raw form again says `El tiempo de la eternidad`. Both models
distinguish the isolated controls “Time will soon end”, “Time will not end”,
“Eternity will soon end” and “Eternity will never end”. They retain the specified
caller/recipient in both Daniel/David directions, and retain both the faith-over-
works and reversed works-over-faith contrast. These observed distinctions are
not a doctrine score or a comprehensive semantic pass.

Production correction changes `Danic` to `Daniel` in exactly 18 calls, covering
all three layouts, two models and three repeats of that example. It preserves
the supplied whitespace and quotes; it inserts no title boundary. The remaining
84 calls have no recorded correction. Correction outputs match the frozen actual
function byte-for-byte.

## Isolated timing and bindings

The retained hymn example's translation-call medians are below, each based on
three separate calls. Correction time is measured separately; its overall median
is 0.023 ms and maximum 0.239 ms. Model load durations, including built-in warmup,
range from 5.08–5.66 seconds E4B and 3.04–3.41 seconds E2B, outside these call times.

| Layout | E4B median | E2B median |
|---|---:|---:|
| Raw | 1,845.9 ms | 866.3 ms |
| Newline | 2,097.5 ms | 934.1 ms |
| Quoted | 2,159.7 ms | 967.9 ms |

These are isolated correction/translation observations, with **n=3 per cell**.
They establish neither p95 performance nor capture-to-caption delivery, browser
visibility or production latency improvement. E4B defaults remain unchanged.

Independent retained-data review checked all six worker hashes, 102 unique calls,
34 comparison cells, source bindings before/after each worker, exact correction
results, input/output hashes, timing sums and actual requested/loaded model IDs.
Recorded local model inventories match the prepared identities: E4B revision
`6be2eaf50f08b88b5dadca0f4a02c0a0880e7bd3`, E2B revision
`0f3495e3e826e95fa21002c327805b840568f489`. This review did not reopen model weights.
The manifest's earlier preparation source `26f854c` remains separately recorded;
the executed worker bindings use `c13f51f` and the unchanged correction export
SHA-256 `3a6ed301978f9dcad4f22d5f74d2a7a1483581870ac1c1543511dbddb59e17eb`.

The completed index SHA-256 is
`8aafe951f9378484528fba57f957f08ee55c64247245c7bc9905e89dfe92ab7f`; comparison SHA-256
is `4f70b05d151d0a9f0cdec904d7332109552c4e346691f4c16de9ccf6a1519df8`.
Raw workers, the manifest, correction module, comparison and readable output
packet remain under `.cache/mac-en-es-closeout/final-sequence-v1/hymn-boundary/`.
The packet index identifies the archived worker, independent `review.json` and
semantic-review receipts separately from these unapproved output observations.
All inputs and outputs remain unapproved and ineligible for training.
