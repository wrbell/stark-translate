# P4 — Marian CT2 vs live Gemma E4B on 8–12-word finals: blinded reference-free review packet

**Why.** Marian-routed finals reach the payload in ≈ 0.8–0.9 s against ≈ 1.6–2.0 s for Gemma-routed ones, but the
conservative Marian route (`conservative-marian-routing`) is quality-gated and blocked on natural Spanish references.
FLEURS has no 8–12-word English sentences and the theological canaries carry lexical targets only, so this band can
only be judged by a bilingual reviewer comparing meaning against the English source. This packet gives that reviewer
the live Gemma output and the production Marian CT2 output for the same finals, blinded.

**Sources.** The twelve series-3 control replays (`ts0911_{A,B}_ctl_r{0,1,2}` and `lb0912_{A,B}_ctl_r{0,1,2}`,
diagnostics SHA256 in `packet/packet.json → provenance.sources`). Kept: finals routed to Gemma whose corrected English
has 8–12 words; deduplicated by normalized English (the repeats replay the same two clips) → **30 unique finals**
(8 words: 10, 9: 8, 10: 2, 11: 4, 12: 6). Hypothesis `gemma_live` = the stored live `spanish_gemma`; hypothesis
`marian_ct2` = `Helsinki-NLP/opus-mt-en-es` through the production CT2 int8 CPU engine (`adapters/marian_ct2/en-es/active`,
`intra_threads 4`, `max_new_tokens 128`) run offline on the identical English by
[`make_marian_packet.py`](make_marian_packet.py) on 2026-09-12T00:00Z (`main` @ `5227a73`, `venv/bin/python`).
No live output drifted across the repeated replays (0 of 30 sources).

**Packet** `031184b1e1eba5188a476cb7` — [`packet/packet.md`](packet/packet.md) (reviewer view),
[`packet/packet.json`](packet/packet.json), [`packet/annotations.jsonl`](packet/annotations.jsonl),
[`packet/review.schema.json`](packet/review.schema.json), [`packet/packet-index.json`](packet/packet-index.json).
A/B labels are assigned per case with an HMAC of a private seed; the mapping key lives outside the docs tree
(`.cache/series4-20260912/P4/private-key.json`, mode 0600) and is not published. 22 of 30 cases have different A/B text;
8 are identical.

**Automatic tripwire (reported, not judged; aggregates only so the packet stays blind).** chrF of Marian against
Gemma: p50 89.4 (min 43.2, max 100). Tier-2 glossary terms (`bible_data/glossary/tier2_master.json`, SHA256
`dd0b5f84…`) occur in 12 of 30 sources (14 instances); both engines render the mapped Spanish term in 14 of 14.
Marian offline latency on this CPU: 30.7 ms p50 / 46.3 ms p95 per final. Output lengths are equal at the median
(8 words).

**Status.** Human review pending; no routing change follows from this packet. It is a 30-item screen of one band on
two sermons and cannot certify Marian finals for production. Once reviewed, the outcome feeds the
`conservative-marian-routing` backlog item.
