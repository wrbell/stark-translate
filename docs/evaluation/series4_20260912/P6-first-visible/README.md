# P6 — first-visible acknowledgement of the first streamed final tokens

**What changed (PR #216, `feat(telemetry): acknowledge the first streamed final tokens per chunk`).** The pipeline
already streamed Gemma tokens to displays in batches, and the displays rendered the first batch immediately, but
stream messages carried no event id and were never acknowledged, so the only measured delivery number was
"speech end → final payload visible". Now:

- `dry_run_ab.py::stream_token_broadcaster` gives the **first** `translation_stream` batch per chunk the deterministic
  event id `<session_id>:stream:<chunk_id>`; later batches keep the broadcast counter ids.
- `_caption_before_send` registers that batch with the render tracker as stage `first_stream` (producer speech end,
  utterance and sample metadata, tokens so far, delivery mode, queue wait), once per client per chunk, in both the
  awaited and the queued delivery modes.
- `displays/caption_telemetry.js` acknowledges that message under the same rules as finals (synchronous DOM
  mutation in a visible tab, double animation frame) and adds `stage: "first_stream"` to the ACK. Audience, A/B and
  OBS displays render streams; church and mobile displays ignore them and therefore produce no first-stream ACK
  (fail-closed). A first batch coalesced away by queued delivery produces no row; later batches cannot substitute.
- `tools/overnight_bench.py`, `tools/mac_evaluation.py` and `tools/tail_screen_report.py` (optional
  `display_metrics_jsonl` per run) report `first_visible_ms` = the `first_stream` row's
  `speech_end_to_ack_upper_bound_ms` (p50/p95, nearest rank), separately from `complete`, never as a gate.
- Definitions: `docs/evaluation/README.md` (schema-2 `first_stream` stage), `displays/CLAUDE.md` (id shape).

**Verification.** Implemented by Codex (`gpt-6-astra`) from a Claude spec in an isolated worktree; reviewed by
Claude; 258 targeted tests including the socket-bound display/operator modules passed locally with the `stt_env`
pytest against the branch; CI lint/test/security green before the squash merge.

**What was not done.** No measurement was taken: the machine-timed replay harness has no display client, so the
series-4 screens report the wire-side first-token number (`translation_started + ttft − speech_end`) and not a
browser ACK. The attended replay with the audience display connected (unlocked Mac) is Willem's Saturday item and
will produce the first `first_visible_ms` numbers alongside the existing `complete` coverage. This change does not
certify visible delivery and does not alter any caption timing.
