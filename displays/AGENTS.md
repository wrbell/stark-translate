# displays/AGENTS.md — Browser Clients (Agent Guide)

> Paired with [`CLAUDE.md`](./CLAUDE.md).

## Ports and surfaces

| Port | Surface |
|------|---------|
| 8080 | Static audience/mobile/church/OBS HTML |
| 8765 | Text WebSocket (partials/finals) |
| 8766 | TTS audio WebSocket |
| 9000 | Operator SPA + `/ws/control` metrics |

Primary operator UI: `displays/operator/` via FastAPI — not legacy `ab_display.html` for Sunday use.

## Protocol notes

- `stage`: `partial` (italic) vs `translation_a` / final
- Caption history: new session clears; same-session reconnect preserves
- Visible render ACKs: separate telemetry from pipeline latency

## Full reference

Message fields, auto-reconnect, multilingual extension plan:
[`CLAUDE.md`](./CLAUDE.md)
