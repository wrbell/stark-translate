# Operator Runbook — stark-translate

> **Audience:** the volunteer running live translation at Stark Road Gospel Hall (Farmington Hills, MI) or a coffee-shop outreach event. Assumes you can open a web browser; no command line required.

---

## What you're running

The **stark-translate operator** turns a microphone in the room into live English/Spanish subtitles on a projector and any phones connected to the local Wi-Fi. Two screens matter:

1. **Operator screen** (your laptop or the church PC): `http://localhost:9000/operator/`
2. **Audience display** (projector / TV): `http://<server-ip>:8080/audience_display.html`

You only have to interact with the operator screen.

---

## Before the event (10 minutes)

### 1. Power on the church PC and the projector

- Wait ~30 seconds after boot for the operator service to come up automatically (systemd starts it at boot).
- If the service didn't auto-start, open a terminal and run:

  ```bash
  cd /opt/stark-translate
  ./run_operator.sh
  ```

  Leave the terminal window open during the event.

### 2. Open the operator UI

- On the church PC, open Firefox or Chrome and go to **http://localhost:9000/operator/**.
- You should see two panels: **Pre-flight** on the left, **Session** on the right, with **Live observability** and **Features** below.

### 3. Pre-flight: all green or yellow

The Pre-flight panel runs five checks:

| Check | Green means | Yellow means | Red means |
|---|---|---|---|
| GPU | CUDA or Apple Silicon detected | running on CPU (slow but works) | n/a |
| Translation models | Both Gemma 4 GGUFs found | one missing or HF NF4 fallback | n/a |
| Microphone | At least one input device | sounddevice unavailable | **no input devices** |
| Adapter manifest | `adapters/manifest.json` parses | manifest absent (using base models) | invalid JSON |
| llama-server | server reachable on port 8090 | server not running (HF fallback) | n/a |

**Red means STOP — fix before the event.** Most common red:

- *Microphone* red: USB mic not plugged in. Plug it into the same port you used last time.

Yellow is fine for an event. Yellow on llama-server just means we'll use the slower HF backend; everything still works.

### 4. Pick the right mic

In the **Session** panel, the **Mic** dropdown lists every input device. Pick the USB lavaliere mic by name (e.g. *"Yeti Stereo Microphone"* — not *"Built-in Microphone"*).

### 5. (Optional) Open the audience display on the projector

- On the projector PC: `http://<church-pc-ip>:8080/audience_display.html`
- Press **F11** for full-screen.
- The display will say *"Connecting…"* until you start a session — that's normal.

---

## During the event

### Start the session

1. Confirm **Language direction**: usually `EN → ES` for the speaker.
2. Click **Start session**. The state pill at the top right turns yellow (`STARTING`) then green (`RUNNING`).
3. The audience display starts showing live subtitles within a few seconds.

### Watch the dashboard while the speaker talks

- **VRAM / CPU** sparklines show how hard the machine is working. Steady is fine; spiking is fine.
- **Latency p50 / p95** shows the typical delay from speech to subtitle. **Under 1 second is healthy.** Above 3 seconds means something is stressed; consider a fallback (below).
- **Confidence mean** below 0.5 means the speaker is too quiet or there's too much background noise — adjust mic position.
- **Recent verses** under Features shows Bible references the system caught (Romans 8:28, John 3:16, etc.). Useful to confirm coverage.

### Mid-session controls

The controls row (just below Start / Stop) is for the speaker pausing or switching languages mid-event. **You don't need them for a normal sermon.**

| Button | When to press |
|---|---|
| **Pause** | Speaker is taking a long break (>1 minute). Stops audio capture; sparklines flatten. |
| **Resume** | Press after Pause. State pill goes back to green. |
| **Flip EN↔ES** | A Spanish-speaking member is taking the mic. Brief outage (~3 s) while we restart. |
| **Fallback to HF** | Translation latency is huge (>5 s) and you suspect llama-server crashed. Restarts on the slower-but-reliable HF backend. Brief outage. |

### Stop the session

Click **Stop session**. State pill goes back to gray (`IDLE`).

### Generate a post-session summary (optional)

After **Stop**, you can click **Generate summary** under Features. This kicks off a background task that produces a 5-sentence English + Spanish summary of what was said. Status appears in the small box; takes 1–5 minutes.

---

## When something goes wrong

### Audience display says "Disconnecting" / no subtitles

- Refresh the audience display browser tab.
- If still broken, click **Stop** then **Start session** in the operator UI.

### Operator UI says state="error"

- Click **Stop**. Wait until the pill says `IDLE`.
- Click **Start session** again.
- If it errors again on start, the GPU is likely out of memory. Power-cycle the church PC.

### USB mic gets unplugged mid-session

- A yellow toast appears on the operator UI: *"Audio devices changed — confirm your mic is still selected."*
- Plug the mic back in. The dropdown will refresh automatically.
- Re-select the mic in the dropdown if it cleared.
- Click **Stop** then **Start session** to resume cleanly.

### "VRAM" reads >90% of the card and stays there

- Click **Fallback to HF** to drop to the lower-VRAM backend.
- If still high, click **Stop** and restart the operator service via the church PC owner.

### Latency sparkline keeps creeping up

- Usually means another process is competing for the GPU (someone left a game open?).
- Click **Stop**, close other GPU apps, **Start** again.

---

## End-of-event checklist (2 minutes)

1. Click **Stop session** if not already stopped.
2. (Optional) Click **Generate summary** and wait for it to finish — the JSON lands in `metrics/`.
3. Close the browser tabs.
4. Power off the projector. Leave the church PC powered on; systemd will keep the operator service running for next time.

---

## Glossary for non-technical operators

| Term | Plain English |
|---|---|
| **Pre-flight** | The checklist that confirms the system can run a session. |
| **Session** | One run of the live translation, from Start to Stop. |
| **VAD** | "Voice Activity Detection" — the system noticing when someone starts/stops talking. Don't change unless told to. |
| **Backend** | Which inference path to use (CUDA = NVIDIA card, MLX = Apple Silicon). Leave on `auto`. |
| **Engine** | Which translation model to use. Leave on `auto`. The system picks llama.cpp if it's available, otherwise HF. |
| **A/B comparison** | Runs two translation models side-by-side. Only useful for development; **leave unchecked for live events**. |
| **TTS** | Text-to-speech (the system reading translations aloud). Leave unchecked unless you have headphones routed for it. |
| **VRAM** | Memory on the graphics card. The sparkline shows how full it is. |
| **Latency** | Time between someone speaking and the subtitle appearing. Lower is better. |
| **p50 / p95** | The "typical" (median) and "almost-worst-case" (95th percentile) latency. p95 should stay under 2 seconds. |
| **Confidence** | How sure the system is about what it heard. 0.0 = totally unsure, 1.0 = certain. Anything above 0.5 is fine. |
| **Adapter** | A small fine-tuning patch that improves accuracy on church-specific vocabulary. The pre-flight check tells you if one is loaded. |

---

## Quick health probe

A quick way to check the service is alive without opening the operator UI:

```
curl http://localhost:9000/healthz
```

Expect a 200 with JSON containing `"status": "ok"`. If that fails, the operator service is down — see "When something goes wrong" above.

## Who to call

| Problem | Who |
|---|---|
| Operator UI won't load at all | The dev who set up the church PC. |
| Pre-flight has red items you can't fix | Same dev. |
| Audience display projector not showing the laptop | The A/V volunteer. Hardware-only, not a software issue. |
| Wrong translation of a specific term (e.g. *Jacobo* vs *Santiago*) | Note the term, the time, and what was said. The dev will use it for fine-tuning next month. |

---

## Pre-event dry-run

Run this once a week before a real event, not the day of:

1. Open the operator UI.
2. All five pre-flight checks should be green or yellow (no red).
3. Click **Start session** with the default settings.
4. Speak a test sentence into the mic ("This is a test — Romans 8:28 says God is good.").
5. Confirm Spanish subtitle appears on the audience display within 2 seconds.
6. Confirm "Romans 8:28" appears in the Recent verses panel.
7. Click **Stop session**. Confirm the state pill returns to `IDLE`.

If any step fails, file an issue with the dev.
