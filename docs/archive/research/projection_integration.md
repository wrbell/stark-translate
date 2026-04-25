# Projection System Integration Research

Research document covering four approaches for integrating SRTranslate's live bilingual EN-to-ES
output into church projection systems. Evaluated for Stark Road Gospel Hall (Farmington Hills, MI).

---

## Current Architecture

The pipeline broadcasts translation results over WebSocket (port 8765) as JSON:

```json
{
  "type": "translation",
  "stage": "partial | stt | translation_a | complete",
  "chunk_id": 1,
  "english": "For God so loved the world",
  "spanish_a": "Porque de tal manera amo Dios al mundo",
  "spanish_b": "Porque tanto amo Dios al mundo",
  "stt_latency_ms": 320.5,
  "latency_a_ms": 650.2,
  "latency_b_ms": 1400.1,
  "e2e_latency_ms": 1720.3,
  "stt_confidence": 0.85
}
```

Existing display pages (`displays/audience_display.html`, `displays/church_display.html`, `displays/mobile_display.html`,
`displays/ab_display.html`) connect to this WebSocket and render text in browsers. The pipeline also runs
an HTTP server on port 8080 to serve these pages over the LAN.

---

## Approach 1: OBS Browser Source Overlay

**Status: Ready to use -- `displays/obs_overlay.html` created.**

### How It Works

OBS Studio supports "Browser Source" -- an embedded Chromium instance that renders any HTML page
and composites it into the OBS scene. The key feature: Browser Sources render `background: transparent`
as alpha transparency, meaning the text floats over whatever video/image is behind it.

### Setup Steps

1. In OBS, add a new **Browser** source to your scene.
2. Set the URL to either:
   - **Local file:** `file:///path/to/SRTranslate/displays/obs_overlay.html`
   - **HTTP:** `http://localhost:8080/displays/obs_overlay.html` (if the HTTP server is running)
3. Set **Width** to your canvas width (e.g., 1920) and **Height** to canvas height (e.g., 1080).
4. Check the **"Custom CSS"** field and ensure it contains:
   ```css
   body { background-color: transparent !important; margin: 0px auto; overflow: hidden; }
   ```
   (This is already set in the HTML, but the OBS custom CSS field reinforces it.)
5. URL parameters for configuration:
   - `?model=a` -- Use Gemma 4B translation (default)
   - `?model=b` -- Use Gemma 12B translation
   - `?english=0` -- Hide English subtitle, show only Spanish
   - `?english=1` -- Show English subtitle (default)
   - `?port=8765` -- WebSocket port (default 8765)
   - `?lines=3` -- Max visible text lines (default 3)

### displays/obs_overlay.html Design

The overlay file (`displays/obs_overlay.html`) differs from the audience display in several ways:

- **Transparent background** -- no black/white fill; gradient backdrop only behind text.
- **Lower-third positioning** -- text anchored to the bottom of the viewport.
- **No UI chrome** -- no header, no fullscreen button, no QR code, no status bar.
- **Text shadow** -- ensures legibility over any background (sermon slides, video, camera feed).
- **Fade animation** -- smooth CSS transitions for text appearing/disappearing.
- **Auto-clear** -- text fades out after 12 seconds of silence.
- **Partial support** -- italic styling for partial (MarianMT) translations, replaced when Gemma
  final translation arrives.
- **Profanity filter** -- same biblical-term-aware filter as other displays.

### Can `displays/audience_display.html` Be Used Directly?

Yes, but with caveats:
- It has a white background, which would show as a solid white rectangle in OBS.
- The header ("Stark Road Gospel Hall") and fullscreen button would be visible.
- The two-column layout is designed for a standalone display, not an overlay.

You could use it with OBS Color Key / Chroma Key to remove the white background, but this is
fragile (white text content could also become transparent). The dedicated `displays/obs_overlay.html` is
the better option.

### Pros

- **Zero additional software** -- OBS is already commonly used for church livestreaming.
- **Works immediately** -- just add a Browser Source and point it at the URL.
- **Full alpha transparency** -- text composites cleanly over any background.
- **WebSocket auto-reconnect** -- the overlay reconnects automatically if the pipeline restarts.
- **CSS-controllable** -- change font size, position, colors without touching Python.
- **NDI-compatible output** -- OBS can output its composited scene as NDI via the obs-ndi plugin,
  which ProPresenter or other software can then consume.

### Cons

- **Requires OBS in the signal chain** -- adds one more application to manage during a service.
- **Single-machine affinity** -- the Browser Source runs inside OBS on the same machine, so
  the pipeline and OBS must be on the same network (or same machine for `localhost`).
- **Limited positioning control** -- text position is baked into the HTML/CSS. To move the overlay,
  you either edit the CSS or resize/reposition the OBS source.

### Effort Estimate

**< 1 hour.** The overlay file is already created. Setup is purely configuration in OBS.

---

## Approach 2: NDI Output

### How It Works

NDI (Network Device Interface) is a video-over-IP protocol used widely in broadcast and AV.
A Python process would render text onto RGBA frames (with alpha channel for transparency) and
send them as an NDI source. Any NDI-capable software (OBS with obs-ndi plugin, ProPresenter,
vMix, Resolume, NewTek TriCaster) could consume this as a video input with transparency.

### Technical Implementation

The `ndi-python` package (`buresu/ndi-python` on GitHub, available on PyPI) provides Python
bindings for the NDI SDK. A sender script would:

1. Listen on the same WebSocket for translation messages.
2. Render text onto a transparent BGRA numpy array using Pillow or Cairo.
3. Send the frame via `ndi.send_send_video_v2()`.

Rough architecture:

```python
import NDIlib as ndi
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Initialize NDI sender
send_settings = ndi.SendCreate()
send_settings.ndi_name = "SRTranslate Captions"
send_instance = ndi.send_create(send_settings)

# Create BGRA frame (1920x1080, transparent)
frame = np.zeros((1080, 1920, 4), dtype=np.uint8)  # BGRA, all alpha=0

# Render text with Pillow
img = Image.fromarray(frame, 'RGBA')
draw = ImageDraw.Draw(img)
font = ImageFont.truetype("/path/to/font.ttf", 48)
draw.text((100, 900), "Porque de tal manera amo Dios al mundo",
          fill=(255, 255, 255, 255), font=font)

# Send via NDI
video_frame = ndi.VideoFrameV2()
video_frame.data = np.array(img)
video_frame.FourCC = ndi.FOURCC_VIDEO_TYPE_BGRA
ndi.send_send_video_v2(send_instance, video_frame)
```

### NDI SDK Requirement

The NDI SDK must be installed on the machine. On macOS, download the NDI SDK from
https://ndi.video/tools/. The `ndi-python` package wraps the SDK's C library.

### ProPresenter NDI Input

ProPresenter 7 can receive NDI inputs as video sources. The SRTranslate NDI source would
appear in ProPresenter's media browser and could be placed as a layer over the main presentation.
This provides a direct ProPresenter integration without the WebSocket/Stage Display API.

### Pros

- **Universal AV compatibility** -- any NDI-capable software can consume the source.
- **True alpha channel** -- NDI natively supports BGRA with transparency.
- **Network-independent** -- NDI works over any local network; the caption source machine
  does not need to be the same machine running OBS/ProPresenter.
- **Multi-consumer** -- multiple receivers can consume the same NDI source simultaneously.
- **ProPresenter native** -- ProPresenter can receive NDI directly, no OBS needed.

### Cons

- **Additional dependency** -- requires NDI SDK installed on the machine, plus `ndi-python`.
- **Frame rendering complexity** -- must handle text layout, word wrapping, font rendering
  in Python (Pillow or Cairo). Less flexible than CSS for styling.
- **CPU overhead** -- rendering and encoding 1080p frames at even 10 fps adds CPU load.
  Not a concern on M3 Pro, but worth noting.
- **Font management** -- must ship/reference specific TTF files for consistent rendering
  across machines.
- **`ndi-python` maturity** -- the package works but is community-maintained; not as
  battle-tested as OBS Browser Sources.

### Effort Estimate

**4-8 hours.** Write a WebSocket-to-NDI bridge script, handle text rendering with proper
word wrapping, test transparency with OBS/ProPresenter NDI input. An additional 1-2 hours
for polishing (font selection, animation, edge cases).

---

## Approach 3: ProPresenter Integration

### ProPresenter 7 API Options

ProPresenter 7 exposes two integration paths:

#### Option A: Stage Display Messages (WebSocket Protocol)

ProPresenter has a "Messages" feature designed for exactly this use case -- displaying
scrolling or static text overlays on stage displays and outputs. The WebSocket API allows
sending arbitrary text as a stage display message.

**Connection:**
- WebSocket endpoint: `ws://<propresenter-ip>:<port>/stagedisplay`
- Authentication required immediately after connection.
- Protocol uses JSON messages.

**Sending a message:**
```json
{
  "action": "messageSend",
  "messageIndex": 0,
  "messageKeys": ["Caption"],
  "messageValues": ["Porque de tal manera amo Dios al mundo"]
}
```

Important: The message must already be configured in ProPresenter's Messages panel with
a placeholder token (e.g., `{{Caption}}`). The API replaces the token value, not the
message itself. This means someone must set up a "Live Translation" message template
in ProPresenter before the API integration works.

**Warning from community docs:** It is easy to crash ProPresenter by sending malformed
messages. Validate all outgoing JSON carefully.

#### Option B: Official REST API (ProPresenter 7.9+)

ProPresenter 7.9 introduced an officially supported REST API. The relevant endpoint:

```
PUT /v1/stage/message
Body: "Porque de tal manera amo Dios al mundo"
```

This is simpler and more stable than the WebSocket protocol. However, it requires
ProPresenter 7.9 or later.

Additional useful endpoints:
- `GET /v1/messages` -- list all configured messages.
- `GET /v1/messages/{id}` -- get a specific message.
- `PUT /v1/messages/{id}/trigger` -- trigger a message to show.
- `DELETE /v1/messages/{id}/trigger` -- hide a message.

#### Option C: NDI Input (via Approach 2)

As described in Approach 2, ProPresenter can consume an NDI source directly as a video
layer. This bypasses the message/API system entirely and gives full control over
text rendering, but requires the NDI sender script.

### Community Resources

- **featherbear/propresenter-stagemessages** -- Node.js tool for sending stage display
  messages to ProPresenter 6/7 over WebSocket.
- **jeffmikels/ProPresenter-API** -- comprehensive community documentation of the
  undocumented WebSocket protocol (covers Pro 6 and 7).
- **openapi.propresenter.com** -- official REST API documentation (7.9+).
- **BenJamesAndo/ProPresenter-OBS** -- script to feed ProPresenter slides into OBS
  for lower thirds (reverse direction of what we want, but useful reference).

### Pros

- **Native integration** -- text appears in ProPresenter's output pipeline, subject to
  the same transitions and formatting as other ProPresenter content.
- **No additional software** -- if the church already runs ProPresenter, no OBS needed.
- **Stage display routing** -- messages can be routed to specific outputs (main,
  stage, broadcast) independently.
- **Operator control** -- the ProPresenter operator can hide/show the translation
  message at any time using their normal workflow.

### Cons

- **ProPresenter dependency** -- requires ProPresenter to be running and configured.
- **Message template setup** -- someone must create the message template in ProPresenter
  with the correct token names before the integration works.
- **Limited formatting** -- message formatting is controlled by ProPresenter's message
  editor, not by our code. Font size, position, and animation are set in ProPresenter.
- **Version sensitivity** -- the WebSocket protocol changed between 7.0 and 7.4.2
  (protocol version 700 vs 701). The REST API requires 7.9+.
- **Crash risk** -- the community docs explicitly warn that malformed WebSocket messages
  can crash ProPresenter. This is not acceptable during a live service.
- **Single-language limitation** -- the Messages feature shows one text string. Showing
  both English and Spanish would require two separate messages or a combined string.

### Effort Estimate

**6-12 hours.** Write a Python WebSocket client that authenticates with ProPresenter
and sends stage messages. Handle reconnection, error recovery, and message formatting.
Test extensively before using in a live service due to crash risk.

For the REST API path (7.9+): **3-5 hours.** Simpler HTTP PUT requests, less crash risk.

---

## Approach 4: PowerPoint Live Captions Comparison

### PowerPoint Live Captions Overview

PowerPoint's built-in "Present with Subtitles" feature provides real-time speech-to-text
with optional translation into 60+ languages. It uses Microsoft Azure Cognitive Services
(cloud-based) for recognition and translation.

### How Our System Compares

| Dimension | PowerPoint Live Captions | SRTranslate |
|-----------|------------------------|-------------|
| **STT Engine** | Azure Speech Services (cloud) | Distil-Whisper via MLX (local) |
| **Translation** | Azure Translator (cloud) | TranslateGemma 4B/12B via MLX (local) |
| **Internet required** | Yes (always) | No (fully on-device) |
| **Latency** | 1-3 seconds (network round-trip) | 200-500ms partials, 1-2s final |
| **Privacy** | Audio sent to Microsoft cloud | All processing stays on-device |
| **Theological vocabulary** | Generic -- mishandles church terms | Custom Whisper prompt + future fine-tuning |
| **Language model bias** | General-purpose | Bible-domain fine-tuned (planned) |
| **Customization** | Font size/position only | Full CSS control, multiple display modes |
| **Cost** | Microsoft 365 subscription | Free (open-source models) |
| **A/B testing** | Not possible | Built-in dual-model comparison |
| **Confidence scoring** | Not exposed | Full segment + word-level confidence |
| **Quality monitoring** | None | CometKiwi, LaBSE, back-translation QE |

### PowerPoint Limitations for Church Use

1. **Cloud dependency** -- requires stable internet. Many church buildings have unreliable
   WiFi, and livestreaming already consumes bandwidth.

2. **Generic vocabulary** -- PowerPoint's speech recognition is trained on general English.
   It frequently misrecognizes theological terms:
   - "atonement" as "at one meant"
   - "propitiation" as "proposition"
   - "mediator" as "media tour"
   - Biblical names (Melchizedek, Nebuchadnezzar) are rarely recognized correctly.

3. **No theological glossary** -- there is no way to provide a custom vocabulary or glossary
   to bias the recognition toward church-specific terms.

4. **Translation quality for religious text** -- Azure Translator is optimized for business
   and conversational language. Religious register, archaic phrasing, and doctrinal precision
   are not its strength. "Grace" may be translated as "charm" instead of "gracia" in the
   theological sense. "Covenant" may become "contrato" instead of "pacto."

5. **No quality feedback** -- there is no API to access confidence scores, quality estimates,
   or error logs. You cannot monitor or improve quality over time.

6. **Presentation coupling** -- Live Captions only works during an active PowerPoint
   slideshow. If the church uses ProPresenter (not PowerPoint) for worship slides, this
   feature is unusable.

### Can We Overlay on PowerPoint?

Yes, but indirectly:
- Run PowerPoint in slideshow mode.
- Use OBS to capture the PowerPoint window.
- Add the `displays/obs_overlay.html` Browser Source on top.
- Output the combined scene to the projector or NDI.

This works but adds OBS to the chain and defeats the simplicity argument for using
PowerPoint captions in the first place.

### Verdict

PowerPoint Live Captions is a reasonable zero-effort baseline for churches that already
use PowerPoint and have reliable internet. However, for Stark Road Gospel Hall's needs
(theological accuracy, offline operation, Spanish translation quality, quality monitoring,
and iterative improvement), SRTranslate is a fundamentally better architecture. The domain
fine-tuning planned in Phases 6-8 will widen this gap significantly.

---

## Recommendation

### Short-term (this week): OBS Browser Source

The OBS Browser Source approach is ready to use today with `displays/obs_overlay.html`.

**Setup checklist:**
1. Start the SRTranslate pipeline (`python dry_run_ab.py`).
2. Open OBS Studio.
3. Add a Browser Source pointed at `http://<mac-ip>:8080/displays/obs_overlay.html?model=a`.
4. Set width/height to match canvas (1920x1080).
5. Position the source at the bottom of the scene (it self-positions as a lower third).
6. Layer it above the camera/presentation source.
7. Output from OBS to the projector (fullscreen projector output or NDI).

This gets bilingual captions on the church projector with minimal setup and no risk
of crashing ProPresenter.

### Medium-term (next month): ProPresenter REST API

If the church runs ProPresenter 7.9+, the REST API (`PUT /v1/stage/message`) is the
cleanest native integration. Write a small Python bridge that reads from the WebSocket
and POSTs to ProPresenter. This gives the operator full control over when captions
are visible and which output they appear on.

### Long-term (if needed): NDI Output

The NDI approach is the most universal but also the most work. Pursue it only if:
- Multiple applications need to consume the caption stream simultaneously.
- The church AV setup requires NDI routing (e.g., through a video switcher).
- The OBS Browser Source approach proves insufficient for some reason.

### Not Recommended: PowerPoint Live Captions

PowerPoint Live Captions should not be the primary solution. It lacks theological
vocabulary support, requires internet, provides no quality monitoring, and cannot be
improved over time. It is useful only as a quick comparison baseline to demonstrate
that SRTranslate produces better results.

---

## Quick Reference: Integration Paths

| Approach | Effort | Dependencies | Quality Control | Best For |
|----------|--------|-------------|-----------------|----------|
| **OBS Browser Source** | < 1 hour | OBS Studio | Full (our pipeline) | Immediate deployment |
| **NDI Output** | 4-8 hours | NDI SDK, ndi-python, Pillow | Full (our pipeline) | Multi-app AV routing |
| **ProPresenter API** | 3-12 hours | ProPresenter 7.9+ | Full (our pipeline) | Native PP integration |
| **PowerPoint Captions** | 0 hours | Microsoft 365, internet | None | Quick baseline comparison |

---

## Sources

- [ProPresenter Official REST API Documentation](https://openapi.propresenter.com/)
- [ProPresenter Community WebSocket API (jeffmikels)](https://github.com/jeffmikels/ProPresenter-API)
- [ProPresenter Stage Messages Tool (featherbear)](https://github.com/featherbear/propresenter-stagemessages)
- [ndi-python -- NDI SDK Python wrapper (buresu)](https://github.com/buresu/ndi-python)
- [ndi-python send_video.py example](https://github.com/buresu/ndi-python/blob/master/example/send_video.py)
- [OBS NDI Plugin (DistroAV)](https://obsproject.com/forum/resources/distroav-network-audio-video-in-obs-studio-using-ndi%C2%AE-technology.528/)
- [OBS Browser Source Transparency Discussion](https://obsproject.com/forum/threads/translucent-transparent-browser-source.59549/)
- [ProPresenter TCP/IP API Documentation](https://support.renewedvision.com/hc/en-us/articles/31606866768147-TCP-IP-Connections-with-ProPresenter-API)
- [PowerPoint Live Captions Guide (UGA)](https://oit.caes.uga.edu/live-captions-and-subtitles-in-powerpoint/)
