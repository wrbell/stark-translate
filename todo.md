# TODO.md — Dry Run: A/B Inference with Live 4-Quadrant Display

> **Goal:** Live mic → Distil-Whisper STT → TranslateGemma 4B + 12B → 4-quadrant browser display (B left, A right, English top, Spanish bottom) with latency metrics.
>
> **Machine:** M3 Pro MacBook, 16GB unified memory
>
> **Output:** Fullscreen browser display connected via WebSocket to the Python backend
>
> **Time estimate:** ~30 min setup + ~25GB model download + 15 min testing

---

## Phase 1: Environment Setup

- [ ] **1.1** System deps
  ```bash
  brew update && brew install python@3.12 ffmpeg portaudio
  ```

- [ ] **1.2** Project + virtualenv
  ```bash
  mkdir -p ~/stt_bilingual
  cd ~/stt_bilingual
  python3.12 -m venv stt_env
  source stt_env/bin/activate
  ```

- [ ] **1.3** PyTorch with MPS
  ```bash
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu
  ```

- [ ] **1.4** Verify MPS
  ```bash
  python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"
  ```

- [ ] **1.5** Install all dependencies
  ```bash
  pip install transformers accelerate
  pip install sounddevice numpy pandas
  pip install sentencepiece protobuf
  pip install bitsandbytes                # 4-bit quantization (both models in RAM)
  pip install websockets                  # WebSocket server for display
  ```

- [ ] **1.6** Install Silero VAD
  ```bash
  pip install silero-vad
  ```

- [ ] **1.7** Mic permissions
  - **System Settings → Privacy & Security → Microphone** → enable Terminal

---

## Phase 2: Model Downloads

> ~25GB total. Use Wi-Fi. Cached after first download.

- [ ] **2.1** Distil-Whisper (~1.5GB)
  ```bash
  python -c "
  from transformers import pipeline; import torch
  p = pipeline('automatic-speech-recognition', model='distil-whisper/distil-large-v3',
               device='mps', torch_dtype=torch.float16)
  print('✓ Distil-Whisper cached')
  "
  ```

- [ ] **2.2** TranslateGemma 4B (~5GB)
  ```bash
  python -c "
  from transformers import AutoTokenizer, AutoModelForCausalLM; import torch
  AutoTokenizer.from_pretrained('google/translategemma-4b')
  AutoModelForCausalLM.from_pretrained('google/translategemma-4b', torch_dtype=torch.float16)
  print('✓ TranslateGemma 4B cached')
  "
  ```
  - **Auth?** `pip install huggingface_hub && huggingface-cli login`, accept license on HF

- [ ] **2.3** TranslateGemma 12B (~24GB)
  ```bash
  python -c "
  from transformers import AutoTokenizer, AutoModelForCausalLM; import torch
  AutoTokenizer.from_pretrained('google/translategemma-12b')
  AutoModelForCausalLM.from_pretrained('google/translategemma-12b', torch_dtype=torch.float16)
  print('✓ TranslateGemma 12B cached')
  "
  ```

- [ ] **2.4** Silero VAD
  ```bash
  python -c "
  import torch
  torch.hub.load('snakers4/silero-vad', 'silero_vad')
  print('✓ Silero VAD cached')
  "
  ```

---

## Phase 3: Create Files

Two files go in `~/stt_bilingual/`:

### 3.1 — `ab_display.html`

- [ ] Save `ab_display.html` into `~/stt_bilingual/ab_display.html`
- Self-contained HTML — no build step, no npm
- Connects to `ws://localhost:8765`
- 4-quadrant layout:
  - **Top-left:** B (12B) English — blue accent
  - **Top-right:** A (4B) English — green accent
  - **Bottom-left:** B (12B) Spanish — blue accent
  - **Bottom-right:** A (4B) Spanish — green accent
- Footer: running chunk count, avg latencies, same-output %, 4B-faster %

### 3.2 — `dry_run_ab.py`

- [ ] Save `dry_run_ab.py` into `~/stt_bilingual/dry_run_ab.py`
- Architecture: Mic → Silero VAD → Distil-Whisper → [Gemma 4B + Gemma 12B] → WebSocket → Browser
- Two modes:
  - **Parallel (default):** Both Gemma models in 4-bit via bitsandbytes — real-time A/B
  - **Swap (fallback):** Loads one model at a time if 4-bit fails on macOS
- Also writes terminal output + CSV metrics alongside the browser display

---

## Phase 4: Run It

- [ ] **4.1** Start the backend
  ```bash
  cd ~/stt_bilingual
  source stt_env/bin/activate
  python dry_run_ab.py
  ```

- [ ] **4.2** Wait for all ✓ checkmarks (1-3 min on first run)

- [ ] **4.3** Open the display
  ```bash
  # In a new terminal tab:
  open ~/stt_bilingual/ab_display.html
  ```
  - Green dot = connected
  - **Cmd+Shift+F** or **F11** for fullscreen

- [ ] **4.4** Speak test phrases — watch all 4 quadrants update:

  | Type | Phrase |
  |------|--------|
  | Simple | "Good morning everyone, welcome to our service." |
  | Scripture | "Let us open our Bibles to Romans chapter 8 verse 28." |
  | Theological | "Justification by faith is the cornerstone of the Gospel." |
  | Names | "The apostle Paul wrote to the Philippians from prison." |
  | Great Commission | "Go therefore and make disciples of all nations, baptizing them in the name of the Father and of the Son and of the Holy Spirit." |
  | Pastoral | "Let us remember our brothers and sisters who are going through difficult times this week." |
  | Numeric | "This letter was written around 55 AD during Paul's third missionary journey." |

- [ ] **4.5** Run 5+ minutes / 10+ utterances

- [ ] **4.6** Ctrl+C for summary stats

- [ ] **4.7** Review CSV: `open ab_metrics_*.csv`

---

## Phase 5: Fallback If bitsandbytes Fails

Edit line ~38 of `dry_run_ab.py`:
```python
SWAP_MODE = True   # was False
```

Swap mode loads/unloads one Gemma model at a time in fp16. Slower cycles (~30-60s per swap) but no quantization dependency.

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| Browser shows "Disconnected" | Backend not running — check terminal for errors |
| `bitsandbytes` install fails | Use `SWAP_MODE = True` |
| OOM with both models | Use `SWAP_MODE = True`; close Chrome/other apps |
| No audio captured | Check mic permissions; `python -c "import sounddevice; print(sounddevice.query_devices())"` |
| WebSocket port in use | Change `WS_PORT` in both files |
| TranslateGemma outputs English | Try `"<2es> {text}"` as prompt |
| HuggingFace 403 | `huggingface-cli login`; accept license on HF |
| Whisper detects wrong language | Add `generate_kwargs={"language": "en"}` to stt_pipe |

---

## After This Works ✅

1. **Analyze CSV** — latency distributions, quality differences
2. **Pick winner** or keep both (4B live, 12B offline)
3. **Build projected caption display** — Spanish top / English bottom for church projector
4. **Start data collection** — yt-dlp from Stark Road YouTube
5. **Fine-tune** on church data (WSL desktop)
6. **Integrate with church AV** (ProPresenter / PowerPoint — homework TBD)
