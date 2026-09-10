// Operator UI controller — Phase 9.1.
// Polls /api/preflight and /api/session/status; wires Start/Stop buttons to
// /api/session/{start,stop}. No framework, no build step.

(function () {
  "use strict";

  const PREFLIGHT_INTERVAL_MS = 4000;
  const STATUS_INTERVAL_MS = 1500;

  const checksEl = document.getElementById("checks");
  const preflightMetaEl = document.getElementById("preflight-meta");
  const statePillEl = document.getElementById("state-pill");
  const statusDetailEl = document.getElementById("status-detail");
  const startBtn = document.getElementById("start-btn");
  const stopBtn = document.getElementById("stop-btn");
  const micSelect = document.getElementById("mic-device");
  const form = document.getElementById("config-form");
  const idleEditedFields = new Set();

  let preflightOk = false;
  let currentState = "idle";

  // ---- helpers ----
  async function getJson(url) {
    const resp = await fetch(url);
    if (!resp.ok) throw new Error(`${url} -> ${resp.status}`);
    return resp.json();
  }
  async function postJson(url, body) {
    const resp = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: body ? JSON.stringify(body) : "{}",
    });
    const data = await resp.json().catch(() => ({}));
    if (!resp.ok) {
      const detail = data.detail || resp.statusText;
      throw new Error(`${url} -> ${resp.status}: ${detail}`);
    }
    return data;
  }

  function setStatePill(state) {
    statePillEl.textContent = state;
    statePillEl.className = "state-pill " + state;
  }

  const pauseBtn = document.getElementById("pause-btn");
  const resumeBtn = document.getElementById("resume-btn");
  const flipBtn = document.getElementById("flip-btn");
  const fallbackBtn = document.getElementById("fallback-btn");

  function updateButtonsForState(state) {
    const isIdle = state === "idle" || state === "error";
    const isRunning = state === "running";
    const isPaused = state === "paused";
    startBtn.disabled = !isIdle || !preflightOk;
    stopBtn.disabled = isIdle || state === "stopping";
    pauseBtn.disabled = !isRunning;
    resumeBtn.disabled = !isPaused;
    flipBtn.disabled = !isRunning;
    fallbackBtn.disabled = !isRunning;
  }

  // ---- preflight ----
  let preflightRequest = 0;
  function renderChecks(payload) {
    checksEl.innerHTML = "";
    for (const c of payload.checks) {
      const li = document.createElement("li");
      const dot = document.createElement("div");
      dot.className = "dot " + c.status;
      const body = document.createElement("div");
      const name = document.createElement("div");
      name.className = "name";
      name.textContent = c.name;
      const detail = document.createElement("div");
      detail.className = "detail";
      detail.textContent = c.detail;
      body.appendChild(name);
      body.appendChild(detail);
      li.appendChild(dot);
      li.appendChild(body);
      checksEl.appendChild(li);
    }
    const counts = payload.status_counts;
    preflightMetaEl.textContent = `${counts.pass} pass · ${counts.warn} warn · ${counts.fail} fail`;
    preflightOk = !!payload.ok;
    updateButtonsForState(currentState);
  }

  async function refreshPreflight() {
    const request = ++preflightRequest;
    try {
      const selected = readForm();
      const query = new URLSearchParams({backend: selected.backend, lang: selected.lang,
        tts: String(selected.tts), diarize: String(selected.diarize)});
      if (selected.mic_device != null) query.set("input_device", String(selected.mic_device));
      const data = await getJson(`/api/preflight?${query}`);
      if (request === preflightRequest) renderChecks(data);
    } catch (e) {
      if (request === preflightRequest) preflightMetaEl.textContent = `preflight error: ${e.message}`;
    }
  }

  // ---- mic + output devices ----
  const outputSelect = document.getElementById("output-device");
  const languageOutputs = ["en", "es"].map(lang => document.getElementById(`output-device-${lang}`));
  const ttsModeSelect = form.elements.namedItem("tts_output_mode");
  const ttsEnabled = form.elements.namedItem("tts");
  const ttsStorageKey = "stark-translate-tts-outputs";

  function readTtsRoute(control) {
    return control.dataset.deviceIndex === control.value ? Number(control.value) : control.value;
  }

  function populateOutput(select, devices, byName) {
    const selected = select.value;
    select.innerHTML = "";
    select.add(new Option(byName ? "use fallback output" : "system default", ""));
    for (const d of devices) {
      const value = String(byName ? d.name : d.index);
      if (!Array.from(select.options).some(opt => opt.value === value)) {
        select.add(new Option(`${d.index}: ${d.name} (${d.channels}ch)${d.default ? " — default" : ""}`, value));
      }
    }
    if (selected && !Array.from(select.options).some(opt => opt.value === selected)) {
      select.add(new Option(`${selected} (unavailable)`, selected));
    }
    select.value = selected;
  }

  // Keep language routes by name across reloads and USB device renumbering.
  try {
    const saved = JSON.parse(localStorage.getItem(ttsStorageKey) || "{}");
    for (const select of [outputSelect, ...languageOutputs]) {
      if (["string", "number"].includes(typeof saved[select.name])) {
        const value = String(saved[select.name]);
        if (value) select.add(new Option(value, value));
        select.value = value;
        if (typeof saved[select.name] === "number") select.dataset.deviceIndex = value;
        idleEditedFields.add(select.name);
      }
    }
    if (["ws", "wav", "both", "local"].includes(saved.tts_output_mode)) {
      ttsModeSelect.value = saved.tts_output_mode;
      idleEditedFields.add(ttsModeSelect.name);
    }
    if (typeof saved.tts === "boolean") {
      ttsEnabled.checked = saved.tts;
      idleEditedFields.add(ttsEnabled.name);
    }
  } catch (e) { /* Storage may be disabled; session controls still work. */ }

  function persistTtsChoices() {
    const saved = {tts: ttsEnabled.checked};
    for (const control of [outputSelect, ...languageOutputs, ttsModeSelect]) {
      saved[control.name] = readTtsRoute(control);
    }
    try { localStorage.setItem(ttsStorageKey, JSON.stringify(saved)); } catch (e) { /* optional storage */ }
  }
  for (const control of [outputSelect, ...languageOutputs, ttsModeSelect, ttsEnabled]) {
    control.addEventListener("change", () => {
      delete control.dataset.deviceIndex;
      persistTtsChoices();
    });
  }
  const toastEl = document.getElementById("toast");
  let knownChangeSeq = 0;
  let toastTimer = null;

  function showToast(msg) {
    if (!toastEl) return;
    toastEl.textContent = msg;
    toastEl.hidden = false;
    toastEl.classList.remove("fade");
    if (toastTimer) clearTimeout(toastTimer);
    toastTimer = setTimeout(() => {
      toastEl.classList.add("fade");
      setTimeout(() => { toastEl.hidden = true; }, 220);
    }, 4500);
  }

  async function refreshDevices(showChangeToast) {
    try {
      const [data, outputs] = await Promise.all([
        getJson("/api/devices"),
        getJson("/api/audio/output-devices"),
      ]);
      // mic
      const selectedMic = micSelect.value;
      micSelect.innerHTML = '<option value="">auto-detect</option>';
      for (const d of data.inputs || []) {
        const opt = document.createElement("option");
        opt.value = d.index;
        opt.textContent = `${d.index}: ${d.name} (${d.channels}ch)`;
        micSelect.appendChild(opt);
      }
      if (selectedMic && !Array.from(micSelect.options).some(opt => opt.value === selectedMic)) {
        micSelect.add(new Option(`${selectedMic} (unavailable)`, selectedMic));
      }
      micSelect.value = selectedMic;
      populateOutput(outputSelect, outputs.outputs || [], false);
      for (const select of languageOutputs) {
        populateOutput(select, outputs.outputs || [], true);
      }
      if (showChangeToast) {
        const counts = `${(data.inputs || []).length} in / ${(data.outputs || []).length} out`;
        showToast(`Audio devices changed — ${counts}. Check your mic and TTS outputs.`);
      }
      knownChangeSeq = data.change_seq || knownChangeSeq;
    } catch (e) {
      // 503 if sounddevice unavailable — leave the placeholder option
    }
  }

  // ---- session status ----
  const languageSelect = form.elements.namedItem("lang");
  const activeStates = ["starting", "running", "paused", "stopping"];
  let statusSeen = false;
  let idleLanguageEdited = false;
  let statusRenderRevision = 0;
  languageSelect.addEventListener("change", () => {
    if (!activeStates.includes(currentState)) idleLanguageEdited = true;
  });
  for (const control of Array.from(form.elements)) {
    control.addEventListener("change", () => {
      if (!activeStates.includes(currentState)) idleEditedFields.add(control.name);
    });
  }

  function syncSessionConfig(config, force) {
    let changed = false;
    for (const control of Array.from(form.elements)) {
      if (control.name === "lang") continue;
      const key = control.name === "output_device" ? "tts_device" : control.name;
      if (!Object.hasOwn(config, key) || (!force && (statusSeen || idleEditedFields.has(control.name)))) continue;
      const value = config[key];
      if (["tts_device_en", "tts_device_es"].includes(key)) {
        changed = changed || control.dataset.deviceIndex !== (typeof value === "number" ? String(value) : undefined);
        if (typeof value === "number") control.dataset.deviceIndex = String(value);
        else delete control.dataset.deviceIndex;
      }
      if (control.type === "checkbox") {
        if (typeof value !== "boolean") continue;
        changed = changed || control.checked !== value;
        control.checked = value;
      } else {
        const selected = value == null ? "" : String(value);
        if (control.tagName === "SELECT" && !Array.from(control.options).some(opt => opt.value === selected)) {
          control.add(new Option(selected || "system default", selected));
        }
        changed = changed || control.value !== selected;
        control.value = selected;
      }
      idleEditedFields.delete(control.name);
    }
    return changed;
  }

  function renderStatus(snap, {forcePreflight = false} = {}) {
    ++statusRenderRevision;
    const wasActive = activeStates.includes(currentState);
    currentState = snap.state || "idle";
    const isActive = activeStates.includes(currentState);
    const confirmedLang = snap.config && snap.config.lang;
    const configChanged = snap.config ? syncSessionConfig(snap.config, isActive || wasActive) : false;
    let languageChanged = false;
    // Live status owns the direction; a volunteer's next-session choice owns
    // it while idle. Also recover the previous direction on an initial reload.
    if ((isActive || wasActive || (!statusSeen && !idleLanguageEdited)) && ["en", "es"].includes(confirmedLang)) {
      languageChanged = languageSelect.value !== confirmedLang;
      languageSelect.value = confirmedLang;
      idleLanguageEdited = false;
    }
    if (snap.config && (configChanged || !statusSeen) && (isActive || wasActive)) persistTtsChoices();
    statusSeen = true;
    for (const control of Array.from(form.elements)) control.disabled = isActive;
    setStatePill(currentState);
    updateButtonsForState(currentState);
    statusDetailEl.textContent = JSON.stringify(snap, null, 2);
    const summaryControl = document.getElementById("summary-btn");
    if (summaryControl) summaryControl.disabled = ["starting", "running", "paused", "stopping"].includes(currentState);
    window.dispatchEvent(new CustomEvent("operator-session", {detail: snap}));
    if (languageChanged || configChanged || forcePreflight) refreshPreflight();
  }

  async function refreshStatus() {
    const revision = statusRenderRevision;
    try {
      const snap = await getJson("/api/session/status");
      // A control response may have confirmed a newer direction meanwhile.
      if (revision === statusRenderRevision) renderStatus(snap);
    } catch (e) {
      if (revision === statusRenderRevision) statusDetailEl.textContent = `status error: ${e.message}`;
    }
  }

  // ---- start / stop ----
  function readForm() {
    // Active controls are disabled, so FormData would silently omit them.
    const value = name => form.elements.namedItem(name).value;
    const checked = name => form.elements.namedItem(name).checked;
    const body = {
      lang: value("lang"),
      backend: value("backend"),
      engine: value("engine"),
      tts: checked("tts"),
      run_ab: checked("run_ab"),
      diarize: checked("diarize"),
      vad_threshold: Number(value("vad_threshold")),
      log_level: "INFO",
    };
    const mic = value("mic_device");
    if (mic) body.mic_device = Number(mic);
    const ttsMode = value("tts_output_mode");
    if (ttsMode) body.tts_output_mode = ttsMode;
    const ttsDevice = value("output_device");
    if (ttsDevice) body.tts_device = Number(ttsDevice);
    for (const lang of ["en", "es"]) {
      const device = value(`tts_device_${lang}`);
      if (device) body[`tts_device_${lang}`] = readTtsRoute(form.elements.namedItem(`tts_device_${lang}`));
    }
    return body;
  }

  startBtn.addEventListener("click", async () => {
    startBtn.disabled = true;
    try {
      const snap = await postJson("/api/session/start", readForm());
      renderStatus(snap);
    } catch (e) {
      statusDetailEl.textContent = `start error: ${e.message}`;
      startBtn.disabled = false;
    }
  });

  stopBtn.addEventListener("click", async () => {
    stopBtn.disabled = true;
    try {
      const snap = await postJson("/api/session/stop");
      renderStatus(snap);
    } catch (e) {
      statusDetailEl.textContent = `stop error: ${e.message}`;
    }
  });

  async function controlClick(url, body, btn) {
    btn.disabled = true;
    try {
      const snap = await postJson(url, body);
      renderStatus(snap, {forcePreflight: url === "/api/control/lang_flip"});
    } catch (e) {
      statusDetailEl.textContent = `${url} error: ${e.message}`;
    }
  }

  pauseBtn.addEventListener("click", () => controlClick("/api/control/pause", null, pauseBtn));
  resumeBtn.addEventListener("click", () => controlClick("/api/control/resume", null, resumeBtn));
  flipBtn.addEventListener("click", () => controlClick("/api/control/lang_flip", null, flipBtn));
  fallbackBtn.addEventListener("click", () => controlClick("/api/control/fallback", { engine: "hf" }, fallbackBtn));

  // ---- live metrics over /ws/control ----
  const vramSpark = document.getElementById("spark-vram");
  const cpuSpark = document.getElementById("spark-cpu");
  const latencySpark = document.getElementById("spark-latency");
  const confidenceSpark = document.getElementById("spark-confidence");
  const metricsMeta = document.getElementById("metrics-meta");
  const metricVramEl = document.getElementById("metric-vram");
  const metricCpuEl = document.getElementById("metric-cpu");
  const metricLatencyEl = document.getElementById("metric-latency");
  const metricConfidenceEl = document.getElementById("metric-confidence");

  const latencyHistory = [];
  const confidenceHistory = [];
  let metricsCohort = "";

  function renderMetrics(snap) {
    const r = snap.resources || {};
    const lat = snap.latency || {};

    const vramSeries = r.vram_mib_recent || [];
    const cpuSeries = r.cpu_percent_recent || [];
    drawSparkline(vramSpark, vramSeries, { color: "#2563aa", fill: "rgba(37,99,170,0.08)" });
    drawSparkline(cpuSpark, cpuSeries, { color: "#c89a16", fill: "rgba(200,154,22,0.08)", min: 0, max: 100 });

    metricVramEl.textContent = r.vram_mib_current ? Math.round(r.vram_mib_current) : "—";
    metricCpuEl.textContent = r.cpu_percent_current != null ? r.cpu_percent_current.toFixed(1) : "—";

    const cohort = `${snap.session_id}:${lat.timing_schema_version}:${lat.basis}`;
    if (cohort !== metricsCohort) {
      latencyHistory.length = 0;
      confidenceHistory.length = 0;
      metricsCohort = cohort;
    }
    if (lat.total_ms_p50 != null) {
      latencyHistory.push(lat.total_ms_p50);
      if (latencyHistory.length > 60) latencyHistory.shift();
    }
    if (lat.confidence_mean != null) {
      confidenceHistory.push(lat.confidence_mean);
      if (confidenceHistory.length > 60) confidenceHistory.shift();
    }
    metricLatencyEl.textContent = lat.total_ms_p50 != null
      ? `${Math.round(lat.total_ms_p50)} / ${Math.round(lat.total_ms_p95)}` : "— / —";
    metricConfidenceEl.textContent = lat.confidence_mean != null ? lat.confidence_mean.toFixed(2) : "—";
    metricLatencyEl.title = lat.basis === "speech_end_to_final_ms"
      ? "Estimated speech end to final payload readiness (server); browser display timing is separate" : "Historical pipeline timing; not speech-end latency";
    drawSparkline(latencySpark, latencyHistory, { color: "#2f6b1a", fill: "rgba(47,107,26,0.08)" });
    drawSparkline(confidenceSpark, confidenceHistory, { color: "#8a4500", min: 0, max: 1 });

    metricsMeta.textContent = `uptime ${Math.round(snap.uptime_s || 0)}s · queue ${snap.queue_depth} · errors ${snap.error_count}`;

    // Audio hotplug detection — re-fetch device list when the watcher's
    // change_seq counter advances.
    if (snap.audio && typeof snap.audio.change_seq === "number" && snap.audio.change_seq > knownChangeSeq) {
      refreshDevices(true);
    }

    // Live diarization (Phase 9.6.1) — current speaker pill + caption view.
    const speakerEl = document.getElementById("metric-speaker");
    const speakerDetailEl = document.getElementById("metric-speaker-detail");
    const captionViewEl = document.getElementById("caption-view");
    const diar = (snap.audio && snap.audio.diarization) || null;
    if (speakerEl && speakerDetailEl) {
      if (diar && diar.current_speaker) {
        speakerEl.textContent = diar.current_speaker;
        const transitions = diar.transitions || 0;
        const recent = (diar.recent || []).length;
        speakerDetailEl.textContent = `${transitions} transitions · ${recent} recent labels`;
      } else {
        speakerEl.textContent = "—";
        speakerDetailEl.textContent = "no diarization data yet";
      }
    }
    if (captionViewEl) {
      const captions = (diar && diar.captions && diar.captions.length)
        ? diar.captions
        : (diar && diar.recent) || [];
      if (!captions.length) {
        captionViewEl.innerHTML = '<li class="empty">no captions yet</li>';
      } else {
        captionViewEl.innerHTML = "";
        for (const rec of captions.slice(-8)) {
          const li = document.createElement("li");
          const spk = rec.speaker || rec.current_speaker;
          const text = rec.english || "";
          if (spk) {
            const tag = document.createElement("span");
            tag.className = "spk";
            tag.textContent = spk + ":";
            li.appendChild(tag);
          }
          li.appendChild(document.createTextNode(text ? " " + text : ""));
          captionViewEl.appendChild(li);
        }
      }
    }
  }

  let metricsWs = null;
  let metricsBackoff = 1000;
  function connectMetrics() {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    const url = `${proto}://${location.host}/ws/control`;
    metricsWs = new WebSocket(url);
    metricsWs.onopen = () => {
      metricsBackoff = 1000;
      metricsMeta.textContent = "connected";
    };
    metricsWs.onmessage = (event) => {
      try {
        renderMetrics(JSON.parse(event.data));
      } catch (e) {
        // ignore malformed frames
      }
    };
    metricsWs.onclose = () => {
      metricsMeta.textContent = `disconnected — retrying in ${metricsBackoff}ms`;
      setTimeout(connectMetrics, metricsBackoff);
      metricsBackoff = Math.min(metricsBackoff * 2, 15000);
    };
    metricsWs.onerror = () => {
      try { metricsWs.close(); } catch (e) {}
    };
  }

  // ---- features (Phase 9.6) ----
  const versesListEl = document.getElementById("verses-list");
  const summaryBtn = document.getElementById("summary-btn");
  const summaryStatusEl = document.getElementById("summary-status");
  let lastVerseChunk = -1;
  let summaryPollTimer = null;

  function renderVerses(highlights) {
    if (!versesListEl) return;
    if (!highlights || highlights.length === 0) {
      versesListEl.innerHTML = '<li class="empty">none yet</li>';
      return;
    }
    versesListEl.innerHTML = "";
    for (const h of highlights.slice(-25).reverse()) {
      const li = document.createElement("li");
      const ref = document.createElement("span");
      ref.className = "ref";
      ref.textContent = h.reference;
      const ctx = document.createElement("span");
      ctx.className = "ctx";
      ctx.textContent = h.context || "";
      li.appendChild(ref);
      li.appendChild(ctx);
      versesListEl.appendChild(li);
      lastVerseChunk = Math.max(lastVerseChunk, h.chunk_id || 0);
    }
  }

  async function refreshVerses() {
    try {
      const data = await getJson("/api/features/verses");
      renderVerses(data.highlights || []);
    } catch (e) {
      // ignore errors during idle state
    }
  }

  async function pollSummary(taskId) {
    try {
      const task = await getJson(`/api/features/summary/${taskId}`);
      summaryStatusEl.textContent = JSON.stringify({
        state: task.state,
        return_code: task.return_code,
        error: task.error,
        result: task.result || null,
      }, null, 2);
      if (task.state === "done" || task.state === "error") {
        if (summaryPollTimer) { clearInterval(summaryPollTimer); summaryPollTimer = null; }
        summaryBtn.disabled = false;
      }
    } catch (e) {
      summaryStatusEl.textContent = `poll error: ${e.message}`;
    }
  }

  summaryBtn.addEventListener("click", async () => {
    summaryBtn.disabled = true;
    summaryStatusEl.textContent = "submitting…";
    try {
      const task = await postJson("/api/features/summary", {});
      summaryStatusEl.textContent = `task ${task.task_id} submitted (state=${task.state})`;
      if (summaryPollTimer) clearInterval(summaryPollTimer);
      summaryPollTimer = setInterval(() => pollSummary(task.task_id), 2000);
    } catch (e) {
      summaryStatusEl.textContent = `error: ${e.message}`;
      summaryBtn.disabled = false;
    }
  });

  form.addEventListener("change", refreshPreflight);

  // ---- bootstrap ----
  refreshPreflight();
  refreshDevices(false);
  refreshStatus();
  refreshVerses();
  connectMetrics();
  setInterval(refreshPreflight, PREFLIGHT_INTERVAL_MS);
  setInterval(refreshStatus, STATUS_INTERVAL_MS);
  setInterval(refreshVerses, 5000);
})();
