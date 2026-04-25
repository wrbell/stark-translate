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
    const isIdle = state === "idle";
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
    try {
      const data = await getJson("/api/preflight");
      renderChecks(data);
    } catch (e) {
      preflightMetaEl.textContent = `preflight error: ${e.message}`;
    }
  }

  // ---- mic + output devices ----
  const outputSelect = document.getElementById("output-device");
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
      const data = await getJson("/api/devices");
      // mic
      micSelect.innerHTML = '<option value="">auto-detect</option>';
      for (const d of data.inputs || []) {
        const opt = document.createElement("option");
        opt.value = d.index;
        opt.textContent = `${d.index}: ${d.name} (${d.channels}ch)`;
        micSelect.appendChild(opt);
      }
      // outputs (preview only — wiring to PiperTTSEngine is deferred to 9.4.1)
      if (outputSelect) {
        outputSelect.innerHTML = '<option value="">system default</option>';
        for (const d of data.outputs || []) {
          const opt = document.createElement("option");
          opt.value = d.index;
          opt.textContent = `${d.index}: ${d.name} (${d.channels}ch)`;
          outputSelect.appendChild(opt);
        }
      }
      if (showChangeToast) {
        const counts = `${(data.inputs || []).length} in / ${(data.outputs || []).length} out`;
        showToast(`Audio devices changed — ${counts}. Confirm your mic is still selected.`);
      }
      knownChangeSeq = data.change_seq || knownChangeSeq;
    } catch (e) {
      // 503 if sounddevice unavailable — leave the placeholder option
    }
  }

  // ---- session status ----
  function renderStatus(snap) {
    currentState = snap.state || "idle";
    setStatePill(currentState);
    updateButtonsForState(currentState);
    statusDetailEl.textContent = JSON.stringify(snap, null, 2);
  }

  async function refreshStatus() {
    try {
      const snap = await getJson("/api/session/status");
      renderStatus(snap);
    } catch (e) {
      statusDetailEl.textContent = `status error: ${e.message}`;
    }
  }

  // ---- start / stop ----
  function readForm() {
    const fd = new FormData(form);
    const body = {
      lang: fd.get("lang"),
      backend: fd.get("backend"),
      engine: fd.get("engine"),
      tts: fd.get("tts") === "on",
      run_ab: fd.get("run_ab") === "on",
      vad_threshold: Number(fd.get("vad_threshold")),
      log_level: "INFO",
    };
    const mic = fd.get("mic_device");
    if (mic) body.mic_device = Number(mic);
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
      renderStatus(snap);
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

  function renderMetrics(snap) {
    const r = snap.resources || {};
    const lat = snap.latency || {};

    const vramSeries = r.vram_mib_recent || [];
    const cpuSeries = r.cpu_percent_recent || [];
    drawSparkline(vramSpark, vramSeries, { color: "#2563aa", fill: "rgba(37,99,170,0.08)" });
    drawSparkline(cpuSpark, cpuSeries, { color: "#c89a16", fill: "rgba(200,154,22,0.08)", min: 0, max: 100 });

    metricVramEl.textContent = r.vram_mib_current ? Math.round(r.vram_mib_current) : "—";
    metricCpuEl.textContent = r.cpu_percent_current != null ? r.cpu_percent_current.toFixed(1) : "—";

    if (lat.n) {
      latencyHistory.push(lat.total_ms_p50);
      if (latencyHistory.length > 60) latencyHistory.shift();
      confidenceHistory.push(lat.confidence_mean);
      if (confidenceHistory.length > 60) confidenceHistory.shift();
      metricLatencyEl.textContent = `${Math.round(lat.total_ms_p50)} / ${Math.round(lat.total_ms_p95)}`;
      metricConfidenceEl.textContent = lat.confidence_mean.toFixed(2);
    } else {
      metricLatencyEl.textContent = "— / —";
      metricConfidenceEl.textContent = "—";
    }
    drawSparkline(latencySpark, latencyHistory, { color: "#2f6b1a", fill: "rgba(47,107,26,0.08)" });
    drawSparkline(confidenceSpark, confidenceHistory, { color: "#8a4500", min: 0, max: 1 });

    metricsMeta.textContent = `uptime ${Math.round(snap.uptime_s || 0)}s · queue ${snap.queue_depth} · errors ${snap.error_count}`;

    // Audio hotplug detection — re-fetch device list when the watcher's
    // change_seq counter advances.
    if (snap.audio && typeof snap.audio.change_seq === "number" && snap.audio.change_seq > knownChangeSeq) {
      refreshDevices(true);
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
        result_keys: task.result ? Object.keys(task.result) : null,
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
