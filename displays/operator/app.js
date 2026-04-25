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

  function updateButtonsForState(state) {
    const isIdle = state === "idle";
    startBtn.disabled = !isIdle || !preflightOk;
    stopBtn.disabled = isIdle || state === "stopping";
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

  // ---- mic devices ----
  async function refreshDevices() {
    try {
      const data = await getJson("/api/devices");
      micSelect.innerHTML = '<option value="">auto-detect</option>';
      for (const d of data.inputs || []) {
        const opt = document.createElement("option");
        opt.value = d.index;
        opt.textContent = `${d.index}: ${d.name} (${d.channels}ch)`;
        micSelect.appendChild(opt);
      }
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

  // ---- bootstrap ----
  refreshPreflight();
  refreshDevices();
  refreshStatus();
  connectMetrics();
  setInterval(refreshPreflight, PREFLIGHT_INTERVAL_MS);
  setInterval(refreshStatus, STATUS_INTERVAL_MS);
})();
