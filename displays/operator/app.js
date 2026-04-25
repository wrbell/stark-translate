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

  // ---- bootstrap ----
  refreshPreflight();
  refreshDevices();
  refreshStatus();
  setInterval(refreshPreflight, PREFLIGHT_INTERVAL_MS);
  setInterval(refreshStatus, STATUS_INTERVAL_MS);
})();
