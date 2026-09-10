// Operator UI controller — layperson layout (Prepare / Live / Sessions / Help / Advanced).
//
// Talks to the FastAPI control plane in operator_app/main.py. Everything beyond
// the v2026.6 contract (/api/capabilities, /api/storage, /api/support/*,
// /api/audio/test-*, readiness/health/outcome/work fields on
// /api/session/status) is optional: a 404/405 or a missing field degrades to
// "not available" text and never to simulated success.
//
// `StarkOperator.create(deps)` wires the page; the browser bootstraps it at
// load time, and the Node tests inject a fake document/fetch instead.

(function (global) {
  "use strict";

  const PREFLIGHT_INTERVAL_MS = 4000;
  const STATUS_INTERVAL_MS = 1500;
  const VERSE_INTERVAL_MS = 5000;
  const AGE_TICK_MS = 1000;
  const STALE_AFTER_MS = 6000;
  const REQUEST_TIMEOUT_MS = 5000;
  const CONTROL_ACK_TIMEOUT_MS = 15000;
  const ACTIVE_STATES = ["starting", "running", "paused", "stopping"];
  // Product profiles installed code can select. Selecting one is not a claim
  // that its models are installed or its hardware certified: readiness decides.
  const KNOWN_PROFILES = {
    standard: {
      label: "Standard (full models for this computer)",
      description: "Uses the models installed for this computer's standard setup.",
    },
    "lite-cpu": {
      label: "Lite — no graphics card",
      description: "Small speech model and fast processor-only translation for every caption. Needs about 8 GB of memory.",
    },
    "lite-cpu-quality": {
      label: "Lite quality — no graphics card, slower final captions",
      description: "Adds a slower processor-only model that improves the final translation. Needs about 16 GB of memory.",
    },
    "lite-cuda-8gb": {
      label: "Lite — 8 GB NVIDIA graphics card",
      description: "For an 8 GB NVIDIA card (original RTX 2070 class). Readiness checks the card; this hardware has not been certified yet.",
    },
  };
  const LEGACY_PROFILE_IDS = {full: "standard"};
  const LANG_TEXT = {en: "English speaker → Spanish captions", es: "Spanish speaker → English captions"};
  const CHECK_TEXT = {
    GPU: {title: "Computer hardware", pass: "Graphics acceleration is available on this computer.", warn: "No graphics acceleration was found. Captions will work but arrive more slowly."},
    "Runtime dependencies": {title: "Software installation", pass: "The required software is installed.", fail: "The installation on this computer is incomplete. Ask the setup owner."},
    Models: {title: "Language models", pass: "Speech recognition and translation models are installed.", fail: "Language models are missing on this computer. Ask the setup owner to run setup for this profile."},
    Microphone: {title: "Microphone", fail: "Plug in the USB microphone, then click Check again."},
    "Adapter manifest": {
      title: "Custom vocabulary (optional)",
      warn: "No custom vocabulary is installed; the standard models are used.",
      fail: "The custom vocabulary file is damaged. Ask the setup owner.",
    },
    "llama-server": {title: "Translation server", warn: "The translation server is not running. Captions will use a slower path."},
    "Managed llama-server": {title: "Translation server (managed)", fail: "The managed translation server is not installed for this profile. Ask the setup owner."},
    "Lite hardware": {title: "Computer memory and cores", fail: "This computer is below the memory or processor floor for the chosen profile."},
    "Lite CUDA": {title: "NVIDIA graphics card", fail: "The NVIDIA card could not be verified for this profile."},
    Diarization: {title: "Speaker labels", fail: "Speaker labels are not available with a Lite profile. Turn them off under Advanced."},
  };
  const PHASE_TEXT = {
    loading_models: "loading language models",
    loading: "loading language models",
    warming: "warming up",
    warmup: "warming up",
    listening: "listening for speech",
    ready: "ready",
    paused: "paused",
    starting: "starting",
    stopping: "stopping",
    input_error: "microphone problem — no sound can be captured",
    unknown: "status unknown",
    idle: "not running",
  };

  // ---- pure helpers (exported for tests) ----------------------------------

  function friendlyCheck(check) {
    const known = CHECK_TEXT[check.name] || {};
    return {
      name: check.name,
      status: check.status,
      detail: check.detail || "",
      title: known.title || check.name,
      advice: known[check.status] || "",
    };
  }

  function countChecks(payload) {
    const counts = {pass: 0, warn: 0, fail: 0};
    const given = payload && payload.status_counts;
    if (given && typeof given === "object") {
      for (const key of Object.keys(counts)) counts[key] = Number(given[key]) || 0;
      return counts;
    }
    for (const check of (payload && payload.checks) || []) {
      if (Object.hasOwn(counts, check.status)) counts[check.status] += 1;
    }
    return counts;
  }

  function readinessSummary(payload) {
    if (!payload || !Array.isArray(payload.checks)) {
      return {ok: false, tone: "pending", counts: {pass: 0, warn: 0, fail: 0}, text: "Readiness has not been checked yet."};
    }
    const counts = countChecks(payload);
    const ok = payload.ok === undefined ? counts.fail === 0 : !!payload.ok;
    if (ok) {
      const text = counts.warn
        ? `Ready to start — ${counts.warn} warning${counts.warn === 1 ? "" : "s"} to be aware of.`
        : "Ready to start.";
      return {ok: true, tone: "ok", counts, text};
    }
    const fails = counts.fail || 1;
    return {ok: false, tone: "fail", counts, text: `${fails} problem${fails === 1 ? "" : "s"} to fix before starting.`};
  }

  function firstLine(text) {
    return String(text || "").split("\n")[0].trim();
  }

  function awaitingFirstHealth(readiness, hasReportedHealth) {
    return !!readiness && readiness.stale === true && !hasReportedHealth &&
      readiness.updated_at == null && readiness.age_s == null && readiness.phase !== "input_error";
  }

  function runningReadiness(readiness, hasReportedHealth) {
    if (!readiness || typeof readiness !== "object") return null;
    const phase = String(readiness.phase || "").toLowerCase();
    const reason = firstLine(readiness.reason);
    const extra = reason ? ` ${reason}` : "";
    if (awaitingFirstHealth(readiness, hasReportedHealth)) {
      return {label: "Waiting for status", tone: "busy", detail: "Waiting for the caption process to report its first status. Stop is available if you need it."};
    }
    if (readiness.stale === true) {
      return {label: "Needs attention", tone: "bad", detail: `Status from the caption process is out of date.${extra} Use Stop, then start the session again.`};
    }
    if (phase === "input_error") {
      return {label: "Needs attention", tone: "bad", detail: `Sound input is unavailable.${extra} Check the selected input and microphone permission. Use Stop to restart.`};
    }
    if (readiness.ready === false) {
      if (["loading", "loading_models", "warming", "warmup", "starting"].includes(phase)) {
        return {label: "Getting ready…", tone: "busy", detail: `Getting ready: ${humanPhase(phase)}.${extra} Stop is available if you need it.`};
      }
      return {label: "Waiting for input", tone: "warn", detail: `Captions are not ready yet.${extra} Check the selected sound input. Stop is available if you need it.`};
    }
    return null; // Legacy snapshots without readiness keep their existing display.
  }

  function describeState(input) {
    const state = input.state || "idle";
    if (input.connection === "stale") {
      const known = describeState({...input, connection: "ok"});
      return {label: "Not connected", tone: "stale", detail: `Last known: ${known.label.replace(/…$/, "")}.`};
    }
    switch (state) {
      case "starting":
        return {label: "Starting…", tone: "busy", detail: "Loading language models. This can take a minute."};
      case "running":
        return runningReadiness(input.readiness, input.hasReportedHealth) || {label: "Live", tone: "ok", detail: "Captions are being sent."};
      case "paused":
        return {label: "Paused", tone: "warn", detail: "Captions are paused."};
      case "stopping":
        return {label: "Stopping…", tone: "busy", detail: "Finishing up."};
      case "error":
        return {label: "Needs attention", tone: "bad", detail: firstLine(input.error) || "The caption process reported a problem."};
      default: {
        if (input.work && input.work.kind) {
          return {label: "Busy", tone: "busy", detail: `The operator service is ${humanWork(input.work)}.`};
        }
        const outcome = input.outcome;
        if (outcome === "failed") return {label: "Stopped", tone: "warn", detail: "The last session failed. See the Help tab."};
        if (outcome === "interrupted") return {label: "Stopped", tone: "neutral", detail: "The last session was interrupted."};
        if (input.preflightOk) {
          return {label: "Ready", tone: "ok", detail: outcome === "completed" ? "The last session finished normally." : "Ready to start captions."};
        }
        return {label: "Not ready", tone: "neutral", detail: input.preflightChecked ? "Fix the red items on the Prepare tab." : "Checking this computer…"};
      }
    }
  }

  function humanWork(work) {
    if (!work || typeof work !== "object" || !work.kind) return "busy";
    const kind = String(work.kind).replace(/[_-]+/g, " ");
    return `busy with ${kind}${work.id ? ` (${work.id})` : ""}`;
  }

  function humanPhase(phase) {
    if (!phase) return "getting ready";
    const key = String(phase).toLowerCase();
    return PHASE_TEXT[key] || key.replace(/[_-]+/g, " ");
  }

  function describeError(error) {
    const detail = error && error.detail;
    const info = {status: error && error.status, code: null, message: "", checks: null, work: null, raw: detail};
    if (detail && typeof detail === "object" && !Array.isArray(detail)) {
      info.code = detail.code || null;
      info.message = detail.message || detail.detail || JSON.stringify(detail);
      if (Array.isArray(detail.checks)) info.checks = detail.checks;
      if (detail.work && typeof detail.work === "object") info.work = detail.work;
    } else if (Array.isArray(detail)) {
      info.message = detail.map(item => (item && item.msg) || JSON.stringify(item)).join("; ");
    } else if (typeof detail === "string" && detail) {
      info.message = detail;
    } else {
      info.message = (error && error.message) || "Unknown error";
    }
    return info;
  }

  function isLocalHost(host) {
    return /^(localhost|127(\.\d+){3}|\[::1\]|::1|0\.0\.0\.0)$/i.test(String(host || ""));
  }

  function hostOf(url) {
    const match = /^[a-z]+:\/\/(\[[^\]]+\]|[^/:?#]+)/i.exec(String(url || ""));
    return match ? match[1] : "";
  }

  // Server-advertised audience URLs win; otherwise derive from the page host.
  function audienceLinks(location, capabilities) {
    const caps = capabilities || {};
    const urls = caps.audience_urls && typeof caps.audience_urls === "object" ? caps.audience_urls : {};
    const ports = caps.display_ports && typeof caps.display_ports === "object" ? caps.display_ports : {};
    const host = (location && location.hostname) || "localhost";
    const httpPort = Number(ports.http) || 8080;
    const wsPort = Number(ports.websocket) || 8765;
    const query = wsPort !== 8765 ? `?port=${wsPort}` : "";
    const audience = urls.audience || `http://${host}:${httpPort}/displays/audience_display.html${query}`;
    const mobile = urls.mobile || `http://${host}:${httpPort}/displays/mobile_display.html${query}`;
    const shareHost = hostOf(mobile) || host;
    const shareable = !isLocalHost(shareHost);
    const note = shareable
      ? "Phones on the same Wi-Fi can open this link while captions are running."
      : "This link only works on this computer. Open the network audience-display bookmark prepared by the setup owner, then click its header for the phone QR code.";
    return {audience, mobile, church: urls.church || null, obs: urls.obs || null, shareable, note, host: shareHost, httpPort, wsPort};
  }

  function formatAge(seconds) {
    if (seconds == null || !Number.isFinite(seconds) || seconds < 0) return "unknown";
    if (seconds < 2) return "just now";
    if (seconds < 90) return `${Math.round(seconds)} s ago`;
    if (seconds < 5400) return `${Math.round(seconds / 60)} min ago`;
    return `${(seconds / 3600).toFixed(1)} h ago`;
  }

  function formatDuration(seconds) {
    if (seconds == null || !Number.isFinite(seconds) || seconds < 0) return "";
    if (seconds < 60) return `${Math.round(seconds)} s`;
    if (seconds < 3600) return `${Math.floor(seconds / 60)} min`;
    return `${Math.floor(seconds / 3600)} h ${Math.floor((seconds % 3600) / 60)} min`;
  }

  function formatBytes(bytes) {
    if (bytes == null || !Number.isFinite(Number(bytes))) return "unknown size";
    const value = Number(bytes);
    if (value < 1024) return `${value} B`;
    if (value < 1024 * 1024) return `${(value / 1024).toFixed(0)} KB`;
    if (value < 1024 * 1024 * 1024) return `${(value / 1024 / 1024).toFixed(1)} MB`;
    return `${(value / 1024 / 1024 / 1024).toFixed(1)} GB`;
  }

  function normalizeProfileId(id) {
    const key = id == null ? "" : String(id);
    return Object.hasOwn(LEGACY_PROFILE_IDS, key) ? LEGACY_PROFILE_IDS[key] : key;
  }

  function isLiteProfile(id) {
    const key = normalizeProfileId(id);
    return !!key && key !== "standard";
  }

  function profileLabel(id) {
    const key = normalizeProfileId(id);
    return (KNOWN_PROFILES[key] && KNOWN_PROFILES[key].label) || key;
  }

  const BACKEND_TEXT = {mlx: "Apple graphics", cuda: "NVIDIA graphics card", cpu: "processor only", auto: "automatic"};
  function backendLabel(id) {
    const key = String(id || "").toLowerCase();
    return BACKEND_TEXT[key] || key;
  }

  // Accepts the real contract (a list of ids) and, for compatibility, objects
  // with {id, label, description, available}. Unknown ids keep their raw name.
  function pickProfiles(capabilities) {
    const raw = capabilities && Array.isArray(capabilities.profiles) ? capabilities.profiles : [];
    const out = [];
    const seen = new Set();
    for (const entry of raw) {
      const profile = typeof entry === "string" ? {id: entry} : entry;
      if (!profile || profile.id == null) continue;
      const id = normalizeProfileId(profile.id);
      if (!id || seen.has(id)) continue;
      seen.add(id);
      const known = KNOWN_PROFILES[id] || {};
      out.push({
        id,
        label: profile.label || known.label || id,
        description: profile.description || known.description || "",
        available: profile.available !== false,
        lite: isLiteProfile(id),
      });
    }
    return out;
  }

  function defaultProfileId(capabilities, profiles) {
    const wanted = normalizeProfileId(capabilities && capabilities.default_profile);
    if (wanted && profiles.some(p => p.id === wanted)) return wanted;
    return profiles.length ? profiles[0].id : "standard";
  }

  function profileNameOf(value) {
    if (!value) return "";
    if (typeof value === "string") return normalizeProfileId(value);
    if (typeof value === "object") return normalizeProfileId(value.name || value.profile || "");
    return "";
  }

  function levelPercent(value) {
    const number = Number(value);
    if (!Number.isFinite(number)) return null;
    return Math.max(0, Math.min(100, Math.round(number * 100)));
  }

  function describeLevel(peak, rms) {
    const percent = levelPercent(peak);
    if (percent === null) return {percent: null, text: "The test finished, but no level was reported."};
    const average = levelPercent(rms);
    const measured = `Measured peak ${percent}%${average !== null ? ` (average ${average}%)` : ""} of full scale.`;
    if (percent === 0) return {percent, text: `${measured} No sound was detected: check the microphone, its cable and the microphone permission.`};
    if (percent < 5) return {percent, text: `${measured} Very quiet: move the microphone closer to the speaker or raise its gain.`};
    if (percent > 90) return {percent, text: `${measured} Too loud and may distort: move the microphone away or lower its gain.`};
    return {percent, text: `${measured} Sound was detected at a usable level.`};
  }

  // ---- the page controller ------------------------------------------------

  function create(deps) {
    deps = deps || {};
    const doc = deps.document || global.document;
    const fetchImpl = deps.fetch || ((url, init) => global.fetch(url, init));
    const loc = deps.location || global.location || {hostname: "localhost", protocol: "http:", host: "localhost", search: ""};
    const storage = deps.localStorage !== undefined ? deps.localStorage : safeStorage();
    const setIntervalImpl = deps.setInterval || ((fn, ms) => global.setInterval(fn, ms));
    const setTimeoutImpl = deps.setTimeout || ((fn, ms) => global.setTimeout(fn, ms));
    const clearTimeoutImpl = deps.clearTimeout || (id => global.clearTimeout(id));
    const clearIntervalImpl = deps.clearInterval || (id => global.clearInterval(id));
    const now = deps.now || (() => Date.now());
    const WS = deps.WebSocket || global.WebSocket;
    const openWindow = deps.open || (url => global.open(url, "_blank", "noopener"));
    const dispatch = deps.dispatchEvent || (event => global.dispatchEvent(event));
    const EventImpl = deps.CustomEvent || global.CustomEvent;
    const clipboard = deps.clipboard !== undefined ? deps.clipboard : global.navigator && global.navigator.clipboard;
    const drawSpark = deps.drawSparkline || global.drawSparkline || (() => {});
    const qr = deps.StarkQR || global.StarkQR || null;
    const captionsLib = deps.StarkCaptions || global.StarkCaptions || null;
    const AbortControllerImpl = deps.AbortController !== undefined ? deps.AbortController : global.AbortController;
    const SearchParams = deps.URLSearchParams || global.URLSearchParams;

    function safeStorage() {
      try { return global.localStorage; } catch (e) { return null; }
    }
    function readStorage(key) {
      try { return storage ? storage.getItem(key) : null; } catch (e) { return null; }
    }
    function writeStorage(key, value) {
      try { if (storage) storage.setItem(key, value); } catch (e) { /* optional storage */ }
    }

    const $ = id => doc.getElementById(id);
    const el = {
      connectionAge: $("connection-age"), statePill: $("state-pill"), stateDetail: $("state-detail"), toast: $("toast"),
      attention: $("attention"), attentionTitle: $("attention-title"), attentionText: $("attention-text"),
      attentionAdvice: $("attention-advice"), attentionStop: $("attention-stop"), attentionDismiss: $("attention-dismiss"),
      form: $("config-form"), micSelect: $("mic-device"), micHint: $("mic-hint"), outputSelect: $("output-device"),
      micTestBtn: $("mic-test-btn"), outputTestBtn: $("output-test-btn"), micLevelBar: $("mic-level-bar"), micTestStatus: $("mic-test-status"),
      deviceNote: $("device-note"),
      readinessSummary: $("readiness-summary"), checks: $("checks"), preflightRefresh: $("preflight-refresh"), preflightMeta: $("preflight-meta"),
      profileField: $("profile-field"), profileSelect: $("profile-select"), profileHint: $("profile-hint"), liteNote: $("lite-note"),
      startHint: $("start-hint"), startBtn: $("start-btn"), startStatus: $("start-status"),
      liveTitle: $("live-title"), liveSubtitle: $("live-subtitle"), liveEvent: $("live-event"), liveLanguage: $("live-language"), liveElapsed: $("live-elapsed"),
      liveLevelBar: $("live-level-bar"), liveLevelText: $("live-level-text"),
      pauseBtn: $("pause-btn"), resumeBtn: $("resume-btn"), stopBtn: $("stop-btn"), flipBtn: $("flip-btn"), fallbackBtn: $("fallback-btn"),
      healthList: $("health-list"), pipelineReadiness: $("pipeline-readiness"),
      captionStatus: $("caption-status"), captionView: $("caption-view"), captionEndpoint: $("caption-endpoint"),
      audienceOpen: $("audience-open"), audienceCopy: $("audience-copy"), audienceUrl: $("audience-url"), audienceNote: $("audience-note"), audienceQr: $("audience-qr"),
      otherDisplays: $("other-displays"),
      versesList: $("verses-list"),
      summaryBtn: $("summary-btn"), summaryCancel: $("summary-cancel"), summaryStatus: $("summary-status"), summaryResult: $("summary-result"),
      summaryNotice: $("summary-notice"), summaryEnglish: $("summary-english"), summarySpanish: $("summary-spanish"), summaryMeta: $("summary-meta"), summaryRaw: $("summary-raw"),
      storageSummary: $("storage-summary"), storageSessions: $("storage-sessions"), storagePreview: $("storage-cleanup-preview"), storageCleanup: $("storage-cleanup"), storageStatus: $("storage-status"),
      supportSession: $("support-session"), supportText: $("support-include-text"), supportAudio: $("support-include-audio"),
      supportPreview: $("support-preview"), supportExport: $("support-export"), supportDownload: $("support-download"), supportStatus: $("support-status"), supportFiles: $("support-files"),
      metricVram: $("metric-vram"), metricCpu: $("metric-cpu"), metricLatency: $("metric-latency"), metricConfidence: $("metric-confidence"),
      metricSpeaker: $("metric-speaker"), metricSpeakerDetail: $("metric-speaker-detail"), metricsMeta: $("metrics-meta"),
      sparkVram: $("spark-vram"), sparkCpu: $("spark-cpu"), sparkLatency: $("spark-latency"), sparkConfidence: $("spark-confidence"),
      statusDetail: $("status-detail"), preflightDetail: $("preflight-detail"), capabilitiesDetail: $("capabilities-detail"),
    };
    const form = el.form;
    const formControls = () => Array.from(form.elements).filter(target => target.name && target.tagName !== "BUTTON");
    const control = name => form.elements.namedItem(name);
    const languageSelect = control("lang");
    const languageOutputs = ["en", "es"].map(lang => $(`output-device-${lang}`));
    const ttsModeSelect = control("tts_output_mode");
    const ttsEnabled = control("tts");
    const idleEditedFields = new Set();

    // Skip identical writes: several targets are aria-live regions, and a
    // rewrite every poll would make screen readers re-announce them.
    function setText(node, text) {
      if (node && node.textContent !== String(text)) node.textContent = text;
    }
    function makeOption(text, value) {
      const option = doc.createElement("option");
      option.textContent = text;
      option.value = String(value);
      return option;
    }
    function hasOption(select, value) {
      return Array.from(select.options).some(opt => opt.value === value);
    }
    function selectedLabel(select, fallback) {
      const option = Array.from(select.options).find(opt => opt.value === select.value);
      return option && select.value ? option.textContent : fallback;
    }

    // ---- tabs -------------------------------------------------------------
    const TAB_KEY = "stark-operator-tab";
    const tabs = Array.from(doc.querySelectorAll('[role="tab"]'));
    let activeTab = "prepare";
    function showTab(name, options) {
      const opts = options || {};
      if (!tabs.some(tab => tab.dataset.tab === name)) return;
      activeTab = name;
      for (const tab of tabs) {
        const selected = tab.dataset.tab === name;
        tab.setAttribute("aria-selected", selected ? "true" : "false");
        tab.tabIndex = selected ? 0 : -1;
        const panel = $(tab.getAttribute("aria-controls"));
        if (panel) panel.hidden = !selected;
        if (selected && opts.focus) tab.focus();
      }
      if (opts.persist !== false) writeStorage(TAB_KEY, name);
      if (name === "sessions") refreshStorage();
      if (name === "help") refreshSupportSessions();
    }
    for (const tab of tabs) tab.addEventListener("click", () => showTab(tab.dataset.tab));
    const tablist = doc.querySelector('[role="tablist"]');
    if (tablist) {
      tablist.addEventListener("keydown", event => {
        const index = tabs.findIndex(tab => tab.dataset.tab === activeTab);
        let next = null;
        if (event.key === "ArrowRight") next = (index + 1) % tabs.length;
        else if (event.key === "ArrowLeft") next = (index - 1 + tabs.length) % tabs.length;
        else if (event.key === "Home") next = 0;
        else if (event.key === "End") next = tabs.length - 1;
        if (next === null) return;
        event.preventDefault();
        showTab(tabs[next].dataset.tab, {focus: true});
      });
    }

    // ---- http helpers -----------------------------------------------------
    function withTimeout(init) {
      const options = {...(init || {})};
      if (AbortControllerImpl) {
        const controller = new AbortControllerImpl();
        options.signal = controller.signal;
        setTimeoutImpl(() => controller.abort(), REQUEST_TIMEOUT_MS);
      }
      return options;
    }
    async function getJson(url, options) {
      const resp = await fetchImpl(url, options && options.timeout ? withTimeout() : undefined);
      if (!resp.ok) {
        const err = new Error(`${url} -> ${resp.status}`);
        err.status = resp.status;
        err.detail = await resp.json().then(data => data && data.detail).catch(() => null);
        throw err;
      }
      return resp.json();
    }
    async function postJson(url, body) {
      const resp = await fetchImpl(url, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: body ? JSON.stringify(body) : "{}",
      });
      const data = await resp.json().catch(() => ({}));
      if (!resp.ok) {
        const detail = data && data.detail !== undefined ? data.detail : resp.statusText;
        const err = new Error(`${url} -> ${resp.status}: ${typeof detail === "string" ? detail : JSON.stringify(detail)}`);
        err.status = resp.status;
        err.detail = detail;
        throw err;
      }
      return data;
    }
    const isMissingEndpoint = err => err && [404, 405, 501].includes(err.status);

    // ---- toast + attention banner ----------------------------------------
    let toastTimer = null;
    function showToast(msg) {
      if (!el.toast) return;
      el.toast.textContent = msg;
      el.toast.hidden = false;
      el.toast.classList.remove("fade");
      if (toastTimer) clearTimeoutImpl(toastTimer);
      toastTimer = setTimeoutImpl(() => {
        el.toast.classList.add("fade");
        setTimeoutImpl(() => { el.toast.hidden = true; }, 220);
      }, 4500);
    }

    let attentionSource = null;
    let dismissedStateError = null;
    function showAttention(input) {
      attentionSource = input.source || "request";
      setText(el.attentionTitle, input.title || "Needs attention");
      setText(el.attentionText, input.text || "");
      if (el.attentionText) el.attentionText.hidden = !input.text;
      setText(el.attentionAdvice, input.advice || "");
      el.attention.classList.toggle("info", input.tone === "info");
      el.attention.hidden = false;
      if (el.attentionStop) el.attentionStop.hidden = attentionSource !== "state";
    }
    function clearAttention(source) {
      if (source && attentionSource !== source) return;
      attentionSource = null;
      el.attention.hidden = true;
    }
    el.attentionDismiss.addEventListener("click", () => {
      if (attentionSource === "state") dismissedStateError = currentError || "";
      clearAttention();
    });

    // ---- preflight ---------------------------------------------------------
    let preflightRequest = 0;
    let preflightOk = false;
    let preflightChecked = false;

    function renderChecks(payload) {
      const checks = Array.isArray(payload.checks) ? payload.checks : [];
      el.checks.replaceChildren();
      for (const raw of checks) {
        const c = friendlyCheck(raw);
        const li = doc.createElement("li");
        const dot = doc.createElement("div");
        dot.className = "dot " + c.status;
        const body = doc.createElement("div");
        const name = doc.createElement("div");
        name.className = "name";
        name.textContent = c.title;
        body.appendChild(name);
        if (c.advice) {
          const advice = doc.createElement("div");
          advice.className = "advice";
          advice.textContent = c.advice;
          body.appendChild(advice);
        }
        const detail = doc.createElement("div");
        detail.className = "detail";
        detail.textContent = c.detail;
        if (c.advice && c.detail) {
          const disclosure = doc.createElement("details");
          const summary = doc.createElement("summary");
          summary.textContent = "Technical details";
          disclosure.appendChild(summary);
          disclosure.appendChild(detail);
          body.appendChild(disclosure);
        } else {
          body.appendChild(detail);
        }
        li.appendChild(dot);
        li.appendChild(body);
        el.checks.appendChild(li);
      }
      const summary = readinessSummary(payload);
      preflightChecked = true;
      preflightOk = summary.ok;
      setText(el.readinessSummary, summary.text);
      el.readinessSummary.className = "readiness-summary " + summary.tone;
      const counts = summary.counts;
      const checkedProfile = profileNameOf(payload.effective_profile && payload.effective_profile.profile)
        || profileNameOf(payload.profile);
      const scope = checkedProfile
        ? ` · checked for ${profileLabel(checkedProfile)}${payload.backend ? ` on ${backendLabel(payload.backend)}` : ""}`
        : payload.backend ? ` · ${backendLabel(payload.backend)}` : "";
      setText(el.preflightMeta, `${counts.pass} pass · ${counts.warn} warn · ${counts.fail} fail${scope}`);
      setText(el.preflightDetail, JSON.stringify(payload, null, 2));
      updateButtonsForState(currentState);
      renderStatePill();
    }

    function preflightFailed(message) {
      preflightOk = false;
      preflightChecked = true;
      setText(el.readinessSummary, `Couldn't check readiness: ${message}`);
      el.readinessSummary.className = "readiness-summary fail";
      setText(el.preflightMeta, "The last readiness check did not complete.");
      updateButtonsForState(currentState);
      renderStatePill();
    }

    async function refreshPreflight() {
      const request = ++preflightRequest;
      try {
        const selected = readForm();
        const query = new SearchParams({backend: selected.backend, lang: selected.lang,
          tts: String(selected.tts), diarize: String(selected.diarize)});
        if (selected.mic_device != null) query.set("input_device", String(selected.mic_device));
        // The same profile the start request will carry, so both agree.
        if (selected.profile) query.set("profile", selected.profile);
        const data = await getJson(`/api/preflight?${query}`, {timeout: true});
        if (request === preflightRequest) renderChecks(data);
      } catch (e) {
        // A failed check must clear any earlier "ready" verdict immediately.
        if (request === preflightRequest) preflightFailed(describeError(e).message);
      }
    }
    el.preflightRefresh.addEventListener("click", () => {
      setText(el.readinessSummary, "Checking…");
      el.readinessSummary.className = "readiness-summary pending";
      refreshPreflight();
    });

    // ---- mic + output devices ---------------------------------------------
    const ttsStorageKey = "stark-translate-tts-outputs";

    function readTtsRoute(target) {
      return target.dataset.deviceIndex === target.value ? Number(target.value) : target.value;
    }

    function deviceLabel(d, duplicates) {
      const base = `${d.name} (${d.channels}ch)${d.default ? " — default" : ""}`;
      return duplicates.has(d.name) ? `${base} #${d.index}` : base;
    }
    function duplicateNames(devices) {
      const seen = new Set(), dupes = new Set();
      for (const d of devices) {
        if (seen.has(d.name)) dupes.add(d.name);
        seen.add(d.name);
      }
      return dupes;
    }

    function populateOutput(select, devices, byName) {
      const selected = select.value;
      select.replaceChildren();
      select.appendChild(makeOption(byName ? "Same as above" : "Computer default", ""));
      const dupes = duplicateNames(devices);
      for (const d of devices) {
        const value = String(byName ? d.name : d.index);
        if (!hasOption(select, value)) select.appendChild(makeOption(deviceLabel(d, dupes), value));
      }
      if (selected && !hasOption(select, selected)) select.appendChild(makeOption(`${selected} (unavailable)`, selected));
      select.value = selected;
    }

    // Keep language routes by name across reloads and USB device renumbering.
    try {
      const saved = JSON.parse(readStorage(ttsStorageKey) || "{}");
      for (const select of [el.outputSelect, ...languageOutputs]) {
        if (["string", "number"].includes(typeof saved[select.name])) {
          const value = String(saved[select.name]);
          if (value && !hasOption(select, value)) select.appendChild(makeOption(value, value));
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
      for (const target of [el.outputSelect, ...languageOutputs, ttsModeSelect]) saved[target.name] = readTtsRoute(target);
      writeStorage(ttsStorageKey, JSON.stringify(saved));
    }
    for (const target of [el.outputSelect, ...languageOutputs, ttsModeSelect, ttsEnabled]) {
      target.addEventListener("change", () => {
        delete target.dataset.deviceIndex;
        persistTtsChoices();
      });
    }

    let knownChangeSeq = 0;
    let deviceRequest = 0;
    async function refreshDevices(showChangeToast) {
      const request = ++deviceRequest;
      try {
        const [data, outputs] = await Promise.all([
          getJson("/api/devices", {timeout: true}),
          getJson("/api/audio/output-devices", {timeout: true}),
        ]);
        if (request !== deviceRequest) return; // a newer listing already landed
        const inputs = data.inputs || [];
        const selectedMic = el.micSelect.value;
        el.micSelect.replaceChildren();
        el.micSelect.appendChild(makeOption("Automatic (computer default)", ""));
        const dupes = duplicateNames(inputs);
        for (const d of inputs) el.micSelect.appendChild(makeOption(deviceLabel(d, dupes), d.index));
        if (selectedMic && !hasOption(el.micSelect, selectedMic)) {
          el.micSelect.appendChild(makeOption(`Previously chosen microphone (unavailable)`, selectedMic));
        }
        el.micSelect.value = selectedMic;
        setText(el.micHint, inputs.length
          ? `${inputs.length} microphone${inputs.length === 1 ? "" : "s"} found. Choose the USB microphone by name.`
          : "No microphones were found. Plug in the USB microphone.");
        populateOutput(el.outputSelect, outputs.outputs || [], false);
        for (const select of languageOutputs) populateOutput(select, outputs.outputs || [], true);
        if (showChangeToast) {
          const counts = `${inputs.length} in / ${(data.outputs || []).length} out`;
          showToast(`Audio devices changed — ${counts}. Check your microphone and speaker choices.`);
        }
        knownChangeSeq = data.change_seq || knownChangeSeq;
      } catch (e) {
        // 503 if sounddevice unavailable — leave the placeholder option
      }
    }

    // ---- capabilities (all optional) --------------------------------------
    let capabilities = {};
    let capabilitiesRequest = 0;
    let capabilitiesJson = null;
    let capabilitiesSeen = false;
    let profilesSupported = false;
    let serverDefaultProfile = "standard";
    const PROFILE_KEY = "stark-operator-profile";

    function applyCapabilities(data) {
      const next = data && typeof data === "object" ? data : {};
      const json = JSON.stringify(next);
      // Status snapshots may embed capabilities on every poll; only rebuild
      // the dependent controls when something actually changed.
      if (json === capabilitiesJson) return;
      capabilitiesJson = json;
      capabilities = next;
      capabilitiesSeen = true;
      setText(el.capabilitiesDetail, Object.keys(capabilities).length ? JSON.stringify(capabilities, null, 2) : "none advertised");
      const before = effectiveProfileId();
      renderProfiles();
      if (effectiveProfileId() !== before) refreshPreflight();
      renderAudience();
      renderAudioTests();
      renderStorageAvailability();
      setText(el.captionEndpoint, captionUrl());
      updateButtonsForState(currentState);
    }
    async function probeCapabilities() {
      const request = ++capabilitiesRequest;
      try {
        const data = await getJson("/api/capabilities", {timeout: true});
        if (request === capabilitiesRequest) applyCapabilities(data);
      } catch (e) {
        if (request !== capabilitiesRequest) return;
        if (isMissingEndpoint(e)) applyCapabilities({});
        else setText(el.capabilitiesDetail, `capabilities unavailable: ${e.message}`);
      }
    }
    const featureFlag = name => (capabilitiesSeen && Object.hasOwn(capabilities, name) ? capabilities[name] !== false : null);

    // The select's empty option follows the operator service's default
    // (STARK_PROFILE, e.g. a Lite launcher). A persisted explicit choice
    // overrides it only while the server still offers that profile.
    function effectiveProfileId() {
      if (!profilesSupported) return "";
      return el.profileSelect.value || serverDefaultProfile;
    }
    function renderProfiles() {
      const profiles = pickProfiles(capabilities);
      profilesSupported = profiles.length > 0;
      const select = el.profileSelect;
      if (!profilesSupported) {
        serverDefaultProfile = "standard";
        el.profileField.hidden = true;
        el.profileHint.hidden = true;
        select.replaceChildren(makeOption("Default for this computer", ""));
        select.value = "";
        applyProfileConstraints();
        return;
      }
      serverDefaultProfile = defaultProfileId(capabilities, profiles);
      const confirmed = ACTIVE_STATES.includes(currentState) && currentSnap && currentSnap.config
        ? profileNameOf(currentSnap.config.profile) : "";
      const persisted = normalizeProfileId(readStorage(PROFILE_KEY) || "");
      const explicit = persisted && profiles.some(p => p.id === persisted) ? persisted : "";
      select.replaceChildren(makeOption(`Server default: ${profileLabel(serverDefaultProfile)}`, ""));
      for (const profile of profiles) {
        const option = makeOption(profile.available ? profile.label : `${profile.label} — not available here`, profile.id);
        if (!profile.available) option.disabled = true;
        select.appendChild(option);
      }
      if (confirmed) {
        if (!hasOption(select, confirmed)) select.appendChild(makeOption(profileLabel(confirmed), confirmed));
        select.value = confirmed;
      } else {
        select.value = explicit;
      }
      el.profileField.hidden = false;
      const effective = effectiveProfileId();
      const chosen = profiles.find(profile => profile.id === effective);
      const parts = [chosen && chosen.description ? chosen.description : ""];
      if (confirmed) parts.push("This is the profile of the running session.");
      else if (!select.value) parts.push("Following the operator service's default. Readiness below confirms whether its models are installed.");
      setText(el.profileHint, parts.filter(Boolean).join(" "));
      el.profileHint.hidden = false;
      applyProfileConstraints();
    }
    el.profileSelect.addEventListener("change", () => {
      writeStorage(PROFILE_KEY, el.profileSelect.value);
      renderProfiles();
    });

    // A Lite profile owns the backend, engine, A/B and speaker-label settings;
    // leaving them editable would send a contradictory (e.g. MLX) request.
    function applyProfileConstraints() {
      const lite = profilesSupported && isLiteProfile(effectiveProfileId());
      const active = ACTIVE_STATES.includes(currentState);
      const bound = ["backend", "engine", "run_ab", "diarize"].map(control).filter(Boolean);
      if (lite && !active) {
        const backend = control("backend");
        if (backend && backend.value !== "auto") backend.value = "auto";
        for (const name of ["run_ab", "diarize"]) {
          const box = control(name);
          if (box) box.checked = false;
        }
      }
      for (const target of bound) target.disabled = active || lite;
      if (el.liteNote) el.liteNote.hidden = !lite;
    }

    // ---- microphone / speaker tests (real, idle-only, work-leased) ---------
    let audioTestBusy = false;
    let audioTestResultShown = false; // a finished test's result line stays until the capability changes
    function audioTestsSupported() {
      return capabilities.audio_tests === true;
    }
    function renderAudioTests() {
      const supported = audioTestsSupported();
      const idleOnly = capabilities.audio_tests_require_idle !== false;
      const blocked = idleOnly && (ACTIVE_STATES.includes(currentState) || !!workBusy);
      el.micTestBtn.disabled = !supported || audioTestBusy || blocked;
      el.outputTestBtn.hidden = !supported;
      el.outputTestBtn.disabled = !supported || audioTestBusy || blocked;
      if (!supported) {
        audioTestResultShown = false;
        setText(el.micTestStatus, "Microphone test is not available from this operator service yet. After starting, say a sentence and watch for the first caption.");
      } else if (!audioTestBusy && !audioTestResultShown) {
        setText(el.micTestStatus, blocked
          ? "Audio tests are available when no session or other job is running."
          : "Click Test microphone, then speak normally for two seconds. Nothing is recorded or saved.");
      }
      if (el.deviceNote) el.deviceNote.hidden = !supported || capabilities.audio_devices_validated !== false;
    }

    function renderLevelBar(percent) {
      const width = percent === null ? 0 : percent;
      el.micLevelBar.style.width = `${width}%`;
      el.micLevelBar.classList.toggle("hot", width > 90);
    }

    async function runAudioTest(kind) {
      if (!audioTestsSupported() || audioTestBusy) return;
      const input = kind === "input";
      const url = input ? "/api/audio/test-input" : "/api/audio/test-output";
      const select = input ? el.micSelect : el.outputSelect;
      const device = select.value ? Number(select.value) : null;
      const deviceName = selectedLabel(select, input ? "the computer's default microphone" : "the computer's default speakers");
      const body = input ? {device, duration_s: 2} : {device, duration_s: 0.4};
      const label = input ? "Microphone test" : "Speaker test";
      audioTestBusy = true;
      renderAudioTests();
      updateButtonsForState(currentState);
      setText(el.micTestStatus, input
        ? `Listening for 2 seconds on ${deviceName}… speak normally.`
        : `Playing a short test tone on ${deviceName}…`);
      try {
        const result = await postJson(url, body);
        if (!result || result.ok !== true) {
          setText(el.micTestStatus, `${label} did not report a result. Nothing can be confirmed.`);
        } else if (input) {
          const level = describeLevel(result.peak, result.rms);
          renderLevelBar(level.percent);
          const seconds = Number(result.duration_s) || 2;
          setText(el.micTestStatus, `${level.text} Listened for ${seconds} s on ${deviceName}${result.recorded === false ? "; nothing was recorded or saved" : ""}.`);
        } else {
          setText(el.micTestStatus, `The server finished playing a short tone on ${deviceName}. The software cannot tell whether it was audible — confirm you heard it.`);
        }
      } catch (e) {
        const info = describeError(e);
        if (input) renderLevelBar(null);
        if (info.code === "work_busy") {
          setText(el.micTestStatus, `The operator service is ${humanWork(info.work) === "busy" ? "busy" : humanWork(info.work)}. Try again when it finishes.`);
        } else if (info.code === "audio_unavailable" || info.status === 422) {
          setText(el.micTestStatus, `${label} could not use ${deviceName}: ${info.message}`);
        } else if (isMissingEndpoint(e)) {
          setText(el.micTestStatus, `${label} is not available from this operator service.`);
        } else {
          setText(el.micTestStatus, `${label} failed: ${info.message}`);
        }
      } finally {
        audioTestBusy = false;
        audioTestResultShown = true;
        renderAudioTests();
        updateButtonsForState(currentState);
        refreshStatus();
      }
    }
    el.micTestBtn.addEventListener("click", () => runAudioTest("input"));
    el.outputTestBtn.addEventListener("click", () => runAudioTest("output"));

    // ---- session status ---------------------------------------------------
    let currentState = "idle";
    let currentError = null;
    let currentSnap = null;
    let hasReportedHealth = false;
    let healthSessionId = null;
    let statusSeen = false;
    let idleLanguageEdited = false;
    let statusRenderRevision = 0;
    let connection = "connecting"; // connecting | ok | stale
    let lastStatusOkAt = null;
    let lastStatusError = "";
    let workBusy = null;
    let pendingControl = null; // {kind: "pause"|"resume", target, since}

    languageSelect.addEventListener("change", () => {
      if (!ACTIVE_STATES.includes(currentState)) idleLanguageEdited = true;
      renderFlipLabel();
    });
    for (const target of formControls()) {
      target.addEventListener("change", () => {
        if (!ACTIVE_STATES.includes(currentState)) idleEditedFields.add(target.name);
        refreshPreflight();
      });
    }

    function syncSessionConfig(config, force) {
      let changed = false;
      for (const target of formControls()) {
        // The direction and the profile are shown from the confirmed session
        // separately; the profile follows the server default when idle.
        if (target.name === "lang" || target.name === "profile") continue;
        const key = target.name === "output_device" ? "tts_device" : target.name;
        if (!Object.hasOwn(config, key) || (!force && (statusSeen || idleEditedFields.has(target.name)))) continue;
        const value = config[key];
        if (["tts_device_en", "tts_device_es"].includes(key)) {
          changed = changed || target.dataset.deviceIndex !== (typeof value === "number" ? String(value) : undefined);
          if (typeof value === "number") target.dataset.deviceIndex = String(value);
          else delete target.dataset.deviceIndex;
        }
        if (target.type === "checkbox") {
          if (typeof value !== "boolean") continue;
          changed = changed || target.checked !== value;
          target.checked = value;
        } else {
          const selected = value == null ? "" : String(value);
          if (target.tagName === "SELECT" && !hasOption(target, selected)) {
            target.appendChild(makeOption(selected || "Computer default", selected));
          }
          changed = changed || target.value !== selected;
          target.value = selected;
        }
        idleEditedFields.delete(target.name);
      }
      return changed;
    }

    function renderStatePill() {
      const described = describeState({
        state: currentState, error: currentError, connection, preflightOk, preflightChecked,
        outcome: currentSnap && currentSnap.outcome, work: workBusy,
        readiness: currentSnap && currentSnap.readiness,
        hasReportedHealth,
      });
      setText(el.statePill, described.label);
      el.statePill.className = "state-pill " + described.tone;
      setText(el.stateDetail, described.detail);
    }

    function renderConnectionAge() {
      if (connection === "connecting") {
        setText(el.connectionAge, "Connecting to the operator service…");
        el.connectionAge.className = "connection";
        return;
      }
      const age = lastStatusOkAt == null ? null : (now() - lastStatusOkAt) / 1000;
      if (connection === "ok" && age != null && age * 1000 > STALE_AFTER_MS) {
        markDisconnected(`No status update for ${Math.round(age)} s.`);
        return;
      }
      if (connection === "stale") {
        setText(el.connectionAge, `Not connected — last update ${formatAge(age)}. ${lastStatusError}`.trim());
        el.connectionAge.className = "connection stale";
      } else {
        setText(el.connectionAge, `Connected · status updated ${formatAge(age)}`);
        el.connectionAge.className = "connection";
      }
    }

    function markDisconnected(reason) {
      connection = "stale";
      lastStatusError = reason || "";
      renderConnectionAge();
      renderStatePill();
      renderLive();
      updateButtonsForState(currentState);
    }

    function renderFlipLabel() {
      const lang = (currentSnap && ACTIVE_STATES.includes(currentState) && currentSnap.config && currentSnap.config.lang) || languageSelect.value;
      setText(el.flipBtn, lang === "es" ? "Switch to English speaker" : "Switch to Spanish speaker");
    }

    function renderHealth(snap) {
      const health = snap && snap.health;
      const active = ACTIVE_STATES.includes(currentState);
      if (!health || typeof health !== "object" || !active) {
        el.healthList.hidden = true;
        el.healthList.replaceChildren();
        renderLiveLevel(null);
        return;
      }
      const items = [];
      const inputAge = Number(health.input_age_s);
      if (health.input_seen === false || (health.input_age_s == null && Object.hasOwn(health, "input_seen"))) {
        items.push({text: "Waiting for the audio feed", bad: currentState === "running"});
      } else if (health.input_age_s != null) {
        items.push({text: Number.isFinite(inputAge) ? `Audio feed updated ${formatAge(inputAge)}` : "Audio feed: unknown", bad: Number.isFinite(inputAge) && inputAge > 30});
      }
      if (health.caption_age_s != null) {
        const age = Number(health.caption_age_s);
        items.push({text: Number.isFinite(age) ? `Last caption ${formatAge(age)}` : "Captions: none yet", bad: false});
      } else if (Object.hasOwn(health, "caption_age_s")) {
        items.push({text: "No captions yet", bad: false});
      }
      if (health.clients != null) items.push({text: `Displays connected: ${health.clients}`, bad: Number(health.clients) === 0});
      if (health.queues && typeof health.queues === "object") {
        const backlog = Object.values(health.queues).reduce((sum, v) => sum + (Number(v) || 0), 0);
        items.push({text: `Backlog: ${backlog}`, bad: backlog > 5});
      }
      const errors = health.error_count != null ? Number(health.error_count)
        : typeof health.errors === "number" ? health.errors
        : Array.isArray(health.errors) ? health.errors.length : null;
      if (errors != null && Number.isFinite(errors)) items.push({text: `Errors: ${errors}`, bad: errors > 0});
      const recording = health.recording;
      if (recording && typeof recording === "object") {
        if (recording.audio_enabled === false) items.push({text: "Original chunk audio is not being saved", bad: false});
        else if (recording.ok === false || Number(recording.required_failures) > 0) items.push({text: "Recording problems — review audio may be incomplete", bad: true});
        else items.push({text: "Recording audio", bad: false});
      } else if (recording != null) {
        items.push({text: recording ? "Recording audio" : "Original chunk audio is not being saved", bad: false});
      }
      if (health.persistence && typeof health.persistence === "object" && health.persistence.ok === false) {
        items.push({text: `Saving problems: ${health.persistence.reason || "some session files failed to save"}`, bad: true});
      }
      if (health.storage && typeof health.storage === "object" && health.storage.low_space === true) {
        items.push({text: `Low disk space (${formatBytes(health.storage.free_bytes)} free)`, bad: true});
      }
      if (Number(health.publish_failures) > 0) items.push({text: `Status updates failing: ${health.publish_failures}`, bad: true});
      el.healthList.replaceChildren(...items.map(item => {
        const li = doc.createElement("li");
        if (item.bad) li.className = "bad";
        const strong = doc.createElement("strong");
        strong.textContent = item.text;
        li.appendChild(strong);
        return li;
      }));
      el.healthList.hidden = items.length === 0;
      renderLiveLevel(Object.hasOwn(health, "input_level") ? health.input_level : null);
    }

    function renderLiveLevel(level) {
      if (!el.liveLevelBar || !el.liveLevelText) return;
      const percent = level == null ? null : levelPercent(level);
      const parent = el.liveLevelBar.parentNode;
      if (parent) parent.hidden = percent === null;
      el.liveLevelText.hidden = percent === null;
      el.liveLevelBar.style.width = `${percent === null ? 0 : percent}%`;
      el.liveLevelBar.classList.toggle("hot", percent !== null && percent > 90);
      if (percent !== null) setText(el.liveLevelText, `Sound level ${percent}%`);
    }

    function renderReadiness(snap) {
      const readiness = snap && snap.readiness;
      if (!readiness || typeof readiness !== "object" || !ACTIVE_STATES.includes(currentState)) {
        el.pipelineReadiness.hidden = true;
        return;
      }
      let text = readiness.ready ? "Captions are ready." : `Getting ready: ${humanPhase(readiness.phase)}.`;
      const plainPhase = String(readiness.phase || "").toLowerCase().replace(/[_-]+/g, " ");
      const reason = String(readiness.reason || "").trim();
      if (!readiness.ready && reason && reason.toLowerCase() !== plainPhase) text += ` ${reason}`;
      if (currentState === "paused" && readiness.phase === "paused" && !readiness.stale) {
        text = "Captions are paused.";
      } else if (awaitingFirstHealth(readiness, hasReportedHealth)) {
        text = "Waiting for the caption process to report its first status.";
      } else if (readiness.stale) {
        const age = readiness.age_s == null ? "" : ` (last update ${formatAge(Number(readiness.age_s))})`;
        text = `The caption process has not reported a recent status${age}${readiness.reason ? ` — ${readiness.reason}` : ""}.`;
      }
      setText(el.pipelineReadiness, text);
      el.pipelineReadiness.hidden = false;
    }

    function renderLive() {
      const snap = currentSnap || {};
      const stale = connection === "stale";
      const readinessState = currentState === "running" ? runningReadiness(snap.readiness, hasReportedHealth) : null;
      let title, subtitle;
      switch (currentState) {
        case "starting": title = "Starting…"; subtitle = "Loading language models. Stop is available if you need it."; break;
        case "running":
          title = readinessState ? readinessState.label : "Live";
          subtitle = readinessState ? readinessState.detail : "Captions are being sent to the audience display.";
          break;
        case "paused": title = "Paused"; subtitle = "Captions are paused. Press Resume to continue."; break;
        case "stopping": title = "Stopping…"; subtitle = "Finishing up."; break;
        case "error": title = "Needs attention"; subtitle = firstLine(currentError) || "The caption process reported a problem. Use Stop and reset."; break;
        default:
          title = "Not running";
          subtitle = snap.outcome === "failed" ? "The last session failed. See the message at the top of the page or the Help tab."
            : snap.outcome === "interrupted" ? "The last session was interrupted."
            : snap.outcome === "completed" ? "The last session finished normally. Start captions from the Prepare tab."
            : "Start captions from the Prepare tab.";
      }
      if (pendingControl && ACTIVE_STATES.includes(currentState)) {
        const waited = now() - pendingControl.since;
        subtitle = `${readinessState ? `${subtitle} ` : ""}${pendingControl.kind === "pause" ? "Pausing" : "Resuming"}… waiting for the caption process to confirm.`;
        if (waited > CONTROL_ACK_TIMEOUT_MS) subtitle += " No confirmation yet; the status may be stale.";
      }
      if (stale) {
        subtitle = `The page has lost contact with the operator service (last known: ${title.toLowerCase()}). ${lastStatusError}`.trim();
        title = "Not connected";
      }
      setText(el.liveTitle, title);
      setText(el.liveSubtitle, subtitle);
      if (el.liveEvent) {
        let event = ACTIVE_STATES.includes(currentState) || currentState === "error" ? snap.last_event || "" : "";
        const ready = snap.readiness && snap.readiness.ready === true && !snap.readiness.stale;
        const oldStartup = /models loaded|loading (?:language )?models|waiting for (?:audio|pipeline|input) readiness|subprocess launching/i.test(event);
        const oldPause = /pause requested.*waiting|waiting.*pause.*acknowledg/i.test(event);
        const oldResume = /resume requested.*waiting|waiting.*resume.*acknowledg/i.test(event);
        if ((ready && (oldStartup || oldResume)) || (currentState === "paused" && oldPause)) event = "";
        setText(el.liveEvent, event);
        el.liveEvent.hidden = !event;
      }
      const lang = snap.config && snap.config.lang;
      let meta = ACTIVE_STATES.includes(currentState) && LANG_TEXT[lang] ? LANG_TEXT[lang] : "";
      const profile = profileNameOf(snap.effective_profile) || (snap.config ? profileNameOf(snap.config.profile) : "");
      if (profile && ACTIVE_STATES.includes(currentState)) meta += `${meta ? " · " : ""}profile: ${profileLabel(profile)}`;
      setText(el.liveLanguage, meta);
      renderElapsed();
      renderHealth(snap);
      renderReadiness(snap);
      renderFlipLabel();
    }

    function renderElapsed() {
      const snap = currentSnap;
      if (!snap || !ACTIVE_STATES.includes(currentState) || !snap.started_at) {
        setText(el.liveElapsed, "");
        return;
      }
      const started = Date.parse(snap.started_at);
      if (!Number.isFinite(started)) { setText(el.liveElapsed, ""); return; }
      setText(el.liveElapsed, `Running for ${formatDuration((now() - started) / 1000)}`);
    }

    function renderStatus(snap, options) {
      const opts = options || {};
      ++statusRenderRevision;
      const previousState = currentState;
      const wasActive = ACTIVE_STATES.includes(currentState);
      if (snap.session_id !== healthSessionId || !ACTIVE_STATES.includes(snap.state)) {
        healthSessionId = snap.session_id || null;
        hasReportedHealth = false;
      }
      if ((snap.readiness && snap.readiness.updated_at != null) || (snap.health && snap.health.updated_at != null)) hasReportedHealth = true;
      currentState = snap.state || "idle";
      currentError = snap.error || null;
      currentSnap = snap;
      workBusy = snap.work && typeof snap.work === "object" && snap.work.kind ? snap.work : null;
      const isActive = ACTIVE_STATES.includes(currentState);
      if (pendingControl && (currentState === pendingControl.target || currentState !== pendingControl.from)) pendingControl = null;
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
      if (!statusSeen && isActive && !readStorage(TAB_KEY)) showTab("live", {persist: false});
      statusSeen = true;
      for (const target of formControls()) target.disabled = isActive;
      if (snap.capabilities && typeof snap.capabilities === "object") applyCapabilities(snap.capabilities);
      if (isActive !== wasActive || previousState !== currentState) renderProfiles();
      else applyProfileConstraints();
      renderStatePill();
      renderConnectionAge();
      updateButtonsForState(currentState);
      renderLive();
      renderAudioTests();
      updateHealthCaptions(snap);
      setText(el.statusDetail, JSON.stringify(snap, null, 2));
      renderSummaryButtons();
      if (currentState === "error") {
        // Keep a request failure visible until dismissed; the state error
        // returns on the next poll after that.
        if (dismissedStateError !== (currentError || "") && attentionSource !== "request") {
          showAttention({
            source: "state",
            title: "Captions need attention",
            text: currentError || snap.last_event || "The caption process reported an error.",
            advice: "Click Stop and reset, wait until the status says Ready, then start again. If it happens again, send this text to the setup owner (Help tab).",
          });
        }
      } else {
        dismissedStateError = null;
        clearAttention("state");
      }
      syncCaptionClient(isActive);
      if (EventImpl) dispatch(new EventImpl("operator-session", {detail: snap}));
      // Recheck after a stop, after a confirmed change, or when a control asks for it.
      if ((!isActive && wasActive) || languageChanged || configChanged || opts.forcePreflight) refreshPreflight();
    }

    async function refreshStatus() {
      const revision = statusRenderRevision;
      try {
        const snap = await getJson("/api/session/status", {timeout: true});
        // A control response may have confirmed a newer state meanwhile.
        if (revision !== statusRenderRevision) return;
        connection = "ok";
        lastStatusOkAt = now();
        lastStatusError = "";
        renderStatus(snap);
      } catch (e) {
        if (revision !== statusRenderRevision) return;
        markDisconnected(`(${e.message})`);
      }
    }

    function updateButtonsForState(state) {
      const stale = connection === "stale";
      const isIdle = state === "idle" || state === "error";
      const isRunning = state === "running";
      const isPaused = state === "paused";
      el.startBtn.disabled = !isIdle || !preflightOk || stale || !!workBusy || audioTestBusy;
      // Stop stays available in "error": the process may still be alive.
      el.stopBtn.disabled = state === "idle" || state === "stopping";
      el.pauseBtn.disabled = !isRunning || stale || !!pendingControl;
      el.resumeBtn.disabled = !isPaused || stale || !!pendingControl;
      el.flipBtn.disabled = !isRunning || stale;
      el.fallbackBtn.disabled = !isRunning || stale;
      if (el.attentionStop) el.attentionStop.disabled = el.stopBtn.disabled;
      setText(el.startHint, audioTestBusy ? "Wait for the audio test to finish."
        : workBusy ? `The operator service is ${humanWork(workBusy)}. Start becomes available when it finishes.`
        : stale ? "Start is unavailable while the page is not connected to the operator service."
        : !isIdle ? "Captions are running. Use the Live tab to pause or stop."
        : preflightOk ? "Everything looks ready." : "Start becomes available when every check above is green or yellow.");
    }

    // ---- start / stop / controls -----------------------------------------
    function readForm() {
      // Active controls are disabled, so FormData would silently omit them.
      const value = name => control(name).value;
      const checked = name => control(name).checked;
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
        if (device) body[`tts_device_${lang}`] = readTtsRoute(control(`tts_device_${lang}`));
      }
      const profile = effectiveProfileId();
      if (profile) body.profile = profile;
      return body;
    }

    function acceptControlResponse(snap, options) {
      connection = "ok";
      lastStatusOkAt = now();
      lastStatusError = "";
      renderStatus(snap, options);
    }

    function handleStartError(e) {
      const info = describeError(e);
      setText(el.startStatus, "Not started.");
      if (info.code === "preflight_failed") {
        if (info.checks) renderChecks({checks: info.checks, ok: false});
        else preflightFailed(info.message);
        showAttention({title: "Can't start yet", text: info.message, advice: "Fix the red items on the Prepare tab, then try again."});
        showTab("prepare");
      } else if (info.code === "work_busy") {
        showAttention({tone: "info", title: "The operator service is busy", text: info.message,
          advice: `It is ${humanWork(info.work)}. Wait for it to finish, then try again.`});
      } else if (info.code === "profile_unavailable") {
        showAttention({title: "This profile is not installed", text: info.message,
          advice: "Choose the server default profile on the Prepare tab, or ask the setup owner to install the selected profile."});
        showTab("prepare");
      } else if (info.status === 422 && typeof info.raw === "string") {
        showAttention({title: "The selected settings can't be used together", text: info.message,
          advice: "Check the profile on the Prepare tab and the technical settings under Advanced, then try again."});
        showTab("prepare");
      } else if (info.status === 409) {
        showAttention({tone: "info", title: "Captions are already running or stopping", text: info.message,
          advice: "Check the Live tab. If the status looks wrong, wait a few seconds and try again."});
      } else {
        showAttention({title: "Couldn't start captions", text: info.message,
          advice: "Check the Prepare tab. If it keeps failing, the Help tab explains how to send details to the setup owner."});
      }
    }

    el.startBtn.addEventListener("click", async () => {
      el.startBtn.disabled = true;
      clearAttention("request");
      setText(el.startStatus, "Starting… loading language models.");
      try {
        const snap = await postJson("/api/session/start", readForm());
        acceptControlResponse(snap);
        setText(el.startStatus, "");
        showTab("live");
      } catch (e) {
        handleStartError(e);
        updateButtonsForState(currentState);
        refreshStatus();
      }
    });

    async function stopSession() {
      el.stopBtn.disabled = true;
      if (el.attentionStop) el.attentionStop.disabled = true;
      pendingControl = null;
      try {
        const snap = await postJson("/api/session/stop");
        acceptControlResponse(snap);
      } catch (e) {
        showAttention({title: "Couldn't stop captions", text: describeError(e).message,
          advice: "Wait a few seconds and try again. If the page is not connected, the operator service itself may need restarting."});
        updateButtonsForState(currentState);
        refreshStatus();
      }
    }
    el.stopBtn.addEventListener("click", stopSession);
    el.attentionStop.addEventListener("click", stopSession);

    async function controlClick(url, body, btn, label, ack) {
      btn.disabled = true;
      const from = currentState;
      try {
        const snap = await postJson(url, body);
        // Pause/resume are acknowledged by the caption process later; until
        // the polled state changes, say so instead of pretending it happened.
        if (ack && snap.state === from) pendingControl = {kind: ack.kind, target: ack.target, from, since: now()};
        acceptControlResponse(snap, {forcePreflight: url === "/api/control/lang_flip"});
      } catch (e) {
        showAttention({title: `Couldn't ${label}`, text: describeError(e).message, advice: "The status below is being refreshed. Try again in a moment."});
        updateButtonsForState(currentState);
        refreshStatus();
      }
    }
    el.pauseBtn.addEventListener("click", () => controlClick("/api/control/pause", null, el.pauseBtn, "pause captions", {kind: "pause", target: "paused"}));
    el.resumeBtn.addEventListener("click", () => controlClick("/api/control/resume", null, el.resumeBtn, "resume captions", {kind: "resume", target: "running"}));
    el.flipBtn.addEventListener("click", () => controlClick("/api/control/lang_flip", null, el.flipBtn, "switch the speaker language"));
    el.fallbackBtn.addEventListener("click", () => controlClick("/api/control/fallback", {engine: "hf"}, el.fallbackBtn, "switch to the fallback engine"));

    // ---- caption preview (WebSocket when open, status-feed captions otherwise)
    const captionModel = captionsLib ? captionsLib.createModel({limit: 50}) : null;
    let captionClient = null;
    let captionSocketState = "closed";
    let healthCaptions = [];
    let healthCaptionsJson = "[]";

    function urlParam(name) {
      try { return new SearchParams(loc.search || "").get(name); } catch (e) { return null; }
    }
    function captionUrl() {
      const ports = capabilities.display_ports || {};
      const port = Number(ports.websocket) || Number(urlParam("caption_port")) || 8765;
      const host = loc.hostname || "localhost";
      const proto = loc.protocol === "https:" ? "wss" : "ws";
      return `${proto}://${host}:${port}`;
    }
    function socketCaptionsLive() {
      return captionSocketState === "open" && !!captionClient;
    }
    function socketRows() {
      return socketCaptionsLive() && captionModel ? captionModel.sentences() : [];
    }
    function healthCaptionRows() {
      return healthCaptions.filter(c => !captionModel || captionModel.acceptsCaption(c));
    }
    function captionRows() {
      const live = socketRows();
      if (live.length) return live.slice(-6);
      return healthCaptionRows().slice(-6).map(c => ({
        source: c.english || "", target: c.spanish_a || "", speaker: c.speaker || "",
        partial: c.stage === "partial", streaming: false,
      }));
    }
    function updateHealthCaptions(snap) {
      const list = snap && snap.health && Array.isArray(snap.health.captions) && ACTIVE_STATES.includes(currentState)
        ? snap.health.captions.map(c => ({...c, session_id: c.session_id || snap.session_id})) : [];
      const json = JSON.stringify(list);
      if (json === healthCaptionsJson) return;
      healthCaptionsJson = json;
      healthCaptions = list;
      if (!socketRows().length) renderCaptions();
    }
    function renderCaptionStatus() {
      if (!ACTIVE_STATES.includes(currentState)) {
        setText(el.captionStatus, "Caption preview connects while captions are running.");
        return;
      }
      if (socketCaptionsLive()) {
        const labels = captionModel ? captionModel.labels() : {};
        const pair = labels.source && labels.target ? ` (${labels.source} → ${labels.target})` : "";
        setText(el.captionStatus, captionModel && captionModel.musicHold()
          ? `Connected${pair} — music or silence detected, waiting for speech.`
          : !socketRows().length && healthCaptionRows().length
            ? `Connected${pair}. Showing the latest captions from the status feed until new ones arrive.`
            : `Connected${pair}. Partial lines are in italics until the final caption replaces them.`);
      } else if (healthCaptionRows().length) {
        setText(el.captionStatus, "Showing the latest captions from the status feed (updates every few seconds).");
      } else {
        setText(el.captionStatus, currentState === "starting"
          ? "Waiting for the caption service to come up…"
          : "Not receiving captions right now — reconnecting. Captions may still reach the audience display.");
      }
    }
    let captionViewKey = "";
    const captionKey = () => `${currentState}:${captionSocketState}`;
    function renderCaptions() {
      captionViewKey = captionKey();
      const sentences = captionRows();
      el.captionView.replaceChildren();
      if (!sentences.length) {
        const li = doc.createElement("li");
        li.className = "empty";
        li.textContent = ACTIVE_STATES.includes(currentState) ? "No captions yet. Speak into the microphone." : "No captions.";
        el.captionView.appendChild(li);
        renderCaptionStatus();
        return;
      }
      for (const s of sentences) {
        const li = doc.createElement("li");
        li.className = s.partial ? "partial" : s.streaming ? "streaming" : "final";
        const src = doc.createElement("span");
        src.className = "src";
        if (s.speaker) {
          const spk = doc.createElement("span");
          spk.className = "spk";
          spk.textContent = `${s.speaker}:`;
          src.appendChild(spk);
        }
        src.appendChild(doc.createTextNode(s.source || ""));
        const tgt = doc.createElement("span");
        tgt.className = "tgt";
        tgt.textContent = s.target || "";
        li.appendChild(src);
        li.appendChild(tgt);
        el.captionView.appendChild(li);
      }
      renderCaptionStatus();
    }
    function syncCaptionClient(active) {
      let changed = false;
      if (active && !captionClient && captionsLib && captionModel && WS) {
        captionClient = captionsLib.connect(captionUrl(), captionModel, {
          WebSocket: WS, setTimeout: setTimeoutImpl, clearTimeout: clearTimeoutImpl, retryMs: 3000,
          onChange: () => renderCaptions(),
          // Opening or losing the socket switches the caption source, so re-render.
          onStatus: state => { captionSocketState = state; renderCaptions(); },
        });
        changed = true;
      } else if (!active && captionClient) {
        captionClient.close();
        captionClient = null;
        captionSocketState = "closed";
        captionModel.reset();
        changed = true;
      }
      // Re-render only when the session state changed; a polled rewrite of an
      // unchanged list would churn the DOM every 1.5 s.
      if (changed || captionKey() !== captionViewKey) renderCaptions();
    }

    // ---- audience display links -------------------------------------------
    let audienceInfo = null;
    function renderAudience() {
      audienceInfo = audienceLinks(loc, capabilities);
      setText(el.audienceUrl, audienceInfo.mobile);
      setText(el.audienceNote, `${audienceInfo.note} The audience pages are served only while captions are running.`);
      el.audienceCopy.disabled = !audienceInfo.shareable;
      let drawn = false;
      if (audienceInfo.shareable && qr && el.audienceQr) {
        try { drawn = qr.draw(el.audienceQr, audienceInfo.mobile, 176); } catch (e) { drawn = false; }
      }
      if (el.audienceQr) el.audienceQr.hidden = !drawn;
      if (el.otherDisplays) {
        const extras = [["Church display", audienceInfo.church], ["Streaming overlay", audienceInfo.obs]].filter(([, url]) => url);
        el.otherDisplays.replaceChildren(...extras.map(([name, url]) => {
          const li = doc.createElement("li");
          const link = doc.createElement("a");
          link.href = url;
          link.target = "_blank";
          link.rel = "noopener";
          link.textContent = `${name}: ${url}`;
          li.appendChild(link);
          return li;
        }));
        el.otherDisplays.hidden = extras.length === 0;
      }
    }
    el.audienceOpen.addEventListener("click", () => {
      if (audienceInfo) openWindow(audienceInfo.audience);
    });
    el.audienceCopy.addEventListener("click", async () => {
      if (!audienceInfo) return;
      if (clipboard && clipboard.writeText) {
        try {
          await clipboard.writeText(audienceInfo.mobile);
          showToast("Phone link copied.");
          return;
        } catch (e) { /* fall through */ }
      }
      showToast("Copying isn't available here — the link is shown below the buttons.");
    });

    // ---- live metrics over /ws/control (Advanced) --------------------------
    const latencyHistory = [];
    const confidenceHistory = [];
    let metricsCohort = "";

    function renderMetrics(snap) {
      const r = snap.resources || {};
      const lat = snap.latency || {};
      drawSpark(el.sparkVram, r.vram_mib_recent || [], {color: "#2563aa", fill: "rgba(37,99,170,0.08)"});
      drawSpark(el.sparkCpu, r.cpu_percent_recent || [], {color: "#c89a16", fill: "rgba(200,154,22,0.08)", min: 0, max: 100});
      setText(el.metricVram, r.vram_mib_current ? String(Math.round(r.vram_mib_current)) : "—");
      setText(el.metricCpu, r.cpu_percent_current != null ? Number(r.cpu_percent_current).toFixed(1) : "—");

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
      setText(el.metricLatency, lat.total_ms_p50 != null ? `${Math.round(lat.total_ms_p50)} / ${Math.round(lat.total_ms_p95)}` : "— / —");
      setText(el.metricConfidence, lat.confidence_mean != null ? Number(lat.confidence_mean).toFixed(2) : "—");
      el.metricLatency.title = lat.basis === "speech_end_to_final_ms"
        ? "Estimated speech end to final payload readiness (server); browser display timing is separate"
        : "Historical pipeline timing; not speech-end latency";
      drawSpark(el.sparkLatency, latencyHistory, {color: "#2f6b1a", fill: "rgba(47,107,26,0.08)"});
      drawSpark(el.sparkConfidence, confidenceHistory, {color: "#8a4500", min: 0, max: 1});
      setText(el.metricsMeta, `uptime ${Math.round(snap.uptime_s || 0)}s · queue ${snap.queue_depth} · errors ${snap.error_count}`);

      // Audio hotplug detection — re-fetch device list when the watcher's
      // change_seq counter advances.
      if (snap.audio && typeof snap.audio.change_seq === "number" && snap.audio.change_seq > knownChangeSeq) {
        refreshDevices(true);
      }

      // Live diarization (Phase 9.6.1) — current speaker (Advanced). The
      // caption preview no longer depends on this data.
      const diar = (snap.audio && snap.audio.diarization) || null;
      if (diar && diar.current_speaker) {
        setText(el.metricSpeaker, diar.current_speaker);
        setText(el.metricSpeakerDetail, `${diar.transitions || 0} transitions · ${(diar.recent || []).length} recent labels`);
      } else {
        setText(el.metricSpeaker, "—");
        setText(el.metricSpeakerDetail, "no diarization data yet");
      }
    }

    let metricsWs = null;
    let metricsBackoff = 1000;
    function connectMetrics() {
      if (!WS) return;
      const proto = loc.protocol === "https:" ? "wss" : "ws";
      const url = `${proto}://${loc.host}/ws/control`;
      try {
        metricsWs = new WS(url);
      } catch (e) {
        setText(el.metricsMeta, "metrics socket unavailable");
        return;
      }
      metricsWs.onopen = () => {
        metricsBackoff = 1000;
        setText(el.metricsMeta, "connected");
      };
      metricsWs.onmessage = event => {
        try { renderMetrics(JSON.parse(event.data)); } catch (e) { /* ignore malformed frames */ }
      };
      metricsWs.onclose = () => {
        setText(el.metricsMeta, `disconnected — retrying in ${metricsBackoff}ms`);
        setTimeoutImpl(connectMetrics, metricsBackoff);
        metricsBackoff = Math.min(metricsBackoff * 2, 15000);
      };
      metricsWs.onerror = () => {
        try { metricsWs.close(); } catch (e) { /* ignore */ }
      };
    }

    // ---- verses (Phase 9.6) -------------------------------------------------
    function renderVerses(highlights) {
      if (!el.versesList) return;
      el.versesList.replaceChildren();
      if (!highlights || highlights.length === 0) {
        const li = doc.createElement("li");
        li.className = "empty";
        li.textContent = "none yet";
        el.versesList.appendChild(li);
        return;
      }
      for (const h of highlights.slice(-25).reverse()) {
        const li = doc.createElement("li");
        const ref = doc.createElement("span");
        ref.className = "ref";
        ref.textContent = h.reference;
        const ctx = doc.createElement("span");
        ctx.className = "ctx";
        ctx.textContent = h.context || "";
        li.appendChild(ref);
        li.appendChild(ctx);
        el.versesList.appendChild(li);
      }
    }
    let verseRequest = 0;
    async function refreshVerses() {
      const request = ++verseRequest;
      try {
        const data = await getJson("/api/features/verses", {timeout: true});
        if (request === verseRequest) renderVerses(data.highlights || []);
      } catch (e) {
        // ignore errors during idle state
      }
    }

    // ---- session summary (bilingual text; raw JSON only under Advanced) -----
    let summaryTask = null;
    let summaryPollTimer = null;

    function renderSummaryButtons() {
      const running = summaryTask && ["pending", "running"].includes(summaryTask.state);
      const blocked = ACTIVE_STATES.includes(currentState) || (!!workBusy && !running);
      el.summaryBtn.disabled = blocked || !!running;
      setText(el.summaryBtn, summaryTask && summaryTask.state === "error" ? "Try again" : "Create summary");
      el.summaryCancel.hidden = !running;
      el.summaryCancel.disabled = !running;
    }

    function renderSummaryResult(result) {
      const data = result && typeof result === "object" ? result : {};
      const metadata = data.metadata && typeof data.metadata === "object" ? data.metadata : {};
      const excerpt = metadata.content_mode === "excerpt" || data.format === "short-session excerpt";
      setText(el.summaryEnglish, data.english || "(no English text)");
      setText(el.summarySpanish, data.spanish || "(no Spanish text)");
      const notices = data.notice ? [data.notice] : [];
      if (metadata.transcript_truncated === true) {
        notices.push("This summary uses the beginning and end of the transcript; the middle was omitted.");
      }
      const notice = notices.join(" ");
      setText(el.summaryNotice, notice);
      el.summaryNotice.hidden = !notice;
      const parts = [];
      parts.push(excerpt ? "Excerpt of the recorded text, not a summary" : `Model summary${data.format ? ` (${data.format})` : ""}`);
      if (data.translation_method) parts.push(`translation: ${data.translation_method}`);
      if (metadata.total_words != null) parts.push(`${metadata.total_words} words`);
      if (metadata.human_reviewed === false) parts.push("not reviewed by a person — treat as a draft");
      setText(el.summaryMeta, parts.join(" · "));
      el.summaryResult.hidden = false;
      setText(el.summaryRaw, JSON.stringify(data, null, 2));
    }

    function renderSummaryTask(task) {
      summaryTask = task;
      const state = task.state;
      if (state === "done") {
        setText(el.summaryStatus, "Finished.");
        renderSummaryResult(task.result);
      } else if (state === "error") {
        setText(el.summaryStatus, `Summary failed: ${task.error || "no reason given"}${task.return_code != null ? ` (exit code ${task.return_code})` : ""}. Click Try again to retry.`);
        el.summaryResult.hidden = true;
        setText(el.summaryRaw, JSON.stringify(task, null, 2));
      } else {
        setText(el.summaryStatus, `Working on the summary (${state})… This can take a few minutes on the computer's language model.`);
        el.summaryResult.hidden = true;
      }
      renderSummaryButtons();
    }

    function stopSummaryPolling() {
      if (summaryPollTimer) { clearIntervalImpl(summaryPollTimer); summaryPollTimer = null; }
    }

    async function pollSummary(taskId) {
      try {
        const task = await getJson(`/api/features/summary/${taskId}`);
        if (!summaryTask || summaryTask.task_id !== taskId) return; // an older task's reply
        renderSummaryTask(task);
        if (task.state === "done" || task.state === "error") stopSummaryPolling();
      } catch (e) {
        setText(el.summaryStatus, `Couldn't check the summary task: ${describeError(e).message}`);
      }
    }

    el.summaryBtn.addEventListener("click", async () => {
      el.summaryBtn.disabled = true;
      setText(el.summaryStatus, "Requesting a summary…");
      try {
        const task = await postJson("/api/features/summary", {});
        stopSummaryPolling();
        renderSummaryTask(task);
        summaryPollTimer = setIntervalImpl(() => pollSummary(task.task_id), 2000);
      } catch (e) {
        const info = describeError(e);
        setText(el.summaryStatus, info.code === "work_busy"
          ? `The operator service is ${humanWork(info.work)}. Try again when it finishes.`
          : `Couldn't start a summary: ${info.message}`);
        renderSummaryButtons();
        el.summaryBtn.disabled = ACTIVE_STATES.includes(currentState);
      }
    });
    el.summaryCancel.addEventListener("click", async () => {
      if (!summaryTask) return;
      el.summaryCancel.disabled = true;
      try {
        const task = await postJson(`/api/features/summary/${summaryTask.task_id}/cancel`);
        renderSummaryTask(task);
        if (task.state === "done" || task.state === "error") stopSummaryPolling();
        else setText(el.summaryStatus, "Cancel requested… waiting for the summary task to stop.");
      } catch (e) {
        setText(el.summaryStatus, `Couldn't cancel the summary: ${describeError(e).message}`);
        renderSummaryButtons();
      }
    });

    // ---- disk space + cleanup (optional endpoints) -------------------------
    let storageRequest = 0;
    let cleanupPreview = null;
    let cleanupVersion = 0;
    let storageSessionsList = [];

    function sessionIdOf(item) {
      if (typeof item === "string") return item;
      return item.session_id || item.session || item.id || "";
    }
    function sessionCompleted(item) {
      return typeof item !== "object" || item.status == null || item.status === "completed";
    }
    function renderStorageAvailability() {
      if (featureFlag("storage") === false) {
        setText(el.storageSummary, "Disk space details are not available from this operator service.");
        el.storagePreview.disabled = true;
        el.storageCleanup.disabled = true;
      }
      if (featureFlag("support") === false) {
        setText(el.supportStatus, "Support bundles are not available from this operator service. Send the error text from the top of the page instead.");
        el.supportPreview.disabled = true;
        el.supportExport.disabled = true;
      } else if (featureFlag("support") === true) {
        el.supportPreview.disabled = false;
      }
    }
    function renderStorage(data) {
      const parts = [];
      if (data.free_bytes != null) parts.push(`Free: ${formatBytes(data.free_bytes)}`);
      if (data.used_bytes != null) parts.push(`Used: ${formatBytes(data.used_bytes)}`);
      storageSessionsList = Array.isArray(data.sessions) ? data.sessions : [];
      if (typeof data.sessions === "number") parts.push(`${data.sessions} sessions stored`);
      else if (storageSessionsList.length) parts.push(`${storageSessionsList.length} sessions stored`);
      if (data.low_space === true) parts.push("Low disk space — free some space before the next session");
      // The server's scope sentence belongs with the summary; the status line
      // is reserved for the outcome of the operator's last action.
      if (data.cleanup_scope) parts.push(String(data.cleanup_scope));
      setText(el.storageSummary, parts.length ? parts.join(" · ") : "Disk space details were not reported.");
      el.storageSessions.replaceChildren(...storageSessionsList.map(item => {
        const li = doc.createElement("li");
        const label = doc.createElement("label");
        const box = doc.createElement("input");
        box.type = "checkbox";
        box.value = sessionIdOf(item);
        box.dataset.session = box.value;
        box.disabled = !sessionCompleted(item);
        label.appendChild(box);
        const text = doc.createElement("span");
        const status = typeof item === "object" && item.status ? ` · ${item.status}` : "";
        text.textContent = `${sessionIdOf(item)}${status}`;
        label.appendChild(text);
        li.appendChild(label);
        const bytes = doc.createElement("span");
        bytes.className = "bytes";
        const size = typeof item === "object" ? (item.cleanup_bytes != null ? item.cleanup_bytes : item.bytes) : null;
        bytes.textContent = size != null ? `logs ${formatBytes(size)}` : "";
        li.appendChild(bytes);
        return li;
      }));
      el.storagePreview.disabled = !storageSessionsList.some(sessionCompleted);
      el.storageCleanup.disabled = true;
      cleanupPreview = null;
    }
    async function refreshStorage() {
      if (featureFlag("storage") === false) return;
      const request = ++storageRequest;
      try {
        const data = await getJson("/api/storage", {timeout: true});
        if (request === storageRequest) renderStorage(data);
      } catch (e) {
        if (request !== storageRequest) return;
        if (isMissingEndpoint(e)) {
          setText(el.storageSummary, "Disk space details are not available from this operator service version.");
          el.storagePreview.disabled = true;
          el.storageCleanup.disabled = true;
        } else {
          setText(el.storageSummary, `Couldn't read disk space: ${describeError(e).message}`);
        }
      }
    }
    function selectedStorageSessions() {
      const boxes = Array.from(el.storageSessions.querySelectorAll('input[type="checkbox"]'));
      const chosen = boxes.filter(box => box.checked && !box.disabled).map(box => box.value);
      return chosen.length ? chosen : boxes.filter(box => !box.disabled).map(box => box.value);
    }
    el.storagePreview.addEventListener("click", async () => {
      const version = ++cleanupVersion;
      el.storagePreview.disabled = true;
      el.storageCleanup.disabled = true;
      cleanupPreview = null;
      const sessions = selectedStorageSessions();
      if (!sessions.length) {
        setText(el.storageStatus, "Only completed sessions can be cleaned up, and none are listed.");
        el.storagePreview.disabled = false;
        return;
      }
      setText(el.storageStatus, "Asking the server what can be removed…");
      try {
        const preview = await postJson("/api/storage/cleanup/preview", {session_ids: sessions});
        if (version !== cleanupVersion) return;
        cleanupPreview = preview;
        const files = Array.isArray(preview.files) ? preview.files : [];
        setText(el.storageStatus, files.length
          ? `${files.length} log file${files.length === 1 ? "" : "s"} (${formatBytes(preview.bytes)}) can be removed. Original audio, diagnostics, corrections and exports are kept. Click Delete the listed files to confirm.`
          : "Nothing to remove for the chosen sessions.");
        el.storageCleanup.disabled = files.length === 0;
      } catch (e) {
        if (version !== cleanupVersion) return;
        setText(el.storageStatus, isMissingEndpoint(e)
          ? "Cleanup is not available from this operator service version."
          : `Couldn't preview cleanup: ${describeError(e).message}`);
      } finally {
        if (version === cleanupVersion) el.storagePreview.disabled = !storageSessionsList.some(sessionCompleted);
      }
    });
    el.storageCleanup.addEventListener("click", async () => {
      if (!cleanupPreview || !cleanupPreview.preview_id) return;
      const version = ++cleanupVersion;
      const previewId = cleanupPreview.preview_id;
      el.storageCleanup.disabled = true;
      setText(el.storageStatus, "Removing files…");
      try {
        const result = await postJson("/api/storage/cleanup", {preview_id: previewId});
        if (version !== cleanupVersion) return;
        const bytes = result.removed_bytes != null ? result.removed_bytes : result.bytes;
        const count = Array.isArray(result.files) ? result.files.length : result.removed;
        setText(el.storageStatus, `Removed ${count != null ? `${count} file${count === 1 ? "" : "s"}` : "the listed logs"}${bytes != null ? ` (${formatBytes(bytes)})` : ""}. Original audio, diagnostics, corrections and exports were kept.`);
        cleanupPreview = null;
        refreshStorage();
      } catch (e) {
        if (version !== cleanupVersion) return;
        setText(el.storageStatus, `Cleanup failed: ${describeError(e).message}`);
      }
    });

    // ---- support bundle (optional endpoints) --------------------------------
    let supportSessionsRequest = 0;
    let supportVersion = 0;
    let supportPreview = null;
    let latestReviewSession = "";

    async function refreshSupportSessions() {
      const request = ++supportSessionsRequest;
      try {
        const data = await getJson("/api/review/sessions", {timeout: true});
        if (request !== supportSessionsRequest) return;
        const previous = el.supportSession.value;
        const sessions = data.sessions || [];
        latestReviewSession = sessions.length ? sessions[0].session : "";
        el.supportSession.replaceChildren(makeOption(latestReviewSession ? `Most recent session (${latestReviewSession})` : "Most recent session", ""));
        for (const s of sessions) {
          const status = s.active ? "live" : s.status || "";
          el.supportSession.appendChild(makeOption(`${s.session}${status ? ` · ${status}` : ""}`, s.session));
        }
        el.supportSession.value = hasOption(el.supportSession, previous) ? previous : "";
      } catch (e) {
        // The list is a convenience; the current session id still works without it.
      }
    }
    function supportSessionId() {
      return el.supportSession.value || latestReviewSession || (currentSnap && currentSnap.session_id) || "";
    }
    function invalidateSupport() {
      supportVersion += 1;
      supportPreview = null;
      el.supportExport.disabled = true;
      el.supportDownload.hidden = true;
      el.supportDownload.removeAttribute("href");
    }
    for (const target of [el.supportSession, el.supportText, el.supportAudio]) target.addEventListener("change", invalidateSupport);
    el.supportPreview.addEventListener("click", async () => {
      invalidateSupport();
      const version = supportVersion;
      const sessionId = supportSessionId();
      el.supportFiles.replaceChildren();
      if (!sessionId) {
        setText(el.supportStatus, "No session to describe yet. Run a session first, or choose one from the list.");
        return;
      }
      el.supportPreview.disabled = true;
      setText(el.supportStatus, "Asking the server what the bundle would contain…");
      try {
        const body = {session_id: sessionId, include_text: !!el.supportText.checked, include_audio: !!el.supportAudio.checked};
        const preview = await postJson("/api/support/preview", body);
        if (version !== supportVersion) return;
        supportPreview = preview;
        const files = Array.isArray(preview.files) ? preview.files : [];
        el.supportFiles.replaceChildren(...files.map(file => {
          const li = doc.createElement("li");
          const name = doc.createElement("span");
          name.textContent = typeof file === "string" ? file : file.path || file.name || JSON.stringify(file);
          li.appendChild(name);
          const bytes = doc.createElement("span");
          bytes.className = "bytes";
          bytes.textContent = typeof file === "object" && file.bytes != null ? formatBytes(file.bytes) : "";
          li.appendChild(bytes);
          return li;
        }));
        const privacy = typeof preview.privacy === "string" ? preview.privacy
          : preview.privacy && typeof preview.privacy === "object"
            ? [preview.privacy.message, preview.privacy.text_included ? "Includes caption text." : "", preview.privacy.audio_included ? "Includes audio clips." : ""].filter(Boolean).join(" ")
            : "";
        setText(el.supportStatus, `Metadata plus ${formatBytes(preview.bytes)} of optional attachments for session ${sessionId}.${privacy ? ` ${privacy}` : ""}`);
        el.supportExport.disabled = !preview.preview_id;
      } catch (e) {
        if (version !== supportVersion) return;
        setText(el.supportStatus, isMissingEndpoint(e)
          ? "Support bundles are not available from this operator service version. Send the error text from the top of the page instead."
          : `Couldn't preview the bundle: ${describeError(e).message}`);
      } finally {
        if (version === supportVersion) el.supportPreview.disabled = false;
      }
    });
    el.supportExport.addEventListener("click", async () => {
      if (!supportPreview || !supportPreview.preview_id) return;
      const version = supportVersion;
      const previewId = supportPreview.preview_id;
      el.supportExport.disabled = true;
      setText(el.supportStatus, "Building the bundle…");
      try {
        const result = await postJson("/api/support/export", {preview_id: previewId});
        if (version !== supportVersion) return;
        if (!result.download_url) throw new Error("the server did not return a download link");
        el.supportDownload.href = result.download_url;
        el.supportDownload.hidden = false;
        setText(el.supportStatus, "Bundle ready. Download it and send it to the setup owner.");
      } catch (e) {
        if (version !== supportVersion) return;
        setText(el.supportStatus, `Couldn't build the bundle: ${describeError(e).message}`);
        el.supportExport.disabled = false;
      }
    });

    // ---- bootstrap ----------------------------------------------------------
    const savedTab = readStorage(TAB_KEY);
    showTab(savedTab && tabs.some(tab => tab.dataset.tab === savedTab) ? savedTab : "prepare", {persist: false});
    renderFlipLabel();
    renderStatePill();
    renderAudience();
    renderProfiles();
    renderAudioTests();
    renderCaptions();
    renderSummaryButtons();
    setText(el.captionEndpoint, captionUrl());
    updateButtonsForState(currentState);

    const timers = [];
    function start() {
      refreshPreflight();
      refreshDevices(false);
      refreshStatus();
      refreshVerses();
      connectMetrics();
      probeCapabilities();
      refreshStorage();
      refreshSupportSessions();
      timers.push(setIntervalImpl(() => { if (!ACTIVE_STATES.includes(currentState)) refreshPreflight(); }, PREFLIGHT_INTERVAL_MS));
      timers.push(setIntervalImpl(refreshStatus, STATUS_INTERVAL_MS));
      timers.push(setIntervalImpl(refreshVerses, VERSE_INTERVAL_MS));
      timers.push(setIntervalImpl(() => { renderConnectionAge(); renderElapsed(); if (pendingControl) renderLive(); }, AGE_TICK_MS));
    }

    return {
      start,
      showTab,
      readForm,
      renderStatus,
      refreshStatus,
      refreshPreflight,
      renderChecks,
      refreshDevices,
      populateOutput,
      updateButtonsForState,
      renderMetrics,
      applyCapabilities,
      probeCapabilities,
      refreshStorage,
      refreshSupportSessions,
      renderConnectionAge,
      renderSummaryTask,
      captionModel,
      get state() { return currentState; },
      get connection() { return connection; },
      get preflightOk() { return preflightOk; },
      get activeTab() { return activeTab; },
      get capabilities() { return capabilities; },
      get captionClient() { return captionClient; },
      get effectiveProfile() { return effectiveProfileId(); },
      get pendingControl() { return pendingControl; },
    };
  }

  global.StarkOperator = {
    create, friendlyCheck, readinessSummary, describeState, describeError, audienceLinks,
    formatAge, formatDuration, formatBytes, pickProfiles, defaultProfileId, normalizeProfileId, isLiteProfile,
    profileLabel, humanPhase, describeLevel, ACTIVE_STATES,
  };

  if (!global.__STARK_OPERATOR_MANUAL__ && global.document && global.document.getElementById("config-form")) {
    global.__starkOperatorApp = create({});
    global.__starkOperatorApp.start();
  }
})(typeof window !== "undefined" ? window : globalThis);
