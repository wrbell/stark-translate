// Loads the real operator page (index.html + widgets + app.js) into a Node vm
// context backed by tests/frontend/fake_dom.js, with routable fetch, fake
// WebSockets, recorded timers and storage. Used by the pytest wrappers.

"use strict";

const fs = require("fs");
const path = require("path");
const vm = require("vm");
const {createDom, Event, CustomEvent} = require("./fake_dom.js");

const ROOT = path.resolve(__dirname, "..", "..");
const OPERATOR = path.join(ROOT, "displays", "operator");

function deferred() {
  let resolve, reject;
  const promise = new Promise((res, rej) => { resolve = res; reject = rej; });
  return {promise, resolve, reject};
}

const settle = async (rounds = 6) => {
  for (let i = 0; i < rounds; i++) await new Promise(resolve => setImmediate(resolve));
};

function response(body, status = 200) {
  return {ok: status >= 200 && status < 300, status, statusText: `HTTP ${status}`,
    json: async () => JSON.parse(JSON.stringify(body))};
}

class FakeWebSocket {
  constructor(url) {
    this.url = url;
    this.readyState = 0;
    this.sent = [];
    FakeWebSocket.instances.push(this);
  }
  open() { this.readyState = 1; if (this.onopen) this.onopen({}); }
  message(data) { if (this.onmessage) this.onmessage({data: typeof data === "string" ? data : JSON.stringify(data)}); }
  send(data) { this.sent.push(data); }
  close() {
    if (this.readyState === 3) return;
    this.readyState = 3;
    if (this.onclose) this.onclose({});
  }
  fail() { if (this.onerror) this.onerror({}); }
}
FakeWebSocket.instances = [];

function preflightPayload(overrides = {}) {
  const checks = overrides.checks || [
    {name: "GPU", status: "pass", detail: "Apple Silicon detected (MLX path)"},
    {name: "Runtime dependencies", status: "pass", detail: "mlx dependencies available"},
    {name: "Models", status: "pass", detail: "mlx-parakeet-v3, mlx-gemma4-e4b; Marian CT2"},
    {name: "Microphone", status: "pass", detail: "1 input device(s): USB Mic"},
    {name: "Adapter manifest", status: "warn", detail: "No adapters/manifest.json — running with base models"},
  ];
  const counts = {pass: 0, warn: 0, fail: 0};
  for (const c of checks) counts[c.status] += 1;
  return {checks, ok: counts.fail === 0, status_counts: counts, backend: "mlx", ...overrides};
}

// The capabilities contract emitted by operator_app/main.py (GET /api/capabilities).
function realCapabilities(overrides = {}) {
  const host = overrides.host || "localhost";
  const http = overrides.http || 8080;
  const ws = overrides.ws || 8765;
  const base = `http://${host}:${http}/displays`;
  const caps = {
    profiles: ["standard", "lite-cpu", "lite-cpu-quality", "lite-cuda-8gb"],
    default_profile: "standard",
    preflight_required: true,
    audio_tests: true,
    audio_tests_require_idle: true,
    audio_devices_validated: false,
    support: true,
    storage: true,
    display_ports: {http, websocket: ws},
    audience_urls: {
      audience: `${base}/audience_display.html?port=${ws}`,
      church: `${base}/church_display.html?port=${ws}`,
      mobile: `${base}/mobile_display.html?port=${ws}`,
      obs: `${base}/obs_overlay.html?port=${ws}`,
    },
  };
  for (const key of Object.keys(overrides)) if (!["host", "http", "ws"].includes(key)) caps[key] = overrides[key];
  return caps;
}

// The health block tools/pipeline_health.py publishes and the runner copies into /api/session/status.
function realHealth(overrides = {}) {
  return {
    schema_version: 1, session_id: "s1", updated_at: 1.0, phase: "ready", input_seen: true, input_age_s: 0.4,
    caption_age_s: 2.5, input_level: 0.31, errors: [], error_count: 0, captions: [],
    recording: {audio_enabled: true, required_failures: 0, ok: true}, control_sequence: 0, publish_failures: 0,
    persistence: {ok: true, completed: 3, failed: 0, pending: 0}, storage: {free_bytes: 50 * 1024 ** 3, low_space: false},
    queues: {audio: 0, capture_handoff: 0, finals: 0, stream_tokens: 0}, clients: 1, age_s: 0.2, stale: false,
    ...overrides,
  };
}

function createHarness(options = {}) {
  const html = fs.readFileSync(path.join(OPERATOR, "index.html"), "utf8");
  const dom = createDom(html);
  const storage = new Map(options.storage || []);
  const intervals = [], timeouts = [], fetchLog = [], opened = [], clipboard = [];
  const state = {
    status: options.status || {state: "idle"},
    preflight: options.preflight || preflightPayload(),
    devices: options.devices || {inputs: [{index: 1, name: "USB Mic", channels: 1}], outputs: [], change_seq: 0},
    outputs: options.outputs || {outputs: [{index: 2, name: "Speakers", channels: 2, default: true}]},
    capabilities: options.capabilities === undefined ? null : options.capabilities, // null -> 404
    storage: options.storageInfo === undefined ? null : options.storageInfo,
    verses: {highlights: []},
    reviewSessions: options.reviewSessions || {sessions: []},
  };
  const routes = new Map();
  const pending = [];
  const context = {
    console,
    URLSearchParams,
    __STARK_OPERATOR_MANUAL__: true,
    document: dom.document,
    Event,
    CustomEvent,
    Option: function Option(text, value) {
      const option = dom.document.createElement("option");
      option.textContent = text;
      if (value !== undefined) option.value = value;
      return option;
    },
    localStorage: {
      getItem: key => (storage.has(key) ? storage.get(key) : null),
      setItem: (key, value) => storage.set(key, String(value)),
      removeItem: key => storage.delete(key),
    },
    location: {hostname: "localhost", host: "localhost:9000", protocol: "http:", search: "", href: "http://localhost:9000/operator/", ...(options.location || {})},
    setInterval: (fn, ms) => { intervals.push({fn, ms}); return intervals.length; },
    setTimeout: (fn, ms) => { timeouts.push({fn, ms}); return timeouts.length; },
    clearTimeout: () => {},
    clearInterval: () => {},
    WebSocket: FakeWebSocket,
    navigator: {clipboard: {writeText: async text => { clipboard.push(text); }}},
    open: url => { opened.push(url); },
    performance: {now: () => Date.now()},
    requestAnimationFrame: fn => fn(),
    devicePixelRatio: 1,
    listeners: {},
    addEventListener(type, fn) { (this.listeners[type] ||= []).push(fn); },
    dispatchEvent(event) { for (const fn of this.listeners[event.type] || []) fn(event); return true; },
    fetch: async (url, init = {}) => {
      fetchLog.push({url, init});
      const pathOnly = url.split("?")[0];
      const method = (init.method || "GET").toUpperCase();
      // Prefer the most specific route so "/x/{id}/cancel" is not swallowed by "/x".
      let best = null;
      for (const [key, handler] of routes) {
        const [routeMethod, routePath] = key.split(" ");
        const matches = routeMethod === method && (pathOnly === routePath || pathOnly.startsWith(routePath + "/") ||
          (routePath.endsWith("*") && pathOnly.startsWith(routePath.slice(0, -1))));
        if (matches && (!best || routePath.length > best.routePath.length)) best = {routePath, handler};
      }
      if (best) return best.handler(url, init);
      if (method === "GET" && pathOnly === "/api/preflight") return response(state.preflight);
      if (method === "GET" && pathOnly === "/api/session/status") return response(state.status);
      if (method === "GET" && pathOnly === "/api/devices") return response(state.devices);
      if (method === "GET" && pathOnly === "/api/audio/output-devices") return response(state.outputs);
      if (method === "GET" && pathOnly === "/api/features/verses") return response(state.verses);
      if (method === "GET" && pathOnly === "/api/review/sessions") return response(state.reviewSessions);
      if (method === "GET" && pathOnly === "/api/capabilities") {
        return state.capabilities ? response(state.capabilities) : response({detail: "Not Found"}, 404);
      }
      if (method === "GET" && pathOnly === "/api/storage") {
        return state.storage ? response(state.storage) : response({detail: "Not Found"}, 404);
      }
      return response({detail: "Not Found"}, 404);
    },
  };
  context.window = context;
  context.globalThis = context;
  context.self = context;
  vm.createContext(context);
  for (const file of ["widgets/sparkline.js", "widgets/qr.js", "widgets/captions.js", "app.js"]) {
    vm.runInContext(fs.readFileSync(path.join(OPERATOR, file), "utf8"), context, {filename: file});
  }
  if (options.loadReview) {
    vm.runInContext(fs.readFileSync(path.join(OPERATOR, "review.js"), "utf8"), context, {filename: "review.js"});
  }
  const clock = {value: Date.now()};
  const app = context.StarkOperator.create({now: () => clock.value});
  const el = id => {
    const node = dom.document.getElementById(id);
    if (!node) throw new Error(`missing element #${id}`);
    return node;
  };
  return {
    app, context, dom, document: dom.document, el, state, storage, intervals, timeouts, fetchLog, opened, clipboard, clock,
    FakeWebSocket, response, deferred, settle, preflightPayload, pending,
    route(method, routePath, handler) { routes.set(`${method.toUpperCase()} ${routePath}`, handler); },
    unroute(method, routePath) { routes.delete(`${method.toUpperCase()} ${routePath}`); },
    lastFetch(pathPrefix, method = "POST") {
      return fetchLog.filter(f => f.url.startsWith(pathPrefix) && (f.init.method || "GET").toUpperCase() === method).pop();
    },
    text: id => el(id).textContent.trim(),
    panelText: name => el(`panel-${name}`).textContent.replace(/\s+/g, " ").trim(),
    runIntervals: () => intervals.map(t => t.fn()),
    runTimeouts: () => { const list = timeouts.splice(0); for (const t of list) t.fn(); },
    sockets: () => FakeWebSocket.instances,
    async start() { app.start(); await settle(); },
    async setStatus(status) { state.status = status; await app.refreshStatus(); await settle(); },
  };
}

module.exports = {createHarness, settle, deferred, response, FakeWebSocket, preflightPayload, realCapabilities, realHealth, ROOT};
