"""Run the shared browser protocol helper with deterministic DOM/RAF boundaries."""

import shutil
import subprocess
from pathlib import Path

import pytest


def test_visible_render_ack_and_hidden_or_unchanged_suppression():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is needed to execute the standalone browser helper")
    script = r"""
const fs = require('fs'), vm = require('vm'), assert = require('assert');
let raf = [], observers = new Set(), sent = [], now = 10;
function mutate() { for (const observer of observers) observer.records.push({}); }
function drainFrames() { while (raf.length) { now += 16; raf.shift()(); } }
const ctx = {
  window: {}, document: {visibilityState: 'visible', body: {}},
  performance: {now: () => now},
  requestAnimationFrame: fn => raf.push(fn),
  MutationObserver: class {
    constructor(fn) { this.fn = fn; this.records = []; }
    observe() { observers.add(this); }
    takeRecords() { const result = this.records; this.records = []; return result; }
    disconnect() { observers.delete(this); this.records = []; }
  }
};
vm.createContext(ctx);
vm.runInContext(fs.readFileSync('displays/caption_telemetry.js', 'utf8'), ctx);
const socket = {readyState: 1, send: data => sent.push(JSON.parse(data))};
function run(visible, shouldMutate) {
  ctx.document.visibilityState = visible;
  const handler = ctx.window.StarkCaptionTelemetry.wrap(socket, () => { if (shouldMutate) mutate(); });
  handler({data: JSON.stringify({type: 'translation', stage: 'complete', event_id: 'session:2'})});
  drainFrames();
}
run('visible', true);
assert.strictEqual(sent.length, 1);
assert.strictEqual(sent[0].receive_to_render_ms, 32);
assert.strictEqual(sent[0].event_id, 'session:2');
assert.strictEqual(sent[0].visible, true);
assert(!('browser_time' in sent[0]));
run('hidden', true);
assert.strictEqual(sent.length, 1);
run('visible', false);
assert.strictEqual(sent.length, 1);
// A stale event ignored by its display must not borrow later DOM mutations.
const stale = ctx.window.StarkCaptionTelemetry.wrap(socket, () => {});
stale({data: JSON.stringify({type: 'translation', stage: 'partial', event_id: 'session:stale'})});
mutate(); // unrelated timer/status update
run('visible', true); // a later caption also changes the DOM before the next frame
assert.strictEqual(sent.length, 2);
assert(!sent.some(event => event.event_id === 'session:stale'));
assert.strictEqual(observers.size, 0);
// Visibility is checked again at the render opportunity.
const hiddenBeforeFrame = ctx.window.StarkCaptionTelemetry.wrap(socket, mutate);
hiddenBeforeFrame({data: JSON.stringify({type: 'translation', event_id: 'session:hidden'})});
ctx.document.visibilityState = 'hidden';
drainFrames();
assert.strictEqual(sent.length, 2);
socket.readyState = 3;
run('visible', true);
assert.strictEqual(sent.length, 2);
assert.strictEqual(observers.size, 0);
for (const name of ['audience_display','ab_display','mobile_display','church_display','obs_overlay']) {
  const html = fs.readFileSync('displays/' + name + '.html', 'utf8');
  assert(html.includes('src="display_connection.js"'));
  assert(html.includes('StarkDisplayConnection.websocketUrl('));
  assert(html.includes('src="caption_telemetry.js"'));
  assert(html.includes('StarkCaptionTelemetry.wrap(ws,'));
  for (const match of html.matchAll(/<script(?:\s[^>]*)?>([\s\S]*?)<\/script>/g)) new vm.Script(match[1]);
}
"""
    subprocess.run([node, "-e", script], cwd=Path(__file__).resolve().parents[1], check=True, capture_output=True)


def test_display_connections_validate_ports_and_preserve_mobile_origin():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is needed to execute the standalone browser helper")
    script = r"""
const fs = require('fs'), vm = require('vm'), assert = require('assert');
const ctx = {window: {}, URL, URLSearchParams};
vm.createContext(ctx);
vm.runInContext(fs.readFileSync('displays/display_connection.js', 'utf8'), ctx);
const config = ctx.window.StarkDisplayConnection;
for (const value of ['', '0', '-1', '65536', '123abc', '12.3', '1e3', 'Infinity', 'NaN', '%202']) {
  assert.strictEqual(config.port(new URL('http://localhost:8080/audience_display.html?port=' + value)), 8765);
}
for (const value of ['1', '8766', '65535']) {
  const page = new URL('http://localhost:8082/audience_display.html?port=' + value);
  assert.strictEqual(config.websocketUrl(page), 'ws://localhost:' + value);
  assert.strictEqual(config.mobileUrl(page), 'http://localhost:8082/mobile_display.html?port=' + value);
}
const securePage = new URL('https://captions.example/displays/audience_display.html?port=9999');
assert.strictEqual(config.websocketUrl(securePage), 'wss://captions.example:9999');
assert.strictEqual(config.mobileUrl(securePage), 'https://captions.example/displays/mobile_display.html?port=9999');
assert.strictEqual(config.websocketUrl(new URL('http://[::1]/?port=1')), 'ws://[::1]:1');
assert(fs.readFileSync('displays/audience_display.html', 'utf8').includes('StarkDisplayConnection.mobileUrl(location)'));
"""
    subprocess.run([node, "-e", script], cwd=Path(__file__).resolve().parents[1], check=True, capture_output=True)
