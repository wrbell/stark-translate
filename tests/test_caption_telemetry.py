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
let raf = [], mutations = [], sent = [], now = 10;
const ctx = {
  window: {}, document: {visibilityState: 'visible', body: {}},
  performance: {now: () => now}, setTimeout: () => 1,
  requestAnimationFrame: fn => raf.push(fn),
  MutationObserver: class {
    constructor(fn) { this.fn = fn; }
    observe() {}
    takeRecords() { const result = mutations; mutations = []; return result; }
    disconnect() {}
  }
};
vm.createContext(ctx);
vm.runInContext(fs.readFileSync('displays/caption_telemetry.js', 'utf8'), ctx);
const socket = {readyState: 1, send: data => sent.push(JSON.parse(data))};
function run(visible, mutate) {
  ctx.document.visibilityState = visible;
  const handler = ctx.window.StarkCaptionTelemetry.wrap(socket, () => { if (mutate) mutations.push({}); });
  handler({data: JSON.stringify({type: 'translation', stage: 'complete', event_id: 'session:2'})});
  while (raf.length) { now += 16; raf.shift()(); }
}
run('visible', true);
assert.strictEqual(sent.length, 1);
assert.strictEqual(sent[0].receive_to_render_ms, 32);
assert.strictEqual(sent[0].event_id, 'session:2');
assert.strictEqual(sent[0].visible, true);
assert(!('browser_time' in sent[0]));
run('hidden', true);
assert.strictEqual(sent.length, 1);
mutations = [];
run('visible', false);
assert.strictEqual(sent.length, 1);
socket.readyState = 3;
run('visible', true);
assert.strictEqual(sent.length, 1);
for (const name of ['audience_display','ab_display','mobile_display','church_display','obs_overlay']) {
  const html = fs.readFileSync('displays/' + name + '.html', 'utf8');
  assert(html.includes('src="caption_telemetry.js"'));
  assert(html.includes('StarkCaptionTelemetry.wrap(ws,'));
  for (const match of html.matchAll(/<script(?:\s[^>]*)?>([\s\S]*?)<\/script>/g)) new vm.Script(match[1]);
}
"""
    subprocess.run([node, "-e", script], cwd=Path(__file__).resolve().parents[1], check=True, capture_output=True)
