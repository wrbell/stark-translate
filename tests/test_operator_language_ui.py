"""Exercise the operator's real JavaScript language controls without a server."""

import shutil
import subprocess
from pathlib import Path

import pytest


def test_confirmed_language_controls_preflight_and_next_start():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is needed to execute the operator UI")
    script = r"""
const fs = require('fs'), vm = require('vm'), assert = require('assert');
const source = fs.readFileSync('displays/operator/app.js', 'utf8');
function makeContext() {
  const select = {value: 'en', handlers: {}, addEventListener(name, fn) {this.handlers[name] = fn;}};
  const button = () => ({handlers: {}, addEventListener(name, fn) {this.handlers[name] = fn;}});
  const context = {
    currentState: 'idle', select, form: {elements: {namedItem: () => select}},
    statusDetailEl: {}, setStatePill() {}, updateButtonsForState() {},
    document: {getElementById: () => null}, window: {dispatchEvent() {}},
    CustomEvent: class {}, FormData: class {get(name) {return name === 'backend' ? 'mlx' : null;}},
    startBtn: button(), stopBtn: button(), pauseBtn: button(), resumeBtn: button(), flipBtn: button(),
    fallbackBtn: button(), preflights: [], posts: [],
  };
  context.refreshPreflight = () => {context.preflights.push(context.select.value);};
  context.postJson = async (url, body) => {context.posts.push({url, body}); return context.nextPost;};
  vm.createContext(context);
  const begin = source.indexOf('  // ---- session status ----');
  const end = source.indexOf('  // ---- live metrics over', begin);
  vm.runInContext(source.slice(begin, end), context);
  return context;
}
const status = (state, lang) => ({state, config: {lang}});
(async () => {
  const c = makeContext();
  c.renderStatus(status('running', 'es')); // Reload into an active ES session.
  assert.strictEqual(c.select.value, 'es');
  assert.strictEqual(c.select.disabled, true);
  assert.strictEqual(c.preflights.at(-1), 'es');
  assert.strictEqual(c.readForm().lang, 'es'); // Disabled controls are absent from FormData.
  c.nextPost = status('idle', 'es');
  await c.stopBtn.handlers.click();
  assert.strictEqual(c.select.disabled, false);
  c.nextPost = status('starting', 'es');
  await c.startBtn.handlers.click();
  assert.strictEqual(c.posts.at(-1).body.lang, 'es');

  let resolveOldStatus;
  c.getJson = () => new Promise(resolve => {resolveOldStatus = resolve;});
  const oldPoll = c.refreshStatus();
  c.nextPost = status('starting', 'en');
  await c.flipBtn.handlers.click();
  assert.strictEqual(c.select.value, 'en');
  assert.strictEqual(c.preflights.at(-1), 'en');
  resolveOldStatus(status('running', 'es'));
  await oldPoll;
  assert.strictEqual(c.select.value, 'en'); // A stale poll cannot undo the flip.
  const before = c.preflights.length;
  await c.flipBtn.handlers.click();
  assert.strictEqual(c.preflights.length, before + 1); // Recheck even if a poll already synced it.

  c.renderStatus(status('paused', 'en'));
  assert.strictEqual(c.select.disabled, true);
  c.renderStatus(status('error', 'en'));
  c.select.value = 'es';
  c.select.handlers.change();
  c.renderStatus(status('idle', 'en'));
  assert.strictEqual(c.select.value, 'es'); // Preserve a deliberate next-session choice.
  c.nextPost = status('starting', 'es');
  await c.startBtn.handlers.click();
  assert.strictEqual(c.posts.at(-1).body.lang, 'es');

  const idleReload = makeContext();
  idleReload.renderStatus(status('idle', 'es'));
  assert.strictEqual(idleReload.select.value, 'es');
  const earlyChoice = makeContext();
  earlyChoice.select.value = 'es';
  earlyChoice.select.handlers.change();
  earlyChoice.renderStatus(status('idle', 'en'));
  assert.strictEqual(earlyChoice.select.value, 'es');

  // A slower EN preflight must not overwrite the completed ES preflight.
  const requests = [], rendered = [];
  const p = {preflightRequest: 0, URLSearchParams, selected: 'en', preflightMetaEl: {},
    readForm() {return {backend: 'mlx', lang: p.selected, tts: false, diarize: false};},
    getJson(url) {return new Promise(resolve => requests.push({url, resolve}));},
    renderChecks(data) {rendered.push(data);},
  };
  vm.createContext(p);
  const first = source.indexOf('  async function refreshPreflight()');
  const last = source.indexOf('  // ---- mic + output devices ----', first);
  vm.runInContext(source.slice(first, last), p);
  const en = p.refreshPreflight();
  p.selected = 'es';
  const es = p.refreshPreflight();
  assert(requests[1].url.includes('lang=es'));
  requests[1].resolve('Whisper ES');
  await es;
  requests[0].resolve('Parakeet EN');
  await en;
  assert.deepStrictEqual(rendered, ['Whisper ES']);
})().catch(error => {console.error(error); process.exit(1);});
"""
    subprocess.run([node, "-e", script], cwd=Path(__file__).resolve().parents[1], check=True, timeout=10)
