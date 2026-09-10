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
function makeContext(storage = new Map()) {
  const control = (name, choices, initial = '', type = 'select-one') => ({
    name, type, tagName: choices ? 'SELECT' : 'INPUT', _value: initial, checked: false,
    options: (choices || []).map(value => ({value})), dataset: {}, handlers: {},
    get value() {return this._value;},
    set value(value) {this._value = this.tagName !== 'SELECT' || this.options.some(o => o.value === String(value)) ? String(value) : '';},
    set innerHTML(value) {this.options = []; this._value = '';},
    add(option) {this.options.push(option);},
    addEventListener(name, fn) {
      const before = this.handlers[name];
      this.handlers[name] = before ? event => {before(event); fn(event);} : fn;
    },
  });
  const controls = {
    lang: control('lang', ['en', 'es'], 'en'), backend: control('backend', ['auto', 'mlx', 'cuda'], 'auto'),
    engine: control('engine', ['auto', 'hf', 'llamacpp'], 'auto'),
    mic_device: control('mic_device', ['']), output_device: control('output_device', ['']),
    tts_device_en: control('tts_device_en', ['']), tts_device_es: control('tts_device_es', ['']),
    tts_output_mode: control('tts_output_mode', ['ws', 'wav', 'both', 'local'], 'ws'),
    vad_threshold: control('vad_threshold', null, '0.3', 'number'),
    tts: control('tts', null, 'on', 'checkbox'), run_ab: control('run_ab', null, 'on', 'checkbox'),
    diarize: control('diarize', null, 'on', 'checkbox'),
  };
  const elements = Object.values(controls);
  elements.namedItem = name => controls[name];
  const select = controls.lang;
  const button = () => ({handlers: {}, addEventListener(name, fn) {this.handlers[name] = fn;}});
  const context = {
    currentState: 'idle', select, controls, storage, form: {elements}, idleEditedFields: new Set(),
    statusDetailEl: {}, setStatePill() {}, updateButtonsForState() {},
    document: {getElementById: id => ({'output-device': controls.output_device,
      'output-device-en': controls.tts_device_en, 'output-device-es': controls.tts_device_es})[id] || null},
    window: {dispatchEvent() {}}, CustomEvent: class {},
    Option: class {constructor(text, value) {this.text = text; this.value = String(value);}},
    localStorage: {getItem: key => storage.get(key), setItem: (key, value) => storage.set(key, value)},
    startBtn: button(), stopBtn: button(), pauseBtn: button(), resumeBtn: button(), flipBtn: button(),
    fallbackBtn: button(), preflights: [], posts: [],
  };
  context.refreshPreflight = () => {context.preflights.push(context.select.value);};
  context.postJson = async (url, body) => {context.posts.push({url, body}); return context.nextPost;};
  vm.createContext(context);
  const devices = source.indexOf('  // ---- mic + output devices ----');
  const begin = source.indexOf('  // ---- session status ----');
  const end = source.indexOf('  // ---- live metrics over', begin);
  vm.runInContext(source.slice(devices, begin), context);
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

  const t = makeContext();
  const activeConfig = {lang: 'es', backend: 'mlx', engine: 'auto', tts: true, run_ab: false,
    diarize: true, vad_threshold: 0.4, mic_device: 0, tts_output_mode: 'local', tts_device: 0,
    tts_device_en: 17, tts_device_es: 'Church Spanish Speakers'};
  const active = {state: 'running', config: activeConfig};
  t.renderStatus(active);
  assert.strictEqual(t.controls.tts.checked, true);
  assert.strictEqual(t.controls.tts_output_mode.value, 'local');
  assert.strictEqual(t.controls.output_device.value, '0');
  assert.strictEqual(t.controls.tts_device_en.value, '17');
  assert.strictEqual(t.controls.tts_device_es.value, 'Church Spanish Speakers');
  assert(Object.values(t.controls).every(control => control.disabled));
  const payload = t.readForm();
  assert.strictEqual(payload.tts, true);
  assert.strictEqual(payload.backend, 'mlx');
  assert.strictEqual(payload.diarize, true);
  assert.strictEqual(payload.mic_device, 0);
  assert.strictEqual(payload.tts_device, 0);
  assert.strictEqual(payload.tts_device_en, 17); // Keep numeric IDs distinct from names.
  assert.strictEqual(payload.tts_device_es, 'Church Spanish Speakers');
  // Device enumeration arrives after status, with those outputs disconnected.
  t.populateOutput(t.controls.tts_device_es, [], true);
  t.populateOutput(t.controls.output_device, [], false);
  assert.strictEqual(t.controls.tts_device_es.value, 'Church Spanish Speakers');
  assert.strictEqual(t.controls.output_device.value, '0');
  t.renderStatus({...active, state: 'idle'});
  assert(Object.values(t.controls).every(control => !control.disabled));
  t.nextPost = {...active, state: 'starting'};
  await t.startBtn.handlers.click();
  assert.strictEqual(t.posts.at(-1).body.tts, true);
  assert.strictEqual(t.posts.at(-1).body.tts_device_en, 17);
  t.renderStatus({...active, state: 'idle'});
  t.controls.tts.checked = false;
  t.controls.tts.handlers.change();
  t.controls.tts_output_mode.value = 'ws';
  t.controls.tts_output_mode.handlers.change();
  t.controls.tts_device_es.value = '';
  t.controls.tts_device_es.handlers.change();
  t.renderStatus({...active, state: 'idle'});
  assert.strictEqual(t.controls.tts.checked, false);
  assert.strictEqual(t.controls.tts_output_mode.value, 'ws');
  assert.strictEqual(t.controls.tts_device_es.value, '');
  const restored = makeContext(t.storage);
  restored.renderStatus({...active, state: 'idle'});
  assert.strictEqual(restored.readForm().tts, false);
  assert.strictEqual(restored.readForm().tts_output_mode, 'ws');
  assert.strictEqual(restored.readForm().tts_device_es, undefined); // Explicitly cleared route survives reload.
  assert.strictEqual(restored.readForm().tts_device_en, 17);
  // A confirmed active configuration supersedes those saved next-start choices.
  restored.renderStatus(active);
  assert.strictEqual(restored.readForm().tts, true);
  assert.strictEqual(restored.readForm().tts_output_mode, 'local');
  const reloadedActive = makeContext(restored.storage);
  assert.strictEqual(reloadedActive.controls.tts.checked, true);

  // A numeric-looking route selected by the user is a device name, not an ID.
  restored.renderStatus({...active, state: 'idle'});
  restored.controls.tts_device_en.handlers.change();
  assert.strictEqual(restored.readForm().tts_device_en, '17');

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
