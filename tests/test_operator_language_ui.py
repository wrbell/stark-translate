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
const {createHarness, deferred, response, preflightPayload} = require(process.cwd() + '/tests/frontend/operator_harness.js');
const assert = require('assert');
const status = (state, lang) => ({state, config: {lang}});
const preflights = h => h.fetchLog.filter(f => f.url.startsWith('/api/preflight'));
const lastPreflightLang = h => new URLSearchParams(preflights(h).at(-1).url.split('?')[1]).get('lang');
const startBody = h => JSON.parse(h.lastFetch('/api/session/start').init.body);
(async () => {
  const h = createHarness();
  h.app.renderChecks(preflightPayload());
  let nextPost = status('idle', 'es');
  for (const path of ['/api/session/start', '/api/session/stop', '/api/control/lang_flip']) h.route('POST', path, () => response(nextPost));
  const select = h.el('lang-select');
  h.app.renderStatus(status('running', 'es')); // Reload into an active ES session.
  assert.strictEqual(select.value, 'es');
  assert.strictEqual(select.disabled, true);
  assert.strictEqual(lastPreflightLang(h), 'es');
  assert.strictEqual(h.app.readForm().lang, 'es'); // Disabled controls are absent from FormData.
  nextPost = status('idle', 'es');
  await h.el('stop-btn').click();
  assert.strictEqual(select.disabled, false);
  nextPost = status('starting', 'es');
  await h.el('start-btn').click();
  assert.strictEqual(startBody(h).lang, 'es');

  const slow = deferred();
  h.route('GET', '/api/session/status', () => slow.promise);
  const oldPoll = h.app.refreshStatus();
  nextPost = status('starting', 'en');
  await h.el('flip-btn').click();
  assert.strictEqual(select.value, 'en');
  assert.strictEqual(lastPreflightLang(h), 'en');
  slow.resolve(response(status('running', 'es')));
  await oldPoll;
  assert.strictEqual(select.value, 'en'); // A stale poll cannot undo the flip.
  const before = preflights(h).length;
  await h.el('flip-btn').click();
  assert.strictEqual(preflights(h).length, before + 1); // Recheck even if a poll already synced it.

  h.app.renderStatus(status('paused', 'en'));
  assert.strictEqual(select.disabled, true);
  h.app.renderStatus(status('error', 'en'));
  select.value = 'es';
  await select.fire('change');
  h.app.renderStatus(status('idle', 'en'));
  assert.strictEqual(select.value, 'es'); // Preserve a deliberate next-session choice.
  nextPost = status('starting', 'es');
  await h.el('start-btn').click();
  assert.strictEqual(startBody(h).lang, 'es');

  const idleReload = createHarness();
  idleReload.app.renderStatus(status('idle', 'es'));
  assert.strictEqual(idleReload.el('lang-select').value, 'es');
  const earlyChoice = createHarness();
  earlyChoice.el('lang-select').value = 'es';
  await earlyChoice.el('lang-select').fire('change');
  earlyChoice.app.renderStatus(status('idle', 'en'));
  assert.strictEqual(earlyChoice.el('lang-select').value, 'es');

  const t = createHarness();
  const c = name => t.el('config-form').elements.namedItem(name);
  const activeConfig = {lang: 'es', backend: 'mlx', engine: 'auto', tts: true, run_ab: false,
    diarize: true, vad_threshold: 0.4, mic_device: 0, tts_output_mode: 'local', tts_device: 0,
    tts_device_en: 17, tts_device_es: 'Church Spanish Speakers'};
  const active = {state: 'running', config: activeConfig};
  t.app.renderStatus(active);
  assert.strictEqual(c('tts').checked, true);
  assert.strictEqual(c('tts_output_mode').value, 'local');
  assert.strictEqual(c('output_device').value, '0');
  assert.strictEqual(c('tts_device_en').value, '17');
  assert.strictEqual(c('tts_device_es').value, 'Church Spanish Speakers');
  assert(Array.from(t.el('config-form').elements).every(control => control.disabled));
  const payload = t.app.readForm();
  assert.strictEqual(payload.tts, true);
  assert.strictEqual(payload.backend, 'mlx');
  assert.strictEqual(payload.diarize, true);
  assert.strictEqual(payload.mic_device, 0);
  assert.strictEqual(payload.tts_device, 0);
  assert.strictEqual(payload.tts_device_en, 17); // Keep numeric IDs distinct from names.
  assert.strictEqual(payload.tts_device_es, 'Church Spanish Speakers');
  // Device enumeration arrives after status, with those outputs disconnected.
  t.app.populateOutput(c('tts_device_es'), [], true);
  t.app.populateOutput(c('output_device'), [], false);
  assert.strictEqual(c('tts_device_es').value, 'Church Spanish Speakers');
  assert.strictEqual(c('output_device').value, '0');
  t.app.renderStatus({...active, state: 'idle'});
  assert(Array.from(t.el('config-form').elements).every(control => !control.disabled));
  t.route('POST', '/api/session/start', () => response({...active, state: 'starting'}));
  await t.el('start-btn').click();
  assert.strictEqual(startBody(t).tts, true);
  assert.strictEqual(startBody(t).tts_device_en, 17);
  t.app.renderStatus({...active, state: 'idle'});
  c('tts').checked = false;
  await c('tts').fire('change');
  c('tts_output_mode').value = 'ws';
  await c('tts_output_mode').fire('change');
  c('tts_device_es').value = '';
  await c('tts_device_es').fire('change');
  t.app.renderStatus({...active, state: 'idle'});
  assert.strictEqual(c('tts').checked, false);
  assert.strictEqual(c('tts_output_mode').value, 'ws');
  assert.strictEqual(c('tts_device_es').value, '');
  const restored = createHarness({storage: [...t.storage]});
  restored.app.renderStatus({...active, state: 'idle'});
  assert.strictEqual(restored.app.readForm().tts, false);
  assert.strictEqual(restored.app.readForm().tts_output_mode, 'ws');
  assert.strictEqual(restored.app.readForm().tts_device_es, undefined); // Explicitly cleared route survives reload.
  assert.strictEqual(restored.app.readForm().tts_device_en, 17);
  // A confirmed active configuration supersedes those saved next-start choices.
  restored.app.renderStatus(active);
  assert.strictEqual(restored.app.readForm().tts, true);
  assert.strictEqual(restored.app.readForm().tts_output_mode, 'local');
  const reloadedActive = createHarness({storage: [...restored.storage]});
  assert.strictEqual(reloadedActive.el('config-form').elements.namedItem('tts').checked, true);

  // A numeric-looking route selected by the user is a device name, not an ID.
  restored.app.renderStatus({...active, state: 'idle'});
  await restored.el('config-form').elements.namedItem('tts_device_en').fire('change');
  assert.strictEqual(restored.app.readForm().tts_device_en, '17');

  // A slower EN preflight must not overwrite the completed ES preflight.
  const p = createHarness();
  const requests = [];
  p.route('GET', '/api/preflight', url => new Promise(resolve => requests.push({url, resolve})));
  p.el('lang-select').value = 'en';
  const en = p.app.refreshPreflight();
  p.el('lang-select').value = 'es';
  const es = p.app.refreshPreflight();
  assert(requests[1].url.includes('lang=es'));
  requests[1].resolve(response(preflightPayload({checks: [{name: 'Models', status: 'pass', detail: 'Whisper ES'}]})));
  await es;
  requests[0].resolve(response(preflightPayload({checks: [{name: 'Models', status: 'pass', detail: 'Parakeet EN'}]})));
  await en;
  assert(p.text('checks').includes('Whisper ES'));
  assert(!p.text('checks').includes('Parakeet EN'));
})().catch(error => {console.error(error); process.exit(1);});
"""
    subprocess.run([node, "-e", script], cwd=Path(__file__).resolve().parents[1], check=True, timeout=30)
