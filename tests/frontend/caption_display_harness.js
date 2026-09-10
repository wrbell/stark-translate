// Runs shipped pages and socket/telemetry handlers; only browser APIs are simulated.
"use strict";
const fs = require('fs'), path = require('path'), vm = require('vm');
const {createDom} = require('./fake_dom.js');
const ROOT = path.resolve(__dirname, '..', '..');
function createDisplay(name) {
  const html = fs.readFileSync(path.join(ROOT, 'displays', name + '.html'), 'utf8');
  const dom = createDom(html), frames = [], sockets = [];
  dom.document.visibilityState = 'visible';
  const elementPrototype = Object.getPrototypeOf(dom.document.body);
  if (!Object.getOwnPropertyDescriptor(elementPrototype, 'firstChild')) {
    Object.defineProperty(elementPrototype, 'firstChild', {get() { return this.childNodes[0] || null; }});
  }
  class WebSocket {
    constructor(url) { this.url = url; this.readyState = 1; this.sent = []; sockets.push(this); }
    send(data) { this.sent.push(JSON.parse(data)); }
    close() { this.readyState = 3; }
  }
  class MutationObserver {
    observe() { this.before = dom.document.body.innerHTML; }
    takeRecords() { return this.before === dom.document.body.innerHTML ? [] : [{}]; }
    disconnect() {}
  }
  const ctx = {document: dom.document, console, URL, URLSearchParams, WebSocket, MutationObserver,
    location: new URL('http://localhost/displays/' + name + '.html?english=1&lines=10'),
    navigator: {}, performance: {now: () => 123}, innerHeight: 900,
    setTimeout: () => 1, clearTimeout: () => {}, setInterval: () => 1, clearInterval: () => {},
    requestAnimationFrame: fn => frames.push(fn), addEventListener: () => {}};
  ctx.window = ctx; vm.createContext(ctx);
  for (const match of html.matchAll(/<script\b([^>]*)>([\s\S]*?)<\/script>/g)) {
    const src = match[1].match(/src="([^"]+)"/);
    vm.runInContext(src ? fs.readFileSync(path.join(ROOT, 'displays', src[1]), 'utf8') : match[2], ctx,
      {filename: src ? src[1] : name + '.html'});
  }
  const socket = sockets[0];
  if (!socket || !socket.onmessage) throw new Error('Production page did not connect');
  let sequence = 0;
  function send(message) {
    socket.onmessage({data: JSON.stringify({event_id: 'test:' + (++sequence), ...message})});
    while (frames.length) frames.shift()();
  }
  const ids = {audience_display: ['english', 'spanish'], ab_display: ['a-en', 'a-es', 'b-en', 'b-es', 'c-en', 'c-es'],
    church_display: ['englishChunks', 'spanishChunks'], mobile_display: ['englishChunks', 'spanishChunks'],
    obs_overlay: ['englishText', 'spanishText']}[name];
  return {send, socket, document: dom.document, text: () => ids.map(id => dom.document.getElementById(id).textContent).join('\n')};
}
module.exports = {createDisplay};
