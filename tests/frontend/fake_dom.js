// Minimal DOM for exercising displays/operator/*.js under Node without jsdom.
//
// Parses the real index.html into a tree and implements the handful of DOM
// APIs the operator scripts use: ids, classes, datasets, attributes, form
// controls (select/option/checkbox semantics), simple selectors, bubbling
// events, and canvas stubs. It is deliberately small; extend it when a script
// starts using something new rather than reaching for a browser.

"use strict";

const VOID = new Set(["meta", "link", "input", "br", "hr", "img", "source", "wbr", "base", "col", "area", "embed", "param", "track"]);
const RAW_TEXT = new Set(["script", "style"]);
const FORM_ASSOCIATED = new Set(["INPUT", "SELECT", "TEXTAREA", "BUTTON", "FIELDSET", "OUTPUT", "OBJECT"]);
const ENTITIES = {amp: "&", lt: "<", gt: ">", quot: '"', apos: "'", nbsp: " ", rarr: "→", harr: "↔", mdash: "—", hellip: "…"};

function decode(text) {
  return text.replace(/&(#x[0-9a-f]+|#\d+|[a-z]+);/gi, (match, body) => {
    if (body[0] === "#") {
      const code = body[1].toLowerCase() === "x" ? parseInt(body.slice(2), 16) : parseInt(body.slice(1), 10);
      return Number.isFinite(code) ? String.fromCodePoint(code) : match;
    }
    return Object.hasOwn(ENTITIES, body) ? ENTITIES[body] : match;
  });
}

class Event {
  constructor(type, init) {
    this.type = type;
    this.bubbles = init && init.bubbles !== undefined ? !!init.bubbles : true;
    this.defaultPrevented = false;
    this.target = null;
    this.currentTarget = null;
    if (init) for (const key of Object.keys(init)) if (!(key in this)) this[key] = init[key];
  }
  preventDefault() { this.defaultPrevented = true; }
  stopPropagation() { this.bubbles = false; }
}

class CustomEvent extends Event {
  constructor(type, init) {
    super(type, init);
    this.detail = init ? init.detail : undefined;
  }
}

class TextNode {
  constructor(text, doc) {
    this.nodeType = 3;
    this.data = String(text);
    this.parentNode = null;
    this.ownerDocument = doc;
  }
  get textContent() { return this.data; }
  set textContent(value) { this.data = String(value); }
}

class ClassList {
  constructor(element) { this.element = element; }
  _list() { return (this.element.getAttribute("class") || "").split(/\s+/).filter(Boolean); }
  _write(list) { this.element.setAttribute("class", list.join(" ")); }
  add(...names) { const list = this._list(); for (const n of names) if (!list.includes(n)) list.push(n); this._write(list); }
  remove(...names) { this._write(this._list().filter(n => !names.includes(n))); }
  contains(name) { return this._list().includes(name); }
  toggle(name, force) {
    const has = this.contains(name);
    const want = force === undefined ? !has : !!force;
    if (want && !has) this.add(name);
    if (!want && has) this.remove(name);
    return want;
  }
}

function matchesCompound(element, compound) {
  let rest = compound;
  const tag = /^[a-zA-Z][\w-]*/.exec(rest);
  if (tag) {
    if (element.tagName !== tag[0].toUpperCase()) return false;
    rest = rest.slice(tag[0].length);
  }
  const parts = rest.match(/(#[\w-]+|\.[\w-]+|\[[^\]]+\])/g) || [];
  for (const part of parts) {
    if (part[0] === "#") { if (element.id !== part.slice(1)) return false; }
    else if (part[0] === ".") { if (!element.classList.contains(part.slice(1))) return false; }
    else {
      const m = /^\[([\w-]+)(?:=("([^"]*)"|'([^']*)'|([^\]]*)))?\]$/.exec(part);
      if (!m) return false;
      const value = m[3] !== undefined ? m[3] : m[4] !== undefined ? m[4] : m[5];
      if (!element.hasAttribute(m[1])) return false;
      if (value !== undefined && element.getAttribute(m[1]) !== value) return false;
    }
  }
  return true;
}

function matchesSelector(element, selector) {
  return selector.split(",").some(alternative => {
    const compounds = alternative.trim().split(/\s+/).filter(Boolean);
    if (!compounds.length) return false;
    if (!matchesCompound(element, compounds[compounds.length - 1])) return false;
    let node = element.parentNode;
    let index = compounds.length - 2;
    while (index >= 0 && node && node.nodeType === 1) {
      if (matchesCompound(node, compounds[index])) index -= 1;
      node = node.parentNode;
    }
    return index < 0;
  });
}

class Element {
  constructor(tagName, doc) {
    this.nodeType = 1;
    this.tagName = tagName.toUpperCase();
    this.attributes = new Map();
    this.childNodes = [];
    this.parentNode = null;
    this.ownerDocument = doc;
    this.listeners = {};
    this.style = {};
    this._value = undefined;
    this._checked = false;
    this.classList = new ClassList(this);
    this.dataset = new Proxy({}, {
      get: (_, key) => (typeof key === "string" ? this.getAttribute("data-" + dash(key)) ?? undefined : undefined),
      set: (_, key, value) => { this.setAttribute("data-" + dash(key), String(value)); return true; },
      deleteProperty: (_, key) => { this.removeAttribute("data-" + dash(key)); return true; },
      has: (_, key) => this.hasAttribute("data-" + dash(key)),
    });
    if (this.tagName === "AUDIO") { this.paused = true; this.pauseCalls = 0; }
    if (this.tagName === "CANVAS") this._context = makeContext();
  }

  // -- attributes --
  getAttribute(name) { return this.attributes.has(name) ? this.attributes.get(name) : null; }
  setAttribute(name, value) { this.attributes.set(name, String(value)); }
  removeAttribute(name) { this.attributes.delete(name); if (name === "src" || name === "href") delete this["_" + name]; }
  hasAttribute(name) { return this.attributes.has(name); }
  get id() { return this.getAttribute("id") || ""; }
  set id(value) { this.setAttribute("id", value); }
  get className() { return this.getAttribute("class") || ""; }
  set className(value) { this.setAttribute("class", value); }
  get name() { return this.getAttribute("name") || ""; }
  set name(value) { this.setAttribute("name", value); }
  get hidden() { return this.hasAttribute("hidden"); }
  set hidden(value) { if (value) this.setAttribute("hidden", ""); else this.removeAttribute("hidden"); }
  get disabled() { return this.hasAttribute("disabled"); }
  set disabled(value) { if (value) this.setAttribute("disabled", ""); else this.removeAttribute("disabled"); }
  get title() { return this.getAttribute("title") || ""; }
  set title(value) { this.setAttribute("title", value); }
  get href() { return this.hasAttribute("href") ? this.getAttribute("href") : undefined; }
  set href(value) { this.setAttribute("href", value); }
  get src() { return this.hasAttribute("src") ? this.getAttribute("src") : undefined; }
  set src(value) { this.setAttribute("src", value); }
  get tabIndex() { return this.hasAttribute("tabindex") ? Number(this.getAttribute("tabindex")) : 0; }
  set tabIndex(value) { this.setAttribute("tabindex", String(value)); }
  get width() { return Number(this.getAttribute("width")) || 0; }
  set width(value) { this.setAttribute("width", String(value)); }
  get height() { return Number(this.getAttribute("height")) || 0; }
  set height(value) { this.setAttribute("height", String(value)); }
  get clientWidth() { return this.width; }
  get clientHeight() { return this.height; }
  get type() {
    if (this.tagName === "SELECT") return this.hasAttribute("multiple") ? "select-multiple" : "select-one";
    if (this.tagName === "INPUT") return (this.getAttribute("type") || "text").toLowerCase();
    if (this.tagName === "BUTTON") return (this.getAttribute("type") || "submit").toLowerCase();
    if (this.tagName === "TEXTAREA") return "textarea";
    return this.getAttribute("type") || "";
  }
  set type(value) { this.setAttribute("type", value); }

  // -- tree --
  get children() { return this.childNodes.filter(n => n.nodeType === 1); }
  get firstElementChild() { return this.children[0] || null; }
  get lastElementChild() { const c = this.children; return c[c.length - 1] || null; }
  appendChild(node) {
    if (node.parentNode) node.parentNode.removeChild(node);
    node.parentNode = this;
    this.childNodes.push(node);
    return node;
  }
  add(option) { return this.appendChild(option); }
  insertBefore(node, ref) {
    if (!ref) return this.appendChild(node);
    if (node.parentNode) node.parentNode.removeChild(node);
    const index = this.childNodes.indexOf(ref);
    node.parentNode = this;
    this.childNodes.splice(index < 0 ? this.childNodes.length : index, 0, node);
    return node;
  }
  removeChild(node) {
    const index = this.childNodes.indexOf(node);
    if (index >= 0) { this.childNodes.splice(index, 1); node.parentNode = null; }
    return node;
  }
  replaceChildren(...nodes) {
    for (const node of this.childNodes.slice()) this.removeChild(node);
    for (const node of nodes) this.appendChild(typeof node === "string" ? new TextNode(node, this.ownerDocument) : node);
  }
  remove() { if (this.parentNode) this.parentNode.removeChild(this); }
  contains(node) {
    for (let n = node; n; n = n.parentNode) if (n === this) return true;
    return false;
  }
  closest(selector) {
    for (let n = this; n && n.nodeType === 1; n = n.parentNode) if (matchesSelector(n, selector)) return n;
    return null;
  }
  matches(selector) { return matchesSelector(this, selector); }
  *descendants() {
    for (const child of this.childNodes) {
      if (child.nodeType !== 1) continue;
      yield child;
      yield* child.descendants();
    }
  }
  querySelectorAll(selector) {
    const out = [];
    for (const node of this.descendants()) if (matchesSelector(node, selector)) out.push(node);
    return out;
  }
  querySelector(selector) { return this.querySelectorAll(selector)[0] || null; }
  getElementsByTagName(tag) { return this.querySelectorAll(tag); }

  // -- text --
  get textContent() {
    return this.childNodes.map(n => (n.nodeType === 3 ? n.data : n.textContent)).join("");
  }
  set textContent(value) {
    this.replaceChildren();
    if (value !== "" && value != null) this.appendChild(new TextNode(String(value), this.ownerDocument));
  }
  get innerHTML() { return serialize(this.childNodes); }
  set innerHTML(value) {
    this.replaceChildren();
    for (const node of parseFragment(String(value), this.ownerDocument)) this.appendChild(node);
  }
  get innerText() { return this.textContent; }

  // -- form controls --
  get options() {
    if (this.tagName !== "SELECT") return [];
    const out = [];
    for (const node of this.descendants()) if (node.tagName === "OPTION") out.push(node);
    return out;
  }
  get selectedIndex() { return this.options.findIndex(opt => opt._selected); }
  get value() {
    if (this.tagName === "SELECT") {
      const options = this.options;
      const selected = options.find(opt => opt._selected) || (this.hasAttribute("size") && Number(this.getAttribute("size")) > 1 ? null : options[0]);
      return selected ? selected.value : "";
    }
    if (this.tagName === "OPTION") return this.hasAttribute("value") ? this.getAttribute("value") : this.textContent;
    if (this.tagName === "INPUT" || this.tagName === "TEXTAREA") {
      return this._value !== undefined ? this._value : (this.getAttribute("value") || "");
    }
    return this._value;
  }
  set value(value) {
    const text = value == null ? "" : String(value);
    if (this.tagName === "SELECT") {
      let found = false;
      for (const opt of this.options) {
        opt._selected = !found && opt.value === text;
        if (opt._selected) found = true;
      }
      return;
    }
    if (this.tagName === "OPTION") { this.setAttribute("value", text); return; }
    this._value = text;
  }
  get text() { return this.textContent; }
  set text(value) { this.textContent = value; }
  get selected() { return !!this._selected; }
  set selected(value) { this._selected = !!value; }
  get checked() { return this._checked; }
  set checked(value) { this._checked = !!value; }
  get form() {
    if (this.hasAttribute("form")) return this.ownerDocument.getElementById(this.getAttribute("form"));
    return this.closest("form");
  }
  get elements() {
    if (this.tagName !== "FORM") return undefined;
    const doc = this.ownerDocument;
    const list = [];
    for (const node of doc.documentElement.descendants()) {
      if (!FORM_ASSOCIATED.has(node.tagName)) continue;
      const owner = node.hasAttribute("form") ? doc.getElementById(node.getAttribute("form")) : node.closest("form");
      if (owner === this) list.push(node);
    }
    list.namedItem = name => list.find(n => n.name === name || n.id === name) || null;
    return list;
  }

  // -- events --
  addEventListener(type, fn) { (this.listeners[type] ||= []).push(fn); }
  removeEventListener(type, fn) { this.listeners[type] = (this.listeners[type] || []).filter(f => f !== fn); }
  dispatchEvent(event) {
    if (!event.target) event.target = this;
    const results = [];
    for (let node = this; node; node = event.bubbles ? node.parentNode : null) {
      if (node.nodeType !== 1 && !(node instanceof Document)) continue;
      event.currentTarget = node;
      for (const fn of (node.listeners[event.type] || []).slice()) results.push(fn.call(node, event));
      if (!event.bubbles) break;
    }
    return Promise.all(results).then(() => !event.defaultPrevented);
  }
  fire(type, init) { return this.dispatchEvent(new Event(type, init)); }
  click() { return this.fire("click"); }
  focus() { this.ownerDocument.activeElement = this; }
  blur() { if (this.ownerDocument.activeElement === this) this.ownerDocument.activeElement = null; }
  scrollIntoView() {}
  pause() { this.paused = true; this.pauseCalls += 1; }
  play() { this.paused = false; }
  getContext() { return this._context; }
}

class Document {
  constructor(root) {
    this.nodeType = 9;
    this.documentElement = root;
    this.listeners = {};
    this.activeElement = null;
    this.visibilityState = "visible";
    root.ownerDocument = this;
    for (const node of root.descendants()) node.ownerDocument = this;
  }
  get body() { return this.documentElement.querySelector("body"); }
  get head() { return this.documentElement.querySelector("head"); }
  get title() { const t = this.documentElement.querySelector("title"); return t ? t.textContent : ""; }
  getElementById(id) {
    if (this.documentElement.id === id) return this.documentElement;
    for (const node of this.documentElement.descendants()) if (node.id === id) return node;
    return null;
  }
  querySelectorAll(selector) {
    const out = matchesSelector(this.documentElement, selector) ? [this.documentElement] : [];
    return out.concat(this.documentElement.querySelectorAll(selector));
  }
  querySelector(selector) { return this.querySelectorAll(selector)[0] || null; }
  createElement(tag) { return new Element(tag, this); }
  createTextNode(text) { return new TextNode(text, this); }
  addEventListener(type, fn) { (this.listeners[type] ||= []).push(fn); }
  removeEventListener(type, fn) { this.listeners[type] = (this.listeners[type] || []).filter(f => f !== fn); }
  dispatchEvent(event) {
    event.target ||= this;
    return Promise.all((this.listeners[event.type] || []).map(fn => fn.call(this, event))).then(() => !event.defaultPrevented);
  }
}

function dash(key) { return String(key).replace(/[A-Z]/g, m => "-" + m.toLowerCase()); }

function makeContext() {
  const calls = [];
  const noop = name => (...args) => { calls.push([name, ...args]); };
  return {
    calls, fillStyle: "", strokeStyle: "", lineWidth: 1,
    fillRect: noop("fillRect"), clearRect: noop("clearRect"), beginPath: noop("beginPath"), moveTo: noop("moveTo"),
    lineTo: noop("lineTo"), stroke: noop("stroke"), fill: noop("fill"), closePath: noop("closePath"), setTransform: noop("setTransform"),
  };
}

function serialize(nodes) {
  return nodes.map(node => {
    if (node.nodeType === 3) return node.data;
    const attrs = Array.from(node.attributes.entries()).map(([k, v]) => (v === "" ? ` ${k}` : ` ${k}="${v}"`)).join("");
    const tag = node.tagName.toLowerCase();
    if (VOID.has(tag)) return `<${tag}${attrs}>`;
    return `<${tag}${attrs}>${serialize(node.childNodes)}</${tag}>`;
  }).join("");
}

function parseFragment(html, doc) {
  const root = new Element("#fragment", doc);
  parseInto(html, root, doc);
  const nodes = root.childNodes.slice();
  for (const node of nodes) node.parentNode = null;
  return nodes;
}

function parseInto(html, root, doc) {
  const stack = [root];
  let index = 0;
  const attrPattern = /([^\s"'<>/=]+)(?:\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s"'=<>`]+)))?/g;
  while (index < html.length) {
    const lt = html.indexOf("<", index);
    if (lt < 0) { appendText(stack[stack.length - 1], html.slice(index), doc); break; }
    if (lt > index) appendText(stack[stack.length - 1], html.slice(index, lt), doc);
    if (html.startsWith("<!--", lt)) {
      const end = html.indexOf("-->", lt);
      index = end < 0 ? html.length : end + 3;
      continue;
    }
    if (html.startsWith("<!", lt)) {
      const end = html.indexOf(">", lt);
      index = end < 0 ? html.length : end + 1;
      continue;
    }
    const gt = html.indexOf(">", lt);
    if (gt < 0) break;
    const raw = html.slice(lt + 1, gt);
    index = gt + 1;
    if (raw[0] === "/") {
      const closing = raw.slice(1).trim().toUpperCase();
      for (let i = stack.length - 1; i > 0; i--) {
        if (stack[i].tagName === closing) { stack.length = i; break; }
      }
      continue;
    }
    const selfClosing = raw.endsWith("/");
    const body = selfClosing ? raw.slice(0, -1) : raw;
    const nameMatch = /^[a-zA-Z][\w:-]*/.exec(body);
    if (!nameMatch) continue;
    const element = new Element(nameMatch[0], doc);
    attrPattern.lastIndex = 0;
    const attrText = body.slice(nameMatch[0].length);
    let m;
    while ((m = attrPattern.exec(attrText))) {
      const value = m[2] !== undefined ? m[2] : m[3] !== undefined ? m[3] : m[4] !== undefined ? m[4] : "";
      element.setAttribute(m[1], decode(value));
    }
    if (element.tagName === "OPTION" && element.hasAttribute("selected")) element._selected = true;
    if (element.tagName === "INPUT" && element.hasAttribute("checked")) element._checked = true;
    stack[stack.length - 1].appendChild(element);
    const tag = element.tagName.toLowerCase();
    if (RAW_TEXT.has(tag) && !selfClosing) {
      const close = html.toLowerCase().indexOf(`</${tag}>`, index);
      const content = html.slice(index, close < 0 ? html.length : close);
      if (content) element.appendChild(new TextNode(content, doc));
      index = close < 0 ? html.length : close + tag.length + 3;
      continue;
    }
    if (!VOID.has(tag) && !selfClosing) stack.push(element);
  }
}

function appendText(parent, text, doc) {
  if (!text) return;
  parent.appendChild(new TextNode(decode(text), doc));
}

function createDom(html) {
  const holder = new Element("#document", null);
  parseInto(html, holder, null);
  const root = holder.children.find(n => n.tagName === "HTML") || holder;
  root.parentNode = null;
  const document = new Document(root);
  return {document, Event, CustomEvent, Element, TextNode};
}

module.exports = {createDom, Event, CustomEvent, Element, Document, TextNode, matchesSelector};
