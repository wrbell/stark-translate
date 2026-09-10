// Read-only caption preview for the operator page.
//
// Connects to the same text WebSocket the audience displays use and keeps a
// small model of what they are showing. It never sends anything back: the
// visible-browser acknowledgments belong to real audience displays only, so
// the operator preview cannot pollute timing measurements.
//
//   const model = StarkCaptions.createModel();
//   model.apply(JSON.parse(frame)) -> true when the view changed
//   model.sentences() -> [{id, source, target, speaker, partial, streaming}]

(function (global) {
  "use strict";

  const SCOPED = ["translation", "translation_start", "translation_stream", "speaker_update", "music_hold"];

  function partialUtteranceId(message) {
    const value = message.utterance_id == null ? message.chunk_id : message.utterance_id;
    return Number.isSafeInteger(value) && value > 0 ? value : null;
  }
  function replacesPartial(message, uid) {
    return message.utterance_id == null ||
      (Number.isSafeInteger(message.utterance_id) && message.utterance_id > 0 && message.utterance_id === uid);
  }

  function createModel(options) {
    const limit = (options && options.limit) || 200;
    let sentences = [];
    let session = null;
    let labels = {source: "", target: ""};
    let musicHold = false;
    const discarded = new Set();
    const finalized = new Set();
    let retiredThrough = 0;

    function published(uid) {
      return Number.isSafeInteger(uid) && uid > 0 && (uid <= retiredThrough || finalized.has(uid));
    }

    function observeFinal(message) {
      if (!session || message.session_id !== session || (message.stage || "complete") !== "complete") return;
      const uid = message.utterance_id;
      if (!Number.isSafeInteger(uid) || uid <= retiredThrough) return;
      finalized.add(uid);
      while (finalized.size > 128) {
        const oldest = Math.min(...finalized);
        finalized.delete(oldest);
        retiredThrough = Math.max(retiredThrough, oldest);
      }
    }

    function reset() {
      sentences = [];
      musicHold = false;
      discarded.clear();
      finalized.clear();
      retiredThrough = 0;
    }

    // Same guard as the audience display: a new session ID resets history so
    // repeated chunk IDs cannot merge with the previous session's captions.
    function accept(message) {
      const incoming = typeof message.session_id === "string" && message.session_id ? message.session_id : null;
      if (message.type === "lang_config") {
        if (incoming && incoming !== session) { session = incoming; reset(); }
        return true;
      }
      return !session || SCOPED.indexOf(message.type) < 0 || incoming === session;
    }

    function acceptsCaption(message) {
      if (session && message.session_id !== session) return false;
      return !(message.stage === "partial" &&
        (discarded.has(partialUtteranceId(message)) || published(partialUtteranceId(message))));
    }

    function apply(message) {
      if (!message || typeof message !== "object") return false;
      if (!accept(message)) return false;
      const type = message.type;
      if (type === "lang_config") {
        labels = {source: message.source_label || "", target: message.target_label || ""};
        return true;
      }
      if (type === "utterance_discarded") {
        const uid = message.utterance_id;
        if (!session || message.session_id !== session || !Number.isSafeInteger(uid) || uid <= 0 || discarded.has(uid)) return false;
        discarded.add(uid);
        sentences = sentences.filter(s => !(s.partial && s.utterance_id === uid));
        // Notify even when only the operator's status-feed fallback has this preview.
        return true;
      }
      if (type === "music_hold") {
        musicHold = !!message.active;
        return true;
      }
      if (type === "speaker_update") {
        const sentence = sentences.find(s => s.id === message.chunk_id);
        if (!sentence) return false;
        sentence.speaker = message.speaker || "";
        return true;
      }
      if (type === "translation_start") {
        const lastPartial = sentences.filter(s => s.partial && replacesPartial(message, s.utterance_id)).pop();
        sentences = sentences.filter(s => !(s.partial && replacesPartial(message, s.utterance_id)) && s.id !== "stream-" + message.chunk_id);
        const nextPartial = sentences.findIndex(s => s.partial);
        sentences.splice(nextPartial < 0 ? sentences.length : nextPartial, 0, {
          id: "stream-" + message.chunk_id,
          source: message.english || "",
          target: lastPartial ? lastPartial.target : "",
          speaker: message.speaker || (lastPartial && lastPartial.speaker) || "",
          partial: false,
          streaming: true,
        });
        return true;
      }
      if (type === "translation_stream") {
        const existing = sentences.find(s => s.id === "stream-" + message.chunk_id);
        if (!existing) return false;
        existing.target = message.partial_spanish_a || "";
        return true;
      }
      if (type !== "translation") return false;
      const source = message.english || "";
      const target = message.spanish_a || "";
      if ((message.stage || "complete") === "partial") {
        const uid = partialUtteranceId(message);
        if (discarded.has(uid) || published(uid)) return false;
        const existing = sentences.find(s => s.id === "p-" + message.chunk_id);
        if (existing) {
          existing.utterance_id = uid;
          existing.source = source;
          existing.target = target;
        } else {
          sentences = sentences.filter(s => !s.partial);
          sentences.push({id: "p-" + message.chunk_id, utterance_id: uid, source, target, speaker: "", partial: true, streaming: false});
        }
        return true;
      }
      // Match capture identity, preserving any newer utterance's preview.
      observeFinal(message);
      sentences = sentences.filter(s => !(s.partial && replacesPartial(message, s.utterance_id)) && s.id !== "stream-" + message.chunk_id);
      const existing = sentences.find(s => s.id === message.chunk_id);
      if (existing) {
        existing.source = source;
        existing.target = target;
        if (message.speaker) existing.speaker = message.speaker;
      } else {
        const nextPartial = sentences.findIndex(s => s.partial || s.streaming);
        sentences.splice(nextPartial < 0 ? sentences.length : nextPartial, 0, {id: message.chunk_id, source, target, speaker: message.speaker || "", partial: false, streaming: false});
      }
      if (sentences.length > limit) sentences = sentences.slice(-limit);
      return true;
    }

    return {
      apply,
      reset,
      acceptsCaption,
      observeFinal,
      sentences: () => sentences.map(s => ({...s})),
      labels: () => ({...labels}),
      musicHold: () => musicHold,
    };
  }

  // Minimal reconnecting client. `deps.WebSocket` and `deps.setTimeout` are
  // injectable for tests. Calling close() stops reconnecting for good.
  function connect(url, model, deps) {
    const WS = deps.WebSocket;
    const schedule = deps.setTimeout;
    const retryMs = deps.retryMs || 3000;
    const onChange = deps.onChange || (() => {});
    const onStatus = deps.onStatus || (() => {});
    let socket = null;
    let closed = false;
    let timer = null;

    function open() {
      if (closed) return;
      onStatus("connecting");
      let ws;
      try {
        ws = new WS(url);
      } catch (e) {
        onStatus("closed");
        timer = schedule(open, retryMs);
        return;
      }
      socket = ws;
      ws.onopen = () => { if (!closed) onStatus("open"); };
      ws.onmessage = event => {
        let message;
        try { message = JSON.parse(event.data); } catch (e) { return; }
        if (model.apply(message)) onChange(model);
      };
      ws.onerror = () => { try { ws.close(); } catch (e) { /* already closed */ } };
      ws.onclose = () => {
        if (socket === ws) socket = null;
        if (closed) return;
        onStatus("closed");
        timer = schedule(open, retryMs);
      };
    }

    open();
    return {
      close() {
        closed = true;
        if (timer && deps.clearTimeout) deps.clearTimeout(timer);
        if (socket) { try { socket.close(); } catch (e) { /* ignore */ } }
        socket = null;
      },
      get connected() { return !!socket && socket.readyState === 1; },
    };
  }

  global.StarkCaptions = {createModel, connect};
})(typeof window !== "undefined" ? window : globalThis);
