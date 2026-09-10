/* Shared caption connection settings for direct, replay, and HTTPS displays. */
(function (global) {
  'use strict';
  function port(location) {
    var value = new URLSearchParams(location.search).get('port');
    var number = Number(value);
    return value && /^\d+$/.test(value) && number >= 1 && number <= 65535 ? number : 8765;
  }
  function websocketUrl(location) {
    var scheme = location.protocol === 'https:' ? 'wss://' : 'ws://';
    return scheme + (location.hostname || 'localhost') + ':' + port(location);
  }
  function mobileUrl(location) {
    var url = new URL('mobile_display.html', location.href);
    url.searchParams.set('port', String(port(location)));
    return url.href;
  }
  // Keep this guard outside connect(): reconnecting to the same session keeps
  // history, while a new lang_config session ID resets it before chunk IDs repeat.
  function partialUtteranceId(message) {
    var value = message.utterance_id == null ? message.chunk_id : message.utterance_id;
    return Number.isSafeInteger(value) && value > 0 ? value : null;
  }
  function sessionGuard(reset, onDiscard) {
    var session = null;
    var discarded = new Set();
    return function (message) {
      var incoming = typeof message.session_id === 'string' && message.session_id ? message.session_id : null;
      if (message.type === 'lang_config') {
        if (incoming && incoming !== session) { session = incoming; discarded.clear(); reset(); }
        return true;
      }
      if (message.type === 'utterance_discarded') {
        // Destructive events require a known session and an explicit utterance ID.
        var uid = message.utterance_id;
        if (session && incoming === session && Number.isSafeInteger(uid) && uid > 0 && !discarded.has(uid)) {
          discarded.add(uid);
          if (onDiscard) onDiscard(uid);
        }
        return false;
      }
      var scoped = ['translation', 'translation_start', 'translation_stream', 'speaker_update', 'music_hold'];
      if (session && scoped.indexOf(message.type) >= 0 && incoming !== session) return false;
      // A queued preview must not resurrect an utterance after its removal.
      return !(message.type === 'translation' && message.stage === 'partial' &&
        discarded.has(partialUtteranceId(message)));
    };
  }
  global.StarkDisplayConnection = {port: port, websocketUrl: websocketUrl, mobileUrl: mobileUrl, sessionGuard: sessionGuard, partialUtteranceId: partialUtteranceId};
})(window);
