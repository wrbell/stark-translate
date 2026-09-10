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
  function sessionGuard(reset) {
    var session = null;
    return function (message) {
      var incoming = typeof message.session_id === 'string' && message.session_id ? message.session_id : null;
      if (message.type === 'lang_config') {
        if (incoming && incoming !== session) { session = incoming; reset(); }
        return true;
      }
      var scoped = ['translation', 'translation_start', 'translation_stream', 'speaker_update', 'music_hold'];
      return !session || scoped.indexOf(message.type) < 0 || incoming === session;
    };
  }
  global.StarkDisplayConnection = {port: port, websocketUrl: websocketUrl, mobileUrl: mobileUrl, sessionGuard: sessionGuard};
})(window);
