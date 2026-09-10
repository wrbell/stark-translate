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
  global.StarkDisplayConnection = {port: port, websocketUrl: websocketUrl, mobileUrl: mobileUrl};
})(window);
