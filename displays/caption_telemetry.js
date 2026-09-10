/* Rendering telemetry uses only browser durations, never cross-host clocks.
 * A double animation frame is a render opportunity, not a measured photon.
 * The server's send-to-ack duration includes the return network journey.
 */
(function (global) {
  'use strict';
  function wrap(socket, handler) {
    return function (event) {
      var received = performance.now();
      var message;
      try { message = JSON.parse(event.data); } catch (_) { return; }
      var caption = message.type === 'translation' && typeof message.event_id === 'string';
      var visible = document.visibilityState === 'visible';
      var changed = false;
      var observer = caption && visible ? new MutationObserver(function (mutations) {
        changed = changed || mutations.length > 0;
      }) : null;
      if (observer) observer.observe(document.body, {childList: true, subtree: true, characterData: true});
      try { handler.call(this, event); }
      catch (error) {
        if (observer) observer.disconnect();
        throw error;
      }
      if (!observer) return;
      requestAnimationFrame(function () {
        requestAnimationFrame(function () {
          changed = changed || observer.takeRecords().length > 0;
          observer.disconnect();
          if (!changed || socket.readyState !== 1 || document.visibilityState !== 'visible') return;
          socket.send(JSON.stringify({
            type: 'caption_rendered', event_id: message.event_id,
            visible: true, receive_to_render_ms: performance.now() - received
          }));
        });
      });
      // Hidden/background tabs may never receive another animation frame.
      setTimeout(function () { observer.disconnect(); }, 5000);
    };
  }
  global.StarkCaptionTelemetry = {wrap: wrap};
})(window);
