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
      var firstStream = message.type === 'translation_stream' &&
        typeof message.event_id === 'string' && message.event_id.indexOf(':stream:') !== -1;
      var caption = (message.type === 'translation' && typeof message.event_id === 'string') || firstStream;
      var visible = document.visibilityState === 'visible';
      var observer = caption && visible ? new MutationObserver(function () {}) : null;
      if (observer) observer.observe(document.body, {childList: true, subtree: true, characterData: true});
      try { handler.call(this, event); }
      catch (error) {
        if (observer) observer.disconnect();
        throw error;
      }
      if (!observer) return;
      // Caption handlers update the DOM synchronously. Stop observing here so
      // an ignored/stale event cannot borrow a later caption or timer mutation.
      var changed = observer.takeRecords().length > 0;
      observer.disconnect();
      if (!changed) return;
      requestAnimationFrame(function () {
        requestAnimationFrame(function () {
          if (socket.readyState !== 1 || document.visibilityState !== 'visible') return;
          var ack = {
            type: 'caption_rendered', event_id: message.event_id,
            visible: true, receive_to_render_ms: performance.now() - received
          };
          if (firstStream) ack.stage = 'first_stream';
          socket.send(JSON.stringify(ack));
        });
      });
    };
  }
  global.StarkCaptionTelemetry = {wrap: wrap};
})(window);
