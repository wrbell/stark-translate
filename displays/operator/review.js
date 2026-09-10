(function () {
  "use strict";
  const el = id => document.getElementById(`review-${id}`);
  let sessions = [], records = [], current = null, dirty = false, busy = false;
  const activeStates = new Set(["starting", "running", "paused", "stopping"]);
  let activeSession = null, pageOffset = 0, staleDraft = null, saveConflict = false;
  let editVersion = 0, refreshVersion = 0;
  const message = text => { el("status").textContent = text; };
  const draftKey = () => current ? `stark-review-${current.session}-${current.chunk_id}` : "";
  async function request(url, method = "GET", body) {
    const response = await fetch(url, {method, headers: {"Content-Type": "application/json"},
      body: body === undefined ? undefined : JSON.stringify(body)});
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || `Request failed (${response.status})`);
    return data;
  }
  function values() {
    return {expected_revision: current.revision, source_lang: el("language").value || null,
      corrected_source_text: el("source").value, corrected_translation_text: el("target").value,
      transcript_approved: el("transcript-approved").checked,
      translation_approved: el("translation-approved").checked, excluded: el("excluded").checked,
      review_note: el("note").value};
  }
  function populate(value) {
    el("language").value = value.source_lang || "";
    el("source").value = value.corrected_source_text || "";
    el("target").value = value.corrected_translation_text || "";
    el("transcript-approved").checked = !!value.transcript_approved;
    el("translation-approved").checked = !!value.translation_approved;
    el("excluded").checked = !!value.excluded;
    el("note").value = value.review_note || "";
    labels();
  }
  function labels() {
    const source = el("language").value;
    el("source-label").textContent = source === "en" ? "English transcript" : source === "es" ? "Spanish transcript" : "Source transcript";
    el("target-label").textContent = source === "en" ? "Spanish translation" : source === "es" ? "English translation" : "Translation";
  }
  function render(record) {
    current = record; dirty = false; saveConflict = false; staleDraft = null;
    el("restore").hidden = true;
    el("editor").hidden = !record;
    if (!record) return;
    populate(record);
    const context = record.context || {};
    el("context").textContent = `Before: ${context.previous || "—"}\nAfter: ${context.next || "—"}`;
    el("original").textContent = JSON.stringify({transcript: record.source_text,
      gemma: record.translation_text, marian: record.spanish_marian,
      priority: record.review_priority, confidence: record.stt_confidence, qe: record.qe_a,
      homophones: record.homophone_flags, near_misses: record.near_miss_flags}, null, 2);
    el("audio").pause();
    el("audio").removeAttribute("src");
    el("audio").hidden = !record.audio_available;
    if (record.audio_available) el("audio").src = `/api/review/${encodeURIComponent(record.session)}/segments/${record.chunk_id}/audio`;
    el("audio-status").textContent = record.audio_available ? "" : "Audio unavailable. Text review is still available; STT audio export will skip this segment.";
    try {
      const saved = JSON.parse(localStorage.getItem(draftKey()) || "null");
      if (saved) {
        if (saved.expected_revision === record.revision) { populate(saved); dirty = true; message("Restored unsaved draft."); }
        else {
          staleDraft = saved; el("restore").hidden = false;
          message("A newer server revision exists. Current saved text is shown; you can restore your local draft for comparison.");
        }
      }
    } catch (e) { /* Local draft storage is optional. */ }
  }
  function changed() {
    if (!current) return;
    dirty = true; editVersion += 1; labels();
    try { localStorage.setItem(draftKey(), JSON.stringify(values())); } catch (e) { /* optional */ }
    message("Unsaved draft — saved locally in this browser.");
  }
  for (const id of ["source", "target", "language", "note", "transcript-approved", "translation-approved", "excluded"]) {
    el(id).addEventListener("input", () => {
      if (id === "source" || id === "language") el("transcript-approved").checked = false;
      if (["source", "target", "language"].includes(id)) el("translation-approved").checked = false;
      changed();
    });
  }
  async function save() {
    if (!current || !dirty) return true;
    if (busy) return false;
    busy = true;
    const sentVersion = editVersion;
    try {
      const result = await request(`/api/review/${encodeURIComponent(current.session)}/segments/${current.chunk_id}`, "PUT", values());
      const key = draftKey();
      current = {...current, ...result};
      if (sentVersion !== editVersion) {
        // An edit made while saving belongs to the next server revision.
        try { localStorage.setItem(key, JSON.stringify(values())); } catch (e) { /* optional */ }
        message(`Saved revision ${result.revision}. Newer edits are still an unsaved draft.`);
        return false;
      }
      dirty = false;
      try { localStorage.removeItem(key); } catch (e) { /* optional */ }
      message(`Saved revision ${result.revision}.`);
      return true;
    } catch (e) { saveConflict = true; message(`Save failed: ${e.message} Your draft remains in this browser. Refresh to load the current revision.`); return false; }
    finally { busy = false; }
  }
  async function refreshRecords(preserve = true) {
    const session = el("session").value;
    if (!session) { records = []; render(null); return; }
    if (busy) return;
    const requestedVersion = ++refreshVersion;
    const query = new URLSearchParams({pending_only: el("pending").checked, flagged_only: el("flagged").checked, limit: 200, offset: pageOffset});
    try {
      const data = await request(`/api/review/${encodeURIComponent(session)}/segments?${query}`);
      if (requestedVersion !== refreshVersion || el("session").value !== session || dirty) return;
      records = data.segments;
      el("items").replaceChildren(...records.map(record => new Option(
        `#${record.chunk_id} · priority ${record.review_priority || 0} · ${record.source_text.slice(0, 55)}`, String(record.chunk_id))));
      const selected = preserve && current && current.session === session
        ? records.find(record => record.chunk_id === current.chunk_id) : null;
      const next = selected || records[0] || null;
      if (next) el("items").value = String(next.chunk_id);
      if (!preserve || !dirty) render(next);
      if (!dirty && !staleDraft) message(`${data.total} matching finalized segments · showing ${records.length ? pageOffset + 1 : 0}–${pageOffset + records.length}.`);
      el("previous-page").disabled = pageOffset === 0;
      el("next-page").disabled = pageOffset + records.length >= data.total;
      el("export").disabled = session === activeSession;
    } catch (e) { message(e.message); }
  }
  async function refreshSessions() {
    try {
      const data = await request("/api/review/sessions"); sessions = data.sessions;
      const selected = el("session").value;
      el("session").replaceChildren(...sessions.map(s => new Option(`${s.session}${s.active ? " · LIVE" : ""} · ${s.pending} pending`, s.session)));
      if (sessions.some(s => s.session === selected)) el("session").value = selected;
      else if (sessions.length) el("session").value = (sessions.find(s => !s.active) || sessions[0]).session;
      activeSession = (sessions.find(s => s.active) || {}).session || null;
      await refreshRecords();
    } catch (e) { message(e.message); }
  }
  el("items").addEventListener("change", async () => {
    const selected = Number(el("items").value);
    if (await save()) render(records.find(r => r.chunk_id === selected));
    else if (current) el("items").value = String(current.chunk_id);
  });
  el("session").addEventListener("change", async () => {
    if (await save()) { current = null; pageOffset = 0; await refreshRecords(false); }
    else if (current) el("session").value = current.session;
  });
  for (const id of ["pending", "flagged"]) el(id).addEventListener("change", async () => { if (await save()) { pageOffset = 0; await refreshRecords(); } });
  el("refresh").addEventListener("click", async () => {
    if (saveConflict) await refreshRecords(false);
    else if (await save()) await refreshSessions();
  });
  el("restore").addEventListener("click", () => {
    if (staleDraft) { populate(staleDraft); el("restore").hidden = true; changed(); }
  });
  for (const [id, delta] of [["previous-page", -200], ["next-page", 200]]) {
    el(id).addEventListener("click", async () => {
      if (await save()) { pageOffset = Math.max(0, pageOffset + delta); await refreshRecords(false); }
    });
  }
  el("save").addEventListener("click", save);
  el("next").addEventListener("click", async () => {
    const index = current ? records.findIndex(r => r.chunk_id === current.chunk_id) : -1;
    if (await save()) {
      const next = records[index + 1];
      if (next) { el("items").value = String(next.chunk_id); render(next); }
      else { current = null; pageOffset = 0; await refreshRecords(false); }
    }
  });
  el("export").addEventListener("click", async () => {
    if (!(await save())) return;
    el("export").disabled = true;
    try {
      const result = await request(`/api/review/${encodeURIComponent(el("session").value)}/export`, "POST", {split: el("split").value});
      el("download").href = result.download_url; el("download").hidden = false;
      message(`Export ready: ${result.stt_samples.en} English and ${result.stt_samples.es} Spanish audio clips; ${result.translation_pairs} translation pairs.`);
    } catch (e) { message(`Export failed: ${e.message}`); }
    finally { el("export").disabled = el("session").value === activeSession; }
  });
  window.addEventListener("operator-session", event => {
    activeSession = activeStates.has(event.detail.state) ? event.detail.session_id : null;
    el("export").disabled = el("session").value === activeSession;
  });
  window.addEventListener("beforeunload", event => { if (dirty) { event.preventDefault(); event.returnValue = ""; } });
  refreshSessions();
  setInterval(() => { if (!dirty && !busy) refreshSessions(); }, 5000);
})();
