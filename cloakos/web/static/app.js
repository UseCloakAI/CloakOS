/* CloakOS (Vector Edition) dashboard. One WebSocket: state at 5 Hz + events as they happen. */
'use strict';

const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const DRIVES = ['energy', 'curiosity', 'boredom', 'attachment', 'fear'];
const ORB = '<span class="bot-dot cloak-orb" aria-hidden="true"><svg viewBox="-52 -52 104 104"><circle class="cs-shape" cx="0" cy="0" r="46"></circle></svg></span>';

let ws = null, S = null, info = null, sayMode = 'talk', page = 'live';

/* ── socket ─────────────────────────────────────────────── */
function connect() {
  ws = new WebSocket((location.protocol === 'https:' ? 'wss://' : 'ws://') + location.host + '/ws');
  ws.onmessage = e => {
    const m = JSON.parse(e.data);
    if (m.type === 'state') render(m);
    else if (m.type === 'event') onEvent(m);
    else if (m.type === 'history') m.events.forEach(ev => onEvent(ev, true));
  };
  ws.onclose = () => { setConn('server'); setTimeout(connect, 1500); };
}
const sendWS = obj => { if (ws && ws.readyState === 1) ws.send(JSON.stringify(obj)); };

/* ── state render ───────────────────────────────────────── */
function buildDrives() {
  $('#drives').innerHTML = DRIVES.map(d =>
    `<div class="drive" data-d="${d}"><span class="drive-name">${d}</span><span class="drive-track"><span class="drive-fill"></span></span><span class="drive-val">—</span></div>`).join('');
  $('#lab-drives').innerHTML = DRIVES.map(d =>
    `<div class="lab-row"><span class="drive-name">${d}</span><input type="range" min="0" max="1" step="0.01" data-d="${d}"><span class="drive-val" data-v="${d}">—</span></div>`).join('');
  $$('#lab-drives input').forEach(i => i.addEventListener('input', () => sendWS({ type: 'poke', drive: i.dataset.d, value: +i.value })));
  $('#axes').innerHTML = ['left', 'right', 'head', 'lift'].map(a =>
    `<div class="axis" data-a="${a}"><span>${a}</span><span class="axis-track"><span class="axis-fill"></span></span><span class="axis-v">0</span></div>`).join('');
}

function setConn(state) {
  const on = state === 'connected', wait = state === 'connecting' || state === 'waiting';
  $$('.status-pill .live-dot').forEach(d => d.className = 'live-dot ' + (on ? 'on' : wait ? 'wait' : 'off'));
  const label = { connected: 'Connected', connecting: 'Connecting', waiting: 'Retrying', disconnecting: 'Closing', disconnected: 'Offline', server: 'No brain' }[state] || state;
  $$('.conn-label').forEach(l => l.textContent = label);
  $('#sb-email').textContent = label.toLowerCase();
  // not connected for a while: offer pairing right on the Live page
  clearTimeout(setConn.t);
  if (on) $('#pair-cta').hidden = true;
  else setConn.t = setTimeout(() => { $('#pair-cta').hidden = false; }, 8000);
}

const HINTS = {
  local_network: ['<b>macOS is blocking CloakOS from reaching Vector.</b> System Settings → Privacy &amp; Security → Local Network: turn on CloakOS, Terminal and WirePod, then restart CloakOS.', 'Open settings', 'x-apple.systempreferences:com.apple.preference.security?Privacy_LocalNetwork'],
  no_wirepod: ['<b>Can\'t find wire-pod.</b> Make sure WirePod is running (menu bar), on the same network as this Mac.', 'Set up', '/setup'],
  not_paired: ['<b>wire-pod doesn\'t know this Vector yet.</b> Pair him once and CloakOS picks him up.', 'Pair Vector', '/pair'],
};
function renderHint(hint) {
  const h = HINTS[hint] || ['<b>Vector isn\'t connected.</b> New robot, or not paired with wire-pod yet?', 'Pair Vector', '/pair'];
  $('#pair-cta-text').innerHTML = h[0]; $('#pair-cta-btn').textContent = h[1]; $('#pair-cta-btn').href = h[2];
}

function render(s) {
  S = s;
  setConn(s.connection);
  if (s.target === 'sim') { clearTimeout(setConn.t); $('#pair-cta').hidden = true; }
  $$('.freeze-btn, .freeze-chip').forEach(b => { b.classList.toggle('on', !!s.frozen); b.title = s.frozen ? 'Unfreeze' : 'Freeze (or triple-tap Vector\'s button)'; });
  document.body.classList.toggle('is-frozen', !!s.frozen);
  renderHint(s.link_hint);
  setModeUI(s.mode);
  const lab = page === 'lab' && document.activeElement && document.activeElement.type === 'range';
  DRIVES.forEach(d => {
    const v = s.drives[d] ?? 0;
    const row = $(`.drive[data-d="${d}"]`);
    $('.drive-fill', row).style.width = (v * 100).toFixed(1) + '%';
    $('.drive-val', row).textContent = v.toFixed(2);
    $(`[data-v="${d}"]`).textContent = v.toFixed(2);
    const inp = $(`#lab-drives input[data-d="${d}"]`);
    if (!lab || document.activeElement !== inp) inp.value = v;
  });
  const se = s.sensors || {};
  $('#v-behavior').textContent = s.mode === 'idle' ? 'Stock' : !s.has_control ? 'Waiting' : s.mode === 'auto' ? s.behavior.replace('_', ' ') : 'Teleop';
  $('#v-battery').textContent = se.charging ? 'Charging' : ({ 1: 'Low', 2: 'OK', 3: 'Full' }[se.battery_level] || '—') + (se.battery_volts ? ` ${se.battery_volts}v` : '');
  $('#v-prox').textContent = se.proximity_mm != null ? Math.round(se.proximity_mm) + 'mm' : (s.connection === 'connected' ? 'Clear' : '—');
  $('#v-body').textContent = se.picked_up ? 'Held' : se.cliff ? 'Cliff!' : se.touching ? 'Petted' : se.on_charger ? 'On charger' : s.connection === 'connected' ? 'Grounded' : '—';
  drawFace(se.face);
  // recording
  const rec = s.recording || {};
  $$('.rec-btn').forEach(b => { b.classList.toggle('on', !!rec.on); b.title = rec.on ? `Recording · ${rec.steps} steps` : 'Record episode'; });
  $('#sb-rec-label').textContent = rec.on ? `Stop · ${rec.steps} steps` : 'Record episode';
  // volume (leave the slider alone while it's being dragged)
  if (s.volume != null && document.activeElement !== $('#vol')) { $('#vol').value = s.volume; volLabel(s.volume); }
  // voice
  const v = s.voice || {};
  $('#voice-sub').textContent = v.error ? 'Voice off: ' + v.error : !v.ready ? 'Voice loading…' : v.speaking ? 'Speaking…' : (v.llm ? 'Voice ready · LLM on' : 'Voice ready · LLM not set');
  // teleop readout
  const a = s.action || {};
  const ax = { left: a.left_mmps / 220, right: a.right_mmps / 220, head: a.head_rate / 4, lift: a.lift_rate / 4 };
  for (const [k, val] of Object.entries(ax)) {
    const row = $(`.axis[data-a="${k}"]`); if (!row) continue;
    const f = $('.axis-fill', row), x = Math.max(-1, Math.min(1, val || 0));
    f.style.left = x < 0 ? (50 + x * 50) + '%' : '50%'; f.style.width = Math.abs(x * 50) + '%';
    $('.axis-v', row).textContent = (k === 'left' || k === 'right') ? Math.round(a[k + '_mmps'] || 0) : (a[k + '_rate'] || 0).toFixed(1);
  }
  $('#tele-mode-btn').textContent = s.mode === 'teleop' ? (s.has_control ? 'Driving' : 'Waiting for control') : 'Take control';
  $('#tele-note').hidden = s.mode === 'teleop';
  // scores (lab)
  if (page === 'lab') $('#kv-scores').innerHTML = Object.entries(s.scores || {}).sort((x, y) => y[1] - x[1])
    .map(([k, v]) => `<dt>${esc(k.replace('_', ' '))}</dt><dd>${v.toFixed(3)}${k === s.behavior ? '  ← running' : ''}</dd>`).join('') || '<dt>—</dt><dd>Auto mode only</dd>';
}

let lastVol = 80;
function volLabel(v) {
  v = +v; $('#vol-val').textContent = v ? v : 'Off';
  $('.vol').classList.toggle('muted', v === 0);
}
function setVolume(v) { v = +v; if (v) lastVol = v; volLabel(v); sendWS({ type: 'volume', value: v }); }
function toggleMute() { const v = +$('#vol').value ? 0 : lastVol || 80; $('#vol').value = v; setVolume(v); }

function drawFace(f) {
  $$('[data-cam]').forEach(cam => {
    let box = $('.cam-face', cam);
    if (!f) { if (box) box.remove(); return; }
    if (!box) { box = document.createElement('div'); box.className = 'cam-face'; cam.appendChild(box); }
    const w = f.size * 100, h = w * 16 / 9;
    Object.assign(box.style, { left: (f.cx * 100 - w / 2) + '%', top: (f.cy * 100 - h / 2) + '%', width: w + '%', height: h + '%' });
  });
}

/* ── camera ─────────────────────────────────────────────── */
// One live stream at a time: the browser only allows ~6 connections to a site, and a stream per
// page (Live, Teleop, Simulator, Map) starved the rest. Only the visible page's camera streams.
function startCamera() {
  $$('[data-cam]').forEach(cam => {
    const img = new Image();
    img.alt = 'Vector camera';
    const inset = $('.eyes-inset', cam);
    cam._img = img;
    img.onload = () => { $('.cam-empty', cam).hidden = true; $('.cam-tag', cam).hidden = false; if (inset) inset.hidden = false; if (!img.isConnected) cam.prepend(img); };
    img.onerror = () => {
      if (!img.getAttribute('src')) return;                    // we stopped it on purpose
      img.remove(); $('.cam-empty', cam).hidden = false; $('.cam-tag', cam).hidden = true; if (inset) inset.hidden = true;
      setTimeout(() => { if (cam._on) img.src = '/camera.mjpg?' + Date.now(); }, 3000);
    };
  });
  syncCamera();
}
function syncCamera() {
  $$('[data-cam]').forEach(cam => {
    const on = !!cam.closest('.page') && !cam.closest('.page').hidden && !document.hidden;
    if (on === !!cam._on || !cam._img) return;
    cam._on = on;
    if (on) cam._img.src = '/camera.mjpg?' + Date.now();
    else cam._img.removeAttribute('src');                       // closes the connection
  });
}
document.addEventListener('visibilitychange', syncCamera);

/* the eyes Cloak is drawing on Vector's screen, mirrored here (works offline too) */
let eyeN = 0;
setInterval(() => {
  if (document.hidden || page !== 'live') return;
  const src = '/eyes.png?' + (eyeN++);
  $$('[data-eyes]').forEach(im => { if (im.offsetParent !== null) im.src = src; });
}, 170);

/* ── event stream ───────────────────────────────────────── */
const EV_TEXT = {
  'sense.face_seen': 'Sees a face', 'sense.face_lost': 'Lost the face', 'sense.touched': 'Being petted',
  'sense.picked_up': 'Picked up', 'sense.put_down': 'Put down', 'sense.cliff': 'Cliff ahead', 'sense.loud_sound': 'Loud sound',
  'sense.heard_wake_word': 'Heard "Hey Vector"', 'sense.button': 'Back button',
};
const FREEZE_TEXT = on => on ? 'Frozen. Triple-tap the button (or ❄) to unfreeze' : 'Unfrozen';
let evGroup = null;
function onEvent(m, quiet) {
  const t = m.topic, d = m.data || {};
  if (window.simEvent) window.simEvent(t);
  if (t === 'voice.said') { evGroup = null; return addSay(d); }
  if (t === 'robot.connection' && d.state !== 'connected' && d.state !== 'disconnected') return;
  if (t === 'record.started' || t === 'record.stopped') loadInfo();
  let tag = t.split('.')[0], text;
  if (EV_TEXT[t]) text = EV_TEXT[t] + (d.simulated ? ' (sim)' : '');
  else if (t.startsWith('drive.')) { const [, n, edge] = t.split('.'); text = `${n} ${edge === 'high' ? 'is high' : 'is low'} · ${(+d.value).toFixed(2)}`; }
  else if (t === 'behavior.changed') text = `→ ${d.name.replace('_', ' ')}`;
  else if (t === 'mode.changed') text = `Mode: ${d.mode}`;
  else if (t === 'freeze.changed') text = FREEZE_TEXT(d.on);
  else if (t === 'command') text = (d.say ? `“${d.say}”` : 'Heard you') + (d.action === 'go_to' ? ` · going to ${d.target}` : d.action !== 'none' ? ` · ${d.action}` : '');
  else if (t === 'goto.arrived') text = `Got to the ${d.target}`;
  else if (t === 'goto.failed') text = `Couldn't reach the ${d.target}`;
  else if (t === 'test.finished') text = `Sim test ${d.title}: ${d.passed ? 'pass' : d.status === 'stopped' ? 'stopped' : 'fail'}`;
  else if (t === 'robot.connection' && d.state === 'connected' && d.name) text = `Connected to ${d.name}`;
  else if (t === 'robot.connection') text = d.state === 'connected' ? 'Connected to Vector' : 'Disconnected';
  else if (t === 'record.started') text = 'Recording started';
  else if (t === 'record.stopped') text = `Saved ${d.steps} steps`;
  else if (t === 'voice.failed') text = `Couldn't answer: ${d.reason}`;
  else return;
  if (!evGroup || !evGroup.isConnected) { evGroup = document.createElement('div'); evGroup.className = 'evs'; appendMsg(evGroup); }
  const row = document.createElement('div');
  row.className = 'ev';
  if (quiet) row.style.animation = 'none';
  row.innerHTML = `<span class="ev-tag ${esc(tag)}">${esc(tag)}</span><span class="ev-txt">${esc(text)}</span><span class="ev-time">${clock(m.t)}</span>`;
  evGroup.appendChild(row);
  while (evGroup.children.length > 14) evGroup.firstChild.remove();
  scrollDown();
}

function addSay(d) {
  if (d.source === 'you') return;   // already shown as the user's bubble
  const el = document.createElement('div');
  el.className = 'msg bot';
  const meta = [d.source === 'llm' ? 'LLM' : d.source === 'cache' ? 'Cached' : d.source, d.trigger && d.trigger !== 'manual' ? d.trigger.replace('.', ' ') : '', d.ready_ms != null ? d.ready_ms + 'ms' : '', d.spoken === false ? 'not spoken: no control' : ''].filter(Boolean).join(' · ');
  el.innerHTML = `<div class="bot-body"><div class="bot-meta">${ORB}<span class="bot-label">Vector</span></div><div class="bot-content"><div class="say-text">${esc(d.text)}</div><div class="say-meta">${esc(meta)}</div></div></div>`;
  appendMsg(el);
}

function addUser(text, mode) {
  evGroup = null;
  const el = document.createElement('div');
  el.className = 'msg user';
  el.innerHTML = `<div class="msg-wrap"><div class="bubble"><div>${esc(text)}</div></div></div>`;
  if (mode === 'talk') el.title = 'Talk to it';
  appendMsg(el);
}

function appendMsg(el) {
  $('#empty-state').style.display = 'none';
  const box = $('#messages');
  box.appendChild(el);
  while (box.children.length > 120) box.firstChild.remove();
  scrollDown();
}
const scrollDown = () => { const a = $('#chat-area'); a.scrollTop = a.scrollHeight; };
const clock = t => new Date((t || Date.now() / 1000) * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });

/* ── composer ───────────────────────────────────────────── */
const input = $('#chat-input');
input.addEventListener('input', () => {
  input.style.height = 'auto'; input.style.height = Math.min(input.scrollHeight, 160) + 'px';
  $('#send-btn').disabled = !input.value.trim();
});
input.addEventListener('keydown', e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); } });
function send() {
  const text = input.value.trim(); if (!text) return;
  addUser(text, sayMode);
  sendWS({ type: sayMode, text });
  input.value = ''; input.dispatchEvent(new Event('input'));
}
function togglePlusMenu(e) { e.stopPropagation(); $('#plus-menu').classList.toggle('open'); $('#plus-btn').classList.toggle('open'); }
function setSayMode(m) {
  sayMode = m;
  $$('#plus-menu .plus-item').forEach(b => b.classList.toggle('on', b.dataset.m === m));
  const chip = $('.say-mode'); chip.dataset.m = m; chip.textContent = m === 'say' ? 'Say it' : 'Tell it';
  input.placeholder = m === 'say' ? 'Make Vector say…' : 'Tell Vector what to do… “go to the mug”';
  $('#plus-menu').classList.remove('open'); $('#plus-btn').classList.remove('open');
  input.focus();
}

/* ── modes ──────────────────────────────────────────────── */
function toggleModeMenu(e) {
  e.stopPropagation();
  const btn = $('.mode-picker .model-btn'), dd = $('.mode-picker .model-dropdown');
  btn.classList.toggle('open', !dd.classList.contains('open')); dd.classList.toggle('open');
}
function pickMode(m) { sendWS({ type: 'mode', mode: m }); setModeUI(m); closeMenus(); }
function setModeUI(m) {
  $('.mode-label').textContent = m[0].toUpperCase() + m.slice(1);
  $('.mode-picker .mode-swatch').dataset.mode = m;
  $$('.mode-picker .model-option').forEach(o => o.classList.toggle('active', o.dataset.mode === m));
}
function closeMenus() {
  $$('.model-dropdown.open,.plus-menu.open').forEach(x => x.classList.remove('open'));
  $$('.model-btn.open,.plus-btn.open').forEach(x => x.classList.remove('open'));
}
document.addEventListener('click', closeMenus);
function toggleRecord() { sendWS({ type: 'record', on: !(S && S.recording && S.recording.on) }); }
function toggleFreeze() { sendWS({ type: 'freeze', on: !(S && S.frozen) }); }
async function deleteEpisode(ev, name) {
  ev.stopPropagation();
  if (!confirm(`Delete recording ${name}? This can't be undone.`)) return;
  const r = await fetch('/api/episodes/' + encodeURIComponent(name), { method: 'DELETE' });
  if (!r.ok) alert("Couldn't delete it (still recording?)");
  loadInfo();
}

/* ── pages + shell (same behavior as Cloak chat) ────────── */
function goPage(p) {
  page = p;
  $$('.page').forEach(s => s.hidden = s.dataset.page !== p);
  $$('[data-nav]').forEach(b => { const on = b.dataset.nav === p; b.classList.toggle('on', on && b.closest('.sb-nav')); if (on) b.setAttribute('aria-current', 'page'); else b.removeAttribute('aria-current'); });
  if (p === 'settings' || p === 'lab') loadInfo();
  if (window.simShown) window.simShown(p === 'sim');
  if (window.trainShown) window.trainShown(p === 'train');
  if (window.mapShown) window.mapShown(p === 'map');
  if (p === 'settings') { loadUpdates(); loadNous(); if (window.calRefresh) window.calRefresh(); }
  syncCamera();
  if (window.innerWidth <= 640) closeMobileSidebar();
}
function toggleSidebar() {
  const el = $('#sidebar');
  if (window.innerWidth <= 640) { const open = !el.classList.contains('collapsed'); el.classList.toggle('collapsed', open); $('#sb-overlay').classList.toggle('show', !open); }
  else el.classList.toggle('collapsed');
}
function closeMobileSidebar() { $('#sidebar').classList.add('collapsed'); $('#sb-overlay').classList.remove('show'); }
function toggleDark() {
  const dark = !document.documentElement.classList.contains('dark');
  document.documentElement.classList.toggle('dark', dark); document.body.classList.toggle('dark', dark);
  localStorage.setItem('cloak_dark', dark ? '1' : '0');
}

// Sidebar motion, ported from cloak.js: wordmark C on the spine, ink glides between nav items.
(function () {
  const sb = $('#sidebar'), c = $('.sb-c', sb), nav = $('.sb-nav', sb);
  const fitC = () => { if (c && c.offsetWidth) $('.sb-word', sb).style.setProperty('--c-w', c.offsetWidth + 'px'); };
  fitC(); if (document.fonts && document.fonts.ready) document.fonts.ready.then(fitC);
  if (window.innerWidth <= 640) sb.classList.add('collapsed');
  const ink = document.createElement('i'); ink.className = 'sb-ink'; ink.setAttribute('aria-hidden', 'true'); nav.prepend(ink);
  let at = null;
  const place = () => {
    if (!nav.offsetHeight) return;
    const on = $('.sb-btn.on', nav);
    if (!on) { ink.classList.remove('show'); at = null; return; }
    const t = on.offsetTop, b = nav.clientHeight - t - on.offsetHeight;
    if (at === null) { ink.classList.add('jump'); ink.classList.remove('up', 'down'); }
    else { ink.classList.toggle('down', t > at); ink.classList.toggle('up', t < at); }
    ink.style.setProperty('--ink-t', t + 'px'); ink.style.setProperty('--ink-b', b + 'px');
    if (at === null) { void ink.offsetWidth; ink.classList.remove('jump'); }
    ink.classList.add('show'); at = t;
  };
  const mo = new MutationObserver(place);
  $$('.sb-btn', nav).forEach(b => mo.observe(b, { attributes: true, attributeFilter: ['class'] }));
  if ('ResizeObserver' in window) new ResizeObserver(() => { fitC(); place(); }).observe(nav);
  place();
  // the footer band is as tall as the composer, like Cloak
  const ia = $('.input-area');
  const sync = () => { const h = ia.getBoundingClientRect().height; if (h) sb.style.setProperty('--input-h', h + 'px'); };
  if ('ResizeObserver' in window) new ResizeObserver(sync).observe(ia);
  sync();
})();

/* ── settings / lab ─────────────────────────────────────── */
async function loadInfo() {
  try { info = await (await fetch('/api/info')).json(); } catch { return; }
  const kv = (el, rows) => { $(el).innerHTML = rows.map(([k, v, cls]) => `<dt>${esc(k)}</dt><dd class="${cls || ''}">${esc(v ?? '—')}</dd>`).join(''); };
  kv('#kv-robot', [['wire-pod', info.wirepod], ['Connection', S ? S.connection : '—'], ['Name', info.robot.name], ['Serial', info.robot.serial], ['Address', info.robot.host], ['Control', S && S.has_control ? 'held by Cloak' : 'stock Vector']]);
  if (info.robot.name) $('#sb-name').textContent = info.robot.name;
  if (info.wirepod) $('#wp-link').href = info.wirepod;
  if (info.product) kv('#kv-about', [['Product', info.product.name], ['Version', info.product.version]]);
  $('#emote-row').innerHTML = (info.emotes || []).map(e => `<button class="chip" onclick="sendWS({type:'emote',name:'${e}'})">${esc(e)}</button>`).join('');
  kv('#kv-voice', [['TTS', info.tts], ...(info.tts_error ? [['TTS error', info.tts_error, 'err']] : [])]);
  const sk = info.skills || {}, ex = t => { const e = (sk[t] || {}).exam || {}; return !sk[t] ? 'not installed (rule-based fallback)' :
    t === 'goto' ? `trained · exam: reaches ${Math.round((e.success || 0) * 100)}%, ${e.bumps_per_run ?? '?'} bumps/run` :
    `trained · exam: ${e.cells_per_min ?? '?'} new cells/min, ${e.bumps_per_min ?? '?'} bumps/min`; };
  const L = info.llm || {};
  kv('#kv-brain', [['Wander', ex('wander')], ['Go to', ex('goto')],
    ['LLM', !L.enabled ? 'off' : L.provider === 'cloak' ? `Cloak · ${L.chat_model} (commands) · ${L.model} (lines)` : `${L.base_url} · ${L.model}`]]);
  const trigs = info.anim_triggers || [];
  $('#anim-list').innerHTML = trigs.map(t => `<option value="${esc(t)}">`).join('');
  $('#trig-list').textContent = trigs.length ? trigs.join('\n') : "Connect to load Vector's animation triggers.";
  $('#trig-list').style.whiteSpace = 'pre';
  $('#sim-row').innerHTML = (info.sim_events || []).map(e => `<button class="chip" onclick="sendWS({type:'simulate',event:'${e}'})">${esc(e.replace(/_/g, ' '))}</button>`).join('');
  $('#conv-list').innerHTML = (info.episodes || []).map(ep => {
    const m = ep.name.match(/^(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})\d{2}Z_(\w+)/);
    const label = m ? `${m[6]} · ${m[2]}/${m[3]} ${m[4]}:${m[5]}` : ep.name;
    return `<div class="conv-item" title="${esc(ep.name)}"><div class="conv-label"><svg class="conv-icon" width="11" height="11" viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="6" fill="currentColor"/></svg><span>${esc(label)}</span><span class="ep-n">${ep.steps}</span></div>` +
      `<button class="conv-del" type="button" title="Delete recording" aria-label="Delete recording" onclick="deleteEpisode(event, '${esc(ep.name)}')"><svg width="12" height="12" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" viewBox="0 0 24 24" aria-hidden="true"><path d="M4 7h16M10 11v6M14 11v6M6 7l1 13h10l1-13M9 7V4h6v3"/></svg></button></div>`;
  }).join('') || '<div class="ep-empty">No episodes yet.</div>';
}
let upd = null;
async function loadUpdates() {
  try { upd = await (await fetch('/api/setup/updates')).json(); } catch { return; }
  if (!upd.managed) { $('#kv-updates').innerHTML = '<dt>Updates</dt><dd>Dev checkout: update with git.</dd>'; return; }
  const st = upd.status || {}, wp = upd.wirepod || {};
  const when = st.checked_at ? new Date(st.checked_at * 1000).toLocaleString() : 'not yet';
  const short = x => (x || '').slice(0, 7) || '—';
  const rows = [
    ['CloakOS', `${short(upd.installed)} · ${upd.channel}`],
    ['Last check', `${when}${st.result ? ' · ' + st.result : ''}`],
    ['wire-pod', wp.managed ? `${wp.installed || '?'} · pinned ${wp.pinned}` : `yours · CloakOS won't touch it (compatible: ${wp.pinned})`],
  ];
  $('#kv-updates').innerHTML = rows.map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join('');
  $('#upd-actions').hidden = false;
  $('#upd-toggle').textContent = `Auto-update: ${upd.auto ? 'on' : 'off'}`;
}
async function toggleAutoUpdate() {
  await fetch('/api/setup/updates', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ auto: !upd.auto }) });
  loadUpdates();
}
async function checkUpdates() {
  await fetch('/api/setup/updates/check', { method: 'POST' });
  $('#kv-updates').insertAdjacentHTML('beforeend', '<dt>Now</dt><dd>Checking… If there\'s an update, CloakOS restarts by itself.</dd>');
}

function playAnim() { const k = $('#anim-key').value.trim(); if (k) sendWS({ type: 'anim', key: k }); }

/* ── teleop input: keyboard + Gamepad API, sent at 20 Hz while active ── */
const keys = new Set();
const KEYMAP = { w: 'w', a: 'a', s: 's', d: 'd', arrowup: 'up', arrowdown: 'down', r: 'r', f: 'f', shift: 'shift' };
const typing = () => { const a = document.activeElement; return a && (a.tagName === 'TEXTAREA' || a.tagName === 'INPUT'); };
addEventListener('keydown', e => {
  if (typing() || !S || S.mode !== 'teleop') return;
  const k = KEYMAP[e.key.toLowerCase()];
  if (e.key === ' ') { keys.clear(); e.preventDefault(); }
  if (k) { keys.add(k); e.preventDefault(); }
  paintKeys();
});
addEventListener('keyup', e => { const k = KEYMAP[e.key.toLowerCase()]; if (k) keys.delete(k); paintKeys(); });
addEventListener('blur', () => { keys.clear(); paintKeys(); });
document.addEventListener('visibilitychange', () => { if (document.hidden) { keys.clear(); paintKeys(); } });
const paintKeys = () => $$('.key[data-k]').forEach(el => el.classList.toggle('on', keys.has(el.dataset.k)));

let padIdx = null;
addEventListener('gamepadconnected', e => { padIdx = e.gamepad.index; $('#pad-state').textContent = 'Gamepad'; $('#pad-state').classList.add('on'); });
addEventListener('gamepaddisconnected', () => { padIdx = null; $('#pad-state').textContent = 'No gamepad'; $('#pad-state').classList.remove('on'); });
const dz = v => Math.abs(v) < 0.12 ? 0 : v;

function teleopAxes() {
  const k = n => keys.has(n) ? 1 : 0;
  const ax = { throttle: k('w') - k('s'), steer: k('d') - k('a'), head: k('up') - k('down'), lift: k('r') - k('f'), boost: k('shift') };
  const gp = padIdx != null && navigator.getGamepads ? navigator.getGamepads()[padIdx] : null;
  if (gp) {
    ax.throttle = ax.throttle || -dz(gp.axes[1] || 0);
    ax.steer = ax.steer || dz(gp.axes[0] || 0);
    ax.head = ax.head || -dz(gp.axes[3] || 0);
    const rt = gp.buttons[7] ? gp.buttons[7].value : 0, lt = gp.buttons[6] ? gp.buttons[6].value : 0;
    ax.lift = ax.lift || (rt - lt);
    ax.boost = ax.boost || (gp.buttons[0] && gp.buttons[0].pressed ? 1 : 0);
  }
  return ax;
}
let wasActive = false;
setInterval(() => {
  if (!S || S.mode !== 'teleop') { wasActive = false; keys.clear(); return; }
  const ax = teleopAxes();
  const active = Object.values(ax).some(v => v !== 0);
  if (active || wasActive) sendWS({ type: 'teleop', ...ax });   // keep-alive for the dead-man, plus one zero on release
  wasActive = active;
}, 50);

/* first run: no settings yet → the setup wizard */
fetch('/api/setup/status').then(r => r.json()).then(st => { if (!st.done) location.replace('/setup'); }).catch(() => {});

/* ── boot ───────────────────────────────────────────────── */
buildDrives();
connect();
if (document.readyState === "complete") startCamera(); else addEventListener("load", startCamera);   // an endless MJPEG would hold the load event open
loadInfo();

/* live detection boxes on every camera view (the Map page draws its own) */
(() => {
  const draw = L => {
    const [fw, fh] = L.frame_size || [640, 360];
    const html = (L.dets || []).map(d => {
      const [x0, y0, x1, y1] = d.box;
      return `<div class="det-box" style="left:${100 * x0 / fw}%;top:${100 * y0 / fh}%;width:${100 * (x1 - x0) / fw}%;height:${100 * (y1 - y0) / fh}%"><span>${d.label.replace(/[<>&]/g, '')}<i>${Math.round(d.score * 100)}%</i></span></div>`;
    }).join('');
    document.querySelectorAll('[data-dets]').forEach(el => { el.innerHTML = html; });
  };
  setInterval(async () => {
    if (document.hidden || !document.querySelector('[data-dets]')) return;
    try { draw(await (await fetch('/api/map/live')).json()); } catch { /* offline */ }
  }, 400);
})();

/* Nous model: shows the loaded version, Update Nous checks and installs the newest shipped one */
async function loadNous(info) {
  try { info = info || await (await fetch('/api/nous')).json(); } catch { return; }
  const sk = Object.entries(info.skills || {}).map(([t, v]) => `${t}${v.exam && v.exam.passed ? ' ✓' : ''}`).join(' · ') || 'no skills loaded';
  const rows = [['Nous', `v${info.installed}`], ['Skills', sk]];
  if (info.latest) rows.push(['Latest', `v${info.latest}`]);
  if (info.message) rows.push(['Now', info.message]);
  $('#kv-nous').innerHTML = rows.map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join('');
}
async function updateNous() {
  const b = $('#nous-btn');
  b.disabled = true; b.textContent = 'Checking…';
  try {
    let info = await (await fetch('/api/nous/check', { method: 'POST' })).json();
    if (info.error) info.message = info.error;
    else if (info.needs_app_update) info.message = `v${info.latest} needs a newer CloakOS. Update the app first (Updates above).`;
    else if (info.update) {
      b.textContent = 'Downloading…';
      info = await (await fetch('/api/nous/update', { method: 'POST' })).json();
      info.message = info.error || (info.updated ? `Updated to v${info.installed}. Active now.` : 'Already current.');
    } else info.message = 'You have the latest Nous.';
    loadNous(info);
  } catch { loadNous({ installed: '?', message: 'Update failed. Try again.' }); }
  b.disabled = false; b.textContent = 'Update Nous';
}
