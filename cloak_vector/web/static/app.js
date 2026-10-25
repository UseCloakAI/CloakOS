/* Cloak for Vector dashboard. One WebSocket: state at 5 Hz + events as they happen. */
'use strict';

const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const DRIVES = ['energy', 'curiosity', 'boredom', 'attachment', 'fear'];
const ORB = '<span class="bot-dot cloak-orb" aria-hidden="true"><svg viewBox="-52 -52 104 104"><circle class="cs-shape" cx="0" cy="0" r="46"></circle></svg></span>';

let ws = null, S = null, info = null, sayMode = 'say', page = 'live';

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
}

function render(s) {
  S = s;
  setConn(s.connection);
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
function startCamera() {
  $$('[data-cam]').forEach(cam => {
    const img = new Image();
    img.alt = 'Vector camera';
    img.onload = () => { $('.cam-empty', cam).hidden = true; $('.cam-tag', cam).hidden = false; if (!img.isConnected) cam.prepend(img); };
    img.onerror = () => { img.remove(); $('.cam-empty', cam).hidden = false; $('.cam-tag', cam).hidden = true; setTimeout(() => { img.src = '/camera.mjpg?' + Date.now(); }, 3000); };
    img.src = '/camera.mjpg';
  });
}

/* ── event stream ───────────────────────────────────────── */
const EV_TEXT = {
  'sense.face_seen': 'Sees a face', 'sense.face_lost': 'Lost the face', 'sense.touched': 'Being petted',
  'sense.picked_up': 'Picked up', 'sense.put_down': 'Put down', 'sense.cliff': 'Cliff ahead', 'sense.loud_sound': 'Loud sound',
  'sense.heard_wake_word': 'Heard "Hey Vector"', 'sense.button': 'Back button',
};
let evGroup = null;
function onEvent(m, quiet) {
  const t = m.topic, d = m.data || {};
  if (t === 'voice.said') { evGroup = null; return addSay(d); }
  if (t === 'robot.connection' && d.state !== 'connected' && d.state !== 'disconnected') return;
  if (t === 'record.started' || t === 'record.stopped') loadInfo();
  let tag = t.split('.')[0], text;
  if (EV_TEXT[t]) text = EV_TEXT[t] + (d.simulated ? ' (sim)' : '');
  else if (t.startsWith('drive.')) { const [, n, edge] = t.split('.'); text = `${n} ${edge === 'high' ? 'is high' : 'is low'} · ${(+d.value).toFixed(2)}`; }
  else if (t === 'behavior.changed') text = `→ ${d.name.replace('_', ' ')}`;
  else if (t === 'mode.changed') text = `Mode: ${d.mode}`;
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
  const chip = $('.say-mode'); chip.dataset.m = m; chip.textContent = m === 'say' ? 'Say it' : 'Talk to it';
  input.placeholder = m === 'say' ? 'Make Vector say…' : 'Talk to Vector…';
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

/* ── pages + shell (same behavior as Cloak chat) ────────── */
function goPage(p) {
  page = p;
  $$('.page').forEach(s => s.hidden = s.dataset.page !== p);
  $$('[data-nav]').forEach(b => { const on = b.dataset.nav === p; b.classList.toggle('on', on && b.closest('.sb-nav')); if (on) b.setAttribute('aria-current', 'page'); else b.removeAttribute('aria-current'); });
  if (p === 'settings' || p === 'lab') loadInfo();
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
  kv('#kv-robot', [['Connection', S ? S.connection : '—'], ['Name', info.robot.name], ['IP', info.robot.ip], ['Control', S && S.has_control ? 'held by Cloak' : 'stock Vector']]);
  kv('#kv-voice', [['TTS', info.tts], ...(info.tts_error ? [['TTS error', info.tts_error, 'err']] : []), ['LLM endpoint', info.llm.base_url], ['LLM model', info.llm.model]]);
  const trigs = info.anim_triggers || [];
  $('#anim-list').innerHTML = trigs.map(t => `<option value="${esc(t)}">`).join('');
  $('#trig-list').textContent = trigs.length ? trigs.join('\n') : "Connect to load Vector's animation triggers.";
  $('#trig-list').style.whiteSpace = 'pre';
  $('#sim-row').innerHTML = (info.sim_events || []).map(e => `<button class="chip" onclick="sendWS({type:'simulate',event:'${e}'})">${esc(e.replace(/_/g, ' '))}</button>`).join('');
  $('#conv-list').innerHTML = (info.episodes || []).map(ep => {
    const m = ep.name.match(/^(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})\d{2}Z_(\w+)/);
    const label = m ? `${m[6]} · ${m[2]}/${m[3]} ${m[4]}:${m[5]}` : ep.name;
    return `<div class="conv-item" title="${esc(ep.name)}"><div class="conv-label"><svg class="conv-icon" width="11" height="11" viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="6" fill="currentColor"/></svg><span>${esc(label)}</span><span class="ep-n">${ep.steps}</span></div></div>`;
  }).join('') || '<div class="ep-empty">No episodes yet.</div>';
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
  if (!S || S.mode !== 'teleop') { wasActive = false; return; }
  const ax = teleopAxes();
  const active = Object.values(ax).some(v => v !== 0);
  if (active || wasActive) sendWS({ type: 'teleop', ...ax });   // keep-alive for the dead-man, plus one zero on release
  wasActive = active;
}, 50);

/* ── boot ───────────────────────────────────────────────── */
buildDrives();
connect();
if (document.readyState === "complete") startCamera(); else addEventListener("load", startCamera);   // an endless MJPEG would hold the load event open
loadInfo();
