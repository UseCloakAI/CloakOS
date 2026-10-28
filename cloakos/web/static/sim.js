/* Simulator page: live 3D view of the MuJoCo world (three.js), world controls, tester. */
import * as THREE from 'three';
import { OrbitControls } from './vendor/OrbitControls.js';
import { adaptive } from './perf.js';

const $ = (s, el = document) => el.querySelector(s);
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const post = (url, body = {}) => fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }).then(r => r.json());

THREE.Object3D.DEFAULT_UP.set(0, 0, 1);   // MuJoCo is z-up

/* ── three.js scene ─────────────────────────────────────── */
const canvas = $('#sim-canvas');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
Object.assign(canvas.style, { position: 'absolute', inset: '0', width: '100%', height: '100%' });   // never let the canvas size its own box
const tune = adaptive(renderer, { max: 1.5 });
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFShadowMap;
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(45, 1, 0.005, 50);
camera.position.set(0.55, -0.55, 0.45);
const controls = new OrbitControls(camera, canvas);
controls.target.set(0, 0, 0.03);
controls.enableDamping = true;
controls.maxPolarAngle = Math.PI * 0.49;
scene.add(new THREE.HemisphereLight(0xffffff, 0x444033, 1.6));
const sun = new THREE.DirectionalLight(0xffffff, 1.6);
sun.position.set(1.2, -0.8, 2.5);
sun.castShadow = true;
sun.shadow.mapSize.set(1024, 1024);
Object.assign(sun.shadow.camera, { left: -2, right: 2, top: 2, bottom: -2, near: 0.1, far: 6 });
scene.add(sun);

let bodies = [], version = -1, robotId = 1, follow = true, shown = false, screenTex = null;
const world = new THREE.Group();
scene.add(world);
const goalMark = new THREE.Mesh(new THREE.RingGeometry(0.035, 0.05, 40), new THREE.MeshBasicMaterial({ color: 0x22c55e, side: THREE.DoubleSide }));
goalMark.visible = false;
scene.add(goalMark);
canvas.addEventListener('dblclick', e => {        // double-click the floor: set a goal for goto policies
  const r = canvas.getBoundingClientRect();
  const ndc = new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
  const ray = new THREE.Raycaster();
  ray.setFromCamera(ndc, camera);
  const hit = new THREE.Vector3();
  if (ray.ray.intersectPlane(new THREE.Plane(new THREE.Vector3(0, 0, 1), -(lastFloorZ || 0)), hit)) post('/api/rl/goal', { x: hit.x, y: hit.y });
});
let lastFloorZ = 0;
// single click (not a drag) on the desk drops the picked thing there
let downAt = null;
canvas.addEventListener('pointerdown', e => { downAt = [e.clientX, e.clientY]; });
canvas.addEventListener('click', e => {
  if (!placing || !downAt || Math.hypot(e.clientX - downAt[0], e.clientY - downAt[1]) > 4) return;
  const r = canvas.getBoundingClientRect();
  const ndc = new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
  const ray = new THREE.Raycaster();
  ray.setFromCamera(ndc, camera);
  const hit = new THREE.Vector3();
  if (ray.ray.intersectPlane(new THREE.Plane(new THREE.Vector3(0, 0, 1), -lastFloorZ), hit)) deskEdit({ add: { name: placing, x: hit.x, y: hit.y } });
});

function bg() {
  const css = getComputedStyle(document.documentElement);
  scene.background = new THREE.Color(css.getPropertyValue('--p2').trim() || '#EAE5DB');
}

function checker(rgba, size) {
  const c = document.createElement('canvas');
  c.width = c.height = 64;
  const g = c.getContext('2d');
  const col = (k) => `rgb(${rgba.slice(0, 3).map(v => Math.round(v * 255 * k)).join(',')})`;
  g.fillStyle = col(1); g.fillRect(0, 0, 64, 64);
  g.fillStyle = col(0.86); g.fillRect(0, 0, 32, 32); g.fillRect(32, 32, 32, 32);
  const t = new THREE.CanvasTexture(c);
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  t.repeat.set(size[0] * 4, size[1] * 4);
  t.colorSpace = THREE.SRGBColorSpace;
  t.magFilter = THREE.NearestFilter;
  return t;
}

const texCache = new Map(), meshCache = new Map();
let walls = [];
/* walls between the camera and the room fade out, so you can always see in */
function fadeWalls() {
  for (const w of walls) {
    const out = w.position.clone().setZ(0).normalize();
    const toCam = camera.position.clone().sub(w.position).setZ(0);
    const target = toCam.dot(out) > 0 ? 0.08 : 1;
    w.material.opacity += (target - w.material.opacity) * 0.2;
    w.material.depthWrite = w.material.opacity > 0.9;
  }
}
function texture(g) {
  const id = g.mat && g.mat.tex;
  if (id == null || id < 0) return null;
  const key = `${version}:${id}`;
  let base = texCache.get(key);
  if (!base) {
    base = new THREE.TextureLoader().load(`/api/sim/texture/${id}?v=${version}`);
    base.colorSpace = THREE.SRGBColorSpace;
    base.wrapS = base.wrapT = THREE.RepeatWrapping;
    base.anisotropy = 8;
    texCache.set(key, base);
  }
  if (g.type === 'mesh') return base;
  // primitives: MuJoCo "texuniform" repeats per metre; approximate with the geom's size
  const t = base.clone();
  const [rx, ry] = g.mat.repeat, [a, b] = g.size;
  t.repeat.set(g.mat.uniform ? Math.max(1, 2 * (a || 3) * rx) : rx, g.mat.uniform ? Math.max(1, 2 * (b || 3) * ry) : ry);
  t.needsUpdate = true;
  return t;
}
function meshGeometry(id) {
  const key = `${version}:${id}`;
  if (!meshCache.has(key)) meshCache.set(key, fetch(`/api/sim/mesh/${id}?v=${version}`).then(r => r.arrayBuffer()).then(buf => {
    const [n, nuv] = new Uint32Array(buf, 0, 2);
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(buf, 8, n * 3), 3));
    const nrm = new Float32Array(buf, 8 + n * 12, n * 3);
    if (nrm.some(v => v !== 0)) geo.setAttribute('normal', new THREE.BufferAttribute(nrm, 3)); else geo.computeVertexNormals();
    if (nuv) geo.setAttribute('uv', new THREE.BufferAttribute(new Float32Array(buf, 8 + n * 24, nuv * 2), 2));
    return geo;
  }));
  return meshCache.get(key);
}
function material(g) {
  const color = new THREE.Color().setRGB(g.rgba[0], g.rgba[1], g.rgba[2], THREE.SRGBColorSpace);
  const m = g.mat || {};
  const map = texture(g);
  return new THREE.MeshStandardMaterial({
    color: map ? 0xffffff : color, map, transparent: g.rgba[3] < 1, opacity: g.rgba[3],
    roughness: 1 - 0.85 * Math.min(1, (m.shininess ?? 0.3) * (m.specular ?? 0.5) * 2), metalness: 0.02,
    emissive: m.emission ? color : 0x000000, emissiveIntensity: m.emission || 0,
  });
}

function mesh(g) {
  const [a, b, c] = g.size;
  let geo;
  switch (g.type) {
    case 'ellipsoid': geo = new THREE.SphereGeometry(1, 24, 16); geo.scale(a, b, c); break;
    case 'mesh': {
      const m = new THREE.Mesh(new THREE.BufferGeometry(), material(g));
      meshGeometry(g.mesh).then(geo => { m.geometry = geo; });
      m.position.set(...g.pos);
      m.quaternion.set(g.quat[1], g.quat[2], g.quat[3], g.quat[0]);
      m.castShadow = m.receiveShadow = true;
      return m;
    }
    case 'plane': geo = new THREE.PlaneGeometry(2 * (a || 3), 2 * (b || 3)); break;
    case 'box': geo = new THREE.BoxGeometry(2 * a, 2 * b, 2 * c); break;
    case 'sphere': geo = new THREE.SphereGeometry(a, 24, 16); break;
    case 'cylinder': geo = new THREE.CylinderGeometry(a, a, 2 * b, 28); geo.rotateX(Math.PI / 2); break;
    case 'capsule': geo = new THREE.CapsuleGeometry(a, 2 * b, 6, 16); geo.rotateX(Math.PI / 2); break;
    default: return null;
  }
  const mat = g.type === 'plane' && !(g.mat && g.mat.tex >= 0)
    ? new THREE.MeshStandardMaterial({ map: checker(g.rgba, [a || 3, b || 3]), roughness: 0.9 })
    : material(g);
  let m = new THREE.Mesh(geo, mat);
  if (g.name === 'screen') {   // Vector's face: show the eyes CloakOS is drawing
    screenTex = screenTex || new THREE.Texture();
    screenTex.colorSpace = THREE.SRGBColorSpace;
    const pg = new THREE.PlaneGeometry(2 * b, 2 * c);
    pg.applyMatrix4(new THREE.Matrix4().makeBasis(new THREE.Vector3(0, 1, 0), new THREE.Vector3(0, 0, 1), new THREE.Vector3(1, 0, 0)));
    m = new THREE.Mesh(pg, new THREE.MeshBasicMaterial({ map: screenTex, color: 0xffffff }));
    m.position.set(g.pos[0] + a + 0.0002, g.pos[1], g.pos[2]);
    return m;
  }
  m.position.set(...g.pos);
  m.quaternion.set(g.quat[1], g.quat[2], g.quat[3], g.quat[0]);
  if ((g.name || '').startsWith('wall')) { mat.transparent = true; walls.push(m); }
  m.castShadow = g.type !== 'plane' && !(g.name || '').startsWith('wall');
  m.receiveShadow = true;
  return m;
}

async function loadScene() {
  const sc = await (await fetch('/api/sim/scene')).json();
  world.clear();
  bodies = [];
  walls = [];
  for (let i = 0; i < (sc.nbody || 0); i++) { const grp = new THREE.Group(); bodies.push(grp); world.add(grp); }
  for (const g of sc.geoms || []) { const m = mesh(g); if (m && bodies[g.body]) bodies[g.body].add(m); }
  robotId = sc.robot || 1;
  lastFloorZ = sc.floor_z || 0;
  version = sc.version;
  $('#sim-world').value = sc.world || 'room';
  $('#sim-seed').placeholder = sc.seed ? `seed ${sc.seed}` : 'seed';
}

/* poses over a websocket, ~30 Hz */
let ws = null, last = null;
function connect() {
  if (!shown) return;
  ws = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws/sim`);
  ws.onmessage = e => {
    const msg = JSON.parse(e.data);
    last = msg;
    if (msg.v !== version) { version = msg.v; loadScene(); return; }
    const p = msg.p;
    for (let i = 0; i < bodies.length && 7 * i + 6 < p.length; i++) {
      const o = 7 * i;
      bodies[i].position.set(p[o], p[o + 1], p[o + 2]);
      bodies[i].quaternion.set(p[o + 4], p[o + 5], p[o + 6], p[o + 3]);
    }
    badges(msg);
    goalMark.visible = !!msg.goal;
    if (msg.goal) goalMark.position.set(msg.goal[0], msg.goal[1], (lastFloorZ || 0) + 0.002);
    running(msg.test);
  };
  ws.onclose = () => { ws = null; if (shown) setTimeout(connect, 1500); };
}

function badges(msg) {
  const s = msg.s || {}, out = [];
  if (msg.frozen) out.push(['Frozen', 'ice']);
  if (s.picked_up) out.push(['Picked up', 'warn']);
  if (s.cliff) out.push(['Cliff', 'warn']);
  if (s.bump) out.push(['Bump', 'warn']);
  if (s.upright != null && s.upright < 0.6) out.push(['Tipped', 'warn']);
  if (s.pitch != null && Math.abs(s.pitch) > 0.25) out.push([`Pitch ${Math.round(s.pitch * 57.3)}°`, '']);
  $('#sim-badges').innerHTML = out.map(([t, c]) => `<span class="sim-badge ${c}">${esc(t)}</span>`).join('');
}

function resize() {
  const r = canvas.parentElement.getBoundingClientRect();
  if (!r.width) return;
  renderer.setSize(Math.min(r.width, 4096), Math.min(r.height, 4096), false);
  camera.aspect = r.width / r.height;
  camera.updateProjectionMatrix();
}
new ResizeObserver(resize).observe(canvas.parentElement);

const tmp = new THREE.Vector3();
function frame() {
  requestAnimationFrame(frame);
  if (!shown) return;
  if (follow && bodies[robotId]) {
    tmp.copy(bodies[robotId].position);
    tmp.z += 0.03;
    const d = tmp.clone().sub(controls.target);
    controls.target.add(d);
    camera.position.add(d);
  }
  controls.update();
  fadeWalls();
  renderer.render(scene, camera);
  tune();
}
frame();

/* Vector's screen texture + the eyes inset, from /eyes.png */
const eyesImg = new Image();
eyesImg.onload = () => {
  if (screenTex) { screenTex.image = eyesImg; screenTex.needsUpdate = true; }
  $('#sim-eyes').src = eyesImg.src;
};
setInterval(() => { if (shown) eyesImg.src = '/eyes.png?' + Date.now(); }, 200);

/* ── controls ───────────────────────────────────────────── */
let status = null;
async function refresh() {
  try { status = await (await fetch('/api/sim')).json(); } catch { return; }
  $$('.sim-target .seg-btn').forEach(b => b.classList.toggle('active', b.dataset.target === status.target));
  $('#sim-empty').hidden = status.target === 'sim' || version > 0;
  const sel = $('#sim-world');
  if (!sel.options.length) sel.innerHTML = Object.entries(status.worlds).map(([k, v]) => `<option value="${k}">${esc(v)}</option>`).join('');
  if (status.world) sel.value = status.world;
  scenarios(status.scenarios);
  results(status.results);
  running(status.running);
  if (status.render_error) $('#sim-empty').textContent = status.render_error;
  const camMsg = $('#page-sim .sim-cam .cam-empty');            // why the camera inset is blank, if we know
  if (camMsg) camMsg.textContent = status.render_error || 'No camera yet';
}
const $$ = (s, el = document) => [...el.querySelectorAll(s)];

window.setTarget = async t => { await post('/api/sim/target', { target: t }); refresh(); };
window.resetWorld = async () => {
  const seed = $('#sim-seed').value.trim();
  if (status && status.target !== 'sim') await post('/api/sim/target', { target: 'sim' });
  await post('/api/sim/reset', { world: $('#sim-world').value, seed: seed || null });
  $('#sim-seed').value = '';
  refresh();
};
window.simPoke = (action, triple) => post('/api/sim/poke', { action, triple: !!triple });
window.simFollow = () => { follow = !follow; $('#sim-follow').classList.toggle('on', follow); };
$('#sim-follow').classList.add('on');
controls.addEventListener('start', () => { /* dragging keeps following; the target just moves with the robot */ });

function scenarios(list) {
  const el = $('#sim-scenarios');
  if (el.dataset.n == list.length) return;
  el.dataset.n = list.length;
  el.innerHTML = list.map(s => `
    <div class="sc-card">
      <div class="sc-head"><b>${esc(s.title)}</b><span class="sc-meta">${esc(s.mode)} · ${Math.round(s.duration)} s</span></div>
      <p>${esc(s.about)}</p>
      <div class="sc-foot"><span class="sc-checks">${s.checks.map(esc).join(' · ')}</span>
        <button class="chip" onclick="runTest('${s.id}')">Run</button></div>
    </div>`).join('');
}
window.runTest = async id => {
  const seed = $('#sim-seed').value.trim();
  const r = await post('/api/sim/test', { scenario: id, seed: seed || null, record: $('#sim-record').checked });
  if (!r.ok) alert('A test is already running.');
  setTimeout(refresh, 300);
};
window.stopTest = () => post('/api/sim/test/stop').then(refresh);
window.replay = (id, seed) => { $('#sim-seed').value = seed; runTest(id); };

function running(t) {
  const el = $('#sim-running');
  el.hidden = !t;
  if (!t) return;
  const pct = Math.min(100, 100 * t.t / t.duration);
  el.innerHTML = `<div class="sc-head"><b>Running: ${esc(t.title)}</b><span class="sc-meta">seed ${t.seed} · ${t.t.toFixed(0)}/${Math.round(t.duration)} s</span></div>
    <div class="sim-bar"><i style="width:${pct}%"></i></div><button class="chip" onclick="stopTest()">Stop</button>`;
}

function results(list) {
  if (!list || !list.length) return;
  $('#sim-results').innerHTML = list.map(r => `
    <div class="res ${r.passed ? 'pass' : 'fail'}">
      <div class="sc-head"><span class="res-tag">${r.passed ? 'Pass' : r.status === 'stopped' ? 'Stopped' : 'Fail'}</span>
        <b>${esc(r.title)}</b><span class="sc-meta">seed ${r.seed} · ${new Date(r.t * 1000).toLocaleTimeString()}</span>
        <button class="chip" onclick="replay('${r.scenario}', ${r.seed})" title="Run again with the same room">Rerun</button></div>
      <ul>${r.checks.map(c => `<li class="${c.ok ? 'ok' : 'no'}">${c.ok ? '✓' : '✗'} ${esc(c.name)} <span>${esc(c.detail)}</span></li>`).join('')}</ul>
    </div>`).join('');
}

/* ── panel tabs + desk builder ─────────────────────────── */
let tab = 'world', placing = null, desk = null, deskTimer = null;
window.simTab = t => {
  tab = t;
  $$('.sim-tabs .sim-tab').forEach(b => { const on = b.dataset.simtab === t; b.classList.toggle('active', on); b.setAttribute('aria-selected', on); });
  $$('.sim-pane').forEach(p => { p.hidden = p.dataset.pane !== t; });
  clearInterval(deskTimer);
  if (t !== 'desk') setPlacing(null);
  if (t === 'desk' && shown) { deskRefresh(); deskTimer = setInterval(deskRefresh, 1000); }
};
function setPlacing(name) {
  placing = name;
  $$('#desk-palette .chip').forEach(b => b.classList.toggle('on', b.dataset.obj === name));
  $('.sim-stage').classList.toggle('placing', !!name);
  $('#desk-hint').textContent = name ? `Click the desk to drop the ${name}. Click it again in the list to stop.` : 'Pick a thing, then click the desk to drop it there.';
}
window.deskPick = name => setPlacing(placing === name ? null : name);
async function deskEdit(body) {
  const r = await post('/api/sim/desk', body);
  if (!r.ok) alert(r.error || 'Couldn\'t change the desk.');
  deskRefresh(); refresh();
}
window.deskSize = () => deskEdit({ size: [+$('#desk-w').value, +$('#desk-d').value] });
window.deskRandom = () => deskEdit({ random: true });
window.deskClear = () => deskEdit({ clear: true });
window.deskRemove = id => deskEdit({ remove: id });
window.deskRun = async () => {
  const pol = $('#desk-policy').value;
  if (!pol) return alert('Train a Desk policy first (Training page).');
  const r = await post('/api/sim/desk/run', { policy: pol });
  if (!r.ok) alert(r.error);
  deskRefresh();
};
window.deskStop = () => post('/api/sim/desk/stop').then(deskRefresh);
async function deskRefresh() {
  try { desk = await (await fetch('/api/sim/desk')).json(); } catch { return; }
  const pal = $('#desk-palette');
  if (!pal.children.length) pal.innerHTML = Object.entries(desk.catalog).map(([n, c]) =>
    `<button class="chip" data-obj="${esc(n)}" onclick="deskPick('${esc(n)}')"><i style="background:rgb(${c.map(v => Math.round(v * 255)).join(',')})"></i>${esc(n)}</button>`).join('');
  const onDesk = !!desk.size;
  if (onDesk && document.activeElement?.type !== 'number') { $('#desk-w').value = desk.size[0]; $('#desk-d').value = desk.size[1]; }
  $('#desk-objects').innerHTML = !onDesk ? '<p class="tele-note">No desk loaded. Hit Build desk.</p>'
    : desk.objects.length ? desk.objects.map(o => `<div class="desk-obj"><span>${esc(o.name)}</span>
        <span class="sc-meta">${o.fell ? 'fell off' : `${Math.round(o.x * 100)}, ${Math.round(o.y * 100)} cm`}</span>
        <button title="Remove" aria-label="Remove ${esc(o.name)}" onclick="deskRemove('${esc(o.id)}')">×</button></div>`).join('')
    : '<p class="tele-note">Nothing on the desk.</p>';
  const sel = $('#desk-policy'), was = sel.value;
  sel.innerHTML = desk.policies.map(p => `<option value="${esc(p)}">${esc(p)}</option>`).join('') || '<option value="">No desk policies yet</option>';
  if (desk.policies.includes(was)) sel.value = was;
  const run = desk.run, live = run && run.running;
  $('#desk-go').hidden = !!live; $('#desk-stop').hidden = !live;
  $('#desk-score').innerHTML = run ? `<div class="desk-score">
      <div class="sc-head"><b>${esc(run.policy)}</b><span class="res-tag">${live ? 'Driving' : 'Done'}</span></div>
      <dl><dt>Time</dt><dd>${run.seconds} s</dd>
        <dt>Desk covered</dt><dd>${run.covered != null ? Math.round(run.covered * 100) + '%' : '—'} · ${run.cells_per_min}/min <span class="sc-meta">(exam wants ≥ 40)</span></dd>
        <dt>Bumps</dt><dd>${run.bumps} · ${run.bumps_per_min}/min <span class="sc-meta">(≤ 1.5)</span></dd>
        <dt>Falls</dt><dd>${run.falls}</dd>
        ${live ? `<dt>Policy</dt><dd>${esc(run.status || '')}</dd>` : ''}</dl></div>` : '';
}

/* the app shell calls these when the page is shown/hidden; tests finishing refresh the list */
window.simShown = on => { shown = on; if (!on && ws) ws.close(); if (on) { bg(); resize(); refresh(); if (!ws) connect(); } simTab(tab); };
window.simEvent = topic => { if (topic.startsWith('test.')) refresh(); };
new MutationObserver(bg).observe(document.documentElement, { attributes: true, attributeFilter: ['class', 'data-theme'] });
