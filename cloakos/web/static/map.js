/* Map page: the world Vector has built from its front camera, FSD-style.
   Live: what it sees right now glows, older things fade; drivable floor, obstacles,
   recognised objects as 3D boxes, its view cone, and a chase camera behind it. */
import * as THREE from 'three';
import { OrbitControls } from './vendor/OrbitControls.js';

const $ = (s, el = document) => el.querySelector(s);
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const post = (url, body = {}) => fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }).then(r => r.json());

THREE.Object3D.DEFAULT_UP.set(0, 0, 1);
const canvas = $('#map-canvas');
Object.assign(canvas.style, { position: 'absolute', inset: '0', width: '100%', height: '100%' });
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(2, devicePixelRatio));
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(55, 1, 0.01, 60);
camera.position.set(-0.5, -0.4, 0.4);
const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
scene.add(new THREE.HemisphereLight(0xffffff, 0x444444, 1.6));
const sun = new THREE.DirectionalLight(0xffffff, 1.1);
sun.position.set(1, -2, 3);
scene.add(sun);

let fsd = true, chase = true, showCone = false, shown = false, timer = null, voxelVersion = -1, fetching = false;
let voxels = null, floor = null, live = null, summary = null;
const FSD = { bg: 0x0b0f14, floor: 0x1d3b5c, fresh: new THREE.Color(0xe8f6ff), old: new THREE.Color(0x4a5560), box: 0xff8a3d };

// Vector: dark body, orange face
const robot = new THREE.Group();
const part = (geo, mat, x, z) => { const m = new THREE.Mesh(geo, mat); m.position.set(x, 0, z); robot.add(m); };
part(new THREE.BoxGeometry(0.065, 0.06, 0.04), new THREE.MeshStandardMaterial({ color: 0x15171a, roughness: 0.6 }), 0, 0.025);
part(new THREE.BoxGeometry(0.035, 0.046, 0.026), new THREE.MeshStandardMaterial({ color: 0x0c0c0e, roughness: 0.2 }), 0.008, 0.058);
part(new THREE.BoxGeometry(0.002, 0.033, 0.017), new THREE.MeshBasicMaterial({ color: 0xd44d2a }), 0.026, 0.058);
scene.add(robot);
const trail = new THREE.Line(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: 0xd44d2a }));
scene.add(trail);
const goal = new THREE.Mesh(new THREE.RingGeometry(0.035, 0.05, 40), new THREE.MeshBasicMaterial({ color: 0x22c55e, side: THREE.DoubleSide }));
goal.visible = false;
scene.add(goal);
const cone = new THREE.LineSegments(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: 0x5ec8ff, transparent: true, opacity: 0.7 }));
const coneFill = new THREE.Mesh(new THREE.BufferGeometry(), new THREE.MeshBasicMaterial({ color: 0x5ec8ff, transparent: true, opacity: 0.07, side: THREE.DoubleSide, depthWrite: false }));
scene.add(cone, coneFill);
const boxes = new THREE.Group();
scene.add(boxes);

function theme() {
  if (fsd) { scene.background = new THREE.Color(FSD.bg); return; }
  const css = getComputedStyle(document.documentElement);
  scene.background = new THREE.Color(css.getPropertyValue('--p2').trim() || '#EAE5DB');
}

/* ── voxels + drivable floor ───────────────────────────── */
let vox = null;      // last decoded voxel payload, kept so the look can switch without refetching
async function loadVoxels() {
  if (fetching) return;
  fetching = true;
  try {
    const buf = await (await fetch('/api/map/voxels')).arrayBuffer();
    const n = new Uint32Array(buf, 0, 1)[0];
    const [ox, oy, oz, vs] = new Float32Array(buf, 4, 4);
    let off = 20;
    const ijk = new Uint16Array(buf.slice(off, off + n * 6)); off += n * 6;
    const rgb = new Uint8Array(buf, off, n * 3); off += n * 3;
    const age = new Uint8Array(buf, off, n); off += n;
    const m = new Uint32Array(buf.slice(off, off + 4))[0]; off += 4;
    const fl = new Uint16Array(buf.slice(off, off + m * 4));
    vox = { n, ox, oy, oz, vs, ijk, rgb: rgb.slice(), age: age.slice(), m, fl };
    paintVoxels();
  } finally { fetching = false; }
}

function paintVoxels() {
  if (voxels) { scene.remove(voxels); voxels.geometry.dispose(); voxels.material.dispose(); voxels = null; }
  if (floor) { scene.remove(floor); floor.geometry.dispose(); floor.material.map.dispose(); floor.material.dispose(); floor = null; }
  const v = vox;
  $('#map-empty').hidden = !!(v && v.n);
  if (!v || !v.n) return;
  const mesh = new THREE.InstancedMesh(new THREE.BoxGeometry(v.vs * 0.9, v.vs * 0.9, v.vs * 0.9),
    new THREE.MeshStandardMaterial({ roughness: 0.85 }), v.n);
  const mat = new THREE.Matrix4(), c = new THREE.Color();
  let floorK = Infinity;
  for (let i = 0; i < v.n; i++) floorK = Math.min(floorK, v.ijk[3 * i + 2]);
  for (let i = 0; i < v.n; i++) {
    const k = v.ijk[3 * i + 2];
    mat.makeTranslation(v.ox + (v.ijk[3 * i] + 0.5) * v.vs, v.oy + (v.ijk[3 * i + 1] + 0.5) * v.vs, v.oz + (k + 0.5) * v.vs);
    mesh.setMatrixAt(i, mat);
    const fresh = Math.max(0, 1 - v.age[i] / 12);            // ~6 s at 2 Hz
    if (fsd) {
      c.copy(FSD.old).lerp(FSD.fresh, fresh);
      if (k <= floorK + 1) c.multiplyScalar(0.45);            // floor-level hits stay dim, like painted ground
    } else {
      const r = v.rgb[3 * i], g = v.rgb[3 * i + 1], b = v.rgb[3 * i + 2];
      if (r + g + b === 0) c.set(0xd44d2a); else c.setRGB(r / 255, g / 255, b / 255, THREE.SRGBColorSpace);
      c.multiplyScalar(0.55 + 0.45 * fresh);
    }
    mesh.setColorAt(i, c);
  }
  mesh.instanceMatrix.needsUpdate = true;
  voxels = mesh;
  scene.add(mesh);
  if (v.m) {   // drivable floor as one texture over the map's extent
    let i0 = 1e9, j0 = 1e9, i1 = 0, j1 = 0;
    for (let q = 0; q < v.m; q++) { const i = v.fl[2 * q], j = v.fl[2 * q + 1]; i0 = Math.min(i0, i); j0 = Math.min(j0, j); i1 = Math.max(i1, i); j1 = Math.max(j1, j); }
    const w = i1 - i0 + 1, h = j1 - j0 + 1;
    const data = new Uint8Array(w * h * 4);
    const fc = new THREE.Color(fsd ? FSD.floor : 0x7fb2d9);
    for (let q = 0; q < v.m; q++) {
      const o = 4 * ((v.fl[2 * q + 1] - j0) * w + (v.fl[2 * q] - i0));
      data[o] = fc.r * 255; data[o + 1] = fc.g * 255; data[o + 2] = fc.b * 255; data[o + 3] = fsd ? 200 : 90;
    }
    const tex = new THREE.DataTexture(data, w, h, THREE.RGBAFormat);
    tex.magFilter = THREE.NearestFilter; tex.needsUpdate = true;
    floor = new THREE.Mesh(new THREE.PlaneGeometry(w * v.vs, h * v.vs),
      new THREE.MeshBasicMaterial({ map: tex, transparent: true, depthWrite: false }));
    floor.position.set(v.ox + (i0 + w / 2) * v.vs, v.oy + (j0 + h / 2) * v.vs, ((live && live.floor_z) || 0) + 0.001);
    scene.add(floor);
  }
}

/* ── live layer: robot, cone, boxes, camera overlay ────── */
function label(text, color) {
  const c = document.createElement('canvas');
  c.width = 256; c.height = 64;
  const g = c.getContext('2d');
  g.fillStyle = color; g.fillRect(0, 0, 256, 64);
  g.fillStyle = '#111'; g.font = 'bold 30px sans-serif'; g.textBaseline = 'middle';
  g.fillText(text.toUpperCase().slice(0, 14), 14, 33);
  const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(c), depthTest: false }));
  s.scale.set(0.14, 0.035, 1);
  return s;
}

function drawLive(L) {
  live = L;
  if (L.robot) {
    robot.position.set(L.robot.pos[0], L.robot.pos[1], L.robot.pos[2] || 0);
    robot.rotation.set(0, 0, L.robot.yaw || 0);
  }
  goal.visible = !!L.goal;
  if (L.goal) goal.position.set(L.goal[0], L.goal[1], 0.003);
  // view cone (16:9, 50° vertical), 1.2 m deep
  const cm = L.cam;
  cone.visible = coneFill.visible = showCone && !!cm;
  if (cm) {
    const p = new THREE.Vector3(...cm.pos), f = new THREE.Vector3(...cm.fwd), l = new THREE.Vector3(...cm.left), u = new THREE.Vector3(...cm.up);
    const d = 1.2, tv = Math.tan(25 * Math.PI / 180) * d, th = tv * 16 / 9;
    const corners = [[1, 1], [-1, 1], [-1, -1], [1, -1]].map(([a, b]) => p.clone().addScaledVector(f, d).addScaledVector(l, a * th).addScaledVector(u, b * tv));
    const seg = [];
    corners.forEach((q, i) => { seg.push(p, q, q, corners[(i + 1) % 4]); });
    cone.geometry.dispose(); cone.geometry = new THREE.BufferGeometry().setFromPoints(seg);
    const tri = []; corners.forEach((q, i) => tri.push(p, q, corners[(i + 1) % 4]));
    coneFill.geometry.dispose(); coneFill.geometry = new THREE.BufferGeometry().setFromPoints(tri);
  }
  // recognised objects as 3D boxes
  boxes.clear();
  for (const o of L.objects || []) {
    const color = o.fresh ? FSD.box : 0x9aa4ad;
    let mn, mx;
    if (o.box) { [mn, mx] = o.box; } else { mn = [o.pos[0] - 0.04, o.pos[1] - 0.04, o.pos[2]]; mx = [o.pos[0] + 0.04, o.pos[1] + 0.04, o.pos[2] + 0.08]; }
    const size = mx.map((v, i) => Math.max(0.02, v - mn[i]));
    const geo = new THREE.EdgesGeometry(new THREE.BoxGeometry(...size));
    const e = new THREE.LineSegments(geo, new THREE.LineBasicMaterial({ color, transparent: true, opacity: o.fresh ? 1 : 0.6 }));
    e.position.set((mn[0] + mx[0]) / 2, (mn[1] + mx[1]) / 2, (mn[2] + mx[2]) / 2);
    boxes.add(e);
    const s = label(o.label, o.fresh ? '#FF8A3D' : '#C9CFD4');
    s.position.set(e.position.x, e.position.y, mx[2] + 0.03);
    boxes.add(s);
  }
  // detections over the live camera
  const layer = $('#det-layer');
  const [fw, fh] = L.frame_size || [640, 360];
  layer.innerHTML = (L.dets || []).map(d => {
    const [x0, y0, x1, y1] = d.box;
    return `<div class="det-box" style="left:${100 * x0 / fw}%;top:${100 * y0 / fh}%;width:${100 * (x1 - x0) / fw}%;height:${100 * (y1 - y0) / fh}%"><span>${esc(d.label)}</span></div>`;
  }).join('');
  $('#map-status').textContent = L.status || '';
  if (L.version !== voxelVersion) { voxelVersion = L.version; loadVoxels(); }
}

/* ── side panel (slower) ───────────────────────────────── */
async function refreshSummary() {
  try { summary = await (await fetch('/api/map')).json(); } catch { return; }
  const s = summary;
  const pts = (s.trail || []).map(p => new THREE.Vector3(p[0], p[1], (p[2] || 0) + 0.003));
  trail.geometry.dispose();
  trail.geometry = new THREE.BufferGeometry().setFromPoints(pts);
  $('#map-objects').innerHTML = (s.objects || []).map(o => `
    <div class="sc-card"><div class="sc-head"><b>${esc(o.label)}</b><span class="sc-meta">seen ${o.seen}× · ${ago(o.last_seen)}</span></div>
      <div class="sc-foot"><span class="sc-checks">${o.pos.map(v => v.toFixed(2)).join(', ')} m · ${esc(o.source)}</span>
      <button class="chip" onclick="mapGo('${o.id}')">Go</button></div></div>`).join('') || '<p class="tele-note">Nothing yet.</p>';
  const st = s.stats || {};
  $('#kv-map').innerHTML = [['Map', s.key || '—'], ['Occupied voxels', st.occupied ?? '—'], ['Seen free', st.free ?? '—'],
    ['Voxel', st.voxel ? `${(st.voxel * 100).toFixed(1)} cm` : '—'],
    ['Depth', (s.key || '').startsWith('sim-') ? 'rendered (sim)' : s.models && s.models.depth ? 'camera model' : 'distance sensor only (install the ml extra)']]
    .map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join('');
}
const ago = t => { const s = Math.max(0, Date.now() / 1000 - t); return s < 60 ? `${s | 0}s ago` : s < 3600 ? `${(s / 60) | 0}m ago` : `${(s / 3600) | 0}h ago`; };

window.mapGo = async id => { const r = await post('/api/map/goto', { id }); if (!r.ok) alert(r.error); };
window.mapFind = async () => {
  const q = $('#map-find').value.trim();
  if (!q) return;
  const r = await post('/api/map/goto', { query: q });
  alert(r.ok ? `Found ${r.object.label} at ${r.object.pos.map(v => v.toFixed(2)).join(', ')} m.${r.note ? ' ' + r.note : ''}` : r.error);
};
window.mapClear = async () => { if (confirm('Forget this map and everything in it?')) { await post('/api/map/clear'); voxelVersion = -1; refreshSummary(); } };
window.mapChase = () => { chase = !chase; $('#map-chase').classList.toggle('on', chase); };
window.mapLook = () => { fsd = !fsd; $('#map-look').classList.toggle('on', fsd); theme(); paintVoxels(); };
window.mapCone = () => { showCone = !showCone; $('#map-cone').classList.toggle('on', showCone); };

function resize() {
  const r = canvas.parentElement.getBoundingClientRect();
  if (!r.width) return;
  renderer.setSize(Math.min(r.width, 4096), Math.min(r.height, 4096), false);
  camera.aspect = r.width / r.height;
  camera.updateProjectionMatrix();
}
new ResizeObserver(resize).observe(canvas.parentElement);

const behind = new THREE.Vector3(), ahead = new THREE.Vector3();
(function frame() {
  requestAnimationFrame(frame);
  if (!shown) return;
  if (chase && live && live.robot) {      // FSD-style chase cam: behind and above, looking past the robot
    const yaw = live.robot.yaw || 0, p = robot.position;
    behind.set(p.x - Math.cos(yaw) * 0.42, p.y - Math.sin(yaw) * 0.42, p.z + 0.26);
    ahead.set(p.x + Math.cos(yaw) * 0.35, p.y + Math.sin(yaw) * 0.35, p.z + 0.02);
    camera.position.lerp(behind, 0.08);
    controls.target.lerp(ahead, 0.12);
  }
  controls.update();
  renderer.render(scene, camera);
})();

let liveTimer = null;
window.mapShown = on => {
  shown = on;
  clearInterval(timer); clearInterval(liveTimer);
  if (on) {
    theme(); resize(); refreshSummary();
    const tick = async () => { try { drawLive(await (await fetch('/api/map/live')).json()); } catch { /* offline */ } };
    tick();
    liveTimer = setInterval(tick, 350);
    timer = setInterval(refreshSummary, 2000);
  }
};
new MutationObserver(theme).observe(document.documentElement, { attributes: true, attributeFilter: ['class', 'data-theme'] });
