/* Map page: the 3D voxel map Vector builds, its path, and the objects it remembers. */
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
const camera = new THREE.PerspectiveCamera(50, 1, 0.01, 60);
camera.position.set(1.4, -1.6, 1.6);
const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
scene.add(new THREE.HemisphereLight(0xffffff, 0x555044, 1.8));
const sun = new THREE.DirectionalLight(0xffffff, 1.2);
sun.position.set(1, -2, 3);
scene.add(sun);
const grid = new THREE.GridHelper(10, 40, 0x888888, 0x555555);
grid.rotation.x = Math.PI / 2;
grid.material.transparent = true; grid.material.opacity = 0.25;
scene.add(grid);

let voxels = null, voxelVersion = -1, shown = false, timer = null, labels = [];
const robot = new THREE.Group();
const part = (geo, mat, x, z) => { const m = new THREE.Mesh(geo, mat); m.position.set(x, 0, z); robot.add(m); };
part(new THREE.BoxGeometry(0.07, 0.06, 0.06), new THREE.MeshStandardMaterial({ color: 0x111111 }), 0, 0.03);
part(new THREE.BoxGeometry(0.01, 0.04, 0.02), new THREE.MeshBasicMaterial({ color: 0xd44d2a }), 0.036, 0.045);
scene.add(robot);
const trail = new THREE.Line(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: 0xd44d2a }));
scene.add(trail);
const goal = new THREE.Mesh(new THREE.RingGeometry(0.035, 0.05, 40), new THREE.MeshBasicMaterial({ color: 0x22c55e, side: THREE.DoubleSide }));
goal.visible = false;
scene.add(goal);

function bg() {
  const css = getComputedStyle(document.documentElement);
  scene.background = new THREE.Color(css.getPropertyValue('--p2').trim() || '#EAE5DB');
}

async function loadVoxels() {
  const buf = await (await fetch('/api/map/voxels')).arrayBuffer();
  const n = new Uint32Array(buf, 0, 1)[0];
  if (voxels) { scene.remove(voxels); voxels.geometry.dispose(); voxels.material.dispose(); voxels = null; }
  $('#map-empty').hidden = n > 0;
  if (!n) return;
  const [ox, oy, oz, vs] = new Float32Array(buf, 4, 4);
  const ijk = new Uint16Array(buf, 20, n * 3);
  const rgb = new Uint8Array(buf, 20 + n * 6, n * 3);
  const mesh = new THREE.InstancedMesh(new THREE.BoxGeometry(vs * 0.92, vs * 0.92, vs * 0.92), new THREE.MeshStandardMaterial({ roughness: 0.9 }), n);
  const m = new THREE.Matrix4(), c = new THREE.Color();
  for (let i = 0; i < n; i++) {
    m.makeTranslation(ox + (ijk[3 * i] + 0.5) * vs, oy + (ijk[3 * i + 1] + 0.5) * vs, oz + (ijk[3 * i + 2] + 0.5) * vs);
    mesh.setMatrixAt(i, m);
    const r = rgb[3 * i], g = rgb[3 * i + 1], b = rgb[3 * i + 2];
    // voxels with no colour (distance-sensor hits) show in the accent colour
    if (r + g + b === 0) c.set(0xd44d2a); else c.setRGB(r / 255, g / 255, b / 255, THREE.SRGBColorSpace);
    mesh.setColorAt(i, c);
  }
  mesh.instanceMatrix.needsUpdate = true;
  voxels = mesh;
  scene.add(mesh);
}

function label(text) {
  const c = document.createElement('canvas');
  c.width = 256; c.height = 64;
  const g = c.getContext('2d');
  g.fillStyle = '#0A0A0A'; g.fillRect(0, 0, 256, 64);
  g.fillStyle = '#F2EEE5'; g.font = 'bold 30px sans-serif'; g.textBaseline = 'middle';
  g.fillText(text.toUpperCase().slice(0, 14), 14, 33);
  const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(c), depthTest: false }));
  s.scale.set(0.16, 0.04, 1);
  return s;
}

let summary = null;
async function refresh() {
  try { summary = await (await fetch('/api/map')).json(); } catch { return; }
  const s = summary;
  $('#map-status').textContent = s.status || '';
  if (s.stats && s.stats.version !== voxelVersion) { voxelVersion = s.stats.version; loadVoxels(); }
  if (s.robot) {
    robot.position.set(s.robot.pos[0], s.robot.pos[1], s.robot.pos[2] || 0);
    robot.rotation.set(0, 0, s.robot.yaw || 0);
  }
  const pts = (s.trail || []).map(p => new THREE.Vector3(p[0], p[1], (p[2] || 0) + 0.003));
  trail.geometry.dispose();
  trail.geometry = new THREE.BufferGeometry().setFromPoints(pts);
  goal.visible = !!s.goal;
  if (s.goal) goal.position.set(s.goal[0], s.goal[1], 0.003);
  labels.forEach(l => scene.remove(l));
  labels = (s.objects || []).map(o => {
    const l = label(o.label);
    l.position.set(o.pos[0], o.pos[1], o.pos[2] + 0.09);
    scene.add(l);
    return l;
  });
  $('#map-objects').innerHTML = (s.objects || []).map(o => `
    <div class="sc-card"><div class="sc-head"><b>${esc(o.label)}</b><span class="sc-meta">seen ${o.seen}× · ${ago(o.last_seen)}</span></div>
      <div class="sc-foot"><span class="sc-checks">${o.pos.map(v => v.toFixed(2)).join(', ')} m · ${esc(o.source)}</span>
      <button class="chip" onclick="mapGo('${o.id}')">Go</button></div></div>`).join('') || '<p class="tele-note">Nothing yet.</p>';
  const st = s.stats || {};
  $('#kv-map').innerHTML = [['Map', s.key || '—'], ['Occupied voxels', st.occupied ?? '—'], ['Seen free', st.free ?? '—'],
    ['Voxel', st.voxel ? `${(st.voxel * 100).toFixed(1)} cm` : '—'], ['Depth', (s.key || '').startsWith('sim-') ? 'rendered (sim)' : s.models && s.models.depth ? 'camera model' : 'distance sensor only (install the ml extra)']]
    .map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join('');
}
const ago = t => { const s = Math.max(0, Date.now() / 1000 - t); return s < 60 ? `${s | 0}s ago` : s < 3600 ? `${(s / 60) | 0}m ago` : `${(s / 3600) | 0}h ago`; };

window.mapGo = async id => { const r = await post('/api/map/goto', { id }); if (!r.ok) alert(r.error); else if (r.note || r.blocked) alert(r.note || r.blocked); refresh(); };
window.mapFind = async () => {
  const q = $('#map-find').value.trim();
  if (!q) return;
  const r = await post('/api/map/goto', { query: q });
  alert(r.ok ? `Found ${r.object.label} at ${r.object.pos.map(v => v.toFixed(2)).join(', ')} m.${r.note ? ' ' + r.note : r.blocked ? ' ' + r.blocked : ' On the way.'}` : r.error);
  refresh();
};
window.mapClear = async () => { if (confirm('Forget this map and everything in it?')) { await post('/api/map/clear'); voxelVersion = -1; refresh(); } };

function resize() {
  const r = canvas.parentElement.getBoundingClientRect();
  if (!r.width) return;
  renderer.setSize(Math.min(r.width, 4096), Math.min(r.height, 4096), false);
  camera.aspect = r.width / r.height;
  camera.updateProjectionMatrix();
}
new ResizeObserver(resize).observe(canvas.parentElement);
(function frame() { requestAnimationFrame(frame); if (!shown) return; controls.update(); renderer.render(scene, camera); })();

window.mapShown = on => {
  shown = on;
  clearInterval(timer);
  if (on) { bg(); resize(); refresh(); timer = setInterval(refresh, 1500); }
};
new MutationObserver(bg).observe(document.documentElement, { attributes: true, attributeFilter: ['class', 'data-theme'] });
