/* Arena: a camera above the real desk (an iPhone via Continuity Camera works) tracks Vector.
   Everything visual happens here in the browser: find the gold top (+ the dot for heading) by
   colour, map it through the 4 desk corners to desk metres, post the pose ~10x a second. */
(() => {
  const $ = (s, el = document) => el.querySelector(s);
  const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const post = (url, body = {}) => fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }).then(r => r.json());
  const video = $('#ar-video'), over = $('#ar-overlay'), map = $('#ar-map');
  const PW = 320;                                  // processing width (pixels); plenty for a 10 cm robot on a desk
  const work = document.createElement('canvas'), wctx = work.getContext('2d', { willReadFrequently: true });
  let shown = false, stream = null, cfg = {}, st = null, H = null, loop = null, poll = null;
  let picking = null, cornerPts = [], last = null, path = [];

  /* ── camera ─────────────────────────────────────────────── */
  async function cameras() {
    try { await navigator.mediaDevices.getUserMedia({ video: true }).then(s => s.getTracks().forEach(t => t.stop())); } catch { }
    const devs = (await navigator.mediaDevices.enumerateDevices()).filter(d => d.kind === 'videoinput');
    const sel = $('#ar-cam');
    sel.innerHTML = '<option value="">Camera…</option>' + devs.map(d => `<option value="${esc(d.deviceId)}">${esc(d.label || 'Camera')}</option>`).join('');
    const want = devs.find(d => d.label && d.label === cfg.camera) || devs.find(d => /iphone/i.test(d.label));
    if (want) { sel.value = want.deviceId; open(want.deviceId); }
  }
  async function open(id) {
    if (stream) stream.getTracks().forEach(t => t.stop());
    stream = null;
    if (!id) return;
    try {
      // the iPhone's sensor is 4:3: asking for 16:9 crops the top and bottom of the desk away
      stream = await navigator.mediaDevices.getUserMedia({ video: { deviceId: { exact: id }, width: { ideal: 1920 }, height: { ideal: 1440 }, aspectRatio: { ideal: 4 / 3 } } });
    } catch (e) { $('#ar-empty').textContent = `Couldn't open that camera: ${e.message}`; $('#ar-empty').hidden = false; return; }
    const track = stream.getVideoTracks()[0], caps = track.getCapabilities ? track.getCapabilities() : {};
    if (caps.zoom) await track.applyConstraints({ advanced: [{ zoom: caps.zoom.min }] }).catch(() => {});   // ultra-wide if offered
    track.onended = () => lost();                  // the iPhone dropped (moved, unlocked, Wi-Fi blip)
    video.srcObject = stream;
    await video.play().catch(() => {});
    $('#ar-empty').hidden = true;
    const label = stream.getVideoTracks()[0].label;
    if (label !== cfg.camera) save({ camera: label });
  }
  $('#ar-cam').addEventListener('change', e => open(e.target.value));

  /* Continuity Camera drops now and then: keep trying the same camera until it's back */
  let retry = null;
  function lost() {
    stream = null;
    $('#ar-empty').textContent = 'The camera disconnected. Reconnecting… (keep the iPhone locked, still, and near the Mac)';
    $('#ar-empty').hidden = false;
    clearInterval(retry);
    retry = setInterval(async () => {
      if (stream) return clearInterval(retry);
      const devs = (await navigator.mediaDevices.enumerateDevices()).filter(d => d.kind === 'videoinput');
      const d = devs.find(x => x.label && x.label === cfg.camera) || devs.find(x => /iphone/i.test(x.label));
      if (d) { $('#ar-cam').value = d.deviceId; await open(d.deviceId); if (stream) clearInterval(retry); }
    }, 3000);
  }
  navigator.mediaDevices && navigator.mediaDevices.addEventListener('devicechange', () => { if (!stream && cfg.camera) lost(); });

  /* ── geometry: image (0..1) <-> desk (metres) ───────────── */
  function solve(A, b) {                           // small Gaussian elimination
    const n = b.length;
    for (let i = 0; i < n; i++) {
      let p = i; for (let r = i + 1; r < n; r++) if (Math.abs(A[r][i]) > Math.abs(A[p][i])) p = r;
      [A[i], A[p]] = [A[p], A[i]]; [b[i], b[p]] = [b[p], b[i]];
      for (let r = i + 1; r < n; r++) { const f = A[r][i] / A[i][i]; for (let c = i; c < n; c++) A[r][c] -= f * A[i][c]; b[r] -= f * b[i]; }
    }
    const x = Array(n).fill(0);
    for (let i = n - 1; i >= 0; i--) { let s = b[i]; for (let c = i + 1; c < n; c++) s -= A[i][c] * x[c]; x[i] = s / A[i][i]; }
    return x;
  }
  function homography(src, dst) {
    const A = [], b = [];
    for (let i = 0; i < 4; i++) {
      const [x, y] = src[i], [u, v] = dst[i];
      A.push([x, y, 1, 0, 0, 0, -u * x, -u * y]); b.push(u);
      A.push([0, 0, 0, x, y, 1, -v * x, -v * y]); b.push(v);
    }
    const h = solve(A, b);
    return [[h[0], h[1], h[2]], [h[3], h[4], h[5]], [h[6], h[7], 1]];
  }
  const apply = (M, [x, y]) => { const w = M[2][0] * x + M[2][1] * y + M[2][2]; return [(M[0][0] * x + M[0][1] * y + M[0][2]) / w, (M[1][0] * x + M[1][1] * y + M[1][2]) / w]; };
  function deskSize() { return cfg.size ? [cfg.size[0] / 100, cfg.size[1] / 100] : null; }
  function rebuild() {
    const s = deskSize();
    H = cfg.corners && cfg.corners.length === 4 && s ? homography(cfg.corners, [[0, 0], [s[0], 0], [s[0], s[1]], [0, s[1]]]) : null;
  }

  /* where the video really sits inside the element (object-fit: contain) */
  function box() {
    const r = over.getBoundingClientRect(), vw = video.videoWidth || 16, vh = video.videoHeight || 9;
    const k = Math.min(r.width / vw, r.height / vh), w = vw * k, h = vh * k;
    return { r, x: (r.width - w) / 2, y: (r.height - h) / 2, w, h };
  }
  const toNorm = e => { const b = box(); return [(e.clientX - b.r.left - b.x) / b.w, (e.clientY - b.r.top - b.y) / b.h]; };

  /* ── colour ─────────────────────────────────────────────── */
  function hsv(r, g, b) {
    r /= 255; g /= 255; b /= 255;
    const mx = Math.max(r, g, b), mn = Math.min(r, g, b), d = mx - mn;
    let h = 0;
    if (d) h = mx === r ? ((g - b) / d) % 6 : mx === g ? (b - r) / d + 2 : (r - g) / d + 4;
    return [(h * 60 + 360) % 360, mx ? d / mx : 0, mx];
  }
  function matcher(c) {
    if (!c) return () => false;
    // tight by default: a wooden desk is gold's neighbour (Match widens it for dim or uneven light)
    const t = +(cfg.tol || 22), ht = t * 0.6, st2 = 0.08 + t / 300, vt = 0.12 + t / 250, useHue = c[1] > 0.2;
    return (h, s, v) => {
      if (Math.abs(v - c[2]) > vt || Math.abs(s - c[1]) > st2) return false;
      if (!useHue) return true;
      const dh = Math.abs(h - c[0]); return Math.min(dh, 360 - dh) < ht;
    };
  }
  function sample(n) {                              // average colour around a clicked point (5x5 in processing pixels)
    const W = work.width, Hh = work.height, img = wctx.getImageData(0, 0, W, Hh).data;
    const cx = Math.round(n[0] * W), cy = Math.round(n[1] * Hh);
    let r = 0, g = 0, b = 0, k = 0;
    for (let y = cy - 2; y <= cy + 2; y++) for (let x = cx - 2; x <= cx + 2; x++) {
      if (x < 0 || y < 0 || x >= W || y >= Hh) continue;
      const i = (y * W + x) * 4; r += img[i]; g += img[i + 1]; b += img[i + 2]; k++;
    }
    return k ? hsv(r / k, g / k, b / k) : null;
  }

  /* ── tracking ───────────────────────────────────────────── */
  function inside(p, poly) {
    let c = false;
    for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
      const [xi, yi] = poly[i], [xj, yj] = poly[j];
      if ((yi > p[1]) !== (yj > p[1]) && p[0] < (xj - xi) * (p[1] - yi) / (yj - yi) + xi) c = !c;
    }
    return c;
  }
  function grow(poly, k) {                          // the desk outline pushed out a little (Vector hangs over edges)
    const cx = poly.reduce((a, p) => a + p[0], 0) / 4, cy = poly.reduce((a, p) => a + p[1], 0) / 4;
    return poly.map(([x, y]) => [cx + (x - cx) * k, cy + (y - cy) * k]);
  }
  function track() {
    if (!video.videoWidth) return null;
    const W = PW, Hh = Math.round(PW * video.videoHeight / video.videoWidth);
    if (work.width !== W || work.height !== Hh) { work.width = W; work.height = Hh; }
    wctx.drawImage(video, 0, 0, W, Hh);
    if (!H || !cfg.gold) return null;
    const img = wctx.getImageData(0, 0, W, Hh).data;
    const isGold = matcher(cfg.gold), isDot = matcher(cfg.dot);
    const poly = grow(cfg.corners.map(([x, y]) => [x * W, y * Hh]), 1.12);
    const xs = poly.map(p => p[0]), ys = poly.map(p => p[1]);
    const x0 = Math.max(0, Math.floor(Math.min(...xs))), x1 = Math.min(W - 1, Math.ceil(Math.max(...xs)));
    const y0 = Math.max(0, Math.floor(Math.min(...ys))), y1 = Math.min(Hh - 1, Math.ceil(Math.max(...ys)));
    const mask = new Uint8Array(W * Hh);
    for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) {
      const i = (y * W + x) * 4, c = hsv(img[i], img[i + 1], img[i + 2]);
      if (isGold(...c) && inside([x, y], poly)) mask[y * W + x] = 1;
      else if (cfg.dot && isDot(...c)) mask[y * W + x] = 2;
    }
    // connected gold blobs; keep the biggest (near the last fix if it's close in size)
    const seen = new Uint8Array(W * Hh); let best = null;
    let deskPx = 0;                                  // the desk's area in the image: Vector's top is a few % of it
    for (let i = 0, j = 3; i < 4; j = i++) deskPx += (poly[j][0] + poly[i][0]) * (poly[j][1] - poly[i][1]) / 2;
    deskPx = Math.abs(deskPx) / 1.12 / 1.12;
    for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) {
      const s = y * W + x;
      if (mask[s] !== 1 || seen[s]) continue;
      const q = [s]; seen[s] = 1; let n = 0, sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0;
      while (q.length) {
        const p = q.pop(), px = p % W, py = (p - px) / W;
        n++; sx += px; sy += py; sxx += px * px; syy += py * py; sxy += px * py;
        for (const o of [p - 1, p + 1, p - W, p + W]) if (o >= 0 && o < mask.length && mask[o] === 1 && !seen[o]) { seen[o] = 1; q.push(o); }
      }
      if (n < 12 || n > deskPx * 0.08) continue;      // specks, or something far too big to be Vector
      const cx = sx / n, cy = sy / n, b = { n, cx, cy, cxx: sxx / n - cx * cx, cyy: syy / n - cy * cy, cxy: sxy / n - cx * cy };
      const near = last && last.px ? Math.hypot(cx - last.px[0] * W, cy - last.px[1] * Hh) : 0;
      b.rank = n - near * 0.5;
      if (!best || b.rank > best.rank) best = b;
    }
    if (!best) return { ok: false };
    const c = [best.cx / W, best.cy / Hh], pos = apply(H, c);
    let yaw = null, dotPx = null;
    if (cfg.dot) {                                  // the dot near the gold top gives which way it faces
      const R = 1.3 * Math.sqrt(best.n); let n = 0, sx = 0, sy = 0;   // on or right next to the gold top, on the desk
      for (let y = Math.max(0, Math.floor(best.cy - R)); y <= Math.min(Hh - 1, best.cy + R); y++)
        for (let x = Math.max(0, Math.floor(best.cx - R)); x <= Math.min(W - 1, best.cx + R); x++)
          if (mask[y * W + x] === 2 && Math.hypot(x - best.cx, y - best.cy) <= R && inside([x, y], poly)) { n++; sx += x; sy += y; }
      if (n >= 2) { dotPx = [sx / n / W, sy / n / Hh]; const d = apply(H, dotPx); yaw = Math.atan2(d[1] - pos[1], d[0] - pos[0]); }
    }
    if (yaw === null) {                             // no dot: the blob's long axis, the way closest to last time
      const a = 0.5 * Math.atan2(2 * best.cxy, best.cxx - best.cyy), tip = apply(H, [(best.cx + Math.cos(a) * 5) / W, (best.cy + Math.sin(a) * 5) / Hh]);
      yaw = Math.atan2(tip[1] - pos[1], tip[0] - pos[0]);
      if (last && last.ok && Math.cos(yaw - last.yaw) < 0) yaw += Math.PI;
    }
    if (cfg.flip) yaw += Math.PI;
    yaw = Math.atan2(Math.sin(yaw), Math.cos(yaw));
    return { ok: true, x: pos[0], y: pos[1], yaw, px: c, dotPx, area: best.n };
  }

  /* ── drawing ────────────────────────────────────────────── */
  function draw(f) {
    const b = box(), dpr = devicePixelRatio || 1;
    if (over.width !== Math.round(b.r.width * dpr)) { over.width = Math.round(b.r.width * dpr); over.height = Math.round(b.r.height * dpr); }
    const g = over.getContext('2d'); g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, b.r.width, b.r.height);
    const P = ([x, y]) => [b.x + x * b.w, b.y + y * b.h];
    const pts = picking === 'corners' ? cornerPts : cfg.corners || [];
    const acc = getComputedStyle(document.documentElement).getPropertyValue('--acc').trim() || '#e8553d';
    g.lineWidth = 2; g.strokeStyle = acc; g.fillStyle = acc; g.font = '700 12px sans-serif';
    if (pts.length) {
      g.beginPath(); pts.forEach((p, i) => { const [x, y] = P(p); i ? g.lineTo(x, y) : g.moveTo(x, y); });
      if (pts.length === 4) g.closePath(); g.stroke();
      pts.forEach((p, i) => { const [x, y] = P(p); g.fillRect(x - 4, y - 4, 8, 8); g.fillText(['front-left', 'front-right', 'back-right', 'back-left'][i], x + 7, y - 7); });
    }
    if (f && f.ok) {
      const [x, y] = P(f.px);
      g.strokeStyle = '#22c55e'; g.lineWidth = 3; g.beginPath(); g.arc(x, y, Math.max(8, Math.sqrt(f.area) * b.w / PW), 0, Math.PI * 2); g.stroke();
      if (f.dotPx) { const [dx, dy] = P(f.dotPx); g.fillStyle = '#22c55e'; g.fillRect(dx - 3, dy - 3, 6, 6); }
    }
    // top-down map: desk, path, Vector with its heading
    const m = map.getContext('2d'), s = deskSize(); m.clearRect(0, 0, map.width, map.height);
    if (!s) return;
    const k = Math.min((map.width - 20) / s[0], (map.height - 20) / s[1]), ox = (map.width - s[0] * k) / 2, oy = (map.height - s[1] * k) / 2;
    const M = ([x, y]) => [ox + x * k, map.height - oy - y * k];
    m.strokeStyle = getComputedStyle(document.documentElement).getPropertyValue('--ink').trim() || '#fff'; m.lineWidth = 2;
    m.strokeRect(ox, oy, s[0] * k, s[1] * k);
    m.strokeStyle = acc; m.globalAlpha = 0.5; m.lineWidth = 1.5; m.beginPath();
    path.forEach((p, i) => { const [x, y] = M(p); i ? m.lineTo(x, y) : m.moveTo(x, y); }); m.stroke(); m.globalAlpha = 1;
    if (f && f.ok) {
      const [x, y] = M([f.x, f.y]), r = 0.05 * k;
      m.fillStyle = '#22c55e'; m.beginPath(); m.arc(x, y, Math.max(4, r), 0, Math.PI * 2); m.fill();
      m.strokeStyle = '#22c55e'; m.lineWidth = 3; m.beginPath(); m.moveTo(x, y); m.lineTo(x + Math.cos(f.yaw) * r * 2.2, y - Math.sin(f.yaw) * r * 2.2); m.stroke();
    }
  }

  /* ── setup clicks ───────────────────────────────────────── */
  function step(msg) { $('#ar-step').hidden = !msg; $('#ar-step').textContent = msg || ''; $('.ar-stage').classList.toggle('picking', !!msg); }
  const CORNER = ['front-left (your side, left)', 'front-right', 'back-right', 'back-left'];
  window.arCorners = () => { picking = 'corners'; cornerPts = []; step(`Click the desk's ${CORNER[0]} corner`); };
  window.arPick = what => { picking = what; step(what === 'gold' ? "Click the middle of Vector's gold top" : 'Click the dot on Vector'); };
  over.addEventListener('click', e => {
    if (!picking || !video.videoWidth) return;
    const n = toNorm(e);
    if (n[0] < 0 || n[1] < 0 || n[0] > 1 || n[1] > 1) return;
    if (picking === 'corners') {
      cornerPts.push(n);
      if (cornerPts.length < 4) return step(`Now the ${CORNER[cornerPts.length]} corner`);
      picking = null; step(null); save({ corners: cornerPts, size: [+$('#ar-w').value, +$('#ar-d').value] });
    } else {
      const c = sample(n); const what = picking; picking = null; step(null);
      if (c) save({ [what]: c });
    }
  });
  window.arSaveSize = () => save({ size: [+$('#ar-w').value, +$('#ar-d').value] });
  window.arTol = v => { cfg.tol = +v; clearTimeout(window._arTolT); window._arTolT = setTimeout(() => save({ tol: +v }), 400); };
  window.arFlip = () => save({ flip: !cfg.flip });
  async function save(patch) {
    cfg = { ...cfg, ...patch }; rebuild();
    const r = await post('/api/arena/config', patch);
    if (r.config) { cfg = r.config; rebuild(); }
  }

  /* ── runs ───────────────────────────────────────────────── */
  window.arRun = async () => {
    const r = await post('/api/arena/run', { policy: $('#ar-policy').value, seconds: +$('#ar-secs').value });
    if (!r.ok) alert(r.error); else path = [];
    refresh();
  };
  window.arStop = () => post('/api/arena/run/stop').then(refresh);

  async function refresh() {
    try { st = await (await fetch('/api/arena')).json(); } catch { return; }
    if (!Object.keys(cfg).length && st.config) {
      cfg = st.config; rebuild();
      if (cfg.size) { $('#ar-w').value = cfg.size[0]; $('#ar-d').value = cfg.size[1]; }
      if (cfg.tol) $('#ar-tol').value = cfg.tol;
    }
    const sel = $('#ar-policy'), was = sel.value;
    sel.innerHTML = `<option value="">Me (Teleop, or whatever is driving)</option>` + st.policies.map(p => `<option value="${esc(p)}">${esc(p)}</option>`).join('');
    if ([...sel.options].some(o => o.value === was)) sel.value = was;
    $('#ar-go').hidden = st.running; $('#ar-stop').hidden = !st.running;
    const run = st.run;
    $('#ar-run').innerHTML = run ? card(run, st.running) : '';
    $('#ar-recent').innerHTML = (st.recent || []).map(r => card(r, false)).join('') || '<p class="tele-note">None yet.</p>';
  }
  function card(r, live) {
    const tag = live ? 'Running' : r.status === 'done' ? 'Done' : r.status === 'fell' ? 'Fell off' : r.status;
    return `<div class="desk-score"><div class="sc-head"><b>${esc(r.driver)}</b><span class="res-tag">${esc(tag)}</span></div>
      <dl><dt>Time</dt><dd>${r.seconds} / ${Math.round(r.planned_s)} s</dd>
      <dt>Desk covered</dt><dd>${Math.round(r.covered * 100)}% · ${r.cells_per_min}/min <span class="sc-meta">(sim exam wants ≥ 40)</span></dd>
      <dt>Edge saves</dt><dd>${r.edge_saves} · ${r.edge_saves_per_min}/min</dd>
      <dt>Falls</dt><dd>${r.falls}</dd>${r.lost_s > 1 ? `<dt>Camera lost it</dt><dd>${r.lost_s} s</dd>` : ''}</dl></div>`;
  }

  function info(f) {
    const parts = [!cfg.corners ? '✗ desk corners not set' : `✓ desk ${cfg.size ? cfg.size.join(' × ') + ' cm' : ''}`,
      !cfg.gold ? '✗ gold top not picked' : '✓ gold top', !cfg.dot ? '– dot not picked (heading guessed from movement)' : '✓ dot'];
    const where = f && f.ok ? `Vector at ${(f.x * 100).toFixed(0)}, ${(f.y * 100).toFixed(0)} cm, facing ${(f.yaw * 57.3).toFixed(0)}°`
      : H && cfg.gold ? "Can't see Vector. Try Match higher, or click the gold top again." : '';
    $('#ar-track').innerHTML = `<p class="tele-note">${parts.map(esc).join('<br>')}</p>${where ? `<b>${esc(where)}</b>` : ''}`;
  }

  /* ── the 10 Hz loop ─────────────────────────────────────── */
  let tick = 0;
  function frame() {
    const f = track();
    if (f && f.ok) { last = f; if (!path.length || Math.hypot(path[path.length - 1][0] - f.x, path[path.length - 1][1] - f.y) > 0.01) { path.push([f.x, f.y]); path = path.slice(-2000); } }
    else if (f) last = { ...(last || {}), ok: false };
    draw(f);
    if (++tick % 5 === 0) info(f);
    if (f && H) post('/api/arena/pose', f.ok ? { ok: true, x: f.x, y: f.y, yaw: f.yaw } : { ok: false }).catch(() => {});
  }

  window.arenaShown = on => {
    shown = on;
    clearInterval(poll);
    if (on) {
      refresh().then(() => { if (!stream) cameras(); });
      if (!loop) loop = setInterval(frame, 100);
      poll = setInterval(refresh, 1000);
    } else if (!(st && st.running)) {              // a run keeps tracking while you look at another page
      clearInterval(loop); loop = null;
    }
  };
})();
