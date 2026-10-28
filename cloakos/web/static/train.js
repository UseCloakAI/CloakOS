/* Training page: start/stop RL runs, chart progress, pick the policy that drives. */
(() => {
  const $ = (s, el = document) => el.querySelector(s);
  const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const post = (url, body = {}) => fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }).then(r => r.json());
  let st = null, chartName = null, timer = null;

  async function refresh() {
    try { st = await (await fetch('/api/rl')).json(); } catch { return; }
    const task = $('#rl-task'), sens = $('#rl-sensors');
    if (!task.options.length) {
      task.innerHTML = Object.entries(st.tasks).map(([k, v]) => `<option value="${k}">${esc(v)}</option>`).join('');
      sens.innerHTML = Object.entries(st.sensors).map(([k, v]) => `<option value="${k}">${esc(v)}</option>`).join('');
      sens.value = 'map';
    }
    formFit();
    $('#rl-missing').hidden = !st.available;
    $('#rl-missing').textContent = st.available ? `Training isn't installed here: ${st.available}.` : '';
    $('#rl-start').hidden = !!st.running;
    $('#rl-start').disabled = !!st.available;
    $('#rl-stop').hidden = !st.running;
    const a = st.active;
    const kv = [['Policy', a.name || 'none'], ['Skill', a.task || '—'], ['Mode', a.on ? 'Learned (driving)' : 'not selected'],
      ['Status', a.blocked || a.status], ['Goal', a.goal ? `${a.goal[0].toFixed(2)}, ${a.goal[1].toFixed(2)} m` : '—']];
    $('#kv-rl-active').innerHTML = kv.map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join('');
    const list = st.policies || [];
    if (!chartName && list.length) chartName = list[0].name;
    $('#rl-policies').innerHTML = list.map(p => {
      const l = p.last || {};
      const stats = [l.steps != null && `${(l.steps / 1000).toFixed(0)}k steps`, l.reward != null && `reward ${l.reward}`,
        l.success != null && p.task === 'goto' && `success ${(l.success * 100).toFixed(0)}%`,
        l.success != null && p.task === 'desk-pixels' && `stays on ${(l.success * 100).toFixed(0)}%`, l.round && `round ${l.round}`, l.fps && `${l.fps} steps/s`].filter(Boolean).join(' · ');
      return `<div class="sc-card ${p.name === chartName ? 'sel' : ''}">
        <div class="sc-head"><b>${esc(p.task)}</b><span class="res-tag">${p.training ? 'Training' : p.can_continue ? 'Paused' : p.ready ? 'Ready' : 'Empty'}</span>
          <span class="sc-meta">${esc(p.sensors)} sensors${p.built_on || p.resumed_from ? ` · built on ${esc(p.built_on || p.resumed_from)}` : ''}${p.teacher ? ` · teacher ${esc(p.teacher)}` : ''} · ${new Date((p.created || 0) * 1000).toLocaleString()}</span></div>
        <p>${esc(stats || 'starting…')}</p>
        <div class="sc-foot"><span class="sc-checks">${esc(p.name)}</span>
          <button class="chip" onclick="rlChart('${p.name}')">Chart</button>
          ${p.can_continue && !st.running ? `<button class="chip" onclick="rlContinue('${p.name}')">Continue</button>` : ''}
          ${p.ready ? `<button class="chip" onclick="rlUse('${p.name}')">Use</button>` : ''}
          ${p.training ? '' : `<button class="chip" onclick="rlDelete('${p.name}')">Delete</button>`}</div>
      </div>`;
    }).join('') || '<p class="tele-note">None yet. Start one above; a quick run takes a few minutes.</p>';
    if (chartName) drawChart(await (await fetch(`/api/rl/progress/${encodeURIComponent(chartName)}`)).json());
  }

  function drawChart(rows) {
    const c = $('#rl-chart');
    const w = c.clientWidth || 600, h = 220;
    c.width = w * devicePixelRatio; c.height = h * devicePixelRatio;
    const g = c.getContext('2d');
    g.scale(devicePixelRatio, devicePixelRatio);
    const css = getComputedStyle(document.documentElement);
    const ink = css.getPropertyValue('--ink').trim(), acc = css.getPropertyValue('--acc').trim(), grn = css.getPropertyValue('--grn').trim();
    g.clearRect(0, 0, w, h);
    g.strokeStyle = ink; g.globalAlpha = 0.15; g.lineWidth = 1;
    for (let i = 0; i <= 4; i++) { g.beginPath(); g.moveTo(40, 10 + i * (h - 40) / 4); g.lineTo(w - 10, 10 + i * (h - 40) / 4); g.stroke(); }
    g.globalAlpha = 1;
    const pts = rows.filter(r => r.reward != null);
    $('#rl-legend').innerHTML = `<span style="color:${acc}">■ reward</span>` + (pts.some(r => r.success) ? ` <span style="color:${grn}">■ success</span>` : '') +
      ` <span class="sc-meta">${esc(chartName || '')}</span>`;
    if (pts.length < 2) { g.fillStyle = ink; g.globalAlpha = 0.5; g.font = '12px sans-serif'; g.fillText('Waiting for the first episodes…', 48, h / 2); g.globalAlpha = 1; return; }
    const xs = pts.map(r => r.steps), ys = pts.map(r => r.reward);
    const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys) || 1;
    const X = v => 40 + (v - x0) / Math.max(1, x1 - x0) * (w - 50);
    const Y = v => 10 + (1 - (v - y0) / Math.max(1e-6, y1 - y0)) * (h - 40);
    const line = (vals, color, yf) => { g.strokeStyle = color; g.lineWidth = 2; g.beginPath(); pts.forEach((r, i) => { const y = yf(vals[i]); i ? g.lineTo(X(xs[i]), y) : g.moveTo(X(xs[i]), y); }); g.stroke(); };
    line(ys, acc, Y);
    if (pts.some(r => r.success)) line(pts.map(r => r.success || 0), grn, v => 10 + (1 - v) * (h - 40));
    g.fillStyle = ink; g.font = '10px sans-serif';
    g.fillText(y1.toFixed(1), 4, 14); g.fillText(y0.toFixed(1), 4, h - 30);
    g.fillText(`${(x0 / 1000).toFixed(0)}k`, 40, h - 12); g.fillText(`${(x1 / 1000).toFixed(0)}k steps`, w - 80, h - 12);
  }

  // Build on = keep training an earlier policy of the same skill (stacking). The pixel student
  // also picks its desk teacher and, optionally, teleop recordings from the real desk.
  const keep = (sel, html) => { const v = [...sel.selectedOptions].map(o => o.value); sel.innerHTML = html; [...sel.options].forEach(o => { o.selected = v.includes(o.value); }); };
  function formFit() {
    if (!st) return;
    const t = $('#rl-task').value, ready = (st.policies || []).filter(p => p.ready && !p.training);
    $('#rl-sensors').disabled = ['balance', 'desk', 'desk-pixels', 'recover'].includes(t);
    keep($('#rl-resume'), '<option value="">Start fresh</option>' + ready.filter(p => p.task === t && (t.startsWith('desk') || p.sensors === $('#rl-sensors').value))
      .map(p => `<option value="${esc(p.name)}">${esc(p.name)}</option>`).join(''));
    const px = t === 'desk-pixels';
    $('#rl-teacher-row').hidden = $('#rl-real-row').hidden = !px;
    keep($('#rl-teacher'), ready.filter(p => p.task === 'desk').map(p => `<option value="${esc(p.name)}">${esc(p.name)}</option>`).join('') || '<option value="">Train a Desk teacher first</option>');
    keep($('#rl-real'), (st.recordings || []).map(e => `<option value="${esc(e.name)}">${esc(e.name)} · ${e.frames} frames</option>`).join(''));
  }

  window.rlStart = async () => {
    const r = await post('/api/rl/train', { task: $('#rl-task').value, sensors: $('#rl-sensors').value, steps: +$('#rl-steps').value,
      resume: $('#rl-resume').value, teacher: $('#rl-teacher').value, real: [...$('#rl-real').selectedOptions].map(o => o.value) });
    if (!r.ok) alert(r.error); else chartName = r.name;
    refresh();
  };
  window.rlContinue = async name => {
    const r = await post('/api/rl/continue', { name });
    if (!r.ok) alert(r.error); else chartName = name;
    refresh();
  };
  window.rlWipe = async () => {
    if (!confirm('Delete ALL training data on this computer? Every policy, every recording and downloaded Nous models. Calibration is kept. This can\'t be undone.')) return;
    const r = await post('/api/rl/wipe');
    alert(r.ok ? `Deleted ${r.policies} policies and ${r.recordings} recordings.` : r.error);
    chartName = null; refresh();
  };
  window.rlStop = async () => { await post('/api/rl/stop'); refresh(); };
  window.rlChart = name => { chartName = name; refresh(); };
  window.rlUse = async name => {
    const r = await post('/api/rl/use', { name });
    if (!r.ok) alert(r.error); else if (r.blocked) alert(`Selected. Note: ${r.blocked}`);
    refresh();
  };
  window.rlDelete = async name => {
    if (!confirm(`Delete policy ${name}?`)) return;
    await fetch(`/api/rl/${encodeURIComponent(name)}`, { method: 'DELETE' });
    if (chartName === name) chartName = null;
    refresh();
  };
  $('#rl-task').addEventListener('change', formFit);
  $('#rl-sensors').addEventListener('change', formFit);

  let calTimer = null;
  const calShow = async () => {
    const c = await (await fetch('/api/rl/calibration')).json();
    const msg = c.running ? c.progress : (c.progress || ''), out = Object.keys(c.result || {}).length ? JSON.stringify(c.result, null, 1) : 'No results yet. Run one of the tests above.';
    document.querySelectorAll('[data-cal-status]').forEach(el => { el.textContent = msg; });
    document.querySelectorAll('[data-cal-out]').forEach(el => { el.textContent = out; });
    if (!c.running) { clearInterval(calTimer); calTimer = null; }
  };
  const calWatch = () => { clearInterval(calTimer); calTimer = setInterval(calShow, 1000); calShow(); };
  window.calRun = async kind => {
    const r = await post('/api/rl/calibration', { kind });
    if (!r.ok) alert(r.error); else calWatch();
  };
  window.calStop = () => post('/api/rl/calibration/stop').then(calShow);
  window.calCopy = () => navigator.clipboard.writeText(document.querySelector('[data-cal-out]').textContent);
  window.calSend = async () => { const r = await post('/api/rl/calibration/send'); alert(r.ok ? 'Sent. Thanks!' : r.error); calShow(); };

  window.calRefresh = calShow;
  window.trainShown = on => {
    if (on) calShow();
    clearInterval(timer);
    if (on) { refresh(); timer = setInterval(refresh, 3000); }
  };
})();
