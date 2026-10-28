/* Training page: start/stop RL runs, chart progress, pick the policy that drives. */
(() => {
  const $ = (s, el = document) => el.querySelector(s);
  const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const post = (url, body = {}) => fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }).then(r => r.json());
  let st = null, chartName = null, timer = null;

  // Plain words for each skill, and what its chart lines mean.
  const DESC = {
    goto: 'Learns to drive to a point by trial and error (PPO) in the simulator, many rooms at once.',
    wander: 'Learns to roam without bumping into things, by trial and error (PPO) in the simulator.',
    desk: 'The desk teacher: learns to roam a random desk without falling off. It cheats (sees the exact desk edges), so it only runs in the sim. Train this first.',
    'desk-pixels': 'The desk student: copies the teacher using only the camera (with the real camera lag), so it can run on the real Vector.',
    recover: 'Learns to get back on its treads after a fall. Experimental: needs a long run.',
  };
  const LINES = { goto: ['reward', 'reaches goal'], wander: ['reward', 'no falls'], desk: ['reward', 'stays on desk'],
    'desk-pixels': ['desk covered', 'stays on desk'], recover: ['reward', 'recovered'] };
  const num = n => n >= 1e6 ? `${(n / 1e6).toFixed(n >= 1e7 ? 0 : 1)}M` : n >= 1e3 ? `${(n / 1e3).toFixed(0)}k` : `${n}`;
  const dur = s => { s = Math.max(0, Math.round(s)); const h = Math.floor(s / 3600), m = Math.floor(s / 60) % 60;
    return h ? `${h}h ${m}m` : m ? `${m}m ${s % 60}s` : `${s}s`; };
  let pending = null, follow = true, chartRows = '';

  async function refresh() {
    try { st = await (await fetch('/api/rl')).json(); } catch { return; }
    const task = $('#rl-task'), sens = $('#rl-sensors');
    if (!task.options.length) {
      task.innerHTML = Object.entries(st.tasks).map(([k, v]) => `<option value="${k}">${esc(v)}</option>`).join('');
      sens.innerHTML = Object.entries(st.sensors).map(([k, v]) => `<option value="${k}">${esc(v)}</option>`).join('');
      sens.value = 'map';
    }
    formFit();
    const list = st.policies || [], run = list.find(p => p.training);
    if (pending === 'start' && run) pending = null;
    if (pending === 'stop' && !run) pending = null;
    window.trainDot && window.trainDot(!!run);
    $('#rl-missing').hidden = !st.available;
    $('#rl-missing').textContent = st.available ? `Training isn't installed here: ${st.available}.` : '';
    const busy = !!st.running || pending === 'start';
    $('#rl-start').hidden = busy && pending !== 'start';
    $('#rl-start').disabled = !!st.available || pending === 'start';
    $('#rl-start').textContent = pending === 'start' ? 'Starting…' : 'Start training';
    $('#rl-stop').hidden = !st.running;
    $('#rl-stop').disabled = !!(run && run.stopping) || pending === 'stop';
    $('#rl-stop').textContent = run && run.stopping || pending === 'stop' ? 'Saving…' : 'Stop and save';
    paintRun(run);

    const a = st.active;
    const kv = [['Policy', a.name || 'none'], ['Skill', a.task || '—'], ['Mode', a.on ? 'Learned (driving)' : 'not selected'],
      ['Status', a.blocked || a.status]];
    if (a.task === 'goto') kv.push(['Goal', a.goal ? `${a.goal[0].toFixed(2)}, ${a.goal[1].toFixed(2)} m` : 'none (double-click the sim floor)']);
    $('#kv-rl-active').innerHTML = kv.map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join('');
    $('#rl-goal-note').hidden = a.task !== 'goto';

    if (run && follow) chartName = run.name;
    if (!list.some(p => p.name === chartName)) chartName = list.length ? list[0].name : null;
    $('#rl-policies').innerHTML = list.map(card).join('') || '<p class="tele-note">None yet. Pick a skill above and hit Start; a quick run takes a few minutes.</p>';
    if (chartName) {
      const rows = await (await fetch(`/api/rl/progress/${encodeURIComponent(chartName)}`)).json();
      const key = chartName + JSON.stringify(rows.slice(-1)) + rows.length;
      if (key !== chartRows) { chartRows = key; drawChart(rows, list.find(p => p.name === chartName)); }
    } else drawChart([], null);
  }

  function paintRun(p) {
    const box = $('#rl-run');
    if (!p && pending !== 'start') { box.hidden = true; return; }
    box.hidden = false;
    const l = (p && p.last) || {}, goal = (p && p.steps) || 0, done = l.steps || 0;
    const startup = !p || !p.last, exam = p && !p.stopping && goal && done >= goal;
    box.classList.toggle('stopping', !!(p && p.stopping));
    $('#rl-run-name').textContent = p ? `${p.task} · ${p.name}` : 'Starting…';
    $('#rl-run-tag').textContent = !p ? 'Starting' : p.stopping ? 'Saving' : startup ? 'Warming up' : exam ? 'Exam' : 'Training';
    const bar = $('#rl-bar-fill').parentElement;
    bar.classList.toggle('wait', startup || exam || !!(p && p.stopping));
    $('#rl-bar-fill').style.width = goal ? `${Math.min(100, done / goal * 100).toFixed(1)}%` : '0';
    const el = p && p.since ? Date.now() / 1000 - p.since : 0;
    const left = l.fps && goal > done ? (goal - done) / l.fps : null;
    const stats = [
      goal && ['Progress', `${num(done)} / ${num(goal)} steps (${Math.min(100, done / goal * 100).toFixed(0)}%)`],
      el && ['Running', dur(el)],
      left && !exam && ['Time left', `about ${dur(left)}`],
      l.fps && ['Speed', `${num(l.fps)} steps/s`],
      l.round && ['Round', `${l.round}`],
    ].filter(Boolean);
    $('#rl-run-stats').innerHTML = stats.map(([k, v]) => `<span>${k} <b>${esc(v)}</b></span>`).join('');
    const tail = ((p && p.log_tail) || '').trim().split('\n').pop() || '';
    $('#rl-run-phase').textContent = !p ? 'Launching the trainer…'
      : p.stopping ? 'Finishing the current step and saving. You can Continue it later.'
      : exam ? 'Done training. Taking the exam on rooms it has never seen…'
      : startup ? (tail && !tail.startsWith('{') ? `${tail[0].toUpperCase()}${tail.slice(1)}…` : 'Loading the simulator. The first point on the chart shows up after the first batch.')
      : tail.startsWith('round') ? `${tail[0].toUpperCase()}${tail.slice(1)}…` : '';
  }

  function card(p) {
    const l = p.last || {}, lines = LINES[p.task] || ['reward', 'success'];
    const stats = [l.steps != null && `${num(l.steps)}${p.steps ? ` / ${num(p.steps)}` : ''} steps`,
      l.reward != null && `${lines[0]} ${l.reward}`, l.success != null && `${lines[1]} ${(l.success * 100).toFixed(0)}%`,
      l.round && `round ${l.round}`].filter(Boolean).join(' · ');
    const ex = p.exam, tag = p.training ? (p.stopping ? 'Saving' : 'Training') : p.can_continue ? 'Paused'
      : ex && ex.passed != null ? (ex.passed ? 'Passed' : 'Failed exam') : p.ready ? 'Ready' : 'Empty';
    const cls = ex && !p.training && !p.can_continue && ex.passed != null ? (ex.passed ? 'res pass' : 'res fail') : '';
    const checks = ex && ex.checks ? Object.entries(ex.checks).map(([k, c]) => `${c.ok ? '✓' : '✗'} ${k.replace(/_/g, ' ')} ${c.value ?? '—'} (want ${c.want})`).join(' · ') : '';
    return `<div class="sc-card ${cls} ${p.name === chartName ? 'sel' : ''}">
      <div class="sc-head"><b>${esc(p.task)}</b><span class="res-tag">${tag}</span>
        <span class="sc-meta">${p.task.startsWith('desk') || p.task === 'recover' ? '' : `${esc(p.sensors)} sensors · `}${p.built_on || p.resumed_from ? `built on ${esc(p.built_on || p.resumed_from)} · ` : ''}${p.teacher ? `teacher ${esc(p.teacher)} · ` : ''}${new Date((p.created || 0) * 1000).toLocaleString()}</span></div>
      <p>${esc(stats || (p.training ? 'warming up…' : 'stopped before the first checkpoint'))}${checks ? `<br>${esc(checks)}` : ''}</p>
      <div class="sc-foot"><span class="sc-checks">${esc(p.name)}</span>
        <button class="chip" onclick="rlChart('${p.name}')">Chart</button>
        ${p.can_continue && !st.running ? `<button class="chip" onclick="rlContinue('${p.name}')">Continue</button>` : ''}
        ${p.ready ? `<button class="chip" onclick="rlUse('${p.name}')">Use</button>` : ''}
        ${p.training ? '' : `<button class="chip" onclick="rlDelete('${p.name}')">Delete</button>`}</div>
    </div>`;
  }

  function drawChart(rows, pol) {
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
    const pts = rows.filter(r => r.reward != null), lines = LINES[pol && pol.task] || ['reward', 'success'];
    $('#rl-legend').innerHTML = `<span style="color:${acc}">■ ${lines[0]}</span>` + (pts.some(r => r.success) ? ` <span style="color:${grn}">■ ${lines[1]} (0–100%)</span>` : '') +
      ` <span class="sc-meta">${esc(chartName || '')}</span>`;
    if (pts.length < 2) { g.fillStyle = ink; g.globalAlpha = 0.5; g.font = '12px sans-serif'; g.fillText(!pol ? 'No runs yet.' : pol.training ? 'Waiting for the first batch… (a minute or two)' : 'Not enough data to chart.', 48, h / 2); g.globalAlpha = 1; return; }
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
    $('#rl-sensors-row').hidden = ['balance', 'desk', 'desk-pixels', 'recover'].includes(t);
    $('#rl-desc').textContent = (DESC[t] || '') + ' It runs in the background; the brain keeps working.';
    keep($('#rl-resume'), '<option value="">Start fresh</option>' + ready.filter(p => p.task === t && (t.startsWith('desk') || p.sensors === $('#rl-sensors').value))
      .map(p => `<option value="${esc(p.name)}">${esc(p.name)}</option>`).join(''));
    const px = t === 'desk-pixels';
    $('#rl-teacher-row').hidden = $('#rl-real-row').hidden = !px;
    keep($('#rl-teacher'), ready.filter(p => p.task === 'desk').map(p => `<option value="${esc(p.name)}">${esc(p.name)}</option>`).join('') || '<option value="">Train a Desk teacher first</option>');
    keep($('#rl-real'), (st.recordings || []).map(e => `<option value="${esc(e.name)}">${esc(e.name)} · ${e.frames} frames</option>`).join(''));
  }

  const fast = () => { refresh(); clearInterval(timer); timer = setInterval(refresh, 1000); setTimeout(() => { clearInterval(timer); timer = setInterval(refresh, 3000); }, 20000); };
  window.rlStart = async () => {
    pending = 'start'; follow = true; paintRun(null); $('#rl-start').disabled = true; $('#rl-start').textContent = 'Starting…';
    const r = await post('/api/rl/train', { task: $('#rl-task').value, sensors: $('#rl-sensors').value, steps: +$('#rl-steps').value,
      resume: $('#rl-resume').value, teacher: $('#rl-teacher').value, real: [...$('#rl-real').selectedOptions].map(o => o.value) });
    if (!r.ok) { pending = null; alert(r.error); } else chartName = r.name;
    fast();
  };
  window.rlContinue = async name => {
    pending = 'start'; follow = true; paintRun(null);
    const r = await post('/api/rl/continue', { name });
    if (!r.ok) { pending = null; alert(r.error); } else chartName = name;
    fast();
  };
  window.rlWipe = async () => {
    if (!confirm('Delete ALL training data on this computer? Every policy, every recording and downloaded Nous models. Calibration is kept. This can\'t be undone.')) return;
    const r = await post('/api/rl/wipe');
    alert(r.ok ? `Deleted ${r.policies} policies and ${r.recordings} recordings.` : r.error);
    chartName = null; refresh();
  };
  window.rlStop = async () => {
    pending = 'stop'; $('#rl-stop').disabled = true; $('#rl-stop').textContent = 'Saving…';
    await post('/api/rl/stop'); fast();
  };
  window.rlChart = name => { chartName = name; chartRows = ''; follow = !!(st && st.running === name); refresh(); };
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
    if (on) { chartRows = ''; refresh(); timer = setInterval(refresh, 3000); }
  };
  // the sidebar's Training button pulses while a run is going, on any page
  setInterval(() => { if (!timer || $('#page-train').hidden) fetch('/api/rl/running').then(r => r.json()).then(x => window.trainDot && window.trainDot(!!x.running)).catch(() => {}); }, 10000);
})();
