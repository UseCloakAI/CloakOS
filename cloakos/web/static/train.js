/* Training page. The top is the guided path: 3 steps, one "how long" choice, one Start button
   (the server picks what's next). Everything specific lives under Advanced. Words used here:
   driver = a trained policy; Simulator driver = the desk teacher (sees exact edges, sim only);
   Camera driver = the desk student (camera only, runs on the real Vector); test = the exam. */
(() => {
  const $ = (s, el = document) => el.querySelector(s);
  const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const post = (url, body = {}) => fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }).then(r => r.json());
  let st = null, chartName = null, timer = null;

  // Plain words for each skill, and what its chart lines mean.
  const DESC = {
    goto: 'Learns to drive to a point by trial and error (PPO) in the simulator, many rooms at once.',
    wander: 'Learns to roam without bumping into things, by trial and error (PPO) in the simulator.',
    desk: 'Simulator driver: learns to roam a random desk without falling off. It sees the exact desk edges, so it only runs in the Simulator. Step 1.',
    'desk-pixels': 'Camera driver: copies the Simulator driver using only the camera (with its real lag), so it runs on your real Vector. Step 2.',
    recover: 'Learns to get back on its treads after a fall. Experimental: needs a long run.',
  };
  const LINES = { goto: ['reward', 'reaches goal'], wander: ['reward', 'no falls'], desk: ['reward', 'stays on desk'],
    'desk-pixels': ['desk covered', 'stays on desk'], recover: ['reward', 'recovered'] };
  const num = n => n >= 1e6 ? `${(n / 1e6).toFixed(n >= 1e7 ? 0 : 1)}M` : n >= 1e3 ? `${(n / 1e3).toFixed(0)}k` : `${n}`;
  const dur = s => { s = Math.max(0, Math.round(s)); const h = Math.floor(s / 3600), m = Math.floor(s / 60) % 60;
    return h ? `${h}h ${m}m` : m ? `${m}m ${s % 60}s` : `${s}s`; };
  let pending = null, follow = true, chartRows = '', trHours = '8', showAll = false;
  const NAME = { desk: 'Simulator driver', 'desk-pixels': 'Camera driver', goto: 'Go-to driver', wander: 'Roaming driver', recover: 'Get-up driver' };
  const STEPS = [['Simulator', 'Learns to drive a desk without falling'], ['Camera', 'Learns to do it from the camera only'], ['Your desk', 'Your real drives make it better']];

  function paintPath() {
    const pl = st.plan; if (!pl) return;
    const done = [!!(pl.sim && pl.sim.passed), !!(pl.cam && pl.cam.passed), false];
    $('#tr-path').innerHTML = STEPS.map(([t, d], i) => {
      const cls = i + 1 === pl.stage ? 'now' : done[i] ? 'done' : '';
      return `<div class="tr-step ${cls}"><span class="n">${done[i] ? '✓ ' : ''}Step ${i + 1}</span><b>${t}</b><span>${d}</span></div>`;
    }).join('');
    const from = pl.continues_from ? `It continues from your best one so far, <i>${esc(pl.continues_from)}</i>.` : 'It starts fresh.';
    const drives = pl.stage >= 2 ? (pl.new_drives ? ` It also learns from ${pl.new_drives} new recorded drive${pl.new_drives > 1 ? 's' : ''}.`
      : pl.stage === 3 ? ' Tip: record a few drives on your desk first (Teleop, then "Record a drive" in the sidebar), they help the most now.' : '') : '';
    const now = st.running || (st.auto && st.auto.on);
    $('#tr-next').innerHTML = `<b>${now ? 'Now' : 'Next'}: ${esc(pl.title)}</b>${esc(pl.why)} ${now ? '' : from}${now ? '' : drives}`;
  }
  window.trLen = h => { trHours = h; document.querySelectorAll('.tr-opt').forEach(b => b.classList.toggle('on', b.dataset.h === h)); };
  window.trStart = async () => {
    pending = 'start'; follow = true; paintRun(null); $('#tr-start').disabled = true; $('#tr-start').textContent = 'Starting…';
    const r = trHours === 'keep' ? await post('/api/rl/keep', { on: true }) : await post('/api/rl/next', { hours: +trHours });
    if (!r.ok || r.error) { pending = null; alert(r.error || 'Could not start'); }
    $('#tr-start').disabled = false; $('#tr-start').textContent = 'Start training';
    fast();
  };
  window.rlMore = () => { showAll = !showAll; refresh(); };

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
    paintPath();
    $('#tr-go').hidden = !!run || pending === 'start' || !!(st.auto && st.auto.on);
    $('#tr-start').disabled = !!st.available;

    const a = st.active;
    const kv = [['Driver', a.name || 'none'], ['Kind', NAME[a.task] || a.task || '—'], ['Driving', a.on ? 'yes' : 'no'],
      ['Status', a.blocked || a.status]];
    if (a.task === 'goto') kv.push(['Goal', a.goal ? `${a.goal[0].toFixed(2)}, ${a.goal[1].toFixed(2)} m` : 'none (double-click the sim floor)']);
    $('#kv-rl-active').innerHTML = kv.map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join('');
    $('#rl-goal-note').hidden = a.task !== 'goto';

    if (run && follow) chartName = run.name;
    if (!list.some(p => p.name === chartName)) chartName = list.length ? list[0].name : null;
    const shown = showAll ? list : list.slice(0, 5);
    $('#rl-policies').innerHTML = shown.map(card).join('') || '<p class="tele-note">None yet. Hit Start training above.</p>';
    $('#rl-more').hidden = list.length <= 5;
    $('#rl-more').textContent = showAll ? 'Show fewer' : `Show all ${list.length}`;
    if (chartName) {
      const rows = await (await fetch(`/api/rl/progress/${encodeURIComponent(chartName)}`)).json();
      const key = chartName + JSON.stringify(rows.slice(-1)) + rows.length;
      if (key !== chartRows) { chartRows = key; drawChart(rows, list.find(p => p.name === chartName)); }
    } else drawChart([], null);
  }

  function paintRun(p) {
    const box = $('#rl-run'), auto = st && st.auto && st.auto.on;
    if (!p && pending !== 'start' && !auto) { box.hidden = true; return; }
    box.hidden = false;
    const l = (p && p.last) || {}, timed = p && p.hours, goal = (p && p.steps) || 0, done = l.steps || 0;
    const frac = timed ? (l.done || 0) : goal ? Math.min(1, done / goal) : 0;
    const startup = !p || !p.last, exam = p && !p.stopping && frac >= 1;
    box.classList.toggle('stopping', !!(p && p.stopping));
    const step = p ? ({ desk: 'Step 1 · Simulator', 'desk-pixels': (st.plan && st.plan.stage === 3 ? 'Step 3 · Your desk' : 'Step 2 · Camera') }[p.task] || NAME[p.task] || p.task) : '';
    $('#rl-run-name').textContent = !p ? (auto ? 'Keep training: starting the next session…' : 'Starting…')
      : `${auto ? 'Keep training · ' : ''}${step}`;
    $('#rl-run-tag').textContent = !p ? 'Starting' : p.stopping ? 'Saving' : startup ? 'Warming up' : exam ? 'Testing' : 'Training';
    const bar = $('#rl-bar-fill').parentElement;
    bar.classList.toggle('wait', !p || startup || exam || !!p.stopping);
    $('#rl-bar-fill').style.width = `${(frac * 100).toFixed(1)}%`;
    const el = p && p.since ? Date.now() / 1000 - p.since : 0;
    const left = timed ? p.hours * 3600 * (1 - frac) : l.fps && goal > done ? (goal - done) / l.fps : null;
    const best = l.success != null ? `stays on the desk ${(l.success * 100).toFixed(0)}% of the time` : null;
    const stats = [
      p && [timed ? 'Done' : 'Progress', `${(frac * 100).toFixed(0)}%`],
      el && ['Running', dur(el)],
      left && !exam && [auto ? 'Left in this session' : 'Left', `about ${dur(left)}`],
      best && ['Right now', best],
      auto && ['Sessions', `${st.auto.sessions || 0} · ${dur((st.auto.trained_s || 0) + el)} in total`],
    ].filter(Boolean);
    $('#rl-run-stats').innerHTML = stats.map(([k, v]) => `<span>${k} <b>${esc(v)}</b></span>`).join('');
    $('#rl-run-phase').textContent = !p ? (auto ? 'Each session is 2 hours and ends with a test; the better driver is always kept.' : 'Launching…')
      : p.stopping ? 'Saving where it got to. You can continue it later.'
      : exam ? 'Done: testing it on desks it has never seen…'
      : startup ? 'Setting up the simulator…' : (auto ? 'It keeps going, session after session, until you press Stop.' : '');
    $('#tr-stop').textContent = p && p.stopping ? 'Saving…' : 'Stop';
    $('#tr-stop').disabled = !!(p && p.stopping);
  }

  function testLine(ex) {
    if (!ex || !ex.checks) return '';
    const c = ex.checks, v = k => c[k] && c[k].value;
    if (c.falls) {
      const parts = [`stays on the desk in ${Math.round((1 - (v('falls') || 0)) * 100)}% of tests`];
      if (v('cells_per_min') != null) parts.push(`explores ${v('cells_per_min')} cells a minute`);
      if (v('bumps_per_min') != null) parts.push(`${v('bumps_per_min')} bumps a minute`);
      if (v('success') != null) parts.unshift(`reaches the goal ${Math.round(v('success') * 100)}% of the time`);
      return parts.join(' · ');
    }
    return Object.entries(c).map(([k, x]) => `${k.replace(/_/g, ' ')} ${x.value}`).join(' · ');
  }
  function card(p) {
    const ex = p.exam, passed = ex && ex.passed;
    const tag = p.training ? (p.stopping ? 'Saving' : 'Training') : p.can_continue ? 'Paused'
      : ex && ex.passed != null ? (passed ? 'Passed test' : 'Not passed yet') : p.ready ? 'Not tested' : 'Empty';
    const cls = !p.training && !p.can_continue && ex && ex.passed != null ? (passed ? 'res pass' : 'res fail') : '';
    const extra = ex ? [ex.parent && (ex.beats_parent ? '▲ better than the one before' : '▼ not better than the one before'),
      ex.picked === 'best' && 'kept its best moment, it drove better than the end'].filter(Boolean).join(' · ') : '';
    const when = new Date((p.created || 0) * 1000).toLocaleString([], { month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit' });
    const hrs = p.trained_s ? ` · trained ${dur(p.trained_s)}` : '';
    const useLabel = p.task === 'desk' ? 'Try in the Simulator' : 'Drive with this';
    return `<div class="sc-card tr-drv ${cls} ${p.name === chartName ? 'sel' : ''}">
      <div class="sc-head"><b>${esc(NAME[p.task] || p.task)}</b><span class="res-tag">${tag}</span>
        <span class="sc-meta">${esc(when)}${hrs}</span></div>
      <p>${esc(testLine(ex) || (p.training ? 'Training now…' : p.ready ? 'Not tested yet' : 'Stopped before it saved anything'))}${extra ? `<br>${esc(extra)}` : ''}</p>
      <div class="sc-foot"><span class="sc-checks" title="${esc(p.name)}">${esc(p.name)}</span>
        ${p.can_continue && !st.running ? `<button class="chip" onclick="rlContinue('${p.name}')">Continue</button>` : ''}
        ${p.ready && !p.training ? `<button class="chip" onclick="rlUse('${p.name}')">${useLabel}</button>` : ''}
        <button class="chip" onclick="rlChart('${p.name}')" title="Show its learning curve under Advanced">Chart</button>
        ${p.training ? '' : `<button class="chip" onclick="rlDelete('${p.name}')" title="Delete">Delete</button>`}</div>
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
    $('#rl-desc').textContent = (DESC[t] || '') + ' The big Start button above already picks the right thing for you.';
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
  window.rlChart = name => { chartName = name; chartRows = ''; follow = !!(st && st.running === name); document.querySelector('.tr-adv').open = true; refresh(); };
  window.rlUse = async name => {
    const r = await post('/api/rl/use', { name });
    if (!r.ok) alert(r.error); else if (r.blocked) alert(`It's driving now, but: ${r.blocked}`);
    refresh();
  };
  window.rlDelete = async name => {
    if (!confirm(`Delete this driver (${name})?`)) return;
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
