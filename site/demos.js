// Interactive pieces of the Exactuary deck. Figures come from data.js (generated from result files).
(function () {
  const D = window.EXA;
  const $ = (s, r = document) => r.querySelector(s);
  const el = (tag, attrs = {}, html = '') => { const e = document.createElement(tag); for (const k in attrs) e.setAttribute(k, attrs[k]); if (html) e.innerHTML = html; return e; };
  const NS = 'http://www.w3.org/2000/svg';
  const svgEl = (tag, attrs = {}) => { const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); return e; };
  const onSlide = (id, fn) => { const s = document.getElementById(id); if (s) { s.addEventListener('slide:enter', e => fn('enter', e.detail.step)); s.addEventListener('slide:step', e => fn('step', e.detail.step)); } };
  const byKey = k => D.models.find(m => m.key === k);
  const SAV = byKey('lifelib_savings');

  function dur(sec) {
    if (sec >= 86400 * 2) return `${(sec / 86400).toFixed(0)} days`;
    if (sec >= 7200) return `${(sec / 3600).toFixed(1)} hours`;
    if (sec >= 120) return `${(sec / 60).toFixed(0)} min`;
    if (sec >= 10) return `${sec.toFixed(0)} s`;
    if (sec >= 1) return `${sec.toFixed(1)} s`;
    return `${sec.toFixed(2)} s`;
  }
  const fmtS = s => s >= 1000 ? Math.round(s).toLocaleString('en-GB') : s >= 10 ? s.toFixed(0) : s >= 1 ? s.toFixed(2) : s >= 0.1 ? s.toFixed(3) : s.toFixed(4);
  const fmtX = x => x >= 100 ? Math.round(x).toLocaleString('en-GB') : x >= 10 ? x.toFixed(0) : x.toFixed(1);
  const sci = v => { if (v === null || v === undefined) return 'Not available'; if (v === 0) return '0'; const e = Math.floor(Math.log10(v)); const m = v / 10 ** e; return `${m.toFixed(1)}e${e}`; };

  function tabs(container, items, onPick, start = 0) {
    container.innerHTML = '';
    const bs = items.map((it, i) => {
      const b = el('button', { type: 'button', role: 'radio', 'aria-checked': i === start ? 'true' : 'false' }, it.label);
      b.addEventListener('click', () => { bs.forEach(x => x.setAttribute('aria-checked', 'false')); b.setAttribute('aria-checked', 'true'); onPick(it, i); });
      container.appendChild(b); return b;
    });
    onPick(items[start], start);
  }

  // ---------- title: one 10M-row run ----------
  (function () {
    const box = $('#title-bars'); if (!box) return;
    const pick = [1, 3, 4, 5, 6].map(i => SAV.steps[i]);
    const names = ['Vectorised, as lifelib supplies it', 'Compilation and parallel cores', 'Plus early exit', 'Move to a GPU', 'Reduce GPU precision'];
    const notes = ['The starting point', 'Every recorded value identical to the original', 'Proved equal in exact arithmetic; results agree to about 11 significant figures', 'Results agree to about ten significant figures', 'Portfolio totals agree to within one part in a million; single policies differ by up to 6%'];
    const t = pick.map(s => s.s * 10), lo = Math.log10(Math.min(...t)) - 0.6, hi = Math.log10(Math.max(...t));
    const sig2 = x => { const p = 10 ** (Math.floor(Math.log10(x)) - 1); return (Math.round(x / p) * p).toLocaleString('en-GB'); };
    const rows = pick.map((s, i) => {
      const r = el('div', { class: 'tb-row' + (i === 0 ? ' orig' : '') },
        `<span class="nm">${names[i]}</span><span class="tk"><span class="bar"></span></span><span class="val">${dur(t[i])}</span><span class="x">${i === 0 ? '' : sig2(t[0] / t[i]) + '×'}</span><span class="note">${notes[i]}</span>`);
      box.appendChild(r); return r;
    });
    const go = () => rows.forEach((r, i) => setTimeout(() => { r.querySelector('.bar').style.width = `${((Math.log10(t[i]) - lo) / (hi - lo)) * 100}%`; }, 250 + i * 140));
    onSlide('title', k => { if (k === 'enter') go(); });
    setTimeout(go, 400);
  })();

  // ---------- one policy ----------
  (function () {
    const cv = $('#op-canvas'), sl = $('#op-month'), out = $('#op-mval'); if (!cv) return;
    const ctx = cv.getContext('2d'), W = cv.width, H = cv.height, N = 240;
    // illustrative 20-year term assurance, age 40 at entry
    const pif = [], prem = [], out_ = [];
    let p = 1;
    for (let m = 0; m < N; m++) {
      const age = 40 + m / 12, qa = 0.0006 * Math.exp(0.085 * (age - 40)), q = 1 - (1 - qa) ** (1 / 12);
      const la = Math.max(0.10 - 0.01 * Math.floor(m / 12), 0.02), l = 1 - (1 - la) ** (1 / 12);
      pif.push(p); prem.push(p * 40); out_.push(p * q * 100000 + p * 4);
      p = p * (1 - q) * (1 - l);
    }
    const maxF = Math.max(...prem, ...out_);
    function draw(upto) {
      ctx.clearRect(0, 0, W, H);
      const L = 46, R = W - 12, T = 16, mid = 300, B = 440;
      ctx.font = '11px "Geist Mono", monospace'; ctx.fillStyle = '#8b8f99';
      ctx.strokeStyle = '#eceef2'; ctx.lineWidth = 1;
      for (let y = 0; y <= 4; y++) { const yy = T + (mid - T - 20) * y / 4; ctx.beginPath(); ctx.moveTo(L, yy); ctx.lineTo(R, yy); ctx.stroke(); }
      ctx.fillText('100%', 6, T + 4); ctx.fillText('50%', 12, T + (mid - T - 20) / 2 + 4); ctx.fillText('0%', 18, mid - 16);
      ctx.fillText('in', 18, mid + 50); ctx.fillText('out', 14, mid + 100);
      for (let yr = 0; yr <= 20; yr += 5) { const x = L + (R - L) * (yr * 12) / N; ctx.fillText(`yr ${yr}`, x - 10, B + 16); }
      const bw = (R - L) / N;
      for (let m = 0; m <= upto && m < N; m++) {
        const x = L + (R - L) * m / N;
        const hp = (prem[m] / maxF) * 60, ho = (out_[m] / maxF) * 60;
        ctx.fillStyle = '#12909a'; ctx.fillRect(x, mid + 70 - hp, Math.max(bw - 0.5, 0.8), hp);
        ctx.fillStyle = '#e5892a'; ctx.fillRect(x, mid + 72, Math.max(bw - 0.5, 0.8), ho);
      }
      ctx.strokeStyle = '#cfd2d9'; ctx.beginPath(); ctx.moveTo(L, mid + 71); ctx.lineTo(R, mid + 71); ctx.stroke();
      ctx.strokeStyle = '#1a2fd0'; ctx.lineWidth = 2; ctx.beginPath();
      for (let m = 0; m <= upto && m < N; m++) { const x = L + (R - L) * m / N, y = T + (mid - T - 20) * (1 - pif[m]); m ? ctx.lineTo(x, y) : ctx.moveTo(x, y); }
      ctx.stroke();
      if (upto > 0) {
        const m = Math.min(upto, N - 1), x = L + (R - L) * m / N, y = T + (mid - T - 20) * (1 - pif[m]);
        ctx.fillStyle = '#1a2fd0'; ctx.beginPath(); ctx.arc(x, y, 4, 0, 7); ctx.fill();
        ctx.fillStyle = '#2f323a'; ctx.font = '13px Manrope, sans-serif';
        ctx.fillText(`${(pif[m] * 100).toFixed(1)}% still in force`, Math.min(x + 10, R - 160), y - 14);
      }
    }
    const upd = () => { const m = +sl.value; out.textContent = m; draw(m); };
    sl.addEventListener('input', upd); upd();
    let timer = null;
    onSlide('one-policy', k => { if (k === 'enter' && +sl.value === 0) { clearInterval(timer); let m = 0; timer = setInterval(() => { m += 3; sl.value = m; upd(); if (m >= N) clearInterval(timer); }, 30); } });
    sl.addEventListener('pointerdown', () => clearInterval(timer));
  })();

  // ---------- cube ----------
  (function () {
    const box = $('#cube'); if (!box) return;
    const s = svgEl('svg', { viewBox: '0 0 560 420' });
    const P = (x, y) => `${x},${y}`;
    const a = [120, 120], w = 300, h = 220, d = [110, -70];
    const face = (pts, fill) => s.appendChild(svgEl('polygon', { points: pts.map(p => P(...p)).join(' '), fill, stroke: '#1a2fd0', 'stroke-width': 1.5 }));
    face([[a[0], a[1]], [a[0] + w, a[1]], [a[0] + w + d[0], a[1] + d[1]], [a[0] + d[0], a[1] + d[1]]], '#e9ecff');
    face([[a[0] + w, a[1]], [a[0] + w + d[0], a[1] + d[1]], [a[0] + w + d[0], a[1] + d[1] + h], [a[0] + w, a[1] + h]], '#d7dcfb');
    face([[a[0], a[1]], [a[0] + w, a[1]], [a[0] + w, a[1] + h], [a[0], a[1] + h]], '#ffffff');
    for (let i = 1; i < 12; i++) s.appendChild(svgEl('line', { x1: a[0] + w * i / 12, y1: a[1], x2: a[0] + w * i / 12, y2: a[1] + h, stroke: '#e2e4e9' }));
    for (let j = 1; j < 8; j++) s.appendChild(svgEl('line', { x1: a[0], y1: a[1] + h * j / 8, x2: a[0] + w, y2: a[1] + h * j / 8, stroke: '#e2e4e9' }));
    s.appendChild(svgEl('rect', { x: a[0], y: a[1] + h * 3 / 8, width: w, height: h / 8, fill: 'rgba(26,47,208,.16)' }));
    const t = (x, y, txt, anchor = 'middle', rot = 0) => { const e = svgEl('text', { x, y, 'text-anchor': anchor, 'font-family': 'Geist Mono, monospace', 'font-size': 13, fill: '#5b5f6a', transform: rot ? `rotate(${rot} ${x} ${y})` : '' }); e.textContent = txt; s.appendChild(e); };
    t(a[0] + w / 2, a[1] + h + 28, 'months →');
    t(a[0] - 18, a[1] + h / 2, 'policies →', 'middle', -90);
    t(a[0] + w + d[0] / 2 + 6, a[1] + d[1] / 2 - 14, 'scenarios ↗', 'middle', 0);
    t(a[0] + w / 2, a[1] + h * 3 / 8 + 18, 'one projection: a row', 'middle');
    box.appendChild(s);
  })();

  // ---------- grid demo: four ways ----------
  (function () {
    const grid = $('#gd-grid'), note = $('#gd-note'), tabsBox = $('#gd-tabs'); if (!grid) return;
    const R = 8, C = 24, ends = [24, 15, 24, 9, 20, 24, 12, 18];
    const cells = [];
    for (let r = 0; r < R; r++) for (let c = 0; c < C; c++) { const i = el('i'); grid.appendChild(i); cells.push(i); }
    const cell = (r, c) => cells[r * C + c];
    const sv = SAV.steps;
    const modes = [
      { label: 'One at a time', title: 'One number at a time', text: 'Python reads and works out each instruction again for every single number. This is simple and flexible, but most of the time goes on handling each instruction, and little on the arithmetic itself.', kv: [['Worked on together', 'one square'], ['Kept in memory', 'one policy'], ['Savings model', `${fmtS(sv[0].s)} s per million rows`]], tick: 35 },
      { label: 'Vectorised', title: 'Vectorised: one month for all policies', text: 'The computer works out a whole column at once, which removes most of the bookkeeping. But months still come one after another, the computer keeps every month of every policy in memory, and it keeps calculating matured policies to the end.', kv: [['Worked on together', 'a column'], ['Kept in memory', 'everything (orange)'], ['Savings model', `${fmtS(sv[1].s)} s per million rows`]], tick: 220 },
      { label: 'Compiled', title: 'Compiled: one policy through all its months', text: 'The model is compiled to machine code. Each policy runs from its first month to its last and passes a small set of values to the next month. The striped squares show early exit, which is a separate change at level 4: the projection stops when nothing is left in force.', kv: [['Worked on together', 'one row, very fast'], ['Kept in memory', 'two numbers per policy'], ['Savings model', `${fmtS(sv[2].s)} s per million rows`]], tick: 22 },
      { label: 'All cores / GPU', title: 'Many policies at once', text: 'Policies do not affect each other, so the computer shares the rows out: 20 at a time across the processor cores here, and thousands at a time on a graphics processor. Each row still runs its months in order.', kv: [['Worked on together', 'many rows'], ['Kept in memory', 'two numbers per policy'], ['Savings model', `${fmtS(sv[4].s)} s (cores) · ${fmtS(sv[6].s)} s (GPU)`]], tick: 160 },
    ];
    let timer = null, mode = 0, active = false;
    function reset() { cells.forEach(c => (c.className = '')); for (let r = 0; r < R; r++) for (let c = ends[r]; c < C; c++) if (mode >= 2) cell(r, c).className = 'skip'; }
    function run() {
      clearInterval(timer); reset(); let k = 0;
      const m = modes[mode];
      timer = setInterval(() => {
        if (!active) return;
        cells.forEach(c => c.classList.remove('now'));
        if (mode === 0) { const r = Math.floor(k / C), c = k % C; if (r >= R) { k = 0; reset(); return; } cell(r, c).classList.add('now', 'done'); k++; }
        else if (mode === 1) { const c = k; if (c >= C) { k = 0; reset(); return; } for (let r = 0; r < R; r++) { cell(r, c).classList.add('now', 'done', 'held'); } k++; }
        else if (mode === 2) {
          let left = k, r = 0; while (r < R && left >= ends[r]) { left -= ends[r]; r++; }
          if (r >= R) { k = 0; reset(); return; }
          if (left === 0 && r > 0) for (let c = 0; c < ends[r - 1]; c++) cell(r - 1, c).classList.add('done');
          cell(r, left).classList.add('now', 'done'); k++;
        } else { const c = k; if (c >= C) { k = 0; reset(); return; } for (let r = 0; r < R; r++) if (c < ends[r]) cell(r, c).classList.add('now', 'done'); k++; }
      }, m.tick);
      note.innerHTML = `<span class="label">${m.label}</span><h3>${m.title}</h3><p>${m.text}</p>` + m.kv.map(([a, b]) => `<div class="kv"><span>${a}</span><b>${b}</b></div>`).join('');
    }
    tabs(tabsBox, modes, (m, i) => { mode = i; run(); });
    const sl = grid.closest('.slide');
    new MutationObserver(() => { active = sl.classList.contains('active'); }).observe(sl, { attributes: true, attributeFilter: ['class'] });
  })();

  // ---------- speed ladder ----------
  (function () {
    const box = $('#lad-chart'), side = $('#lad-side'), tb = $('#lad-tabs'); if (!box) return;
    const tip = el('div', { class: 'lad-tip' }); box.appendChild(tip);
    const descr = [
      ['Compilation', 'Each policy runs through its months as machine code, passing two or three numbers forward.'],
      ['Parallel cores', 'All 20 processor cores share the policies. Results are identical to one core.'],
      ['Early exit', 'A projection ends once no policies remain in force, not at the longest term.'],
      ['Move to a GPU', 'Thousands of policies at once, keeping about 16 digits. The last digits can differ.'],
      ['Reduce GPU precision', 'Keeping about seven digits. Many times faster on most graphics processors, and less accurate.'],
    ];
    side.innerHTML = descr.map(([a, b], i) => `<div class="ls-step" data-i="${i}"><b>${a}</b><span>${b}</span></div>`).join('');
    let model = SAV, step = 0;
    const W = 860, rowH = 50, L = 250, Rr = 150;
    function render() {
      box.querySelectorAll('svg').forEach(s => s.remove());
      const st = model.steps, n = st.length, H = 34 + n * rowH;
      const s = svgEl('svg', { viewBox: `0 0 ${W} ${H}` });
      const lo = -3, hi = 6, x = v => L + (W - L - Rr) * (Math.log10(v) - lo) / (hi - lo);
      const ax = svgEl('g', { class: 'ax' });
      for (let e = lo; e <= hi; e++) {
        ax.appendChild(svgEl('line', { x1: x(10 ** e), x2: x(10 ** e), y1: 20, y2: H }));
        const t = svgEl('text', { x: x(10 ** e), y: 12, 'text-anchor': 'middle' }); t.textContent = `${(10 ** e).toLocaleString('en-GB')} s`; ax.appendChild(t);
      }
      s.appendChild(ax);
      const nOrig = st.filter(z => z.kind === 'original').length, vec = st[nOrig - 1].s;
      st.forEach((z, i) => {
        const g = svgEl('g', { class: 'row' + (i >= nOrig + step ? ' hidden' : '') }), y = 28 + i * rowH;
        const lab = svgEl('text', { x: L - 14, y: y + 20, 'text-anchor': 'end', class: 'lbl' }); lab.textContent = z.label; g.appendChild(lab);
        g.appendChild(svgEl('rect', { class: 'bar', x: L, y: y + 6, width: Math.max(2, x(z.s) - L), height: 20, rx: 0, fill: z.kind === 'original' ? '#cfd2d9' : '#1a2fd0' }));
        g.appendChild(svgEl('rect', { x: x(z.s) - 4, y: y + 6, width: 4, height: 20, rx: 2, fill: z.kind === 'original' ? '#cfd2d9' : '#1a2fd0' }));
        const v = svgEl('text', { x: x(z.s) + 8, y: y + 21, class: 'val' }); v.textContent = `${fmtS(z.s)} s`; g.appendChild(v);
        if (i > 0) { const m = svgEl('text', { x: W - 4, y: y + 21, 'text-anchor': 'end', class: 'mul' }); m.textContent = `×${fmtX(st[i - 1].s / z.s)}`; g.appendChild(m); }
        const hit = svgEl('rect', { class: 'hit', x: 0, y: y, width: W, height: rowH }); g.appendChild(hit);
        hit.addEventListener('mousemove', ev => {
          const b = box.getBoundingClientRect(), sc = b.width / box.offsetWidth;
          tip.innerHTML = `<b>${z.label}</b><div><span>Per million rows</span><span>${fmtS(z.s)} s</span></div><div><span>10 million rows</span><span>${dur(z.s * 10)}</span></div>` +
            (i > 0 ? `<div><span>Compared with the previous step</span><span>×${fmtX(st[i - 1].s / z.s)}</span></div>` : '') +
            (i >= nOrig ? `<div><span>Compared with the vectorised original</span><span>×${fmtX(vec / z.s)}</span></div>` : '');
          tip.style.left = `${Math.min((ev.clientX - b.left) / sc + 14, box.offsetWidth - 240)}px`; tip.style.top = `${(ev.clientY - b.top) / sc + 12}px`; tip.classList.add('on');
        });
        hit.addEventListener('mouseleave', () => tip.classList.remove('on'));
        s.appendChild(g);
      });
      box.insertBefore(s, tip);
      side.querySelectorAll('.ls-step').forEach(d => d.classList.toggle('on', +d.dataset.i < step));
    }
    const items = D.models.map(m => ({ label: m.name, m }));
    tabs(tb, items, it => { model = it.m; render(); });
    onSlide('ladder', (k, st) => { step = st; render(); });
  })();

  // ---------- steps table ----------
  (function () {
    const tb = $('#steps-body'); if (!tb) return;
    const st = SAV.steps;
    const rows = [
      ['Vectorised, as lifelib supplies it', st[1].s, st[0].s, 'The computer works out all policies together, one month at a time', '<span class="yes">Yes</span>: it is the reference', 'No conversion for these lifelib examples'],
      ['Compilation', st[2].s, st[1].s, 'Each policy runs as machine code, keeping only a few numbers', '<span class="yes">Yes</span> for recorded values on the test portfolios', 'Rewrite the model in a simple, restricted form'],
      ['Parallel cores', st[3].s, st[2].s, 'The cores share the policies', '<span class="yes">Yes</span>', 'None'],
      ['Early exit', st[4].s, st[3].s, 'Projections end early', '<span class="part">Equal in exact arithmetic</span> under stated assumptions', 'A proof that stopping loses nothing'],
      ['Move to a GPU', st[5].s, st[4].s, 'The same sums on different hardware', '<span class="part">Last digits differ</span>', 'A graphics processor, and error bounds'],
      ['Reduce GPU precision', st[6].s, st[5].s, 'About seven digits instead of 16', '<span class="part">No: depends on the output; see section 06</span>', 'Error bounds, checked for each reported output'],
    ];
    tb.innerHTML = rows.map(r => `<tr><td>${r[0]}</td><td class="num">×${fmtX(r[2] / r[1])}</td><td>${r[3]}</td><td>${r[4]}</td><td>${r[5]}</td></tr>`).join('');
  })();

  // ---------- hero row ----------
  (function () {
    const box = $('#hero-row'); if (!box) return;
    const st = SAV.steps;
    const pick = [[0, 'One policy at a time'], [1, 'Vectorised, as lifelib supplies it'], [2, 'Compilation'], [4, 'Plus parallel cores and early exit'], [5, 'Move to a GPU'], [6, 'Reduce GPU precision']];
    box.innerHTML = pick.map(([i, lab]) => `<div class="${st[i].kind === 'kernel' ? 'k' : ''}"><span class="label">${st[i].kind === 'kernel' ? 'Fast model' : 'Original'}</span><span class="big-num">${dur(st[i].s * 10)}</span><span class="lab">${lab}</span></div>`).join('');
  })();

  // ---------- defects ----------
  (function () {
    const box = $('#df-chart'), tb = $('#df-tabs'); if (!box) return;
    const names = { 'portfolio totals within 1e-6, sample portfolio': 'Portfolio totals agree to one part in a million', 'per-policy present values within 1e-9, sample portfolio': 'Each policy\'s present values agree to one part per billion', 'every intermediate exact, sample portfolio': 'Every recorded value exactly, sample portfolio', 'every intermediate exact, all portfolios': 'Every recorded value exactly, sample and unusual cases' };
    tabs(tb, D.models.map(m => ({ label: m.name, m })), it => {
      const rows = it.m.defects;
      box.innerHTML = rows.map((r, i) => `<div class="df-row ${i < 2 ? 'weak' : ''}"><div class="nm"><span>${names[r.check] || r.check}</span><b>${r.caught} of ${r.of} · ${Math.round(100 * r.caught / r.of)}%</b></div><div class="df-track"><div class="df-bar" style="width:0"></div></div></div>`).join('') +
        `<p class="src" style="margin-top:12px">Only planted errors that still let the model run are counted. ${it.m.name} model.</p>`;
      requestAnimationFrame(() => box.querySelectorAll('.df-bar').forEach((b, i) => { b.style.width = `${100 * rows[i].caught / rows[i].of}%`; }));
    });
  })();

  // ---------- dominoes ----------
  (function () {
    const box = $('#dom'); if (!box) return;
    const ds = [];
    for (let i = 0; i < 12; i++) { const d = el('div', { class: 'd' + (i === 0 ? ' base' : '') }, `<span>${i === 0 ? 'month 0' : i}</span>`); box.appendChild(d); ds.push(d); }
    box.appendChild(el('div', { class: 'more' }, '… month 1,140'));
    let timers = [];
    onSlide('dominoes', (k, st) => {
      timers.forEach(clearTimeout); timers = [];
      ds.forEach(d => d.classList.remove('fall'));
      if (st >= 3) ds.forEach((d, i) => timers.push(setTimeout(() => d.classList.add('fall'), 200 + i * 120)));
    });
  })();

  // ---------- floats ----------
  (function () {
    const inp = $('#fl-in'); if (!inp) return;
    const mark = (s, ref) => { let out = '', ok = true; for (let i = 0; i < s.length; i++) { if (ok && s[i] !== ref[i]) ok = false; out += ok ? s[i] : `<span class="bad">${s[i]}</span>`; } return out; };
    function upd() {
      const t = inp.value.trim(), x = Number(t);
      if (!isFinite(x) || t === '') { ['#fl-64', '#fl-32', '#fl-can'].forEach(q => ($(q).textContent = 'Enter a finite number')); return; }
      const exact = t.includes('.') ? t + '0'.repeat(25) : t + '.' + '0'.repeat(25);
      const d = x.toPrecision(21), f = Math.fround(x).toPrecision(21);
      const norm = v => { const n = Number(v); return n.toFixed(Math.max(0, 20 - Math.floor(Math.log10(Math.abs(n) || 1)))).slice(0, 23); };
      $('#fl-64').innerHTML = mark(norm(d), exact);
      $('#fl-32').innerHTML = mark(norm(f), exact);
      const c = Math.fround(Math.fround(1e6 + x) - 1e6);
      $('#fl-can').innerHTML = `${c} <span class="label" style="margin-left:8px">exact result: ${t}</span>`;
    }
    inp.addEventListener('input', upd); upd();
  })();

  // ---------- interval ----------
  (function () {
    const s = $('#iv-svg'); if (!s) return;
    const g = (attrs, step) => { const e = svgEl('g', attrs); if (step) e.setAttribute('data-iv', step); s.appendChild(e); return e; };
    const add = (p, tag, attrs, txt) => { const e = svgEl(tag, attrs); if (txt) e.textContent = txt; p.appendChild(e); return e; };
    const base = g({});
    add(base, 'line', { x1: 30, x2: 730, y1: 160, y2: 160, stroke: '#cfd2d9', 'stroke-width': 2 });
    add(base, 'rect', { x: 330, y: 136, width: 100, height: 48, fill: 'rgba(26,47,208,.10)', stroke: '#1a2fd0', 'stroke-width': 1.5 });
    add(base, 'text', { x: 330, y: 206, 'text-anchor': 'middle' }, 'lower');
    add(base, 'text', { x: 430, y: 206, 'text-anchor': 'middle' }, 'upper');
    add(base, 'line', { x1: 372, x2: 372, y1: 128, y2: 192, stroke: '#0c0d12', 'stroke-width': 2, 'stroke-dasharray': '3 3' });
    add(base, 'text', { x: 372, y: 118, 'text-anchor': 'middle', class: 'lbl' }, 'exact answer (unknown)');
    add(base, 'text', { x: 380, y: 236, 'text-anchor': 'middle', class: 'lbl' }, 'the exact answer must lie inside the box');
    const s1 = g({}, 1);
    add(s1, 'circle', { cx: 365, cy: 160, r: 7, fill: '#1a2fd0', stroke: '#fff', 'stroke-width': 2 });
    add(s1, 'circle', { cx: 392, cy: 160, r: 7, fill: '#3d50e6', stroke: '#fff', 'stroke-width': 2 });
    add(s1, 'text', { x: 300, y: 60, class: 'lbl' }, 'processor, and GPU at 16 digits:');
    add(s1, 'text', { x: 300, y: 80, class: 'lbl' }, 'inside or next to the box');
    add(s1, 'line', { x1: 370, y1: 86, x2: 375, y2: 150, stroke: '#8b8f99' });
    const s2 = g({}, 2);
    add(s2, 'circle', { cx: 610, cy: 160, r: 7, fill: '#e5892a', stroke: '#fff', 'stroke-width': 2 });
    add(s2, 'text', { x: 610, y: 128, 'text-anchor': 'middle', class: 'lbl' }, 'GPU at seven digits');
    add(s2, 'line', { x1: 330, x2: 610, y1: 280, y2: 280, stroke: '#e5892a', 'stroke-width': 2 });
    add(s2, 'line', { x1: 330, x2: 330, y1: 272, y2: 288, stroke: '#e5892a', 'stroke-width': 2 });
    add(s2, 'line', { x1: 610, x2: 610, y1: 272, y2: 288, stroke: '#e5892a', 'stroke-width': 2 });
    add(s2, 'text', { x: 470, y: 306, 'text-anchor': 'middle', class: 'lbl' }, 'its error is at most this distance, whatever the device did');
    const apply = st => s.querySelectorAll('[data-iv]').forEach(e => e.classList.toggle('off', +e.dataset.iv > st));
    apply(0); onSlide('interval', (k, st) => apply(st));
  })();

  // ---------- certified distances ----------
  (function () {
    const hd = $('#cert-head'), bd = $('#cert-body'); if (!hd) return;
    const cols = [["original's summation (float64)", 'Fast-model values, original summation order'], ['gpu float64', 'GPU, 16 digits'], ['cpu float32', 'Processor, seven digits'], ['gpu float32', 'GPU, seven digits']];
    hd.innerHTML = `<tr><th>Model</th><th class="num">Rows</th>${cols.map(c => `<th class="num">${c[1]}</th>`).join('')}<th class="num">Net cashflow, GPU at seven digits</th></tr>`;
    bd.innerHTML = D.models.map(m => {
      const v = m.certify.variants;
      return `<tr><td>${m.name}</td><td class="num">${m.certify.rows.toLocaleString('en-GB')}</td>` + cols.map(c => { const w = v[c[0]].worst; return `<td class="num ${w > 1e-3 ? 'hi' : ''}">${sci(w)}</td>`; }).join('') +
        `<td class="num ${v['gpu float32'].net_worst > 1e-3 ? 'hi' : ''}">${sci(v['gpu float32'].net_worst)}</td></tr>`;
    }).join('');
  })();
})();
