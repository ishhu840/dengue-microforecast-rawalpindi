#!/usr/bin/env python3
"""Assemble the single-file web app.

Everything -- forecasts, boundaries, styles, script -- is inlined into one
index.html. Two reasons: the page opens by double-click with no web server,
which matters when it is handed to someone at a district health office; and it
has no external dependency to break, so no map tile provider, CDN or font host
sits between the user and the forecast.

The map and the season chart are drawn as inline SVG rather than with a mapping
or charting library, for the same reason.

Two views. **Next week** is the live forecast from real-time weather, carrying
the caveat that with no case feed its UC ranking is only a historical-burden
prior. **2024 season** is the validated replay, where every week was forecast
from earlier data only and is shown against what actually happened -- that view
is the evidence the method works.

Run: python src/08_build_app.py   (after 06 and 07; ideally after 09-12 too)
"""

from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
APP_DATA = APP / "data"


def rings(geometry: dict) -> list:
    if geometry["type"] == "Polygon":
        return geometry["coordinates"]
    if geometry["type"] == "MultiPolygon":
        return [ring for polygon in geometry["coordinates"] for ring in polygon]
    return []


def project(geojson: dict) -> tuple[dict, dict]:
    """Lon/lat to SVG coordinates.

    Equirectangular, with longitude squeezed by cos(latitude) so the city is not
    stretched sideways. Over ~25km of one city the error against a proper
    projection is far below the width of a stroke.
    """
    xs, ys = [], []
    for feature in geojson["features"]:
        for ring in rings(feature["geometry"]):
            for lon, lat in ring:
                xs.append(lon)
                ys.append(lat)

    min_x, max_x, min_y, max_y = min(xs), max(xs), min(ys), max(ys)
    scale_x = math.cos(math.radians((min_y + max_y) / 2))
    width, height = (max_x - min_x) * scale_x, max_y - min_y
    unit = 1000.0 / max(width, height)

    shapes = {}
    for feature in geojson["features"]:
        paths = []
        for ring in rings(feature["geometry"]):
            points = [f"{(lon - min_x) * scale_x * unit:.1f},{(max_y - lat) * unit:.1f}"
                      for lon, lat in ring]
            if len(points) > 2:
                paths.append("M" + "L".join(points) + "Z")
        if paths:
            shapes[feature["properties"]["uc"]] = " ".join(paths)

    return shapes, {"w": round(width * unit, 1), "h": round(height * unit, 1)}


HTML = """<title>Rawalpindi Dengue Forecast</title>
<style>
:root {
  --bg:#f5f6f8; --panel:#fff; --panel2:#fafbfc; --ink:#12161c; --muted:#5d6673;
  --line:#e1e5ea; --accent:#2f6f9f; --accent-soft:rgba(47,111,159,.09);
  --warn-bg:#fdf6e3; --warn-line:#e0c56a; --warn-ink:#6b5416;
  --green:#7fb069; --watch:#4b9cd3; --yellow:#e8b931; --orange:#e07b39; --red:#c8412f;
  --shadow:0 1px 2px rgba(18,22,28,.05), 0 6px 20px rgba(18,22,28,.05);
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg:#0e1116; --panel:#161a20; --panel2:#1b2027; --ink:#e9edf2; --muted:#98a2af;
  --line:#252b34; --accent:#6fb3e0; --accent-soft:rgba(111,179,224,.12);
  --warn-bg:#2a2416; --warn-line:#6b5a24; --warn-ink:#e8d08a;
  --shadow:0 1px 2px rgba(0,0,0,.45), 0 6px 22px rgba(0,0,0,.35);
} }
:root[data-theme="dark"] {
  --bg:#0e1116; --panel:#161a20; --panel2:#1b2027; --ink:#e9edf2; --muted:#98a2af;
  --line:#252b34; --accent:#6fb3e0; --accent-soft:rgba(111,179,224,.12);
  --warn-bg:#2a2416; --warn-line:#6b5a24; --warn-ink:#e8d08a;
  --shadow:0 1px 2px rgba(0,0,0,.45), 0 6px 22px rgba(0,0,0,.35);
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
  font:15px/1.55 ui-sans-serif,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  -webkit-font-smoothing:antialiased}
.wrap{max-width:1200px;margin:0 auto;padding:30px 20px 72px}

header{margin-bottom:22px}
h1{font-size:24px;margin:0 0 5px;letter-spacing:-.015em;font-weight:680}
.tagline{color:var(--muted);font-size:14px;margin:0;max-width:66ch}
h2{font-size:12.5px;margin:0 0 14px;letter-spacing:.05em;text-transform:uppercase;
   color:var(--muted);font-weight:650}

.card{background:var(--panel);border:1px solid var(--line);border-radius:14px;
      padding:20px;box-shadow:var(--shadow)}
.grid{display:grid;grid-template-columns:minmax(0,1.1fr) minmax(0,1fr);gap:18px}
@media(max-width:920px){.grid{grid-template-columns:minmax(0,1fr)}}
section{margin-top:18px}

.tabs{display:inline-flex;background:var(--panel);border:1px solid var(--line);
      border-radius:11px;padding:4px;gap:4px;margin-bottom:18px;box-shadow:var(--shadow)}
.tabs button{background:none;border:0;border-radius:8px;padding:8px 18px;
  font:inherit;font-size:14px;font-weight:550;color:var(--muted);cursor:pointer}
.tabs button.on{background:var(--accent);color:#fff}
.tabs button:not(.on):hover{background:var(--accent-soft);color:var(--accent)}

.banner{background:var(--warn-bg);border:1px solid var(--warn-line);color:var(--warn-ink);
  border-radius:11px;padding:13px 16px;font-size:13px;line-height:1.5;margin-bottom:18px}
.banner b{font-weight:680}

.stepper{display:flex;align-items:center;gap:14px;flex-wrap:wrap}
.stepper button{background:var(--panel2);color:var(--ink);border:1px solid var(--line);
  border-radius:9px;padding:7px 15px;font:inherit;font-size:14px;cursor:pointer}
.stepper button:hover:not(:disabled){border-color:var(--accent);color:var(--accent)}
.stepper button:disabled{opacity:.35;cursor:default}
.stepper input[type=range]{flex:1;min-width:180px;accent-color:var(--accent)}
.weeklabel{font-variant-numeric:tabular-nums;font-weight:670;min-width:128px;font-size:15px}

.big{display:flex;align-items:baseline;gap:11px;flex-wrap:wrap}
.big .n{font-size:46px;font-weight:700;letter-spacing:-.025em;font-variant-numeric:tabular-nums;
  line-height:1}
.big .band{color:var(--muted);font-size:13.5px;font-variant-numeric:tabular-nums}
.actual{margin-top:11px;font-size:13.5px;color:var(--muted);padding-top:11px;
  border-top:1px solid var(--line)}
.actual b{color:var(--ink);font-variant-numeric:tabular-nums}
.hit{color:var(--green);font-weight:650}.miss{color:var(--orange);font-weight:650}

.legend{display:flex;gap:15px;flex-wrap:wrap;margin:15px 0 2px;font-size:12.5px}
.legend span{display:inline-flex;align-items:center;gap:6px;color:var(--muted)}
.dot{width:11px;height:11px;border-radius:3px;display:inline-block;flex:none}

svg.map{width:100%;height:auto;display:block}
svg.map path{stroke:var(--panel);stroke-width:1.1;cursor:pointer;transition:opacity .12s}
svg.map path:hover{opacity:.7}
svg.map path.hot{stroke:var(--ink);stroke-width:1.6}
svg.map path.sel{stroke:var(--ink);stroke-width:2.6}

svg.chart{width:100%;height:auto;display:block}
.chart .grid-line{stroke:var(--line);stroke-width:1}
.chart .band{fill:var(--accent);opacity:.15}
.chart .pred{fill:none;stroke:var(--accent);stroke-width:2.4}
.chart .act{fill:none;stroke:var(--ink);stroke-width:2;stroke-dasharray:5 3}
.chart text{fill:var(--muted);font-size:12px}
.chart .cursor{stroke:var(--ink);stroke-width:1.2;opacity:.4}

table{width:100%;border-collapse:collapse;font-size:13.5px}
th{text-align:left;font-weight:620;color:var(--muted);font-size:11px;text-transform:uppercase;
   letter-spacing:.03em;padding:7px;border-bottom:1px solid var(--line);white-space:nowrap}
td{padding:8px 7px;border-bottom:1px solid var(--line);font-variant-numeric:tabular-nums}
td.uc{font-variant-numeric:normal;min-width:118px;white-space:normal}
tbody tr:last-child td{border-bottom:0}
tr.row:hover{background:var(--accent-soft);cursor:pointer}
tr.on{background:var(--accent-soft)}
.pill{display:inline-block;padding:2px 9px;border-radius:999px;font-size:11px;
  font-weight:680;color:#fff;letter-spacing:.01em}
.tablewrap{overflow-x:auto}.scroll{max-height:430px;overflow-y:auto}

.note{font-size:12.5px;color:var(--muted);margin-top:13px;line-height:1.55}
.kv{display:grid;grid-template-columns:1fr auto;gap:6px 16px;font-size:13.5px}
.kv div:nth-child(even){font-variant-numeric:tabular-nums;font-weight:640}
.kv .sep{grid-column:1/-1;height:1px;background:var(--line);margin:7px 0}
.finding{border-left:3px solid var(--accent);padding-left:15px;margin:15px 0}
.finding b{display:block;margin-bottom:3px}
.hide{display:none}
footer{margin-top:26px;color:var(--muted);font-size:12px;text-align:center}
</style>

<div class="wrap">
<header>
  <h1>Rawalpindi Dengue Forecast</h1>
  <p class="tagline">Weekly dengue risk for the 87 Union Councils of Rawalpindi Tehsil —
  how many cases, in which neighbourhoods, and how confident.</p>
</header>

<div class="tabs">
  <button id="tab-live" class="on">Next week (live)</button>
  <button id="tab-replay">2024 season (validated)</button>
</div>

<div id="view-live">
  <div class="banner" id="livebanner"></div>
  <div class="grid">
    <div class="card">
      <h2>Where — alert level by Union Council</h2>
      <svg class="map" id="lmap" role="img" aria-label="Live alert map"></svg>
      <div class="legend" id="llegend"></div>
    </div>
    <div>
      <div class="card">
        <h2>How many — city total</h2>
        <div class="big"><span class="n" id="lcity">—</span><span class="band" id="lweek"></span></div>
        <div class="kv" style="margin-top:15px" id="lcounts"></div>
      </div>
      <div class="card" style="margin-top:18px">
        <h2>Highest risk next week</h2>
        <div class="tablewrap scroll"><table id="ltable">
          <thead><tr><th>Union Council</th><th>Expected</th><th>80% range</th><th>Alert</th></tr></thead>
          <tbody></tbody></table></div>
      </div>
    </div>
  </div>
  <section class="card">
    <h2>What this live forecast can and cannot tell you</h2>
    <div class="tablewrap"><table id="ltrust"></table></div>
    <p class="note" id="ltrustnote"></p>
  </section>
</div>

<div id="view-replay" class="hide">
  <div class="banner" id="replaybanner"></div>

  <div class="card">
    <h2>The whole season — forecast against what happened</h2>
    <svg class="chart" id="chart" role="img" aria-label="Season forecast versus actual"></svg>
    <div class="legend">
      <span><i class="dot" style="background:var(--accent)"></i>Forecast</span>
      <span><i class="dot" style="background:var(--accent);opacity:.3"></i>80% range</span>
      <span><i class="dot" style="background:var(--ink)"></i>Actual</span>
    </div>
  </div>

  <section class="card stepper">
    <button id="prev">&larr;</button>
    <span class="weeklabel" id="weeklabel"></span>
    <input type="range" id="slider" min="0" max="0" value="0">
    <button id="next">&rarr;</button>
  </section>

  <div class="grid">
    <div class="card">
      <h2>Where — alert level by Union Council</h2>
      <svg class="map" id="rmap" role="img" aria-label="Replay alert map"></svg>
      <div class="legend" id="rlegend"></div>
      <p class="note" id="mapnote"></p>
    </div>
    <div>
      <div class="card">
        <h2>How many — city total</h2>
        <div class="big"><span class="n" id="rcity">—</span><span class="band" id="rband"></span></div>
        <div class="actual" id="ractual"></div>
        <div class="kv" style="margin-top:15px" id="rcounts"></div>
      </div>
      <div class="card" style="margin-top:18px">
        <h2>Highest risk this week</h2>
        <div class="tablewrap scroll"><table id="rtable">
          <thead><tr><th>Union Council</th><th>Expected</th><th>80% range</th>
          <th>Last wk</th><th>Alert</th><th>Actual</th></tr></thead><tbody></tbody></table></div>
      </div>
    </div>
  </div>

  <section class="card">
    <h2>Does the alert colour mean anything?</h2>
    <p class="note" style="margin:0 0 13px">Every UC-week of the season, grouped by the
    colour the map showed <em>before</em> that week happened.</p>
    <div class="tablewrap"><table id="alerttable"></table></div>
  </section>

  <section class="grid">
    <div class="card">
      <h2>Measured uncertainty</h2>
      <div class="kv" id="coverage"></div>
      <p class="note">Coverage is the share of weeks the band actually contained the true
      count. A well-calibrated 80% band contains 80%. The all-UC-weeks rows over-cover
      because most UC-weeks are zero and an interval starting at zero cannot be narrowed;
      the rows restricted to UCs that reported cases are the meaningful ones.</p>
    </div>
    <div class="card"><h2>What the study found</h2><div id="findings"></div></div>
  </section>

  <section class="card">
    <h2>Model comparison — mean over held-out seasons</h2>
    <div class="tablewrap"><table id="comptable"></table></div>
    <p class="note">Lower MAE, RMSE and Poisson deviance are better; higher Top-10 hit is
    better. "Static share" is the rule the currently deployed alert app uses, and it is
    given the true city total for the week — information it would not have in practice.</p>
  </section>
</div>

<footer id="stamp"></footer>
</div>

<script>
const DATA = __DATA__, LIVE = __LIVE__, SHAPES = __SHAPES__, BOX = __BOX__;
const HEX = {Red:'#c8412f',Orange:'#e07b39',Yellow:'#e8b931',Watch:'#4b9cd3',Green:'#7fb069'};
const ORDER = ['Red','Orange','Yellow','Watch','Green'];
const $ = id => document.getElementById(id);
const fmt = n => n.toLocaleString('en-US');

let index = DATA.weeks.reduce((b,w,i,a)=>w.city_expected>a[b].city_expected?i:b,0);
let selected = null;
let view = 'live';

function paintMap(svgId, ucs) {
  const by = {}; ucs.forEach(u => by[u.uc] = u);
  const svg = $(svgId);
  svg.setAttribute('viewBox', `0 0 ${BOX.w} ${BOX.h}`);
  svg.innerHTML = Object.entries(SHAPES).map(([name, d]) => {
    const u = by[name];
    const fill = u ? HEX[u.alert] : '#9aa3ad';
    const hot = u && (u.alert === 'Red' || u.alert === 'Orange');
    const cls = name === selected ? 'sel' : (hot ? 'hot' : '');
    const t = u ? `${name} — ${u.expected} expected, ${u.alert}` +
      (u.actual != null ? `, actual ${u.actual}` : '') : name;
    return `<path class="${cls}" d="${d}" fill="${fill}" data-uc="${name}"><title>${t}</title></path>`;
  }).join('');
  svg.querySelectorAll('path').forEach(p => p.addEventListener('click', () => {
    selected = selected === p.dataset.uc ? null : p.dataset.uc; render();
  }));
}

function legend(id, counts) {
  $(id).innerHTML = ORDER.filter(k => counts[k]).map(k =>
    `<span><i class="dot" style="background:${HEX[k]}"></i>${k} · ${counts[k]}</span>`).join('');
}

function tally(ucs) { const c = {}; ucs.forEach(u => c[u.alert] = (c[u.alert]||0)+1); return c; }

function bindRows(tableId) {
  $(tableId).querySelectorAll('tr.row').forEach(tr => tr.addEventListener('click', () => {
    selected = selected === tr.dataset.uc ? null : tr.dataset.uc; render(); }));
}

/* ---------------- live ---------------- */
function renderLive() {
  if (!LIVE) {
    $('view-live').innerHTML = '<div class="card">Live forecast not generated yet. ' +
      'Run <code>python src/12_live_forecast.py</code>, then rebuild.</div>';
    return;
  }
  $('livebanner').innerHTML = `<b>Live forecast from real-time weather.</b> ${LIVE.caveat}`;
  $('lcity').textContent = fmt(Math.round(LIVE.city_expected));
  $('lweek').textContent =
    `cases expected · week ${LIVE.target.week}, starting ${LIVE.target.week_start}`;

  const counts = tally(LIVE.ucs);
  legend('llegend', counts);
  $('lcounts').innerHTML = ORDER.filter(k => counts[k]).map(k =>
    `<div><i class="dot" style="background:${HEX[k]}"></i> ${k} UCs</div><div>${counts[k]}</div>`)
    .join('') + `<div class="sep"></div><div>Weather source</div><div>${LIVE.weather_source}</div>`;

  $('ltable').querySelector('tbody').innerHTML = LIVE.ucs.slice(0, 25).map(u => `
    <tr class="row ${u.uc===selected?'on':''}" data-uc="${u.uc}">
      <td class="uc">${u.uc}</td><td>${u.expected.toFixed(1)}</td>
      <td>${u.low80}–${u.high80}</td>
      <td><span class="pill" style="background:${HEX[u.alert]}">${u.alert}</span></td></tr>`).join('');
  bindRows('ltable');
  paintMap('lmap', LIVE.ucs);

  const v = LIVE.engine_validation || [];
  const folds = [...new Set(v.map(r => r.fold))].sort();
  const byEngine = {};
  v.forEach(r => { (byEngine[r.engine] = byEngine[r.engine] || {})[r.fold] = r; });
  $('ltrust').innerHTML =
    `<thead><tr><th>Engine</th>${folds.map(f=>`<th>MAE ${f}</th>`).join('')}
     ${folds.map(f=>`<th>Top-10 hit ${f}</th>`).join('')}</tr></thead><tbody>` +
    Object.entries(byEngine).map(([name, rows]) => `<tr>
      <td class="uc">${name}</td>
      ${folds.map(f=>`<td>${rows[f]?rows[f].MAE.toFixed(3):'—'}</td>`).join('')}
      ${folds.map(f=>`<td>${rows[f]?(rows[f].Top10Hit*100).toFixed(0)+'%':'—'}</td>`).join('')}
      </tr>`).join('') + '</tbody>';
  $('ltrustnote').innerHTML =
    'The weather-only engine behind the map above has the same top-10 hit rate as a fixed ' +
    'historical share, in both held-out seasons. That is the honest limit: <b>weather ' +
    'predicts when the season turns, not which neighbourhood is affected.</b> Which UC is ' +
    'really at risk lives only in recent case counts — connect a weekly feed and the full ' +
    'engine (top row) takes over.';
}

/* ---------------- season chart ---------------- */
function drawChart() {
  const W=1000, H=270, L=48, R=14, T=16, B=30;
  const weeks = DATA.weeks;
  const max = Math.max(...weeks.map(w => Math.max(w.city_high80, w.city_actual ?? 0))) * 1.06;
  const x = i => L + i*(W-L-R)/(weeks.length-1);
  const y = v => T + (H-T-B)*(1 - v/max);

  const band = weeks.map((w,i)=>`${x(i)},${y(w.city_high80)}`).join(' ') + ' ' +
               weeks.map((w,i)=>`${x(i)},${y(w.city_low80)}`).reverse().join(' ');
  const line = k => weeks.map((w,i)=>`${x(i)},${y(w[k] ?? 0)}`).join(' ');

  const ticks = [0,.25,.5,.75,1].map(f => {
    const v = max*f;
    return `<line class="grid-line" x1="${L}" y1="${y(v)}" x2="${W-R}" y2="${y(v)}"/>
            <text x="${L-9}" y="${y(v)+4}" text-anchor="end">${Math.round(v)}</text>`;
  }).join('');
  const labels = weeks.map((w,i)=> i%4===0
    ? `<text x="${x(i)}" y="${H-8}" text-anchor="middle">w${w.week}</text>` : '').join('');

  $('chart').setAttribute('viewBox', `0 0 ${W} ${H}`);
  $('chart').innerHTML = `${ticks}
    <polygon class="band" points="${band}"/>
    <polyline class="act" points="${line('city_actual')}"/>
    <polyline class="pred" points="${line('city_expected')}"/>
    <line class="cursor" id="cursor" x1="0" y1="${T}" x2="0" y2="${H-B}"/>
    ${labels}`;
  moveCursor();
}
function moveCursor() {
  const c = $('cursor'); if (!c) return;
  const px = 48 + index*(1000-48-14)/(DATA.weeks.length-1);
  c.setAttribute('x1', px); c.setAttribute('x2', px);
}

/* ---------------- replay ---------------- */
function renderReplay() {
  const w = DATA.weeks[index];
  $('replaybanner').innerHTML = `<b>Validated replay.</b> ${DATA.mode_note}`;
  $('weeklabel').textContent = `${w.year} · week ${w.week}`;
  $('slider').value = index;
  $('prev').disabled = index === 0;
  $('next').disabled = index === DATA.weeks.length-1;
  $('rcity').textContent = fmt(Math.round(w.city_expected));
  $('rband').textContent = `cases forecast · 80% range ${fmt(w.city_low80)}–${fmt(w.city_high80)}`;

  if (w.city_actual != null) {
    const hit = w.city_actual >= w.city_low80 && w.city_actual <= w.city_high80;
    $('ractual').innerHTML = `Actually happened: <b>${fmt(w.city_actual)}</b> cases · ` +
      `<span class="${hit?'hit':'miss'}">${hit?'inside the band':'outside the band'}</span>`;
  } else { $('ractual').textContent = 'No observed count for this week.'; }

  const counts = w.alert_counts || {};
  legend('rlegend', counts);
  $('rcounts').innerHTML = ORDER.filter(k => counts[k]).map(k =>
    `<div><i class="dot" style="background:${HEX[k]}"></i> ${k} UCs</div><div>${counts[k]}</div>`).join('');
  $('mapnote').textContent = selected
    ? `Selected: ${selected}. Click again to clear.`
    : 'Click a Union Council to highlight it. Hover for its numbers.';

  $('rtable').querySelector('tbody').innerHTML = w.ucs
    .filter(u => u.expected >= 0.2 || u.alert !== 'Green' || u.uc === selected)
    .slice(0,40).map(u => `
    <tr class="row ${u.uc===selected?'on':''}" data-uc="${u.uc}">
      <td class="uc">${u.uc}${u.onset_probability != null
        ? `<br><span style="color:var(--muted);font-size:11.5px">start risk ${Math.round(u.onset_probability*100)}%</span>`
        : ''}</td>
      <td>${u.expected.toFixed(1)}</td><td>${u.low80}–${u.high80}</td><td>${u.last_week}</td>
      <td><span class="pill" style="background:${HEX[u.alert]}">${u.alert}</span></td>
      <td>${u.actual == null ? '—' : u.actual}</td></tr>`).join('');
  bindRows('rtable');
  paintMap('rmap', w.ucs);
  moveCursor();
}

/* ---------------- evidence ---------------- */
function renderEvidence() {
  const ev = DATA.replay_evaluation;
  $('alerttable').innerHTML =
    `<thead><tr><th>Alert</th><th>UC-weeks</th><th>% of map</th><th>Mean actual</th>
     <th>% with any case</th><th>% with 5+</th><th>% of all cases caught</th></tr></thead><tbody>` +
    ev.alerts.map(a => `<tr>
      <td><span class="pill" style="background:${HEX[a.alert]}">${a.alert}</span></td>
      <td>${fmt(a.UC_weeks_fired)}</td><td>${a.share_of_all}%</td><td>${a.mean_actual_cases}</td>
      <td>${a.pct_with_any_case}%</td><td>${a.pct_with_5plus}%</td>
      <td><b>${a.pct_of_all_cases}%</b></td></tr>`).join('') + '</tbody>';

  $('coverage').innerHTML = ev.coverage.map(c =>
    `<div>${c.scope} · nominal ${c.nominal}</div><div>${(c.PICP*100).toFixed(1)}%</div>`).join('') +
    `<div class="sep"></div><div>City total · nominal 80%</div><div>${(ev.city.PICP80*100).toFixed(0)}%</div>` +
    `<div>City MAE (cases/week)</div><div>${ev.city.MAE}</div>` +
    `<div>City error at peak season</div><div>${ev.city.MAPE_active}%</div>` +
    `<div>Top-10 UC hit rate</div><div>${(ev.top10_hit*100).toFixed(0)}%</div>`;

  const d = DATA.distance_decay, o = DATA.onset_spatial_test;
  const dA = o.map(r => (r.dAUC>0?'+':'') + r.dAUC.toFixed(3)).join(' and ');
  const knox = DATA.knox ? DATA.knox.pooled.filter(k => k.time_days === 7) : [];
  const at = m => { const k = knox.find(x => x.space_m === m); return k ? k.mean_ratio.toFixed(2) : '?'; };
  const mh = DATA.microhotspot;
  $('findings').innerHTML = `
    <div class="finding"><b>Neighbouring UCs add nothing.</b>
      Adding an adjacency graph changed onset AUC by ${dA} across two held-out seasons —
      opposite signs, neither significant. At UC scale, correlation does not even fall with
      distance (${d[0].mean_corr} at 0–2&nbsp;km vs ${d[d.length-1].mean_corr} at 20&nbsp;km+).</div>
    <div class="finding"><b>But that is a resolution artefact.</b>
      On household coordinates, cases close in space are close in time far more often than
      chance, decaying sharply: ${at(100)}× excess at 100&nbsp;m, ${at(200)}× at 200&nbsp;m,
      ${at(2000)}× at 2&nbsp;km (all p=0.002). Local transmission is real at 100–200&nbsp;m —
      Union Councils are kilometres wide, so aggregating averages it away.</div>
    ${mh ? `<div class="finding"><b>Rings beat the map per km².</b>
      A ${mh.by_radius[1].radius_m}&nbsp;m ring around every case reported this week catches
      ${mh.by_radius[1].pct_caught}% of next week's over ${mh.by_radius[1].mean_area_km2}&nbsp;km²
      — five times more area-efficient than Red+Orange UCs, with no model at all.</div>` : ''}
    <div class="finding"><b>Reporting is fast.</b>
      Median ${DATA.reporting_delay.median_days} days from onset to confirmation;
      ${DATA.reporting_delay.within_7_days_pct}% within a week (n=${fmt(DATA.reporting_delay.n)}).</div>
    <div class="finding"><b>Onset warning is partial.</b>
      Of ${ev.onset.first_activations} UCs that first reported during the season,
      ${ev.onset.flagged_week_before} were flagged the week before
      (${(ev.onset.recall*100).toFixed(0)}%).</div>`;

  $('comptable').innerHTML =
    `<thead><tr><th>Model</th><th>MAE</th><th>RMSE</th><th>Poisson deviance</th>
     <th>Top-10 hit</th></tr></thead><tbody>` +
    DATA.model_comparison.map(m => `<tr><td class="uc">${m.model}</td>
      <td>${m.MAE.toFixed(3)}</td><td>${m.RMSE.toFixed(3)}</td>
      <td>${m.PoissonDev.toFixed(3)}</td><td>${(m.Top10Hit*100).toFixed(0)}%</td></tr>`).join('') +
    '</tbody>';

  $('stamp').textContent = `Generated ${DATA.generated_at.replace('T',' ')} · ` +
    `${DATA.weeks.length} replayed weeks · 87 Union Councils · Rawalpindi Tehsil`;
}

function render() { view === 'live' ? renderLive() : renderReplay(); }
function setView(v) {
  view = v; selected = null;
  $('tab-live').classList.toggle('on', v === 'live');
  $('tab-replay').classList.toggle('on', v === 'replay');
  $('view-live').classList.toggle('hide', v !== 'live');
  $('view-replay').classList.toggle('hide', v !== 'replay');
  if (v === 'replay') drawChart();
  render();
}
$('tab-live').onclick = () => setView('live');
$('tab-replay').onclick = () => setView('replay');
$('slider').max = DATA.weeks.length - 1;
$('slider').oninput = e => { index = +e.target.value; renderReplay(); };
$('prev').onclick = () => { if (index > 0) { index--; renderReplay(); } };
$('next').onclick = () => { if (index < DATA.weeks.length-1) { index++; renderReplay(); } };
document.addEventListener('keydown', e => {
  if (view !== 'replay') return;
  if (e.key === 'ArrowLeft' && index > 0) { index--; renderReplay(); }
  if (e.key === 'ArrowRight' && index < DATA.weeks.length-1) { index++; renderReplay(); }
});

renderEvidence();
setView('live');
</script>
"""


def main() -> None:
    payload = json.loads((APP_DATA / "forecasts.json").read_text())
    boundaries = json.loads((APP_DATA / "uc_boundaries.geojson").read_text())
    live_path = APP_DATA / "live_forecast.json"
    live = json.loads(live_path.read_text()) if live_path.exists() else None

    shapes, box = project(boundaries)
    html = (HTML
            .replace("__DATA__", json.dumps(payload, separators=(",", ":")))
            .replace("__LIVE__", json.dumps(live, separators=(",", ":")) if live else "null")
            .replace("__SHAPES__", json.dumps(shapes, separators=(",", ":")))
            .replace("__BOX__", json.dumps(box)))

    # Two copies on purpose: the repo root is what GitHub Pages serves, and
    # app/ keeps the page next to the data it was built from for local use.
    for out in (ROOT / "index.html", APP / "index.html"):
        out.write_text(html, encoding="utf-8")

    size = (ROOT / "index.html").stat().st_size / 1e6
    print(f"wrote index.html at repo root and in app/  "
          f"({size:.2f} MB, {len(shapes)} UC shapes, live={'yes' if live else 'no'})")


if __name__ == "__main__":
    main()
