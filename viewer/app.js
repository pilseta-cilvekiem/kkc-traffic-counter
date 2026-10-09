// Public viewer for `public.crossings` (see supabase/schema.sql).
//
// The publishable key is meant to be public: row-level security lets it read crossings and
// nothing else. PostgREST aggregates are off on this project, so the page fetches the raw
// rows for the chosen interval (paged) and buckets them here, in Riga wall-clock time.

"use strict";

const SUPABASE_URL = "https://koqvsbumzhrsnstmurol.supabase.co";
const SUPABASE_KEY = "sb_publishable_k3G34JPZk_8OfAWqSM6LGA_zybPtqu2";
const TABLE = "crossings";
const TZ = "Europe/Riga";
const PAGE = 1000;              // PostgREST's default max rows per request
const MAX_BUCKETS = 2000;       // beyond this a finer split is refused, not drawn

// The board's class names, grouped into what a reader cares about. `rider(person)` is a
// person moving at riding pace with no bicycle detected under them: mostly scooter riders.
const GROUPS = [
  { key: "bicycles", label: "Bicycles", classes: ["bicycle"] },
  { key: "riders", label: "Scooters & riders", classes: ["rider(person)"] },
  { key: "pedestrians", label: "Pedestrians", classes: ["pedestrian"] },
  { key: "cars", label: "Cars", classes: ["car"] },
];
const OTHER = { key: "other", label: "Other", classes: [] };
const DIRS = [
  { key: "in", label: "From center" },
  { key: "out", label: "To center" },
];
const DIR_LABEL = Object.fromEntries(DIRS.map((d) => [d.key, d.label]));
const CLASS_GROUP = {};
for (const g of GROUPS) for (const c of g.classes) CLASS_GROUP[c] = g.key;
const groupOf = (cls) => CLASS_GROUP[cls] || OTHER.key;

const $ = (id) => document.getElementById(id);
const state = { rows: [], seenOther: false, chart: null, loadSeq: 0 };

// --- time zone arithmetic -----------------------------------------------------------------
//
// "Wall" times are Riga calendar times encoded as UTC milliseconds (Date.UTC of the local
// parts), so bucketing is plain UTC calendar arithmetic. Only converting to and from real
// instants needs the zone.

const partsFmt = new Intl.DateTimeFormat("en-GB", {
  timeZone: TZ, hourCycle: "h23",
  year: "numeric", month: "2-digit", day: "2-digit",
  hour: "2-digit", minute: "2-digit", second: "2-digit",
});

function instantToWall(ms) {
  const p = {};
  for (const { type, value } of partsFmt.formatToParts(new Date(ms))) p[type] = +value;
  return Date.UTC(p.year, p.month - 1, p.day, p.hour, p.minute, p.second);
}

function wallToInstant(wall) {
  // Riga's offset at the guessed instant, refined once for the DST edge.
  let t = wall - (instantToWall(wall) - wall);
  t = wall - (instantToWall(t) - t);
  return t;
}

const DAY = 86400e3;
const isoDate = (wall) => new Date(wall).toISOString().slice(0, 10);
const parseDate = (s) => { const [y, m, d] = s.split("-").map(Number); return Date.UTC(y, m - 1, d); };
const todayWall = () => { const w = instantToWall(Date.now()); return w - (w % DAY); };

function floorBucket(wall, unit) {
  const d = new Date(wall);
  switch (unit) {
    case "hour": return wall - (wall % 3600e3);
    case "day": return wall - (wall % DAY);
    case "week": { const day = wall - (wall % DAY); return day - ((d.getUTCDay() + 6) % 7) * DAY; }
    case "month": return Date.UTC(d.getUTCFullYear(), d.getUTCMonth(), 1);
  }
}

function nextBucket(wall, unit) {
  const d = new Date(wall);
  switch (unit) {
    case "hour": return wall + 3600e3;
    case "day": return wall + DAY;
    case "week": return wall + 7 * DAY;
    case "month": return Date.UTC(d.getUTCFullYear(), d.getUTCMonth() + 1, 1);
  }
}

function isoWeek(wall) {
  const d = new Date(wall);
  const thu = wall + (3 - ((d.getUTCDay() + 6) % 7)) * DAY;
  const year = new Date(thu).getUTCFullYear();
  return [year, 1 + Math.floor((thu - Date.UTC(year, 0, 1)) / (7 * DAY))];
}

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
function bucketLabel(wall, unit) {
  const d = new Date(wall);
  const dd = String(d.getUTCDate()).padStart(2, "0");
  const mon = MONTHS[d.getUTCMonth()];
  switch (unit) {
    case "hour": return `${dd} ${mon} ${String(d.getUTCHours()).padStart(2, "0")}:00`;
    case "day": return `${dd} ${mon} ${d.getUTCFullYear()}`;
    case "week": { const [y, w] = isoWeek(wall); return `${y} W${String(w).padStart(2, "0")} (from ${dd} ${mon})`; }
    case "month": return `${mon} ${d.getUTCFullYear()}`;
  }
}

function bucketKeyForCsv(wall, unit) {
  const s = new Date(wall).toISOString();
  switch (unit) {
    case "hour": return s.slice(0, 13).replace("T", " ") + ":00";
    case "day": return s.slice(0, 10);
    case "week": { const [y, w] = isoWeek(wall); return `${y}-W${String(w).padStart(2, "0")}`; }
    case "month": return s.slice(0, 7);
  }
}

// --- filters ------------------------------------------------------------------------------

// Each preset: its wall-date range [from, to] inclusive, and the split it opens with.
function presetRange(preset) {
  const t = todayWall();
  const d = new Date(t);
  const y = d.getUTCFullYear(), m = d.getUTCMonth();
  switch (preset) {
    case "today": return [t, t, "hour"];
    case "yesterday": return [t - DAY, t - DAY, "hour"];
    case "7d": return [t - 6 * DAY, t, "day"];
    case "30d": return [t - 29 * DAY, t, "day"];
    case "this-month": return [Date.UTC(y, m, 1), t, "day"];
    case "last-month": return [Date.UTC(y, m - 1, 1), Date.UTC(y, m, 0), "day"];
    case "90d": return [t - 89 * DAY, t, "week"];
    case "this-year": return [Date.UTC(y, 0, 1), t, "month"];
  }
  return null;
}

function readFilters() {
  const from = $("from").value, to = $("to").value;
  return {
    preset: $("preset").value,
    from, to,
    bucket: $("bucket").value,
    stack: $("stack").value,
    groups: [...document.querySelectorAll("input[name=group]:checked")].map((i) => i.value),
    dirs: [...document.querySelectorAll("input[name=dir]:checked")].map((i) => i.value),
  };
}

// Filters live in the URL hash so a view can be linked to.
function writeHash(f) {
  const p = new URLSearchParams();
  p.set("range", f.preset);
  if (f.preset === "custom") { p.set("from", f.from); p.set("to", f.to); }
  p.set("by", f.bucket);
  if (f.stack !== "group") p.set("chart", f.stack);
  if (f.groups.length !== allGroups().length) p.set("types", f.groups.join(","));
  if (f.dirs.length !== DIRS.length) p.set("dir", f.dirs.join(","));
  history.replaceState(null, "", "#" + p.toString());
}

function readHash() {
  const p = new URLSearchParams(location.hash.slice(1));
  const preset = p.get("range");
  if (preset && [...$("preset").options].some((o) => o.value === preset)) $("preset").value = preset;
  if ($("preset").value === "custom") {
    if (/^\d{4}-\d\d-\d\d$/.test(p.get("from") || "")) $("from").value = p.get("from");
    if (/^\d{4}-\d\d-\d\d$/.test(p.get("to") || "")) $("to").value = p.get("to");
  } else {
    applyPreset(false);
  }
  if (["hour", "day", "week", "month"].includes(p.get("by"))) $("bucket").value = p.get("by");
  if (p.get("chart") === "direction") $("stack").value = "direction";
  if (p.has("types")) {
    const on = new Set(p.get("types").split(","));
    document.querySelectorAll("input[name=group]").forEach((i) => (i.checked = on.has(i.value)));
  }
  if (p.has("dir")) {
    const on = new Set(p.get("dir").split(","));
    document.querySelectorAll("input[name=dir]").forEach((i) => (i.checked = on.has(i.value)));
  }
}

function applyPreset(setBucket = true) {
  const r = presetRange($("preset").value);
  if (!r) return;
  $("from").value = isoDate(r[0]);
  $("to").value = isoDate(r[1]);
  if (setBucket) $("bucket").value = r[2];
}

const allGroups = () => (state.seenOther ? [...GROUPS, OTHER] : GROUPS);

function renderGroupToggles() {
  const prev = new Map([...document.querySelectorAll("input[name=group]")].map((i) => [i.value, i.checked]));
  $("group-toggles").innerHTML = allGroups().map((g) => `
    <label><input type="checkbox" name="group" value="${g.key}" ${prev.get(g.key) === false ? "" : "checked"}>
    <span class="swatch" style="background:var(--${g.key})"></span>${g.label}</label>`).join("");
}

// --- data ---------------------------------------------------------------------------------

async function fetchRows(fromInstant, toInstant, onProgress) {
  const rows = [];
  const base = `${SUPABASE_URL}/rest/v1/${TABLE}?select=event_id,occurred_at,class_name,direction`
    + `&occurred_at=gte.${encodeURIComponent(new Date(fromInstant).toISOString())}`
    + `&occurred_at=lt.${encodeURIComponent(new Date(toInstant).toISOString())}`
    + `&order=occurred_at.asc,event_id.asc`;
  for (let offset = 0; ; offset += PAGE) {
    const res = await fetch(`${base}&limit=${PAGE}&offset=${offset}`, {
      headers: { apikey: SUPABASE_KEY, Authorization: `Bearer ${SUPABASE_KEY}` },
    });
    if (!res.ok) throw new Error(`Supabase ${res.status}: ${(await res.text()).slice(0, 200)}`);
    const page = await res.json();
    rows.push(...page);
    onProgress(rows.length);
    if (page.length < PAGE) return rows;
  }
}

function aggregate(rows, f, startWall, endWall) {
  const buckets = [];
  for (let w = floorBucket(startWall, f.bucket); w < endWall; w = nextBucket(w, f.bucket)) {
    buckets.push(w);
    if (buckets.length > MAX_BUCKETS) return null;
  }
  const index = new Map(buckets.map((w, i) => [w, i]));
  const groups = new Set(f.groups), dirs = new Set(f.dirs);
  // counts[groupKey][dir][bucketIndex]
  const counts = {};
  for (const g of allGroups()) counts[g.key] = { in: new Array(buckets.length).fill(0), out: new Array(buckets.length).fill(0) };
  for (const r of rows) {
    const g = groupOf(r.class_name);
    if (!groups.has(g) || !dirs.has(r.direction)) continue;
    const i = index.get(floorBucket(instantToWall(Date.parse(r.occurred_at)), f.bucket));
    if (i !== undefined) counts[g][r.direction][i]++;
  }
  return { buckets, counts };
}

// --- rendering ----------------------------------------------------------------------------

const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const fmt = (n) => n.toLocaleString("en-US").replace(/,/g, " ");
const sum = (a) => a.reduce((s, v) => s + v, 0);

function renderCards(agg, f) {
  const shown = allGroups().filter((g) => f.groups.includes(g.key));
  $("cards").innerHTML = shown.map((g) => {
    const tin = sum(agg.counts[g.key].in), tout = sum(agg.counts[g.key].out);
    return `<div class="card" style="--c:var(--${g.key})">
      <div class="name">${g.label}</div>
      <div class="total">${fmt(tin + tout)}</div>
      <div class="split">
        ${f.dirs.includes("in") ? `<span>From center <b>${fmt(tin)}</b></span>` : ""}
        ${f.dirs.includes("out") ? `<span>To center <b>${fmt(tout)}</b></span>` : ""}
      </div></div>`;
  }).join("");
}

function renderChart(agg, f) {
  const labels = agg.buckets.map((w) => bucketLabel(w, f.bucket));
  const shown = allGroups().filter((g) => f.groups.includes(g.key));
  let datasets;
  if (f.stack === "direction") {
    datasets = DIRS.filter((d) => f.dirs.includes(d.key)).map((d) => ({
      label: d.label,
      data: agg.buckets.map((_, i) => sum(shown.map((g) => agg.counts[g.key][d.key][i]))),
      backgroundColor: cssVar(`--dir-${d.key}`),
    }));
  } else {
    datasets = shown.map((g) => ({
      label: g.label,
      data: agg.buckets.map((_, i) => sum(f.dirs.map((d) => agg.counts[g.key][d][i]))),
      backgroundColor: cssVar(`--${g.key}`),
    }));
  }
  const text = cssVar("--muted"), grid = cssVar("--border");
  const dirNote = f.dirs.length === DIRS.length ? "both directions" : DIR_LABEL[f.dirs[0]] || "no direction";
  $("chart-title").textContent = `Crossings per ${f.bucket}, ${dirNote}`;
  if (state.chart) state.chart.destroy();
  state.chart = new Chart($("chart"), {
    type: "bar",
    data: { labels, datasets },
    options: {
      maintainAspectRatio: false,
      animation: false,
      interaction: { mode: "index", intersect: false },
      scales: {
        x: { stacked: true, ticks: { color: text, maxRotation: 0, autoSkip: true, autoSkipPadding: 12 }, grid: { display: false } },
        y: { stacked: true, beginAtZero: true, ticks: { color: text, precision: 0 }, grid: { color: grid } },
      },
      plugins: {
        legend: { labels: { color: text, boxWidth: 12 } },
        tooltip: { callbacks: { footer: (items) => `Total: ${fmt(sum(items.map((i) => i.parsed.y)))}` } },
      },
    },
  });
}

// One column pair per type: From center / To center, then a grand total.
function tableModel(agg, f) {
  const shown = allGroups().filter((g) => f.groups.includes(g.key));
  const dirs = DIRS.filter((d) => f.dirs.includes(d.key));
  const cols = [];
  for (const g of shown) for (const d of dirs) cols.push({ g, d, data: agg.counts[g.key][d.key] });
  const rows = agg.buckets.map((w, i) => {
    const vals = cols.map((c) => c.data[i]);
    return { wall: w, vals, total: sum(vals) };
  });
  const totals = cols.map((c) => sum(c.data));
  return { shown, dirs, cols, rows, totals, grand: sum(totals) };
}

function renderTable(agg, f) {
  const m = tableModel(agg, f);
  const cell = (v) => `<td class="${v ? "" : "zero"}">${fmt(v)}</td>`;
  $("table").innerHTML = `
    <thead>
      <tr><th></th>${m.shown.map((g) => `<th colspan="${m.dirs.length}">${g.label}</th>`).join("")}<th></th></tr>
      <tr><th>${f.bucket[0].toUpperCase() + f.bucket.slice(1)}</th>${m.cols.map((c) => `<th>${c.d.label}</th>`).join("")}<th>Total</th></tr>
    </thead>
    <tbody>${m.rows.map((r) => `<tr><td>${bucketLabel(r.wall, f.bucket)}</td>${r.vals.map(cell).join("")}${cell(r.total)}</tr>`).join("")}</tbody>
    <tfoot><tr><td>Total</td>${m.totals.map(cell).join("")}${cell(m.grand)}</tr></tfoot>`;
}

// --- downloads ----------------------------------------------------------------------------

function csvEscape(v) {
  const s = String(v);
  return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

function download(name, lines) {
  const blob = new Blob(["﻿" + lines.map((l) => l.map(csvEscape).join(",")).join("\r\n") + "\r\n"], { type: "text/csv;charset=utf-8" });
  const a = Object.assign(document.createElement("a"), { href: URL.createObjectURL(blob), download: name });
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

const localStamp = (iso) => new Date(instantToWall(Date.parse(iso))).toISOString().slice(0, 19).replace("T", " ");

// Raw events honour the interval and the type/direction filters, so what is downloaded is
// what the page shows. The board's own class name is kept beside the group.
function downloadRaw() {
  const f = readFilters();
  const groups = new Set(f.groups), dirs = new Set(f.dirs);
  const lines = [["event_id", "occurred_at_utc", "occurred_at_riga", "class_name", "type", "direction", "direction_label"]];
  for (const r of state.rows) {
    const g = groupOf(r.class_name);
    if (!groups.has(g) || !dirs.has(r.direction)) continue;
    lines.push([r.event_id, r.occurred_at, localStamp(r.occurred_at), r.class_name,
      allGroups().find((x) => x.key === g).label, r.direction, DIR_LABEL[r.direction] || r.direction]);
  }
  download(`crossings_raw_${f.from}_${f.to}.csv`, lines);
}

function downloadTable() {
  const f = readFilters();
  if (!state.agg) return;
  const m = tableModel(state.agg, f);
  const lines = [[f.bucket, ...m.cols.map((c) => `${c.g.label} - ${c.d.label}`), "Total"]];
  for (const r of m.rows) lines.push([bucketKeyForCsv(r.wall, f.bucket), ...r.vals, r.total]);
  lines.push(["Total", ...m.totals, m.grand]);
  download(`crossings_by_${f.bucket}_${f.from}_${f.to}.csv`, lines);
}

// --- main ---------------------------------------------------------------------------------

function setStatus(msg, error = false) {
  $("status").textContent = msg;
  $("status").classList.toggle("error", error);
}

// Re-fetch only when the interval changes; every other filter re-draws from memory.
let loadedRange = null;

async function refresh() {
  const f = readFilters();
  writeHash(f);
  if (!f.from || !f.to || f.from > f.to) { setStatus("Pick a start date on or before the end date.", true); return; }

  const startWall = parseDate(f.from), endWall = parseDate(f.to) + DAY;
  const key = `${f.from}/${f.to}`;
  if (loadedRange !== key) {
    const seq = ++state.loadSeq;
    $("dl-raw").disabled = $("dl-table").disabled = true;
    setStatus("Loading…");
    try {
      const rows = await fetchRows(wallToInstant(startWall), wallToInstant(endWall),
        (n) => seq === state.loadSeq && setStatus(`Loading… ${fmt(n)} events`));
      if (seq !== state.loadSeq) return; // a newer interval was picked meanwhile
      state.rows = rows;
      loadedRange = key;
      const other = rows.some((r) => groupOf(r.class_name) === OTHER.key);
      if (other !== state.seenOther) { state.seenOther = other; renderGroupToggles(); }
    } catch (e) {
      if (seq === state.loadSeq) setStatus(`Could not load data: ${e.message}`, true);
      return;
    }
  }

  const agg = aggregate(state.rows, readFilters(), startWall, endWall);
  if (!agg) {
    setStatus(`Too many ${f.bucket}s in this interval — pick a coarser split or a shorter interval.`, true);
    return;
  }
  state.agg = agg;
  const ff = readFilters();
  renderCards(agg, ff);
  renderChart(agg, ff);
  renderTable(agg, ff);
  $("dl-raw").disabled = $("dl-table").disabled = false;
  const last = state.rows.length ? `, last at ${localStamp(state.rows[state.rows.length - 1].occurred_at)}` : "";
  setStatus(`${fmt(state.rows.length)} events in ${f.from === f.to ? f.from : `${f.from} … ${f.to}`}${last}.`);
}

function init() {
  renderGroupToggles();
  $("preset").value = "7d";
  applyPreset();
  readHash();

  $("preset").addEventListener("change", () => { applyPreset(); refresh(); });
  for (const id of ["from", "to"]) $(id).addEventListener("change", () => { $("preset").value = "custom"; refresh(); });
  for (const id of ["bucket", "stack"]) $(id).addEventListener("change", refresh);
  document.querySelector(".controls").addEventListener("change", (e) => {
    if (e.target.name === "group" || e.target.name === "dir") refresh();
  });
  $("dl-raw").addEventListener("click", downloadRaw);
  $("dl-table").addEventListener("click", downloadTable);
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => state.agg && renderChart(state.agg, readFilters()));
  refresh();
}

init();
