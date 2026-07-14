// script.js — dashboard wiring. Talks to the FastAPI backend defined in app.py.
// Every number on this page comes from a live API call -- nothing here is hand-typed data.

const state = {
  stats: null,
  events: { total: 0, limit: 50, offset: 0, events: [] },
  monthOptions: [],
};

const EXAMPLE_REASONS = [
  "Bundling area photocell 1,2 & 3 not working",
  "BDM HMD cleaning",
  "OD2 continuously active after 1st pass, exit HMD fault",
  "Encoder position feedback fault, billet stuck",
  "LVDT transducer reading drift on stand 3",
];

async function apiGet(path, params = {}) {
  const url = new URL(API_BASE_URL + path, window.location.href);
  Object.entries(params).forEach(([k, v]) => { if (v !== "" && v != null) url.searchParams.set(k, v); });
  const res = await fetch(url);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `Request failed (${res.status})`);
  }
  return res.json();
}

async function apiPost(path, payload) {
  const res = await fetch(API_BASE_URL + path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `Request failed (${res.status})`);
  }
  return res.json();
}

function fmtMin(n) {
  if (n == null || isNaN(n)) return "—";
  return Math.round(n).toLocaleString() + " min";
}

// ------------------------------------------------------------- Loaders --
async function loadStats() {
  const data = await apiGet("/stats");
  state.stats = data;
  renderKpis(data);
  renderCoverage(data);
  renderParetoBars(data);
  renderSeverityTable(data);
  renderAlerts(data);
  populateDeviceFilter(data);

  document.getElementById("brandSub").innerHTML =
    `${data.months_covered[0] || ""} – ${data.months_covered[data.months_covered.length - 1] || ""}` +
    `<span class="sep">&middot;</span>${data.total_events} total events logged`;
}

async function loadModelInfo() {
  try {
    const info = await apiGet("/model_info");
    document.getElementById("modelInfoFooter").textContent =
      `MODEL: ${info.winner_model.toUpperCase()} · HOLDOUT ACCURACY: ${(info.holdout_accuracy * 100).toFixed(1)}% ` +
      `(${info.holdout_month}) · CLASSES: ${info.n_classes} · API: ${API_BASE_URL || "same-origin"}`;
  } catch (e) {
    document.getElementById("modelInfoFooter").textContent = "MODEL: not trained yet — run `python -m src.train`";
  }
}

async function loadTrend() {
  const promises = Object.keys(DEVICE_COLORS).slice(0, 5).map(() => null); // placeholder, real call below
  const all = await apiGet("/forecast/all_devices");
  renderTrendChart(all);
  renderForecastSummary(all);
}

async function loadEvents() {
  const search = document.getElementById("logSearch").value;
  const device = document.getElementById("logDeviceFilter").value;
  const month = document.getElementById("logMonthFilter").value;
  const data = await apiGet("/events", {
    search, device, month,
    limit: state.events.limit,
    offset: state.events.offset,
  });
  state.events = { ...state.events, ...data };
  renderEventLog();
}

// ------------------------------------------------------------- Renderers --
function renderKpis(d) {
  document.getElementById("kpiTotalMinutes").textContent = fmtMin(d.total_field_device_delay_minutes);
  document.getElementById("kpiTotalMinutesFoot").textContent =
    `${d.months_covered.length} months · ${d.months_covered[0]}–${d.months_covered[d.months_covered.length-1]}`;

  document.getElementById("kpiTaggedEvents").textContent = d.total_tagged_events.toLocaleString();
  document.getElementById("kpiTaggedEventsFoot").textContent = `${d.unresolved_events} events unresolved`;

  if (d.top_contributor_volume) {
    document.getElementById("kpiTopVolume").textContent = formatDevice(d.top_contributor_volume.device);
    document.getElementById("kpiTopVolumeFoot").textContent =
      `${fmtMin(d.top_contributor_volume.total_minutes)} · ${d.top_contributor_volume.events} events`;
  }
  if (d.top_contributor_severity) {
    document.getElementById("kpiTopSeverity").textContent = formatDevice(d.top_contributor_severity.device);
    document.getElementById("kpiTopSeverityFoot").textContent =
      `${d.top_contributor_severity.avg_minutes.toFixed(1)} min avg per incident`;
  }
}

function renderCoverage(d) {
  document.getElementById("coveragePct").textContent = `${d.coverage_pct}%`;
  document.getElementById("coverageBarFill").style.width = `${d.coverage_pct}%`;
}

function renderParetoBars(d) {
  const container = document.getElementById("paretoBars");
  const rows = [...d.pareto_by_device].sort((a, b) => b.total_minutes - a.total_minutes);
  const max = rows.length ? rows[0].total_minutes : 1;
  container.innerHTML = rows.map(r => `
    <div class="bar-row">
      <span class="name">${formatDevice(r.device)}</span>
      <div class="bar-track"><div class="bar-fill" style="width:${(r.total_minutes / max * 100).toFixed(1)}%; background:${deviceColor(r.device)}"></div></div>
      <span class="val">${Math.round(r.total_minutes)}</span>
    </div>
  `).join("");
}

function renderSeverityTable(d) {
  const rows = [...d.pareto_by_device].sort((a, b) => b.avg_minutes - a.avg_minutes);
  document.getElementById("sevTableBody").innerHTML = rows.map(r => `
    <tr>
      <td><span class="device-chip"><span class="device-dot" style="background:${deviceColor(r.device)}"></span>${formatDevice(r.device)}</span></td>
      <td class="num">${r.avg_minutes.toFixed(1)}</td>
      <td class="num">${Math.round(r.max_minutes)} min</td>
      <td class="num">${r.events}</td>
      <td><span class="risk-badge" style="background:${riskColor(r.risk_level)}22; color:${riskColor(r.risk_level)};">${r.risk_level}${r.is_estimated ? " · no sheet" : ""}</span></td>
    </tr>
  `).join("");
}

function renderAlerts(d) {
  const rows = [...d.pareto_by_device];
  const byVolume = [...rows].sort((a, b) => b.total_minutes - a.total_minutes);
  const bySeverity = [...rows].sort((a, b) => b.avg_minutes - a.avg_minutes);
  const gaps = rows.filter(r => r.is_estimated).sort((a, b) => b.total_minutes - a.total_minutes).slice(0, 5);

  const alerts = [];

  if (bySeverity[0]) {
    alerts.push({
      type: "warning",
      title: `${formatDevice(bySeverity[0].device)} — highest severity per incident`,
      body: `Averages ${bySeverity[0].avg_minutes.toFixed(1)} min per event across ${bySeverity[0].events} incidents` +
        (bySeverity[0].rpn ? ` and carries an FMEA RPN of ${bySeverity[0].rpn} (${bySeverity[0].risk_level}).` : `, with no FMEA sheet on file (risk: UNKNOWN).`),
    });
  }
  if (byVolume[0]) {
    alerts.push({
      type: "info",
      title: `${formatDevice(byVolume[0].device)} — volume leader, not necessarily severity leader`,
      body: `${byVolume[0].events} events (most of any device), ${byVolume[0].avg_minutes.toFixed(1)} min average. Priority here is process discipline, not necessarily redesign.`,
    });
  }
  if (gaps.length) {
    alerts.push({
      type: "critical",
      title: `FMEA gap: ${gaps.map(g => formatDevice(g.device)).join(", ")}`,
      body: `${gaps.length} device(s) among the top empirical contributors have no FMEA sheet on file. Reported as UNKNOWN risk, not fabricated as "Low" — closing this gap would make the risk ranking fully evidence-based.`,
    });
  }
  alerts.push({
    type: "info",
    title: "Data coverage gap",
    body: `${d.unresolved_events} of ${d.total_events} total delay events (${(100 - d.coverage_pct).toFixed(1)}%) could not be resolved to a specific field device — either genuinely non-field-device delays or reason text too ambiguous. Reported as-is, not rounded up.`,
  });

  document.getElementById("alertsContainer").innerHTML = alerts.map(a => `
    <div class="alert ${a.type}">
      <p class="alert-title">${a.title}</p>
      <p class="alert-body">${a.body}</p>
    </div>
  `).join("");
}

function populateDeviceFilter(d) {
  const sel = document.getElementById("logDeviceFilter");
  const existing = new Set(Array.from(sel.options).map(o => o.value));
  d.pareto_by_device.forEach(r => {
    if (!existing.has(r.device)) {
      const opt = document.createElement("option");
      opt.value = r.device;
      opt.textContent = formatDevice(r.device);
      sel.appendChild(opt);
    }
  });
  const monthSel = document.getElementById("logMonthFilter");
  d.months_covered.forEach(m => {
    const opt = document.createElement("option");
    opt.value = m; opt.textContent = m;
    monthSel.appendChild(opt);
  });
}

let trendChartInstance = null;
function renderTrendChart(allForecasts) {
  const top5 = [...state.stats.pareto_by_device].sort((a, b) => b.total_minutes - a.total_minutes).slice(0, 5).map(r => r.device);
  const labels = (allForecasts[top5[0]] || { monthly_totals: [] }).monthly_totals.map(m => m.month);

  const datasets = top5.map(device => {
    const series = (allForecasts[device] || { monthly_totals: [] }).monthly_totals;
    const byMonth = Object.fromEntries(series.map(s => [s.month, s.mins]));
    return {
      label: formatDevice(device),
      data: labels.map(m => byMonth[m] ?? 0),
      borderColor: deviceColor(device),
      backgroundColor: deviceColor(device) + "33",
      tension: 0.35,
      pointRadius: 3,
      borderWidth: 2,
    };
  });

  const ctx = document.getElementById("trendChart").getContext("2d");
  if (trendChartInstance) trendChartInstance.destroy();
  trendChartInstance = new Chart(ctx, {
    type: "line",
    data: { labels, datasets },
    options: {
      responsive: true,
      interaction: { mode: "index", intersect: false },
      plugins: { legend: { position: "bottom", labels: { color: "#8194A6", boxWidth: 10, font: { size: 11 } } } },
      scales: {
        x: { ticks: { color: "#5A6C7E" }, grid: { color: "#182230" } },
        y: { ticks: { color: "#5A6C7E" }, grid: { color: "#182230" } },
      },
    },
  });
}

function renderForecastSummary(allForecasts) {
  const topDevice = [...state.stats.pareto_by_device].sort((a, b) => b.total_minutes - a.total_minutes)[0]?.device;
  // Aggregate "ALL" forecast isn't a single endpoint call here; approximate using sum across target devices' rolling forecasts.
  let rollingSum = 0, anyTrend = false, trendSum = 0, allReliable = true;
  Object.values(allForecasts).forEach(f => {
    rollingSum += f.rolling_avg_forecast || 0;
    if (f.trend_reliable) { trendSum += f.trend_forecast || 0; anyTrend = true; }
    else { allReliable = false; }
  });

  document.getElementById("forecastRolling").textContent = fmtMin(rollingSum);
  document.getElementById("forecastTrend").textContent = anyTrend && allReliable ? fmtMin(trendSum) : "Unreliable";
  document.getElementById("forecastNote").textContent = allReliable
    ? "Enough months with strong linear fit for every device — trend shown with confidence."
    : "3-month rolling average is the primary figure — trend line withheld or partial where R² is too low / too few months exist.";
}

function renderEventLog() {
  const { events, total, limit, offset } = state.events;
  document.getElementById("logCount").textContent = `${total} matching events`;
  document.getElementById("logTableBody").innerHTML = events.length ? events.map(e => `
    <tr>
      <td class="mono">${e.date || "—"}</td>
      <td>${e.month}</td>
      <td><span class="device-chip"><span class="device-dot" style="background:${deviceColor(e.device)}"></span>${formatDevice(e.device)}</span></td>
      <td class="mins">${e.mins != null ? Math.round(e.mins) : "—"}</td>
      <td class="reason">${escapeHtml(e.reason_text || "")}</td>
    </tr>
  `).join("") : `<tr><td colspan="5" style="color:var(--text-dim);">No events match this filter.</td></tr>`;

  const start = total === 0 ? 0 : offset + 1;
  const end = Math.min(offset + limit, total);
  document.getElementById("logPageInfo").textContent = `${start}–${end} of ${total}`;
  document.getElementById("logPrevBtn").disabled = offset === 0;
  document.getElementById("logNextBtn").disabled = end >= total;
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

// ------------------------------------------------------------- Predict panel --
function openPredictPanel() {
  document.getElementById("overlay").classList.add("open");
  document.getElementById("predictPanel").classList.add("open");
  document.getElementById("reasonInput").focus();
}
function closePredictPanel() {
  document.getElementById("overlay").classList.remove("open");
  document.getElementById("predictPanel").classList.remove("open");
}

function renderExampleChips() {
  document.getElementById("exampleChips").innerHTML = EXAMPLE_REASONS.map(r =>
    `<button type="button" class="example-chip" data-text="${escapeHtml(r)}">${escapeHtml(r.slice(0, 34))}${r.length > 34 ? "…" : ""}</button>`
  ).join("");
  document.querySelectorAll(".example-chip").forEach(chip => {
    chip.addEventListener("click", () => {
      document.getElementById("reasonInput").value = chip.dataset.text;
    });
  });
}

async function runPredict() {
  const reasonText = document.getElementById("reasonInput").value.trim();
  const minsVal = document.getElementById("minsInput").value;
  const errorEl = document.getElementById("predictError");
  const btn = document.getElementById("runPredictBtn");
  errorEl.innerHTML = "";

  if (reasonText.length < 3) {
    errorEl.innerHTML = `<div class="error-banner">Enter a delay description (at least 3 characters).</div>`;
    return;
  }

  btn.disabled = true;
  btn.innerHTML = `<span class="spinner"></span> Classifying…`;

  try {
    const payload = { reason_text: reasonText };
    if (minsVal !== "") payload.mins = parseFloat(minsVal);
    const result = await apiPost("/predict", payload);
    renderPredictResult(result);
  } catch (e) {
    errorEl.innerHTML = `<div class="error-banner">${escapeHtml(e.message)}</div>`;
    document.getElementById("resultBlock").classList.remove("show");
  } finally {
    btn.disabled = false;
    btn.textContent = "Classify Delay";
  }
}

function renderPredictResult(r) {
  document.getElementById("resultDot").style.background = deviceColor(r.predicted_device);
  document.getElementById("resultDeviceName").textContent = formatDevice(r.predicted_device);
  document.getElementById("resultConfidence").textContent = `${(r.confidence * 100).toFixed(1)}%`;

  document.getElementById("lowConfBanner").style.display = r.low_confidence ? "block" : "none";

  document.getElementById("candidatesList").innerHTML = r.top_candidates.map(c => `
    <div class="candidate-row">
      <span class="cname">${formatDevice(c.device)}</span>
      <div class="candidate-track"><div class="candidate-fill" style="width:${(c.confidence * 100).toFixed(1)}%; background:${deviceColor(c.device)}"></div></div>
      <span class="cval">${(c.confidence * 100).toFixed(0)}%</span>
    </div>
  `).join("");

  const riskEl = document.getElementById("resultRisk");
  riskEl.textContent = r.fmea_risk_level + (r.fmea_rpn ? ` (RPN ${r.fmea_rpn})` : "");
  riskEl.style.color = riskColor(r.fmea_risk_level);

  document.getElementById("resultMinutes").textContent = r.estimated_delay_minutes != null
    ? `${r.estimated_delay_minutes} min (as entered)`
    : "Not provided";

  document.getElementById("resultBlock").classList.add("show");
}

// ------------------------------------------------------------------ Init --
function wireEvents() {
  document.getElementById("openPredictBtn").addEventListener("click", openPredictPanel);
  document.getElementById("closePredictBtn").addEventListener("click", closePredictPanel);
  document.getElementById("overlay").addEventListener("click", closePredictPanel);
  document.addEventListener("keydown", e => { if (e.key === "Escape") closePredictPanel(); });
  document.getElementById("runPredictBtn").addEventListener("click", runPredict);
  document.getElementById("reasonInput").addEventListener("keydown", e => {
    if (e.key === "Enter" && e.ctrlKey) runPredict();
  });

  let searchDebounce;
  document.getElementById("logSearch").addEventListener("input", () => {
    clearTimeout(searchDebounce);
    searchDebounce = setTimeout(() => { state.events.offset = 0; loadEvents(); }, 300);
  });
  document.getElementById("logDeviceFilter").addEventListener("change", () => { state.events.offset = 0; loadEvents(); });
  document.getElementById("logMonthFilter").addEventListener("change", () => { state.events.offset = 0; loadEvents(); });
  document.getElementById("logPrevBtn").addEventListener("click", () => {
    state.events.offset = Math.max(0, state.events.offset - state.events.limit);
    loadEvents();
  });
  document.getElementById("logNextBtn").addEventListener("click", () => {
    state.events.offset += state.events.limit;
    loadEvents();
  });
}

async function init() {
  renderExampleChips();
  wireEvents();
  try {
    await loadStats();
    await loadTrend();
    await loadEvents();
    await loadModelInfo();
  } catch (e) {
    document.getElementById("alertsContainer").innerHTML =
      `<div class="alert critical"><p class="alert-title">Could not reach the API</p>` +
      `<p class="alert-body">${escapeHtml(e.message)} — make sure the backend is running (uvicorn app:app --reload --port 8000) and that /mnt/user-data isn't blocking CORS. API base: ${API_BASE_URL}</p></div>`;
  }
}

init();